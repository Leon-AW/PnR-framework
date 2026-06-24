"""
Core Model Components
=====================

Implements the Frozen Foundation and Expert Adapter architecture for Patch-and-Route.

Architecture Overview:
- **Frozen Foundation**: Base LLM (Mistral-7B) with 4-bit quantization, parameters frozen
- **Expert Adapters**: LoRA modules that encode domain-specific knowledge
- **PatchAndRouteLLM**: Orchestrator class managing foundation + adapters

Key Design Decisions:
1. 4-bit quantization via bitsandbytes for memory efficiency
2. LoRA targets attention layers (q_proj, k_proj, v_proj, o_proj) for maximum impact
3. Adapter weights are saved separately, enabling hot-swapping
4. Foundation is loaded once, adapters attached/detached dynamically

References:
- QLoRA: https://arxiv.org/abs/2305.14314
- LoRA: https://arxiv.org/abs/2106.09685
- PEFT: https://github.com/huggingface/peft
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class QuantizationType(Enum):
    """Quantization options for the Frozen Foundation."""
    NONE = "none"      # Full precision (FP16/BF16)
    INT8 = "int8"      # 8-bit quantization
    INT4 = "int4"      # 4-bit quantization (recommended)


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class FrozenFoundationConfig:
    """Configuration for the Frozen Foundation (base LLM).
    
    Attributes:
        model_id: HuggingFace model identifier.
        quantization: Quantization type for memory efficiency.
        device_map: Device placement strategy ("auto" recommended).
        torch_dtype: Data type for non-quantized layers.
        trust_remote_code: Allow remote code execution (some models require this).
        use_cache: Enable KV cache (disable for training with grad checkpointing).
        attn_implementation: Attention implementation ("flash_attention_2" if available).
    """
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    quantization: QuantizationType = QuantizationType.INT4
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.bfloat16
    trust_remote_code: bool = True
    use_cache: bool = False  # Disable for training with gradient checkpointing
    attn_implementation: str | None = None  # Auto-detect
    
    def get_bnb_config(self) -> BitsAndBytesConfig | None:
        """Create BitsAndBytesConfig based on quantization setting.
        
        Returns:
            BitsAndBytesConfig or None if no quantization.
        """
        if self.quantization == QuantizationType.NONE:
            return None
        
        if self.quantization == QuantizationType.INT8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        
        if self.quantization == QuantizationType.INT4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        
        return None


@dataclass
class ExpertConfig:
    """Configuration for Expert Adapters (LoRA modules).
    
    Attributes:
        name: Unique identifier for this adapter.
        r: LoRA rank (higher = more capacity, more parameters).
        lora_alpha: Scaling factor (typically 2x rank).
        lora_dropout: Dropout probability for regularization.
        target_modules: Model layers to apply LoRA to.
        bias: Bias handling ("none", "all", or "lora_only").
        task_type: PEFT task type (CAUSAL_LM for text generation).
        modules_to_save: Additional modules to make trainable.
    """
    name: str = "expert_adapter"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ])
    bias: str = "none"
    task_type: TaskType = TaskType.CAUSAL_LM
    modules_to_save: list[str] | None = None
    
    def to_lora_config(self) -> LoraConfig:
        """Convert to PEFT LoraConfig.
        
        Returns:
            Configured LoraConfig instance.
        """
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=self.task_type,
            modules_to_save=self.modules_to_save,
        )


# =============================================================================
# Main Model Class
# =============================================================================

class PatchAndRouteLLM:
    """Orchestrator for Frozen Foundation + Expert Adapters.
    
    Manages the complete lifecycle:
    1. Load quantized base model (Frozen Foundation)
    2. Attach LoRA adapters (Expert Adapters)
    3. Train adapters while keeping foundation frozen
    4. Save/load adapter checkpoints
    5. Switch between adapters at inference time
    
    Example:
        ```python
        # Initialize
        llm = PatchAndRouteLLM()
        
        # Load foundation (only once)
        llm.load_frozen_foundation()
        
        # Attach and train an expert
        llm.attach_expert(ExpertConfig(name="base_v1"))
        model, tokenizer = llm.get_training_components()
        # ... train with SFTTrainer ...
        llm.save_expert("checkpoints/base_v1")
        
        # Later: load a different expert
        llm.detach_expert()
        llm.load_expert("checkpoints/patch_2021")
        ```
    """
    
    def __init__(
        self,
        foundation_config: FrozenFoundationConfig | None = None,
    ) -> None:
        """Initialize the orchestrator.
        
        Args:
            foundation_config: Configuration for the base model.
        """
        self.foundation_config = foundation_config or FrozenFoundationConfig()
        
        self._model: PreTrainedModel | PeftModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._current_expert: str | None = None
        self._is_peft_model: bool = False
        
        logger.info("Initialized PatchAndRouteLLM")
        logger.info(f"  Model: {self.foundation_config.model_id}")
        logger.info(f"  Quantization: {self.foundation_config.quantization.value}")
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def model(self) -> PreTrainedModel | PeftModel:
        """Get the current model (foundation or PEFT-wrapped)."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_frozen_foundation() first.")
        return self._model
    
    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Get the tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_frozen_foundation() first.")
        return self._tokenizer
    
    @property
    def current_expert(self) -> str | None:
        """Get the name of the currently attached expert."""
        return self._current_expert
    
    @property
    def is_foundation_loaded(self) -> bool:
        """Check if the foundation model is loaded."""
        return self._model is not None
    
    @property
    def has_expert_attached(self) -> bool:
        """Check if an expert adapter is attached."""
        return self._is_peft_model
    
    # -------------------------------------------------------------------------
    # Foundation Loading
    # -------------------------------------------------------------------------
    
    def load_frozen_foundation(self) -> None:
        """Load the Frozen Foundation (quantized base model).
        
        This should be called once at the start. The foundation remains
        in memory while experts are attached/detached.
        """
        if self._model is not None:
            logger.warning("Model already loaded. Skipping reload.")
            return
        
        logger.info("=" * 60)
        logger.info("LOADING FROZEN FOUNDATION")
        logger.info("=" * 60)
        logger.info(f"Model: {self.foundation_config.model_id}")
        logger.info(f"Quantization: {self.foundation_config.quantization.value}")
        
        # Get quantization config
        bnb_config = self.foundation_config.get_bnb_config()
        
        # Detect attention implementation
        attn_impl = self.foundation_config.attn_implementation
        if attn_impl is None:
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
                logger.info("Flash Attention 2 detected, enabling...")
            except ImportError:
                attn_impl = "sdpa"  # PyTorch 2.0 scaled dot product attention
                logger.info("Using SDPA attention (Flash Attention not available)")
        
        # Load model
        model_kwargs: dict[str, Any] = {
            "device_map": self.foundation_config.device_map,
            "torch_dtype": self.foundation_config.torch_dtype,
            "trust_remote_code": self.foundation_config.trust_remote_code,
            "use_cache": self.foundation_config.use_cache,
            "attn_implementation": attn_impl,
        }

        # Compute max_memory from actual free GPU memory so that
        # device_map="auto" doesn't overshoot when another process already
        # occupies part of the GPU.  Without this, accelerate uses 90% of
        # *total* GPU memory which can exceed *free* memory and silently
        # dispatches layers to CPU, breaking bitsandbytes 4-bit training.
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            # Reserve 10 % of free memory as a buffer
            safe_gib = int(free_bytes * 0.90 / (1024 ** 3))
            model_kwargs["max_memory"] = {0: f"{safe_gib}GiB", "cpu": "30GiB"}
            logger.info(f"GPU memory budget: {safe_gib} GiB (free={free_bytes/1024**3:.1f} GiB)")

        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        logger.info("Loading model weights...")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.foundation_config.model_id,
            **model_kwargs,
        )
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.foundation_config.model_id,
            trust_remote_code=self.foundation_config.trust_remote_code,
        )
        
        # Configure tokenizer for training
        self._configure_tokenizer()
        
        # Prepare for k-bit training if quantized
        if bnb_config is not None:
            logger.info("Preparing model for k-bit training...")
            self._model = prepare_model_for_kbit_training(
                self._model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        
        logger.info("✓ Frozen Foundation loaded successfully")
        self._log_model_info()
    
    def _configure_tokenizer(self) -> None:
        """Configure tokenizer for training compatibility."""
        if self._tokenizer is None:
            return
        
        # Set padding token (required for batched training)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            logger.info("Set pad_token = eos_token")
        
        # Left padding for decoder-only models (better for generation)
        self._tokenizer.padding_side = "right"  # Right for training
        
        logger.info("✓ Tokenizer configured")
    
    # -------------------------------------------------------------------------
    # Expert Adapter Management
    # -------------------------------------------------------------------------
    
    def attach_expert(self, config: ExpertConfig) -> None:
        """Attach a new Expert Adapter to the foundation.
        
        Creates a new LoRA adapter and wraps the model with PEFT.
        
        Args:
            config: Configuration for the expert adapter.
        """
        if self._model is None:
            raise RuntimeError("Foundation not loaded. Call load_frozen_foundation() first.")
        
        if self._is_peft_model:
            logger.warning(f"Expert '{self._current_expert}' already attached. Detaching first...")
            self.detach_expert()
        
        logger.info(f"Attaching Expert Adapter: {config.name}")
        logger.info(f"  LoRA rank: {config.r}")
        logger.info(f"  LoRA alpha: {config.lora_alpha}")
        logger.info(f"  Target modules: {config.target_modules}")
        
        # Create LoRA config
        lora_config = config.to_lora_config()
        
        # Wrap model with PEFT
        self._model = get_peft_model(self._model, lora_config)
        self._is_peft_model = True
        self._current_expert = config.name

        # When the base model uses device_map="auto" (quantized), newly added
        # LoRA trainable parameters may be initialized on CPU.  Move them to
        # cuda:0 — which is always the allocated GPU because SLURM (or the
        # caller) sets CUDA_VISIBLE_DEVICES before launch.  We can't rely on
        # iterating base_model.parameters() to find the target device because
        # bitsandbytes Params4bit objects may report .device == 'cpu' even
        # when the model is live on GPU.
        if torch.cuda.is_available():
            target_device = torch.device("cuda:0")
            moved = 0
            for _name, param in self._model.named_parameters():
                if param.requires_grad and param.device.type != "cuda":
                    param.data = param.data.to(target_device)
                    moved += 1
            if moved:
                logger.info(f"  Moved {moved} LoRA parameter tensors → {target_device}")

        # Log trainable parameters
        self._log_trainable_params()
        
        logger.info(f"✓ Expert '{config.name}' attached")
    
    def detach_expert(self) -> None:
        """Detach the current Expert Adapter.
        
        Returns the model to the base foundation state.
        """
        if not self._is_peft_model:
            logger.warning("No expert attached. Nothing to detach.")
            return
        
        logger.info(f"Detaching Expert Adapter: {self._current_expert}")
        
        # For QLoRA (4-bit quantized models), we need to be careful about how we unload.
        # The cleanest approach is to delete the adapter and clear PEFT state.
        try:
            # Method 1: Use delete_adapter if available (PEFT >= 0.6)
            if hasattr(self._model, "delete_adapter"):
                adapter_name = self._model.active_adapter
                if isinstance(adapter_name, list):
                    adapter_name = adapter_name[0] if adapter_name else "default"
                self._model.delete_adapter(adapter_name)
                # Get the base model after deletion
                if hasattr(self._model, "base_model"):
                    self._model = self._model.base_model.model
                logger.info(f"Deleted adapter via delete_adapter()")
            # Method 2: Use unload() 
            elif hasattr(self._model, "unload"):
                self._model = self._model.unload()
                logger.info(f"Unloaded adapter via unload()")
            # Method 3: Fallback - get base model
            else:
                if hasattr(self._model, "base_model"):
                    base = self._model.base_model
                    # Navigate to the actual model if needed
                    if hasattr(base, "model"):
                        self._model = base.model
                    else:
                        self._model = base
                logger.info(f"Extracted base_model")
        except Exception as e:
            logger.warning(f"Error during adapter detach: {e}. Attempting base_model extraction.")
            if hasattr(self._model, "base_model"):
                self._model = self._model.base_model
        
        # Clean up any lingering PEFT attributes that cause the warning
        for attr in ["peft_config", "active_adapter", "active_adapters"]:
            if hasattr(self._model, attr):
                try:
                    delattr(self._model, attr)
                except AttributeError:
                    pass  # Some attributes may be properties
        
        self._is_peft_model = False
        self._current_expert = None
        
        logger.info("✓ Expert detached, foundation restored")
    
    def load_expert(self, checkpoint_path: str | Path) -> None:
        """Load a saved Expert Adapter from checkpoint.
        
        Args:
            checkpoint_path: Path to the saved adapter checkpoint.
        """
        if self._model is None:
            raise RuntimeError("Foundation not loaded. Call load_frozen_foundation() first.")
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if self._is_peft_model:
            logger.warning(f"Expert '{self._current_expert}' already attached. Detaching first...")
            self.detach_expert()
        
        logger.info(f"Loading Expert Adapter from: {checkpoint_path}")
        
        # Load adapter from checkpoint
        self._model = PeftModel.from_pretrained(
            self._model,
            checkpoint_path,
            is_trainable=True,
        )
        self._is_peft_model = True
        self._current_expert = checkpoint_path.name
        
        logger.info(f"✓ Expert loaded from {checkpoint_path}")
    
    def save_expert(self, output_path: str | Path) -> Path:
        """Save the current Expert Adapter to disk.
        
        Args:
            output_path: Directory to save the adapter.
            
        Returns:
            Path to the saved checkpoint.
        """
        if not self._is_peft_model:
            raise RuntimeError("No expert attached. Nothing to save.")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving Expert Adapter to: {output_path}")
        
        # Save adapter weights only (not full model)
        self._model.save_pretrained(output_path)
        
        # Save tokenizer alongside
        self._tokenizer.save_pretrained(output_path)
        
        logger.info(f"✓ Expert '{self._current_expert}' saved to {output_path}")
        return output_path
    
    # -------------------------------------------------------------------------
    # Training Interface
    # -------------------------------------------------------------------------
    
    def get_training_components(self) -> tuple[PeftModel, PreTrainedTokenizerBase]:
        """Get model and tokenizer for training.
        
        Returns:
            Tuple of (peft_model, tokenizer) ready for SFTTrainer.
            
        Raises:
            RuntimeError: If no expert is attached.
        """
        if not self._is_peft_model:
            raise RuntimeError(
                "No expert attached. Call attach_expert() before training."
            )
        
        return self._model, self._tokenizer
    
    def get_inference_components(self) -> tuple[PreTrainedModel | PeftModel, PreTrainedTokenizerBase]:
        """Get model and tokenizer for inference.
        
        Returns:
            Tuple of (model, tokenizer) for generation.
        """
        return self.model, self.tokenizer
    
    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    
    def _log_model_info(self) -> None:
        """Log model information."""
        if self._model is None:
            return
        
        # Count parameters
        total_params = sum(p.numel() for p in self._model.parameters())
        
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Model dtype: {self._model.dtype}")
        
        # Memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"  GPU memory allocated: {allocated:.2f} GB")
            logger.info(f"  GPU memory reserved: {reserved:.2f} GB")
    
    def _log_trainable_params(self) -> None:
        """Log trainable parameter count after attaching adapter."""
        if self._model is None:
            return
        
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        pct = 100 * trainable / total if total > 0 else 0
        
        logger.info(f"  Trainable parameters: {trainable:,} ({pct:.2f}%)")
        logger.info(f"  Frozen parameters: {total - trainable:,}")
    
    def print_model_info(self) -> None:
        """Print detailed model information."""
        print("=" * 60)
        print("PATCH-AND-ROUTE MODEL INFO")
        print("=" * 60)
        print(f"Model ID: {self.foundation_config.model_id}")
        print(f"Quantization: {self.foundation_config.quantization.value}")
        print(f"Expert attached: {self._current_expert or 'None'}")
        
        if self._model is not None:
            total = sum(p.numel() for p in self._model.parameters())
            trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            print(f"Total parameters: {total:,}")
            print(f"Trainable parameters: {trainable:,}")
            print(f"Trainable %: {100 * trainable / total:.2f}%")
            
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                print(f"GPU memory: {allocated:.2f} GB")
        
        print("=" * 60)


# =============================================================================
# Convenience Functions
# =============================================================================

def load_model_for_training(
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
    adapter_name: str = "expert_adapter",
    lora_r: int = 16,
    lora_alpha: int = 32,
    quantization: QuantizationType = QuantizationType.INT4,
) -> tuple[PeftModel, PreTrainedTokenizerBase]:
    """Convenience function to load model ready for training.
    
    Args:
        model_id: HuggingFace model identifier.
        adapter_name: Name for the LoRA adapter.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        quantization: Quantization type.
        
    Returns:
        Tuple of (peft_model, tokenizer) ready for SFTTrainer.
    """
    foundation_config = FrozenFoundationConfig(
        model_id=model_id,
        quantization=quantization,
    )
    
    expert_config = ExpertConfig(
        name=adapter_name,
        r=lora_r,
        lora_alpha=lora_alpha,
    )
    
    llm = PatchAndRouteLLM(foundation_config=foundation_config)
    llm.load_frozen_foundation()
    llm.attach_expert(expert_config)
    
    return llm.get_training_components()


def load_model_for_inference(
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
    adapter_path: str | Path | None = None,
    quantization: QuantizationType = QuantizationType.INT4,
) -> tuple[PreTrainedModel | PeftModel, PreTrainedTokenizerBase]:
    """Convenience function to load model for inference.
    
    Args:
        model_id: HuggingFace model identifier.
        adapter_path: Optional path to saved adapter checkpoint.
        quantization: Quantization type.
        
    Returns:
        Tuple of (model, tokenizer) ready for generation.
    """
    foundation_config = FrozenFoundationConfig(
        model_id=model_id,
        quantization=quantization,
        use_cache=True,  # Enable KV cache for inference
    )
    
    llm = PatchAndRouteLLM(foundation_config=foundation_config)
    llm.load_frozen_foundation()
    
    if adapter_path is not None:
        llm.load_expert(adapter_path)
    
    return llm.get_inference_components()

