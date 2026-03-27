"""
Training Engine
===============

Implements the training pipeline for Expert Adapters in the Patch-and-Route framework.

Key Design Decisions:
1. Uses SFTTrainer from TRL for instruction-tuning with chat templates
2. Supports streaming datasets (IterableDataset) with max_steps instead of epochs
3. Implements buffer shuffling for proper training mixing in streaming mode
4. Handles chat template application for Mistral-style models
5. Adapter-Aware: Can train multiple adapters sequentially on same foundation

The training process creates Expert Adapters that encode domain-specific knowledge
while keeping the Frozen Foundation parameters unchanged.

Supported Adapter Types:
- Base Adapter (base_v1): Trained on pre-2019 temporal + US geographic data
- Temporal Patches (patch_temp_YYYY): Trained on data from specific years
- Geographic Patches (patch_geo_COUNTRY): Trained on country-specific data
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from datasets import IterableDataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig
from peft import PeftModel

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for Expert Adapter training.
    
    Optimized defaults for training LoRA adapters on quantized base models
    with streaming datasets.
    
    Attributes:
        output_dir: Directory for checkpoints and logs.
        max_steps: Total training steps (required for streaming datasets).
        per_device_train_batch_size: Batch size per GPU.
        gradient_accumulation_steps: Steps to accumulate before update.
        learning_rate: Peak learning rate for AdamW.
        lr_scheduler_type: Learning rate schedule ("cosine" recommended).
        warmup_ratio: Fraction of steps for warmup.
        max_seq_length: Maximum sequence length (truncation).
        logging_steps: Steps between logging.
        save_steps: Steps between checkpoint saves.
        save_total_limit: Maximum checkpoints to keep.
        fp16: Use FP16 mixed precision (auto-disabled if bf16 available).
        bf16: Use BF16 mixed precision (preferred if available).
        gradient_checkpointing: Enable gradient checkpointing for memory.
        optim: Optimizer to use ("adamw_torch" or "paged_adamw_8bit").
        dataloader_num_workers: Workers for data loading.
        seed: Random seed for reproducibility.
        report_to: Logging integrations (e.g., ["wandb", "tensorboard"]).
    """
    # Output
    output_dir: str = "checkpoints/situatedqa_base_v1"
    
    # Training duration (max_steps required for streaming)
    max_steps: int = 1000
    
    # Batch configuration
    per_device_train_batch_size: int = 1  # Reduced for 14B models on 24GB GPU
    gradient_accumulation_steps: int = 16  # Effective batch = 16
    
    # Learning rate
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Sequence length
    max_seq_length: int = 2048
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 50  # Evaluate every N steps to track generalization
    eval_strategy: str = "steps"
    save_total_limit: int = 5
    load_best_model_at_end: bool = True  # Auto-select best checkpoint by eval loss
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Precision (auto-configured based on hardware)
    fp16: bool = False
    bf16: bool = False
    
    # Memory optimization
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {"use_reentrant": False})
    optim: str = "paged_adamw_8bit"  # Memory-efficient optimizer
    
    # Data loading
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Reproducibility
    seed: int = 42
    
    # Logging (use "none" to avoid tensorboard dependency issues)
    report_to: list[str] = field(default_factory=lambda: ["none"])

    # Progress bars — disable_tqdm=False keeps them on even in non-TTY environments
    # (e.g. SLURM log files). Steps are then visible via `tail -f *.out`.
    disable_tqdm: bool = False
    
    # Streaming-specific
    dataset_buffer_size: int = 10_000  # Shuffle buffer for streaming

    # Regularization (Generalization)
    neftune_noise_alpha: float | None = 5.0  # NEFTune noise for better generalization

    # MLflow experiment tracking
    mlflow_experiment: str = "pnr-training"
    mlflow_run_name: str | None = None
    mlflow_tracking_uri: str = "sqlite:///mlruns.db"
    
    def __post_init__(self):
        """Auto-configure precision based on hardware capabilities."""
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported() and not self.fp16:
                self.bf16 = True
                logger.info("Auto-enabled BF16 (hardware supported)")
            elif not self.bf16:
                self.fp16 = True
                logger.info("Auto-enabled FP16 (BF16 not supported)")
    
    def to_training_arguments(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments.
        
        Returns:
            Configured TrainingArguments instance.
        """
        return TrainingArguments(
            output_dir=self.output_dir,
            max_steps=self.max_steps,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            fp16=self.fp16,
            bf16=self.bf16,
            gradient_checkpointing=self.gradient_checkpointing,
            gradient_checkpointing_kwargs=self.gradient_checkpointing_kwargs,
            optim=self.optim,
            dataloader_num_workers=self.dataloader_num_workers,
            dataloader_pin_memory=self.dataloader_pin_memory,
            seed=self.seed,
            report_to=self.report_to,
            disable_tqdm=self.disable_tqdm,
            # Streaming-specific settings
            max_grad_norm=1.0,
            remove_unused_columns=False,  # Important for custom formatting
        )
    
    def to_sft_config(self) -> SFTConfig:
        """Convert to TRL SFTConfig.
        
        Returns:
            Configured SFTConfig instance.
        """
        return SFTConfig(
            output_dir=self.output_dir,
            max_steps=self.max_steps,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_train_batch_size,  # Must match train BS to avoid OOM during eval
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            eval_steps=self.eval_steps,
            eval_strategy=self.eval_strategy,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            fp16=self.fp16,
            bf16=self.bf16,
            gradient_checkpointing=self.gradient_checkpointing,
            gradient_checkpointing_kwargs=self.gradient_checkpointing_kwargs,
            optim=self.optim,
            dataloader_num_workers=self.dataloader_num_workers,
            dataloader_pin_memory=self.dataloader_pin_memory,
            seed=self.seed,
            report_to=self.report_to,
            disable_tqdm=self.disable_tqdm,
            max_grad_norm=1.0,
            remove_unused_columns=False,
            # SFT-specific settings
            max_length=self.max_seq_length,  # TRL 0.27+ renamed max_seq_length to max_length
            packing=False,  # Disable packing for chat format
            dataset_text_field=None,  # We use formatting_func instead
            neftune_noise_alpha=self.neftune_noise_alpha,  # Inject noise into embeddings
        )


# =============================================================================
# Training Engine
# =============================================================================

class PatchAndRouteTrainer:
    """Training engine for Expert Adapters in the Patch-and-Route framework.
    
    Handles the complete training pipeline:
    1. Dataset preparation with buffer shuffling
    2. Chat template application for instruction formatting
    3. SFTTrainer setup with streaming support
    4. Checkpoint management for continual learning
    
    Example:
        ```python
        # Initialize components
        llm = PatchAndRouteLLM()
        llm.load_frozen_foundation()
        llm.attach_expert(ExpertConfig(name="situatedqa_base"))
        model, tokenizer = llm.get_training_components()
        
        # Load data
        loader = SituatedQALoader()
        stable_stream, _ = loader.get_temporal_streams()
        formatted_data = loader.get_formatted_stream(stable_stream)
        
        # Train
        trainer = PatchAndRouteTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=formatted_data,
        )
        trainer.train()
        ```
    """
    
    def __init__(
        self,
        model: PreTrainedModel | PeftModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: IterableDataset,
        config: TrainingConfig | None = None,
        eval_dataset: IterableDataset | None = None,
        formatting_func: Callable[[dict[str, Any]], str] | None = None,
    ):
        """Initialize the training engine.
        
        Args:
            model: The PEFT-wrapped model to train.
            tokenizer: Tokenizer with chat template.
            train_dataset: Streaming training dataset (already formatted).
            config: Training configuration.
            eval_dataset: Optional evaluation dataset.
            formatting_func: Custom function to format examples to text.
                           If None, uses apply_chat_template on 'messages' field.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or TrainingConfig()
        self._formatting_func = formatting_func
        
        # Validate tokenizer setup
        self._validate_tokenizer()
        
        # Build trainer
        self.trainer: SFTTrainer | None = None
        
        logger.info("Initialized PatchAndRouteTrainer")
        logger.info(f"  Output dir: {self.config.output_dir}")
        logger.info(f"  Max steps: {self.config.max_steps}")
        logger.info(f"  Effective batch size: {self._effective_batch_size}")
    
    @property
    def _effective_batch_size(self) -> int:
        """Calculate effective batch size with gradient accumulation."""
        return (
            self.config.per_device_train_batch_size 
            * self.config.gradient_accumulation_steps
        )
    
    def _validate_tokenizer(self) -> None:
        """Validate tokenizer configuration for training.
        
        Raises:
            ValueError: If tokenizer is misconfigured.
        """
        # Check pad token
        if self.tokenizer.pad_token is None:
            raise ValueError(
                "Tokenizer must have a pad_token. "
                "Set tokenizer.pad_token = tokenizer.eos_token"
            )
        
        # Override chat template for training.
        # IMPORTANT: The stock DeepSeek-R1 tokenizer template strips <think>
        # blocks from assistant messages (designed for multi-turn inference).
        # During training we MUST preserve <think> blocks so the model learns
        # the Chain-of-Thought pattern. Always use this training-safe template.
        _TRAINING_CHAT_TEMPLATE = (
            "{{ bos_token }}"
            "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] }}"
            "{% set loop_messages = messages[1:] %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% endif %}"
            "{% for message in loop_messages %}"
            "{% if message['role'] == 'user' %}"
            "<｜User｜>{{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}"
            "<｜Assistant｜>{{ message['content'] }}<｜end▁of▁sentence｜>"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}<｜Assistant｜>{% endif %}"
        )
        if self.tokenizer.chat_template is None:
            logger.warning(
                "Tokenizer has no chat_template. "
                "Setting training-safe DeepSeek-R1 template."
            )
        else:
            logger.info(
                "Overriding tokenizer chat_template with training-safe "
                "template (preserves <think> blocks)."
            )
        self.tokenizer.chat_template = _TRAINING_CHAT_TEMPLATE
        
        logger.info("✓ Tokenizer validated")
    
    def _default_formatting_func(self, example: dict[str, Any]) -> str:
        """Default formatting function using chat template.
        
        Applies the tokenizer's chat template to the 'messages' field.
        
        Args:
            example: Dataset example with 'messages' field.
            
        Returns:
            Formatted string ready for tokenization.
        """
        messages = example.get("messages", [])
        
        if not messages:
            logger.warning("Example has empty 'messages' field")
            return ""
        
        # Apply chat template
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        return formatted
    
    def _prepare_dataset(
        self,
        dataset: IterableDataset,
        shuffle: bool = True,
    ) -> IterableDataset:
        """Prepare dataset for training with shuffling.
        
        For streaming datasets, buffer shuffling is critical to ensure
        proper mixing of examples during training.
        
        Args:
            dataset: Input streaming dataset.
            shuffle: Whether to apply buffer shuffling.
            
        Returns:
            Prepared streaming dataset.
        """
        if shuffle:
            # Check if this is a streaming (Iterable) dataset or regular Dataset
            from datasets import IterableDataset
            if isinstance(dataset, IterableDataset):
                # Streaming datasets use buffer_size for shuffle
                logger.info(
                    f"Applying buffer shuffle (buffer_size={self.config.dataset_buffer_size})"
                )
                dataset = dataset.shuffle(
                    seed=self.config.seed,
                    buffer_size=self.config.dataset_buffer_size,
                )
            else:
                # Regular datasets don't use buffer_size
                logger.info("Shuffling dataset")
                dataset = dataset.shuffle(seed=self.config.seed)
        
        return dataset
    
    def build_trainer(self) -> SFTTrainer:
        """Build the SFTTrainer instance.
        
        Returns:
            Configured SFTTrainer ready for training.
        """
        # Prepare datasets
        train_data = self._prepare_dataset(self.train_dataset, shuffle=True)
        eval_data = None
        if self.eval_dataset is not None:
            eval_data = self._prepare_dataset(self.eval_dataset, shuffle=False)
        
        # Get formatting function
        formatting_func = self._formatting_func or self._default_formatting_func

        # Build SFT config
        sft_config = self.config.to_sft_config()

        # Optionally attach step-level MLflow callback
        callbacks = []
        try:
            from src.utils.mlflow_tracker import MLflowStepCallback, _MLFLOW_AVAILABLE
            if _MLFLOW_AVAILABLE:
                callbacks.append(MLflowStepCallback())
                logger.info("MLflowStepCallback registered for step-level logging")
        except ImportError:
            pass

        # Create trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=train_data,
            eval_dataset=eval_data,
            processing_class=self.tokenizer,
            formatting_func=formatting_func,
            callbacks=callbacks if callbacks else None,
        )

        logger.info("✓ SFTTrainer built successfully")
        return self.trainer
    
    def train(self, resume_from_checkpoint: str | bool | None = None) -> dict[str, Any]:
        """Run the training loop.
        
        Args:
            resume_from_checkpoint: Path to checkpoint or True to auto-detect.
            
        Returns:
            Training metrics dictionary.
        """
        logger.info("=" * 60)
        logger.info("STARTING EXPERT ADAPTER TRAINING")
        logger.info("=" * 60)
        logger.info(f"Max steps: {self.config.max_steps}")
        logger.info(f"Batch size: {self._effective_batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Output: {self.config.output_dir}")
        logger.info(f"MLflow experiment: {self.config.mlflow_experiment}")
        logger.info("=" * 60)

        # Import tracker (graceful no-op if mlflow missing)
        try:
            from src.utils.mlflow_tracker import PnRTracker, _MLFLOW_AVAILABLE
            _use_mlflow = _MLFLOW_AVAILABLE
        except ImportError:
            _use_mlflow = False

        def _build_and_run():
            if self.trainer is None:
                self.build_trainer()
            train_result = self.trainer.train(
                resume_from_checkpoint=resume_from_checkpoint
            )
            metrics = train_result.metrics
            logger.info("=" * 60)
            logger.info("TRAINING COMPLETE")
            logger.info("=" * 60)
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}")
            logger.info("=" * 60)
            return metrics

        if _use_mlflow:
            with PnRTracker(
                experiment_name=self.config.mlflow_experiment,
                run_name=self.config.mlflow_run_name,
                tracking_uri=self.config.mlflow_tracking_uri,
            ) as tracker:
                tracker.log_training_config(self.config)
                metrics = _build_and_run()
                tracker.log_metrics(metrics)
                tracker.log_gpu_memory()
                tracker.log_adapter_artifact(self.config.output_dir)
        else:
            metrics = _build_and_run()

        return metrics
    
    def save_model(self, output_dir: str | Path | None = None) -> Path:
        """Save the trained adapter.
        
        Args:
            output_dir: Output directory (defaults to config.output_dir).
            
        Returns:
            Path to saved model.
        """
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving trained adapter to: {output_dir}")
        
        # Save adapter weights
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info("✓ Model saved successfully")
        return output_dir


# =============================================================================
# Convenience Functions
# =============================================================================

def train_base_expert(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: IterableDataset,
    output_dir: str = "checkpoints/situatedqa_base_v1",
    max_steps: int = 1000,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    **kwargs,
) -> dict[str, Any]:
    """Convenience function to train a Base Expert Adapter.
    
    Simplified interface for training an expert on stable knowledge.
    
    Args:
        model: PEFT-wrapped model.
        tokenizer: Configured tokenizer.
        train_dataset: Formatted streaming dataset.
        output_dir: Checkpoint directory.
        max_steps: Training steps.
        learning_rate: Peak learning rate.
        batch_size: Per-device batch size.
        **kwargs: Additional TrainingConfig overrides.
        
    Returns:
        Training metrics.
        
    Example:
        ```python
        metrics = train_base_expert(
            model=model,
            tokenizer=tokenizer,
            train_dataset=stable_stream,
            output_dir="checkpoints/base_v1",
            max_steps=2000,
        )
        ```
    """
    config = TrainingConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        **kwargs,
    )
    
    trainer = PatchAndRouteTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config=config,
    )
    
    metrics = trainer.train()
    trainer.save_model()
    
    return metrics


# =============================================================================
# Adapter-Aware Training Interface
# =============================================================================

def train_adapter(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: IterableDataset,
    adapter_name: str,
    output_dir: str | None = None,
    max_steps: int = 1000,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    save_steps: int = 100,
    logging_steps: int = 10,
    **kwargs,
) -> dict[str, Any]:
    """Train a specific adapter with automatic checkpoint path management.
    
    This is the primary interface for training adapters in the Patch-and-Route
    framework. It handles checkpoint organization automatically based on adapter type.
    
    Args:
        model: PEFT-wrapped model (must have adapter already attached).
        tokenizer: Configured tokenizer with chat template.
        dataset: Formatted streaming dataset (with 'messages' field).
        adapter_name: Unique identifier for this adapter. Used to derive output path.
                     Examples: "base_v1", "patch_temp_2021", "patch_geo_india"
        output_dir: Optional explicit output directory. If None, derived from adapter_name
                   as "checkpoints/{adapter_name}".
        max_steps: Total training steps (required for streaming datasets).
        learning_rate: Peak learning rate for AdamW optimizer.
        batch_size: Per-device training batch size.
        gradient_accumulation_steps: Steps to accumulate before weight update.
        save_steps: Steps between checkpoint saves.
        logging_steps: Steps between logging.
        **kwargs: Additional TrainingConfig overrides.
        
    Returns:
        Dictionary of training metrics.
        
    Example:
        ```python
        from src.models.core import PatchAndRouteLLM, ExpertConfig
        from src.data.loader import SituatedQALoader
        
        # Load model once
        llm = PatchAndRouteLLM()
        llm.load_frozen_foundation()
        
        # Train Base adapter
        llm.attach_expert(ExpertConfig(name="base_v1"))
        model, tokenizer = llm.get_training_components()
        
        loader = SituatedQALoader()
        base_data = loader.format_stream(loader.get_base_stream())
        
        metrics = train_adapter(
            model=model,
            tokenizer=tokenizer,
            dataset=base_data,
            adapter_name="base_v1",
            max_steps=2000,
        )
        
        # Save and detach for next adapter
        llm.save_expert("checkpoints/base_v1")
        llm.detach_expert()
        
        # Train Temporal Patch
        llm.attach_expert(ExpertConfig(name="patch_temp_2021"))
        model, tokenizer = llm.get_training_components()
        
        temp_data = loader.format_stream(loader.get_temporal_patch_stream())
        
        train_adapter(
            model=model,
            tokenizer=tokenizer,
            dataset=temp_data,
            adapter_name="patch_temp_2021",
        )
        ```
    """
    # Derive output directory from adapter name if not specified
    if output_dir is None:
        output_dir = f"checkpoints/{adapter_name}"
    
    logger.info("=" * 60)
    logger.info(f"TRAINING ADAPTER: {adapter_name}")
    logger.info("=" * 60)
    logger.info(f"Output: {output_dir}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    logger.info("=" * 60)
    
    # Build configuration
    config = TrainingConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        **kwargs,
    )
    
    # Create and run trainer
    trainer = PatchAndRouteTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        config=config,
    )
    
    metrics = trainer.train()
    
    # Save final checkpoint
    trainer.save_model()
    
    logger.info(f"✓ Adapter '{adapter_name}' training complete")
    logger.info(f"  Saved to: {output_dir}")
    
    return metrics


def train_multiple_adapters(
    llm,  # PatchAndRouteLLM instance
    adapter_configs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Train multiple adapters sequentially on the same foundation.
    
    Useful for batch training of multiple patches in one script.
    
    Args:
        llm: PatchAndRouteLLM instance with loaded foundation.
        adapter_configs: List of adapter configurations, each containing:
            - name: Adapter name
            - dataset: Formatted streaming dataset
            - expert_config: Optional ExpertConfig (uses defaults if not provided)
            - max_steps: Optional training steps override
            - learning_rate: Optional learning rate override
            
    Returns:
        Dictionary mapping adapter names to their training metrics.
        
    Example:
        ```python
        from src.models.core import PatchAndRouteLLM, ExpertConfig
        from src.data.loader import SituatedQALoader
        
        llm = PatchAndRouteLLM()
        llm.load_frozen_foundation()
        
        loader = SituatedQALoader()
        
        configs = [
            {
                "name": "base_v1",
                "dataset": loader.format_stream(loader.get_base_stream()),
                "max_steps": 2000,
            },
            {
                "name": "patch_geo_india",
                "dataset": loader.format_stream(loader.get_geo_patch_stream("India")),
                "max_steps": 500,
            },
        ]
        
        all_metrics = train_multiple_adapters(llm, configs)
        ```
    """
    from src.models.core import ExpertConfig
    
    all_metrics: dict[str, dict[str, Any]] = {}
    
    for i, cfg in enumerate(adapter_configs):
        adapter_name = cfg["name"]
        dataset = cfg["dataset"]
        
        logger.info(f"\n{'=' * 60}")
        logger.info(f"ADAPTER {i + 1}/{len(adapter_configs)}: {adapter_name}")
        logger.info(f"{'=' * 60}\n")
        
        # Get or create expert config
        expert_config = cfg.get("expert_config", ExpertConfig(name=adapter_name))
        if expert_config.name != adapter_name:
            expert_config.name = adapter_name
        
        # Attach adapter
        llm.attach_expert(expert_config)
        model, tokenizer = llm.get_training_components()
        
        # Train
        metrics = train_adapter(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            adapter_name=adapter_name,
            max_steps=cfg.get("max_steps", 1000),
            learning_rate=cfg.get("learning_rate", 2e-4),
            batch_size=cfg.get("batch_size", 4),
        )
        
        all_metrics[adapter_name] = metrics
        
        # Save and detach for next adapter
        llm.save_expert(f"checkpoints/{adapter_name}")
        llm.detach_expert()
        
        logger.info(f"✓ Completed adapter {i + 1}/{len(adapter_configs)}")
    
    return all_metrics

