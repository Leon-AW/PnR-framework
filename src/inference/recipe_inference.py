"""
RECIPE Inference Wrapper
========================

Thin wrapper around RECIPE (Retrieval-Augmented Continuous Prompt lEarning)
that provides the same interface as PatchAndRouteInference.generate(), returning
an object with .response, .adapter_loaded, and .routing_result attributes.

RECIPE (Chen et al., EMNLP 2024) enables lifelong knowledge editing without
modifying LLM parameters.  It maintains a knowledge retrieval repository K_t
and a trainable prompt encoder.  At inference time, a continuous prompt p_k is
retrieved and prepended to the LLM's input embeddings to steer the response.

Architecture
------------
Trainable components (LLM frozen):
    f_rm   : RoBERTa encoder — text → pooled representation
    MLP_K  : pooled f_rm output → knowledge representation r_k  (Eq. 6)
    MLP_P  : r_k → continuous prompt tokens p_k  (Eq. 7)
    MLP_Q  : pooled f_rm output → query representation r̃_q  (Eq. 8)
    Θ      : Knowledge Sentinel — trainable parameter in knowledge space

Knowledge retrieval (Eq. 9):
    j = argmax_{τ} r̃_q · r_{k_τ}
    KS(q) = p_{k_j}  if  r̃_q · r_{k_j} > r̃_q · r_Θ  else ∅

LLM inference with editing on the fly (Section 4.3):
    output = f_llm([p_k ; emb(q)])   (p_k prepended to word embeddings)

Reference
---------
Chen et al., "Lifelong Knowledge Editing for LLMs with Retrieval-Augmented
Continuous Prompt Learning", EMNLP 2024.
Repo: https://github.com/qizhou000/RECIPE
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RECIPEConfig:
    """Hyperparameters for RECIPE.

    Attributes:
        encoder_model: HuggingFace model ID for the knowledge representation
            encoder f_rm.  RoBERTa-base is the default used in the paper.
        d_k: Dimension of the knowledge representation space.  Used as the
            output dimension of MLP_K and MLP_Q and as the dimension of the
            Knowledge Sentinel Θ.
        cpt_length: Number of Continuous Prompt Tokens (CPTs) l prepended to
            the LLM input.  The paper finds l=3 optimal (Section 5.2.1).
        temperature: InfoNCE temperature τ for the contrastive prompt-learning
            losses L_no and L_so (Eq. 17).
        lambda_loc: Weight of the locality KL divergence in L_edit (Eq. 12-13).
        llm_hidden_size: Hidden dimension d_llm of the target LLM.  Set
            automatically by train_recipe_baseline.py and stored in the
            checkpoint config so the module can be reconstructed at inference
            time without loading the LLM first.
    """
    encoder_model: str = "roberta-base"
    d_k: int = 512
    cpt_length: int = 3
    temperature: float = 1.0
    lambda_loc: float = 0.5
    llm_hidden_size: int = 4096  # Mistral-7B / LLaMA-2-7B / LLaMA-3-8B

    def to_dict(self) -> dict:
        return {
            "encoder_model": self.encoder_model,
            "d_k": self.d_k,
            "cpt_length": self.cpt_length,
            "temperature": self.temperature,
            "lambda_loc": self.lambda_loc,
            "llm_hidden_size": self.llm_hidden_size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RECIPEConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# RECIPE Trainable Module
# =============================================================================

class RECIPEModule(nn.Module):
    """Full set of trainable RECIPE components.

    Bundles f_rm (RoBERTa), MLP_K, MLP_P, MLP_Q, and the Knowledge Sentinel Θ
    into a single nn.Module so that they can be jointly trained and saved as a
    single state_dict.

    Args:
        config: RECIPEConfig with architecture hyperparameters.
    """

    def __init__(self, config: RECIPEConfig) -> None:
        super().__init__()
        self.config = config

        # --- f_rm: RoBERTa knowledge representation encoder ---
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(config.encoder_model)
        encoder_hidden: int = self.encoder.config.hidden_size  # 768 for roberta-base
        pooled_dim: int = encoder_hidden * 3  # max + min + avg concatenated

        # --- MLP_K: pooled f_rm → r_k  (Eq. 6) ---
        self.mlp_k = nn.Sequential(
            nn.Linear(pooled_dim, config.d_k * 2),
            nn.GELU(),
            nn.Linear(config.d_k * 2, config.d_k),
        )

        # --- MLP_P: r_k → p_k = (l, d_llm)  (Eq. 7) ---
        self.mlp_p = nn.Sequential(
            nn.Linear(config.d_k, config.d_k * 2),
            nn.GELU(),
            nn.Linear(config.d_k * 2, config.cpt_length * config.llm_hidden_size),
        )
        self.cpt_length = config.cpt_length
        self.llm_hidden_size = config.llm_hidden_size

        # --- MLP_Q: pooled f_rm → r̃_q  (Eq. 8) ---
        self.mlp_q = nn.Sequential(
            nn.Linear(pooled_dim, config.d_k * 2),
            nn.GELU(),
            nn.Linear(config.d_k * 2, config.d_k),
        )

        # --- Knowledge Sentinel Θ: trainable d_k vector  (Section 4.2) ---
        # Represents "no relevant knowledge" in the knowledge representation
        # space.  Trained with the sentinel-oriented contrastive loss L_so to
        # be most similar to locality (irrelevant) queries.
        self.knowledge_sentinel = nn.Parameter(torch.randn(config.d_k) * 0.02)

    # -------------------------------------------------------------------------
    # Pooling helper
    # -------------------------------------------------------------------------

    @staticmethod
    def _pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute masked max + min + avg pooling and concatenate.

        Args:
            last_hidden_state: (batch, seq_len, hidden) — RoBERTa output.
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding.

        Returns:
            Pooled representation of shape (batch, hidden * 3).
        """
        # Expand mask: (batch, seq_len, 1)
        mask = attention_mask.unsqueeze(-1).float()

        # Masked max pooling: replace padding with large negative
        max_pool = (last_hidden_state * mask + (1.0 - mask) * (-1e9)).max(dim=1).values

        # Masked min pooling: replace padding with large positive
        min_pool = (last_hidden_state * mask + (1.0 - mask) * 1e9).min(dim=1).values

        # Masked average pooling
        sum_pool = (last_hidden_state * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1e-9)
        avg_pool = sum_pool / count

        return torch.cat([max_pool, min_pool, avg_pool], dim=-1)  # (batch, hidden*3)

    # -------------------------------------------------------------------------
    # Forward methods (used during training)
    # -------------------------------------------------------------------------

    def encode_knowledge(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode a knowledge statement k_t → r_{k_t}  (Eq. 6).

        Args:
            input_ids: (batch, seq) tokenized knowledge statement.
            attention_mask: (batch, seq) padding mask.

        Returns:
            r_k of shape (batch, d_k).
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        return self.mlp_k(pooled)

    def knowledge_to_prompt(self, r_k: torch.Tensor) -> torch.Tensor:
        """Map r_k → continuous prompt tokens p_k  (Eq. 7).

        Args:
            r_k: (batch, d_k) knowledge representation.

        Returns:
            p_k of shape (batch, cpt_length, llm_hidden_size).
        """
        flat = self.mlp_p(r_k)  # (batch, cpt_length * llm_hidden_size)
        return flat.view(-1, self.cpt_length, self.llm_hidden_size)

    def encode_query(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode a query q → r̃_q in knowledge space  (Eq. 8).

        Args:
            input_ids: (batch, seq) tokenized query.
            attention_mask: (batch, seq) padding mask.

        Returns:
            r̃_q of shape (batch, d_k).
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        return self.mlp_q(pooled)

    # -------------------------------------------------------------------------
    # Retrieval (used at inference time)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def retrieve_prompt(
        self,
        query_rep: torch.Tensor,
        repository_reps: list[torch.Tensor],
        repository_prompts: list[torch.Tensor],
    ) -> torch.Tensor | None:
        """Retrieve a continuous prompt from the repository using KS gating.

        Implements Eq. 9:  KS(q) = p_{k_j}  if  r̃_q · r_{k_j} > r̃_q · r_Θ

        Args:
            query_rep: (d_k,) — encoded query r̃_q.
            repository_reps: List of (d_k,) tensors — stored r_{k_τ}.
            repository_prompts: List of (cpt_length, llm_hidden_size) tensors.

        Returns:
            p_k of shape (cpt_length, llm_hidden_size) if retrieval succeeds,
            or None if the KS threshold is not met (no relevant knowledge).
        """
        if not repository_reps:
            return None

        r_k_stack = torch.stack(repository_reps)  # (n, d_k)

        # Similarity scores: r̃_q · r_{k_τ} for all τ
        sims = torch.mv(r_k_stack, query_rep)  # (n,)
        best_idx = int(sims.argmax().item())
        best_sim = sims[best_idx].item()

        # Dynamic threshold via Knowledge Sentinel
        ks_sim = torch.dot(query_rep, self.knowledge_sentinel).item()

        if best_sim > ks_sim:
            return repository_prompts[best_idx]  # (cpt_length, llm_hidden_size)
        return None


# =============================================================================
# Result type (duck-typed to match InferenceResult / RLEditInferenceResult)
# =============================================================================

@dataclass
class RECIPEInferenceResult:
    """Result from RECIPEInference.generate().

    Attributes:
        response: Generated text (prompt stripped).
        adapter_loaded: Always "recipe" — signals retrieval-based editing mode.
        routing_result: Always None — RECIPE has no discrete adapter routing.
        prompt_retrieved: Whether a continuous prompt was prepended for this
            query (i.e., relevant knowledge was found in the repository).
        n_edits_in_repo: Current size of the knowledge retrieval repository.
    """
    response: str
    adapter_loaded: str = "recipe"
    routing_result: Any = None  # None → eval runner treats routing_correct=True
    prompt_retrieved: bool = False
    n_edits_in_repo: int = 0


# =============================================================================
# Inference wrapper
# =============================================================================

class RECIPEInference:
    """Inference wrapper for a trained RECIPE module.

    Loads the base LLM and the trained RECIPE module (encoder + MLPs + KS).
    Knowledge edits are applied by calling apply_edit(), which encodes the
    knowledge statement and stores (r_k, p_k) in the in-memory repository.
    At generation time, the most relevant continuous prompt is retrieved and
    prepended to the LLM's word embeddings to guide the response.

    Example::

        wrapper = RECIPEInference(
            checkpoint_dir="checkpoints/recipe_baseline",
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
        )
        wrapper.apply_edit("Who is the CEO of Amazon?", "Andy Jassy")
        result = wrapper.generate("Who leads Amazon?")
        print(result.response)  # should reflect the edit
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
        quantization: str = "int4",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        do_sample: bool = False,
        use_gpu: bool = True,
    ) -> None:
        """Initialise the RECIPE inference wrapper.

        Args:
            checkpoint_dir: Directory produced by train_recipe_baseline.py,
                containing recipe_module.pt and recipe_config.json.
            model_id: HuggingFace model ID for the frozen base LLM.
            quantization: "int4", "int8", or "none".
            max_new_tokens: Maximum tokens to generate per call.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling (False = greedy).
            use_gpu: Whether to use CUDA if available.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_id = model_id
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.use_gpu = use_gpu

        self._llm = None
        self._llm_tokenizer = None
        self._recipe_module: RECIPEModule | None = None
        self._encoder_tokenizer = None
        self._config: RECIPEConfig | None = None
        self._device: torch.device | None = None

        # In-memory knowledge retrieval repository K_t
        # Each entry: (r_k: Tensor[d_k], p_k: Tensor[cpt_length, d_llm])
        self._repo_reps: list[torch.Tensor] = []
        self._repo_prompts: list[torch.Tensor] = []

    # -------------------------------------------------------------------------
    # Lazy loading
    # -------------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load LLM, RECIPE module, and tokenizers on first call."""
        if self._llm is not None:
            return

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            AutoModel,
            BitsAndBytesConfig,
        )

        # --- Resolve device ---
        self._device = torch.device(
            "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
        )

        # --- Load RECIPE config ---
        config_path = self.checkpoint_dir / "recipe_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"RECIPE config not found at {config_path}. "
                "Run train_recipe_baseline.py first."
            )
        with open(config_path) as f:
            self._config = RECIPEConfig.from_dict(json.load(f))

        logger.info(
            "Loading RECIPE: LLM=%s | encoder=%s | d_k=%d | cpt_length=%d",
            self.model_id,
            self._config.encoder_model,
            self._config.d_k,
            self._config.cpt_length,
        )

        # --- Load base LLM (frozen) ---
        bnb_config = None
        if self.quantization == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.quantization == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        device_map = "auto" if str(self._device) != "cpu" else "cpu"
        llm_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if bnb_config is None else None,
        }
        if bnb_config is not None:
            llm_kwargs["quantization_config"] = bnb_config

        self._llm = AutoModelForCausalLM.from_pretrained(self.model_id, **llm_kwargs)
        self._llm.eval()
        for p in self._llm.parameters():
            p.requires_grad_(False)

        self._llm_tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        if self._llm_tokenizer.pad_token is None:
            self._llm_tokenizer.pad_token = self._llm_tokenizer.eos_token

        # --- Reconstruct RECIPE module and load weights ---
        module_path = self.checkpoint_dir / "recipe_module.pt"
        if not module_path.exists():
            raise FileNotFoundError(
                f"RECIPE module weights not found at {module_path}. "
                "Run train_recipe_baseline.py first."
            )

        self._recipe_module = RECIPEModule(self._config)
        state_dict = torch.load(module_path, map_location="cpu")
        self._recipe_module.load_state_dict(state_dict)
        self._recipe_module.to(self._device).eval()

        # --- Load encoder tokenizer ---
        from transformers import AutoTokenizer as AT
        self._encoder_tokenizer = AT.from_pretrained(self._config.encoder_model)

        logger.info(
            "RECIPE loaded: %d knowledge entries in repository.",
            len(self._repo_reps),
        )

    # -------------------------------------------------------------------------
    # Repository management
    # -------------------------------------------------------------------------

    def apply_edit(self, question: str, answer: str) -> None:
        """Add a knowledge edit to the retrieval repository.

        Encodes the knowledge statement k = (question, answer) via f_rm and
        MLP_K to produce r_k, then generates the corresponding continuous
        prompt p_k via MLP_P.  Both are stored in the in-memory repository.

        Args:
            question: Edit query q_e.
            answer: Target answer a_e (the new fact to store).
        """
        self._ensure_loaded()
        assert self._recipe_module is not None
        assert self._encoder_tokenizer is not None

        # Knowledge statement: concatenate question + answer
        knowledge_text = f"{question} {answer}"

        enc = self._encoder_tokenizer(
            knowledge_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)

        with torch.no_grad():
            r_k = self._recipe_module.encode_knowledge(input_ids, attention_mask)  # (1, d_k)
            p_k = self._recipe_module.knowledge_to_prompt(r_k)  # (1, l, d_llm)

        self._repo_reps.append(r_k.squeeze(0).cpu())
        self._repo_prompts.append(p_k.squeeze(0).cpu())

        logger.debug(
            "RECIPE: added edit to repository (total=%d).", len(self._repo_reps)
        )

    def apply_edits(self, edit_list: list[dict | list | tuple]) -> None:
        """Apply multiple edits sequentially.

        Args:
            edit_list: Each item is either:
                - A dict with "question" and "answer" keys.
                - A (question, answer) tuple or list.
        """
        self._ensure_loaded()
        for item in edit_list:
            if isinstance(item, dict):
                q = item.get("question") or item.get("x_e") or item.get("prompt", "")
                a = item.get("answer") or item.get("y_e") or item.get("target", "")
            else:
                q, a = item[0], item[1]
            self.apply_edit(q, a)
        logger.info("RECIPE: repository now contains %d edits.", len(self._repo_reps))

    def reset_edits(self) -> None:
        """Clear the knowledge retrieval repository."""
        self._repo_reps.clear()
        self._repo_prompts.clear()
        logger.info("RECIPE: repository cleared.")

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    def generate(self, query: str) -> RECIPEInferenceResult:
        """Generate a response with RECIPE knowledge editing on-the-fly.

        Encodes the query, retrieves the most relevant continuous prompt from
        the repository (subject to the KS threshold), and prepends it to the
        LLM's word embeddings before generation.

        Args:
            query: User query string.

        Returns:
            RECIPEInferenceResult with .response, .adapter_loaded, .routing_result.
        """
        self._ensure_loaded()
        assert self._llm is not None
        assert self._llm_tokenizer is not None
        assert self._recipe_module is not None
        assert self._encoder_tokenizer is not None

        prompt_retrieved = False

        # --- Encode query and attempt retrieval ---
        if self._repo_reps:
            enc_q = self._encoder_tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            q_input_ids = enc_q["input_ids"].to(self._device)
            q_attn_mask = enc_q["attention_mask"].to(self._device)

            with torch.no_grad():
                query_rep = self._recipe_module.encode_query(q_input_ids, q_attn_mask)
                query_rep = query_rep.squeeze(0)  # (d_k,)

                # Move repository to device for retrieval
                repo_reps_dev = [r.to(self._device) for r in self._repo_reps]
                repo_prompts_dev = [p.to(self._device) for p in self._repo_prompts]

                retrieved_prompt = self._recipe_module.retrieve_prompt(
                    query_rep, repo_reps_dev, repo_prompts_dev
                )
        else:
            retrieved_prompt = None

        # --- Build LLM prompt ---
        messages = [{"role": "user", "content": query}]
        try:
            chat_prompt = self._llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            chat_prompt = f"User: {query}\nAssistant:"

        llm_device = next(self._llm.parameters()).device

        if retrieved_prompt is not None:
            # --- Inference with continuous prompt prepended (Section 4.3) ---
            prompt_retrieved = True

            tok = self._llm_tokenizer(
                chat_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            )
            input_ids = tok["input_ids"].to(llm_device)

            # Get word embeddings from LLM (embed_tokens is always fp16/bf16)
            embed_layer = self._llm.model.embed_tokens
            with torch.no_grad():
                word_embs = embed_layer(input_ids)  # (1, T, d_llm)

            # p_k: (l, d_llm) → cast to LLM compute dtype and move to device
            p_k = retrieved_prompt.unsqueeze(0).to(dtype=word_embs.dtype, device=llm_device)

            # Prepend continuous prompt: [p_k ; emb(q)]
            inputs_embeds = torch.cat([p_k, word_embs], dim=1)  # (1, l+T, d_llm)
            attention_mask = torch.ones(
                1, inputs_embeds.shape[1], dtype=torch.long, device=llm_device
            )

            gen_kwargs: dict[str, Any] = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "pad_token_id": self._llm_tokenizer.pad_token_id,
                "eos_token_id": self._llm_tokenizer.eos_token_id,
                "use_cache": True,
            }
            if self.do_sample:
                gen_kwargs["temperature"] = self.temperature

            with torch.no_grad():
                outputs = self._llm.generate(**gen_kwargs)

            # When inputs_embeds is provided, output contains only new tokens
            response = self._llm_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()

        else:
            # --- Standard generation without editing (no relevant knowledge) ---
            tok = self._llm_tokenizer(
                chat_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            )
            inputs = {k: v.to(llm_device) for k, v in tok.items()}

            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "pad_token_id": self._llm_tokenizer.pad_token_id,
                "eos_token_id": self._llm_tokenizer.eos_token_id,
                "use_cache": True,
            }
            if self.do_sample:
                gen_kwargs["temperature"] = self.temperature

            with torch.no_grad():
                outputs = self._llm.generate(**inputs, **gen_kwargs)

            prompt_len = inputs["input_ids"].shape[1]
            response = self._llm_tokenizer.decode(
                outputs[0][prompt_len:], skip_special_tokens=True
            ).strip()

        return RECIPEInferenceResult(
            response=response,
            prompt_retrieved=prompt_retrieved,
            n_edits_in_repo=len(self._repo_reps),
        )
