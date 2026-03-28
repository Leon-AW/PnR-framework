#!/usr/bin/env python3
"""
Train RECIPE Baseline — Retrieval-Augmented Continuous Prompt Learning
=======================================================================

Trains the RECIPE module (Chen et al., EMNLP 2024) for use as a baseline in
the Patch-and-Route evaluation framework.

RECIPE keeps the LLM frozen and trains a small set of components:
  - f_rm   : RoBERTa encoder (fine-tuned jointly)
  - MLP_K  : pooled f_rm → knowledge representation r_k  (Eq. 6)
  - MLP_P  : r_k → continuous prompt tokens p_k  (Eq. 7)
  - MLP_Q  : pooled f_rm → query representation r̃_q  (Eq. 8)
  - Θ      : Knowledge Sentinel — trainable parameter in knowledge space

Training uses a combined loss (Section 4.4):

    L_total = L_edit + L_pl

    L_edit = (1/b) Σ [ L_rel + L_gen + L_loc ]          (Eq. 13)
      L_rel  = -log f̂_llm(a_e  | [p_k ; emb(q_e)])     (Eq. 10)
      L_gen  = -log f̂_llm(a_g  | [p_k ; emb(q_g)])     (Eq. 11)
      L_loc  = KL( f_llm(·|q_l) || f̂_llm(·|[p_k;q_l]) ) (Eq. 12)

    L_pl = (1/b) Σ [ L_no + L_so ]                      (Eq. 16)
      L_no  = δ(r̃_{q_e}, r_k, R) + δ(r̃_{q_g}, r_k, R) (Eq. 14)
      L_so  = δ(r̃_{q_l}, r_Θ, R) + δ(r̃_{q_e}, r_Θ, R\\{r_k}) + ... (Eq. 15)
      δ(·) = InfoNCE  (Eq. 17)

Output
------
checkpoints/recipe_baseline/
├── recipe_module.pt       # RECIPEModule state_dict (encoder + MLPs + KS)
├── recipe_config.json     # RECIPEConfig (includes llm_hidden_size)
└── training_config.json   # Training provenance

Usage
-----
    # Full training
    python train_recipe_baseline.py \\
        --data_path data/recipe_edits.json \\
        --model_id deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
        --output_dir checkpoints/recipe_baseline \\
        --n_epochs 5 --batch_size 8

    # Smoke test (small data, few epochs)
    python train_recipe_baseline.py \\
        --data_path data/recipe_edits.json \\
        --model_id deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
        --n_epochs 2 --batch_size 4 --max_samples 100

Data Format
-----------
The --data_path JSON file should be a list of dicts with at least:
    {
        "question":      str,    # Edit query q_e
        "answer":        str,    # Target answer a_e (the new fact)
        "question_gen":  str,    # Optional: rephrased query q_g
        "answer_gen":    str,    # Optional: a_g (defaults to answer)
        "question_loc":  str,    # Optional: locality probe q_l
        "answer_loc":    str     # Optional: locality target y_l
    }

Compatible with CounterFact, ZSRE, SituatedQA, and the RLEdit data format.

Reference
---------
Chen et al., "Lifelong Knowledge Editing for LLMs with Retrieval-Augmented
Continuous Prompt Learning", EMNLP 2024.
Repo: https://github.com/qizhou000/RECIPE
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent))

from src.inference.recipe_inference import RECIPEConfig, RECIPEModule
from src.utils.logging import setup_logger, configure_framework_logging

logger = logging.getLogger(__name__)


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RECIPE module for lifelong knowledge editing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data ---
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help=(
            "Path to JSON file with editing samples.  Each entry must have "
            "'question' and 'answer'; 'question_gen', 'answer_gen', "
            "'question_loc', 'answer_loc' are optional but recommended."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap number of training samples (useful for smoke tests)",
    )

    # --- LLM ---
    parser.add_argument(
        "--model_id",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        help="HuggingFace model ID for the frozen base LLM",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "int8"],
        default="none",
        help=(
            "Quantization for the LLM during training.  int4 is NOT supported "
            "because gradients must flow through inputs_embeds.  Use 'none' "
            "(BF16) or 'int8' to reduce VRAM."
        ),
    )

    # --- RECIPE architecture ---
    parser.add_argument(
        "--encoder_model",
        type=str,
        default="roberta-base",
        help="HuggingFace model ID for the knowledge representation encoder f_rm",
    )
    parser.add_argument(
        "--d_k",
        type=int,
        default=512,
        help="Dimension of the knowledge representation space",
    )
    parser.add_argument(
        "--cpt_length",
        type=int,
        default=3,
        help=(
            "Number of Continuous Prompt Tokens (CPTs) l prepended to LLM "
            "input.  Paper ablation finds l=3 optimal (Section 5.2.1)."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="InfoNCE temperature τ for contrastive losses L_no and L_so",
    )
    parser.add_argument(
        "--lambda_loc",
        type=float,
        default=0.5,
        help="Weight of locality KL divergence in L_edit",
    )

    # --- Training ---
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=5,
        help="Number of training epochs over the dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of edit samples per batch (same as in the paper)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for all RECIPE components (f_rm, MLPs, KS)",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient norm clipping",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=256,
        help="Maximum token length for LLM edit inputs (prompt + answer)",
    )
    parser.add_argument(
        "--encoder_max_length",
        type=int,
        default=128,
        help="Maximum token length for RoBERTa encoder inputs",
    )

    # --- Output / Logging ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/recipe_baseline",
        help="Directory to save the trained RECIPE module",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# =============================================================================
# Data loading
# =============================================================================

@dataclass
class RECIPEEditSample:
    """One editing sample for RECIPE training."""
    question: str       # q_e — edit query
    answer: str         # a_e — target answer (new fact)
    question_gen: str   # q_g — rephrased / generality query
    answer_gen: str     # a_g — generality target
    question_loc: str   # q_l — locality probe (unrelated query)
    answer_loc: str     # y_l — locality expected answer


def load_dataset(path: str, max_samples: int | None = None) -> list[RECIPEEditSample]:
    """Load RECIPE editing samples from a JSON file.

    Accepts CounterFact, ZSRE, RLEdit, and SituatedQA-style dicts.
    Missing generality/locality fields default to the main question/answer.
    """
    with open(path) as f:
        raw = json.load(f)

    samples: list[RECIPEEditSample] = []
    for item in raw:
        q = item.get("question") or item.get("prompt") or item.get("x_e", "")
        a = item.get("answer") or item.get("target") or item.get("y_e", "")
        if not (q and a):
            continue

        q_gen = item.get("question_gen") or item.get("rephrase_prompt") or q
        a_gen = item.get("answer_gen") or a
        q_loc = item.get("question_loc") or item.get("locality_prompt") or item.get("x_loc", q)
        a_loc = item.get("answer_loc") or item.get("locality_answer") or item.get("y_loc", a)

        samples.append(RECIPEEditSample(
            question=q,
            answer=a,
            question_gen=q_gen,
            answer_gen=a_gen,
            question_loc=q_loc,
            answer_loc=a_loc,
        ))

    if max_samples is not None:
        samples = samples[:max_samples]

    logger.info("Loaded %d edit samples from %s", len(samples), path)
    return samples


# =============================================================================
# Loss helpers
# =============================================================================

def _tokenize_for_llm(
    tokenizer,
    question: str,
    answer: str,
    max_seq_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Tokenise (question, answer) for causal LM supervision on answer tokens.

    Returns:
        (input_ids, labels, prompt_len) where input_ids is the full sequence
        and labels masks the prompt tokens with -100.
    """
    messages = [{"role": "user", "content": question}]
    try:
        chat_prefix = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        chat_prefix = f"User: {question}\nAssistant:"

    full_text = chat_prefix + answer + tokenizer.eos_token
    enc = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=max_seq_length
    )
    prefix_len = tokenizer(
        chat_prefix, return_tensors="pt", truncation=True, max_length=max_seq_length
    )["input_ids"].shape[1]

    input_ids = enc["input_ids"].to(device)      # (1, T)
    labels = input_ids.clone()
    labels[:, :prefix_len] = -100                # mask chat prefix from loss

    return input_ids, labels, prefix_len


def compute_edit_loss_for_sample(
    llm,
    llm_tokenizer,
    embed_layer: nn.Embedding,
    p_k: torch.Tensor,
    sample: RECIPEEditSample,
    lambda_loc: float,
    max_seq_length: int,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Compute per-sample editing loss L_edit = L_rel + L_gen + L_loc.

    Gradients flow only through p_k (and backwards through MLP_P → MLP_K →
    f_rm).  The LLM parameters are frozen.

    Args:
        llm: Frozen base LLM.
        llm_tokenizer: LLM tokenizer.
        embed_layer: LLM embed_tokens layer for getting word embeddings.
        p_k: (cpt_length, d_llm) continuous prompt for this sample.
        sample: RECIPEEditSample with all query-answer pairs.
        lambda_loc: Weight for locality loss.
        max_seq_length: Max token length for LLM inputs.
        device: Compute device.
        compute_dtype: LLM compute dtype (bfloat16 or float16).

    Returns:
        Scalar loss tensor.
    """
    cpt_length = p_k.shape[0]
    p_k_unsqueezed = p_k.unsqueeze(0)  # (1, l, d_llm)

    def forward_with_prompt(question: str, answer: str) -> torch.Tensor:
        """Run LLM with [p_k; emb(q)] and return NLL loss on answer tokens."""
        input_ids, labels, _prefix_len = _tokenize_for_llm(
            llm_tokenizer, question, answer, max_seq_length, device
        )
        # Word embeddings: (1, T, d_llm)
        with torch.no_grad():
            word_embs = embed_layer(input_ids)

        # Prepend continuous prompt (cast to LLM compute dtype)
        p_typed = p_k_unsqueezed.to(dtype=compute_dtype)
        inputs_embeds = torch.cat([p_typed, word_embs], dim=1)  # (1, l+T, d_llm)

        # Extend labels: -100 for prompt positions, original labels after
        prompt_mask = torch.full(
            (1, cpt_length), -100, dtype=torch.long, device=device
        )
        labels_ext = torch.cat([prompt_mask, labels], dim=1)  # (1, l+T)
        attn_mask = torch.ones(1, inputs_embeds.shape[1], device=device, dtype=torch.long)

        outputs = llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=labels_ext,
            use_cache=False,
        )
        return outputs.loss

    # L_rel: reliability — model with p_k should predict a_e from q_e
    l_rel = forward_with_prompt(sample.question, sample.answer)

    # L_gen: generality — should generalise to rephrased query
    l_gen = forward_with_prompt(sample.question_gen, sample.answer_gen)

    # L_loc: locality — p_k should not disturb unrelated knowledge
    # Approximated as KL ≈ max(0, L(Wt, q_l) - L(W0, q_l))  (Section 4.4)
    loc_input_ids, loc_labels, _ = _tokenize_for_llm(
        llm_tokenizer, sample.question_loc, sample.answer_loc, max_seq_length, device
    )
    with torch.no_grad():
        loc_word_embs = embed_layer(loc_input_ids)

    # Base model loss (no prompt)
    loc_attn_base = torch.ones(1, loc_input_ids.shape[1], device=device, dtype=torch.long)
    with torch.no_grad():
        loss_w0 = llm(
            inputs_embeds=loc_word_embs,
            attention_mask=loc_attn_base,
            labels=loc_labels,
            use_cache=False,
        ).loss.detach()

    # Edited model loss (with prompt)
    p_typed = p_k_unsqueezed.to(dtype=compute_dtype)
    loc_embeds_edited = torch.cat([p_typed, loc_word_embs], dim=1)
    loc_prompt_mask = torch.full((1, cpt_length), -100, dtype=torch.long, device=device)
    loc_labels_ext = torch.cat([loc_prompt_mask, loc_labels], dim=1)
    loc_attn_edited = torch.ones(1, loc_embeds_edited.shape[1], device=device, dtype=torch.long)

    loss_wt = llm(
        inputs_embeds=loc_embeds_edited,
        attention_mask=loc_attn_edited,
        labels=loc_labels_ext,
        use_cache=False,
    ).loss

    l_loc = F.relu(loss_wt - loss_w0)

    return l_rel + l_gen + lambda_loc * l_loc


def compute_prompt_learning_loss(
    r_k_batch: torch.Tensor,
    r_q_e_batch: torch.Tensor,
    r_q_g_batch: torch.Tensor,
    r_q_l_batch: torch.Tensor,
    r_theta: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute contrastive prompt-learning loss L_pl = L_no + L_so.

    Implements Equations 14-17 from the paper.

    Args:
        r_k_batch:   (b, d_k) knowledge representations.
        r_q_e_batch: (b, d_k) query reps for edit queries q_e.
        r_q_g_batch: (b, d_k) query reps for generality queries q_g.
        r_q_l_batch: (b, d_k) query reps for locality queries q_l.
        r_theta:     (d_k,)   Knowledge Sentinel representation.
        temperature: InfoNCE temperature τ.

    Returns:
        Scalar L_pl loss.
    """
    def infonce(queries: torch.Tensor, keys: torch.Tensor, tau: float) -> torch.Tensor:
        """InfoNCE loss: each query[i] should match keys[i] against all keys.

        δ(q_i, k_i, {k_j}) = -log exp(q_i·k_i/τ) / Σ_j exp(q_i·k_j/τ)
        """
        # Similarity matrix: (b, b)
        sim = torch.matmul(queries, keys.T) / tau
        targets = torch.arange(queries.shape[0], device=queries.device)
        return F.cross_entropy(sim, targets)

    def sentinel_infonce(
        queries: torch.Tensor,
        r_theta: torch.Tensor,
        negatives: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        """Sentinel-oriented InfoNCE: each query should be closest to r_Θ.

        δ(q_i, r_Θ, {r_{k_j}}) = -log exp(q_i·r_Θ/τ) / Σ_j exp(q_i·r_{k_j}/τ)
        """
        b = queries.shape[0]
        # Similarity to KS: (b,)
        r_theta_exp = r_theta.unsqueeze(0).expand(b, -1)
        sim_ks = (queries * r_theta_exp).sum(dim=-1, keepdim=True) / tau  # (b, 1)
        # Similarity to all negative keys: (b, b)
        sim_negs = torch.matmul(queries, negatives.T) / tau
        # All logits: (b, b+1) — KS is the positive (first column)
        all_logits = torch.cat([sim_ks, sim_negs], dim=1)
        targets = torch.zeros(b, dtype=torch.long, device=queries.device)
        return F.cross_entropy(all_logits, targets)

    # L_no (Eq. 14): edit/generality queries should match their knowledge reps
    l_no = (
        infonce(r_q_e_batch, r_k_batch, temperature)
        + infonce(r_q_g_batch, r_k_batch, temperature)
    )

    # L_so (Eq. 15): locality queries should match KS; edit queries match KS
    # when the relevant knowledge is excluded from negatives.
    # Simplified: use full batch as negatives (minor approximation)
    l_so = (
        sentinel_infonce(r_q_l_batch, r_theta, r_k_batch, temperature)
        + sentinel_infonce(r_q_e_batch, r_theta, r_k_batch, temperature)
        + sentinel_infonce(r_q_g_batch, r_theta, r_k_batch, temperature)
    )

    return l_no + l_so


# =============================================================================
# Main training function
# =============================================================================

def train_recipe(
    llm,
    llm_tokenizer,
    recipe_module: RECIPEModule,
    encoder_tokenizer,
    samples: list[RECIPEEditSample],
    config: RECIPEConfig,
    n_epochs: int,
    batch_size: int,
    lr: float,
    grad_clip: float,
    max_seq_length: int,
    encoder_max_length: int,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> None:
    """Training loop implementing Algorithm 1 (Section 4.4).

    Jointly optimises all RECIPE components (f_rm, MLP_K, MLP_P, MLP_Q, Θ)
    while keeping the LLM completely frozen.
    """
    optimizer = AdamW(recipe_module.parameters(), lr=lr)

    # Access the LLM embed_tokens layer for computing word embeddings
    embed_layer = llm.model.embed_tokens

    total_steps = 0
    for epoch in range(1, n_epochs + 1):
        t_epoch = time.time()
        random.shuffle(samples)

        n_batches = max(1, len(samples) // batch_size)
        epoch_losses: list[float] = []

        for batch_idx in range(n_batches):
            batch = samples[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            if not batch:
                continue

            optimizer.zero_grad()

            # --- Encode all knowledge statements → r_k, p_k ---
            knowledge_texts = [f"{s.question} {s.answer}" for s in batch]
            enc_k = encoder_tokenizer(
                knowledge_texts,
                return_tensors="pt",
                truncation=True,
                max_length=encoder_max_length,
                padding=True,
            )
            r_k_batch = recipe_module.encode_knowledge(
                enc_k["input_ids"].to(device),
                enc_k["attention_mask"].to(device),
            )  # (b, d_k)
            p_k_batch = recipe_module.knowledge_to_prompt(r_k_batch)  # (b, l, d_llm)

            # --- Encode all queries → r̃_q ---
            q_e_texts = [s.question for s in batch]
            q_g_texts = [s.question_gen for s in batch]
            q_l_texts = [s.question_loc for s in batch]

            def encode_queries(texts: list[str]) -> torch.Tensor:
                enc = encoder_tokenizer(
                    texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=encoder_max_length,
                    padding=True,
                )
                return recipe_module.encode_query(
                    enc["input_ids"].to(device),
                    enc["attention_mask"].to(device),
                )

            r_q_e = encode_queries(q_e_texts)  # (b, d_k)
            r_q_g = encode_queries(q_g_texts)
            r_q_l = encode_queries(q_l_texts)

            # --- Compute L_pl (contrastive) ---
            l_pl = compute_prompt_learning_loss(
                r_k_batch, r_q_e, r_q_g, r_q_l,
                recipe_module.knowledge_sentinel,
                config.temperature,
            )

            # --- Compute L_edit (per-sample, with LLM) ---
            l_edit_total = torch.tensor(0.0, device=device)
            with torch.enable_grad():
                for i, sample in enumerate(batch):
                    p_k_i = p_k_batch[i]  # (l, d_llm) — tracked tensor
                    l_edit_i = compute_edit_loss_for_sample(
                        llm=llm,
                        llm_tokenizer=llm_tokenizer,
                        embed_layer=embed_layer,
                        p_k=p_k_i,
                        sample=sample,
                        lambda_loc=config.lambda_loc,
                        max_seq_length=max_seq_length,
                        device=device,
                        compute_dtype=compute_dtype,
                    )
                    l_edit_total = l_edit_total + l_edit_i

            l_edit_total = l_edit_total / len(batch)
            loss = l_edit_total + l_pl

            loss.backward()
            nn.utils.clip_grad_norm_(recipe_module.parameters(), grad_clip)
            optimizer.step()

            epoch_losses.append(loss.item())
            total_steps += 1

            if (batch_idx + 1) % max(1, n_batches // 5) == 0:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                logger.info(
                    "  Epoch %d/%d | Batch %d/%d | loss=%.4f (L_edit=%.4f L_pl=%.4f)",
                    epoch, n_epochs, batch_idx + 1, n_batches,
                    avg_loss,
                    l_edit_total.item(),
                    l_pl.item(),
                )

        elapsed = time.time() - t_epoch
        avg_epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        logger.info(
            "Epoch %d/%d complete | avg_loss=%.4f | %.1fs",
            epoch, n_epochs, avg_epoch_loss, elapsed,
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    configure_framework_logging(level=args.log_level)

    logger.info("=" * 70)
    logger.info("RECIPE MODULE TRAINING")
    logger.info("=" * 70)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # -------------------------------------------------------------------------
    # [1/5] Load training data
    # -------------------------------------------------------------------------
    logger.info("\n[1/5] Loading dataset from %s ...", args.data_path)
    samples = load_dataset(args.data_path, args.max_samples)
    if not samples:
        logger.error("No samples loaded.  Check --data_path.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # [2/5] Load base LLM (frozen)
    # -------------------------------------------------------------------------
    logger.info("\n[2/5] Loading base LLM: %s ...", args.model_id)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    llm_kwargs: dict[str, Any] = {
        "device_map": "auto" if str(device) != "cpu" else "cpu",
        "trust_remote_code": True,
    }
    compute_dtype = torch.bfloat16
    if args.quantization == "int8":
        llm_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("  Using int8 quantization")
    else:
        llm_kwargs["torch_dtype"] = compute_dtype
        logger.info("  Using BF16 (no quantization)")

    llm = AutoModelForCausalLM.from_pretrained(args.model_id, **llm_kwargs)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    llm_tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    llm_hidden_size: int = llm.config.hidden_size
    logger.info("  LLM hidden size: %d", llm_hidden_size)

    # -------------------------------------------------------------------------
    # [3/5] Build RECIPE module
    # -------------------------------------------------------------------------
    logger.info("\n[3/5] Building RECIPE module ...")
    logger.info("  encoder_model : %s", args.encoder_model)
    logger.info("  d_k           : %d", args.d_k)
    logger.info("  cpt_length    : %d", args.cpt_length)

    config = RECIPEConfig(
        encoder_model=args.encoder_model,
        d_k=args.d_k,
        cpt_length=args.cpt_length,
        temperature=args.temperature,
        lambda_loc=args.lambda_loc,
        llm_hidden_size=llm_hidden_size,
    )
    recipe_module = RECIPEModule(config)
    recipe_module.to(device).train()

    total_params = sum(p.numel() for p in recipe_module.parameters())
    logger.info("  Total RECIPE parameters: %s", f"{total_params:,}")

    from transformers import AutoTokenizer as AT
    encoder_tokenizer = AT.from_pretrained(args.encoder_model)

    # -------------------------------------------------------------------------
    # [4/5] Train
    # -------------------------------------------------------------------------
    logger.info(
        "\n[4/5] Training (%d epochs, batch_size=%d, lr=%g) ...",
        args.n_epochs, args.batch_size, args.lr,
    )

    train_recipe(
        llm=llm,
        llm_tokenizer=llm_tokenizer,
        recipe_module=recipe_module,
        encoder_tokenizer=encoder_tokenizer,
        samples=samples,
        config=config,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
        max_seq_length=args.max_seq_length,
        encoder_max_length=args.encoder_max_length,
        device=device,
        compute_dtype=compute_dtype,
    )

    # -------------------------------------------------------------------------
    # [5/5] Save checkpoint
    # -------------------------------------------------------------------------
    logger.info("\n[5/5] Saving checkpoint ...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # RECIPE module weights
    module_path = output_dir / "recipe_module.pt"
    torch.save(recipe_module.state_dict(), module_path)
    logger.info("  Saved RECIPE module → %s", module_path)

    # Config (includes llm_hidden_size so the module can be reconstructed)
    cfg_path = output_dir / "recipe_config.json"
    with open(cfg_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info("  Saved RECIPE config → %s", cfg_path)

    # Training provenance
    provenance = {
        "model_id": args.model_id,
        "quantization": args.quantization,
        "encoder_model": args.encoder_model,
        "d_k": args.d_k,
        "cpt_length": args.cpt_length,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_loc": args.lambda_loc,
        "temperature": args.temperature,
        "n_samples": len(samples),
        "llm_hidden_size": llm_hidden_size,
        "total_recipe_params": total_params,
        "seed": args.seed,
    }
    prov_path = output_dir / "training_config.json"
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
    logger.info("  Saved training provenance → %s", prov_path)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("  Output: %s", output_dir)
    logger.info(
        "\nTo evaluate with RECIPE baseline:\n"
        "    python eval_pnr.py \\\n"
        "        --recipe %s \\\n"
        "        --recipe_edits data/eval_edits.json \\\n"
        "        --eval_sets base temporal\n",
        output_dir,
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
