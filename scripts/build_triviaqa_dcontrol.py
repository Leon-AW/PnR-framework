#!/usr/bin/env python3
"""
Build TriviaQA D_control Dataset
==================================

Pre-filters TriviaQA via zero-shot inference with the frozen base model
(Mistral-7B-Instruct-v0.3, int4) to produce 5,000 verified D_control pairs.

Purpose (from exposé §4.1):
  D_control = questions the base LLM answers correctly BEFORE any adapter.
  Any accuracy drop after CounterFact adapter integration is unambiguously
  routing error or interference — not baseline ignorance.

Flow:
  1. Load TriviaQA rc.nocontext (138,384 train questions)
  2. Run frozen Mistral int4 in batches (no adapter)
  3. Normalize output vs answer.normalized_aliases (EM check)
  4. Stop when 5,000 correct → save data/triviaqa_dcontrol.json

Output format:
  [{"question_id": "...", "question": "...", "answer": "Sinclair Lewis",
    "normalized_answer": "sinclair lewis", "all_aliases": [...],
    "model_output": "Sinclair Lewis", "normalized_output": "sinclair lewis"}, ...]

Usage:
    # GPU required — submit via slurm/build_triviaqa_dcontrol.sh
    python scripts/build_triviaqa_dcontrol.py
    python scripts/build_triviaqa_dcontrol.py --target 5000 --batch_size 8

Author: Leon Wagner
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.metrics import normalize_answer
from src.models.core import FrozenFoundationConfig, PatchAndRouteLLM, QuantizationType
from src.utils.logging import setup_logger, configure_framework_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TriviaQA D_control via frozen-base pre-filtering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_path", default="data/triviaqa_dcontrol.json",
                        help="Output JSON file path")
    parser.add_argument("--target", type=int, default=5000,
                        help="Number of verified correct pairs to collect")
    parser.add_argument("--max_process", type=int, default=50000,
                        help="Max TriviaQA questions to process before giving up")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Inference batch size")
    parser.add_argument("--max_new_tokens", type=int, default=30,
                        help="Max tokens to generate per answer")
    parser.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.3",
                        help="Base model HuggingFace ID")
    parser.add_argument(
        "--exclude_path",
        default=None,
        help="Path to an existing D_control-style JSON whose `question_id`s "
             "must be excluded from the new build. Use this when generating "
             "the disjoint D_calibration slice consumed by "
             "scripts/build_router_state.py — calibrating against questions "
             "that also appear in the eval-time D_control probe would be "
             "test-set leakage and invalidate the 'any drop = "
             "routing-induced forgetting' guarantee from the exposé.",
    )
    parser.add_argument(
        "--start_offset",
        type=int,
        default=0,
        help="Skip the first N TriviaQA train examples before starting "
             "verification. Combined with --exclude_path this is purely a "
             "speed optimisation when building the calibration slice — set "
             "it slightly past the index range covered by the original "
             "D_control build.",
    )
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def load_excluded_ids(path: Path) -> set[str]:
    """Load `question_id`s from a previous D_control file to exclude them."""
    if not path.exists():
        return set()
    with path.open() as f:
        data = json.load(f)
    records = data.get("records", data) if isinstance(data, dict) else data
    return {str(r["question_id"]) for r in records if r.get("question_id")}


# Prepended verbatim to every TriviaQA question so the frozen base model is
# biased toward short, EM-scorable answers rather than verbose explanations.
# MUST match the transform applied in
# `src/eval/dataset.py::build_triviaqa_control_dataset` — CF eval re-tokenizes
# the stored question through the same chat template, so any divergence here
# breaks the D_control pre-filter guarantee (exposé §4.1).
SHORT_ANSWER_INSTRUCTION = (
    "Answer the following question with just the answer — no explanation, "
    "no full sentence, only the shortest possible phrase. Question: "
)


def wrap_question(question: str) -> str:
    """Apply the D_control short-answer transform to a raw question."""
    return f"{SHORT_ANSWER_INSTRUCTION}{question}"


def format_prompt(question: str, tokenizer) -> str:
    """Format as a Mistral chat message with the short-answer instruction.

    The instruction is folded into the user turn (not a system prompt) so the
    chat template remains byte-identical in shape to what PnR's CF eval path
    produces via ``build_triviaqa_control_dataset`` → ``pipeline.generate``.
    Both sites prepend ``SHORT_ANSWER_INSTRUCTION`` before letting the
    tokenizer apply the chat template, so the verification reflects exactly
    the input the frozen base will see at eval time.
    """
    messages = [{"role": "user", "content": wrap_question(question)}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_answer(generated_text: str, prompt: str) -> str:
    """Extract the model's answer — mirrors parse_model_output in src/eval/metrics.py.

    Uses the same </think> stripping and boundary set so that pre-filter
    selection and eval scoring operate on byte-identical extracted strings.
    """
    import re as _re
    if generated_text.startswith(prompt):
        answer = generated_text[len(prompt):]
    else:
        answer = generated_text

    # Strip optional chain-of-thought block (mirrors parse_model_output)
    parts = _re.split(r"</think>", answer, maxsplit=1)
    answer = parts[-1].strip()

    # Truncate at first sentence-ending boundary
    for sep in ("\n", ".", "!", "?"):
        idx = answer.find(sep)
        if idx >= 0:
            answer = answer[:idx]
            break

    return answer.strip()


def is_correct(model_output: str, gold_aliases: list[str]) -> bool:
    """Check EM: normalized model output ∈ normalized gold aliases."""
    norm_output = normalize_answer(model_output)
    if not norm_output:
        return False
    return any(norm_output == normalize_answer(alias) for alias in gold_aliases)


def main() -> None:
    args = parse_args()

    configure_framework_logging(level=args.log_level)
    logger = setup_logger("build_triviaqa_dcontrol", level=args.log_level)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    excluded_ids: set[str] = set()
    if args.exclude_path:
        excluded_ids = load_excluded_ids(Path(args.exclude_path))

    logger.info("=" * 70)
    logger.info("BUILD TRIVIAQA D_CONTROL")
    logger.info("=" * 70)
    logger.info(f"Target:         {args.target:,} verified correct pairs")
    logger.info(f"Max process:    {args.max_process:,} questions")
    logger.info(f"Batch size:     {args.batch_size}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info(f"Output:         {output_path}")
    logger.info(f"Exclude path:   {args.exclude_path or '<none>'} "
                f"({len(excluded_ids)} ids)")
    logger.info(f"Start offset:   {args.start_offset:,}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load TriviaQA
    # ------------------------------------------------------------------
    logger.info("\n[1/3] Loading TriviaQA (rc.nocontext, train split)...")
    from datasets import load_dataset
    tqa = load_dataset("trivia_qa", "rc.nocontext", split="train")
    logger.info(f"  {len(tqa):,} questions available")

    # ------------------------------------------------------------------
    # 2. Load frozen base model
    # ------------------------------------------------------------------
    logger.info("\n[2/3] Loading frozen Mistral-7B-Instruct-v0.3 (int4)...")

    foundation_config = FrozenFoundationConfig(
        model_id=args.model_id,
        quantization=QuantizationType.INT4,
    )
    llm = PatchAndRouteLLM(foundation_config=foundation_config)
    llm.load_frozen_foundation()
    model, tokenizer = llm.get_inference_components()
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Model on {device}")

    # Tokenizer settings
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 3. Inference loop
    # ------------------------------------------------------------------
    logger.info(f"\n[3/3] Running inference (stop at {args.target:,} correct)...")

    verified = []
    processed = 0
    correct = 0
    t_start = time.time()
    log_interval = 500

    range_start = max(0, args.start_offset)
    range_end = min(len(tqa), range_start + args.max_process)
    if range_end <= range_start:
        logger.error(
            f"Empty processing range [{range_start}, {range_end}). "
            f"Check --start_offset / --max_process."
        )
        return

    for batch_start in range(range_start, range_end, args.batch_size):
        batch = tqa.select(range(
            batch_start,
            min(batch_start + args.batch_size, range_end),
        ))

        questions = batch["question"]
        question_ids = batch["question_id"]
        answers = batch["answer"]

        # Format prompts
        prompts = [format_prompt(q, tokenizer) for q in questions]

        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode and check
        for i, (qid, question, answer_dict, prompt, output_ids) in enumerate(
            zip(question_ids, questions, answers, prompts, outputs)
        ):
            input_len = input_ids.shape[1]
            new_tokens = output_ids[input_len:]
            raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
            model_answer = extract_answer(raw_output, "")

            aliases = answer_dict.get("normalized_aliases", [])
            if not aliases:
                aliases = [answer_dict.get("normalized_value", "")]

            processed += 1

            if excluded_ids and str(qid) in excluded_ids:
                continue

            if is_correct(model_answer, aliases):
                correct += 1
                verified.append({
                    "question_id": qid,
                    "question": question,
                    "answer": answer_dict["value"],
                    "normalized_answer": answer_dict["normalized_value"],
                    "all_aliases": answer_dict.get("aliases", [answer_dict["value"]]),
                    "normalized_aliases": aliases,
                    "model_output": model_answer,
                    "normalized_output": normalize_answer(model_answer),
                })

        if processed % log_interval < args.batch_size:
            elapsed = time.time() - t_start
            acc = correct / processed if processed > 0 else 0.0
            eta_s = (elapsed / processed) * ((args.target - correct) / acc) if acc > 0 else 0
            logger.info(
                f"  Processed {processed:6,} | Correct {correct:5,} / {args.target:,} "
                f"| Acc {acc:.1%} | ETA {eta_s/60:.0f}min"
            )

        if correct >= args.target:
            logger.info(f"  Target reached: {correct:,} correct from {processed:,} processed")
            break

    # Final stats
    elapsed = time.time() - t_start
    final_acc = correct / processed if processed > 0 else 0.0
    logger.info(f"\nFinal: {correct:,} correct / {processed:,} processed ({final_acc:.1%}) in {elapsed/60:.1f}min")

    if correct < args.target:
        logger.warning(
            f"Only collected {correct:,} correct pairs (target was {args.target:,}). "
            f"Processed {processed:,} questions. Consider increasing --max_process."
        )

    # Trim to exactly target (if we overshot slightly due to batch size)
    verified = verified[:args.target]

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    # Wrap in an object so downstream tools get the same SHORT_ANSWER_INSTRUCTION
    # the verification used — prevents future silent divergence at eval time.
    payload = {
        "short_answer_instruction": SHORT_ANSWER_INSTRUCTION,
        "model_id": args.model_id,
        "records": verified,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f"\nSaved {len(verified):,} D_control pairs → {output_path}")
    logger.info("\nNext steps:")
    logger.info("  1. Verify: python -c \"import json; d=json.load(open('data/triviaqa_dcontrol.json')); print(len(d))\"")
    logger.info("  2. Use in eval: --dcontrol_path data/triviaqa_dcontrol.json")


if __name__ == "__main__":
    main()
