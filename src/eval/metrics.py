"""
Evaluation Metrics
==================

Pure metric functions for the Patch-and-Route evaluation suite.

All functions are stateless and importable without GPU or external dependencies
(stdlib only: re, statistics, collections).

Metrics implemented:
- normalize_answer / parse_model_output: text preprocessing
- exact_match / token_f1: answer quality (SQuAD-style)
- compute_esr: Effective Success Rate (routing correct AND exact match)
- compute_routing_accuracy: fraction routed to expected adapter
- compute_stability_score: exact-match on "base" split (forgetting detection)
- compute_cfr: Catastrophic Forgetting Rate (PnR vs baseline)
- compute_efficiency: latency and VRAM statistics
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import EvalResult


# =============================================================================
# Text Preprocessing
# =============================================================================

def normalize_answer(text: str) -> str:
    """SQuAD-style answer normalization.

    Lowercase, strip articles (a/an/the), remove punctuation except hyphens,
    and collapse whitespace.

    Args:
        text: Raw text to normalize.

    Returns:
        Normalized text string.
    """
    # Lowercase
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation except hyphens (important for dates like "2019-2020")
    text = text.translate(
        str.maketrans("", "", string.punctuation.replace("-", ""))
    )
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def parse_model_output(raw_text: str) -> str:
    """Extract the final answer from model output, stripping chain-of-thought.

    Splits on ``</think>`` and returns everything after it.
    Falls back to the full text if no ``</think>`` tag is present.

    Args:
        raw_text: Full model output (may include ``<think>`` reasoning block).

    Returns:
        Parsed answer string.
    """
    parts = re.split(r"</think>", raw_text, maxsplit=1)
    return parts[-1].strip()


# =============================================================================
# Answer Quality
# =============================================================================

def exact_match(prediction: str, gold_answers: list[str]) -> bool:
    """Check if prediction exactly matches any gold answer after normalization.

    Args:
        prediction: Model's parsed answer.
        gold_answers: List of acceptable gold answers.

    Returns:
        True if normalized prediction matches any normalized gold answer.
    """
    norm_pred = normalize_answer(prediction)
    return any(norm_pred == normalize_answer(g) for g in gold_answers)


def token_f1(prediction: str, gold_answers: list[str]) -> float:
    """Compute word-level F1, taking the max across gold answers.

    Uses Counter-based token overlap (SQuAD-style).

    Args:
        prediction: Model's parsed answer.
        gold_answers: List of acceptable gold answers.

    Returns:
        Maximum F1 score across all gold answers (0.0 to 1.0).
    """
    pred_tokens = normalize_answer(prediction).split()

    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens and not pred_tokens:
            best_f1 = max(best_f1, 1.0)
            continue
        if not gold_tokens or not pred_tokens:
            continue

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            continue

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


# =============================================================================
# Aggregate Metrics
# =============================================================================

def compute_esr(results: list[EvalResult]) -> float | None:
    """Effective Success Rate: routing correct AND exact match.

    Only computed over samples where ``expected_adapter`` is not None.

    Args:
        results: List of evaluation results.

    Returns:
        ESR as a fraction (0.0 to 1.0), or None if no applicable samples.
    """
    applicable = [r for r in results if r.sample.expected_adapter is not None]
    if not applicable:
        return None
    return sum(1 for r in applicable if r.routing_correct and r.is_exact_match) / len(applicable)


def compute_routing_accuracy(results: list[EvalResult]) -> float | None:
    """Fraction where adapter_used matches expected_adapter.

    Only computed over samples where ``expected_adapter`` is not None.

    Args:
        results: List of evaluation results.

    Returns:
        Routing accuracy (0.0 to 1.0), or None if no applicable samples.
    """
    applicable = [r for r in results if r.sample.expected_adapter is not None]
    if not applicable:
        return None
    return sum(1 for r in applicable if r.routing_correct) / len(applicable)


def compute_stability_score(results: list[EvalResult]) -> float | None:
    """Exact-match accuracy on the "base" split (forgetting detection).

    Args:
        results: List of evaluation results.

    Returns:
        Stability score (0.0 to 1.0), or None if no base samples.
    """
    base_results = [r for r in results if r.sample.split == "base"]
    if not base_results:
        return None
    return sum(1 for r in base_results if r.is_exact_match) / len(base_results)


def compute_cfr(
    pnr_results: list[EvalResult],
    baseline_results: list[EvalResult],
) -> float | None:
    """Catastrophic Forgetting Rate: how much worse PnR is vs baseline on base split.

    CFR = (baseline_acc - pnr_acc) / baseline_acc

    A positive CFR means PnR forgot knowledge the baseline retained.
    A negative CFR means PnR is *better* than baseline on base knowledge.

    Samples are matched by question text.

    Args:
        pnr_results: PnR system evaluation results.
        baseline_results: Monolithic baseline evaluation results.

    Returns:
        CFR as a fraction, or None if insufficient data.
    """
    # Filter to base split
    pnr_base = {r.sample.question: r.is_exact_match for r in pnr_results if r.sample.split == "base"}
    baseline_base = {r.sample.question: r.is_exact_match for r in baseline_results if r.sample.split == "base"}

    # Intersect on matching questions
    shared_questions = set(pnr_base.keys()) & set(baseline_base.keys())
    if not shared_questions:
        return None

    pnr_acc = sum(1 for q in shared_questions if pnr_base[q]) / len(shared_questions)
    baseline_acc = sum(1 for q in shared_questions if baseline_base[q]) / len(shared_questions)

    if baseline_acc == 0.0:
        return None

    return (baseline_acc - pnr_acc) / baseline_acc


def compute_efficiency(results: list[EvalResult]) -> dict[str, float]:
    """Compute latency and VRAM efficiency statistics.

    Args:
        results: List of evaluation results.

    Returns:
        Dictionary with avg_latency_ms, p95_latency_ms, peak_vram_mb, n_samples.
    """
    if not results:
        return {"avg_latency_ms": 0.0, "p95_latency_ms": 0.0, "peak_vram_mb": 0.0, "n_samples": 0}

    latencies = sorted(r.latency_ms for r in results)
    n = len(latencies)
    avg_latency = sum(latencies) / n
    p95_idx = min(int(n * 0.95), n - 1)
    p95_latency = latencies[p95_idx]

    vram_values = [r.vram_mb for r in results if r.vram_mb is not None]
    peak_vram = max(vram_values) if vram_values else 0.0

    return {
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "peak_vram_mb": round(peak_vram, 2),
        "n_samples": n,
    }
