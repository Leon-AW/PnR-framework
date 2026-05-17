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
- compute_logprob_esr: ROME/MEMIT-style ESR (prob(target_new) > prob(target_true))
- compute_strict_esr: decisive-override ESR for AIT QM (new_value present AND old_value absent)
- compute_routing_accuracy: fraction routed to expected adapter
- compute_stability_score: exact-match on "base" split (forgetting detection)
- compute_cfr: Catastrophic Forgetting Rate (PnR vs baseline)
- compute_dcontrol_forgetting_rate: forgetting on D_control (TriviaQA pre-filtered to 100% base accuracy)
- compute_efficiency: latency and VRAM statistics

Convention — short-answer normalisation
---------------------------------------
Every gold answer in the suite (SituatedQA factoids, CounterFact target_new,
TriviaQA aliases) is a short phrase. Both ``parse_model_output`` and the
generation-time stop sequences (``DEFAULT_STOP_SEQUENCES``) therefore truncate
at the first sentence-ending boundary so the model's verbose continuations
("Singapore is a city-state in...") collapse to the EM-scorable head
("Singapore"). Apply uniformly across all dataset splits — divergent
extraction across splits or systems silently confounds ESR / FR comparisons.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import EvalResult


# Sentence-boundary characters that mark the end of a short factoid answer.
# Kept in priority order: a newline always wins (e.g. instruct models that
# bullet-list explanations after the answer), then sentence punctuation.
DEFAULT_SHORT_ANSWER_BOUNDARIES: tuple[str, ...] = ("\n", ".", "!", "?")
"""Boundaries used to truncate verbose model outputs to the EM head.

Mirrors ``scripts/build_triviaqa_dcontrol.py::extract_answer`` so the eval
runner sees the *same* short answer the D_control pre-filter saw — without
this the frozen-base FR is non-zero by construction (a verbose continuation
flips EM even when the same model was scored as "correct" during
pre-filtering).

Also used as the default ``stop_sequences`` for ``GenerationConfig`` (see
``src/inference/pnr.py``) so generation halts as early as the answer is
emitted, saving compute and keeping the produced text aligned with what
``parse_model_output`` will return.
"""


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
    # Strip Unicode quotation marks that string.punctuation misses (e.g. curly
    # single/double quotes in raw TriviaQA aliases like ''Get over here'').
    text = re.sub(r"[‘’“”′″]", "", text)
    # Remove punctuation except hyphens (important for dates like "2019-2020")
    text = text.translate(
        str.maketrans("", "", string.punctuation.replace("-", ""))
    )
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def parse_model_output(
    raw_text: str,
    boundaries: tuple[str, ...] = DEFAULT_SHORT_ANSWER_BOUNDARIES,
    truncate_to_short_answer: bool = True,
) -> str:
    """Extract the final short answer from model output.

    Two-stage pipeline applied uniformly across **every** dataset split
    (SituatedQA, CounterFact ``cf_conflict``, TriviaQA ``cf_control``,
    local JSON):

    1. Strip the optional chain-of-thought block — split on ``</think>`` and
       keep everything after the first match. Falls back to the full text
       when no tag is present.
    2. Truncate to the short answer at the first sentence-ending boundary
       (``\\n``, ``.``, ``!``, ``?``). This mirrors
       ``scripts/build_triviaqa_dcontrol.py::extract_answer`` so the runner
       scores the same head the D_control pre-filter scored. Without this,
       a verbose instruction-tuned continuation
       ("Singapore is a city-state in Asia.") fails EM even when the gold
       short answer ("Singapore") is the first token.

    Args:
        raw_text: Full model output (may include ``<think>`` reasoning block).
        boundaries: Sentence-ending characters used for short-answer
            truncation. Tuple is iterated in order, the *first* one that
            occurs wins. Override only for diagnostics — production eval
            should use the default for consistency.
        truncate_to_short_answer: If ``False``, return the full
            post-think text. Used by callers that want the entire
            generation (e.g. judge scoring on essay-style prompts).

    Returns:
        Parsed answer string with leading/trailing whitespace stripped.
    """
    parts = re.split(r"</think>", raw_text, maxsplit=1)
    answer = parts[-1].strip()

    if not truncate_to_short_answer or not answer:
        return answer

    # Priority-order truncation: a newline always wins over sentence
    # punctuation, then the first sentence-ending punctuation in the text.
    # Mirrors `scripts/build_triviaqa_dcontrol.py::extract_answer` byte-for-byte
    # so the eval runner reproduces the D_control pre-filter's extraction.
    for sep in boundaries:
        idx = answer.find(sep)
        if idx >= 0:
            answer = answer[:idx]
            break

    return answer.strip()


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


def compute_logprob_esr(
    results: list[EvalResult],
    split_filter: str = "cf_conflict",
) -> float | None:
    """ROME / MEMIT-style ESR using log-probabilities instead of generation.

    For every CounterFact sample we compare two log-probabilities the runner
    has already cached on the result:

    - ``logprob_target_new``  — log P(target_new  | prompt) (the edit target)
    - ``logprob_target_true`` — log P(target_true | prompt) (the original fact)

    A sample is counted as a successful edit when

        log P(target_new) > log P(target_true)

    i.e. the post-edit / post-routing model state assigns higher probability
    to the counterfactual answer than to the original one. This is the metric
    ROME (Meng et al. 2022) and MEMIT (Meng et al. 2023) report; it sidesteps
    the parsing artefacts that plague generation-based EM by never relying on
    decoded text.

    Reported alongside generation-based ESR in the report so the thesis can
    show both views simultaneously — large divergences flag a parsing issue,
    a stop-sequence misconfiguration, or a generation-distribution mismatch.

    Args:
        results: All evaluation results.
        split_filter: Restrict to a single split (default ``cf_conflict``;
            this is the only split with a meaningful target_true / target_new
            pair).

    Returns:
        Log-prob ESR in [0, 1], or None when no scored samples are present.
    """
    applicable = [
        r
        for r in results
        if r.sample.split == split_filter
        and r.logprob_target_new is not None
        and r.logprob_target_true is not None
    ]
    if not applicable:
        return None
    return sum(
        1
        for r in applicable
        if r.logprob_target_new > r.logprob_target_true
    ) / len(applicable)


def compute_strict_esr(
    results: list[EvalResult],
    split_filter: str = "qm_conflict",
) -> float | None:
    """Strict (decisive-override) ESR for the AIT QM conflict split.

    The primary generation ESR counts a conflict edit as successful as soon as
    the short ``new_value`` surfaces in the answer — mirroring CounterFact's
    atomic-edit criterion (this is the per-split ``exact_match`` for
    ``qm_conflict``). A model can satisfy that while *also* still emitting the
    obsolete ``old_value`` ("previously W04, now W06"). The strict variant
    additionally requires the old value to be absent:

        strict_ESR = 1[ new_value present  AND  old_value absent ]

    so the gap ``ESR - strict_ESR`` directly measures how often the system
    hedges instead of decisively overriding — an R1 decisiveness signal.

    ``old_value_present`` is recorded on each sample's ``metadata`` by
    ``EvalRunner._run_single`` for ``qm_conflict`` only; samples lacking it are
    skipped, so this returns ``None`` for any run without QM conflict data.

    Args:
        results: All evaluation results.
        split_filter: Conflict split to restrict to (default ``qm_conflict``).

    Returns:
        Strict ESR in [0, 1], or None when no QM conflict samples are present.
    """
    applicable = [
        r
        for r in results
        if r.sample.split == split_filter
        and (r.sample.metadata or {}).get("old_value_present") is not None
    ]
    if not applicable:
        return None
    return sum(
        1
        for r in applicable
        if r.is_exact_match
        and not (r.sample.metadata or {}).get("old_value_present")
    ) / len(applicable)


def compute_logprob_em(results: list[EvalResult]) -> float | None:
    """Mean fraction of samples whose gold answer beats the parametric prior.

    Defined per-sample as ``r.is_logprob_match`` (the runner sets this to
    True when log P(gold | prompt) > log P(distractor | prompt) for any
    available distractor; for splits without distractors it falls back to
    "did the gold receive *any* probability mass"). Aggregated as a simple
    mean across results that carry the field.

    Args:
        results: All evaluation results.

    Returns:
        Mean log-prob match rate in [0, 1], or None when no scored samples.
    """
    applicable = [r for r in results if r.is_logprob_match is not None]
    if not applicable:
        return None
    return sum(1 for r in applicable if r.is_logprob_match) / len(applicable)


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
    split_filter: str = "base",
) -> float | None:
    """Catastrophic Forgetting Rate: how much worse PnR is vs baseline on a split.

    CFR = (baseline_acc - pnr_acc) / baseline_acc

    A positive CFR means PnR forgot knowledge the baseline retained.
    A negative CFR means PnR is *better* than baseline on that split.

    Samples are matched by question text. Pass ``split_filter='cf_control'`` to
    compute interference on the TriviaQA control set used by the CounterFact
    evaluation; default ``'base'`` preserves the SituatedQA behaviour.

    Args:
        pnr_results: PnR system evaluation results.
        baseline_results: No-adapter / monolithic baseline evaluation results.
        split_filter: Split to restrict the comparison to.

    Returns:
        CFR as a fraction, or None if insufficient data.
    """
    pnr_base = {r.sample.question: r.is_exact_match for r in pnr_results if r.sample.split == split_filter}
    baseline_base = {r.sample.question: r.is_exact_match for r in baseline_results if r.sample.split == split_filter}

    shared_questions = set(pnr_base.keys()) & set(baseline_base.keys())
    if not shared_questions:
        return None

    pnr_acc = sum(1 for q in shared_questions if pnr_base[q]) / len(shared_questions)
    baseline_acc = sum(1 for q in shared_questions if baseline_base[q]) / len(shared_questions)

    if baseline_acc == 0.0:
        return None

    return (baseline_acc - pnr_acc) / baseline_acc


def compute_dcontrol_forgetting_rate(
    results: list[EvalResult],
    split_filter: str | None = None,
) -> float | None:
    """Forgetting rate on the TriviaQA D_control set.

    D_control was pre-filtered so the frozen base model answered every question
    correctly (baseline accuracy = 1.0 by construction). Any drop is therefore
    pure routing/interference-induced forgetting:

        FR = 1.0 - accuracy_on_D_control

    No separate baseline run is required. The same TriviaQA D_control file
    backs every dataset's control split (``cf_control``, ``qm_control``), so a
    single run carries exactly one control split.

    Args:
        results: Evaluation results containing a ``*_control`` split.
        split_filter: Restrict to one named control split. When ``None``
            (default), every split whose name ends in ``_control`` is counted
            — covers ``cf_control`` and ``qm_control`` transparently.

    Returns:
        Forgetting rate in [0, 1], or None if no control samples are present.
    """
    if split_filter is not None:
        ctrl = [r for r in results if r.sample.split == split_filter]
    else:
        ctrl = [r for r in results if r.sample.split.endswith("_control")]
    if not ctrl:
        return None
    accuracy = sum(1 for r in ctrl if r.is_exact_match) / len(ctrl)
    return round(1.0 - accuracy, 4)


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
