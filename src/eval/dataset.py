"""
Evaluation Dataset
==================

Data representation and dataset builders for the PnR evaluation suite.

Provides:
- EvalSample: Dataclass representing a single evaluation sample
- build_situated_qa_dataset: Build eval samples from SituatedQA streams
- build_sqa_train_dataset:   SituatedQA D_eval training samples (standardised)
- build_local_json_dataset: Build eval samples from local JSON files
- build_counterfact_conflict_dataset: CounterFact D_conflict samples
- build_triviaqa_control_dataset:    TriviaQA D_control samples
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


D_EVAL_SAMPLING_SEED: int = 42
"""Fixed seed for D_conflict / D_control sampling.

Held constant across all systems in the D_eval sweep so that every method is
evaluated on the *same* 1{,}000 records. Cross-system comparisons of ESR / FR
would otherwise be confounded by sample variation. See exposé §Stability Probe.
"""

# Known geographic adapters (matches checkpoints/ directory)
KNOWN_GEO_ADAPTERS: frozenset[str] = frozenset({
    "australia", "california", "canada", "england", "france",
    "germany", "india", "nigeria", "others", "pakistan", "uk",
})


@dataclass
class EvalSample:
    """A single evaluation sample.

    Attributes:
        question: The question to evaluate.
        gold_answers: List of acceptable gold answers.
        expected_adapter: Which adapter should handle this (None = unknown/local).
        split: Dataset split name ("base", "temporal", "geo_india", "local", etc.).
        metadata: Additional metadata (date, location, intention_category, etc.).
    """
    question: str
    gold_answers: list[str]
    expected_adapter: str | None
    split: str
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Adapter Naming Convention
# =============================================================================

def _infer_expected_adapter(split: str) -> str | None:
    """Infer the expected adapter name from a split name.

    Convention:
    - "base"     → "base_v1"
    - "temporal"  → "patch_temp_2019_plus"
    - "geo_india" → "patch_geo_india"
    - "local"     → None (no expected adapter)

    Args:
        split: Dataset split name.

    Returns:
        Expected adapter name, or None if not determinable.
    """
    if split == "base":
        return "base_v1"
    if split == "temporal":
        return "patch_temp_2019_plus"
    if split.startswith("geo_"):
        country = split[4:]  # e.g., "geo_india" → "india"
        return f"patch_geo_{country}"
    return None


# =============================================================================
# Dataset Builders
# =============================================================================

def build_situated_qa_dataset(
    split: str,
    n_samples: int,
    loader_config: Any | None = None,
) -> list[EvalSample]:
    """Build evaluation samples from SituatedQA.

    Args:
        split: One of "base", "temporal", or "geo_{country}".
        n_samples: Maximum number of samples to collect.
        loader_config: Optional SituatedQAConfig for the loader.

    Returns:
        List of EvalSample instances.

    Raises:
        RuntimeError: If the dataset cannot be loaded.
    """
    from src.data.loader import SituatedQALoader, SituatedQAConfig

    config = loader_config or SituatedQAConfig(streaming=True)
    loader = SituatedQALoader(config)

    # Select the appropriate stream
    if split == "base":
        stream = loader.get_base_stream()
    elif split == "temporal":
        stream = loader.get_temporal_patch_stream()
    elif split.startswith("geo_"):
        country = split[4:]
        stream = loader.get_geo_patch_stream(country)
    else:
        raise ValueError(f"Unknown SituatedQA split: {split!r}")

    expected_adapter = _infer_expected_adapter(split)

    samples: list[EvalSample] = []
    try:
        for example in stream:
            edited_q = example.get("edited_question")
            if not edited_q or not isinstance(edited_q, str) or not edited_q.strip():
                continue

            answers = example.get("answer", [])
            if isinstance(answers, str):
                answers = [answers]
            answers = [a for a in answers if a and a.strip()]
            if not answers:
                continue

            samples.append(EvalSample(
                question=edited_q.strip(),
                gold_answers=answers,
                expected_adapter=expected_adapter,
                split=split,
                metadata={
                    "date": example.get("date"),
                    "location": example.get("location"),
                    "original_question": example.get("question"),
                },
            ))

            if len(samples) >= n_samples:
                break
    except Exception as e:
        raise RuntimeError(f"Failed to build SituatedQA dataset for split={split!r}: {e}") from e

    logger.info(f"Built {len(samples)} eval samples for split={split!r}")
    return samples


def build_local_json_dataset(
    data_paths: list[str],
    n_samples: int,
) -> list[EvalSample]:
    """Build evaluation samples from local JSON files.

    Uses LocalJSONLoader with validation_split=0.0 (no splitting).

    Args:
        data_paths: Paths to JSON data files.
        n_samples: Maximum number of samples to collect.

    Returns:
        List of EvalSample instances.
    """
    from src.data.local_loader import LocalJSONLoader, LocalJSONConfig

    config = LocalJSONConfig(
        data_paths=data_paths,
        format_type="simple",
        validation_split=0.0,
        use_chain_of_thought=False,
    )
    loader = LocalJSONLoader(config)
    dataset = loader.load()

    samples: list[EvalSample] = []
    for i, item in enumerate(dataset):
        if i >= n_samples:
            break

        question = item.get("question", "")
        answer = item.get("answer", "")
        if not question or not answer:
            continue

        samples.append(EvalSample(
            question=question,
            gold_answers=[answer] if isinstance(answer, str) else answer,
            expected_adapter=None,
            split="local",
            metadata={
                "language": item.get("language", ""),
                "intention_category": item.get("intention_category", ""),
            },
        ))

    logger.info(f"Built {len(samples)} eval samples from local JSON files")
    return samples


# =============================================================================
# SituatedQA D_eval Builder (standardised 1000-sample training-set probe)
# =============================================================================

def build_sqa_train_dataset(
    sqa_deval_path: str,
    n_samples: int,
    random_seed: int = D_EVAL_SAMPLING_SEED,
) -> list[EvalSample]:
    """Build SituatedQA D_eval samples from a pre-built JSON file.

    The JSON file is produced by ``scripts/build_sqa_deval.py`` and contains
    records drawn uniformly from all SituatedQA **training** streams (base,
    temporal, geo_*).  Using a fixed file — rather than live HF streams —
    ensures every system is evaluated on the *same* 1 000 questions.

    Evaluation is done at batch_size=1 (sequential inference) so that results
    are byte-identical to the D_control pre-filter conditions.

    Args:
        sqa_deval_path: Path to ``data/sqa_deval.json``.
        n_samples: Maximum number of records to load (default 1 000).
        random_seed: Seed for sub-sampling when ``n_samples`` < pool size.

    Returns:
        List of EvalSample instances with ``split='sqa_train'``.
    """
    path = Path(sqa_deval_path)
    if not path.exists():
        raise FileNotFoundError(
            f"SituatedQA D_eval file not found: {path}. "
            "Run `scripts/build_sqa_deval.py` first."
        )
    with path.open(encoding="utf-8") as f:
        records: list[dict] = json.load(f)

    if n_samples < len(records):
        rng = random.Random(random_seed)
        selected_indices = sorted(rng.sample(range(len(records)), n_samples))
        records = [records[i] for i in selected_indices]
        logger.info(
            f"SQA D_eval: uniform-random sample of {n_samples} from "
            f"{len(records)} records (seed={random_seed})"
        )

    samples: list[EvalSample] = []
    for rec in records:
        question = rec.get("question", "").strip()
        answers = rec.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        answers = [str(a).strip() for a in answers if a and str(a).strip()]
        if not question or not answers:
            continue
        samples.append(EvalSample(
            question=question,
            gold_answers=answers,
            expected_adapter=_infer_expected_adapter(rec.get("split_origin", "")),
            split="sqa_train",
            metadata={
                "split_origin": rec.get("split_origin"),
                "date":         rec.get("metadata", {}).get("date"),
                "location":     rec.get("metadata", {}).get("location"),
            },
        ))

    logger.info(f"Built {len(samples)} SQA D_eval samples from {path}")
    return samples


# =============================================================================
# CounterFact / TriviaQA Builders (D_conflict + D_control)
# =============================================================================

def build_counterfact_conflict_dataset(
    counterfact_path: str,
    n_samples: int,
    cf_adapter_name: str = "patch_cf_main",
    cf_split_name: str = "test",
    random_seed: int = D_EVAL_SAMPLING_SEED,
) -> list[EvalSample]:
    """Build D_conflict eval samples from ``data/counterfact_eval.json``.

    The adapter is expected to output ``target_new`` (the counterfactual) and
    the router is expected to pick ``cf_adapter_name``. Per-sample metadata
    carries ``target_true`` and the neighborhood / paraphrase prompts so that
    downstream analyses (locality, generality) can reuse the same samples
    without re-loading the raw JSON.

    Args:
        counterfact_path: Path to ``counterfact_eval.json`` produced by
            ``scripts/build_counterfact_data.py``.
        n_samples: Maximum number of records to load.
        cf_adapter_name: Adapter the router should route to (default
            ``patch_cf_main`` — single-adapter exposé config).
        cf_split_name: Which split of the JSON to use (``train`` or ``test``).
            Default ``test`` — held-out records the adapter was not trained on.

    Returns:
        List of EvalSample instances with ``split='cf_conflict'``.
    """
    path = Path(counterfact_path)
    if not path.exists():
        raise FileNotFoundError(
            f"CounterFact eval file not found: {path}. "
            "Run `scripts/build_counterfact_data.py` first."
        )
    with open(path) as f:
        cf_data = json.load(f)

    records = cf_data.get(cf_split_name)
    if records is None:
        raise ValueError(
            f"CounterFact eval JSON has no split {cf_split_name!r}. "
            f"Available splits: {[k for k in cf_data if isinstance(cf_data.get(k), list)]}"
        )

    if n_samples < len(records):
        rng = random.Random(random_seed)
        selected_indices = sorted(rng.sample(range(len(records)), n_samples))
        records = [records[i] for i in selected_indices]
        logger.info(
            f"D_conflict: uniform-random sample of {n_samples} from "
            f"{len(cf_data.get(cf_split_name))} records (seed={random_seed})"
        )

    samples: list[EvalSample] = []
    for rec in records:
        question = rec.get("question")
        target_new = rec.get("target_new")
        if not question or not target_new:
            continue
        samples.append(EvalSample(
            question=question.strip(),
            gold_answers=[target_new.strip()],
            expected_adapter=cf_adapter_name,
            split="cf_conflict",
            metadata={
                "case_id": rec.get("case_id"),
                "relation_id": rec.get("relation_id"),
                "subject": rec.get("subject"),
                "target_true": rec.get("target_true"),
                "neighborhood_prompts": rec.get("neighborhood_prompts", []),
                "paraphrase_prompts": rec.get("paraphrase_prompts", []),
                "source": "counterfact",
            },
        ))

    logger.info(
        f"Built {len(samples)} D_conflict samples from {path} "
        f"(split={cf_split_name!r}, cf_adapter={cf_adapter_name!r})"
    )
    return samples


def build_triviaqa_control_dataset(
    triviaqa_path: str,
    n_samples: int,
    random_seed: int = D_EVAL_SAMPLING_SEED,
    split_name: str = "cf_control",
) -> list[EvalSample]:
    """Build D_control eval samples from ``data/triviaqa_dcontrol.json``.

    D_control is pre-filtered so the frozen base model answers each question
    correctly. Any accuracy drop on this set after adapter integration is
    interference — the routing fired on a query it should have ignored, or a
    shared layer leaked the counterfactual into unrelated inputs.

    Samples set ``expected_adapter=None`` so routing accuracy is not scored
    against a specific adapter — the key signal is EM preservation relative to
    the no-adapter baseline (computed via CFR).

    Args:
        triviaqa_path: Path to ``triviaqa_dcontrol.json`` produced by
            ``scripts/build_triviaqa_dcontrol.py``.
        n_samples: Maximum number of records to load.

    Returns:
        List of EvalSample instances with ``split='cf_control'``.
    """
    path = Path(triviaqa_path)
    if not path.exists():
        raise FileNotFoundError(
            f"TriviaQA D_control file not found: {path}. "
            "Run `scripts/build_triviaqa_dcontrol.py` first."
        )
    with open(path) as f:
        tq_payload = json.load(f)

    # The build script writes an object carrying the short-answer instruction
    # alongside the records so verification and eval stay byte-identical; fall
    # back to a bare list for older dumps (no transform applied in that case).
    if isinstance(tq_payload, dict) and "records" in tq_payload:
        tq_records = tq_payload["records"]
        short_instr = tq_payload.get("short_answer_instruction", "")
    else:
        tq_records = tq_payload
        short_instr = ""

    if n_samples < len(tq_records):
        rng = random.Random(random_seed)
        selected_indices = sorted(rng.sample(range(len(tq_records)), n_samples))
        full_pool_size = len(tq_records)
        tq_records = [tq_records[i] for i in selected_indices]
        logger.info(
            f"D_control: uniform-random sample of {n_samples} from "
            f"{full_pool_size} records (seed={random_seed})"
        )

    samples: list[EvalSample] = []
    for rec in tq_records:
        question = rec.get("question")
        # Prefer pre-normalized aliases (used by the D_control pre-filter in
        # build_triviaqa_dcontrol.py::is_correct). Raw all_aliases can contain
        # Unicode curly quotes that survive normalize_answer's ASCII-only
        # punctuation stripping, causing spurious EM mismatches for the frozen
        # base — the very model that was pre-filtered to 100% accuracy.
        aliases = rec.get("normalized_aliases") or rec.get("all_aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        answer = rec.get("normalized_answer") or rec.get("answer")
        gold = [a for a in (aliases + ([answer] if answer else [])) if a]
        gold = list(dict.fromkeys(gold))  # dedupe, preserve order

        if not question or not gold:
            continue
        wrapped_q = f"{short_instr}{question.strip()}" if short_instr else question.strip()
        samples.append(EvalSample(
            question=wrapped_q,
            gold_answers=gold,
            expected_adapter=None,
            split=split_name,
            metadata={
                "question_id": rec.get("question_id"),
                "raw_question": question.strip(),
                "normalized_answer": rec.get("normalized_answer"),
                "source": "triviaqa",
                "short_answer_instruction": short_instr,
            },
        ))

    logger.info(f"Built {len(samples)} D_control samples from {path}")
    return samples


def build_qm_conflict_dataset(
    qm_conflict_path: str,
    n_samples: int,
    qm_adapter_name: str = "patch_qm_current",
    random_seed: int = D_EVAL_SAMPLING_SEED,
) -> list[EvalSample]:
    """Build D_conflict eval samples from ``data/qm_conflict_pairs.json``.

    The adapter is expected to output ``answer_new`` (the current correct fact).
    ``answer_old`` is stored in metadata for downstream FR / backward-interference
    analysis. Semi-synthetic pairs produced by ``scripts/build_qm_conflict_pairs.py``.

    Args:
        qm_conflict_path: Path to ``data/qm_conflict_pairs.json``.
        n_samples: Maximum number of records to load.
        qm_adapter_name: Adapter the router should route to.
        random_seed: RNG seed for reproducible subsampling.

    Returns:
        List of EvalSample instances with ``split='qm_conflict'``.
    """
    path = Path(qm_conflict_path)
    if not path.exists():
        raise FileNotFoundError(
            f"QM conflict pairs file not found: {path}. "
            "Run `scripts/build_qm_conflict_pairs.py` first."
        )
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    if n_samples < len(records):
        rng = random.Random(random_seed)
        selected_indices = sorted(rng.sample(range(len(records)), n_samples))
        full_pool_size = len(records)
        records = [records[i] for i in selected_indices]
        logger.info(
            f"D_conflict (QM): uniform-random sample of {n_samples} from "
            f"{full_pool_size} records (seed={random_seed})"
        )

    samples: list[EvalSample] = []
    for rec in records:
        question = rec.get("question")
        answer_new = rec.get("answer_new")
        if not question or not answer_new:
            continue
        samples.append(EvalSample(
            question=question.strip(),
            gold_answers=[answer_new.strip()],
            expected_adapter=qm_adapter_name,
            split="qm_conflict",
            metadata={
                "id": rec.get("id"),
                "answer_old": rec.get("answer_old"),
                "changed_attribute": rec.get("changed_attribute"),
                "old_value": rec.get("old_value"),
                "new_value": rec.get("new_value"),
                "language": rec.get("language"),
                "intention_category": rec.get("intention_category"),
                "complexity_level": rec.get("complexity_level"),
                "source_file": rec.get("source_file"),
                "source": "qm_conflict",
            },
        ))

    logger.info(
        f"Built {len(samples)} D_conflict (QM) samples from {path} "
        f"(qm_adapter={qm_adapter_name!r})"
    )
    return samples
