"""
Evaluation Dataset
==================

Data representation and dataset builders for the PnR evaluation suite.

Provides:
- EvalSample: Dataclass representing a single evaluation sample
- build_situated_qa_dataset: Build eval samples from SituatedQA streams
- build_local_json_dataset: Build eval samples from local JSON files
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

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
