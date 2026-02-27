"""
LLM-as-a-Judge
===============

Optional LLM-based evaluation scoring for the PnR evaluation suite.

Uses the same PatchAndRouteInference pipeline with ``skip_routing=True``
so that judge prompts are never routed to domain-specific adapters.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from src.inference import PatchAndRouteInference
    from .dataset import EvalSample
    from .runner import EvalResult

logger = logging.getLogger(__name__)


class LLMJudge:
    """Scores predictions using the LLM as a judge.

    The judge prompt asks the model to rate an answer 1-5 against a reference.
    ``skip_routing=True`` ensures the base model handles all judge queries
    (no domain adapter contamination).

    Example::

        judge = LLMJudge(pipeline)
        results = judge.score_batch(eval_results)
    """

    JUDGE_PROMPT = (
        "You are an expert evaluator. Score the following answer 1-5.\n"
        "Question: {question}\n"
        "Reference: {gold}\n"
        "Prediction: {prediction}\n"
        "Scoring: 1=wrong, 2=mostly wrong, 3=partial, 4=mostly correct, 5=fully correct.\n"
        "Respond with only a single digit. Score:"
    )

    def __init__(self, pipeline: PatchAndRouteInference) -> None:
        """Initialize the judge.

        Args:
            pipeline: PatchAndRouteInference instance (shared with eval runner).
        """
        self.pipeline = pipeline

    def score(self, sample: EvalSample, parsed_answer: str) -> float | None:
        """Score a single prediction.

        Args:
            sample: The evaluation sample (question + gold answers).
            parsed_answer: The model's parsed answer.

        Returns:
            Score as float (1.0-5.0), or None if parsing fails.
        """
        prompt = self.JUDGE_PROMPT.format(
            question=sample.question,
            gold=sample.gold_answers[0] if sample.gold_answers else "",
            prediction=parsed_answer,
        )

        try:
            result = self.pipeline.generate(prompt, skip_routing=True)
            match = re.search(r"[1-5]", result.response)
            if match:
                return float(match.group())
            logger.debug(f"Judge returned unparseable response: {result.response!r}")
            return None
        except Exception as e:
            logger.warning(f"Judge scoring failed: {e}")
            return None

    def score_batch(self, results: list[EvalResult]) -> list[EvalResult]:
        """Score all results in a batch, mutating judge_score in place.

        Args:
            results: List of EvalResult objects.

        Returns:
            The same list with judge_score fields populated.
        """
        for result in tqdm(results, desc="LLM Judge", unit="sample"):
            result.judge_score = self.score(result.sample, result.parsed_answer)

        scored = sum(1 for r in results if r.judge_score is not None)
        logger.info(f"LLM Judge scored {scored}/{len(results)} samples")
        return results
