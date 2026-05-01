"""
Evaluation Suite
================

End-to-end evaluation for the Patch-and-Route framework.

Measures:
- Answer quality: exact match, token F1
- Routing correctness: routing accuracy, ESR
- Forgetting: stability score, CFR
- Efficiency: latency, VRAM
- Quality (optional): LLM-as-a-judge scoring
"""

from .runner import EvalRunner, EvalConfig, EvalResult
from .dataset import EvalSample, build_situated_qa_dataset, build_local_json_dataset
from .metrics import (
    DEFAULT_SHORT_ANSWER_BOUNDARIES,
    normalize_answer,
    parse_model_output,
    exact_match,
    token_f1,
    compute_esr,
    compute_logprob_em,
    compute_logprob_esr,
    compute_routing_accuracy,
    compute_stability_score,
    compute_cfr,
    compute_efficiency,
)
from .judge import LLMJudge

__all__ = [
    "EvalRunner",
    "EvalConfig",
    "EvalResult",
    "EvalSample",
    "build_situated_qa_dataset",
    "build_local_json_dataset",
    "LLMJudge",
    "DEFAULT_SHORT_ANSWER_BOUNDARIES",
    "normalize_answer",
    "parse_model_output",
    "exact_match",
    "token_f1",
    "compute_esr",
    "compute_logprob_em",
    "compute_logprob_esr",
    "compute_routing_accuracy",
    "compute_stability_score",
    "compute_cfr",
    "compute_efficiency",
]
