"""
External LLM Judge (Gemma-4-26B-A4B)
=====================================

Supplementary correctness scorer for the PnR evaluation suite. Produces a
binary CORRECT/INCORRECT verdict per (question, gold, prediction) triple using
a model from a different family than the systems under evaluation (Gemma-4 vs
Mistral-7B), so the judge is not grading its own homework.

Why binary, not 1-5: Zheng et al., 2024 ("Judging LLM-as-a-Judge with MT-Bench
and Chatbot Arena", NeurIPS 2024) document substantial mid-range clustering and
length bias on Likert-style judge prompts. Binary output plus an explicit
anti-length, anti-style clause in the prompt mitigates both failure modes.

Reproducibility: open-weights model, deterministic decoding (do_sample=False,
temperature=0). The exact prompt template lives in this file as a module-level
constant. Any change requires bumping JUDGE_PROMPT_VERSION and re-running the
calibration protocol.

This module DOES NOT replace EM/F1/ESR/FR. It augments them.
"""

from __future__ import annotations

import logging
import re as _re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


JUDGE_MODEL_ID: str = "google/gemma-4-26B-A4B-it"
"""Instruction-tuned variant; required for reliable format following."""

JUDGE_PROMPT_VERSION: str = "v1.0"
"""Bump this when the prompt or output format changes."""

JUDGE_MAX_NEW_TOKENS: int = 8
"""Enough for CORRECT or INCORRECT plus a stop token."""


JUDGE_PROMPT_FACTOID: str = """\
You are an impartial evaluator of factual question-answering systems. Your job is to decide whether a system prediction conveys the SAME FACTUAL CONTENT as a reference answer.

Question: {question}
Reference answer(s) (any one is acceptable): {gold}
System prediction: {prediction}

Rules:
- Different wording, ordering, or formatting are acceptable. Examples:
  "August 15, 1947" matches "15 August 1947".
  "Paris" matches "the city of Paris".
  "8" matches "eight".
- Extra surrounding explanation does NOT invalidate a correct answer, as long as the core fact is asserted somewhere in the prediction.
- A prediction that contradicts the reference is INCORRECT, even if it sounds confident.
- A prediction that is irrelevant, evasive, refuses to answer, or is empty is INCORRECT.
- Length, style, fluency, and politeness are irrelevant. Score ONLY on factual correctness with respect to the reference.

Respond with EXACTLY one word: CORRECT or INCORRECT. No other text. No punctuation. No explanation."""


JUDGE_PROMPT_COUNTERFACT: str = """\
You are an impartial evaluator of knowledge-editing systems. The system has been edited with a counterfactual fact, and you must decide whether its prediction asserts the counterfactual content (NOT the original true fact).

Question: {question}
Counterfactual target the system was edited to assert: {gold}
System prediction: {prediction}

Rules:
- The reference is the COUNTERFACTUAL target, not the real-world fact. The system is considered CORRECT iff its prediction asserts the counterfactual content.
- Different wording, ordering, or formatting are acceptable.
- A prediction that asserts the original true fact is INCORRECT (the edit failed).
- A prediction that is irrelevant, evasive, or empty is INCORRECT.
- Length, style, and politeness are irrelevant.

Respond with EXACTLY one word: CORRECT or INCORRECT. No other text. No punctuation. No explanation."""


@dataclass(frozen=True)
class JudgeVerdict:
    """Result of judging a single (question, gold, prediction) triple."""

    is_correct: bool | None
    raw_response: str
    prompt_version: str
    judge_model_id: str


class ExternalJudge:
    """Stateful wrapper around the Gemma-4 judge model.

    Loads once, scores many. Thread-unsafe because it owns one model instance and
    one CUDA context.
    """

    def __init__(
        self,
        model_id: str = JUDGE_MODEL_ID,
        quantization: str = "int4",
        device: str = "cuda",
        max_new_tokens: int = JUDGE_MAX_NEW_TOKENS,
    ) -> None:
        if quantization not in {"int4", "int8", "none"}:
            raise ValueError("quantization must be one of: int4, int8, none")
        self.model_id = model_id
        self.quantization = quantization
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        """Load model and tokenizer. Idempotent."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            if self.quantization in {"int4", "int8"}:
                raise RuntimeError(
                    "bitsandbytes quantization requested, but transformers "
                    "BitsAndBytesConfig is unavailable"
                ) from exc
            BitsAndBytesConfig = None

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            self.device = "cpu"

        quantization_config = None
        if self.quantization == "int4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "int8":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        logger.info("Loading judge tokenizer: %s", self.model_id)
        # Gemma-4 ships `extra_special_tokens` as a list (e.g. ["<|video|>"]),
        # but transformers 4.57.x expects a dict and crashes in
        # _set_model_specific_special_tokens with `'list' object has no
        # attribute 'keys'`. Passing {} bypasses the parent-class init path;
        # the dropped token is multimodal and unused by a text-only judge.
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            extra_special_tokens={},
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info(
            "Loading judge model: %s (quantization=%s)",
            self.model_id,
            self.quantization,
        )
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs,
        )
        if self.device == "cpu":
            self._model.to("cpu")
        self._model.eval()

    def score(
        self,
        question: str,
        gold: list[str],
        prediction: str,
        dataset_kind: str = "factoid",
    ) -> JudgeVerdict:
        """Judge a single triple."""
        if dataset_kind not in {"factoid", "counterfact"}:
            raise ValueError("dataset_kind must be 'factoid' or 'counterfact'")
        if self._model is None or self._tokenizer is None:
            self.load()

        prompt_str = self._build_prompt(question, gold, prediction, dataset_kind)
        chat = [{"role": "user", "content": prompt_str}]

        encoded = self._tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # transformers ≥5.x returns a BatchEncoding; ≤4.x returns a bare tensor.
        if hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids
        else:
            input_ids = encoded

        target_device = self._model.device if self.device == "cuda" else "cpu"
        input_ids = input_ids.to(target_device)

        import torch

        with torch.no_grad():
            output = self._model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        response_ids = output[0][input_ids.shape[1] :]
        raw = self._tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        ).strip()
        is_correct = self._parse_response(raw)

        return JudgeVerdict(
            is_correct=is_correct,
            raw_response=raw,
            prompt_version=JUDGE_PROMPT_VERSION,
            judge_model_id=self.model_id,
        )

    @staticmethod
    def _build_prompt(
        question: str,
        gold: list[str],
        prediction: str,
        dataset_kind: str,
    ) -> str:
        """Pick the right template and fill it."""
        gold_str = " | ".join(str(g) for g in gold if g) or "(none)"
        pred_str = (prediction or "").strip() or "(empty prediction)"
        question_str = (question or "").strip() or "(empty question)"
        template = (
            JUDGE_PROMPT_COUNTERFACT
            if dataset_kind == "counterfact"
            else JUDGE_PROMPT_FACTOID
        )
        return template.format(
            question=question_str,
            gold=gold_str,
            prediction=pred_str,
        )

    @staticmethod
    def _parse_response(raw: str) -> bool | None:
        """Robust parser returning True, False, or None for off-spec output."""
        upper = (raw or "").upper().strip()
        has_correct = bool(_re.search(r"\bCORRECT\b", upper))
        has_incorrect = bool(_re.search(r"\bINCORRECT\b", upper))
        if has_correct and not has_incorrect:
            return True
        if has_incorrect and not has_correct:
            return False
        return None


def _run_smoke() -> int:
    """Run six hand-crafted smoke tests against the live judge model."""
    cases = [
        (
            "Who painted the Mona Lisa?",
            ["Leonardo da Vinci"],
            "Leonardo da Vinci",
            True,
        ),
        (
            "Who painted the Mona Lisa?",
            ["Leonardo da Vinci"],
            "The Mona Lisa was painted by Leonardo da Vinci in the early 1500s.",
            True,
        ),
        (
            "Who painted the Mona Lisa?",
            ["Leonardo da Vinci"],
            "Pablo Picasso",
            False,
        ),
        (
            "Who painted the Mona Lisa?",
            ["Leonardo da Vinci"],
            "I don't know.",
            False,
        ),
        (
            "When did India gain independence?",
            ["15 August 1947"],
            "August 15, 1947",
            True,
        ),
        (
            "When did India gain independence?",
            ["15 August 1947"],
            "",
            False,
        ),
    ]

    judge = ExternalJudge()
    passed = 0
    for i, (question, gold, prediction, expected) in enumerate(cases, start=1):
        verdict = judge.score(question, gold, prediction)
        ok = verdict.is_correct is expected
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] {i}/6 expected={expected} "
            f"got={verdict.is_correct} raw={verdict.raw_response!r}"
        )
    print(f"{passed}/6 PASS")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(_run_smoke())
