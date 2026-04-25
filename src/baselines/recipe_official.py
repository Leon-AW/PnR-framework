"""Inference wrapper around the official RECIPE implementation.

Loads the author's code from ``external/RECIPE`` (cloned from
https://github.com/qizhou000/RECIPE) and exposes a ``generate(query)`` API
compatible with ``src/eval/runner.py``.

Checkpoints produced by ``external/RECIPE/train_recipe.py`` are expected at
something like::

    external/RECIPE/train_records/recipe/mistral-7b/<run>/checkpoints/<ckpt>

Pass the absolute path to that ``<ckpt>`` file as ``checkpoint_path``.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL_RECIPE = _REPO_ROOT / "external" / "RECIPE"


def _ensure_official_on_path() -> None:
    p = str(_OFFICIAL_RECIPE)
    if p not in sys.path:
        sys.path.insert(0, p)


@dataclass
class RECIPEOfficialResult:
    response: str
    adapter_loaded: str = "recipe_official"
    routing_result: Any = None
    prompt_retrieved: bool = False
    n_edits_in_repo: int = 0


class RECIPEOfficialInference:
    """Wraps the official RECIPE editor for evaluation."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_name: str = "mistral-7b",
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
        quantization: str = "int4",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        do_sample: bool = False,
        use_gpu: bool = True,
    ) -> None:
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.model_name = model_name
        self.model_id = model_id
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.use_gpu = use_gpu

        self._editor = None
        self._tokenizer = None
        self._device: str | None = None

    def _ensure_loaded(self) -> None:
        if self._editor is not None:
            return

        _ensure_official_on_path()

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # The official config_path is relative; temporarily cwd into the repo
        # so ``configs/recipe/mistral-7b.yaml`` resolves correctly.
        import os
        prev_cwd = os.getcwd()
        os.chdir(_OFFICIAL_RECIPE)
        try:
            from utils.utils import (  # type: ignore
                get_model_editor_config_path,
                model_path_map,
                set_tokenizer_pad_id,
            )
            from editors.recipe.recipe import RECIPE, RECIPEConfig  # type: ignore

            self._device = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
            model_path, config_path = get_model_editor_config_path(self.model_name, "recipe")
            # Prefer the explicit HF hub id from our edit to model_path_map
            model_path = model_path_map.get("mistral-7b-instruct", self.model_id)

            logger.info("Loading base LLM %s (quantization=%s) ...", model_path, self.quantization)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            set_tokenizer_pad_id(tokenizer)

            load_kwargs: dict = {"device_map": self._device}
            if self.quantization == "int4":
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif self.quantization == "int8":
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                load_kwargs["dtype"] = torch.bfloat16

            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

            config = RECIPEConfig.from_yaml(config_path)
            roberta_path = model_path_map["roberta-base"]
            self._editor = RECIPE(model, tokenizer, config, self._device, roberta_path)
            self._editor.load_ckpt(self.checkpoint_path, restrict=True, load_opt=False)
            self._editor.set_train(False)
            self._tokenizer = tokenizer
        finally:
            os.chdir(prev_cwd)

    def apply_edits(self, edit_list: list) -> None:
        self._ensure_loaded()
        requests = []
        for item in edit_list:
            if isinstance(item, dict):
                q = item.get("question") or item.get("prompt") or item.get("src") or ""
                a = item.get("answer") or item.get("target_new") or item.get("alt") or ""
            else:
                q, a = item[0], item[1]
            requests.append({"prompt": q, "subject": q, "target_new": a})
        self._editor.edit_batch(requests)
        logger.info(
            "RECIPE-official: repository now contains %d edits.",
            len(self._editor.knowledge_base_nl) - 1,  # minus the prototype row
        )

    def reset_edits(self) -> None:
        self._ensure_loaded()
        self._editor.restore_to_original_model()

    def generate(self, query: str) -> RECIPEOfficialResult:
        self._ensure_loaded()
        assert self._editor is not None
        assert self._tokenizer is not None

        # RECIPE was trained on completion-style prompts: `pt2xym(prompt, target)`
        # tokenizes `prompt + " " + target` and trains the continuous prompt to
        # steer next-token prediction at the end of `prompt`. Wrapping `query`
        # in Mistral-Instruct's chat template pushes `[/INST]` between RECIPE's
        # injected prompt and the generation point, diluting the edit's effect
        # and letting the instruct prior dominate with verbose essay answers.
        # Feed the raw query — matches recipe.py::edit_batch and data.py::pt2xym.
        tok = self._tokenizer(query, return_tensors="pt", truncation=True, max_length=4096)
        model = self._editor.model
        llm_device = next(model.parameters()).device
        input_ids = tok["input_ids"].to(llm_device)
        attention_mask = tok["attention_mask"].to(llm_device)

        # Cap at 30 tokens: RECIPE edits are factoid completions, not essays.
        max_new = min(self.max_new_tokens, 30)

        # use_cache=False is REQUIRED. RECIPE's begin_layer hook prepends
        # prompt_token_n continuous prompts to the input embeddings on the
        # first forward; the wrapped model.forward extends cache_position +
        # position_ids by the same amount so position 0..N+extra-1 land in
        # the cache correctly. But on every subsequent generation step,
        # `is_first_pass=False` short-circuits all the position/cache fixes
        # (recipe.py:133-138), while transformers.generate still writes the
        # next token to cache_position=N — overwriting a cached prompt token
        # and reading from the wrong slot. Result: TF accuracy 0.88 vs
        # generation EM 0.001. Disabling the cache forces every step through
        # the first-pass branch, which is correct.
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                use_cache=False,
            )

        prompt_len = input_ids.shape[1]
        response = self._tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

        # Completion semantics: first line / first sentence is the answer.
        for sep in ("\n", "."):
            if sep in response:
                response = response[: response.index(sep)].strip()
                break

        # Introspect whether the editor injected a prompt for this query
        adopted = getattr(self._editor, "adopted_prompts", [])
        prompt_retrieved = bool(adopted) and any(len(p) > 0 for p in adopted)

        return RECIPEOfficialResult(
            response=response,
            prompt_retrieved=prompt_retrieved,
            n_edits_in_repo=max(0, len(self._editor.knowledge_base_nl) - 1),
        )
