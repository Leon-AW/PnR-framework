## Implementation Plan — LLM-as-a-Judge with Gemma-4-26B-A4B (Supplementary Practical Metric)

> **Audience:** another LLM-coding-agent that has access to this repository
> (`/vol/fob-vol1/mi23/wagnerql/PnR-framework`) and can run SLURM jobs on the
> `gruenau10` (3× A100-80GB) compute node. This document is self-contained:
> all rationale, file paths, function signatures, and acceptance criteria are
> spelled out so the work can be completed end-to-end without further questions.

---

## 1. Context and Goal

### Why
The thesis primary metrics are **EM, F1, ESR, FR (forgetting rate)** as required
by the exposé (`docs/short_expose.tex` §4.2). On factoid datasets, strict EM is
often too harsh: a system that answers `"India gained independence on August 15, 1947"`
gets EM=0 against gold `"15 August 1947"` even though the answer is factually
indistinguishable. F1 partially mitigates this; an LLM judge mitigates it further
and, more importantly, captures the **practical-utility dimension** that an
enterprise CL framework actually cares about ("would a human accept this answer?").

We add a Gemma-4-26B-A4B-based judge as a **supplementary** column on every
results table — it does **not** replace EM/F1/ESR/FR. The exposé R1/R2 metric
definitions remain unchanged, ensuring continued literature comparability with
ROME/MEMIT/RECIPE/T-Patcher/MEND.

### Goal — concrete deliverables
1. A reproducible, post-hoc judge pipeline that consumes any `eval_results/<run_name>/results.json` and emits per-record `judge_score` (binary correct/incorrect) plus split-level and overall `judge_accuracy` aggregates.
2. A small human-calibration protocol that produces **Cohen's κ** between Gemma judge and human annotation on 100 stratified records. Without κ ≥ 0.6, the judge metric is reported with an explicit "uncalibrated" caveat in the thesis.
3. SLURM integration so any new D_eval sweep can be judge-scored in ≤ 3 h on `gruenau10` without slowing down primary metrics.
4. Integration into the existing `EvalResult.judge_score` field (already exists at `src/eval/runner.py:169`) and the `summary` block in `report.json`.

### Non-goals (do NOT do these)
- Do **not** change the definition of ESR or FR. They remain target_false-matching and 1−accuracy_on_D_control respectively.
- Do **not** modify the existing `src/eval/judge.py` (Mistral self-judge). Leave it for backwards compatibility; the new code lives in `src/eval/external_judge.py`.
- Do **not** invoke the judge inline during `eval_pnr.py` runs. Judge-scoring is **post-hoc only** — we do not want to slow down primary D_eval sweeps and we want freedom to re-judge with different prompts/models without re-running the expensive system runs.
- Do **not** use the judge to retroactively change the headline numbers in the thesis tables. Judge-Acc is a *new column*, not a replacement.

---

## 2. Existing State (read this first before coding)

### Files that exist and are relevant
| Path | Purpose | Touch? |
|---|---|---|
| `src/eval/judge.py` | Old Mistral self-judge with 1-5 scale. Methodologically weak. | **Leave alone.** |
| `src/eval/runner.py` | Eval orchestration. `EvalResult.judge_score: float \| None` already exists at line ~169. `to_dict()` already serialises it. | Read-only reference. |
| `eval_pnr.py` | Primary eval CLI. Has a `--use_llm_judge` flag (the old judge). | Read-only reference. |
| `src/eval/dataset.py` | Defines `EvalSample` (`question`, `gold_answers: list[str]`, `expected_adapter`, `split`, `metadata`). `D_EVAL_SAMPLING_SEED = 42`. | Read-only reference. |
| `eval_results/<run_name>/results.json` | List of dicts, one per sample. Each has `question`, `gold_answers`, `parsed_answer`, `is_exact_match`, `f1`, `judge_score` (currently `null`), `split`, `metadata`. | **Will be mutated** by the new scorer (writes back to `judge_score`). |
| `eval_results/<run_name>/report.json` | Summary block (`exact_match_overall`, `f1_overall`, `esr`, `dcontrol_forgetting_rate`, …). | **Will be augmented** with `judge_accuracy_overall` and `by_split[<split>].judge_accuracy`. |
| `~/.cache/huggingface` symlink | Already redirected to `/vol/tmp/wagnerql/.cache/huggingface` (140 TB). HF downloads are automatic. | No action needed. |
| `slurm/eval_deval.sh` | Reference for SLURM header (gruenau10 pinning, conda activate, etc.). | Pattern source. |

### Conventions (must respect)
- **Compute node:** all SLURM jobs use `#SBATCH --nodelist=gruenau10` (see `docs/roadmap.md` §"Compute Convention").
- **Conda env:** activate via `source /usr/local/anaconda3-2024.06/etc/profile.d/conda.sh && conda activate pnr`.
- **Storage:** anything > a few MB goes to `/vol/tmp/wagnerql/`, symlinked into the repo if a stable path is needed. The HF cache is already redirected so model downloads are automatic — `google/gemma-4-26B-A4B-it` will land in `/vol/tmp/wagnerql/.cache/huggingface/hub/...` (~14 GB at int4, ~50 GB at fp16 download size).
- **Logging:** SLURM output goes to `logs/<jobname>_<jobid>.out` and `logs/<jobname>_<jobid>.err`.
- **Reproducibility:** all randomness uses `D_EVAL_SAMPLING_SEED = 42` from `src/eval/dataset.py`.

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  Primary D_eval sweep (already running, do NOT touch)            │
│   slurm/submit_deval_sweep.sh  →  9× SLURM jobs                  │
│     → eval_results/<run_name>/results.json + report.json         │
└──────────────────────────┬───────────────────────────────────────┘
                           │   (post-hoc, days later, independent)
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  NEW: Judge-scoring pipeline                                     │
│                                                                  │
│  scripts/score_with_judge.py <run_name> [<run_name> ...]         │
│      ├─ Loads google/gemma-4-26B-A4B-it (int4, bnb)              │
│      ├─ For each record:                                         │
│      │     prompt = build_judge_prompt(q, gold, pred, dataset)   │
│      │     resp   = gemma.generate(prompt, max_new_tokens=8)     │
│      │     score  = parse_judge_response(resp)  # bool or None   │
│      ├─ Mutates results.json in place (judge_score field)        │
│      └─ Recomputes & augments report.json:                       │
│            summary.judge_accuracy_overall                        │
│            summary.judge_disagreement.{em_only,judge_only,both}  │
│            by_split.<split>.judge_accuracy                       │
│                                                                  │
│  src/eval/external_judge.py  (the model wrapper + prompt logic)  │
│                                                                  │
│  slurm/score_with_judge.sh  (SLURM batch wrapper)                │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  Calibration (one-off, manual)                                   │
│                                                                  │
│  scripts/sample_for_human_calibration.py <run_name> [...]        │
│      → eval_results/_calibration/calibration_to_annotate.csv     │
│      (100 records, stratified by split × EM × Judge agreement)   │
│                                                                  │
│  HUMAN: opens CSV, fills `human_label` column (0/1)              │
│                                                                  │
│  scripts/compute_judge_kappa.py                                  │
│      → eval_results/_calibration/calibration_report.json         │
│      ({cohen_kappa, agreement_rate, n, breakdown_by_cell})       │
└──────────────────────────────────────────────────────────────────┘
```

The judge pipeline is **completely decoupled** from the primary eval. It reads
`results.json` files, runs the judge model, writes back. It can be re-run with
a different judge or prompt without rerunning any system inference.

---

## 4. Implementation Tasks (ordered)

### Task 1 — `src/eval/external_judge.py` (new file, ~300 LOC)

Create this file. It encapsulates the judge model and the prompt logic.

#### 1.1 Imports / module docstring

```python
"""
External LLM Judge (Gemma-4-26B-A4B)
=====================================

Supplementary correctness scorer for the PnR evaluation suite. Produces a
binary CORRECT/INCORRECT verdict per (question, gold, prediction) triple
using a model from a *different* family than the systems under evaluation
(Gemma-4 vs Mistral-7B), so the judge is not grading its own homework.

Why binary, not 1-5: Zheng et al., 2024 ("Judging LLM-as-a-Judge with
MT-Bench and Chatbot Arena", NeurIPS 2024) document substantial mid-range
clustering and length bias on Likert-style judge prompts. Binary output
plus an explicit anti-length, anti-style clause in the prompt mitigates
both failure modes.

Reproducibility: open-weights model, deterministic decoding (do_sample=False,
temperature=0). The exact prompt template lives in this file as a module-level
constant — any change requires bumping JUDGE_PROMPT_VERSION and re-running the
calibration protocol.

This module DOES NOT replace EM/F1/ESR/FR. It augments them.
"""
```

#### 1.2 Constants

```python
JUDGE_MODEL_ID: str = "google/gemma-4-26B-A4B-it"
"""Instruction-tuned variant; required for reliable format following."""

JUDGE_PROMPT_VERSION: str = "v1.0"
"""Bump this when the prompt or output format changes. Stored in report.json
so we can detect mixed-version judging across a sweep."""

JUDGE_MAX_NEW_TOKENS: int = 8
"""Just enough for "CORRECT" or "INCORRECT" plus a stop token. Saves latency."""

# Two prompt variants — pick at runtime based on the dataset of the sample.
# Both share the same anti-bias preamble and binary output contract.

JUDGE_PROMPT_FACTOID: str = """\
You are an impartial evaluator of factual question-answering systems. Your job is to decide \
whether a system prediction conveys the SAME FACTUAL CONTENT as a reference answer.

Question: {question}
Reference answer(s) (any one is acceptable): {gold}
System prediction: {prediction}

Rules:
- Different wording, ordering, or formatting are acceptable. Examples:
    "August 15, 1947" matches "15 August 1947".
    "Paris" matches "the city of Paris".
    "8" matches "eight".
- Extra surrounding explanation does NOT invalidate a correct answer, as long \
as the core fact is asserted somewhere in the prediction.
- A prediction that contradicts the reference is INCORRECT, even if it sounds confident.
- A prediction that is irrelevant, evasive, refuses to answer, or is empty is INCORRECT.
- Length, style, fluency, and politeness are irrelevant. Score ONLY on factual \
correctness with respect to the reference.

Respond with EXACTLY one word: CORRECT or INCORRECT. No other text. No punctuation. No explanation."""

JUDGE_PROMPT_COUNTERFACT: str = """\
You are an impartial evaluator of knowledge-editing systems. The system has been edited \
with a counterfactual fact, and you must decide whether its prediction asserts the \
counterfactual content (NOT the original true fact).

Question: {question}
Counterfactual target the system was edited to assert: {gold}
System prediction: {prediction}

Rules:
- The reference is the COUNTERFACTUAL target, not the real-world fact. The system is \
considered CORRECT iff its prediction asserts the counterfactual content.
- Different wording, ordering, or formatting are acceptable.
- A prediction that asserts the *original* true fact is INCORRECT (the edit failed).
- A prediction that is irrelevant, evasive, or empty is INCORRECT.
- Length, style, and politeness are irrelevant.

Respond with EXACTLY one word: CORRECT or INCORRECT. No other text. No punctuation. No explanation."""
```

#### 1.3 `class ExternalJudge`

```python
@dataclass
class JudgeVerdict:
    """Result of judging a single (question, gold, prediction) triple."""
    is_correct: bool | None    # True = CORRECT, False = INCORRECT, None = unparseable
    raw_response: str           # Whatever the judge model emitted (for audit)
    prompt_version: str         # JUDGE_PROMPT_VERSION at time of judging
    judge_model_id: str         # JUDGE_MODEL_ID at time of judging


class ExternalJudge:
    """Stateful wrapper around the Gemma-4 judge model.

    Loads once, scores many. Thread-unsafe (single CUDA context).

    Usage:
        judge = ExternalJudge()
        judge.load()  # downloads model on first call (~14 GB int4, ~5 min)
        verdict = judge.score(
            question="when did India gain independence",
            gold=["15 August 1947"],
            prediction="India gained independence on August 15, 1947.",
            dataset_kind="factoid",  # or "counterfact"
        )
        # verdict.is_correct == True
    """

    def __init__(
        self,
        model_id: str = JUDGE_MODEL_ID,
        quantization: str = "int4",  # "int4" | "int8" | "none"
        device: str = "cuda",
        max_new_tokens: int = JUDGE_MAX_NEW_TOKENS,
    ) -> None:
        self.model_id = model_id
        self.quantization = quantization
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        """Load model + tokenizer. Idempotent."""
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        if self.quantization == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_config = None

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self._model.eval()

    def score(
        self,
        question: str,
        gold: list[str],
        prediction: str,
        dataset_kind: str = "factoid",  # "factoid" | "counterfact"
    ) -> JudgeVerdict:
        """Judge a single triple. Returns a JudgeVerdict."""
        if self._model is None:
            self.load()

        prompt_str = self._build_prompt(question, gold, prediction, dataset_kind)
        chat = [{"role": "user", "content": prompt_str}]
        prompt_ids = self._tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        import torch
        with torch.no_grad():
            output = self._model.generate(
                prompt_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        response_ids = output[0][prompt_ids.shape[1]:]
        raw = self._tokenizer.decode(response_ids, skip_special_tokens=True).strip()
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
        """Pick the right template and fill it. Empty predictions / gold get
        sane defaults so the judge sees a well-formed prompt."""
        gold_str = " | ".join(g for g in gold if g) or "(none)"
        pred_str = (prediction or "").strip() or "(empty prediction)"
        if dataset_kind == "counterfact":
            tpl = JUDGE_PROMPT_COUNTERFACT
        else:
            tpl = JUDGE_PROMPT_FACTOID
        return tpl.format(question=question.strip(), gold=gold_str, prediction=pred_str)

    @staticmethod
    def _parse_response(raw: str) -> bool | None:
        """Robust parser. Returns True/False/None.

        The model is instructed to emit exactly CORRECT or INCORRECT, but
        real models occasionally append punctuation, lowercase, or add a leading
        space. We accept any of these. We reject anything that contains both
        markers (model said both) or neither (off-spec).
        """
        upper = raw.upper()
        # Order matters: 'INCORRECT' contains 'CORRECT' as a substring,
        # so we must test for INCORRECT first.
        has_incorrect = bool(_re.search(r"\bINCORRECT\b", upper))
        has_correct = bool(_re.search(r"\bCORRECT\b", upper)) and not has_incorrect
        if has_incorrect and not has_correct:
            return False
        if has_correct and not has_incorrect:
            return True
        return None  # unparseable
```

(Use `import re as _re` at module top.)

#### 1.4 Self-test (smoke)

Add a small `if __name__ == "__main__":` block that runs ~6 hand-crafted triples against the judge and asserts the expected verdicts. This lets a human verify the judge is sane on a known dataset before unleashing it on 18k records:

| # | question | gold | prediction | expected |
|---|---|---|---|---|
| 1 | "Who painted the Mona Lisa?" | ["Leonardo da Vinci"] | "Leonardo da Vinci" | CORRECT |
| 2 | "Who painted the Mona Lisa?" | ["Leonardo da Vinci"] | "The Mona Lisa was painted by Leonardo da Vinci in the early 1500s." | CORRECT |
| 3 | "Who painted the Mona Lisa?" | ["Leonardo da Vinci"] | "Pablo Picasso" | INCORRECT |
| 4 | "Who painted the Mona Lisa?" | ["Leonardo da Vinci"] | "I don't know." | INCORRECT |
| 5 | "When did India gain independence?" | ["15 August 1947"] | "August 15, 1947" | CORRECT |
| 6 | "When did India gain independence?" | ["15 August 1947"] | "" | INCORRECT |

Run via `python -m src.eval.external_judge`. Failure means the prompt or model config needs adjusting *before* the SLURM job.

---

### Task 2 — `scripts/score_with_judge.py` (new, ~250 LOC)

CLI entry point for post-hoc scoring. Argparse signature:

```text
python scripts/score_with_judge.py <run_name> [<run_name> ...]
    [--results_dir eval_results]
    [--dataset_kind {auto, factoid, counterfact}]
    [--only_disagreement] [--force]
    [--max_records N] [--quantization {int4, int8, none}]
    [--log_level INFO]
```

Behaviour:

- For each `<run_name>` argument:
  1. Read `<results_dir>/<run_name>/results.json` (list of dicts). Abort with clear error if missing.
  2. For each record:
     - If `record["judge_score"]` is non-null and `--force` is not set, skip.
     - If `--only_disagreement` is set, skip records where `is_exact_match=True` (we only score the EM-misses, halves compute).
     - Determine `dataset_kind`:
       - `--dataset_kind auto`: infer from `record["split"]`. Splits starting with `cf_` → `counterfact`. Anything else → `factoid`. Override with explicit flag if needed.
     - For CounterFact `cf_conflict`, `gold_answers` already contains `target_false` (verify by reading `src/eval/dataset.py::build_counterfact_conflict_dataset`; if not, also read `metadata["target_false"]` and prefer that).
     - Call `judge.score(...)`, store `verdict.is_correct` (cast to `bool` or leave `None`) into `record["judge_score"]`. Also persist `record["judge_raw"]`, `record["judge_prompt_version"]`, `record["judge_model_id"]` for audit.
  3. Write back to `results.json` (atomic: write to `results.json.tmp`, fsync, rename).
  4. Augment `report.json`:
     ```python
     summary["judge_accuracy_overall"] = mean of (judge_score == True) over non-null
     summary["judge_unparseable_rate"] = fraction of None / total
     summary["judge_disagreement"] = {
         "em_correct_judge_correct":  count where is_exact_match=True  and judge_score=True,
         "em_correct_judge_wrong":    count where is_exact_match=True  and judge_score=False,
         "em_wrong_judge_correct":    count where is_exact_match=False and judge_score=True,
         "em_wrong_judge_wrong":      count where is_exact_match=False and judge_score=False,
         "em_correct_judge_null":     count where is_exact_match=True  and judge_score is None,
         "em_wrong_judge_null":       count where is_exact_match=False and judge_score is None,
     }
     summary["judge_meta"] = {
         "model_id": JUDGE_MODEL_ID,
         "prompt_version": JUDGE_PROMPT_VERSION,
         "scored_at_utc": iso8601_now(),
         "n_scored": <count of records with non-null judge_score>,
         "n_skipped_em_match": <if --only_disagreement>,
     }
     by_split[<split>]["judge_accuracy"] = mean of (judge_score == True) on records of that split
     ```
  5. Log a one-line summary per split:
     ```
     [run=pnr_deval split=cf_conflict]  EM=0.045  F1=0.123  Judge=0.317  (n=1000, unparseable=2)
     ```

Resume-safety:
- Re-running without `--force` is a no-op for already-scored records.
- Mid-run crashes leave a partial `results.json` — the tmp-then-rename pattern guarantees we never write garbage; the next run picks up where it left off.
- Print progress every 50 records via `tqdm`.

Performance:
- Load model once at script start. Loop over all `<run_name>` arguments using the same loaded model (do NOT reload between runs).
- Average judge call ≈ 0.3-0.6 s on int4 with prompt-cached chat template. 18k records ≈ 90-180 min.

---

### Task 3 — `slurm/score_with_judge.sh` (new, ~50 LOC)

Pattern after `slurm/eval_deval.sh`. Header:

```bash
#!/bin/bash
#SBATCH --job-name=judge_score
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --nodelist=gruenau10
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source /usr/local/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate pnr

export TQDM_MININTERVAL=10
export TQDM_NCOLS=100

echo "Job ID: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}  Started: $(date)"
python scripts/score_with_judge.py "$@"
echo "Finished: $(date)"
```

Caller pattern (do NOT add to `submit_deval_sweep.sh` — keep the judge step manual):

```bash
sbatch slurm/score_with_judge.sh \
    frozen_base_deval monolithic_deval lora_rag_deval pnr_deval \
    xlora_deval recipe_deval morpheus_deval parallel_deval morpheus_nobypass_deval
```

---

### Task 4 — `scripts/sample_for_human_calibration.py` (new, ~120 LOC)

Stratified random sampler for the human-annotation calibration step.

CLI:
```text
python scripts/sample_for_human_calibration.py <run_name> [<run_name> ...]
    [--n_per_cell 25]
    [--output eval_results/_calibration/calibration_to_annotate.csv]
```

Logic:
1. Concatenate `results.json` records from each `<run_name>`, tagging each with its source run.
2. Build four buckets by `(is_exact_match, judge_score)`:
   - cell_A: EM=True,  Judge=True
   - cell_B: EM=True,  Judge=False
   - cell_C: EM=False, Judge=True
   - cell_D: EM=False, Judge=False
3. Skip records where `judge_score` is None (unparseable).
4. From each bucket, draw `--n_per_cell` records uniformly at random with `random.Random(D_EVAL_SAMPLING_SEED).sample(...)` (seed = 42 from `src/eval/dataset.py`).
5. Shuffle the combined 100 records (so the human cannot infer the cell from row order — the human must judge blind).
6. Emit a CSV with columns:
   `id, run_name, split, question, gold_answers, prediction, em_correct, judge_score, human_label, notes`
   Where `human_label` and `notes` are **left empty**. The human fills these in.

Use Python's stdlib `csv` module. Quote everything. Multi-line predictions get `\n` preserved.

The output path's parent dir (`eval_results/_calibration/`) should be created if missing.

---

### Task 5 — `scripts/compute_judge_kappa.py` (new, ~80 LOC)

Reads the human-annotated CSV back and computes Cohen's κ.

CLI:
```text
python scripts/compute_judge_kappa.py
    [--input eval_results/_calibration/calibration_to_annotate.csv]
    [--output eval_results/_calibration/calibration_report.json]
```

Logic:
1. Read the CSV (must have at least one non-empty `human_label`). Skip rows with empty `human_label`.
2. Coerce `human_label` and `judge_score` to {0, 1}. Reject the row with a clear error if a value is neither.
3. Compute:
   - **Cohen's κ** via `sklearn.metrics.cohen_kappa_score(human, judge)` if sklearn is present; otherwise the manual formula:
     ```
     po = agreement_rate
     pe = sum_k P(human=k) * P(judge=k)
     kappa = (po - pe) / (1 - pe)
     ```
   - **Agreement rate** = mean(human == judge)
   - **Confusion matrix** `{TP, FP, FN, TN}` with judge as "predictor" and human as "ground truth".
   - **Per-cell breakdown**: how many of cell_A/B/C/D were right.
4. Write `calibration_report.json`:
   ```json
   {
     "n_annotated": 87,
     "n_total_in_csv": 100,
     "agreement_rate": 0.85,
     "cohen_kappa": 0.69,
     "judge_model_id": "google/gemma-4-26B-A4B-it",
     "prompt_version": "v1.0",
     "confusion": {"tp": 32, "fp": 6, "fn": 7, "tn": 42},
     "by_cell": {
       "em_true_judge_true":   {"n": 23, "human_agrees_with_judge": 22},
       "em_true_judge_false":  {"n": 18, "human_agrees_with_judge": 14},
       "em_false_judge_true":  {"n": 22, "human_agrees_with_judge": 19},
       "em_false_judge_false": {"n": 24, "human_agrees_with_judge": 23}
     }
   }
   ```
5. Print a one-line headline:
   ```
   Cohen's κ = 0.69  (agreement = 0.85, n=87, judge=Gemma-4-26B-A4B-it/v1.0)
   ```
6. **Verdict thresholds (just print, do not error):**
   - κ ≥ 0.81 → "almost perfect agreement (Landis & Koch)"
   - κ ≥ 0.61 → "substantial agreement — judge metric is publication-grade"
   - κ ≥ 0.41 → "moderate — usable as supplementary, with caveat"
   - κ <  0.41 → "fair or worse — DO NOT report judge metric without remediation; revise prompt"

---

### Task 6 — Update `docs/roadmap.md`

Add a new section after §5c (Ablation Studies):

```markdown
#### 5d. Supplementary Practical-Utility Evaluation (LLM-as-Judge)

**Status: Plan in `plan.md`, implementation not started.**

The exposé-mandated metrics (EM, F1, ESR, FR) are kept as primary. We supplement
them with a binary correctness verdict from `google/gemma-4-26B-A4B-it`
(int4, MoE, 25.2 B total / 3.8 B active params, Apr 2026 release) — chosen
because it is from a different model family than the systems under evaluation
and runs on a single A100-80GB.

Pipeline:
1. Run the primary D_eval sweep as usual (`submit_deval_sweep.sh`).
2. Post-hoc: `sbatch slurm/score_with_judge.sh <run_names...>` →
   mutates `eval_results/<run>/results.json`, augments `report.json`
   with `judge_accuracy_overall` and `judge_disagreement`.
3. Calibrate: `python scripts/sample_for_human_calibration.py ...`,
   manually annotate 100 records, `python scripts/compute_judge_kappa.py`.
4. Report alongside EM/F1: tables get a `Judge-Acc` column, footer cites
   Cohen's κ vs human on the calibration set.

Methodological caveats baked into the design:
- Binary output (no Likert mid-range bias)
- Anti-length-bias clause in prompt (Zheng et al. 2024)
- Different model family from systems under test (no self-evaluation bias)
- Open-weights judge (reproducible)
- Calibration via Cohen's κ; thesis only claims judge-metric publication-grade
  if κ ≥ 0.61
```

Also add to the §"Immediate Priorities" list a new low-priority entry:
```markdown
8. **(Optional) Run LLM-as-Judge supplementary evaluation** — after D_eval
   completes, post-hoc score all 9 runs with Gemma-4-26B-A4B per `plan.md`.
   ~3 h on gruenau10. Strengthens practical-utility claim in thesis Discussion.
```

---

## 5. Calibration Protocol (the human step)

This is the only manual step. Budget ~50 minutes.

1. After Task 4 (sampler) runs, open `eval_results/_calibration/calibration_to_annotate.csv` in any editor that handles CSV well (LibreOffice Calc, VSCode with CSV plugin).
2. For each row, read the `question`, `gold_answers`, and `prediction`. Fill `human_label` with `1` if you would accept the prediction as factually correct (in the spirit of the gold), `0` otherwise.
3. Use `notes` to flag ambiguous cases ("technically true but evasive", "partial answer", "answer correct but for wrong reason").
4. Save. Run Task 5 (`compute_judge_kappa.py`).

**Important:** the human must NOT look at the `judge_score` or `em_correct` columns while annotating. Either delete them temporarily or annotate in a fresh sheet. This keeps the κ comparison honest.

---

## 6. Reporting Format (what goes in the thesis)

After successful scoring + calibration, results tables expand from:

| Method | EM | F1 | ESR | FR |
|---|---|---|---|---|

to:

| Method | EM | F1 | **Judge-Acc** | ESR | FR |
|---|---|---|---|---|---|

with a single footnote in the chapter:

> *Judge-Acc is supplementary, computed post-hoc using `google/gemma-4-26B-A4B-it`
> (int4, deterministic decoding, prompt template v1.0). Cohen's κ vs. one human
> annotator on a stratified sample of 100 records is X.XX (agreement rate Y.YY).
> See Appendix Z for the prompt and calibration details.*

If κ < 0.61: drop the Judge-Acc column from the main table and discuss the
metric only qualitatively in a Limitations subsection.

---

## 7. Acceptance Criteria

The implementation is complete when ALL of these are true:

1. `python -m src.eval.external_judge` runs the 6-triple smoke test and reports `6/6 PASS`.
2. `python scripts/score_with_judge.py frozen_base_deval` runs to completion on the smallest existing results file, mutates `results.json` (judge_score field non-null on most records), and adds `judge_accuracy_overall` to `report.json`. Re-running with the same args is a no-op.
3. `sbatch slurm/score_with_judge.sh frozen_base_deval` queues, lands on gruenau10, finishes within 30 min, and produces a non-empty stdout log without Python tracebacks.
4. `python scripts/sample_for_human_calibration.py frozen_base_deval pnr_deval` produces a 100-row CSV with stratified buckets of ~25 each.
5. After manual annotation of ≥ 50 rows, `python scripts/compute_judge_kappa.py` prints a κ value and writes `calibration_report.json` containing `cohen_kappa` and `confusion`.
6. The new section in `docs/roadmap.md` exists and is consistent with the rest of the doc.
7. `ReadLints` reports zero new lint errors in the touched / new files.
8. The existing primary D_eval pipeline is untouched (run `git diff -- src/eval/runner.py eval_pnr.py slurm/eval_deval.sh slurm/submit_deval_sweep.sh` — should be empty).

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Gemma-4-26B-A4B not on HuggingFace yet, or under license that blocks download | Verify with `huggingface_hub.HfApi().model_info("google/gemma-4-26B-A4B-it")`. If gated: try `Qwen3-30B-A3B-Instruct` as drop-in (similar MoE class, similar quality, no gate). Update `JUDGE_MODEL_ID`. |
| MoE model has weird VRAM footprint (all experts loaded) → OOM on A100-80GB | Drop to int8 (~26 GB) or, last resort, int4 + `device_map="balanced"` to spread across all 3 GPUs on gruenau10. |
| Judge unparseable rate > 5% (model breaks output format) | Inspect `judge_raw` for the offending records, refine prompt (add "ONLY one of these two words: CORRECT or INCORRECT") and bump `JUDGE_PROMPT_VERSION` → re-run scoring. |
| κ < 0.41 (judge unreliable) | Inspect cells where human-judge disagreement is concentrated. Common cases: (a) verbose-but-correct on numeric answers, (b) judge accepting near-misses on dates, (c) judge rejecting valid synonyms. Tune prompt rules accordingly. |
| Length bias confirmed empirically (judge systematically prefers longer predictions) | Add a permutation test: re-judge a subsample with `pred ↔ gold` swapped; verdicts should match. If they systematically don't, the prompt's anti-length clause needs strengthening. |

---

## 9. Files to Create or Modify (summary)

| File | Action | Approx. LOC |
|---|---|---|
| `src/eval/external_judge.py` | **NEW** | ~300 |
| `scripts/score_with_judge.py` | **NEW** | ~250 |
| `scripts/sample_for_human_calibration.py` | **NEW** | ~120 |
| `scripts/compute_judge_kappa.py` | **NEW** | ~80 |
| `slurm/score_with_judge.sh` | **NEW** | ~30 |
| `docs/roadmap.md` | **MODIFY** (add §5d, add priority 8) | ~30 |
| `eval_pnr.py`, `src/eval/runner.py`, `slurm/eval_deval.sh`, `slurm/submit_deval_sweep.sh`, `src/eval/judge.py` | **DO NOT TOUCH** | 0 |

Total new code ≈ 800 LOC, all of it isolated from the primary eval pipeline.

---

## 10. Out of Scope (for later, separate plans)

- A *trained* (small specialised) judge model fine-tuned on the human annotations. Adds complexity for marginal gain unless κ < 0.41.
- Multi-judge ensemble (Gemma + Qwen + Claude API). Diminishing returns on a master's thesis.
- Online (streaming) judge embedded in `eval_pnr.py`. Post-hoc is preferred for the reasons in §1 (Non-goals).
- Judging FR / ESR by re-defining them through the judge. The exposé-bound metrics stay as-is; only an *additional* `Judge-Acc` column is added.
