# Patch-and-Route (PnR)

**A Modular "Patch-and-Route" Framework for Continual Learning in LLMs**

Reference implementation for the master's thesis of the same name (Leon Wagner, Humboldt-Universität zu Berlin). PnR lets a large language model integrate **conflicting, domain-specific knowledge updates without catastrophic forgetting**, at a per-update cost far below full retraining and with negligible inference overhead.

> The full thesis (LaTeX source) lives in [`docs/master-thesis/`](docs/master-thesis/).

---

## Table of Contents

- [The Idea in One Minute](#the-idea-in-one-minute)
- [Headline Results](#headline-results)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Baselines](#baselines)
- [Open-Stream Stress Test & Mitigation](#open-stream-stress-test--mitigation)
- [MORPHEUS (Exploratory Architecture)](#morpheus-exploratory-architecture)
- [Experiment Tracking (MLflow)](#experiment-tracking-mlflow)
- [Tests](#tests)
- [Citation](#citation)
- [License](#license)

---

## The Idea in One Minute

Large language models forget. When you retrain a model on new facts, old and new knowledge share the same weights, so gradient descent overwrites the old globally — **catastrophic forgetting**. Retrieval (RAG) sidesteps this but never *internalises* knowledge, and parameter-editing methods (ROME/MEMIT) degrade the model after repeated edits.

PnR takes a different stance: **inhibition over deletion**. Instead of overwriting entrenched parametric knowledge, it *routes around* it.

- The **foundation model is frozen** — its parameters never change.
- New knowledge lives in small, isolated **LoRA experts** (a "base adapter" for the initial corpus, and "knowledge patches" for each conflicting update).
- A **two-stage router** picks the right expert per query and pulls the expert's own training text back into the context window (**Source-Replay**).

This relocates the stability–plasticity dilemma from the *parameter* level to the *architecture* level. The thesis is explicit that this **reframes** continual learning into a tractable, localised, measurable **open-set recognition problem on the routing gate** — and then measures exactly how well that gate holds up.

---

## Headline Results

Across three structurally different update types (atomic facts, temporal updates, long-form enterprise QA), discrete routing into isolated parametric experts is — among the evaluated systems — **the only family that achieves both non-trivial edit success *and* ~0 % forgetting at the same time**. It sits alone on the joint edit-success / forgetting Pareto frontier; every baseline collapses one of the two axes.

| System | CF ESR | SQA ESR | QM ESR | Forgetting ↓ |
|---|---:|---:|---:|---:|
| Frozen base | 0.0 | 0.0 | 1.2 | **0.6** |
| X-LoRA (soft gating) | 0.0 | 0.4 | 0.0 | 74.8 |
| Monolithic LoRA | 0.0 | 20.1 | 23.4 | 100.0 |
| LoRA + RAG | 7.7 | 29.2 | 21.4 | 99.5 |
| RECIPE | 0.3 | 19.8 | 50.0 | 47.8 |
| Parallel Orchestrator (PnR ensemble) | **33.5** | **86.6** | 57.0 | **0.6** |
| **PnR (default, hard routing)** | 30.4 | 86.4 | **62.4** | **0.6** |

*ESR = Edit-Success Rate (%); Forgetting = `1 − accuracy` on a control set the frozen base answers perfectly by construction. CF = CounterFact, SQA = SituatedQA, QM = AIT Quality-Management corpus.*

Other findings:

- **Efficiency.** PnR inference ≈ **457 ms/query** vs. 429 ms for the frozen base (~7 % overhead) and **27 s/query** for X-LoRA (~60× slower). Peak VRAM stays at single-adapter level (~5.4 GB) because experts load one at a time.
- **Update cost.** One PnR patch ≈ 419 s; a monolithic full-corpus retrain ≈ 2 463 s. Cumulatively over `K` updates PnR scales **O(K)** while monolithic retraining scales **O(K²)** (3.46× the steps after 6 updates).
- The **0.6 % forgetting floor = 6 of 1 000 records**, missed identically by PnR *and* the frozen base — i.e. routing-induced forgetting ≈ 0.

For the honest limitations (open-stream leak, bilingual OOD residual), see [Open-Stream Stress Test & Mitigation](#open-stream-stress-test--mitigation).

---

## Architecture

```
                          ┌─────────────────────────────┐
   query ───────────────► │  Stage 1: Domain Gate        │   MiniLM + MLP, 4-way
                          │  {cf, sqa, qm, ood_trivia}   │   classifier (macro-F1 0.978)
                          └──────────────┬───────────────┘
                                         │
                  ood / general          │  in-domain
            ┌────────────────────────────┤
            ▼                            ▼
   ┌─────────────────┐      ┌──────────────────────────────────────┐
   │  Frozen base    │      │  Stage 2: Dispatcher                  │
   │  (no expert)    │      │  • Time-Aware Centroid Router (hard)   │
   └─────────────────┘      │      cosine vs. centroids, τ≈0.45,     │
                            │      newest-wins tie-break             │
                            │  • OR Parallel Orchestrator (ensemble) │
                            └──────────────┬─────────────────────────┘
                                           │  winning expert + Source-Replay
                                           ▼
                            ┌──────────────────────────────────────┐
                            │  Frozen base + hot-swapped LoRA expert │
                            │  + retrieved training chunks in prompt │
                            └──────────────────────────────────────┘
                                           │
                                           ▼  answer
```

**Components**

- **Frozen foundation** — `mistralai/Mistral-7B-Instruct-v0.3`, 4-bit NF4 quantization (double-quant, BF16 compute) via `bitsandbytes`. Never updated.
- **Expert pool** — QLoRA adapters on all seven projections (`q/k/v/o_proj`, `gate/up/down_proj`), dropout 0.05. Knowledge aligned with base priors trains at `r=16, α=32`; knowledge that *contradicts* the base trains at `r=32, α=64` (higher spectral strength to override priors). Optimised with paged AdamW 8-bit.
- **Stage-1 domain gate** (`src/routing/domain_classifier.py`) — MiniLM-L6-v2 sentence encoder + small MLP head. Out-of-domain queries go straight to the frozen base; the expert pool is never touched.
- **Stage-2 dispatcher** — two interchangeable conflict-resolution strategies sharing the same gate, pool, and base:
  - **Time-Aware Centroid Router** (default, hard routing — `src/routing/centroid_router.py`): cosine similarity of the query embedding against per-adapter centroids, winner-takes-all above a threshold, ties broken in favour of the newer timestamp.
  - **Parallel Orchestrator** (ensemble — `src/routing/parallel_orchestrator.py`): all qualifying experts answer independently, then a Branch-Solve-Merge synthesis pass resolves conflicts by recency.
- **Source-Replay** (`src/routing/source_replay.py`) — always-on retrieval of the winning expert's own training chunks into the prompt. The LoRA shifts the *distribution*; the retrieved text supplies the *exact tokens*.
- **Open-set / Mahalanobis detector** (`src/routing/openset_detector.py`) — optional, switchable veto that sends confident-but-out-of-distribution Stage-1 predictions back to the frozen base (Ledoit-Wolf shrinkage, per-class thresholds at a pre-committed 5 % false-reject budget).

---

## Repository Layout

```
PnR-framework/
├── eval_pnr.py                 # Main evaluation CLI (PnR + all baselines)
├── eval_morpheus_continual.py  # Continual-learning (sequential-domain) eval
├── src/
│   ├── inference/              # PatchAndRouteInference, prompt builder, RAG, embeddings, vector store
│   ├── routing/                # centroid router, domain gate, open-set detector, orchestrator, source-replay
│   ├── training/               # PatchAndRouteTrainer, TrainingConfig, train_adapter()
│   ├── baselines/              # X-LoRA, LoRA+RAG, official RECIPE wrappers
│   ├── morpheus/               # exploratory 6-subsystem cognitive architecture (+165 unit tests)
│   ├── eval/                   # EvalRunner, dataset builders, metrics, LLM-as-judge
│   ├── data/                   # semantic / structure-aware chunkers, local JSON loader
│   └── utils/                  # config IO, logging, MLflow tracker
├── train/                      # training entry-point scripts (base, patches, baselines)
├── scripts/                    # data building, router setup, stress test, plotting, judging
├── slurm/                      # SLURM batch jobs (training, eval, data build, sweeps)
├── tests/morpheus/             # pytest suite for the MORPHEUS subsystems
├── examples/router_demo.py     # standalone routing demo
├── docs/master-thesis/         # the thesis (LaTeX)
├── environment.yml             # conda env "pnr"  (recommended)
├── requirements.txt            # pip-only fallback
└── pyproject.toml              # package "patch-and-route"
```

> Note: the model wrapper `PatchAndRouteLLM` lives in `src/models/core.py` (loaded by the inference, training, and routing code). All higher-level entry points import it for you — you normally interact through `src.inference.PatchAndRouteInference`.

---

## Installation

**Requirements:** Python 3.11, an NVIDIA GPU with CUDA (training/inference use 4-bit quantization; evaluation was run on an A100).

### Conda (recommended)

```bash
conda env create -f environment.yml      # creates env "pnr"
conda activate pnr
# update later with:  conda env update -f environment.yml --prune
```

or use the helper, which checks for conda and creates/updates the env:

```bash
./setup_env.sh
conda activate pnr
```

### pip

```bash
pip install -r requirements.txt          # includes X-LoRA from git
# or, for the package + dev tools:
pip install -e ".[dev]"
```

### Verify the GPU stack

```bash
python scripts/validate_gpu_setup.py
```

Key dependencies: `torch>=2.7`, `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `sentence-transformers`, `faiss-cpu`, `chromadb`, `mlflow`.

---

## Quick Start

### Routing demo (no GPU needed)

```bash
python examples/router_demo.py
```

Walks through the Time-Aware Centroid Router with mock embeddings: registering adapters with centroids, routing queries, detecting conflicts, and running Source-Replay.

### Inference in Python

```python
from src.inference import PatchAndRouteInference, GenerationConfig

pnr = PatchAndRouteInference(
    model_id="mistralai/Mistral-7B-Instruct-v0.3",
    checkpoints_dir="checkpoints",        # discovers trained experts + centroids
    quantization="int4",
)

result = pnr.generate("Who is the current Prime Minister of the United Kingdom?")
print(result.text)          # answer
print(result.adapter_used)  # which expert routing selected
```

A ready-made factory is also available:

```python
from src.inference.pnr import create_inference_pipeline
```

---

## Data Preparation

The framework is evaluated on four datasets, each probing a different kind of update. Build scripts live in `scripts/`.

| Dataset | What it probes | Build scripts |
|---|---|---|
| **SituatedQA (SQA)** | temporal updates (pre-2019 = stable base, post-2019 = update stream) | `build_sqa_deval.py` |
| **CounterFact (CF)** | atomic factoid edits, split into 6 thematic knowledge patches | `build_counterfact_data.py`, `build_counterfact_relation_clusters.py` |
| **AIT QM corpus** | long-form bilingual (DE/EN) enterprise document QA; 500 verified conflict pairs | `build_qm_train_data.py`, `build_qm_conflict_pairs.py`, `build_qm_stable_facts.py`, `build_qm_deval.py` |
| **D_control (TriviaQA)** | stability probe — 1 000 items the frozen base answers correctly by construction | `build_triviaqa_dcontrol.py` |

After building datasets and training experts, compute routing state:

```bash
python scripts/compute_centroids.py      # per-adapter centroids
python scripts/build_router_state.py     # serialised router state
python scripts/probe_router_routing.py   # sanity-check routing decisions
```

---

## Training

All training uses streaming datasets, `max_steps` (not epochs), QLoRA on the frozen base, and an effective batch size of 16 (`per_device=1 × grad_accum=16`). Checkpoints land in `checkpoints/{adapter_name}/`.

### PnR experts

```bash
# 1) Base expert on SituatedQA "stable facts" (pre-cutoff temporal + US geo)
python train/train_base_adapter.py --output_dir checkpoints/base_v1 --max_steps 1000

# 2) A single knowledge patch (temporal or geographic)
python train/train_patch.py --type temporal --cutoff_year 2019
python train/train_patch.py --type geo --country India

# 3) Or train the whole expert matrix automatically
python train/train_all_patches.py --max_geo_patches 10

# CounterFact patches (atomic edits) and QM patch (long-form, current answer)
python train/train_counterfact_patch.py --data_path data/counterfact_pairs.json
python train/train_qm_patch.py --data_path data/qm_train.jsonl --answer_field answer_new
```

Programmatic equivalent:

```python
from src.training.trainer import train_adapter, TrainingConfig
train_adapter(adapter_name="patch_geo_india", dataset=..., config=TrainingConfig(max_steps=1000))
```

### Baseline training

```bash
python train/train_monolithic_baseline.py --situatedqa --max_steps 2000   # single LoRA, no routing
python train/train_qm_monolithic.py                                       # sequential = catastrophic forgetting demo
python train/train_xlora_baseline.py --checkpoints_dir checkpoints        # trains the soft-gating head only
python train/train_rag_baseline.py --data_path ... --docs_path ...        # LoRA tuned for RAG context
```

---

## Evaluation

`eval_pnr.py` is the single entry point. It loads the frozen base + experts + router, evaluates each requested split, and computes **EM, F1, routing accuracy, ESR, and stability**, with optional LLM-as-judge and length-normalised log-prob scoring. Results are logged to MLflow and written as JSON to `--output_dir`.

```bash
# Default = PnR routing
python eval_pnr.py --eval_sets base temporal geo_india --n_samples 200 \
    --experiment_name pnr-evaluation --run_name pnr_v1

# Baseline: frozen base (stability "Pass 1")
python eval_pnr.py --no_adapter --eval_sets base temporal --n_samples 100 --run_name frozen_base

# Baseline: monolithic LoRA (bypasses routing)
python eval_pnr.py --monolithic checkpoints/monolithic_v1 --eval_sets base --n_samples 100 --run_name monolithic
```

### Splits (`--eval_sets`)

`base`, `temporal`, `geo_<country>`, `local`, `cf_conflict`, `cf_control`, `sqa_train`, `qm_conflict`, `qm_stable`, `qm_control`. Some splits require their data path (e.g. `cf_control` needs `--triviaqa_dcontrol_path`; `qm_conflict` needs `--qm_conflict_path`).

### System / baseline selectors

| Flag | System |
|---|---|
| *(none)* | **PnR** routing (Time-Aware Centroid Router) — default |
| `--parallel` | PnR **Parallel Orchestrator** (ensemble; see `--parallel_max_adapters`, `--parallel_planner`, `--warm_context`) |
| `--no_adapter` | frozen base model |
| `--monolithic <path>` | single LoRA adapter, routing bypassed |
| `--xlora <ckpt>` | X-LoRA soft gating |
| `--recipe_official <ckpt>` | official RECIPE (EMNLP 2024); add `--recipe_official_edits` |
| `--lora_rag <adapter>` | LoRA + RAG hybrid (`--lora_rag_index`) |
| `--morpheus` | MORPHEUS multi-system architecture (see below) |

Other useful flags: `--n_samples`, `--model_id`, `--checkpoints_dir`, `--quantization {int4,int8,none}`, `--similarity_threshold`, `--domain_classifier_path`, `--use_llm_judge`, `--compute_logprob`.

### Metrics (as used in the thesis)

- **ESR** (Edit-Success Rate) — greedy-decoding edit success; exact match for CF/SQA, strict containment for long-form QM (new value present *and* old value absent).
- **TF-ESR** (Teacher-Forcing ESR) — `P(new | q) > P(old | q)` under teacher forcing; the standard ROME/MEMIT/RECIPE efficacy measure (`--compute_logprob`).
- **Forgetting Rate** — `1 − accuracy(D_control)`; the control set is filtered so the frozen base scores 100 % by construction, so any drop is routing interference.
- **Judge** — binary LLM-as-judge verdict using a different model family (Gemma) to avoid self-grading (`--use_llm_judge`; post-hoc via `scripts/score_with_judge.py`).
- **Efficiency** — per-query latency and peak VRAM; per-update training cost (`scripts/benchmark_update_cost.py`).

### Reproducing the figures

```bash
python scripts/plot_pareto.py                 # ESR vs. forgetting Pareto frontier
python scripts/plot_update_cost_scaling.py    # O(K) vs O(K²) update cost
python scripts/summarize_results.py           # aggregate results.json files
```

---

## Baselines

| Baseline | What it is | Code |
|---|---|---|
| **Frozen base** | unadapted model; edit-success lower bound, stability reference | `--no_adapter` |
| **Monolithic LoRA** | one LoRA retrained on the whole accumulated corpus | `train/train_monolithic_baseline.py` |
| **LoRA + RAG** | monolithic fine-tune plus retrieval over new documents | `src/baselines/lora_rag.py` |
| **X-LoRA** | mixture of LoRA experts with continuous token/layer-level soft gating (Buehler & Buehler) | `src/baselines/xlora.py` |
| **RECIPE** | retrieval-augmented lifelong editing via learned continuous prompts (Chen et al., EMNLP 2024) | `src/baselines/recipe_official.py` |
| **Vanilla RAG** | standalone document-QA RAG, independent of the routing framework | `src/inference/vanilla_rag.py` |

---

## Open-Stream Stress Test & Mitigation

PnR's stability holds *by construction* — the cost is moved onto the gate. To measure that honestly, the open-stream stress test sends 1 000 held-out queries from 5 unseen domains (PubMedQA, LegalBench, financial-QA-10K, SciQ, Natural Questions) at the Stage-1 gate.

```bash
python scripts/build_openstream_testset.py
python scripts/run_openstream_stress.py
```

Finding: the gate leaks ~31 % of unseen queries into experts, almost entirely because a 4-way softmax has no "none-of-the-above" class — the failure is localised to one replaceable component, not the routing principle.

The open-set Mahalanobis detector mitigates this:

```bash
python scripts/build_openstream_testset_fresh.py   # disjoint fit/cal/test
python scripts/fit_openset_detector.py             # fit + calibrate at α=5%
python scripts/run_openstream_mitigation.py
python scripts/sweep_openset_alpha.py              # threshold sweep
```

It cuts the English OOD leak from 17.2 % to 5.8 % at a 2.7 % recall cost. The German residual is structural and honestly reported: the bilingual `qm` class makes German OOD queries hard to distinguish.

---

## MORPHEUS (Exploratory Architecture)

`src/morpheus/` contains an exploratory, brain-inspired six-subsystem extension (stable core / expert bank / fast buffer / consolidation / knowledge store / meta-controller) with a prototype router and a sleep-style consolidation cycle. It is the most experimental part of the codebase and the only part with a dedicated unit-test suite.

```bash
# static inference (same metrics as PnR)
python eval_pnr.py --morpheus --eval_sets base temporal --n_samples 100

# continual-learning eval: sequential domains, forgetting curve, expert lifecycle
python eval_morpheus_continual.py --domains cf sqa qm --architecture morpheus
python eval_morpheus_continual.py --routing_only --domains cf sqa qm   # no LLM needed
```

```python
from src.morpheus import MorpheusInference, MorpheusConfig
```

---

## Experiment Tracking (MLflow)

Training and evaluation log to a local MLflow store (no server required). Every training run is wrapped in `PnRTracker`; step-level metrics come from `MLflowStepCallback`. If MLflow is not installed, tracking degrades to a no-op.

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db    # → http://localhost:5000
```

All training/eval scripts accept `--experiment_name` and `--run_name`.

---

## Tests

The MORPHEUS subsystems are covered by ~165 unit tests:

```bash
pip install -e ".[dev]"
pytest tests/morpheus/                # all subsystem tests
pytest tests/morpheus/test_router.py  # one subsystem
```

---

## Citation

```bibtex
@mastersthesis{wagner2026pnr,
  title  = {A Modular ``Patch-and-Route'' Framework for Continual Learning in LLMs},
  author = {Wagner, Leon},
  school = {Humboldt-Universit\"at zu Berlin},
  year   = {2026}
}
```

---

## License

MIT — see `pyproject.toml`.
</content>
