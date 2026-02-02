# Patch-and-Route Framework

> A Modular Framework for Continual Learning in Enterprise LLMs

[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/Leon-AW/PnR-framework)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This framework implements the **Patch-and-Route** architecture for continual learning in Large Language Models, enabling domain-specific knowledge integration without catastrophic forgetting.

## Core Concepts

| Term | Description |
|------|-------------|
| **Frozen Foundation** | Base LLM with frozen parameters (e.g., Mistral-7B). Never modified. |
| **Expert Pool** | Collection of domain-specific LoRA adapters (Base Adapters + Knowledge Patches) |
| **Base Adapter** | Large adapter trained on initial domain corpus (e.g., `QM_Base_Adapter_v1`) |
| **Knowledge Patch** | Small adapter for specific updates (e.g., `CEO_Patch_v2` storing "The CEO is Müller") |
| **Two-Level Router** | Hybrid routing: Manual domain selection + Intelligent Dispatcher (Micro-Router) |

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Patch-and-Route Framework                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    Two-Level Routing System                         │ │
│  │  ┌──────────────────────┐    ┌────────────────────────────────────┐ │ │
│  │  │ Level 1: Domain      │    │ Level 2: Intelligent Dispatcher    │ │ │
│  │  │ Selection (Manual)   │───▶│ (Micro-Router)                     │ │ │
│  │  │ [QM] [IT] [Legal]    │    │ • Centroid Router (embedding-based)│ │ │
│  │  └──────────────────────┘    │ • Parallel-Orchestrator (ensemble) │ │ │
│  │                              └────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────┐    ┌─────────────────────────────────────────────┐ │
│  │ Frozen Foundation│    │              Expert Pool                    │ │
│  │  (Mistral-7B)   │◄───│  ┌──────────┐ ┌──────────┐ ┌──────────┐    │ │
│  │   4-bit quant   │    │  │QM_Base   │ │CEO_Patch │ │ISO_Patch │    │ │
│  │                 │    │  │Adapter_v1│ │   _v2    │ │   _v3    │    │ │
│  └─────────────────┘    │  └──────────┘ └──────────┘ └──────────┘    │ │
│                         └─────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Environment Setup

**Prerequisites:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and NVIDIA GPU with CUDA

```bash
# Clone repository
git clone git@github.com:Leon-AW/PnR-framework.git
cd PnR-framework

# Create conda environment
conda env create -f environment.yml
conda activate pnr

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Train Base Expert Adapter

```bash
python train_base_adapter.py \
    --model_id "mistralai/Mistral-7B-Instruct-v0.3" \
    --quantization int4 \
    --max_steps 2000 \
    --batch_size 4 \
    --gradient_accumulation 4 \
    --output_dir checkpoints/situatedqa_base_v1
```

### 3. Use Trained Adapter

```python
from src.models.core import PatchAndRouteLLM, ExpertConfig

# Load model with trained adapter
llm = PatchAndRouteLLM(model_id="mistralai/Mistral-7B-Instruct-v0.3")
llm.load_frozen_foundation()
llm.load_expert("checkpoints/situatedqa_base_v1")

model, tokenizer = llm.get_training_components()
```

## Project Structure

```
PnR-framework/
├── src/
│   ├── __init__.py         # Package init (v0.1.0)
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py       # SituatedQA & CounterFact streaming loaders
│   ├── models/
│   │   ├── __init__.py
│   │   └── core.py         # PatchAndRouteLLM model manager
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py      # SFTTrainer for streaming datasets
│   └── utils/
│       ├── __init__.py
│       ├── config.py       # Configuration serialization (JSON)
│       └── logging.py      # Centralized logging setup
├── train_base_adapter.py   # Main training entry point
├── environment.yml         # Conda environment (Python 3.11)
├── requirements.txt        # Pip dependencies (fallback)
├── pyproject.toml          # Project metadata and dependencies
└── setup_env.sh            # Environment setup script
```

## Key Features

### Streaming Data Loading
Handles large datasets without disk storage using HuggingFace `datasets` streaming:

```python
from src.data.loader import SituatedQALoader, SituatedQAConfig

config = SituatedQAConfig(
    streaming=True,
    temporal_cutoff_year=2019,
    buffer_size=10_000,
)

loader = SituatedQALoader(config)
stream_stable, stream_update = loader.get_temporal_streams()
```

### Memory-Efficient Training
4-bit quantization + LoRA for training on consumer GPUs:

```python
from src.models.core import PatchAndRouteLLM, FrozenFoundationConfig, QuantizationType

config = FrozenFoundationConfig(
    model_id="mistralai/Mistral-7B-Instruct-v0.3",
    quantization=QuantizationType.INT4,  # ~4GB VRAM
)

llm = PatchAndRouteLLM(foundation_config=config)
llm.load_frozen_foundation()
llm.attach_expert(name="my_expert", r=16, lora_alpha=32)
```

### Temporal Data Filtering
SituatedQA split for continual learning experiments:

| Stream | Filter | Purpose |
|--------|--------|---------|
| `stream_stable` | year < 2019 | Base Adapter training |
| `stream_update` | year ≥ 2019 | Knowledge update evaluation |

## API Reference

### Core Classes

#### `PatchAndRouteLLM`
Main model manager for Frozen Foundation and Expert Pool.

```python
from src.models.core import PatchAndRouteLLM, FrozenFoundationConfig, ExpertConfig, QuantizationType

# Initialize with configuration
config = FrozenFoundationConfig(
    model_id="mistralai/Mistral-7B-Instruct-v0.3",
    quantization=QuantizationType.INT4,
)
llm = PatchAndRouteLLM(foundation_config=config)

# Load base model and attach expert
llm.load_frozen_foundation()
llm.attach_expert(ExpertConfig(name="my_expert", r=16, lora_alpha=32))

# Get components for training
model, tokenizer = llm.get_training_components()

# Save/load trained adapter
llm.save_expert("checkpoints/my_expert")
llm.load_expert("checkpoints/my_expert")
```

#### `SituatedQALoader`
Streaming data loader with temporal filtering.

```python
from src.data.loader import SituatedQALoader, SituatedQAConfig

config = SituatedQAConfig(
    streaming=True,
    temporal_cutoff_year=2019,
    buffer_size=10_000,
    include_context=True,
)
loader = SituatedQALoader(config)

# Get temporally-split streams
stream_stable, stream_update = loader.get_temporal_streams()

# Get formatted stream for training
train_stream = loader.get_formatted_stream(stream_stable, shuffle=True)
```

#### `CounterFactLoader`
Evaluation loader for knowledge editing.

```python
from src.data.loader import CounterFactLoader, DataConfig

loader = CounterFactLoader(DataConfig(streaming=True))
eval_stream = loader.get_evaluation_stream()

# Each example contains: prompt, target_true, target_false, subject
```

#### `PatchAndRouteTrainer`
Training engine with SFTTrainer wrapper.

```python
from src.training.trainer import PatchAndRouteTrainer, TrainingConfig

config = TrainingConfig(
    output_dir="checkpoints/my_expert",
    max_steps=1000,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
)

trainer = PatchAndRouteTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_stream,
    config=config,
)

trainer.train()
trainer.save_model()
```

## Training Configuration

### Model & LoRA

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_id` | `mistralai/Mistral-7B-Instruct-v0.3` | Base model |
| `--quantization` | `int4` | Quantization type (`none`, `int8`, `int4`) |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha scaling factor |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_steps` | 1000 | Training steps |
| `--batch_size` | 4 | Per-device batch size |
| `--gradient_accumulation` | 4 | Gradient accumulation steps |
| `--learning_rate` | 2e-4 | Peak learning rate (cosine scheduler) |
| `--max_seq_length` | 2048 | Maximum sequence length |

### Data

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cutoff_year` | 2019 | Temporal split threshold |
| `--buffer_size` | 10000 | Shuffle buffer size |

### Output

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output_dir` | `checkpoints/situatedqa_base_v1` | Checkpoint directory |
| `--save_steps` | 100 | Steps between saves |
| `--logging_steps` | 10 | Steps between logs |
| `--seed` | 42 | Random seed |
| `--log_level` | `INFO` | Logging verbosity |

Full options: `python train_base_adapter.py --help`

## Hardware Requirements

| Configuration | VRAM | Batch Size |
|--------------|------|------------|
| Minimum | 8 GB | 1 |
| Recommended | 16 GB | 4 |
| Optimal | 24 GB | 8 |

## Dependencies

### Core ML Stack
- **torch** >= 2.1.0
- **transformers** >= 4.40.0
- **peft** >= 0.10.0
- **trl** >= 0.8.0
- **datasets** >= 2.18.0
- **bitsandbytes** >= 0.43.0
- **accelerate** >= 0.28.0

### Development
- **black** >= 24.0.0
- **ruff** >= 0.3.0
- **mypy** >= 1.9.0

See `pyproject.toml` for full dependency list.

## Datasets

Three-dataset evaluation plan:

| Dataset | Type | Purpose | Split Strategy |
|---------|------|---------|----------------|
| **[SituatedQA](https://huggingface.co/datasets/situated_qa)** | Public, temporal | Temporal dynamics evaluation | Pre-2019 = stable, Post-2019 = updates |
| **[CounterFact-Tracing](https://huggingface.co/datasets/NeelNanda/counterfact-tracing)** | Public, counterfactual | Controlled editing (21,919 items) | D_Target (100 edits), D_Control (stability) |
| **AIT QM Corpus** | Proprietary | Enterprise case study | 250-item QA benchmark + curated updates |

### Evaluation Metrics

- **ESR** (Editing Success Rate): % of D_Target where model outputs t_false
- **CFR** (Catastrophic Forgetting Rate): Change in probability of t_true in D_Control (target: ~0%)
- **Efficacy**: Fraction of queries producing updated answer when Knowledge Patches present

## Roadmap

Based on the Master's Thesis timeline (6 months).

### Phase 1: Foundation & Infrastructure (Month 1)
- [x] Frozen Foundation with 4-bit/8-bit quantization
- [x] Base Expert Adapter training (LoRA)
- [x] Streaming data with temporal filtering
- [x] SituatedQA and CounterFact data loaders
- [x] Chat template formatting for instruction tuning
- [x] Configuration serialization (JSON)
- [x] Centralized logging system
- [x] Data preparation pipeline (PDF → Markdown → QA pairs)
- [x] QM corpus preprocessing (AIT proprietary data)

### Phase 2: Core Framework & Base Adapters (Month 2)
- [ ] Expert Pool management system
- [ ] Two-Level Routing structure:
  - [ ] Level 1: Manual Domain Selection (UI "Hard Switch")
  - [ ] Level 2: Intelligent Dispatcher interface (Micro-Router)
- [ ] Train domain Base Adapters:
  - [ ] `SituatedQA_Base_Adapter_t1` (pre-2019 stable facts)
  - [ ] `QM_Base_Adapter_v1` (AIT corpus)

### Phase 3: Router Architectures (Month 3)
- [ ] **Time-Aware Centroid Router with Source-Replay** (embedding-based):
  - [ ] Adapter centroid computation from training data
  - [ ] Cosine similarity routing
  - [ ] Scoped Retrieval for conflict resolution (RAG-augmented)
- [ ] **Parallel-Orchestrator Architecture** (ensemble & synthesis):
  - [ ] Intelligent Router (Query Planner)
  - [ ] Parallel Execution Engine (multi-LoRA batch inference)
  - [ ] Context Synthesis Agent (The Resolver)
- [ ] Probe-and-Judge conflict detection mechanism
- [ ] X-LoRA baseline integration

### Phase 4: Knowledge Patches & Baselines (Month 4)
- [ ] Train Knowledge Patches:
  - [ ] Temporal patches for SituatedQA (post-2019 updates)
  - [ ] QM updates (CEO changes, role modifications)
  - [ ] Counterfactual patches for controlled editing experiments
- [ ] Implement baseline models:
  - [ ] Monolithic LoRA fine-tuning
  - [ ] LoRA + RAG hybrid
  - [ ] L2R (Learning to Route)
  - [ ] X-LoRA (soft-gating)

### Phase 5: Evaluation (Month 5)
- [ ] **O2 - Efficiency**: Wall-clock time, GPU VRAM, FLOPs comparison
- [ ] **O3a - Conflict Resolution**: ESR, flip/reversibility rates (CounterFact-Tracing)
- [ ] **O3b - Cooperative Composition**: Multi-hop accuracy, synthesis quality
- [ ] **O4 - Stability**: CFR targeting 0% on D_Control
- [ ] Ablation studies comparing router variants

### Phase 6: Analysis & Finalization (Month 6)
- [ ] Quantitative and qualitative analysis
- [ ] Hypothesis validation (H1-H3)
- [ ] Public GitHub release with documentation
- [ ] Master's Thesis submission

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Format code
black src/ train_base_adapter.py

# Lint
ruff check src/ train_base_adapter.py

# Type checking
mypy src/
```

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for Transformers, PEFT, and TRL
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for quantization
- Austrian Institute of Technology (AIT) for research collaboration

