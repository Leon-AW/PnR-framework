# Patch-and-Route Framework

> A Modular Framework for Continual Learning in Enterprise LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This framework implements the **Patch-and-Route** architecture for continual learning in Large Language Models, enabling domain-specific knowledge integration without catastrophic forgetting.

## Core Concepts

| Term | Description |
|------|-------------|
| **Frozen Foundation** | Base LLM with frozen parameters (e.g., Mistral-7B) |
| **Expert Pool** | Collection of domain-specific LoRA adapters |
| **Knowledge Router** | Time-Aware Centroid Router for dynamic adapter selection |
| **Source-Replay** | RAG-style retrieval from older conflicting adapters |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Patch-and-Route Pipeline                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐     ┌─────────────────────┐                           │
│  │  User Query  │────▶│   Centroid Router   │                           │
│  └──────────────┘     │  (Embed + Match)    │                           │
│                       └──────────┬──────────┘                           │
│                                  │                                       │
│                    ┌─────────────┼─────────────┐                        │
│                    ▼             ▼             ▼                        │
│              ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│              │ Adapter  │  │ Adapter  │  │ Adapter  │  Expert Pool     │
│              │  Base    │  │  Geo_DE  │  │ Temp_23  │                   │
│              │(centroid)│  │(centroid)│  │(centroid)│                   │
│              └──────────┘  └──────────┘  └──────────┘                   │
│                    │             │             │                        │
│                    └─────────────┼─────────────┘                        │
│                                  │                                       │
│                    ┌─────────────▼─────────────┐                        │
│                    │    Conflict Detection     │                        │
│                    │   (Multiple Matches?)     │                        │
│                    └─────────────┬─────────────┘                        │
│                                  │                                       │
│           ┌──────────────────────┼──────────────────────┐               │
│           │ Winner (T_new)       │              Loser (T_old)           │
│           ▼                      │                      ▼               │
│  ┌─────────────────┐             │         ┌─────────────────┐          │
│  │  Weight Loading │             │         │  Source-Replay  │          │
│  │  (Load LoRA)    │             │         │  (FAISS RAG)    │          │
│  └────────┬────────┘             │         └────────┬────────┘          │
│           │                      │                  │                   │
│           │                      │     Retrieved Context               │
│           │                      │          ▼                          │
│           │              ┌───────────────────────────┐                  │
│           └─────────────▶│     Prompt Builder        │◀─────────────────┘
│                          │ [System] + [Context] +    │                   │
│                          │ [Query]                   │                   │
│                          └───────────┬───────────────┘                   │
│                                      │                                   │
│                                      ▼                                   │
│                          ┌───────────────────────────┐                   │
│                          │   Frozen Foundation       │                   │
│                          │   (Mistral-7B + LoRA)     │                   │
│                          └───────────┬───────────────┘                   │
│                                      │                                   │
│                                      ▼                                   │
│                          ┌───────────────────────────┐                   │
│                          │       Response            │                   │
│                          └───────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
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
    --max_steps 2000 \
    --batch_size 4 \
    --output_dir checkpoints/situatedqa_base_v1
```

### 3. Compute Centroids for Routing

```bash
python scripts/compute_centroids.py \
    --checkpoints_dir checkpoints/ \
    --embedding_model /path/to/KaLM-Embedding-Gemma3-12B \
    --output_dir router_state/ \
    --index_for_replay
```

### 4. Run Inference with Routing

```python
from src.inference import PatchAndRouteInference

# Initialize pipeline
pipeline = PatchAndRouteInference(
    model_id="mistralai/Mistral-7B-Instruct-v0.3",
    router_path="router_state/",
    embedding_model_path="/path/to/KaLM-Embedding-Gemma3-12B",
)

# Query with automatic routing
result = pipeline.generate("Who is the Chancellor of Germany in 2023?")

print(result.response)
print(f"Adapter used: {result.adapter_loaded}")
print(f"Had conflict: {result.routing_result.has_conflict}")
```

## Project Structure

```
PnR-framework/
├── src/
│   ├── data/
│   │   └── loader.py           # SituatedQA & CounterFact streaming loaders
│   ├── models/
│   │   └── core.py             # PatchAndRouteLLM model manager
│   ├── routing/
│   │   ├── base.py             # BaseRouter abstract class (Strategy Pattern)
│   │   ├── centroid_router.py  # Time-Aware Centroid Router
│   │   ├── manifest.py         # Adapter registration & centroids
│   │   └── source_replay.py    # FAISS-based retrieval for T_old
│   ├── training/
│   │   └── trainer.py          # SFTTrainer for streaming datasets
│   ├── inference.py            # Unified inference pipeline
│   └── utils/
│       ├── config.py           # Configuration management
│       └── logging.py          # Centralized logging
├── scripts/
│   └── compute_centroids.py    # Offline centroid computation
├── examples/
│   └── router_demo.py          # Router demonstration
├── checkpoints/                # Trained adapter checkpoints
├── train_base_adapter.py       # Main training entry point
├── environment.yml             # Conda environment
└── requirements.txt            # Pip dependencies (fallback)
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

### Time-Aware Routing
Automatic adapter selection with conflict resolution:

```python
from src.routing import CentroidRouter

# Initialize router
router = CentroidRouter(
    embedding_model_path="/path/to/embedding-model",
    similarity_threshold=0.65,
)

# Register adapters from checkpoints
router.register_from_checkpoints("checkpoints/")

# Compute centroids (offline)
router.compute_all_centroids()

# Route query (online)
result = router.route("Who is the German Chancellor?")
print(f"Winner: {result.winner_adapter}")  # e.g., "patch_geo_germany"
print(f"Conflict: {result.has_conflict}")
```

### Source-Replay (Conflict Resolution)
When multiple adapters match, the **newest wins** (Weight Loading), older adapters contribute via **retrieval**:

| Adapter Role | Mechanism | Description |
|--------------|-----------|-------------|
| **T_new** (Winner) | Weight Loading | LoRA weights loaded into model |
| **T_old** (Loser) | Source-Replay | Training data retrieved via FAISS |

The retrieved context is injected into the prompt, ensuring both old and new knowledge inform the response.

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_id` | `mistralai/Mistral-7B-Instruct-v0.3` | Base model |
| `--max_steps` | 1000 | Training steps |
| `--batch_size` | 4 | Per-device batch size |
| `--learning_rate` | 2e-4 | Peak learning rate |
| `--lora_r` | 16 | LoRA rank |
| `--cutoff_year` | 2019 | Temporal split threshold |

Full options: `python train_base_adapter.py --help`

## Hardware Requirements

| Configuration | VRAM | Batch Size |
|--------------|------|------------|
| Minimum | 8 GB | 1 |
| Recommended | 16 GB | 4 |
| Optimal | 24 GB | 8 |

## Datasets

- **[SituatedQA](https://huggingface.co/datasets/situated_qa)** - Temporally-situated questions (Zhang & Choi, 2021)
- **[CounterFact-Tracing](https://huggingface.co/datasets/NeelNanda/counterfact-tracing)** - Knowledge editing evaluation (Nanda, 2022)

## Roadmap

- [x] Frozen Foundation with 4-bit quantization
- [x] Base Expert Adapter training (LoRA)
- [x] Streaming data with temporal filtering
- [x] Multi-expert inference (Expert Pool)
- [x] Knowledge Router implementation (Time-Aware Centroid Router)
- [x] Conflict resolution (Source-Replay mechanism)
- [ ] Evaluation pipeline
- [ ] Parallel Orchestrator (Section 4.4.2)

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for Transformers, PEFT, and TRL
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for quantization
- Austrian Institute of Technology (AIT) for research collaboration

