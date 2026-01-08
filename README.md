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
| **Knowledge Router** | Dynamic routing mechanism for adapter selection (planned) |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Patch-and-Route                       │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │ Frozen Foundation│    │       Expert Pool          │ │
│  │  (Mistral-7B)   │◄───│  ┌─────┐ ┌─────┐ ┌─────┐  │ │
│  │   4-bit quant   │    │  │LoRA │ │LoRA │ │LoRA │  │ │
│  └─────────────────┘    │  │Base │ │Exp.1│ │Exp.2│  │ │
│                         │  └─────┘ └─────┘ └─────┘  │ │
│                         └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
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
│   ├── data/
│   │   └── loader.py       # SituatedQA & CounterFact streaming loaders
│   ├── models/
│   │   └── core.py         # PatchAndRouteLLM model manager
│   ├── training/
│   │   └── trainer.py      # SFTTrainer for streaming datasets
│   └── utils/
│       ├── config.py       # Configuration management
│       └── logging.py      # Centralized logging
├── train_base_adapter.py   # Main training entry point
├── environment.yml         # Conda environment
└── requirements.txt        # Pip dependencies (fallback)
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
- [ ] Multi-expert inference (Expert Pool)
- [ ] Knowledge Router implementation
- [ ] conflict resolution
- [ ] Evaluation pipeline

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for Transformers, PEFT, and TRL
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for quantization
- Austrian Institute of Technology (AIT) for research collaboration

