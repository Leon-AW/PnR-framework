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
# Single GPU (default: DeepSeek-R1-Distill-Qwen-14B)
# Optimized for 24GB VRAM: batch_size=1, grad_accum=16
python train_monolithic_baseline.py \
    --data_paths data/archive.json \
    --max_steps 2000 \
    --batch_size 1 \
    --gradient_accumulation 16 \
    --max_seq_length 1024 \
    --output_dir checkpoints/base_v1
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
from src.models.core import PatchAndRouteLLM, FrozenFoundationConfig, ExpertConfig

# Load model with trained adapter
llm = PatchAndRouteLLM()
llm.load_frozen_foundation()
llm.load_expert("checkpoints/base_v1")

model, tokenizer = llm.get_training_components()
```

### 5. Use Trained Adapter

```python
from src.models.core import PatchAndRouteLLM

llm = PatchAndRouteLLM()
llm.load_frozen_foundation()
llm.load_expert("checkpoints/base_v1")

# For RAG adapter, format input with documents:
# [Documents:]
# --- Document 1 ---
# {chunk_content}
#
# [Question:]
# {user_question}
```

## Local JSON Fine-Tuning

Train on your own QA datasets with two baseline approaches.

### Dataset Setup

```
PnR-framework/
├── data/
│   ├── archive.json          # Your QA JSON files
│   ├── current.json
│   └── documents/            # Source documents (for RAG)
│       ├── doc1.md
│       └── subfolder/
│           └── doc2.md
```

### JSON Format

```json
[
  {
    "question": "What is the company's refund policy?",
    "answer": "Our refund policy allows returns within 30 days of purchase.",
    "analysis": "CoT reasoning (not used in training)",
    "evidence_snippet": "Returns are accepted within 30 days",
    "file_path": "policies/refunds.md",
    "language": "en",
    "intention_category": "P"
  },
  {
    "question": "Who won the 2030 election?",
    "answer": "I don't have information about future events.",
    "intention_category": "N"
  }
]
```

| Field | Required | Description |
|-------|----------|-------------|
| `question` | Yes | User question |
| `answer` | Yes | Target output (model should generate this) |
| `analysis` | No | CoT reasoning (excluded from training) |
| `evidence_snippet` | RAG only | Text to match for finding relevant chunk |
| `file_path` | RAG only | Path to source document (relative to docs_path) |
| `language` | No | Language code for filtering |
| `intention_category` | No | "N" = negative/unanswerable sample |

### Monolithic Baseline

Single adapter trained on combined datasets (simple question → answer format):

```bash
# Single dataset
python train_monolithic_baseline.py \
    --data_paths data/archive.json \
    --output_dir checkpoints/monolithic_v1

# Multiple datasets combined
python train_monolithic_baseline.py \
    --data_paths data/archive.json data/current.json \
    --output_dir checkpoints/monolithic_combined \
    --max_steps 2000

# With options (24GB VRAM optimization)
python train_monolithic_baseline.py \
    --data_paths data/archive.json \
    --output_dir checkpoints/monolithic_v1 \
    --max_steps 1000 \
    --batch_size 1 \
    --gradient_accumulation 16 \
    --lora_r 16 \
    --language_filter en \
    --no_negatives
```

### RAG Baseline

Separate adapters optimized for RAG retrieval context with noise injection:

```bash
# Train domain-specific adapter
python train_rag_baseline.py \
    --data_path data/archive.json \
    --docs_path data/documents/ \
    --adapter_name archive_rag \
    --output_dir checkpoints/

# Another domain
python train_rag_baseline.py \
    --data_path data/current.json \
    --docs_path data/documents/ \
    --adapter_name current_rag \
    --output_dir checkpoints/

# Custom settings (optimized for 24GB VRAM)
python train_rag_baseline.py \
    --data_path data/archive.json \
    --docs_path data/documents/ \
    --adapter_name archive_rag \
    --noise_min 1 --noise_max 3 \
    --chunk_size 500 \
    --max_seq_length 1024 \
    --batch_size 1 \
    --gradient_accumulation 16
```

### Configuration Options

#### Monolithic Baseline (`train_monolithic_baseline.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_paths` | Required | JSON files (multiple allowed) |
| `--output_dir` | `checkpoints/monolithic_v1` | Checkpoint directory |
| `--system_prompt` | Default prompt | Custom system prompt |
| `--no_negatives` | False | Exclude negative samples |
| `--language_filter` | None | Filter by language code |
| `--validation_split` | 0.1 | Validation fraction |

#### RAG Baseline (`train_rag_baseline.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | Required | Single JSON file |
| `--docs_path` | Required | Documents directory |
| `--adapter_name` | `rag_baseline` | Name for adapter |
| `--noise_min` | 1 | Min noise chunks |
| `--noise_max` | 2 | Max noise chunks |
| `--chunk_size` | 750 | Target chunk tokens |
| `--max_doc_tokens` | 2500 | Threshold for chunking |
| `--max_seq_length` | 1024 | Sequence length (1024 for 24GB VRAM) |

### Quick Test (Smoke Test)

```bash
python train_monolithic_baseline.py \
    --data_paths data/test.json \
    --output_dir checkpoints/test \
    --max_steps 50
```

## VanillaRAG Deployment

Deploy trained RAG adapters for document Q&A.

```python
from src.inference import VanillaRAG, VanillaRAGConfig

config = VanillaRAGConfig(
    model_name="checkpoints/QM_rag/merged",
    load_in_4bit=True,
)
rag = VanillaRAG(config)

# Index documents
rag.index_directory("data/documents/", pattern="**/*.md")

# Query
result = rag.query("What is the procedure for hardness testing?")
print(result["answer"])
print(result["sources"])

# Interactive REPL
rag.interactive_session()
```

## Project Structure

```
PnR-framework/
├── src/
│   ├── data/
│   │   └── loader.py                    # SituatedQA & CounterFact streaming loaders
│   ├── data_loaders/
│   │   ├── local_loader.py              # Local JSON dataset loader
│   │   ├── chunker.py                   # Document chunking for RAG
│   │   └── structure_aware_chunker.py   # Structure-preserving chunker (tables, lists)
│   ├── inference/
│   │   ├── vanilla_rag.py               # Standalone RAG pipeline
│   │   ├── embeddings.py                # Embedding model wrapper
│   │   ├── vector_store.py              # FAISS/ChromaDB backends
│   │   ├── merge_adapter.py             # LoRA → merged model
│   │   └── convert_to_gguf.py           # Merged → GGUF conversion
│   ├── models/
│   │   └── core.py                      # PatchAndRouteLLM model manager
│   ├── routing/
│   │   ├── base.py                      # BaseRouter abstract class (Strategy Pattern)
│   │   ├── centroid_router.py           # Time-Aware Centroid Router
│   │   ├── manifest.py                  # Adapter registration & centroids
│   │   └── source_replay.py             # FAISS-based retrieval for T_old
│   ├── training/
│   │   └── trainer.py                   # SFTTrainer for streaming datasets
│   └── utils/
│       ├── config.py                    # Configuration management
│       └── logging.py                   # Centralized logging
├── scripts/
│   ├── compute_centroids.py             # Offline centroid computation
│   ├── merge_and_convert.sh             # Adapter → GGUF pipeline
│   └── start_llama_server.sh            # llama.cpp server launcher
├── examples/
│   └── router_demo.py                   # Router demonstration
├── checkpoints/                         # Trained adapter checkpoints
├── train_base_adapter.py                # SituatedQA training entry point
├── train_monolithic_baseline.py         # Monolithic JSON training
├── train_rag_baseline.py                # RAG baseline training
├── interactive_inference.py             # Interactive routing demo
├── environment.yml                      # Conda environment (Python 3.11)
├── requirements.txt                     # Pip dependencies (fallback)
└── pyproject.toml                       # Project metadata
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

### Structure-Aware Chunking
Preserves tables, lists, and section hierarchies in QM documents:

```python
from src.data_loaders import StructureAwareChunker, StructuredChunkConfig

config = StructuredChunkConfig(
    max_chunk_tokens=750,
    table_max_tokens=1500,
    list_max_tokens=500,
    include_breadcrumb=True,
)
chunker = StructureAwareChunker(config)
chunks = chunker.chunk_document("path/to/qm_doc.md")
```

## API Reference

### Core Classes

#### `PatchAndRouteLLM`
Main model manager for Frozen Foundation and Expert Pool.

```python
from src.models.core import PatchAndRouteLLM, FrozenFoundationConfig, ExpertConfig, QuantizationType

# Initialize with default DeepSeek-R1-Distill-Qwen-14B
llm = PatchAndRouteLLM()

# Or specify custom configuration
config = FrozenFoundationConfig(
    model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
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

#### `LocalJSONLoader`
Loader for local JSON QA datasets with simple and RAG formats.

```python
from src.data import LocalJSONLoader, LocalJSONConfig

# Simple format (monolithic baseline)
config = LocalJSONConfig(
    data_paths=["data/qa.json"],
    format_type="simple",
    include_negatives=True,
    validation_split=0.1,
)
loader = LocalJSONLoader(config)
dataset = loader.load()  # Returns Dataset or DatasetDict with train/test

# RAG format with document chunking
config = LocalJSONConfig(
    data_paths=["data/qa.json"],
    docs_base_path="data/documents/",
    format_type="rag",
    noise_chunks=(1, 2),  # Inject 1-2 noise chunks
)
loader = LocalJSONLoader(config)
dataset = loader.load()
```

#### `VanillaRAG`
Standalone RAG pipeline for document Q&A.

```python
from src.inference import VanillaRAG, VanillaRAGConfig

config = VanillaRAGConfig(
    model_name="checkpoints/QM_rag/merged",
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    vector_store_type="faiss",
    top_k=5,
    load_in_4bit=True,
)
rag = VanillaRAG(config)

# Index documents
rag.index_directory("data/documents/", pattern="**/*.md")

# Query with RAG
result = rag.query("What is the procedure for hardness testing?")
print(result["answer"])
print(result["sources"])

# Interactive REPL
rag.interactive_session()
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
| `--model_id` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | Base model |
| `--quantization` | `int4` | Quantization type (`none`, `int8`, `int4`) |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha scaling factor |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_steps` | 1000 | Training steps |
| `--batch_size` | 1 | Per-device batch size (14B model) |
| `--gradient_accumulation` | 16 | Gradient accumulation steps |
| `--learning_rate` | 2e-4 | Peak learning rate (cosine scheduler) |
| `--max_seq_length` | 1024 | Maximum sequence length |

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

### RAG & Embeddings
- **sentence-transformers** >= 2.2.0
- **faiss-cpu** >= 1.7.4
- **chromadb** >= 0.4.0

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
- [x] Multi-expert inference (Expert Pool)
- [x] Knowledge Router implementation (Time-Aware Centroid Router)
- [x] Conflict resolution (Source-Replay mechanism)
- [ ] Evaluation pipeline
- [ ] Parallel Orchestrator (Section 4.4.2)
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
