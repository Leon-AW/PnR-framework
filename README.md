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

# For NVIDIA Blackwell GPUs (RTX 6000 Ada / B100), install PyTorch Nightly with CUDA 12.8:
./setup_blackwell_env.sh

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Train Base Expert Adapter

```bash
# Single GPU/MIG instance (default: DeepSeek-R1-Distill-Qwen-14B)
# Optimized for 24GB VRAM: batch_size=1, grad_accum=16
python train_monolithic_baseline.py \
    --data_paths data/archive.json \
    --max_steps 2000 \
    --batch_size 1 \
    --gradient_accumulation 16 \
    --max_seq_length 1024 \
    --output_dir checkpoints/base_v1

# Specify MIG devices (e.g., use first 2 MIG partitions)
python train_monolithic_baseline.py \
    --data_paths data/archive.json \
    --target_devices 0 1 \
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
from src.inference import PatchAndRouteInference

# Initialize pipeline
pipeline = PatchAndRouteInference(
    model_id="mistralai/Mistral-7B-Instruct-v0.3",
    router_path="router_state/",
    embedding_model_path="/path/to/KaLM-Embedding-Gemma3-12B",
)
### 3. Run on Cluster (SLURM)

```bash
# IMPORTANT: Validate GPU setup BEFORE submitting jobs
python validate_gpu_setup.py --dry-run

# Single GPU/MIG (recommended for reliability)
sbatch run_training_single_gpu.sh

# Or use the default script
sbatch run_training_slurm.sh

# Multi-GPU with accelerate (for faster training)
sbatch run_training_multi_gpu.sh
```

#### GPU Validation

Always run validation before training to catch configuration issues:

```bash
# Check all available GPUs
python validate_gpu_setup.py

# Check specific devices
python validate_gpu_setup.py --target-devices 0 1

# Full validation with memory estimates
python validate_gpu_setup.py --dry-run
```

#### MIG (Multi-Instance GPU) Notes

If your cluster uses MIG-enabled GPUs (common on A100, H100, Blackwell):

1. Each MIG instance appears as a separate CUDA device
2. Use `--target_devices 0` for single MIG instance training
3. Request `--gres=gpu:1` in SLURM for single-device training
4. The validation script will detect MIG instances automatically

#### Why Single-GPU Training is Recommended

For **14B models with LoRA** on **24GB GPUs/MIG instances**, single-GPU training is optimal:

| Multi-GPU Issue | Impact |
|-----------------|--------|
| Each process loads full model | 4x 18GB = 72GB needed, but GPUs share memory |
| LoRA trains only ~0.1% of params | Gradient sync overhead dominates compute savings |
| Memory-bound workload | Batch size stays at 1 regardless of GPU count |
| MIG memory isolation | Processes compete for same physical memory |

**Result**: Multi-GPU is barely faster and often fails with OOM.

**When multi-GPU helps**: Full fine-tuning (not LoRA) with 48GB+ GPUs and batch size > 1

### 3. Use Trained Adapter

```python
from src.models.core import PatchAndRouteLLM, FrozenFoundationConfig, ExpertConfig

# Load model with trained adapter (defaults to DeepSeek-R1-Distill-Qwen-14B)
llm = PatchAndRouteLLM()
llm.load_frozen_foundation()
llm.load_expert("checkpoints/base_v1")

# Query with automatic routing
result = pipeline.generate("Who is the Chancellor of Germany in 2023?")

print(result.response)
print(f"Adapter used: {result.adapter_loaded}")
print(f"Had conflict: {result.routing_result.has_conflict}")
```

## Local JSON Fine-Tuning

Train on your own QA datasets with two baseline approaches:

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

# With options
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

# Custom chunking settings
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

### Using Trained Adapters

```python
from src.models.core import PatchAndRouteLLM

llm = PatchAndRouteLLM()
llm.load_frozen_foundation()
llm.load_expert("checkpoints/monolithic_v1")

model, tokenizer = llm.get_training_components()

# For RAG adapter, format input with documents:
# [Documents:]
# --- Document 1 ---
# {chunk_content}
#
# [Question:]
# {user_question}
```

## RAG Chatbot Deployment

Deploy trained RAG adapters as interactive chatbots with two options.

> **Data Privacy**: All processing happens locally on your hardware. No data is sent to external APIs. Models are downloaded once from HuggingFace, then everything runs offline. See [docs/rag_chatbot.md](docs/rag_chatbot.md#data-privacy--offline-operation) for details.

| Option | Use Case | Interface |
|--------|----------|-----------|
| **VanillaRAG** | Programmatic access, CLI testing | Python API / Terminal |
| **OpenWebUI + llama.cpp** | End-user chat interface | Web UI |

### Quick Start: VanillaRAG

```python
from src.inference import VanillaRAG, VanillaRAGConfig

config = VanillaRAGConfig(
    model_name="checkpoints/QM_rag/merged",
    load_in_4bit=True,
)
rag = VanillaRAG(config)

# Index QM documents
rag.index_directory("src/data/documents/DE/LKR/", pattern="**/*.md")

# Query
result = rag.query("Wie ist der Ablauf bei der Mikrohärteprüfung?")
print(result["answer"])
```

### Quick Start: OpenWebUI

```bash
# 1. Activate conda environment
conda activate pnr

# 2. Merge adapter and convert to GGUF (skip --skip-merge if not yet merged)
./scripts/merge_and_convert.sh --skip-merge

# 3. Start llama.cpp server
./scripts/start_llama_server.sh

# 4. Start OpenWebUI (new terminal)
./scripts/setup_openwebui.sh

# 5. Open http://localhost:3000
#    - Create admin account
#    - Upload documents to Workspace > Knowledge
#    - Chat and reference docs with # (citations appear at bottom of responses)
```

### Key Features

- **Structure-Aware Chunking**: Preserves tables, lists, and section hierarchies in QM documents
- **Multilingual Embeddings**: German/English support via `paraphrase-multilingual-MiniLM-L12-v2`
- **Flexible Storage**: FAISS (in-memory) or ChromaDB (persistent)
- **Quantization Options**: q4_k_m (~10GB VRAM) to q8_0 (~16GB VRAM)
- **RAG Citations**: OpenWebUI displays clickable source references with relevance scores
- **HPC-Ready**: Apptainer (no Docker/sudo), conda-built llama.cpp

**Full documentation**: [docs/rag_chatbot.md](docs/rag_chatbot.md)

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
│   ├── __init__.py              # Package init (v0.1.0)
│   ├── data_loaders/
│   │   ├── __init__.py
│   │   ├── local_loader.py      # Local JSON dataset loader
│   │   ├── chunker.py           # Document chunking for RAG
│   │   └── structure_aware_chunker.py  # QM-document aware chunking
│   ├── inference/               # RAG chatbot deployment
│   │   ├── __init__.py
│   │   ├── vanilla_rag.py       # Standalone RAG pipeline
│   │   ├── embeddings.py        # Embedding model wrapper
│   │   ├── vector_store.py      # FAISS/ChromaDB backends
│   │   ├── merge_adapter.py     # LoRA → merged model
│   │   └── convert_to_gguf.py   # Merged → GGUF conversion
│   ├── models/
│   │   ├── __init__.py
│   │   └── core.py              # PatchAndRouteLLM model manager
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py           # SFTTrainer for streaming datasets
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Configuration serialization (JSON)
│       └── logging.py           # Centralized logging setup
├── scripts/                     # Deployment scripts
│   ├── start_llama_server.sh    # llama.cpp server launcher
│   ├── setup_openwebui.sh       # OpenWebUI Docker setup
│   └── merge_and_convert.sh     # Full adapter → GGUF pipeline
├── docs/
│   └── rag_chatbot.md           # RAG chatbot documentation
├── data/                        # Your datasets (create this)
│   ├── *.json                   # QA JSON files
│   └── documents/               # Source documents for RAG
├── train_base_adapter.py        # SituatedQA training
├── train_monolithic_baseline.py # Monolithic JSON training
├── train_rag_baseline.py        # RAG baseline training
├── environment.yml              # Conda environment (Python 3.11)
├── requirements.txt             # Pip dependencies (fallback)
├── pyproject.toml               # Project metadata and dependencies
├── setup_blackwell_env.sh       # PyTorch Nightly setup for Blackwell
└── run_training_slurm.sh        # SLURM submission script
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
## API Reference

### Core Classes

#### `PatchAndRouteLLM`
Main model manager for Frozen Foundation and Expert Pool.

```python
from src.models.core import PatchAndRouteLLM, FrozenFoundationConfig, ExpertConfig, QuantizationType

# Initialize with default DeepSeek-R1-Distill-Qwen-14B
llm = PatchAndRouteLLM()

# Or specify custom configuration with MIG targeting
config = FrozenFoundationConfig(
    model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    quantization=QuantizationType.INT4,
    target_devices=[0, 1],  # Use specific MIG instances
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

#### `LocalJSONLoader`
Loader for local JSON QA datasets with simple and RAG formats.

```python
from src.data import LocalJSONLoader, LocalJSONConfig, create_simple_loader, create_rag_loader

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

# Factory functions for convenience
loader = create_simple_loader(["data/qa.json"])
loader = create_rag_loader("data/qa.json", "data/documents/")
```

#### `SemanticChunker`
Document chunking for RAG-based fine-tuning.

```python
from src.data import SemanticChunker, ChunkConfig

config = ChunkConfig(
    max_doc_tokens=2500,   # Whole doc if smaller
    chunk_size=750,        # Target chunk size
    chunk_overlap=75,      # Overlap between chunks
)
chunker = SemanticChunker(config)

# Chunk a document
chunks = chunker.chunk_document("path/to/doc.md")

# Find chunk matching evidence
relevant = chunker.find_relevant_chunk(chunks, "evidence text")

# Get noise chunks
noise = chunker.get_noise_chunks(all_chunks, exclude=[relevant], n=2)

# Build context string
context = chunker.build_context(relevant, noise, shuffle=True)
```

#### `StructureAwareChunker`
Structure-preserving chunker for QM documents (tables, lists, headers).

```python
from src.data_loaders import StructureAwareChunker, StructuredChunkConfig

config = StructuredChunkConfig(
    max_chunk_tokens=750,
    table_max_tokens=1500,  # Keep tables atomic
    list_max_tokens=500,    # Keep lists together
    include_breadcrumb=True,
    include_path=True,
)
chunker = StructureAwareChunker(config)

chunks = chunker.chunk_document("path/to/qm_doc.md")

for chunk in chunks:
    print(f"[{chunk.content_type}] {chunk.section_breadcrumb}")
    print(chunk.format_with_context())
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

