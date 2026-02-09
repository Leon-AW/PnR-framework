#!/bin/bash
# =============================================================================
# Start llama.cpp Server
# =============================================================================
#
# Starts llama-server with the quantized QM-RAG model.
#
# Usage:
#   ./scripts/start_llama_server.sh
#   ./scripts/start_llama_server.sh --model path/to/model.gguf
#   ./scripts/start_llama_server.sh --port 8080 --ctx-size 8192
#
# Requirements:
#   - llama.cpp installed with llama-server binary
#   - GGUF model file (run merge_and_convert.sh first)
#
# =============================================================================

set -e

# Ensure conda libraries are available (llama.cpp was built against conda's libstdc++)
if [[ -n "$CONDA_PREFIX" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Default configuration
MODEL="${MODEL:-checkpoints/QM_rag_cot_v2/gguf/merged-q5_k_m.gguf}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-49152}"
N_GPU_LAYERS="${N_GPU_LAYERS:-35}"
# Note: conda sets $HOST to the build triplet (e.g. x86_64-conda-linux-gnu),
# so we use a different variable name to avoid conflicts
HOST="${LLAMA_HOST:-0.0.0.0}"
THREADS="${THREADS:-8}"
BATCH_SIZE="${BATCH_SIZE:-512}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)
            MODEL="$2"
            shift 2
            ;;
        --port|-p)
            PORT="$2"
            shift 2
            ;;
        --ctx-size|-c)
            CTX_SIZE="$2"
            shift 2
            ;;
        --n-gpu-layers|-ngl)
            N_GPU_LAYERS="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --threads|-t)
            THREADS="$2"
            shift 2
            ;;
        --batch-size|-b)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model, -m PATH       Path to GGUF model (default: checkpoints/QM_rag_cot_v2/gguf/merged-q5_k_m.gguf)"
            echo "  --port, -p PORT        Server port (default: 8080)"
            echo "  --ctx-size, -c SIZE    Context size (default: 49152)"
            echo "  --n-gpu-layers, -ngl N Number of GPU layers (default: 35)"
            echo "  --host HOST            Host address (default: 0.0.0.0)"
            echo "  --threads, -t N        Number of threads (default: 8)"
            echo "  --batch-size, -b N     Batch size (default: 512)"
            echo "  --help, -h             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Find llama-server binary
LLAMA_SERVER=""
for cmd in "llama-server" "server"; do
    if command -v "$cmd" &> /dev/null; then
        LLAMA_SERVER="$cmd"
        break
    fi
done

# Check common installation paths
if [[ -z "$LLAMA_SERVER" ]]; then
    for path in "$HOME/llama.cpp/build/bin/llama-server" \
                "$HOME/llama.cpp/llama-server" \
                "/opt/llama.cpp/build/bin/llama-server" \
                "./llama.cpp/build/bin/llama-server"; do
        if [[ -x "$path" ]]; then
            LLAMA_SERVER="$path"
            break
        fi
    done
fi

if [[ -z "$LLAMA_SERVER" ]]; then
    echo "Error: llama-server not found."
    echo ""
    echo "Please install llama.cpp:"
    echo "  git clone https://github.com/ggml-org/llama.cpp"
    echo "  cd llama.cpp"
    echo "  make GGML_CUDA=1  # or just 'make' for CPU-only"
    exit 1
fi

# Check model file exists
# DeepSeek-R1 chat template (ensures correct <｜User｜>/<｜Assistant｜> format)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$SCRIPT_DIR/deepseek-r1.jinja}"

if [[ ! -f "$MODEL" ]]; then
    echo "Error: Model file not found: $MODEL"
    echo ""
    echo "Please run the merge and convert process first:"
    echo "  ./scripts/merge_and_convert.sh"
    exit 1
fi

echo "=============================================="
echo "Starting llama.cpp Server"
echo "=============================================="
echo "Model:         $MODEL"
echo "Port:          $PORT"
echo "Context Size:  $CTX_SIZE"
echo "GPU Layers:    $N_GPU_LAYERS"
echo "Host:          $HOST"
echo "Threads:       $THREADS"
echo "Batch Size:    $BATCH_SIZE"
echo "Chat Template: $CHAT_TEMPLATE"
echo "=============================================="
echo ""
echo "API Endpoint:  http://$HOST:$PORT/v1"
echo "Health Check:  http://$HOST:$PORT/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=============================================="

# Start server with DeepSeek-R1 chat template
TEMPLATE_ARGS=""
if [[ -f "$CHAT_TEMPLATE" ]]; then
    TEMPLATE_ARGS="--jinja --chat-template-file $CHAT_TEMPLATE"
else
    echo "Warning: Chat template not found at $CHAT_TEMPLATE"
    echo "Using model's embedded template (may be incorrect for DeepSeek-R1)"
fi

exec "$LLAMA_SERVER" \
    --model "$MODEL" \
    --port "$PORT" \
    --host "$HOST" \
    --ctx-size "$CTX_SIZE" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --threads "$THREADS" \
    --batch-size "$BATCH_SIZE" \
    --parallel 2 \
    --cont-batching \
    $TEMPLATE_ARGS
