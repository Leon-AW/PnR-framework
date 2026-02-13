#!/bin/bash
# =============================================================================
# Start Advanced RAG Server
# =============================================================================
#
# Starts the FastAPI RAG server that sits between OpenWebUI and llama.cpp,
# providing hybrid retrieval (FAISS + BM25), reranking, and citation tracking.
#
# Usage:
#   ./scripts/start_rag_server.sh
#   ./scripts/start_rag_server.sh --port 8000 --llama-port 8080
#   ./scripts/start_rag_server.sh --start-llama
#   ./scripts/start_rag_server.sh --log-level debug
#
# Requirements:
#   - Conda env 'pnr' activated
#   - Indices built with: python scripts/index_documents_advanced.py --source all
#   - llama.cpp server running (or use --start-llama)
#
# =============================================================================

set -e

# Ensure CUDA and C++ libraries are available
if [[ -n "$CONDA_PREFIX" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    # pip-installed nvidia packages store CUDA libs under site-packages/nvidia/*/lib/
    NVIDIA_SITE_PKGS="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia"
    if [[ -d "$NVIDIA_SITE_PKGS" ]]; then
        for nvidia_lib_dir in "$NVIDIA_SITE_PKGS"/*/lib; do
            [[ -d "$nvidia_lib_dir" ]] && export LD_LIBRARY_PATH="$nvidia_lib_dir:$LD_LIBRARY_PATH"
        done
    fi
fi

# Default configuration
RAG_PORT="${RAG_PORT:-8000}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
RAG_LOG_LEVEL="${RAG_LOG_LEVEL:-info}"
START_LLAMA=false
INDEX_BASE_DIR="${RAG_INDEX_BASE_DIR:-./qm_vectorstore_advanced}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            RAG_PORT="$2"
            shift 2
            ;;
        --llama-port)
            LLAMA_PORT="$2"
            shift 2
            ;;
        --log-level)
            RAG_LOG_LEVEL="$2"
            shift 2
            ;;
        --start-llama)
            START_LLAMA=true
            shift
            ;;
        --index-dir)
            INDEX_BASE_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --port N          RAG server port (default: 8000)"
            echo "  --llama-port N    llama.cpp server port (default: 8080)"
            echo "  --log-level LEVEL Log level: debug, info, warning (default: info)"
            echo "  --start-llama     Start llama.cpp server alongside RAG server"
            echo "  --index-dir DIR   Index directory (default: ./qm_vectorstore_advanced)"
            echo "  --help, -h        Show this help"
            echo ""
            echo "Environment variables:"
            echo "  RAG_PORT, RAG_LLAMA_URL, RAG_LOG_LEVEL, RAG_INDEX_BASE_DIR"
            echo "  RAG_ENABLE_RERANKING, RAG_ENABLE_CITATIONS, RAG_RERANK_TOP_K"
            echo ""
            echo "Prerequisites:"
            echo "  1. Build indices: python scripts/index_documents_advanced.py --source all"
            echo "  2. Start llama.cpp: ./scripts/start_llama_server.sh"
            echo "     (or use --start-llama to start both)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if indices exist
if [[ ! -d "$INDEX_BASE_DIR" ]]; then
    echo "Error: Index directory not found: $INDEX_BASE_DIR"
    echo ""
    echo "Please build indices first:"
    echo "  python scripts/index_documents_advanced.py --source all"
    exit 1
fi

# Optionally start llama.cpp server
LLAMA_PID=""
if [[ "$START_LLAMA" == "true" ]]; then
    echo "=============================================="
    echo "Starting llama.cpp server (port $LLAMA_PORT)..."
    echo "=============================================="

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PORT=$LLAMA_PORT "$SCRIPT_DIR/start_llama_server.sh" &
    LLAMA_PID=$!

    # Wait for llama.cpp to be ready
    echo "Waiting for llama.cpp to start..."
    for i in $(seq 1 60); do
        if curl -s -o /dev/null "http://localhost:$LLAMA_PORT/health" 2>/dev/null; then
            echo "llama.cpp is ready!"
            break
        fi
        if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
            echo "Error: llama.cpp process exited unexpectedly"
            exit 1
        fi
        sleep 2
    done
    echo ""
fi

# Export configuration
export RAG_PORT
export RAG_LLAMA_URL="http://localhost:$LLAMA_PORT"
export RAG_LOG_LEVEL
export RAG_INDEX_BASE_DIR="$INDEX_BASE_DIR"

echo "=============================================="
echo "Starting Advanced RAG Server"
echo "=============================================="
echo "RAG Server Port:   $RAG_PORT"
echo "llama.cpp URL:     $RAG_LLAMA_URL"
echo "Index Directory:   $INDEX_BASE_DIR"
echo "Log Level:         $RAG_LOG_LEVEL"
echo "=============================================="
echo ""
echo "Endpoints:"
echo "  Health:     http://localhost:$RAG_PORT/health"
echo "  Chat:       http://localhost:$RAG_PORT/v1/chat/completions"
echo "  Models:     http://localhost:$RAG_PORT/v1/models"
echo "  Stats:      http://localhost:$RAG_PORT/admin/stats"
echo ""
echo "Press Ctrl+C to stop"
echo "=============================================="

# Cleanup on exit
cleanup() {
    if [[ -n "$LLAMA_PID" ]] && kill -0 "$LLAMA_PID" 2>/dev/null; then
        echo "Stopping llama.cpp server (PID: $LLAMA_PID)..."
        kill "$LLAMA_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Start the RAG server
exec python -m src.inference.rag_server
