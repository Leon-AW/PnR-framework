#!/bin/bash
# =============================================================================
# Setup and Start OpenWebUI (Apptainer/Singularity)
# =============================================================================
#
# Sets up OpenWebUI using Apptainer (rootless, no sudo required),
# configured to use the llama.cpp server as a backend.
#
# Usage:
#   ./scripts/setup_openwebui.sh                           # Start (advanced-rag mode)
#   ./scripts/setup_openwebui.sh --mode advanced-rag       # Use external RAG server
#   ./scripts/setup_openwebui.sh --mode builtin-rag        # Use OpenWebUI's built-in RAG
#   ./scripts/setup_openwebui.sh --stop                    # Stop
#   ./scripts/setup_openwebui.sh --status                  # Check status
#   ./scripts/setup_openwebui.sh --llama-port 8080         # Custom llama port
#
# Requirements:
#   - Apptainer or Singularity installed (no sudo needed)
#   - For builtin-rag: llama.cpp server running
#   - For advanced-rag: RAG server running (start_rag_server.sh) + llama.cpp
#
# =============================================================================

set -e

# Default configuration
WEBUI_PORT="${WEBUI_PORT:-3000}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
RAG_PORT="${RAG_PORT:-8000}"
RAG_MODE="${RAG_MODE:-advanced-rag}"  # "advanced-rag" or "builtin-rag"
DATA_DIR="${DATA_DIR:-$HOME/.openwebui}"
SIF_DIR="${SIF_DIR:-$HOME/.openwebui/images}"
SIF_FILE="$SIF_DIR/openwebui.sif"
PID_FILE="$DATA_DIR/openwebui.pid"
LOG_FILE="$DATA_DIR/openwebui.log"
IMAGE_URL="docker://ghcr.io/open-webui/open-webui:main"

# RAG configuration
EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKING_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE=750
CHUNK_OVERLAP=75
RAG_TOP_K=5
RAG_RELEVANCE_THRESHOLD=0.0

# Detect container runtime (prefer apptainer over singularity)
CONTAINER_CMD=""
for cmd in apptainer singularity; do
    if command -v "$cmd" &> /dev/null; then
        CONTAINER_CMD="$cmd"
        break
    fi
done

if [[ -z "$CONTAINER_CMD" ]]; then
    echo "Error: Neither apptainer nor singularity found."
    echo "Install apptainer: https://apptainer.org/docs/admin/main/installation.html"
    exit 1
fi

# Parse command line arguments
ACTION="start"
REBUILD_IMAGE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --stop)
            ACTION="stop"
            shift
            ;;
        --restart)
            ACTION="restart"
            shift
            ;;
        --logs)
            ACTION="logs"
            shift
            ;;
        --status)
            ACTION="status"
            shift
            ;;
        --rebuild)
            REBUILD_IMAGE=true
            shift
            ;;
        --webui-port)
            WEBUI_PORT="$2"
            shift 2
            ;;
        --llama-port)
            LLAMA_PORT="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --mode)
            RAG_MODE="$2"
            shift 2
            ;;
        --rag-port)
            RAG_PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Actions:"
            echo "  --stop           Stop OpenWebUI"
            echo "  --restart        Restart OpenWebUI"
            echo "  --logs           Show logs (tail -f)"
            echo "  --status         Show running status"
            echo "  --rebuild        Re-pull the container image"
            echo ""
            echo "Options:"
            echo "  --webui-port N   OpenWebUI port (default: 3000)"
            echo "  --llama-port N   llama.cpp server port (default: 8080)"
            echo "  --rag-port N     RAG server port (default: 8000)"
            echo "  --mode MODE      RAG mode: advanced-rag or builtin-rag (default: advanced-rag)"
            echo "  --data-dir DIR   Data directory (default: ~/.openwebui)"
            echo "  --help, -h       Show this help"
            echo ""
            echo "Modes:"
            echo "  advanced-rag   OpenWebUI → RAG Server (port $RAG_PORT) → llama.cpp (port $LLAMA_PORT)"
            echo "                 Requires: start_rag_server.sh + start_llama_server.sh"
            echo "  builtin-rag    OpenWebUI → llama.cpp directly (OpenWebUI handles RAG)"
            echo "                 Requires: start_llama_server.sh"
            echo ""
            echo "Runtime: $CONTAINER_CMD"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$DATA_DIR" "$SIF_DIR"

# Helper: check if process is running
is_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        # Stale PID file
        rm -f "$PID_FILE"
    fi
    return 1
}

# Helper: stop the process
stop_openwebui() {
    if is_running; then
        local pid
        pid=$(cat "$PID_FILE")
        echo "Stopping OpenWebUI (PID: $pid)..."
        kill "$pid" 2>/dev/null || true
        # Wait up to 10 seconds for graceful shutdown
        for i in $(seq 1 10); do
            if ! kill -0 "$pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        # Force kill if still running
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
        echo "OpenWebUI stopped."
    else
        echo "OpenWebUI is not running."
    fi
}

# Handle actions
case $ACTION in
    stop)
        stop_openwebui
        exit 0
        ;;
    logs)
        if [[ -f "$LOG_FILE" ]]; then
            tail -f "$LOG_FILE"
        else
            echo "No log file found at: $LOG_FILE"
            echo "Is OpenWebUI running?"
        fi
        exit 0
        ;;
    status)
        if is_running; then
            local_pid=$(cat "$PID_FILE")
            echo "OpenWebUI is running (PID: $local_pid)"
            echo "Access: http://localhost:$WEBUI_PORT"
            # Check if port is responding
            if curl -s -o /dev/null -w "" "http://localhost:$WEBUI_PORT/health" 2>/dev/null; then
                echo "Health: OK"
            else
                echo "Health: not responding (may still be starting)"
            fi
        else
            echo "OpenWebUI is not running."
        fi
        exit 0
        ;;
    restart)
        stop_openwebui
        sleep 2
        ;;
esac

# Check if already running
if is_running; then
    echo "OpenWebUI is already running (PID: $(cat "$PID_FILE"))."
    echo "Use --restart to restart, or --stop to stop."
    echo ""
    echo "Access: http://localhost:$WEBUI_PORT"
    exit 0
fi

# Pull image if not present or rebuild requested
if [[ ! -f "$SIF_FILE" ]] || [[ "$REBUILD_IMAGE" == "true" ]]; then
    echo "=============================================="
    echo "Pulling OpenWebUI image..."
    echo "This is a one-time download (~1.5GB)"
    echo "=============================================="
    $CONTAINER_CMD pull --force "$SIF_FILE" "$IMAGE_URL"
    echo "Image saved to: $SIF_FILE"
    echo ""
fi

# Validate mode
if [[ "$RAG_MODE" != "advanced-rag" && "$RAG_MODE" != "builtin-rag" ]]; then
    echo "Error: Unknown mode '$RAG_MODE'. Use 'advanced-rag' or 'builtin-rag'."
    exit 1
fi

# Configure based on mode
if [[ "$RAG_MODE" == "advanced-rag" ]]; then
    # advanced-rag: OpenWebUI → RAG Server → llama.cpp
    # RAG server handles retrieval, so we disable OpenWebUI's built-in RAG
    OPENAI_BASE_URL="http://localhost:$RAG_PORT/v1"
    MODE_LABEL="Advanced RAG (external server on port $RAG_PORT)"
else
    # builtin-rag: OpenWebUI → llama.cpp directly (OpenWebUI does RAG)
    OPENAI_BASE_URL="http://localhost:$LLAMA_PORT/v1"
    MODE_LABEL="Built-in RAG (OpenWebUI handles retrieval)"
fi

echo "=============================================="
echo "Starting OpenWebUI ($CONTAINER_CMD)"
echo "=============================================="
echo "Mode:             $MODE_LABEL"
echo "WebUI Port:       $WEBUI_PORT"
if [[ "$RAG_MODE" == "advanced-rag" ]]; then
echo "RAG Server Port:  $RAG_PORT"
fi
echo "llama.cpp Port:   $LLAMA_PORT"
echo "API Base URL:     $OPENAI_BASE_URL"
echo "Data Directory:   $DATA_DIR"
echo "SIF Image:        $SIF_FILE"
if [[ "$RAG_MODE" == "builtin-rag" ]]; then
echo "Embedding Model:  $EMBEDDING_MODEL"
echo "Reranking Model:  $RERANKING_MODEL"
echo "Hybrid Search:    enabled"
echo "RAG Top-K:        $RAG_TOP_K"
fi
echo "Log File:         $LOG_FILE"
echo "=============================================="

# Start OpenWebUI using the container's built-in start.sh
# Apptainer uses host networking by default, so:
# - No port mapping needed (the app listens directly on host ports)
# - llama.cpp server is reachable at localhost
# Note: We use --pwd to set the working directory since start.sh expects to be in /app/backend

# Build environment variable arguments based on mode
ENV_ARGS=(
    --env "PORT=$WEBUI_PORT"
    --env "HOST=0.0.0.0"
    --env "OPENAI_API_BASE_URL=$OPENAI_BASE_URL"
    --env "OPENAI_API_KEY=not-needed"
    --env "ENABLE_RAG_WEB_SEARCH=false"
    --env "DEFAULT_MODELS=llama.cpp"
    --env "ANONYMIZED_TELEMETRY=false"
    --env "SCARF_NO_ANALYTICS=true"
    --env "DO_NOT_TRACK=true"
    --env "ENABLE_OLLAMA_API=false"
)

if [[ "$RAG_MODE" == "builtin-rag" ]]; then
    # Built-in RAG: configure OpenWebUI's RAG settings
    ENV_ARGS+=(
        --env "RAG_EMBEDDING_MODEL=$EMBEDDING_MODEL"
        --env "RAG_EMBEDDING_ENGINE="
        --env "RAG_RERANKING_MODEL=$RERANKING_MODEL"
        --env "RAG_RERANKING_ENGINE="
        --env "RAG_CHUNK_SIZE=$CHUNK_SIZE"
        --env "RAG_CHUNK_OVERLAP=$CHUNK_OVERLAP"
        --env "RAG_TOP_K=$RAG_TOP_K"
        --env "RAG_RELEVANCE_THRESHOLD=$RAG_RELEVANCE_THRESHOLD"
        --env "ENABLE_RAG_HYBRID_SEARCH=true"
    )
fi
# Note: In advanced-rag mode we omit OpenWebUI RAG env vars since our
# external RAG server handles retrieval, embedding, and reranking.

nohup $CONTAINER_CMD run \
    --pwd /app/backend \
    --bind "$DATA_DIR:/app/backend/data" \
    "${ENV_ARGS[@]}" \
    --writable-tmpfs \
    "$SIF_FILE" \
    > "$LOG_FILE" 2>&1 &

# Save PID for management
echo $! > "$PID_FILE"

echo ""
echo "=============================================="
echo "OpenWebUI Starting! (PID: $(cat "$PID_FILE"))"
echo "=============================================="
echo ""
echo "Waiting for OpenWebUI to be ready..."

# Wait for the server to become healthy (up to 120 seconds)
READY=false
for i in $(seq 1 120); do
    if ! is_running; then
        echo ""
        echo "Error: OpenWebUI process exited unexpectedly."
        echo "Check logs: cat $LOG_FILE"
        exit 1
    fi
    if curl -s -o /dev/null -w "" "http://localhost:$WEBUI_PORT/health" 2>/dev/null; then
        READY=true
        break
    fi
    printf "."
    sleep 1
done

echo ""
if [[ "$READY" == "true" ]]; then
    echo ""
    echo "=============================================="
    echo "OpenWebUI is ready!"
    echo "=============================================="
    echo ""
    echo "Mode:             $MODE_LABEL"
    echo "Access:           http://localhost:$WEBUI_PORT"
    echo ""
    echo "First-time setup:"
    echo "  1. Create an admin account at http://localhost:$WEBUI_PORT"
    echo "  2. Go to Admin Panel > Settings > Connections"
    echo "  3. Verify OpenAI API connection"
    echo ""
    if [[ "$RAG_MODE" == "advanced-rag" ]]; then
    echo "Advanced RAG Mode:"
    echo "  - Retrieval handled by external RAG server (port $RAG_PORT)"
    echo "  - Hybrid search: FAISS + BM25 with RRF fusion"
    echo "  - Cross-encoder reranking for precision"
    echo "  - Automatic data source routing (AIT/LKR)"
    echo "  - Citations appended to responses"
    echo "  - No document upload needed (pre-indexed)"
    echo ""
    echo "  To switch to built-in RAG:"
    echo "    ./scripts/setup_openwebui.sh --restart --mode builtin-rag"
    else
    echo "Built-in RAG Configuration (pre-configured via env vars):"
    echo "  - Chunk Size: $CHUNK_SIZE"
    echo "  - Chunk Overlap: $CHUNK_OVERLAP"
    echo "  - Hybrid Search: enabled (BM25 + Embedding)"
    echo "  - Reranking: $RERANKING_MODEL"
    echo "  - Top-K: $RAG_TOP_K"
    echo "  - Web Search: disabled (data privacy)"
    echo ""
    echo "To upload QM documents:"
    echo "  1. Go to Workspace > Knowledge"
    echo "  2. Create a collection (e.g. 'QM-Dokumente')"
    echo "  3. Upload your QM PDF/MD files"
    echo "  4. In chat, use # to reference a document or collection"
    echo "  5. Citations appear as clickable references below responses"
    echo ""
    echo "  To switch to advanced RAG:"
    echo "    ./scripts/setup_openwebui.sh --restart --mode advanced-rag"
    fi
else
    echo "Warning: OpenWebUI did not respond within 120 seconds."
    echo "It may still be starting. Check logs:"
    echo "  cat $LOG_FILE"
fi
echo ""
echo "Commands:"
echo "  View logs:    ./scripts/setup_openwebui.sh --logs"
echo "  Stop:         ./scripts/setup_openwebui.sh --stop"
echo "  Status:       ./scripts/setup_openwebui.sh --status"
echo "  Restart:      ./scripts/setup_openwebui.sh --restart"
echo "  Rebuild:      ./scripts/setup_openwebui.sh --rebuild"
echo "=============================================="
