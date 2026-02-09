#!/bin/bash
# =============================================================================
# Merge LoRA Adapter and Convert to GGUF
# =============================================================================
#
# Complete pipeline to convert a trained LoRA adapter to a deployable GGUF model:
#   1. Merge LoRA adapter with base model
#   2. Convert merged model to GGUF format
#   3. Quantize to specified precision
#
# Usage:
#   ./scripts/merge_and_convert.sh
#   ./scripts/merge_and_convert.sh --adapter checkpoints/QM_rag/checkpoint-500
#   ./scripts/merge_and_convert.sh --quantize q8_0
#
# =============================================================================

set -e

# Ensure conda libraries are available (llama.cpp was built against conda's libstdc++)
if [[ -n "$CONDA_PREFIX" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Default configuration
ADAPTER_PATH="${ADAPTER_PATH:-checkpoints/QM_rag/checkpoint-1000}"
BASE_MODEL="${BASE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"
MERGED_OUTPUT="${MERGED_OUTPUT:-checkpoints/QM_rag/merged}"
GGUF_OUTPUT="${GGUF_OUTPUT:-checkpoints/QM_rag/gguf}"
QUANTIZE="${QUANTIZE:-q4_k_m}"
DTYPE="${DTYPE:-float16}"
SKIP_MERGE="${SKIP_MERGE:-false}"
SKIP_CONVERT="${SKIP_CONVERT:-false}"
USE_CPU="${USE_CPU:-false}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --adapter|-a)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --base-model|-b)
            BASE_MODEL="$2"
            shift 2
            ;;
        --merged-output|-m)
            MERGED_OUTPUT="$2"
            shift 2
            ;;
        --gguf-output|-g)
            GGUF_OUTPUT="$2"
            shift 2
            ;;
        --quantize|-q)
            QUANTIZE="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --skip-merge)
            SKIP_MERGE="true"
            shift
            ;;
        --skip-convert)
            SKIP_CONVERT="true"
            shift
            ;;
        --cpu)
            USE_CPU="true"
            shift
            ;;
        --list-quantizations)
            python -m src.inference.convert_to_gguf --list_quantizations
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --adapter, -a PATH      Path to LoRA adapter (default: checkpoints/QM_rag/checkpoint-1000)"
            echo "  --base-model, -b MODEL  Base model name (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)"
            echo "  --merged-output, -m DIR Merged model output (default: checkpoints/QM_rag/merged)"
            echo "  --gguf-output, -g DIR   GGUF output directory (default: checkpoints/QM_rag/gguf)"
            echo "  --quantize, -q TYPE     Quantization type (default: q4_k_m)"
            echo "  --dtype TYPE            Model dtype (default: float16)"
            echo "  --skip-merge            Skip merge step (use existing merged model)"
            echo "  --skip-convert          Skip conversion step (merge only)"
            echo "  --cpu                   Force CPU-only merging (avoids CUDA errors)"
            echo "  --list-quantizations    List available quantization options"
            echo "  --help, -h              Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "LoRA Merge and GGUF Conversion Pipeline"
echo "=============================================="
echo "Adapter Path:    $ADAPTER_PATH"
echo "Base Model:      $BASE_MODEL"
echo "Merged Output:   $MERGED_OUTPUT"
echo "GGUF Output:     $GGUF_OUTPUT"
echo "Quantization:    $QUANTIZE"
echo "Data Type:       $DTYPE"
echo "=============================================="
echo ""

# Check adapter exists
if [[ "$SKIP_MERGE" != "true" ]] && [[ ! -d "$ADAPTER_PATH" ]]; then
    echo "Error: Adapter not found at: $ADAPTER_PATH"
    exit 1
fi

# Step 1: Merge adapter
if [[ "$SKIP_MERGE" != "true" ]]; then
    echo ""
    echo "=============================================="
    echo "Step 1: Merging LoRA Adapter"
    echo "=============================================="

    # Build device_map argument
    DEVICE_ARG="--device_map auto"
    if [[ "$USE_CPU" == "true" ]]; then
        DEVICE_ARG="--device_map cpu"
        echo "[CPU MODE] Using CPU for merging (this may take 10-30 minutes)"
    fi

    python -m src.inference.merge_adapter \
        --adapter_path "$ADAPTER_PATH" \
        --output_path "$MERGED_OUTPUT" \
        --base_model "$BASE_MODEL" \
        --dtype "$DTYPE" \
        $DEVICE_ARG

    echo "Merge complete!"
else
    echo "Skipping merge step (--skip-merge)"
fi

# Step 2: Convert to GGUF
if [[ "$SKIP_CONVERT" != "true" ]]; then
    echo ""
    echo "=============================================="
    echo "Step 2: Converting to GGUF"
    echo "=============================================="

    # Check merged model exists
    if [[ ! -d "$MERGED_OUTPUT" ]]; then
        echo "Error: Merged model not found at: $MERGED_OUTPUT"
        echo "Run without --skip-merge first."
        exit 1
    fi

    python -m src.inference.convert_to_gguf \
        --model_path "$MERGED_OUTPUT" \
        --output_path "$GGUF_OUTPUT" \
        --quantize "$QUANTIZE"

    echo "Conversion complete!"
else
    echo "Skipping conversion step (--skip-convert)"
fi

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Output files:"
if [[ "$SKIP_MERGE" != "true" ]]; then
    echo "  Merged Model: $MERGED_OUTPUT"
fi
if [[ "$SKIP_CONVERT" != "true" ]]; then
    MODEL_NAME=$(basename "$MERGED_OUTPUT")
    echo "  GGUF (F16):   $GGUF_OUTPUT/${MODEL_NAME}-f16.gguf"
    if [[ "$QUANTIZE" != "none" ]]; then
        echo "  GGUF (Quant): $GGUF_OUTPUT/${MODEL_NAME}-${QUANTIZE}.gguf"
    fi
fi
echo ""
echo "Next steps:"
echo "  1. Start llama.cpp server:"
echo "     ./scripts/start_llama_server.sh --model $GGUF_OUTPUT/${MODEL_NAME:-merged}-${QUANTIZE}.gguf"
echo ""
echo "  2. Start OpenWebUI:"
echo "     ./scripts/setup_openwebui.sh"
echo ""
echo "  3. Or use VanillaRAG directly:"
echo "     python -m src.inference.vanilla_rag --model $MERGED_OUTPUT --interactive"
echo "=============================================="
