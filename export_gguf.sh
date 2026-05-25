#!/usr/bin/env bash
# export_gguf.sh
#
# Merge LoRA adapter into Gemma 4 MoE base, export to Q4_K_M GGUF.
#
# Usage:
#   bash export_gguf.sh --adapter outputs/sft/checkpoint-2000 \
#                       --output  outputs/gguf/gemma4-moe-sft
#
# Requirements:
#   - ~55 GB VRAM or system RAM (bf16 load)
#   - git, cmake, python3 in PATH
#   - .venv activated or VENV_PYTHON set

set -euo pipefail

ADAPTER=""
OUTPUT=""
LLAMA_CPP_DIR="$(dirname "$0")/llama.cpp"
VENV_PYTHON="${VENV_PYTHON:-$(dirname "$0")/.venv/bin/python}"

usage() {
    echo "Usage: $0 --adapter <checkpoint_dir> --output <output_prefix>"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --adapter) ADAPTER="$2"; shift 2 ;;
        --output)  OUTPUT="$2";  shift 2 ;;
        *) usage ;;
    esac
done

[[ -z "$ADAPTER" || -z "$OUTPUT" ]] && usage

MERGED_DIR="${OUTPUT}_merged_bf16"
GGUF_F16="${OUTPUT}_f16.gguf"
GGUF_Q4="${OUTPUT}_q4km.gguf"

# ---------------------------------------------------------------------------
# Step 1: merge LoRA + save bf16 HF model
# ---------------------------------------------------------------------------
echo "=== Step 1: merge LoRA adapter into bf16 base model ==="
"$VENV_PYTHON" "$(dirname "$0")/export_merged.py" \
    --adapter "$ADAPTER" \
    --output  "$MERGED_DIR"

# ---------------------------------------------------------------------------
# Step 2: build llama.cpp if not already present
# ---------------------------------------------------------------------------
echo "=== Step 2: build llama.cpp ==="
if [[ ! -f "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]]; then
    if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
        git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
    fi
    cmake -B "$LLAMA_CPP_DIR/build" -DGGML_CUDA=ON "$LLAMA_CPP_DIR"
    cmake --build "$LLAMA_CPP_DIR/build" --config Release -j "$(nproc)"
fi

# ---------------------------------------------------------------------------
# Step 3: convert to f16 GGUF
# ---------------------------------------------------------------------------
echo "=== Step 3: convert to f16 GGUF ==="
"$VENV_PYTHON" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$MERGED_DIR" \
    --outfile "$GGUF_F16" \
    --outtype f16

# ---------------------------------------------------------------------------
# Step 4: quantize to Q4_K_M
# ---------------------------------------------------------------------------
echo "=== Step 4: quantize to Q4_K_M ==="
"$LLAMA_CPP_DIR/build/bin/llama-quantize" "$GGUF_F16" "$GGUF_Q4" Q4_K_M

echo ""
echo "Done. Pull this file to your local server:"
echo "  $GGUF_Q4"
echo ""
echo "Run locally with:"
echo "  ./llama-cli -m $GGUF_Q4 -n 512 --n-gpu-layers 99 -p \"<your prompt>\""
