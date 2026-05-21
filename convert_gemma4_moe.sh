#!/usr/bin/env bash
# convert_gemma4_moe.sh
# Pre-quantize Gemma 4 26B MoE expert FFN weights to NF4 and save cache.
#
# Usage:
#   ./convert_gemma4_moe.sh                              # default paths
#   ./convert_gemma4_moe.sh --output-dir /data/moe_cache
#   ./convert_gemma4_moe.sh --model-name google/gemma-4-26b-A4B-it --output-dir /data/moe_cache
#   ./convert_gemma4_moe.sh --verify --output-dir /data/moe_cache

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

if [[ ! -d "$VENV" ]]; then
    echo "ERROR: virtualenv not found at $VENV" >&2
    exit 1
fi

echo "=== Gemma 4 MoE Expert Pre-Quantization ==="
echo "Start: $(date)"

"$VENV/bin/python" "$SCRIPT_DIR/convert_gemma4_moe.py" "$@"

echo "End: $(date)"
