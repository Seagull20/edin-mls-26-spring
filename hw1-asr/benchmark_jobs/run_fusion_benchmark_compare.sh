#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNS="${RUNS:-3}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/logs/fusion_compare_$TIMESTAMP"

mkdir -p "$LOG_DIR"

echo "Fusion benchmark artifacts will be written to: $LOG_DIR"

python "$SCRIPT_DIR/benchmark_fusion_detailed.py" \
    glm_asr_triton_example \
    --runs "$RUNS" \
    --json-out "$LOG_DIR/triton_example.json" \
    | tee "$LOG_DIR/triton_example.out"

python "$SCRIPT_DIR/benchmark_fusion_detailed.py" \
    glm_asr_triton_template \
    --runs "$RUNS" \
    --json-out "$LOG_DIR/triton_template_fused.json" \
    | tee "$LOG_DIR/triton_template_fused.out"

python "$SCRIPT_DIR/analyze_fusion_benchmark.py" \
    --baseline "$LOG_DIR/triton_example.json" \
    --candidate "$LOG_DIR/triton_template_fused.json" \
    | tee "$LOG_DIR/comparison.out"

echo "Comparison complete."
echo "Artifacts: $LOG_DIR"
