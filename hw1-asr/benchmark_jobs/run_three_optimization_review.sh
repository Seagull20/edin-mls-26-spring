#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS="${RUNS:-5}"
WARMUP="${WARMUP:-1}"
RUN_DETAILED="${RUN_DETAILED:-1}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/logs/three_opt_review_$TIMESTAMP"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

run_case() {
    local mode="$1"
    local name="$2"
    local folder="$3"
    shift 3

    echo ""
    echo "============================================================"
    echo "[$mode] $name"
    echo "============================================================"

    "$PYTHON_BIN" "$SCRIPT_DIR/benchmark_with_flags.py" \
        "$folder" \
        --mode "$mode" \
        --warmup "$WARMUP" \
        --runs "$RUNS" \
        --json-out "$LOG_DIR/${name}_${mode}.json" \
        "$@" | tee "$LOG_DIR/${name}_${mode}.out"
}

compare_case() {
    local mode="$1"
    local baseline_name="$2"
    local candidate_name="$3"

    echo ""
    echo "------------------------------------------------------------"
    echo "Compare [$mode] $candidate_name vs $baseline_name"
    echo "------------------------------------------------------------"

    "$PYTHON_BIN" "$SCRIPT_DIR/compare_benchmark_json.py" \
        --baseline "$LOG_DIR/${baseline_name}_${mode}.json" \
        --candidate "$LOG_DIR/${candidate_name}_${mode}.json" \
        | tee "$LOG_DIR/compare_${candidate_name}_vs_${baseline_name}_${mode}.out"
}

echo "Three-optimization benchmark artifacts will be written to: $LOG_DIR"
echo "Warmup: $WARMUP, Runs: $RUNS, Run detailed: $RUN_DETAILED"

run_case e2e baseline_example glm_asr_triton_example \
    --linear-backend cublas \
    --linear-autotune 0 \
    --flash-attn-fusion 0 \
    --decoder-qkv-fusion 0

run_case e2e block_size_report glm_asr_triton_template \
    --linear-backend triton \
    --linear-autotune 1 \
    --flash-attn-fusion 0 \
    --decoder-qkv-fusion 0

run_case e2e block_size_control glm_asr_triton_template \
    --linear-backend triton \
    --linear-autotune 0 \
    --flash-attn-fusion 0 \
    --decoder-qkv-fusion 0

run_case e2e flash_attention glm_asr_triton_template \
    --linear-backend cublas \
    --linear-autotune 0 \
    --flash-attn-fusion 1 \
    --decoder-qkv-fusion 0

run_case e2e decoder_qkv glm_asr_triton_template \
    --linear-backend cublas \
    --linear-autotune 0 \
    --flash-attn-fusion 0 \
    --decoder-qkv-fusion 1

compare_case e2e baseline_example block_size_report
compare_case e2e block_size_control block_size_report
compare_case e2e baseline_example flash_attention
compare_case e2e baseline_example decoder_qkv

if [[ "$RUN_DETAILED" == "1" ]]; then
    run_case detailed baseline_example glm_asr_triton_example \
        --linear-backend cublas \
        --linear-autotune 0 \
        --flash-attn-fusion 0 \
        --decoder-qkv-fusion 0

    run_case detailed block_size_report glm_asr_triton_template \
        --linear-backend triton \
        --linear-autotune 1 \
        --flash-attn-fusion 0 \
        --decoder-qkv-fusion 0

    run_case detailed block_size_control glm_asr_triton_template \
        --linear-backend triton \
        --linear-autotune 0 \
        --flash-attn-fusion 0 \
        --decoder-qkv-fusion 0

    run_case detailed flash_attention glm_asr_triton_template \
        --linear-backend cublas \
        --linear-autotune 0 \
        --flash-attn-fusion 1 \
        --decoder-qkv-fusion 0

    run_case detailed decoder_qkv glm_asr_triton_template \
        --linear-backend cublas \
        --linear-autotune 0 \
        --flash-attn-fusion 0 \
        --decoder-qkv-fusion 1

    compare_case detailed baseline_example block_size_report
    compare_case detailed block_size_control block_size_report
    compare_case detailed baseline_example flash_attention
    compare_case detailed baseline_example decoder_qkv
fi

echo ""
echo "All benchmark artifacts written to: $LOG_DIR"
