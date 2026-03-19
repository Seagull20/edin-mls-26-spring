#!/bin/bash
#SBATCH --job-name=refactor-qkv-e2e
#SBATCH --partition=Teaching
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/s2866920/edin-mls-26-spring/hw1-asr/benchmark_jobs/logs/%x-%j.out
#SBATCH --error=/home/s2866920/edin-mls-26-spring/hw1-asr/benchmark_jobs/logs/%x-%j.err

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mls
export PYTHONUNBUFFERED=1

RUNS="${RUNS:-5}"
TOTAL_ROUNDS="${TOTAL_ROUNDS:-2}"

SCRIPT_DIR="/home/s2866920/edin-mls-26-spring/hw1-asr/benchmark_jobs"
PROJECT_ROOT="/home/s2866920/edin-mls-26-spring/hw1-asr"
cd "$PROJECT_ROOT"

LOG_DIR="$SCRIPT_DIR/logs/refactor_qkv_e2e_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "Refactored DecoderRMSNormQKV — E2E Comparison"
echo "============================================"
echo "Node: $(hostname)"
echo "Branch: $(git branch --show-current)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Rounds: $TOTAL_ROUNDS, Profile runs/round: $RUNS"
echo ""

for round in $(seq 1 "$TOTAL_ROUNDS"); do
    echo "=== Round $round / $TOTAL_ROUNDS ==="

    echo "--- Baseline (triton_example, unfused) ---"
    python "$SCRIPT_DIR/benchmark_fusion_detailed.py" \
        glm_asr_triton_example \
        --runs "$RUNS" \
        --json-out "$LOG_DIR/baseline_round${round}.json" \
        | tee "$LOG_DIR/baseline_round${round}.out"

    echo "--- Refactored fused (triton_template, Plan C) ---"
    python "$SCRIPT_DIR/benchmark_fusion_detailed.py" \
        glm_asr_triton_template \
        --runs "$RUNS" \
        --json-out "$LOG_DIR/fused_round${round}.json" \
        | tee "$LOG_DIR/fused_round${round}.out"

    echo "--- Comparison ---"
    python "$SCRIPT_DIR/analyze_fusion_benchmark.py" \
        --baseline "$LOG_DIR/baseline_round${round}.json" \
        --candidate "$LOG_DIR/fused_round${round}.json" \
        | tee "$LOG_DIR/comparison_round${round}.out"

    echo ""
done

echo "All rounds complete. Artifacts in: $LOG_DIR"
