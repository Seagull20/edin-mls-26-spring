#!/bin/bash
#SBATCH --job-name=rmsnorm-qkv-e2e-compare
#SBATCH --partition=Teaching
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/s2866920/edin-mls-26-spring/hw1-asr/benchmark_jobs/logs/%x-%j.out
#SBATCH --error=/home/s2866920/edin-mls-26-spring/hw1-asr/benchmark_jobs/logs/%x-%j.err

# Usage:
#   sbatch -w saxa benchmark_rmsnorm_qkv_e2e_compare.sbatch
#   RUNS=5 TOTAL_ROUNDS=3 sbatch -w saxa benchmark_rmsnorm_qkv_e2e_compare.sbatch

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mls
export PYTHONUNBUFFERED=1

RUNS="${RUNS:-5}"
TOTAL_ROUNDS="${TOTAL_ROUNDS:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$SCRIPT_DIR/logs/rmsnorm_qkv_e2e_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "DecoderRMSNormQKV End-to-End Comparison"
echo "============================================"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Rounds: $TOTAL_ROUNDS, Profile runs/round: $RUNS"
echo "Log directory: $LOG_DIR"
echo ""

for round in $(seq 1 "$TOTAL_ROUNDS"); do
    echo "=== Round $round / $TOTAL_ROUNDS ==="

    # 1. Baseline (unfused - triton_example)
    echo "--- Baseline (triton_example, QKV unfused) ---"
    python "$SCRIPT_DIR/benchmark_fusion_detailed.py" \
        glm_asr_triton_example \
        --runs "$RUNS" \
        --json-out "$LOG_DIR/baseline_round${round}.json" \
        | tee "$LOG_DIR/baseline_round${round}.out"

    # 2. Candidate (fused - triton_template with DecoderRMSNormQKV.FUSED=True)
    echo "--- Candidate (triton_template, QKV fused) ---"
    python "$SCRIPT_DIR/benchmark_fusion_detailed.py" \
        glm_asr_triton_template \
        --runs "$RUNS" \
        --json-out "$LOG_DIR/fused_round${round}.json" \
        | tee "$LOG_DIR/fused_round${round}.out"

    # 3. Compare
    echo "--- Comparison ---"
    python "$SCRIPT_DIR/analyze_fusion_benchmark.py" \
        --baseline "$LOG_DIR/baseline_round${round}.json" \
        --candidate "$LOG_DIR/fused_round${round}.json" \
        | tee "$LOG_DIR/comparison_round${round}.out"

    echo ""
done

echo "All rounds complete. Artifacts in: $LOG_DIR"
