#!/bin/bash
#SBATCH --job-name=rmsnorm-qkv-microbench
#SBATCH --partition=Teaching
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/s2866920/edin-mls-26-spring/hw1-asr/benchmark_jobs/logs/%x-%j.out
#SBATCH --error=/home/s2866920/edin-mls-26-spring/hw1-asr/benchmark_jobs/logs/%x-%j.err

# Usage:
#   sbatch -w saxa benchmark_rmsnorm_qkv_microbench.sbatch
#   sbatch -w crannog01 benchmark_rmsnorm_qkv_microbench.sbatch
#   RUNS=300 WARMUP=100 sbatch -w saxa benchmark_rmsnorm_qkv_microbench.sbatch

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mls
export PYTHONUNBUFFERED=1

RUNS="${RUNS:-200}"
WARMUP="${WARMUP:-50}"
TOTAL_ROUNDS="${TOTAL_ROUNDS:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$SCRIPT_DIR/logs/rmsnorm_qkv_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "DecoderRMSNormQKV Micro-Benchmark"
echo "============================================"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Rounds: $TOTAL_ROUNDS, Runs/round: $RUNS, Warmup: $WARMUP"
echo "Log directory: $LOG_DIR"
echo ""

for round in $(seq 1 "$TOTAL_ROUNDS"); do
    echo "=== Round $round / $TOTAL_ROUNDS ==="
    python "$SCRIPT_DIR/microbench_decoder_rmsnorm_qkv.py" \
        --warmup "$WARMUP" \
        --runs "$RUNS" \
        --json-out "$LOG_DIR/round_${round}.json" \
        | tee "$LOG_DIR/round_${round}.out"
    echo ""
done

echo "All rounds complete. Artifacts in: $LOG_DIR"
