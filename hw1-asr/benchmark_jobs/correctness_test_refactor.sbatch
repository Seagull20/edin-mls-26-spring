#!/bin/bash
#SBATCH --job-name=refactor-qkv-correctness
#SBATCH --partition=Teaching
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/s2866920/edin-mls-26-spring/hw1-asr/benchmark_jobs/logs/%x-%j.out
#SBATCH --error=/home/s2866920/edin-mls-26-spring/hw1-asr/benchmark_jobs/logs/%x-%j.err

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mls
export PYTHONUNBUFFERED=1

cd /home/s2866920/edin-mls-26-spring/hw1-asr

echo "============================================"
echo "DecoderRMSNormQKV Refactor — Correctness Test"
echo "============================================"
echo "Node: $(hostname)"
echo "Branch: $(git branch --show-current)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "=== Test 1: End-to-end ASR correctness (fused=True, default) ==="
./benchmark.sh glm_asr_triton_template --runs 3
echo ""

echo "=== Test 2: Baseline comparison (triton_example, unfused) ==="
./benchmark.sh glm_asr_triton_example --runs 3
echo ""

echo "=== Test 3: Microbenchmark correctness ==="
python benchmark_jobs/microbench_decoder_rmsnorm_qkv.py --warmup 10 --runs 50
echo ""

echo "All correctness tests complete."
