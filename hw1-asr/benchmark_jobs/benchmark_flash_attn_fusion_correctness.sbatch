#!/bin/bash
#SBATCH --job-name=flash-attn-fusion-correctness
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa,crannog01,crannog02
#SBATCH --nodes=1
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/s2866920/edin-mls-26-spring/hw1-asr/benchmark_jobs/logs/%x-%j.out
#SBATCH --error=/home/s2866920/edin-mls-26-spring/hw1-asr/benchmark_jobs/logs/%x-%j.err

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mls
export PYTHONUNBUFFERED=1

cd /home/s2866920/edin-mls-26-spring/hw1-asr

echo "===== Test 1: Flash Attention Fusion ENABLED ====="
FLASH_ATTN_FUSION=1 ./benchmark.sh glm_asr_triton_template

echo ""
echo "===== Test 2: Flash Attention Fusion DISABLED (baseline) ====="
FLASH_ATTN_FUSION=0 ./benchmark.sh glm_asr_triton_template

echo ""
echo "===== Test 3: Triton Example (reference baseline) ====="
./benchmark.sh glm_asr_triton_example
