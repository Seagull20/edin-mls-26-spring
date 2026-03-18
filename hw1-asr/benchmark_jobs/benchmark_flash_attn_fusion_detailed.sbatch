#!/bin/bash
#SBATCH --job-name=flash-attn-fusion-detailed
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

RUNS=5

echo "============================================================"
echo "  DETAILED BENCHMARK: Flash Attention Fusion ENABLED"
echo "============================================================"
FLASH_ATTN_FUSION=1 ./benchmark_detailed.sh glm_asr_triton_template --runs $RUNS

echo ""
echo "============================================================"
echo "  DETAILED BENCHMARK: Flash Attention Fusion DISABLED"
echo "============================================================"
FLASH_ATTN_FUSION=0 ./benchmark_detailed.sh glm_asr_triton_template --runs $RUNS

echo ""
echo "============================================================"
echo "  DETAILED BENCHMARK: Triton Example (reference baseline)"
echo "============================================================"
./benchmark_detailed.sh glm_asr_triton_example --runs $RUNS
