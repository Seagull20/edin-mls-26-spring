#!/bin/bash
#SBATCH --job-name=refactor-e2e-quick
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

cd /home/s2866920/edin-mls-26-spring/hw1-asr

echo "Node: $(hostname), Branch: $(git branch --show-current)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== E2E ASR correctness (fused Plan C) ==="
./benchmark.sh glm_asr_triton_template --runs 3
