#!/bin/bash
#SBATCH --job-name=refactor-qkv-test
#SBATCH --partition=Teaching
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/s2866920/edin-mls-26-spring/hw1-asr/benchmark_jobs/logs/%x-%j.out
#SBATCH --error=/home/s2866920/edin-mls-26-spring/hw1-asr/benchmark_jobs/logs/%x-%j.err

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mls
export PYTHONUNBUFFERED=1

cd /home/s2866920/edin-mls-26-spring/hw1-asr

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Branch: $(git branch --show-current)"
echo ""

python benchmark_jobs/test_refactor_correctness.py
