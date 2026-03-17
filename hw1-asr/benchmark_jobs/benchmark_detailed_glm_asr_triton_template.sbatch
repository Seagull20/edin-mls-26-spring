#!/bin/bash
#SBATCH --job-name=glm-asr-benchmark-detailed
#SBATCH --partition=Teaching
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/s2866920/edin-mls-26-spring/hw1-asr/benchmark_jobs/logs/%x-%j.out
#SBATCH --error=/home/s2866920/edin-mls-26-spring/hw1-asr/benchmark_jobs/logs/%x-%j.err

set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mls

cd /home/s2866920/edin-mls-26-spring/hw1-asr
./benchmark_detailed.sh glm_asr_triton_template
