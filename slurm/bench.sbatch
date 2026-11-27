#!/bin/bash
# Time every config for a few steps and report peak GPU memory (about 15 minutes, ~2 SU).
#   sbatch slurm/bench.sbatch            then read logs/bench-<jobid>.out
#   CONFIGS="configs/lstm.yaml configs/st1_plain.yaml" sbatch --export=ALL slurm/bench.sbatch
#SBATCH --job-name=bench
#SBATCH -A msml612-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out

module load python
cd ~/scratch.msml612/612_Project
source ~/scratch.msml612/venvs/bike/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv
for c in ${CONFIGS:-configs/*.yaml}; do
  python -m src.train --config "$c" --bench 2>&1 | grep -E "BENCH|OutOfMemory|Error" | tail -1 \
    || echo "FAILED $c"
done
