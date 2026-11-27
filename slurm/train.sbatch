#!/bin/bash
# Train one config on Zaratan. Default GPU: a 1/7 A100 slice (5 GB, 7 SU/hour).
#   sbatch --export=ALL,CONFIG=configs/st5_quantile.yaml,SEED=0 slurm/train.sbatch
# Free but preemptible (the run resumes from state.pt when requeued):
#   sbatch -p scavenger --gres=gpu:1 --export=ALL,CONFIG=...,SEED=0 slurm/train.sbatch
#SBATCH --job-name=bike
#SBATCH -A msml612-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 06:00:00
#SBATCH --requeue
#SBATCH -o logs/%x-%j.out

module load python
cd ~/scratch.msml612/612_Project
source ~/scratch.msml612/venvs/bike/bin/activate      # its activate script also unsets PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "CONFIG=$CONFIG SEED=${SEED:-0} JOB=$SLURM_JOB_ID RESTART=${SLURM_RESTART_COUNT:-0}"
python -m src.train --config "$CONFIG" --seed "${SEED:-0}" --max-hours ${MAX_HOURS:-5}
