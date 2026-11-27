#!/bin/bash
# Train one config on one A100 on Zaratan.
#   sbatch --export=ALL,CONFIG=configs/st5_quantile.yaml,SEED=0 slurm/train.sbatch
#SBATCH --job-name=bike
#SBATCH -A msml612-class
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out

module load python
cd ~/scratch.msml612/612_Project
source ~/scratch.msml612/venvs/bike/bin/activate      # its activate script also unsets PYTHONPATH

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "CONFIG=$CONFIG SEED=${SEED:-0} JOB=$SLURM_JOB_ID"
python -m src.train --config "$CONFIG" --seed "${SEED:-0}" --max-hours 3.3
