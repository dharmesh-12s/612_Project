#!/bin/bash
# Submit every experiment. Usage: bash slurm/submit_all.sh [seeds...]   (default: seed 0)
# Example for the extra seeds of the full model:  SEEDS="1 2" CONFIGS="configs/st5_quantile.yaml" bash slurm/submit_all.sh
cd ~/scratch.msml612/612_Project
mkdir -p logs
SEEDS=${SEEDS:-0}
CONFIGS=${CONFIGS:-$(ls configs/*.yaml)}
for c in $CONFIGS; do
  for s in $SEEDS; do
    name=$(basename "$c" .yaml)
    sbatch --job-name="$name-s$s" --export=ALL,CONFIG="$c",SEED="$s" slurm/train.sbatch
  done
done
squeue -u "$USER"
