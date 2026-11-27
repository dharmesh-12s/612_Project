#!/bin/bash
# Submit experiments.  Usage examples (from the repo root on Zaratan):
#   bash slurm/submit_all.sh                                   # every config, seed 0, 5 GB A100 slice
#   SEEDS="1 2" CONFIGS="configs/st5_quantile.yaml" bash slurm/submit_all.sh
#   SBATCH_ARGS="-p scavenger --gres=gpu:1" CONFIGS="configs/st1_plain.yaml" bash slurm/submit_all.sh
cd ~/scratch.msml612/612_Project
mkdir -p logs
SEEDS=${SEEDS:-0}
CONFIGS=${CONFIGS:-$(ls configs/*.yaml)}
for c in $CONFIGS; do
  for s in $SEEDS; do
    name=$(basename "$c" .yaml)
    sbatch $SBATCH_ARGS --job-name="$name-s$s" --export=ALL,CONFIG="$c",SEED="$s" slurm/train.sbatch
  done
done
squeue -u "$USER"
