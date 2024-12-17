#!/bin/bash
#SBATCH --job-name=dr-sup
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/job-%j.out
#SBATCH --exclude=head075

#SBATCH --array=1-5

# 1. copy the squashed dataset to the nodes /tmp 
rsync -ah --progress /home/myasincifci/data/DR.sqfs /tmp

apptainer run --nv -B /tmp/DR.sqfs:/data/DR:image-src=/ /home/myasincifci/containers/main/main.sif \
    python template/train.py \
        --config-name dr-no-no-color