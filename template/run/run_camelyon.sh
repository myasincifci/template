#!/bin/bash
#SBATCH --job-name=ca-dann
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=40gb:1
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/job-%j.out

#SBATCH --array=1-5

# 1. copy the squashed dataset to the nodes /tmp 
rsync -ah --progress /home/myasincifci/data/camelyon17_v1.0.sqfs /temp/

apptainer run --nv -B /temp/camelyon17_v1.0.sqfs:/data/camelyon17_v1.0:image-src=/ /home/myasincifci/containers/main/main.sif \
    python template/train.py \
        --config-name camelyon-color-aug-mixstyle