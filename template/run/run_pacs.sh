#!/bin/bash
#SBATCH --job-name=dp-camelyon
#SBATCH --partition=gpu-7d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/job-%j.out

#SBATCH --array=1-3

# 1. copy the squashed dataset to the nodes /tmp 
rsync -ah --progress /home/myasincifci/dispatch_smol/data/PACS.hdf5 /tmp

apptainer run --nv -B /tmp:/data /home/myasincifci/containers/main/main.sif \
    python template/train.py \
        --config-name pacs-mixstyle-dann