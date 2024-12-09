#!/bin/bash
#SBATCH --job-name=pacs-new
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/job-%j.out
#SBATCH --exclude=head075

#SBATCH --array=1-5

# 1. copy the squashed dataset to the nodes /tmp 
rsync -ah --progress /home/myasincifci/dispatch_smol/data/PACS.hdf5 /tmp

apptainer run --nv -B /tmp:/data /home/myasincifci/containers/main/main.sif \
    python template/train.py \
        --config-name pacs-no-no-none