#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=internvla-test
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00

# Ambiente
source ~/anaconda3/etc/profile.d/conda.sh
conda activate internvla-m1

# Path
cd ~/repo/VLA-Bench/robosuite_test/InternVLA-M1/test

export MODEL_PATH="/home/A.CARDAMONE7/checkpoints/InternVLA-M1-LIBERO-Goal/checkpoints/steps_30000_pytorch_model.pt"

python test.py