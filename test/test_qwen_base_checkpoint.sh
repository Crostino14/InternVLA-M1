#!/bin/bash
 #SBATCH --account=did_robot_learning_359
#SBATCH --job-name=test_qwen_base
#SBATCH --partition=gpuq
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --qos=high
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/test_qwen_base_checkpoint_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/test_qwen_base_checkpoint_%j.err

cd /home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/InternVLA-M1

# Activate conda environment
source /home/A.CARDAMONE7/anaconda3/etc/profile.d/conda.sh
conda activate internvla-m1

# Run the test
python test_qwen_base_checkpoint.py
