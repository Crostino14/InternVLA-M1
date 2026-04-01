#!/bin/bash
#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=test_qwen_internvla_checkpoint
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --partition=gpuq
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/test_qwen_internvla_checkpoint_%j.out

cd /home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/InternVLA-M1

# Setup environment
source /home/A.CARDAMONE7/anaconda3/bin/activate
conda activate internvla-m1

# Run test
python test_qwen_in_internvla_checkpoint.py
