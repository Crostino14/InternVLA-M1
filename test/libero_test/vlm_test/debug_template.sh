#!/bin/bash
#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=debug_template
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/debug_template_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/debug_template_%j.err

LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"
INTERNVLA_ROOT="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/InternVLA-M1"
WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/InternVLA-M1/libero_test"

export PYTHONPATH=${INTERNVLA_ROOT}:${LIBERO_PATH}:${WORK_DIR}:$PYTHONPATH
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export TOKENIZERS_PARALLELISM=false

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate internvla-m1

python /home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/InternVLA-M1/test/debug_template.py