#!/bin/bash
#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=test_internvla_grounding
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/test_internvla_grounding_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/test_internvla_grounding_%j.err
#SBATCH --time=01:30:00

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/InternVLA-M1"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"
INTERNVLA_ROOT="/home/A.CARDAMONE7/repo/InternVLA-M1"

# ── Environment ───────────────────────────────────────────────────────────────
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=${LIBERO_PATH}:${INTERNVLA_ROOT}:${WORK_DIR}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS##*:}
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate internvla-m1

echo "=========================================="
echo "InternVLA-M1 with Grounding Integration"
echo "=========================================="
echo "Job ID:         $SLURM_JOB_ID"
echo "GPU:            $CUDA_VISIBLE_DEVICES"
echo "Start:          $(date)"
echo "=========================================="

cd ${WORK_DIR}

srun python test_internvla_with_grounding.py

echo "=========================================="
echo "Finish: $(date)"
echo "=========================================="
