#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=extract_emb_internvla
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=07:00:00
#SBATCH --array=0-3
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/extract_emb_internvla_%A_%a.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/extract_emb_internvla_%A_%a.err

MODEL_PATH="/mnt/beegfs/a.cardamone7/checkpoints/InternVLA-M1-LIBERO-Goal/checkpoints/steps_30000_pytorch_model.pt"
WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/InternVLA-M1/libero_test/embeddings_test"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"
INTERNVLA_ROOT="/home/A.CARDAMONE7/repo/InternVLA-M1"
OUTPUT_DIR="/mnt/beegfs/a.cardamone7/outputs/embeddings/internvla/libero_goal_finetuned/"
EMB_ENV_NAME="internvla-m1-embeddings"

TASK_SUITE="libero_goal"
TASK_RANGE="0-9"
NUM_ROLLOUTS=10
FIRST_STEP_ONLY=true
USE_COT=false

COMMAND_LEVELS=("default" "l1" "l2" "l3")
COMMAND_LEVEL=${COMMAND_LEVELS[$SLURM_ARRAY_TASK_ID]}

export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=${LIBERO_PATH}:${INTERNVLA_ROOT}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS##*:}
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export USE_JAX=0
export USE_TORCH=1

source $HOME/anaconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "${EMB_ENV_NAME}"; then
  echo "ERROR: conda env '${EMB_ENV_NAME}' not found."
  echo "Create it first as a clone of 'internvla-m1' with:"
  echo "  conda create -y -n ${EMB_ENV_NAME} --clone internvla-m1"
  exit 1
fi
conda activate ${EMB_ENV_NAME}

cd ${WORK_DIR}

FIRST_STEP_FLAG=""
if [ "$FIRST_STEP_ONLY" = true ]; then
  FIRST_STEP_FLAG="--first_step_only"
fi

USE_COT_FLAG=""
if [ "$USE_COT" = true ]; then
  USE_COT_FLAG="--use_cot"
fi

echo "=========================================="
echo "InternVLA-M1 Text Embedding Extraction"
echo "=========================================="
echo "Job ID:         $SLURM_JOB_ID (array: $SLURM_ARRAY_TASK_ID)"
echo "Model:          $MODEL_PATH"
echo "Task suite:     $TASK_SUITE"
echo "Task range:     $TASK_RANGE"
echo "Command level:  $COMMAND_LEVEL"
echo "Rollouts/task:  $NUM_ROLLOUTS"
echo "First step:     $FIRST_STEP_ONLY"
echo "Use CoT:        $USE_COT"
echo "Conda env:      $EMB_ENV_NAME"
echo "Start:          $(date)"
echo "=========================================="

python extract_text_embeddings_rollout_internvla.py \
  --model_path ${MODEL_PATH} \
  --task_suite ${TASK_SUITE} \
  --task_range ${TASK_RANGE} \
  --command_levels ${COMMAND_LEVEL} \
  --output_dir ${OUTPUT_DIR} \
  --resolution 256 \
  --num_steps_wait 10 \
  --num_rollouts ${NUM_ROLLOUTS} \
  --seed 0 \
  --cfg_scale 1.5 \
  --num_ddim_steps 10 \
  ${FIRST_STEP_FLAG} \
  ${USE_COT_FLAG}

echo "Finish: $(date)"
