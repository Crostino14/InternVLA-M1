#!/bin/bash
#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=internvla_task_comp_eval
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --array=2
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/eval_internvla_taskcomp_%a_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/eval_internvla_taskcomp_%a_%j.err


# ── Array → seed/task_group mapping ──────────────────────────────────────────
# 5 task comp tasks split in 3 groups × 3 seeds = 9 array jobs (0-8)
ARRAY_ID=$SLURM_ARRAY_TASK_ID
SEED=$((ARRAY_ID / 3))
TASK_GROUP=$((ARRAY_ID % 3))

case $TASK_GROUP in
  0) TASK_START=0; TASK_END=1 ;;
  1) TASK_START=2; TASK_END=3 ;;
  2) TASK_START=4; TASK_END=4 ;;
esac


# ── Configuration ─────────────────────────────────────────────────────────────
COMP_LEVEL="l1"             # l1 | l2

MODEL_PATH="/home/A.CARDAMONE7/checkpoints/InternVLA-M1-LIBERO-Goal/checkpoints/steps_30000_pytorch_model.pt"

WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/InternVLA-M1/libero_test"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"
INTERNVLA_ROOT="/home/A.CARDAMONE7/repo/InternVLA-M1"
OUTPUT_DIR="/mnt/beegfs/a.cardamone7/outputs"

NUM_TRIALS_PER_TASK=50
NUM_STEPS_WAIT=10
ENV_IMG_RES=256

ID_NOTE="internvla_task_comp_${COMP_LEVEL}_seed${SEED}_tasks${TASK_START}-${TASK_END}"


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
echo "InternVLA-M1 Task Composition Evaluation"
echo "=========================================="
echo "Job ID:         $SLURM_JOB_ID"
echo "Array ID:       $ARRAY_ID"
echo "Model:          $MODEL_PATH"
echo "Comp level:     $COMP_LEVEL"
echo "Task range:     ${TASK_START}-${TASK_END}  | Seed: $SEED"
echo "Start:          $(date)"
echo "=========================================="


cd ${WORK_DIR}


srun python run_internvla_libero_task_comp.py \
  --model_path          ${MODEL_PATH}           \
  --comp_level          ${COMP_LEVEL}           \
  --task_start          ${TASK_START}           \
  --task_end            ${TASK_END}             \
  --seed                ${SEED}                 \
  --run_number          ${SEED}                 \
  --num_trials_per_task ${NUM_TRIALS_PER_TASK}  \
  --num_steps_wait      ${NUM_STEPS_WAIT}       \
  --env_img_res         ${ENV_IMG_RES}          \
  --run_id_note         ${ID_NOTE}              \
  --local_log_dir       ${OUTPUT_DIR}/logs      \


echo "Finish: $(date)"