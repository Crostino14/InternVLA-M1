#!/bin/bash
#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=task7_internvla
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --array=6,16,26         # 10 task × 3 seed = 30 job
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/TEST_10eps_eval_internvla_task7_%a_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/TEST_10eps_eval_internvla_task7_%a_%j.err

# ── Array → SEED + TASK_INDEX (1 job = 1 task, 1 seed) ──────────────────────
# ARRAY_ID: 0-9  → seed 0, task 0-9
# ARRAY_ID: 10-19 → seed 1, task 0-9
# ARRAY_ID: 20-29 → seed 2, task 0-9
ARRAY_ID=$SLURM_ARRAY_TASK_ID
SEED=$((ARRAY_ID / 10))
TASK_INDEX=$((ARRAY_ID % 10))
TASK_RANGE="${TASK_INDEX}-${TASK_INDEX}"

# ── Configuration ─────────────────────────────────────────────────────────────
CHANGE_COMMAND=true
COMMAND_LEVEL="l3"

MODEL_PATH="/mnt/beegfs/a.cardamone7/checkpoints/InternVLA_L3_finetune_libero_goal/internvla_l3_eps10_l3_spatial_finetune/checkpoints/steps_46000_pytorch_model.pt"
TASK_SUITE="libero_goal"

WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/InternVLA-M1/libero_test"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"
INTERNVLA_ROOT="/home/A.CARDAMONE7/repo/InternVLA-M1"
OUTPUT_DIR="/mnt/beegfs/a.cardamone7/outputs"

NUM_TRIALS_PER_TASK=50
NUM_STEPS_WAIT=10
ENV_IMG_RES=256

ID_NOTE="TEST_10eps_internvla_${TASK_SUITE}_${COMMAND_LEVEL}_seed${SEED}_task${TASK_INDEX}"

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
echo "InternVLA-M1 LIBERO Evaluation — Single Task"
echo "=========================================="
echo "Job ID:         $SLURM_JOB_ID"
echo "Array ID:       $ARRAY_ID"
echo "Seed:           $SEED"
echo "Task index:     $TASK_INDEX  (1-based: $((TASK_INDEX + 1)))"
echo "Task range:     $TASK_RANGE"
echo "Command level:  $COMMAND_LEVEL"
echo "Start:          $(date)"
echo "=========================================="

cd ${WORK_DIR}

srun python run_internvla_libero_eval.py \
  --model_path          ${MODEL_PATH}         \
  --task_suite_name     ${TASK_SUITE}         \
  --task_range          ${TASK_RANGE}         \
  --change_command      ${CHANGE_COMMAND}     \
  --command_level       ${COMMAND_LEVEL}      \
  --seed                ${SEED}               \
  --run_number          ${SEED}               \
  --use_cot             true                   \
  --selected_version    1                      \
  --num_trials_per_task ${NUM_TRIALS_PER_TASK} \
  --num_steps_wait      ${NUM_STEPS_WAIT}     \
  --env_img_res         ${ENV_IMG_RES}        \
  --run_id_note         ${ID_NOTE}            \
  --local_log_dir       ${OUTPUT_DIR}/logs    \

echo "Finish: $(date)"