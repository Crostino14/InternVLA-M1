#!/bin/bash
#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=L1v_internvla_libero_eval
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-11          # 10 tasks
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/eval_internvla_L1_Variations_10eps_50000_%a_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/eval_internvla_L1_Variations_10eps_50000_%a_%j.err

# ── Array → seed/task_group mapping (same as TinyVLA script) ─────────────────
ARRAY_ID=$SLURM_ARRAY_TASK_ID
SEED=$((ARRAY_ID / 4))
TASK_GROUP=$((ARRAY_ID % 4))
case $TASK_GROUP in
  0) TASK_RANGE="0-2" ;;
  1) TASK_RANGE="3-5" ;;
  2) TASK_RANGE="6-8" ;;
  3) TASK_RANGE="9-9" ;;
esac

# ── Configuration ─────────────────────────────────────────────────────────────
CHANGE_COMMAND=true
COMMAND_LEVEL="l1"          # l1 | l2 | l3 | all | all_no_default | default

MODEL_PATH="/mnt/beegfs/a.cardamone7/checkpoints/InternVLA_L3_Variations_finetune_libero_goal/internvla_l3_eps10_l3_spatial_finetune/checkpoints/steps_50000_pytorch_model.pt"
TASK_SUITE="libero_goal"

WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/InternVLA-M1/libero_test"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"
INTERNVLA_ROOT="/home/A.CARDAMONE7/repo/InternVLA-M1"
OUTPUT_DIR="/mnt/beegfs/a.cardamone7/outputs"

NUM_TRIALS_PER_TASK=50
NUM_STEPS_WAIT=10
ENV_IMG_RES=256

ID_NOTE="Original_${COMMAND_LEVEL}_Variations_10eps_finetuned_50000steps_seed${SEED}_${TASK_RANGE}"

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
echo "InternVLA-M1 LIBERO Evaluation"
echo "=========================================="
echo "Job ID:         $SLURM_JOB_ID"
echo "Model:          $MODEL_PATH"
echo "Task suite:     $TASK_SUITE"
echo "Command level:  $COMMAND_LEVEL"
echo "Task range:     $TASK_RANGE  | Seed: $SEED"
echo "Start:          $(date)"
echo "=========================================="

cd ${WORK_DIR}

srun python run_internvla_libero_eval.py \
  --model_path       ${MODEL_PATH}         \
  --task_suite_name  ${TASK_SUITE}         \
  --task_range       ${TASK_RANGE}         \
  --change_command   ${CHANGE_COMMAND}     \
  --command_level    ${COMMAND_LEVEL}      \
  --seed             ${SEED}               \
  --run_number       ${SEED}               \
  --use_cot True                           \
  --use_versions     false                  \
  --num_trials_per_task ${NUM_TRIALS_PER_TASK} \
  --num_steps_wait   ${NUM_STEPS_WAIT}     \
  --env_img_res      ${ENV_IMG_RES}        \
  --run_id_note      ${ID_NOTE}            \
  --local_log_dir    ${OUTPUT_DIR}/logs    \

echo "Finish: $(date)"