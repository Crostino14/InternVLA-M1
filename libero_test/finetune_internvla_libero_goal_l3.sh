#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=50eps_finetune_internvla_libero_goal_l3
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/finetune_50eps_internvla_l3_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/finetune_50eps_internvla_l3_%j.err

SCRIPT_PATH="$(realpath $0)"

# --- Environment Setup ---
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export WANDB_MODE=offline

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate internvla-m1

# --- Project Paths ---
INTERNVLA_ROOT="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/InternVLA-M1"
export PYTHONPATH="${INTERNVLA_ROOT}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# --- Cache Directory ---
mkdir -p /tmp/$USER/triton_cache
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache

# --- Distributed Training Vars ---
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000

# --- Parameters (with defaults) ---
CHECKPOINT_PATH="${1:-/mnt/beegfs/a.cardamone7/checkpoints/InternVLA-M1-LIBERO-Goal/checkpoints/steps_30000_pytorch_model.pt}"
PRETRAINED_CHECKPOINT="/mnt/beegfs/a.cardamone7/checkpoints/InternVLA-M1-LIBERO-Goal/checkpoints/steps_30000_pytorch_model.pt"
RUN_ID_NOTE="${2:-l3_spatial_finetune}"
MAX_EPISODES_PER_TASK="${3:-50}"
MAX_STEPS="${4:-50000}"
SAVE_INTERVAL="${5:-500}"
RUN_ROOT_DIR="${6:-/mnt/beegfs/a.cardamone7/checkpoints/InternVLA_L3_Variations_finetune_libero_goal}"
WANDB_PROJECT="${7:-InternVLA_L3_SpatialGrounding}"
WANDB_ENTITY="${8:-agostino-cardamone2001}"
QWEN_VLM_PATH="/mnt/beegfs/a.cardamone7/checkpoints/Qwen2.5-VL-3B-Instruct"

RUN_DIR="${RUN_ROOT_DIR}/internvla_l3_eps${MAX_EPISODES_PER_TASK}_${RUN_ID_NOTE}"

# --- Auto-resume: cerca l'ultimo checkpoint salvato ---
LAST_CKPT=$(ls ${RUN_DIR}/checkpoints/steps_*_pytorch_model.pt 2>/dev/null \
  | sort -t_ -k2 -n | tail -1)

#LAST_CKPT="/mnt/beegfs/a.cardamone7/checkpoints/InternVLA_L3_finetune_libero_goal/internvla_l3_eps50_l3_spatial_finetune/checkpoints/#steps_24000_pytorch_model.pt"

if [ -n "$LAST_CKPT" ]; then
  LAST_STEP=$(echo "$LAST_CKPT" | grep -oP 'steps_\K[0-9]+')
  echo ">>> RIPRENDENDO DA: $LAST_CKPT (step $LAST_STEP / $MAX_STEPS)"
  CHECKPOINT_PATH="$LAST_CKPT"
else
  echo ">>> NESSUN CHECKPOINT TROVATO — parto dal pretrained"
  CHECKPOINT_PATH="$PRETRAINED_CHECKPOINT"
  LAST_STEP=0
fi

echo "======================================================"
echo "InternVLA Fine-tuning on LIBERO Goal L3"
echo "======================================================"
echo "Checkpoint Path:        $CHECKPOINT_PATH"
echo "Run ID Note:            $RUN_ID_NOTE"
echo "Max Episodes per Task:  $MAX_EPISODES_PER_TASK"
echo "Max Steps:              $MAX_STEPS"
echo "Save Interval:          $SAVE_INTERVAL"
echo "Run Root Directory:     $RUN_ROOT_DIR"
echo "WandB Project:          $WANDB_PROJECT"
echo "WandB Entity:           $WANDB_ENTITY"
echo "Qwen VLM Path:          $QWEN_VLM_PATH"
echo "======================================================"

# Assign a unique port based on the Slurm job ID
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "Using MASTER_PORT=$MASTER_PORT"

accelerate launch \
  --config_file ${INTERNVLA_ROOT}/InternVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  --main_process_port ${MASTER_PORT} \
  ${INTERNVLA_ROOT}/InternVLA/training/train_internvla.py \
  --config_yaml ${INTERNVLA_ROOT}/InternVLA/config/training/internvla_cotrain_libero_l3.yaml \
  --datasets.vla_data.per_device_batch_size 8 \
  --framework.action_model.repeated_diffusion_steps 10 \
  --trainer.gradient_accumulation_steps 4 \
  --datasets.vla_data.data_mix "libero_goal_l3_finetune" \
  --datasets.vla_data.data_root_dir "/mnt/beegfs/a.cardamone7/datasets/lerobot_libero_goal_l3" \
  --datasets.vla_data.max_episodes_per_task ${MAX_EPISODES_PER_TASK} \
  --framework.action_model.future_action_window_size 7 \
  --trainer.num_warmup_steps 1500 \
  --trainer.max_train_steps ${MAX_STEPS} \
  --trainer.save_interval ${SAVE_INTERVAL} \
  --trainer.pretrained_checkpoint=${CHECKPOINT_PATH} \
  --trainer.resume_step=${LAST_STEP} \
  --run_root_dir ${RUN_ROOT_DIR} \
  --run_id "internvla_l3_eps${MAX_EPISODES_PER_TASK}_${RUN_ID_NOTE}" \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_entity ${WANDB_ENTITY} \
  --is_debug False \
  --framework.qwenvl.base_vlm ${QWEN_VLM_PATH}
