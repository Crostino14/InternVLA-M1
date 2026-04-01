#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=internvla_vqa_eval
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/internvla_vqa_eval_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/internvla_vqa_eval_%j.err

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# Path to the base model from Hugging Face.
# The model will be downloaded automatically if not in the cache.
BASE_MODEL_PATH="/mnt/beegfs/a.cardamone7/checkpoints/InternVLA-M1"

# Path to your local fine-tuned checkpoint file.
FINETUNED_CHECKPOINT_PATH=""

WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/InternVLA-M1/test/vqa_test"
PROMPTS_JSON="/mnt/beegfs/a.cardamone7/outputs/vqa_test/spatial_benchmark.json"
OUTPUT_JSON="${WORK_DIR}/vqa_results_internvla.json"
OUTPUT_BBOX_DIR="${WORK_DIR}/bboxes"

# This should be the name of the conda environment for InternVLA
# Modify if the name is different.
ENV_NAME="internvla-m1"

# --------------------------------------------------------------------------
# SLURM Environment Setup
# --------------------------------------------------------------------------
echo "=========================================="
echo "InternVLA-M1 VQA Evaluation"
echo "=========================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Start time:      $(date)"
echo "Base Model Path: $BASE_MODEL_PATH"
echo "Checkpoint Path: $FINETUNED_CHECKPOINT_PATH"
echo "Prompts JSON:    $PROMPTS_JSON"
echo "Output JSON:     $OUTPUT_JSON"
echo "Output BBox Dir: $OUTPUT_BBOX_DIR"
echo "=========================================="

# Activate Conda environment
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS##*:}

# Navigate to the working directory
cd "${WORK_DIR}"

echo "Working directory: $(pwd)"
echo "Python executable: $(which python)"
echo "Conda env:         ${CONDA_DEFAULT_ENV}"
echo ""

# --------------------------------------------------------------------------
# Run Evaluation Script
# --------------------------------------------------------------------------
srun python internvla_vqa_eval.py \
    --model_path "${BASE_MODEL_PATH}" \
    --checkpoint_path "${FINETUNED_CHECKPOINT_PATH}" \
    --prompts_json "${PROMPTS_JSON}" \
    --output_json "${OUTPUT_JSON}" \
    --output_bbox_dir "${OUTPUT_BBOX_DIR}"

EXIT_CODE=$?

# --------------------------------------------------------------------------
# Finalization
# --------------------------------------------------------------------------
echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "VQA evaluation completed successfully."
    echo "Results saved to: ${OUTPUT_JSON}"
    echo "Bounding box images saved in: ${OUTPUT_BBOX_DIR}"
else
    echo "VQA evaluation failed with exit code ${EXIT_CODE}."
fi
echo "Finish time: $(date)"
echo "=========================================="

exit ${EXIT_CODE}
