#!/bin/bash
#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=hf_download
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/hf_download_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/hf_download_%j.err

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate internvla-m1

export HF_TOKEN="hf_il_tuo_token"
export HF_HUB_ENABLE_HF_TRANSFER=1

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot',
    repo_type='dataset',
    local_dir='/mnt/beegfs/a.cardamone7/datasets/lerobot_libero_goal',
    max_workers=8,
    resume_download=True,
    token='YOUR_HF_TOKEN'
)
print('Download completato')
"