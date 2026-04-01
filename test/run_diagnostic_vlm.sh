#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=internvla_vlm_diagnostic
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/diagnostic_vlm_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/diagnostic_vlm_%j.err

echo "=========================================="
echo "InternVLA-M1 VLM Diagnostic"
echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Start:    $(date)"
echo "=========================================="

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH="/home/A.CARDAMONE7/checkpoints/InternVLA-M1-LIBERO-Goal/checkpoints/steps_30000_pytorch_model.pt"
WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/InternVLA-M1/test"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"
OUTPUT_DIR="/mnt/beegfs/a.cardamone7/outputs"
FRAME_PATH="${OUTPUT_DIR}/diagnostic_frame.png"

# ── Environment ────────────────────────────────────────────────────────────────
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=${LIBERO_PATH}:${WORK_DIR}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS##*:}
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate internvla-m1

echo "Python:    $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

cd ${WORK_DIR}

# ── Step 1: salva una frame dall'env ───────────────────────────────────────────
echo "--- STEP 1: Saving env frame ---"
srun python - <<EOF
import os, sys
sys.path.insert(0, "${LIBERO_PATH}")
sys.path.insert(0, "${WORK_DIR}")
import cv2
import numpy as np
from PIL import Image
from libero.libero import benchmark
from utils.libero_utils import get_libero_env, get_libero_dummy_action

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_goal"]()
task = task_suite.get_task(6)   # task 7: cream cheese → bowl
initial_states = task_suite.get_task_init_states(6)

env, task_desc, orig_desc = get_libero_env(task, "tiny_vla", resolution=256)
env.reset()
obs = env.set_init_state(initial_states[0])
for _ in range(10):
    obs, _, _, _ = env.step(get_libero_dummy_action("tiny_vla"))

agentview = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
agentview = cv2.resize(agentview, (224, 224))
Image.fromarray(agentview.astype(np.uint8)).save("${FRAME_PATH}")
print(f"Frame saved → ${FRAME_PATH}")
print(f"Task L0: {orig_desc}")
print(f"Task L3: {task_desc}")
EOF

# ── Step 2: diagnostic VLM ─────────────────────────────────────────────────────
echo ""
echo "--- STEP 2: VLM Diagnostic ---"
srun python - <<'EOF'
import os, sys, torch
sys.path.insert(0, "/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO")
sys.path.insert(0, "/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/InternVLA-M1/test")

from PIL import Image
from qwen_vl_utils import process_vision_info
from InternVLA.model.framework.M1 import InternVLA_M1

MODEL_PATH = "/home/A.CARDAMONE7/checkpoints/InternVLA-M1-LIBERO-Goal/checkpoints/steps_30000_pytorch_model.pt"
FRAME_PATH = "/mnt/beegfs/a.cardamone7/outputs/diagnostic_frame.png"

print("Loading model...", flush=True)
model = InternVLA_M1.from_pretrained(MODEL_PATH).to("cuda").eval()

# ── Trova il processor ────────────────────────────────────────────────────────
print("\n=== qwen_vl_interface attrs ===", flush=True)
for k, v in model.qwen_vl_interface.__dict__.items():
    if not k.startswith("_"):
        print(f"  .{k}  =  {type(v).__name__}", flush=True)

processor = None
for candidate in ["processor", "tokenizer", "image_processor", "vlm_processor"]:
    if hasattr(model.qwen_vl_interface, candidate):
        processor = getattr(model.qwen_vl_interface, candidate)
        print(f"\n✅ Processor: qwen_vl_interface.{candidate} → {type(processor).__name__}", flush=True)
        break

if processor is None:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    print(f"\n⚠️  Processor fallback HF: {type(processor).__name__}", flush=True)

vlm   = model.qwen_vl_interface
frame = Image.open(FRAME_PATH)

# ── Test 1: text generation ───────────────────────────────────────────────────
print("\n=== TEST 1: Text Generation ===", flush=True)
for q in [
    "What object is in front of the stove?",
    "Where is the cream cheese?",
    "List all visible objects on the table.",
]:
    msg = [{"role": "user", "content": [
        {"type": "image", "image": frame},
        {"type": "text",  "text": q}
    ]}]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    if isinstance(text, list): text = text[0]
    image_inputs, _ = process_vision_info(msg)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = vlm.model.generate(**inputs, max_new_tokens=64, do_sample=False)
    answer = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Q: {q}\nA: {answer}\n", flush=True)

# ── Test 2: cosine similarity ─────────────────────────────────────────────────
print("=== TEST 2: Cosine Similarity L0 vs L3 ===", flush=True)
def get_last_hidden(instruction):
    msg = [{"role": "user", "content": [
        {"type": "image", "image": frame},
        {"type": "text",  "text": instruction}
    ]}]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    if isinstance(text, list): text = text[0]
    image_inputs, _ = process_vision_info(msg)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = vlm.model(**inputs, output_hidden_states=True)
    return out.hidden_states[-1].mean(dim=1).squeeze().cpu().float()

l0  = get_last_hidden("Put the cream cheese in the bowl")
l3  = get_last_hidden("Put the object in front of the stove on the bowl")
cot = get_last_hidden("Put the object in front of the stove on the bowl. Locate the key object needed.")

cos = torch.nn.functional.cosine_similarity
print(f"cos(L0,  L3)     = {cos(l0.unsqueeze(0), l3.unsqueeze(0)).item():.4f}", flush=True)
print(f"cos(L0,  CoT-L3) = {cos(l0.unsqueeze(0), cot.unsqueeze(0)).item():.4f}", flush=True)
print(f"cos(L3,  CoT-L3) = {cos(l3.unsqueeze(0), cot.unsqueeze(0)).item():.4f}", flush=True)
print("DONE", flush=True)
EOF


echo ""
echo "=========================================="
echo "Diagnostic complete: $(date)"
echo "Output log: /mnt/beegfs/a.cardamone7/outputs/logs/diagnostic_vlm_${SLURM_JOB_ID}.out"
echo "=========================================="