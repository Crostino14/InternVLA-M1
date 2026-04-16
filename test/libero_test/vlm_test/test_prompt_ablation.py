#!/usr/bin/env python3
"""
Test InternVLA-M1 with BoxPrompting from ST4VLA paper.
Section C.5 Ablation Study on Spatial Prompt Formulations.
"""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['DEVICE'] = "cuda"

# Add paths
sys.path.insert(0, "/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO")
sys.path.insert(0, "/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/InternVLA-M1/libero_test")
sys.path.insert(0, "/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/InternVLA-M1")
sys.path.insert(0, "/home/A.CARDAMONE7/anaconda3/envs/libero/lib/python3.11/site-packages")

from InternVLA.model.framework.M1 import InternVLA_M1


BOX_PROMPT = "Figure out how to execute it, then locate the key object needed. Give the box coordinates according to the instruction"


def get_libero_frames():
    """Load a LIBERO task and capture real frames."""
    from libero.libero import benchmark
    from utils.libero_utils import get_libero_env
    
    print("[DEBUG] Loading LIBERO benchmark...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_goal"]()
    
    task = task_suite.get_task(0)
    initial_states = task_suite.get_task_init_states(0)
    
    env, task_description, _ = get_libero_env(
        task, "franka", change_command=False, command_level=None, resolution=256,
    )
    
    env.reset()
    obs = env.set_init_state(initial_states[0])
    
    # Stabilize
    dummy_action = [0.0] * 7
    for _ in range(10):
        obs, _, _, _ = env.step(dummy_action)
    
    # Extract frames
    agentview = obs['agentview_image'][::-1, ::-1]
    wrist = obs['robot0_eye_in_hand_image'][::-1, ::-1]
    
    agentview = cv2.resize(agentview, (224, 224))
    wrist = cv2.resize(wrist, (224, 224))
    
    agentview = Image.fromarray(agentview.astype(np.uint8))
    wrist = Image.fromarray(wrist.astype(np.uint8))
    
    print(f"[DEBUG] Captured frames: {task_description}")
    return agentview, wrist, task_description


def main():
    print("\n" + "="*80)
    print("InternVLA-M1: BoxPrompting from ST4VLA (Section C.5)")
    print("="*80)
    
    # Load model
    model_path = "/home/A.CARDAMONE7/checkpoints/InternVLA-M1-LIBERO-Goal/checkpoints/steps_30000_pytorch_model.pt"
    print(f"\n[INFO] Loading model from {model_path}...")
    model = InternVLA_M1.from_pretrained(model_path)
    model = model.to("cuda").eval()
    
    # Get LIBERO frames
    print("\n" + "="*80)
    print("Loading LIBERO frames...")
    print("="*80)
    agentview, wrist, task_description = get_libero_frames()
    batch_images = [[agentview, wrist]]
    
    print(f"\nBase task instruction: {task_description}")
    
    # Combine task with BoxPrompting
    full_instruction = f"{task_description}. {BOX_PROMPT}"
    instructions = [full_instruction]
    
    print(f"\n" + "="*80)
    print("BoxPrompting Test")
    print("="*80)
    print(f"Full instruction:\n  {full_instruction}\n")
    
    try:
        with torch.inference_mode():
            result = model.predict_action(
                batch_images=batch_images,
                instructions=instructions,
                cfg_scale=1.5,
                use_ddim=True,
                num_ddim_steps=5,
            )
        
        actions = result["normalized_actions"]
        print(f"✓ Actions generated!")
        print(f"  Shape: {actions.shape}")
        print(f"  First timestep action: {actions[0, 0, :].tolist()}")
        
        print("\n" + "="*80)
        print("RESULT")
        print("="*80)
        print("✓ BoxPrompting works: M1 generates actions when asked for bbox")
        print(f"  But remember: M1 ONLY outputs actions [7,], not coordinates")
        print(f"  The prompt influences hidden_states via VLM processing")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
