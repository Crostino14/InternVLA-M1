#!/usr/bin/env python3
"""
Test InternVLA-M1 with Grounding Integration
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

from InternVLA_with_grounding import create_model_with_grounding


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
    print("Testing InternVLA-M1 WITH Grounding Integration")
    print("="*80)
    
    # Create model with grounding
    model_path = "/home/A.CARDAMONE7/checkpoints/InternVLA-M1-LIBERO-Goal/checkpoints/steps_30000_pytorch_model.pt"
    wrapper = create_model_with_grounding(model_path, use_grounding=True)
    
    # Get LIBERO frames
    print("\n" + "="*80)
    print("Loading LIBERO frames...")
    print("="*80)
    agentview, wrist, task_description = get_libero_frames()
    
    batch_images = [[agentview, wrist]]
    instructions = [task_description]
    
    print("\n" + "="*80)
    print("Predicting actions WITH grounding...")
    print("="*80)
    print(f"Instruction: {task_description}")
    
    result = wrapper.predict_action_with_grounding(
        batch_images=batch_images,
        instructions=instructions,
        cfg_scale=1.5,
        use_ddim=True,
        num_ddim_steps=10,
    )
    
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    
    print(f"\n1. Actions shape: {result['normalized_actions'].shape}")
    print(f"   (batch, timesteps, action_dim) = (1, 8, 7)")
    
    if "grounding" in result:
        grounding = result["grounding"]
        print(f"\n2. Grounding detections:")
        for detection in grounding["detections"]:
            print(f"   Instruction: {detection['instruction']}")
            print(f"   Detected objects: {len(detection['bboxes'])}")
            for bbox_obj in detection["bboxes"]:
                bbox = bbox_obj["bbox"]
                label = bbox_obj["label"]
                print(f"     - {label}: {bbox}")
    
    print("\n" + "="*80)
    print("SUCCESS: InternVLA-M1 with Grounding is working!")
    print("="*80)


if __name__ == "__main__":
    main()
