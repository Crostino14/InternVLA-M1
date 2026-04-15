#!/usr/bin/env python3
"""
Debug script: inspect what the VLM actually produces during prediction.
Uses REAL LIBERO frames to test bbox generation.
"""

import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['DEVICE'] = "cuda"

from InternVLA.model.framework.M1 import InternVLA_M1

# Add LIBERO paths
sys.path.insert(0, "/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO")
sys.path.insert(0, "/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/InternVLA-M1/libero_test")

def get_libero_frames():
    """
    Load a LIBERO task and capture real frames from the environment.
    Returns: (agentview_img, wrist_img, task_description)
    """
    from libero.libero import benchmark
    from utils.libero_utils import get_libero_env
    
    print("[DEBUG] Loading LIBERO benchmark...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_goal"]()
    
    print(f"[DEBUG] Loading task 0 from LIBERO...")
    task = task_suite.get_task(0)
    initial_states = task_suite.get_task_init_states(0)
    
    env, task_description, _ = get_libero_env(
        task,
        "franka",  # Use Franka robot (standard for LIBERO/InternVLA-M1)
        change_command=False,
        command_level=None,
        resolution=256,
    )
    
    print(f"[DEBUG] Resetting environment...")
    env.reset()
    obs = env.set_init_state(initial_states[0])
    
    # Stabilize physics
    dummy_action = [0.0] * 7
    for _ in range(10):
        obs, _, _, _ = env.step(dummy_action)
    
    # Extract frames
    agentview = obs['agentview_image'][::-1, ::-1]  # Flip as in official code
    wrist = obs['robot0_eye_in_hand_image'][::-1, ::-1]
    
    agentview = cv2.resize(agentview, (224, 224))
    wrist = cv2.resize(wrist, (224, 224))
    
    agentview = Image.fromarray(agentview.astype(np.uint8))
    wrist = Image.fromarray(wrist.astype(np.uint8))
    
    print(f"[DEBUG] Captured frames from LIBERO task: {task_description}")
    print(f"        agentview: {agentview.size}, wrist: {wrist.size}")
    
    return agentview, wrist, task_description


def deep_inspect(obj, name="obj", depth=0, max_depth=4, visited=None):
    """Recursively inspect object structure."""
    if visited is None:
        visited = set()
    
    indent = "  " * depth
    obj_id = id(obj)
    
    if obj_id in visited or depth > max_depth:
        return
    visited.add(obj_id)
    
    print(f"{indent}{name}: {type(obj).__name__}", end="")
    
    if isinstance(obj, dict):
        print(f" (keys: {list(obj.keys())[:5]}...)" if len(obj) > 5 else f" (keys: {list(obj.keys())})")
        for k, v in list(obj.items())[:3]:  # First 3 items
            deep_inspect(v, f".{k}", depth+1, max_depth, visited)
    elif isinstance(obj, (list, tuple)):
        print(f" (len={len(obj)})")
        if len(obj) > 0:
            deep_inspect(obj[0], "[0]", depth+1, max_depth, visited)
    elif isinstance(obj, torch.Tensor):
        print(f" shape={obj.shape}, dtype={obj.dtype}")
    elif isinstance(obj, np.ndarray):
        print(f" shape={obj.shape}, dtype={obj.dtype}")
    else:
        print()


def monkeypatch_predict_action(model):
    """Wrap predict_action to capture all intermediate outputs."""
    original_predict = model.predict_action
    
    def wrapped_predict_action(batch_images, instructions, **kwargs):
        print("\n" + "="*80)
        print("MONKEYPATCHED PREDICT_ACTION CALLED")
        print("="*80)
        print(f"batch_images: {len(batch_images)} samples, each with {len(batch_images[0])} views")
        print(f"instructions: {instructions}")
        
        # Inline the prediction to capture qwenvl_outputs
        from InternVLA.training.trainer_utils.metrics import resize_images
        
        train_obs_image_size = getattr(model.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        instructions = [instruction.lower() for instruction in instructions]

        inferface_inputs = model.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        qwen_inputs = inferface_inputs

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = model.qwen_vl_interface(
                **qwen_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            
            print("\n[DEBUG] qwenvl_outputs structure:")
            print(f"  Keys: {list(qwenvl_outputs.keys())}")
            if hasattr(qwenvl_outputs, 'hidden_states'):
                print(f"  hidden_states: {len(qwenvl_outputs.hidden_states)} layers")
                for i, layer_hs in enumerate(qwenvl_outputs.hidden_states):
                    if hasattr(layer_hs, 'shape'):
                        print(f"    Layer {i}: {layer_hs.shape}")
            
            if hasattr(qwenvl_outputs, 'logits'):
                logits = qwenvl_outputs.logits
                print(f"  logits shape: {logits.shape}")
                
                # Try to decode the logits to text to see if bbox are there
                try:
                    tokenizer = model.qwen_vl_interface.processor.tokenizer
                    predicted_ids = torch.argmax(logits, dim=-1)
                    predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=False)
                    print(f"  Decoded text (first 500 chars): {predicted_text[:500]}")
                except Exception as e:
                    print(f"  Could not decode logits: {e}")
        
        # Call original
        result = original_predict(batch_images, instructions, **kwargs)
        
        print("\nRETURN VALUE FROM predict_action:")
        deep_inspect(result, "result")
        
        print("\nRETURN KEYS:")
        if isinstance(result, dict):
            for key in result.keys():
                val = result[key]
                if isinstance(val, np.ndarray):
                    print(f"  {key}: ndarray {val.shape} {val.dtype}")
                else:
                    print(f"  {key}: {type(val).__name__}")
        
        return result
    
    model.predict_action = wrapped_predict_action


def main():
    model_path = "/home/A.CARDAMONE7/checkpoints/InternVLA-M1-LIBERO-Goal/checkpoints/steps_30000_pytorch_model.pt"
    
    print(f"Loading model from: {model_path}")
    model = InternVLA_M1.from_pretrained(model_path)
    model = model.to("cuda").eval()
    
    print("\n" + "="*80)
    print("Loading real LIBERO frames...")
    print("="*80)
    
    # Get real frames from LIBERO
    agentview_img, wrist_img, task_description = get_libero_frames()
    
    # Monkeypatch for debugging
    monkeypatch_predict_action(model)
    
    batch_images = [[agentview_img, wrist_img]]
    instructions = [task_description]
    
    print("\n" + "="*80)
    print("TEST 1: CALLING predict_action WITHOUT CoT (baseline)")
    print("="*80)
    print(f"Task instruction: {task_description}")
    
    with torch.inference_mode():
        result_baseline = model.predict_action(
            batch_images=batch_images,
            instructions=instructions,
            cfg_scale=1.5,
            use_ddim=True,
            num_ddim_steps=10,
            use_cot_grounding=False,
        )
    
    print("\n→ Result WITHOUT CoT: normalized_actions shape", result_baseline["normalized_actions"].shape)
    
    print("\n" + "="*80)
    print("TEST 2: CALLING predict_action WITH CoT grounding")
    print("="*80)
    print(f"Task instruction: {task_description}")
    
    with torch.inference_mode():
        result_cot = model.predict_action(
            batch_images=batch_images,
            instructions=instructions,
            cfg_scale=1.5,
            use_ddim=True,
            num_ddim_steps=10,
            use_cot_grounding=True,  # Enable CoT grounding
        )
    
    print("\n→ Result WITH CoT: normalized_actions shape", result_cot["normalized_actions"].shape)
    
    print("\n" + "="*80)
    print("FINAL RESULT:")
    print("="*80)
    print(json.dumps({
        "baseline": str(result_baseline["normalized_actions"].shape),
        "with_cot": str(result_cot["normalized_actions"].shape)
    }, indent=2))


if __name__ == "__main__":
    main()
