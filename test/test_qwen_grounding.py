#!/usr/bin/env python3
"""
Test if Qwen2.5-VL (standalone, not InternVLA-M1) generates bounding boxes
with CoT prompt for grounding.
"""

import torch
from PIL import Image
import cv2
import numpy as np
import os
import sys
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['DEVICE'] = "cuda"

# Add LIBERO paths
sys.path.insert(0, "/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO")
sys.path.insert(0, "/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/InternVLA-M1/libero_test")


def get_libero_frames():
    """Load a LIBERO task and capture real frames from the environment."""
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
        "franka",
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
    agentview = obs['agentview_image'][::-1, ::-1]
    wrist = obs['robot0_eye_in_hand_image'][::-1, ::-1]
    
    agentview = cv2.resize(agentview, (224, 224))
    wrist = cv2.resize(wrist, (224, 224))
    
    agentview = Image.fromarray(agentview.astype(np.uint8))
    wrist = Image.fromarray(wrist.astype(np.uint8))
    
    print(f"[DEBUG] Captured frames from LIBERO task: {task_description}")
    
    return agentview, wrist, task_description


def test_qwen_grounding():
    """Test Qwen2.5-VL for grounding generation."""
    
    print("\n" + "="*80)
    print("Testing Qwen2.5-VL Grounding Capabilities")
    print("="*80)
    
    # Load using exact pattern from InternVLA codebase
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    print(f"\n[INFO] Loading {model_id}...")
    
    # Load model exactly as InternVLA does it
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        attn_implementation="flash_attention_2",
        torch_dtype="auto",
        device_map="cuda",
    )
    
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    
    model.eval()
    
    # Get LIBERO frames
    print("\n" + "="*80)
    print("Loading LIBERO frames...")
    print("="*80)
    agentview, wrist, task_description = get_libero_frames()
    
    # Test 1: Plain instruction without CoT
    print("\n" + "="*80)
    print("TEST 1: Plain instruction (no CoT)")
    print("="*80)
    
    prompt_plain = f"In this image, what action should I take? {task_description}"
    print(f"Prompt: {prompt_plain}")
    
    messages = [
        [{"role": "user", "content": [
            {"type": "image", "image": agentview},
            {"type": "image", "image": wrist},
            {"type": "text", "text": prompt_plain},
        ]}]
    ]
    
    text = processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
    
    generated_text_plain = processor.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated text (first 300 chars):\n{generated_text_plain[:300]}\n")
    
    # Test 2: CoT prompt asking for bounding boxes
    print("\n" + "="*80)
    print("TEST 2: CoT prompt asking for bounding boxes")
    print("="*80)
    
    prompt_cot = (
        f"Your task is: {task_description}. "
        "To identify the target objects for your task, locate their bounding boxes "
        "in [x1,y1,x2,y2] format on a 224x224 image. "
        "First locate the bounding boxes, then generate the action."
    )
    print(f"Prompt: {prompt_cot[:150]}...")
    
    messages_cot = [
        [{"role": "user", "content": [
            {"type": "image", "image": agentview},
            {"type": "image", "image": wrist},
            {"type": "text", "text": prompt_cot},
        ]}]
    ]
    
    text_cot = processor.apply_chat_template(messages_cot[0], tokenize=False, add_generation_prompt=True)
    image_inputs_cot, video_inputs_cot = process_vision_info(messages_cot)
    
    inputs_cot = processor(
        text=[text_cot],
        images=image_inputs_cot,
        videos=video_inputs_cot,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    
    with torch.inference_mode():
        output_ids_cot = model.generate(**inputs_cot, max_new_tokens=512, temperature=0.1)
    
    generated_text_cot = processor.decode(output_ids_cot[0], skip_special_tokens=True)
    print(f"Generated text (first 500 chars):\n{generated_text_cot[:500]}\n")
    
    # Check if bbox are present
    print("\n" + "="*80)
    print("ANALYSIS:")
    print("="*80)
    
    has_coords_plain = any(char.isdigit() and char != ' ' for char in generated_text_plain)
    has_coords_cot = any('[' in generated_text_cot and ']' in generated_text_cot 
                         and any(c.isdigit() for c in generated_text_cot))
    
    print(f"Plain instruction - contains numbers: {has_coords_plain}")
    print(f"CoT prompt - contains bracket coordinates: {has_coords_cot}")
    
    if "[" in generated_text_cot and "]" in generated_text_cot:
        # Extract potential bbox
        import re
        coords = re.findall(r'\[[\d\s,]+\]', generated_text_cot)
        if coords:
            print(f"Found potential bboxes: {coords}")
    
    return generated_text_plain, generated_text_cot


if __name__ == "__main__":
    test_qwen_grounding()
