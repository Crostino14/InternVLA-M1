#!/usr/bin/env python3
"""
Test the Qwen2.5-VL model INSIDE the BASE InternVLA-M1 checkpoint (from HuggingFace).
NOT the LIBERO-fine-tuned version.

This tests the hypothesis: Does the base checkpoint have a working Qwen?
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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


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


def test_qwen_base_checkpoint():
    """Test Qwen2.5-VL INSIDE BASE InternVLA-M1 checkpoint (from HuggingFace)."""
    
    print("\n" + "="*80)
    print("Testing Qwen2.5-VL INSIDE BASE InternVLA-M1 Checkpoint (HuggingFace)")
    print("="*80)
    
    # Load BASE InternVLA-M1 Qwen from local checkpoint (safetensors format)
    model_path = "/mnt/beegfs/a.cardamone7/checkpoints/InternVLA-M1"
    print(f"\n[INFO] Loading BASE InternVLA-M1 Qwen from local path: {model_path}...")
    print("[INFO] Loading with transformers (supports safetensors format)...")
    
    # Load the Qwen model directly using transformers (which supports safetensors)
    try:
        qwenvl = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, device_map="cuda")
        print(f"[INFO] Successfully loaded Qwen model from safetensors")
        print(f"[INFO] Qwen model device: {next(qwenvl.parameters()).device}")
    except Exception as e:
        print(f"[ERROR] Failed to load with device_map, trying without...")
        qwenvl = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path)
        qwenvl = qwenvl.to("cuda")
        print(f"[INFO] Qwen model loaded and moved to CUDA")
    
    qwenvl = qwenvl.eval()
    
    # Load tokenizer and processor from HuggingFace
    print("[INFO] Loading tokenizer and processor from HuggingFace...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    tokenizer = processor.tokenizer
    print("[INFO] Tokenizer and processor loaded successfully")
    
    # Get LIBERO frames
    print("\n" + "="*80)
    print("Loading LIBERO frames...")
    print("="*80)
    agentview, wrist, task_description = get_libero_frames()
    
    # Prepare images for Qwen
    from qwen_vl_utils import process_vision_info
    
    print(f"\nBase task instruction: {task_description}")
    
    # Test 1: BoxPrompting
    print(f"\n" + "="*80)
    print("TEST 1: BoxPrompting (from ST4VLA paper)")
    print("="*80)
    
    box_prompt = "Figure out how to execute it, then locate the target objects needed. Give the box coordinates according to the instruction"
    full_instruction_1 = f"{task_description}. {box_prompt}"
    
    print(f"Full instruction:\n  {full_instruction_1}\n")
    
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": agentview},
                    {"type": "image", "image": wrist},
                    {"type": "text", "text": full_instruction_1},
                ],
            }
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **{"image_input_type": "pil"}
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to("cuda")
        
        with torch.inference_mode():
            generated_ids = qwenvl.generate(**inputs, max_new_tokens=512, temperature=0.7)
        
        response_1 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("Response (BoxPrompting):")
        print(response_1)
        print("\n" + "-"*80)
        
        # Check for quality indicators
        if "coordinates" in response_1.lower() or "bbox" in response_1.lower():
            print("✓ Contains bbox/coordinates keywords")
        if "[" in response_1 and "]" in response_1 and response_1.count("[") > 2:
            print("✓ Contains bracket structures (expected for JSON)")
        if len(response_1.strip()) < 50:
            print("⚠ WARNING: Response is very short (may be corrupted)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: CoT Prompting
    print(f"\n" + "="*80)
    print("TEST 2: CoT Prompting (proven to work with standalone Qwen)")
    print("="*80)
    
    cot_prompt = "Your task is: Open the middle layer of the drawer. To identify target objects, you should: 1) Look at the image. 2) Find the target object. 3) Locate their bounding boxes in [x1,y1,x2,y2] format. Return as JSON."
    full_instruction_2 = cot_prompt
    
    print(f"Full instruction:\n  {full_instruction_2}\n")
    
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": agentview},
                    {"type": "image", "image": wrist},
                    {"type": "text", "text": full_instruction_2},
                ],
            }
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **{"image_input_type": "pil"}
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to("cuda")
        
        with torch.inference_mode():
            generated_ids = qwenvl.generate(**inputs, max_new_tokens=512, temperature=0.7)
        
        response_2 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("Response (CoT):")
        print(response_2)
        print("\n" + "-"*80)
        
        # Check for bbox patterns
        import re
        bbox_pattern = r'\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]'
        if re.search(bbox_pattern, response_2):
            print("✓✓✓ FOUND BBOX COORDINATES! Model is working correctly!")
        
        if "bbox" in response_2.lower() or '{"' in response_2:
            print("✓ Contains valid-looking JSON structure")
        
        if len(response_2.strip()) < 50:
            print("⚠ WARNING: Response is very short (may be corrupted)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
        COMPARISON: BASE vs LIBERO-fine-tuned Checkpoints
        ==================================================

        If BASE checkpoint produces coherent text with bbox → Fine-tuning broke Qwen
        If BASE checkpoint also produces garbage → Problem is elsewhere (model loading?)

        Expected scenario: BASE should work, LIBERO-fine-tuned should not.
""")


if __name__ == "__main__":
    test_qwen_base_checkpoint()
