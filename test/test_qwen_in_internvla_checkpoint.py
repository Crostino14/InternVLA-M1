#!/usr/bin/env python3
"""
Test the Qwen2.5-VL model INSIDE the InternVLA-M1 checkpoint directly.
Does it generate bbox when prompted with BoxPrompting?
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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


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


def test_qwen_inside_checkpoint():
    """Test Qwen2.5-VL INSIDE InternVLA-M1 checkpoint."""
    
    print("\n" + "="*80)
    print("Testing Qwen2.5-VL INSIDE InternVLA-M1 Checkpoint")
    print("="*80)
    
    # Load InternVLA-M1 from local checkpoint (LIBERO fine-tuned)
    model_path = "/mnt/beegfs/a.cardamone7/checkpoints/InternVLA-M1-LIBERO-Goal/checkpoints/steps_30000_pytorch_model.pt"
    print(f"\n[INFO] Loading InternVLA-M1 from {model_path}...")
    model = InternVLA_M1.from_pretrained(model_path)
    model = model.to("cuda").eval()
    
    # Extract the Qwen VL model from inside
    # In InternVLA_M1, the VLM is stored as self.qwen_vl_interface.model
    qwenvl = model.qwen_vl_interface.model
    print(f"[INFO] Extracted Qwen model from checkpoint: {type(qwenvl)}")
    print(f"[INFO] Qwen model device: {next(qwenvl.parameters()).device}")
    
    # Load tokenizer and processor from HuggingFace (they're not in the checkpoint)
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
    
    # Stack images: agent view + wrist
    batch_images = [[agentview, wrist]]
    
    print(f"\nBase task instruction: {task_description}")
    
    # Test 1: BoxPrompting
    print(f"\n" + "="*80)
    print("TEST 1: BoxPrompting")
    print("="*80)
    
    box_prompt = "Figure out how to execute it, then locate the target objects needed. Give the box coordinates according to the instruction"
    full_instruction_1 = f"{task_description}. {box_prompt}"
    
    print(f"Full instruction:\n  {full_instruction_1}\n")
    
    try:
        # Process images for Qwen
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
        
        # Prepare input
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
        
        # Generate response
        with torch.inference_mode():
            generated_ids = qwenvl.generate(**inputs, max_new_tokens=512, temperature=0.7)
        
        response_1 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("Response (BoxPrompting):")
        print("\n" + "="*80)
        print(response_1)
        print("="*80 + "\n")
        
        # Save response to file for full inspection
        with open("/mnt/beegfs/a.cardamone7/outputs/response_boxprompting.txt", "w") as f:
            f.write(response_1)
        print(f"[DEBUG] Full response saved to /mnt/beegfs/a.cardamone7/outputs/response_boxprompting.txt")
        
        # Check if contains bbox-like patterns
        if "[" in response_1 and "]" in response_1:
            print("\n✓ Contains bracket-like structure (possible bbox)")
        if "bbox" in response_1.lower() or "coordinates" in response_1.lower():
            print("✓ Contains bbox/coordinates keywords")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: CoT Prompting (per comparison)
    print(f"\n" + "="*80)
    print("TEST 2: CoT Prompting (for comparison)")
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
        print("\n" + "="*80)
        print(response_2)
        print("="*80 + "\n")
        
        # Save response to file for full inspection
        with open("/mnt/beegfs/a.cardamone7/outputs/response_cot.txt", "w") as f:
            f.write(response_2)
        print(f"[DEBUG] Full response saved to /mnt/beegfs/a.cardamone7/outputs/response_cot.txt")
        
        # Check for bbox patterns
        import re
        bbox_pattern = r'\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]'
        if re.search(bbox_pattern, response_2):
            print("\n✓ FOUND BBOX COORDINATES!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
        This test shows whether the Qwen2.5 model INSIDE the InternVLA-M1 checkpoint
        can generate bounding boxes when prompted appropriately.

        If it does → The wrapper approach is correct (use Qwen from checkpoint)
        If it doesn't → The model was fine-tuned to ignore bbox generation
    """)


if __name__ == "__main__":
    test_qwen_inside_checkpoint()
