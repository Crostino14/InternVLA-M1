#!/usr/bin/env python3
"""
Inspect the InternVLA-M1 model structure to find the Qwen2.5-VL model inside.
"""

import os
import sys
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['DEVICE'] = "cuda"

sys.path.insert(0, "/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/InternVLA-M1")
sys.path.insert(0, "/home/A.CARDAMONE7/anaconda3/envs/internvla-m1/lib/python3.10/site-packages")

from InternVLA.model.framework.M1 import InternVLA_M1


def inspect_model():
    """Inspect model structure."""
    
    print("\n" + "="*80)
    print("Inspecting InternVLA-M1 Model Structure")
    print("="*80)
    
    # Load InternVLA-M1
    model_path = "/home/A.CARDAMONE7/checkpoints/InternVLA-M1-LIBERO-Goal/checkpoints/steps_30000_pytorch_model.pt"
    print(f"\n[INFO] Loading InternVLA-M1 from {model_path}...")
    model = InternVLA_M1.from_pretrained(model_path)
    
    print("\n" + "="*80)
    print("Model Attributes")
    print("="*80)
    
    # Get all attributes
    attributes = dir(model)
    print(f"\nTotal attributes: {len(attributes)}")
    
    # Filter non-dunder attributes
    relevant_attrs = [attr for attr in attributes if not attr.startswith('_')]
    print(f"Relevant attributes: {len(relevant_attrs)}")
    print("\nRelevant attributes:")
    for attr in sorted(relevant_attrs):
        try:
            val = getattr(model, attr)
            if not callable(val):
                print(f"  {attr}: {type(val).__name__}")
        except:
            pass
    
    print("\n" + "="*80)
    print("Named Modules (looking for Qwen)")
    print("="*80)
    
    for name, module in model.named_modules():
        if "qwen" in name.lower() or "vl" in name.lower():
            print(f"\n✓ Found: {name}")
            print(f"  Type: {type(module).__name__}")
            print(f"  Module: {module.__class__.__module__}.{module.__class__.__name__}")
    
    print("\n" + "="*80)
    print("All Named Modules")
    print("="*80)
    
    modules_dict = dict(model.named_modules())
    print(f"\nTotal modules: {len(modules_dict)}")
    
    print("\nTop-level modules:")
    for name in list(modules_dict.keys())[:20]:
        print(f"  {name}")
    
    print("\n" + "="*80)
    print("Looking for vision-language related modules")
    print("="*80)
    
    vision_modules = [name for name in modules_dict.keys() if 
                     any(keyword in name.lower() for keyword in 
                         ['qwen', 'vl', 'vision', 'language', 'llm', 'transformer'])]
    
    print(f"\nFound {len(vision_modules)} vision-language modules:")
    for name in vision_modules[:30]:
        module = modules_dict[name]
        print(f"  {name}: {type(module).__name__}")


if __name__ == "__main__":
    inspect_model()
