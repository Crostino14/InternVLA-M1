#!/usr/bin/env python3
"""
InternVLA-M1 with Grounding Integration.

This module extends InternVLA-M1 to include grounding by:
1. Using Qwen2.5-VL to generate bounding boxes (via CoT prompt)
2. Using InternVLA-M1 to predict actions from images
3. Combining both outputs for complete scene understanding
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import re


class InternVLAGroundingWrapper:
    """Wraps InternVLA-M1 with Qwen2.5-VL grounding capability."""
    
    def __init__(
        self,
        m1_model,
        use_grounding: bool = True,
        grounding_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    ):
        """
        Initialize wrapper.
        
        Args:
            m1_model: InternVLA_M1 instance
            use_grounding: Whether to use Qwen for grounding
            grounding_model_id: Model ID for grounding
        """
        self.m1_model = m1_model
        self.use_grounding = use_grounding
        
        if use_grounding:
            print(f"[INFO] Loading grounding model: {grounding_model_id}")
            self.grounding_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                grounding_model_id,
                attn_implementation="flash_attention_2",
                torch_dtype="auto",
                device_map="cuda",
            )
            self.grounding_processor = AutoProcessor.from_pretrained(grounding_model_id)
            self.grounding_processor.tokenizer.padding_side = "left"
            self.grounding_model.eval()
        else:
            self.grounding_model = None
            self.grounding_processor = None
    
    def predict_action_with_grounding(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 5,
        **kwargs,
    ) -> Dict:
        """
        Predict actions WITH grounding bounding boxes.
        
        Args:
            batch_images: List of multi-view image lists
            instructions: Task instructions
            cfg_scale: Classifier-free guidance scale
            use_ddim: Whether to use DDIM sampling
            num_ddim_steps: Number of DDIM steps
        
        Returns:
            {
                "normalized_actions": np.ndarray [B, T, 7],
                "grounding": {
                    "instruction": str,
                    "bboxes": List[Dict] (format: [{"bbox": [x1,y1,x2,y2], "label": str}, ...]),
                    "raw_response": str,
                }
            }
        """
        
        # 1. Generate actions using M1
        print("[INFO] Predicting actions with InternVLA-M1...")
        actions = self.m1_model.predict_action(
            batch_images=batch_images,
            instructions=instructions,
            cfg_scale=cfg_scale,
            use_ddim=use_ddim,
            num_ddim_steps=num_ddim_steps,
            **kwargs,
        )
        
        # 2. Generate grounding if enabled
        grounding_data = None
        if self.use_grounding:
            print("[INFO] Generating bounding boxes with Qwen VL...")
            grounding_data = self._extract_grounding(batch_images, instructions)
        
        result = {
            "normalized_actions": actions["normalized_actions"],
        }
        
        if grounding_data:
            result["grounding"] = grounding_data
        
        return result
    
    def _extract_grounding(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
    ) -> Dict:
        """Extract bounding boxes using Qwen2.5-VL with CoT prompt."""
        
        # CoT prompt for grounding
        cot_prompt_template = (
            "Your task is: {instruction}. "
            "To identify the target objects for your task, locate their bounding boxes "
            "in [x1,y1,x2,y2] format on a 224x224 image. "
            "Return the result as a JSON list with 'bbox_2d' and 'label' fields. "
            "Example: [{{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"object_name\"}}]"
        )
        
        all_bboxes = []
        
        for images, instruction in zip(batch_images, instructions):
            cot_prompt = cot_prompt_template.format(instruction=instruction)
            
            message = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img} for img in images
                ] + [{"type": "text", "text": cot_prompt}]
            }]
            
            text = self.grounding_processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info([message])
            
            inputs = self.grounding_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            
            with torch.inference_mode():
                output_ids = self.grounding_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                )
            
            generated_text = self.grounding_processor.decode(
                output_ids[0], skip_special_tokens=True
            )
            
            # Extract JSON from response
            bboxes = self._parse_bbox_response(generated_text)
            all_bboxes.append({
                "instruction": instruction,
                "bboxes": bboxes,
                "raw_response": generated_text,
            })
        
        return {
            "detections": all_bboxes,
            "model": "Qwen2.5-VL-3B-Instruct",
        }
    
    @staticmethod
    def _parse_bbox_response(response: str) -> List[Dict]:
        """Extract bounding boxes from model response."""
        bboxes = []
        
        # Find JSON arrays in the response
        json_pattern = r'\[\s*\{[^}]*\}\s*(?:,\s*\{[^}]*\})*\s*\]'
        matches = re.findall(json_pattern, response)
        
        if matches:
            try:
                json_str = matches[0]
                parsed = json.loads(json_str)
                
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            bbox_key = next(
                                (k for k in item.keys() if 'bbox' in k.lower()),
                                None
                            )
                            if bbox_key:
                                bboxes.append({
                                    "bbox": item[bbox_key],
                                    "label": item.get("label", "object"),
                                })
            except json.JSONDecodeError:
                pass
        
        return bboxes


def create_model_with_grounding(
    m1_model_path: str,
    use_grounding: bool = True,
) -> InternVLAGroundingWrapper:
    """Factory function to create InternVLA-M1 with grounding."""
    from InternVLA.model.framework.M1 import InternVLA_M1
    
    print(f"[INFO] Loading InternVLA-M1 from {m1_model_path}...")
    m1_model = InternVLA_M1.from_pretrained(m1_model_path)
    m1_model = m1_model.to("cuda").eval()
    
    wrapper = InternVLAGroundingWrapper(m1_model, use_grounding=use_grounding)
    return wrapper
