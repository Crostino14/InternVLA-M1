#!/usr/bin/env python3
"""
internvla_vqa_eval.py
---------------------
VQA evaluation script for InternVLA-M1 (Qwen2.5-VL-3B-Instruct backbone).
"""
import argparse
import json
import logging
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Color palette for bounding boxes
BBOX_COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 204, 55), (36, 204, 222), (36, 121, 255),
    (122, 84, 255), (180, 84, 255),
]

# ---------------------------------------------------------------------------
# Image Preprocessing & BBox Drawing
# ---------------------------------------------------------------------------
def extract_first_frame(video_path: str) -> np.ndarray:
    """Extracts the first frame from a video file."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError(f"Failed to read the first frame from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def parse_bbox_from_text(generated_text: str) -> List[Tuple[str, List[int]]]:
    """Parses bounding box data from the model's generated text."""
    pattern = r'<ref>(.*?)</ref><box>\[\[(\d+),(\d+),(\d+),(\d+)\]\]</box>'
    matches = re.findall(pattern, generated_text)
    results = []
    for match in matches:
        name, x1, y1, x2, y2 = match
        results.append((name.strip(), [int(x1), int(y1), int(x2), int(y2)]))
    return results

def convert_bbox_coordinates(box: List[int], width: int, height: int) -> List[int]:
    """Converts normalized [0, 999] coordinates to pixel coordinates."""
    x1, y1, x2, y2 = box
    px_x1 = int(x1 / 999 * width)
    px_y1 = int(y1 / 999 * height)
    px_x2 = int(x2 / 999 * width)
    px_y2 = int(y2 / 999 * height)
    return [px_x1, px_y1, px_x2, px_y2]

def draw_bboxes_on_image(
    image: Image.Image,
    bboxes: List[Tuple[str, List[int]]],
    object_list: List[str]
) -> Image.Image:
    """Draws bounding boxes and labels on an image."""
    draw = ImageDraw.Draw(image)
    
    # Create a color map for objects in this specific image
    color_map = {obj.lower(): BBOX_COLORS[i % len(BBOX_COLORS)] for i, obj in enumerate(object_list)}

    try:
        font = ImageFont.truetype("dejavu-sans-mono.ttf", size=12)
    except IOError:
        font = ImageFont.load_default()

    for name, box in bboxes:
        color = color_map.get(name.lower(), (255, 255, 255)) # Default to white if not in list
        pixel_box = convert_bbox_coordinates(box, image.width, image.height)
        draw.rectangle(pixel_box, outline=color, width=3)
        
        text_position = (pixel_box[0], pixel_box[1] - 12)
        if text_position[1] < 0:
            text_position = (pixel_box[0], pixel_box[3])
            
        draw.text(text_position, name, fill=color, font=font)
        
    return image

# ---------------------------------------------------------------------------
# VQA Model Class
# ---------------------------------------------------------------------------
class VQAModel:
    def __init__(self, model_path: str, checkpoint_path: str = None):
        log.info(f"Loading processor from: {model_path}")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        log.info(f"Loading model from: {model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
        )

        if checkpoint_path and Path(checkpoint_path).exists():
            log.info(f"Loading fine-tuned weights from: {checkpoint_path}")
            try:
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                # The checkpoint might contain keys for other components (e.g., action expert)
                # We use strict=False to only load the VLM weights.
                load_result = self.model.load_state_dict(state_dict, strict=False)
                
                if load_result.missing_keys:
                    log.warning(f"Missing keys: {load_result.missing_keys}")
                if load_result.unexpected_keys:
                    log.warning(f"Unexpected keys: {load_result.unexpected_keys}")
                log.info("Successfully loaded checkpoint weights.")

            except Exception as e:
                log.error(f"Failed to load checkpoint: {e}")
                raise

    @torch.inference_mode()
    def answer(self, image: Image.Image, prompt: str, max_new_tokens: int = 128) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to("cuda")

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,       # penalizza token già visti
            no_repeat_ngram_size=6,       # vieta n-grammi di 6 token ripetuti
            temperature=None,             # disabilita esplicitamente (evita il warning)
            top_p=None,
            top_k=None,
        )

        input_token_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[:, input_token_len:]
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        generated_text = generated_text.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
        return generated_text

# ---------------------------------------------------------------------------
# Evaluation Logic
# ---------------------------------------------------------------------------
def humanize_name(value: str) -> str:
    """Convert internal MuJoCo-like object names to readable labels."""
    if not isinstance(value, str):
        return value
    x = value.strip().lower()
    if not x:
        return x
    x = x.replace("_main", "")
    x = re.sub(r"_\d+\b", "", x)
    x = x.replace("_", " ")
    x = " ".join(x.split())
    if x in {"flat stove", "flat stove 1"}:
        return "stove"
    return x

def score_object_listing(generated: str, expected: List[str]) -> float:
    """Calculates recall for object listing questions."""
    gen_list = {humanize_name(s.strip()) for s in generated.split(',') if s.strip()}
    exp_list = {humanize_name(s.strip()) for s in expected if s.strip()}
    
    if not exp_list:
        return 1.0 if not gen_list else 0.0
        
    matches = gen_list.intersection(exp_list)
    recall = len(matches) / len(exp_list)
    return recall

def main(args):
    log.info(f"Starting InternVLA VQA evaluation. Job ID: {os.getenv('SLURM_JOB_ID', 'N/A')}")
    
    # Determine if model_path is a local path or a Hub ID
    model_path_or_id = args.model_path
    if not Path(model_path_or_id).exists():
        log.info(f"'{model_path_or_id}' not found locally. Assuming it's a Hugging Face Hub ID.")
    else:
        log.info(f"Found local model path: '{model_path_or_id}'")

    log.info(f"Loading prompts from: {args.prompts_json}")
    
    with open(args.prompts_json, 'r') as f:
        benchmark_data = json.load(f)

    model = VQAModel(model_path_or_id, args.checkpoint_path)
    
    results = {"metadata": benchmark_data["metadata"], "tasks": []}
    
    # Metrics accumulators
    metrics = {
        "type0": {"correct": 0, "total": 0},
        "type1": {"correct": 0, "total": 0},
        "inv_type1": {"correct": 0, "total": 0},
        "type2": {"correct": 0, "total": 0},
        "type3": {"total_recall": 0, "total": 0, "per_task": {}},
    }

    os.makedirs(args.output_bbox_dir, exist_ok=True)

    for task_idx, task_data in enumerate(tqdm(benchmark_data["tasks"], desc="Tasks")):
        task_name = task_data["task_name"]
        task_results = deepcopy(task_data)
        task_results["questions"] = []

        frame_path = Path(task_data["first_frame_path"])
        if not frame_path.exists():
            log.warning(f"Frame not found for task '{task_name}': {frame_path}. Skipping task.")
            continue
            
        image = Image.open(frame_path).convert("RGB")
        
        # --- Type-3 Grounding Question (New) ---
        objects_in_scene = [humanize_name(o) for o in task_data.get("objects_in_scene", [])]
        objects_str = ", ".join(objects_in_scene)
        
        type3_prompt = (
            f"Detect the following objects: {objects_str}, in this image and provide their bounding boxes. "
            "For each detected object use EXACTLY this format: "
            f"<ref>object_name</ref><box>[[x1,y1,x2,y2]]</box>\n"
            f"Detect: {objects_str}"
        )
        
        log.info(f"[{task_name}_q_grounding] Running Type-3 grounding...")
        generated_text = model.answer(image, type3_prompt, max_new_tokens=512)
        
        detected_boxes = parse_bbox_from_text(generated_text)
        
        if not detected_boxes:
            log.warning(f"[{task_name}_q_grounding] No <ref>/<box> tags found in model output.")
        
        detected_names = {name.lower() for name, _ in detected_boxes}
        expected_names = {name.lower() for name in objects_in_scene}
        
        grounding_recall = len(detected_names.intersection(expected_names)) / len(expected_names) if expected_names else 0.0
        
        metrics["type3"]["total_recall"] += grounding_recall
        metrics["type3"]["total"] += 1
        metrics["type3"]["per_task"][task_name] = {
            "recall": grounding_recall,
            "detected": list(detected_names),
            "missing": list(expected_names - detected_names),
            "raw_output": generated_text,
        }
        
        log.info(f"[{task_name}_q_grounding] Recall: {grounding_recall:.2f} "
                 f"({len(detected_names.intersection(expected_names))}/{len(expected_names)})")

        # Save annotated image
        annotated_image = draw_bboxes_on_image(image.copy(), detected_boxes, objects_in_scene)
        bbox_task_dir = Path(args.output_bbox_dir) / f"task{task_idx}"
        bbox_task_dir.mkdir(exist_ok=True)
        save_path = bbox_task_dir / f"{task_name}_bbox.png"
        annotated_image.save(save_path)
        log.info(f"Saved bbox image to {save_path}")

        # --- Existing Question Types ---
        for q_idx, q_data in enumerate(tqdm(task_data["questions"], desc=f"Questions for {task_name}", leave=False)):
            q_type = q_data["type"]
            prompt = q_data["filled_prompt"]
            q_id = q_data["question_id"]
            
            time.sleep(args.question_delay_sec)
            
            answer = model.answer(image, prompt)
            
            q_result = deepcopy(q_data)
            q_result["generated_answer"] = answer
            
            correct = False
            if q_type == "type0_object_listing":
                gt = q_data["ground_truth"]["type0_object_listing"]
                recall = score_object_listing(answer, gt["expected_objects"])
                correct = (recall >= args.similarity_threshold)
                q_result["recall"] = recall
                metrics["type0"]["correct"] += int(correct)
                metrics["type0"]["total"] += 1

            elif q_type == "type1_spatial_relation":
                gt = q_data["ground_truth"].get("type1_spatial_relation", {})
                
                # Supporta sia 'relation_label' che varianti alternative nel JSON
                relation_label = (
                    gt.get("relation_label") or
                    gt.get("label") or
                    gt.get("answer") or
                    gt.get("relation") or
                    gt.get("expected_answer")
                )
                
                if relation_label is None:
                    log.warning(f"[{q_id}] Struttura ground truth inattesa: {gt}. Skip.")
                    correct = False
                else:
                    correct = relation_label.lower() in answer.lower()
                
                metrics["type1"]["correct"] += int(correct)
                metrics["type1"]["total"] += 1

                # Inverse relation - stessa logica difensiva
                inv_gt = q_data["ground_truth"].get("inv_type1_spatial_relation", {})
                inv_prompt = inv_gt.get("filled_prompt")
                if inv_prompt:
                    inv_answer = model.answer(image, inv_prompt)
                    inv_label = (
                        inv_gt.get("relation_label") or
                        inv_gt.get("label") or
                        inv_gt.get("answer")
                    )
                    inv_correct = (inv_label.lower() in inv_answer.lower()) if inv_label else False
                    metrics["inv_type1"]["correct"] += int(inv_correct)
                    metrics["inv_type1"]["total"] += 1
                    q_result["inv_generated_answer"] = inv_answer
                    q_result["inv_correct"] = inv_correct

            elif q_type == "type2_object_identification":
                gt = q_data["ground_truth"]["type2_object_identification"]
                correct = gt["target_object"].lower() in answer.lower()
                metrics["type2"]["correct"] += int(correct)
                metrics["type2"]["total"] += 1

            q_result["correct"] = correct
            log.info(f"[{q_id}] {'✓' if correct else '✗'} | GT: '{q_data.get('ground_truth', {}).get(q_type, {}).get('relation_label') or q_data.get('ground_truth', {}).get(q_type, {}).get('target_object')}' | Gen: '{answer}'")
            task_results["questions"].append(q_result)
        
        results["tasks"].append(task_results)
        time.sleep(args.task_delay_sec)

    # --- Final Metrics ---
    results["final_metrics"] = {
        "type0_object_listing": f"{metrics['type0']['correct']}/{metrics['type0']['total']} = {metrics['type0']['correct']/metrics['type0']['total']:.2%}" if metrics['type0']['total'] > 0 else "N/A",
        "type1_spatial_relation": f"{metrics['type1']['correct']}/{metrics['type1']['total']} = {metrics['type1']['correct']/metrics['type1']['total']:.2%}" if metrics['type1']['total'] > 0 else "N/A",
        "inv_type1_spatial_relation": f"{metrics['inv_type1']['correct']}/{metrics['inv_type1']['total']} = {metrics['inv_type1']['correct']/metrics['inv_type1']['total']:.2%}" if metrics['inv_type1']['total'] > 0 else "N/A",
        "type2_object_identification": f"{metrics['type2']['correct']}/{metrics['type2']['total']} = {metrics['type2']['correct']/metrics['type2']['total']:.2%}" if metrics['type2']['total'] > 0 else "N/A",
        "type3_grounding_recall": f"{metrics['type3']['total_recall']/metrics['type3']['total']:.2%}" if metrics['type3']['total'] > 0 else "N/A",
    }
    results["type3_grounding"] = {
        "per_task": metrics["type3"]["per_task"],
        "mean_recall": metrics["type3"]["total_recall"] / metrics["type3"]["total"] if metrics["type3"]["total"] > 0 else 0.0
    }

    log.info("\n" + "="*50 + "\nFINAL METRICS\n" + "="*50)
    log.info(f"Type-0 (object listing)       : {results['final_metrics']['type0_object_listing']}")
    log.info(f"Type-1 (spatial relation)     : {results['final_metrics']['type1_spatial_relation']}")
    log.info(f"Type-1 (inverse relation)     : {results['final_metrics']['inv_type1_spatial_relation']}")
    log.info(f"Type-2 (object identification): {results['final_metrics']['type2_object_identification']}")
    log.info(f"Type-3 (grounding recall)     : {results['final_metrics']['type3_grounding_recall']}")
    log.info("="*50)

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InternVLA-M1 VQA Evaluation Script")
    parser.add_argument("--model_path", type=str, default="/mnt/beegfs/a.cardamone7/checkpoints/InternVLA-M1", help="Path al VLM spazialmente pre-addestrato (Stage 1).")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path al checkpoint Action Expert. None per VQA evaluation.")
    parser.add_argument("--prompts_json", type=str, default="/mnt/beegfs/a.cardamone7/outputs/vqa_test/spatial_benchmark.json", help="Path to the JSON file with prompts.")
    parser.add_argument("--output_json", type=str, default="./vqa_results_internvla.json", help="Path to save the output JSON results.")
    parser.add_argument("--output_bbox_dir", type=str, default="./output/bounding_boxes", help="Directory to save images with bounding boxes.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens for generation.")
    parser.add_argument("--question_delay_sec", type=float, default=1.0, help="Delay between questions.")
    parser.add_argument("--task_delay_sec", type=float, default=2.0, help="Delay between tasks.")
    parser.add_argument("--similarity_threshold", type=float, default=0.80, help="Similarity threshold for Type-0 recall.")
    
    args = parser.parse_args()
    main(args)
