import sys
import os
import logging
import glob
import re
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['DEVICE'] = "cuda"


import torch
import cv2
import numpy as np
import time
import argparse
import tqdm
import json
import pickle
from copy import deepcopy
from typing import Optional
from dataclasses import dataclass
from enum import Enum
from PIL import Image


import draccus


from InternVLA.model.framework.M1 import InternVLA_M1


from libero.libero import benchmark
from utils.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from utils.robot_utils import set_seed_everywhere, set_env_seed, DATE_TIME


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)



class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT  = "libero_object"
    LIBERO_GOAL    = "libero_goal"
    LIBERO_10      = "libero_10"
    LIBERO_90      = "libero_90"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT:  280,
    TaskSuite.LIBERO_GOAL:    300,
    TaskSuite.LIBERO_10:      520,
    TaskSuite.LIBERO_90:      400,
}

SPATIAL_COT_PROMPT = (
    "Your task is {instruction}. First, identify key objects and their positions using spatial relations: "
    "left/right of, in front/behind, next to, above/below. Then execute the action."
)

def log_message(message: str, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def _to_jsonable(obj):
    """Recursively convert objects to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, Image.Image):
        return f"<PIL.Image mode={obj.mode} size={obj.size}>"
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _extract_bbox_like_fields(pred_dict: dict) -> dict:
    """
    Best-effort extraction of grounding outputs from model predictions.
    Keeps only keys that look like bbox/grounding fields.
    """
    bbox_like = {}
    if not isinstance(pred_dict, dict):
        return bbox_like

    candidate_tokens = ("bbox", "box", "region", "ground", "loc")
    for key, value in pred_dict.items():
        k = str(key).lower()
        if any(tok in k for tok in candidate_tokens):
            bbox_like[str(key)] = _to_jsonable(value)
    return bbox_like


def _iter_bbox_arrays(obj, prefix=""):
    """Yield (path, np.ndarray) for entries that look like bbox tensors [..., 4]."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            next_prefix = f"{prefix}.{k}" if prefix else str(k)
            yield from _iter_bbox_arrays(v, next_prefix)
        return

    if isinstance(obj, (list, tuple)):
        try:
            arr = np.asarray(obj)
            if arr.ndim >= 1 and arr.shape[-1] == 4 and np.issubdtype(arr.dtype, np.number):
                yield (prefix or "boxes", arr.astype(np.float32))
                return
        except Exception:
            pass
        for i, v in enumerate(obj):
            next_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
            yield from _iter_bbox_arrays(v, next_prefix)
        return

    if isinstance(obj, np.ndarray):
        if obj.ndim >= 1 and obj.shape[-1] == 4 and np.issubdtype(obj.dtype, np.number):
            yield (prefix or "boxes", obj.astype(np.float32))


def _to_pixel_xyxy(box_xyxy, width: int, height: int, model_img_size: int = 224):
    """Convert bbox [x1,y1,x2,y2] to pixel coordinates using simple scale heuristics."""
    b = np.asarray(box_xyxy, dtype=np.float32).reshape(-1)
    if b.size != 4:
        return None

    x1, y1, x2, y2 = b.tolist()
    vals = np.array([x1, y1, x2, y2], dtype=np.float32)

    if np.all(np.isfinite(vals)):
        if np.max(np.abs(vals)) <= 1.05:
            x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height
        elif np.max(np.abs(vals)) <= float(model_img_size) * 1.2:
            sx = width / float(model_img_size)
            sy = height / float(model_img_size)
            x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy

    x1, x2 = sorted([int(round(x1)), int(round(x2))])
    y1, y2 = sorted([int(round(y1)), int(round(y2))])

    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _draw_bboxes_on_image(image: np.ndarray, bbox_entries):
    """Draw bbox entries on a copy of image. bbox_entries is list[(label, xyxy)]."""
    out = image.copy()
    h, w = out.shape[:2]
    color = (0, 255, 0)
    text_color = (255, 255, 255)

    drawn = 0
    for label, box in bbox_entries:
        pix = _to_pixel_xyxy(box, w, h)
        if pix is None:
            continue
        x1, y1, x2, y2 = pix
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            label,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            text_color,
            1,
            cv2.LINE_AA,
        )
        drawn += 1
    return out, drawn


def save_vlm_bbox_frame(frame_dir: str, timestep: int, agentview: np.ndarray,
                        wrist: np.ndarray, pred_meta: dict) -> int:
    """Save side-by-side frame with bbox overlays. Returns number of drawn boxes."""
    bbox_dict = pred_meta.get("bbox_like", {}) if isinstance(pred_meta, dict) else {}

    bbox_entries = []
    if bbox_dict:
        for path, arr in _iter_bbox_arrays(bbox_dict):
            flat = np.asarray(arr, dtype=np.float32).reshape(-1, 4)
            for i, row in enumerate(flat):
                bbox_entries.append((f"{path}[{i}]", row))

    agent_annot, n_agent = _draw_bboxes_on_image(agentview, bbox_entries)
    wrist_annot, n_wrist = _draw_bboxes_on_image(wrist, bbox_entries)

    header_h = 28
    canvas_h = max(agent_annot.shape[0], wrist_annot.shape[0]) + header_h
    canvas_w = agent_annot.shape[1] + wrist_annot.shape[1]
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    canvas[header_h:header_h + agent_annot.shape[0], :agent_annot.shape[1]] = agent_annot
    canvas[header_h:header_h + wrist_annot.shape[0], agent_annot.shape[1]:] = wrist_annot

    title = f"t={timestep} | boxes={min(n_agent, n_wrist)}"
    cv2.putText(canvas, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (agent_annot.shape[1], header_h), (agent_annot.shape[1], canvas_h - 1), (100, 100, 100), 1)

    os.makedirs(frame_dir, exist_ok=True)
    out_path = os.path.join(frame_dir, f"frame_t{timestep:04d}.png")
    ok = cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for path: {out_path}")
    return min(n_agent, n_wrist)



# ─── InternVLA-M1 policy wrapper ─────────────────────────────────────────────


class InternVLA_M1_policy:
    """Policy wrapper for InternVLA-M1."""

    def __init__(self, model_path: str, device: str = "cuda", use_cot: bool = True):
        from InternVLA.model.framework.share_tools import read_mode_config
        from omegaconf import OmegaConf

        log_message(f"Loading InternVLA-M1 from {model_path}")

        run_dir     = os.path.dirname(os.path.dirname(model_path))
        config_path = os.path.join(run_dir, "config.yaml")
        cfg = OmegaConf.load(config_path)

        self.model = InternVLA_M1(config=cfg)

        state_dict = torch.load(model_path, map_location="cpu")
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        log_message(f"Loaded checkpoint with {len(missing)} missing keys and {len(unexpected)} unexpected keys")
        if missing:
            print(f"⚠️ MISSING KEYS (first 10): {missing[:10]}")
        if unexpected:
            print(f"⚠️ UNEXPECTED KEYS (first 10): {unexpected[:10]}")

        self.model = self.model.to(device).eval()
        self.device = device
        self.use_cot = use_cot

        model_config, norm_stats = read_mode_config(model_path)
        
        EMBODIMENT_KEY = "franka"
        if EMBODIMENT_KEY not in norm_stats:
            fallback = "new_embodiment" if "new_embodiment" in norm_stats else list(norm_stats.keys())[0]
            log_message(f"WARNING: '{EMBODIMENT_KEY}' not in norm_stats. "
                        f"Available: {list(norm_stats.keys())}. Using '{fallback}' as fallback.")
            EMBODIMENT_KEY = fallback

        # Unnormalization stats
        action_stats      = norm_stats[EMBODIMENT_KEY]["action"]
        self.action_mask  = np.array(action_stats["mask"], dtype=bool)
        self.action_high  = np.array(action_stats["max"],  dtype=np.float32)
        self.action_low   = np.array(action_stats["min"],  dtype=np.float32)

        # Chunk size da config
        self.chunk_size = (
            model_config["framework"]["action_model"]["future_action_window_size"] + 1
        )
        log_message(f"InternVLA-M1 loaded. chunk_size={self.chunk_size}")

    def predict(self, agentview_img, wrist_img, instruction,
                cfg_scale=1.5, num_ddim_steps=10, return_metadata: bool = False):
        
        if self.use_cot:
            instruction = SPATIAL_COT_PROMPT.replace("{instruction}", instruction)
        
        print("=== PREDICTION INSTRUCTION ===")
        print(instruction)
        
        #proc = self.model.qwen_vl_interface.processor
        #dummy_msg = [{"role": "user", "content": [
        #    {"type": "image", "image": Image.fromarray(agentview_img)},
        #    {"type": "image", "image": Image.fromarray(wrist_img)},
        #    {"type": "text",  "text": instruction}
        #]}]
        #dummy_text   = proc.apply_chat_template(dummy_msg, tokenize=False, add_generation_prompt=True)
        #n_img_tokens = dummy_text.count("<|image_pad|>")
        #print(f"[DEBUG] image_tokens={n_img_tokens} (attesi: ≥2) | '{instruction}'", flush=True)

        view1 = Image.fromarray(agentview_img)
        view2 = Image.fromarray(wrist_img)
        with torch.inference_mode():
            pred = self.model.predict_action(
                batch_images=[[view1, view2]],      # due viste come nel training
                instructions=[instruction],
                cfg_scale=cfg_scale,
                use_ddim=True,
                num_ddim_steps=num_ddim_steps,
            )
        #print(f"[DEBUG] pred keys: {list(pred.keys())}", flush=True)
        normalized = np.clip(pred["normalized_actions"][0], -1, 1)  # [T, 7]
        #print(f"[DEBUG] normalized actions mean={normalized.mean():.4f} std={normalized.std():.4f} min={normalized.min():.4f} max={normalized.max():.4f}")
        #print(f"[DEBUG] action_low={self.action_low}, action_high={self.action_high}")
        #print(f"[DEBUG] action_mask={self.action_mask}")
        
        #img_hash = int(np.sum(agentview_img.astype(np.int64)) % 1e9)

        # Gripper (dim 6): threshold sul normalized raw — come nel codice ufficiale
        normalized[:, 6] = np.where(normalized[:, 6] < 0.5, 0.0, 1.0)

        # Unnormalizzazione min/max — identica a M1Inference.unnormalize_actions
        actions = np.where(
            self.action_mask,
            0.5 * (normalized + 1) * (self.action_high - self.action_low) + self.action_low,
            normalized,
        )
        #print(f"[DEBUG] final actions={actions[0]}")
        if not return_metadata:
            return actions  # [T, 7], gripper in {0.0, 1.0}

        metadata = {
            "pred_keys": list(pred.keys()) if isinstance(pred, dict) else [],
            "bbox_like": _extract_bbox_like_fields(pred),
        }
        return actions, metadata



# ─── Observation helper ───────────────────────────────────────────────────────


def get_obs_internvla(obs):
    """
    Ritorna (agentview, wrist) come uint8 RGB array a 224x224.
    Il doppio flip [::-1, ::-1] è confermato dal codice ufficiale InternVLA-M1
    per allineare l'orientamento OpenGL con il preprocessing del training.
    """
    agentview = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
    wrist      = np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1])
    agentview  = cv2.resize(agentview, (224, 224))
    wrist      = cv2.resize(wrist,     (224, 224))
    return agentview.astype(np.uint8), wrist.astype(np.uint8)


def binarize_gripper_for_robosuite(gripper_val: float) -> float:
    """
    Converte il gripper unnormalizzato {0.0, 1.0} in delta_qpos robosuite:
      0.0 (close) → +1.0
      1.0 (open)  → -1.0
    Identico a _binarize_gripper_open nel codice ufficiale.
    """
    return 1.0 - 2.0 * float(gripper_val > 0.5)


def extract_eef_state(obs) -> np.ndarray:
    """Extract end-effector xyz position from robosuite observation dict."""
    candidate_keys = [
        "robot0_eef_pos",
        "eef_pos",
        "robot0_gripper_pos",
    ]
    for key in candidate_keys:
        if key in obs:
            eef = np.asarray(obs[key], dtype=np.float32).reshape(-1)
            if eef.size >= 3:
                return eef[:3]
    raise KeyError(
        f"Could not find end-effector position in obs. Tried keys={candidate_keys}. "
        f"Available keys={list(obs.keys())}"
    )


def _parse_levels(cfg) -> list:
    if not cfg.change_command or not cfg.command_level:
        return [None]
    presets = {
        "all":            [None, "l1", "l2", "l3"],
        "all_no_default": ["l1", "l2", "l3"],
        "default":        [None],
    }
    if cfg.command_level in presets:
        return presets[cfg.command_level]
    if "," in cfg.command_level:
        return [l.strip() for l in cfg.command_level.split(",")]
    return [cfg.command_level]


# ─── Episode runner ───────────────────────────────────────────────────────────


def run_episode(
    cfg,
    env,
    task_description: str,
    policy: InternVLA_M1_policy,
    initial_state=None,
    collect_vlm_bbox: bool = False,
    frame_save_dir: Optional[str] = None,
    log_file=None,
):
    env.reset()

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.reset()

    # Aspetta stabilizzazione fisica
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action("tiny_vla"))

    max_timesteps = TASK_MAX_STEPS.get(cfg.task_suite_name, 300)
    chunk_size    = policy.chunk_size   # da config (es. 8)
    query_freq    = chunk_size          # query ogni chunk_size step (no temporal agg)

    # Use task_description directly without grounding
    image_list, action_list, state_list = [], [], []
    success = False
    current_chunk = None

    try:
        for t in range(max_timesteps):
            agentview, wrist = get_obs_internvla(obs)
            image_list.append(agentview.copy())
            state_list.append(extract_eef_state(obs).copy())

            # Query policy ogni chunk_size step
            if t % query_freq == 0:
                if collect_vlm_bbox:
                    current_chunk, pred_meta = policy.predict(
                        agentview,
                        wrist,
                        task_description,
                        return_metadata=True,
                    )
                    #if frame_save_dir:
                    #    drawn = save_vlm_bbox_frame(frame_save_dir, t, agentview, wrist, pred_meta)
                    #    if drawn > 0:
                    #        log_message(f"Saved bbox frame @ t={t} with {drawn} boxes", log_file)
                    #    else:
                    #        log_message(
                    #            f"Saved frame @ t={t} without bbox (model keys={pred_meta.get('pred_keys', [])})",
                    #            log_file,
                    #        )
                else:
                    current_chunk = policy.predict(
                                                    agentview, wrist, task_description,
                                                    cfg_scale=cfg.cfg_scale,
                                                    num_ddim_steps=cfg.num_ddim_steps,
                                                    )
                # current_chunk: [chunk_size, 7]

            # Esegui l'azione corrente nel chunk
            action_raw = current_chunk[t % chunk_size]          # [7]

            # Converti gripper in formato delta_qpos robosuite
            gripper_env = binarize_gripper_for_robosuite(action_raw[6])
            env_action  = np.concatenate([action_raw[:6], [gripper_env]])

            obs, _, done, _ = env.step(env_action.tolist())
            action_list.append(env_action)

            if done:
                success = True
                break

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)
        success = False

    replay_traj = {
        "images":       image_list,
        "states":       state_list,
        "task_command": task_description,
        "actions":      action_list,
    }
    return success, replay_traj



# ─── Task runner ─────────────────────────────────────────────────────────────


def run_task(cfg, task_suite, task_id, policy, log_file,
             total_episodes=0, total_successes=0, run_id: Optional[str] = None):

    task           = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)

    # Guardia: selected_version richiede change_command + command_level
    if cfg.selected_version is not None:
        assert cfg.change_command and cfg.command_level is not None, (
            f"selected_version={cfg.selected_version} richiede "
            f"change_command=True e command_level non-None"
        )

    # ── Discovery delle versioni disponibili ──────────────────────────────
    available_versions = []
    has_base_syn = False
    base_syn_filename = None

    if cfg.change_command and cfg.command_level is not None:
        base_name = task.bddl_file.replace(".bddl", "")
        try:
            from libero.libero import get_libero_path
            bddl_folder = os.path.join(get_libero_path("bddl_files"), task.problem_folder)
        except Exception:
            bddl_folder = os.path.dirname(task.bddl_file)

        # file base: *_syn_l3.bddl
        base_syn_file = f"{base_name}_syn_{cfg.command_level}.bddl"
        base_syn_path = os.path.join(bddl_folder, os.path.basename(base_syn_file))
        if os.path.isfile(base_syn_path):
            has_base_syn = True
            base_syn_filename = os.path.basename(base_syn_path)
            log_message(f"Found base syn file: {base_syn_path}", log_file)
        else:
            log_message(f"Warning: base syn file not found: {base_syn_path}", log_file)

        # file versionati: *_syn_l3_v1.bddl, *_syn_l3_v2.bddl, ...
        pattern = f"{base_name}_syn_{cfg.command_level}_v"
        try:
            for filename in os.listdir(bddl_folder):
                if pattern.lower() in filename.lower() and filename.endswith(".bddl"):
                    match = re.search(r"_v(\d+)", filename, re.IGNORECASE)
                    if match:
                        available_versions.append((int(match.group(1)), filename))
        except Exception as e:
            log_message(f"Warning: Could not list version files: {e}", log_file)

        available_versions.sort()

    elif cfg.change_command and cfg.command_level is not None and not cfg.use_versions:
        # cerca il file base _syn_l3.bddl senza suffisso _vN
        base_name = task.bddl_file.replace('.bddl', '')
        try:
            from libero.libero import get_libero_path
            bddl_folder = os.path.join(get_libero_path("bddl_files"), task.problem_folder)
        except Exception:
            bddl_folder = os.path.dirname(task.bddl_file)

        base_syn_file = f"{base_name}_syn_{cfg.command_level}.bddl"
        base_syn_path = os.path.join(bddl_folder, os.path.basename(base_syn_file))
        if os.path.isfile(base_syn_path):
            # usa -1 come sentinel per "file base senza versione"
            available_versions = [(-1, os.path.basename(base_syn_path))]
            log_message(f"Found base syn file: {base_syn_path}", log_file)
        else:
            log_message(f"Warning: base syn file not found: {base_syn_path}", log_file)

    # ── Versioni da testare ───────────────────────────────────────────────
    # -1  -> file base *_syn_l3.bddl
    # >=0 -> file versionato *_syn_l3_vN.bddl
    if cfg.selected_version is not None:
        versions_to_test = [cfg.selected_version]
    else:
        versions_to_test = []
        if has_base_syn:
            versions_to_test.append(-1)   # syn_l3
        versions_to_test.extend([v[0] for v in available_versions])  # l3_v*

        if not versions_to_test:
            versions_to_test = [None]

    log_message("=" * 80, log_file)
    log_message(f"TASK {task_id + 1}/{task_suite.n_tasks}", log_file)
    log_message(f"Versions to test: {versions_to_test}", log_file)
    log_message("=" * 80, log_file)

    task_results_per_version = {}
    task_canonical_description = None  # descrizione stabile (prima versione)

    # ── Loop versioni ─────────────────────────────────────────────────────
    for version_to_test in versions_to_test:
        ablation_bddl_file = None

        if version_to_test == -1:
            version_label = "syn_base"
            ablation_bddl_file = base_syn_filename
        elif version_to_test is not None:
            version_label = f"v{version_to_test}"
            selected_files = [v[1] for v in available_versions if v[0] == version_to_test]
            if selected_files:
                ablation_bddl_file = selected_files[0]
        else:
            version_label = "default"

        env, task_description, original_description = get_libero_env(
            task,
            "InternVLA-M1",
            change_command=cfg.change_command,
            command_level=cfg.command_level,
            ablation_bddl_file=ablation_bddl_file,
            resolution=cfg.env_img_res,
        )
        if task_canonical_description is None:
            # Keep a stable, printable task label across command versions.
            task_canonical_description = (
                original_description
                or task_description
                or getattr(task, "name", None)
                or f"task_{task_id + 1}"
            )
        log_message("=" * 80, log_file)
        log_message(f"Testing VERSION: {version_label}", log_file)
        log_message(f"Original Command:  {original_description}", log_file)
        log_message(f"Variation Command: {task_description}", log_file)
        log_message("=" * 80, log_file)

        task_episodes = task_successes = 0

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task),
                                     desc=f"Version {version_label}"):
            initial_state    = initial_states[episode_idx]
            collect_vlm_bbox = bool(cfg.save_vlm_bboxes and episode_idx == cfg.bbox_episode_index)
            level_name       = cfg.command_level if cfg.command_level is not None else "default"
            frame_save_dir   = os.path.join(
                cfg.bbox_frame_root, level_name,
                f"{task_id + 1}", f"episode_{episode_idx + 1}"
            ) if collect_vlm_bbox else None

            success, replay_traj = run_episode(
                cfg, env, task_description, policy,
                initial_state, collect_vlm_bbox, frame_save_dir, log_file,
            )

            task_episodes   += 1
            total_episodes  += 1
            if success:
                task_successes  += 1
                total_successes += 1

            save_rollout_video(
                replay_traj, total_episodes,
                success=success,
                task_description=task_description,
                log_file=log_file,
                dataset_name=cfg.task_suite_name,
                run=cfg.run_number,
                change_command=cfg.change_command,
                command_level=cfg.command_level,
            )
            log_message(
                f"[{version_label}] ep {episode_idx+1} | success={success} | "
                f"running: {total_successes}/{total_episodes} "
                f"({100*total_successes/total_episodes:.1f}%)", log_file
            )

        # ── Risultati per versione ────────────────────────────────────────
        version_sr = task_successes / task_episodes if task_episodes > 0 else 0.0
        log_message(f"\n{'='*80}", log_file)
        log_message(f"VERSION {version_label} RESULTS:", log_file)
        log_message(f"  Episodes:     {task_episodes}", log_file)
        log_message(f"  Successes:    {task_successes}", log_file)
        log_message(f"  Success Rate: {version_sr:.1%}", log_file)
        log_message(f"{'='*80}\n", log_file)

        task_results_per_version[version_label] = {
            "success_rate": version_sr,
            "episodes":     task_episodes,
            "successes":    task_successes,
        }

        try:
            env.close()
        except Exception:
            pass

    # ── Aggregazione per-task (usa contatori locali, non cumulativi) ──────
    total_task_episodes  = sum(r["episodes"]  for r in task_results_per_version.values())
    total_task_successes = sum(r["successes"] for r in task_results_per_version.values())
    task_sr = (total_task_successes / total_task_episodes
               if total_task_episodes > 0 else 0.0)

    # ── Summary per versione ──────────────────────────────────────────────
    if len(versions_to_test) > 1:
        log_message("=" * 80, log_file)
        log_message("SUMMARY BY VERSION:", log_file)
        log_message("-" * 80, log_file)
        for vlabel, res in task_results_per_version.items():
            log_message(
                f"  {vlabel:>10}: {res['success_rate']:.1%} "
                f"({res['successes']}/{res['episodes']} episodes)", log_file
            )
        log_message("-" * 80, log_file)
        log_message(
            f"  {'OVERALL':>10}: {task_sr:.1%} "
            f"({total_task_successes}/{total_task_episodes} episodes)", log_file
        )
        log_message("=" * 80, log_file)
    else:
        log_message("=" * 80, log_file)
        log_message(
            f"TASK SUCCESS RATE: {task_sr:.1%} "
            f"({total_task_successes}/{total_task_episodes} episodes)", log_file
        )
        log_message("=" * 80, log_file)

    log_message(f"Task SR: {task_sr:.4f}", log_file)
    return total_episodes, total_successes, task_canonical_description, task_sr, total_task_episodes


# ─── Config ──────────────────────────────────────────────────────────────────


@dataclass
class GenerateConfig:
    # Model
    model_path: str = "/home/A.CARDAMONE7/checkpoints/InternVLA-M1-LIBERO-Goal/checkpoints/steps_30000_pytorch_model.pt"

    # LIBERO
    task_suite_name:     str = TaskSuite.LIBERO_GOAL
    num_steps_wait:      int = 10
    num_trials_per_task: int = 50
    env_img_res:         int = 256
    task_range:          str = "0-9"

    # Command variations
    change_command: bool          = False
    command_level:  Optional[str] = None
    selected_version:  Optional[int] = None
    use_versions: bool = True
    num_ddim_steps: int   = 10
    cfg_scale:      float = 1.5

    # Logging
    run_id_note:   Optional[str] = None
    local_log_dir: str           = "./experiments/logs"
    summary_file:  Optional[str] = None
    save_vlm_bboxes:  bool       = True
    bbox_episode_index: int      = 0
    bbox_frame_root:   str       = "/mnt/beegfs/a.cardamone7/outputs/rollouts/libero_goal/syntactic_variation/InternVLA/frame"

    seed:       int  = 7
    run_number: int  = 0
    debug:      bool = False
    use_cot: bool = False


# ─── Main ─────────────────────────────────────────────────────────────────────


@draccus.wrap()
def run_libero_eval(cfg: GenerateConfig):
    set_env_seed(cfg.seed)
    print("=== CONFIGURATION ===")
    print(f"Model Path: {cfg.model_path}")
    print(f"Use CoT: {cfg.use_cot}")
    policy = InternVLA_M1_policy(cfg.model_path, use_cot=cfg.use_cot)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict[cfg.task_suite_name]()

    task_start, task_end = map(int, cfg.task_range.split('-'))

    levels = _parse_levels(cfg)

    all_results  = {}
    task_results = {}

    for level in levels:
        level_name = level if level is not None else "default"
        cfg.command_level  = level
        cfg.change_command = (level is not None)
        set_env_seed(cfg.seed)

        run_id = (f"EVAL-{cfg.task_suite_name}-internvla_m1-{DATE_TIME}"
                  f"--{cfg.run_id_note or ''}-{level_name}")
        os.makedirs(cfg.local_log_dir, exist_ok=True)
        log_file = open(os.path.join(cfg.local_log_dir, run_id + ".txt"), "w")

        log_message(f"=== LEVEL: {level_name.upper()} ===", log_file)
        task_results[level_name] = {}
        total_episodes = total_successes = 0

        for task_id in tqdm.tqdm(
            range(task_start, min(task_end + 1, task_suite.n_tasks)),
            desc=f"Level {level_name}"
        ):
            total_episodes, total_successes, t_name, t_sr, t_eps = run_task(
                cfg, task_suite, task_id, policy, log_file,
                total_episodes, total_successes,
                run_id=run_id,
            )
            task_results[level_name][t_name] = {"success_rate": t_sr, "episodes": t_eps}

        final_sr = total_successes / total_episodes if total_episodes > 0 else 0.0
        all_results[level_name] = {
            "success_rate":    final_sr,
            "total_episodes":  total_episodes,
            "total_successes": total_successes,
        }
        log_message(f"LEVEL {level_name.upper()} SR: {final_sr:.4f}", log_file)

        if cfg.summary_file:
            with open(cfg.summary_file, "w") as f:
                json.dump({"task_results": task_results, "overall": all_results}, f, indent=2)

        log_file.close()

    # Tabella finale
    print("\n" + "=" * 100)
    print("DETAILED RESULTS TABLE")
    print("=" * 100)
    print(f"{'Task':<50} | {'Success Rate':>12} | {'Successes/Total':>15}")
    print("-" * 100)

    all_tasks = list(next(iter(task_results.values())).keys())
    for t in all_tasks:
        # Somma episodi e successi su tutti i level (o usa il primo se single-level)
        for l, res in task_results.items():
            sr  = res.get(t, {}).get("success_rate", 0.0)
            eps = res.get(t, {}).get("episodes", 0)
            sr = float(sr) if sr is not None else 0.0
            eps = int(eps) if eps is not None else 0
            succ = int(round(sr * eps))
            task_label = str(t) if t is not None else "unknown_task"
            print(f"{task_label:<50} | {sr:>11.1%} | {succ:>6}/{eps:<8}")

    print("-" * 100)

    # Riga OVERALL
    for l, res in all_results.items():
        overall_sr   = res["success_rate"]
        total_eps    = res["total_episodes"]
        total_succ   = res["total_successes"]
        print(f"{'OVERALL':<50} | {overall_sr:>11.1%} | {total_succ:>6}/{total_eps:<8}")

    print("=" * 100)



if __name__ == "__main__":
    run_libero_eval()