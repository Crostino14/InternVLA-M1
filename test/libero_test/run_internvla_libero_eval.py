"""Run InternVLA-M1 rollouts on LIBERO-Goal and report task success rate.

This script is the evaluation entry point for the InternVLA-M1 branch of the
thesis pipeline. It loads one checkpoint, runs closed-loop rollouts on LIBERO
tasks (including optional BDDL task variants for syntactic levels), and logs
task success rate per task and per variant. The code assumes the checkpoint was
already fine-tuned on the nonoops variant and only handles evaluation, not
training. It is used for syntactic generalization studies (L1/L2/L3) and for
baseline runs on the same task suite.
"""

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
    """Names of LIBERO benchmark suites accepted by this evaluator.

    Attributes:
        LIBERO_SPATIAL (str): Spatial subset in LIBERO.
        LIBERO_OBJECT (str): Object-focused subset in LIBERO.
        LIBERO_GOAL (str): LIBERO-Goal suite used in the thesis experiments.
        LIBERO_10 (str): Ten-task subset from LIBERO.
        LIBERO_90 (str): Ninety-task subset from LIBERO.
    """

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
    """Write one log line to console and to the optional run log.

    Args:
        message (str): Text to write.
        log_file (Optional[IO[str]]): File handle opened in write/append mode.

    Returns:
        None: This function only writes side effects.

    Raises:
        OSError: If file writing fails.
    """
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()

# ─── InternVLA-M1 policy wrapper ─────────────────────────────────────────────


class InternVLA_M1_policy:
    """Inference wrapper for InternVLA-M1 in LIBERO rollout evaluation.

    The class keeps the loaded model, action unnormalization stats, and action
    chunk size. `predict()` returns one action chunk conditioned on two camera
    views and one instruction.

    Attributes:
        model (InternVLA_M1): Model in eval mode on the selected device.
        device (str): Inference device, usually `"cuda"`.
        use_cot (bool): If true, prepends a spatial prompt to instructions.
        action_mask (np.ndarray): Boolean mask for unnormalization dimensions.
        action_high (np.ndarray): Per-dimension max values for action scaling.
        action_low (np.ndarray): Per-dimension min values for action scaling.
        chunk_size (int): Number of actions per action chunk.
    """

    def __init__(self, model_path: str, device: str = "cuda", use_cot: bool = True):
        """Load checkpoint and metadata needed for rollout-time inference.

        Args:
            model_path (str): Path to an InternVLA-M1 checkpoint (`.pt` file).
            device (str): Torch device string, e.g. `"cuda"` or `"cpu"`.
            use_cot (bool): Enables the spatial CoT prompt template.

        Returns:
            None: The constructor initializes instance attributes in place.

        Raises:
            FileNotFoundError: If checkpoint or inferred config file is missing.
            RuntimeError: If the checkpoint cannot be loaded into the model.
            KeyError: If normalization stats are missing expected keys.
        """
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

        # Stats used to map normalized outputs to end-effector delta (R^7).
        action_stats      = norm_stats[EMBODIMENT_KEY]["action"]
        self.action_mask  = np.array(action_stats["mask"], dtype=bool)
        self.action_high  = np.array(action_stats["max"],  dtype=np.float32)
        self.action_low   = np.array(action_stats["min"],  dtype=np.float32)

        # Action chunk length from training configuration.
        self.chunk_size = (
            model_config["framework"]["action_model"]["future_action_window_size"] + 1
        )
        log_message(f"InternVLA-M1 loaded. chunk_size={self.chunk_size}")

    def predict(self, agentview_img, wrist_img, instruction,
                cfg_scale=1.5, num_ddim_steps=10, return_metadata: bool = False):
        """Predict one action chunk for the current rollout step.

        Args:
            agentview_img (np.ndarray): Agent-view RGB frame, shape
                ``[H, W, 3]``, dtype ``np.uint8``.
            wrist_img (np.ndarray): Wrist-view RGB frame, shape
                ``[H, W, 3]``, dtype ``np.uint8``.
            instruction (str): Language instruction for the current task.
            cfg_scale (float): Guidance scale for diffusion sampling.
            num_ddim_steps (int): Number of DDIM denoising steps.
            return_metadata (bool): If true, also returns model-output metadata.

        Returns:
            np.ndarray | tuple[np.ndarray, dict]: Action chunk with shape
            ``[T, 7]`` and dtype ``np.float32``. If `return_metadata` is true,
            returns ``(actions, metadata)``.

        Raises:
            RuntimeError: If model inference fails.
            KeyError: If expected output keys are missing.

        Example:
            >>> chunk = policy.predict(agent_img, wrist_img, "Turn on the stove")
            >>> print(chunk.shape)  # [T, 7]
        """
        
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
        #print(f"[DEBUG] image_tokens={n_img_tokens} (expected: >=2) | '{instruction}'", flush=True)

        view1 = Image.fromarray(agentview_img)
        view2 = Image.fromarray(wrist_img)
        with torch.inference_mode():
            pred = self.model.predict_action(
                batch_images=[[view1, view2]],
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

        # Keep binary gripper command before environment conversion.
        normalized[:, 6] = np.where(normalized[:, 6] < 0.5, 0.0, 1.0)

        # Recover physical action scale from normalized range [-1, 1].
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
        }
        return actions, metadata



# ─── Observation helper ───────────────────────────────────────────────────────


def get_obs_internvla(obs):
    """Convert environment observations into InternVLA-M1 image inputs.

    Args:
        obs (dict): Observation dict from LIBERO/robosuite.

    Returns:
        tuple[np.ndarray, np.ndarray]: Agent and wrist RGB frames resized to
        ``[224, 224, 3]``, dtype ``np.uint8``.

    Raises:
        KeyError: If required camera keys are missing in `obs`.
        cv2.error: If image resize fails.

    Note:
        A double flip is applied to match orientation used in InternVLA-M1
        preprocessing.
    """
    agentview = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
    wrist      = np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1])
    agentview  = cv2.resize(agentview, (224, 224))
    wrist      = cv2.resize(wrist,     (224, 224))
    return agentview.astype(np.uint8), wrist.astype(np.uint8)


def binarize_gripper_for_robosuite(gripper_val: float) -> float:
    """Map model gripper output to robosuite gripper command convention.

    Args:
        gripper_val (float): Gripper value in the model domain.

    Returns:
        float: Gripper command where close is ``+1.0`` and open is ``-1.0``.

    Raises:
        ValueError: If `gripper_val` is NaN.
    """
    return 1.0 - 2.0 * float(gripper_val > 0.5)


def extract_eef_state(obs) -> np.ndarray:
    """Read end-effector position from observation keys used in LIBERO.

    Args:
        obs (dict): Observation dict from the environment.

    Returns:
        np.ndarray: End-effector position ``[3]`` with dtype ``np.float32``.

    Raises:
        KeyError: If no supported key for end-effector position is found.
    """
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
    """Expand level presets into an explicit command-level list.

    Args:
        cfg (GenerateConfig): Runtime config with `change_command` and
            `command_level` fields.

    Returns:
        list[Optional[str]]: Levels to evaluate. `None` means default command.

    Raises:
        AttributeError: If expected config fields are missing.
    """
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
    """Run one rollout with InternVLA-M1 and collect rollout data.

    This loop is where policy inference meets simulator control. It queries one
    action chunk every `chunk_size` steps, applies end-effector delta (R^7)
    controls, and stops when the environment reports success.

    Args:
        cfg: Runtime config.
        env: LIBERO environment instance.
        task_description (str): Instruction used for this rollout.
        policy (InternVLA_M1_policy): Loaded model wrapper.
        initial_state: Optional deterministic initial state for the task.
        collect_vlm_bbox (bool): If true, asks the policy for extra metadata.
        frame_save_dir (Optional[str]): Reserved path for optional debug frames.
        log_file: Optional file handle for persistent logs.

    Returns:
        tuple[bool, dict]: ``(success, replay_traj)`` where `replay_traj`
        contains `images`, `states`, `task_command`, and `actions`.

    Raises:
        RuntimeError: Internal errors are caught and converted to `success=False`.

    Example:
        >>> success, replay = run_episode(cfg, env, text, policy, init_state)
        >>> print(success, len(replay["actions"]))
    """
    env.reset()

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.reset()

    # Wait a few steps after reset for simulator stabilization.
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action("tiny_vla"))

    max_timesteps = TASK_MAX_STEPS.get(cfg.task_suite_name, 300)
    chunk_size    = policy.chunk_size
    query_freq    = chunk_size

    # Use task_description directly without grounding
    image_list, action_list, state_list = [], [], []
    success = False
    current_chunk = None

    try:
        for t in range(max_timesteps):
            agentview, wrist = get_obs_internvla(obs)
            image_list.append(agentview.copy())
            state_list.append(extract_eef_state(obs).copy())

            if t % query_freq == 0:
                if collect_vlm_bbox:
                    current_chunk, pred_meta = policy.predict(
                        agentview,
                        wrist,
                        task_description,
                        return_metadata=True,
                    )
                else:
                    current_chunk = policy.predict(
                                                    agentview, wrist, task_description,
                                                    cfg_scale=cfg.cfg_scale,
                                                    num_ddim_steps=cfg.num_ddim_steps,
                                                    )
                # current_chunk: [chunk_size, 7]

            action_raw = current_chunk[t % chunk_size]          # [7]

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
    """Run all rollouts for one task and its BDDL task variants.

    This function handles variant discovery (`*_syn_<level>.bddl` and optional
    `*_syn_<level>_vN.bddl`) and computes task success rate over all rollouts.
    It is used in syntactic generalization runs (L1/L2/L3) and default-command
    baseline runs.

    Args:
        cfg: Runtime config.
        task_suite: LIBERO benchmark suite instance.
        task_id (int): Zero-based task index in the suite.
        policy (InternVLA_M1_policy): Loaded policy wrapper.
        log_file: Open log handle for this evaluation run.
        total_episodes (int): Running rollout counter across tasks.
        total_successes (int): Running success counter across tasks.
        run_id (Optional[str]): Run identifier used by the caller.

    Returns:
        tuple[int, int, str, float, int]: Updated global counters, canonical
        task name, task success rate, and number of rollouts for this task.

    Raises:
        AssertionError: If `selected_version` is used with inconsistent config.
        OSError: If BDDL task variant discovery fails on filesystem access.

    Example:
        >>> stats = run_task(cfg, suite, 0, policy, log_file)
        >>> print(stats[3])  # task success rate
    """

    task           = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)

    if cfg.selected_version is not None:
        assert cfg.change_command and cfg.command_level is not None, (
            f"selected_version={cfg.selected_version} requires "
            f"change_command=True and command_level is not None"
        )

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

        base_syn_file = f"{base_name}_syn_{cfg.command_level}.bddl"
        base_syn_path = os.path.join(bddl_folder, os.path.basename(base_syn_file))
        if os.path.isfile(base_syn_path):
            has_base_syn = True
            base_syn_filename = os.path.basename(base_syn_path)
            log_message(f"Found base syn file: {base_syn_path}", log_file)
        else:
            log_message(f"Warning: base syn file not found: {base_syn_path}", log_file)

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
        base_name = task.bddl_file.replace('.bddl', '')
        try:
            from libero.libero import get_libero_path
            bddl_folder = os.path.join(get_libero_path("bddl_files"), task.problem_folder)
        except Exception:
            bddl_folder = os.path.dirname(task.bddl_file)

        base_syn_file = f"{base_name}_syn_{cfg.command_level}.bddl"
        base_syn_path = os.path.join(bddl_folder, os.path.basename(base_syn_file))
        if os.path.isfile(base_syn_path):
            # Use -1 as a sentinel for the base BDDL task variant.
            available_versions = [(-1, os.path.basename(base_syn_path))]
            log_message(f"Found base syn file: {base_syn_path}", log_file)
        else:
            log_message(f"Warning: base syn file not found: {base_syn_path}", log_file)

    # ── Testing version ───────────────────────────────────────────────
    # -1  -> base file *_syn_l3.bddl
    # >=0 -> versioned file *_syn_l3_vN.bddl
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
    task_canonical_description = None
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

    total_task_episodes  = sum(r["episodes"]  for r in task_results_per_version.values())
    total_task_successes = sum(r["successes"] for r in task_results_per_version.values())
    task_sr = (total_task_successes / total_task_episodes
               if total_task_episodes > 0 else 0.0)

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
    """Run the full evaluation loop and print summary tables.

    Args:
        cfg (GenerateConfig): Runtime configuration parsed by draccus.

    Returns:
        None: Results are written to logs, optional JSON, and stdout.

    Raises:
        KeyError: If `task_suite_name` is not in LIBERO benchmark registry.
        ValueError: If `task_range` cannot be parsed as `"start-end"`.
        OSError: If log or summary paths cannot be written.
    """
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

    print("\n" + "=" * 100)
    print("DETAILED RESULTS TABLE")
    print("=" * 100)
    print(f"{'Task':<50} | {'Success Rate':>12} | {'Successes/Total':>15}")
    print("-" * 100)

    all_tasks = list(next(iter(task_results.values())).keys())
    for t in all_tasks:
        for l, res in task_results.items():
            sr  = res.get(t, {}).get("success_rate", 0.0)
            eps = res.get(t, {}).get("episodes", 0)
            sr = float(sr) if sr is not None else 0.0
            eps = int(eps) if eps is not None else 0
            succ = int(round(sr * eps))
            task_label = str(t) if t is not None else "unknown_task"
            print(f"{task_label:<50} | {sr:>11.1%} | {succ:>6}/{eps:<8}")

    print("-" * 100)

    # OVERALL row.
    for l, res in all_results.items():
        overall_sr   = res["success_rate"]
        total_eps    = res["total_episodes"]
        total_succ   = res["total_successes"]
        print(f"{'OVERALL':<50} | {overall_sr:>11.1%} | {total_succ:>6}/{total_eps:<8}")

    print("=" * 100)



if __name__ == "__main__":
    run_libero_eval()