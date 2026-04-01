"""
run_internvla_eval_task_comp.py

Evaluates an InternVLA-M1 trained policy on custom LIBERO task composition scenarios.
Tests task-level generalization: the model must apply known primitives to new object/target
combinations never seen during training.

Custom tasks (all share the libero_goal scene):
  L1:
    1. Put the plate on the top of the cabinet
    2. Put the plate on the stove
    3. Put the cream cheese on the top of the cabinet
    4. Put the cream cheese on the plate
    5. Open the top layer of the drawer and put the cream cheese inside
  L2:
    1. Open the middle drawer of the cabinet
    2. Put the bowl on the stove
    3. Put the cream cheese in the bowl
    4. Push the plate to the front of the stove
    5. Put the bowl on top of the cabinet
"""

import sys
import os
import logging
import gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['DEVICE'] = "cuda"

import torch
import cv2
import numpy as np
import time
import tqdm
import json
import imageio
import draccus
from dataclasses import dataclass
from typing import Optional
from enum import Enum
from PIL import Image

from InternVLA.model.framework.M1 import InternVLA_M1

from libero.libero import get_libero_path
from libero.libero.benchmark import Task
from libero.libero.envs import OffScreenRenderEnv

from utils.libero_utils import (
    get_libero_dummy_action,
    quat2axisangle,
    extract_command_from_bddl,
)
from utils.robot_utils import set_seed_everywhere, DATE_TIME


# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def log_message(message: str, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


# ─── Task Composition Registries ──────────────────────────────────────────────

TASK_COMP_L1_TASKS = [
    {
        "bddl_file": "put_the_plate_on_top_of_the_cabinet_task_comp_l1.bddl",
        "init_states_from": "push_the_plate_to_the_front_of_the_stove",
    },
    {
        "bddl_file": "put_the_plate_on_the_stove_task_comp_l1.bddl",
        "init_states_from": "push_the_plate_to_the_front_of_the_stove",
    },
    {
        "bddl_file": "put_the_cream_cheese_on_top_of_the_cabinet_task_comp_l1.bddl",
        "init_states_from": "put_the_cream_cheese_in_the_bowl",
    },
    {
        "bddl_file": "put_the_cream_cheese_on_the_plate_task_comp_l1.bddl",
        "init_states_from": "put_the_cream_cheese_in_the_bowl",
    },
    {
        "bddl_file": "open_the_top_drawer_and_put_the_cream_cheese_inside_task_comp_l1.bddl",
        "init_states_from": "open_the_top_drawer_and_put_the_bowl_inside",
    },
]

TASK_COMP_L2_TASKS = [
    {
        "bddl_file": "open_the_middle_drawer_of_the_cabinet_task_comp_l2.bddl",
        "init_states_from": "open_the_middle_drawer_of_the_cabinet",
    },
    {
        "bddl_file": "put_the_bowl_on_the_stove_task_comp_l2.bddl",
        "init_states_from": "put_the_bowl_on_the_stove",
    },
    {
        "bddl_file": "put_the_cream_cheese_in_the_bowl_task_comp_l2.bddl",
        "init_states_from": "put_the_cream_cheese_in_the_bowl",
    },
    {
        "bddl_file": "push_the_plate_to_the_front_of_the_stove_task_comp_l2.bddl",
        "init_states_from": "push_the_plate_to_the_front_of_the_stove",
    },
    {
        "bddl_file": "put_the_bowl_on_top_of_the_cabinet_task_comp_l2.bddl",
        "init_states_from": "put_the_cream_cheese_in_the_bowl",
    },
]

TASK_COMP_REGISTRY = {
    "l1": TASK_COMP_L1_TASKS,
    "l2": TASK_COMP_L2_TASKS,
}

TASK_MAX_STEPS = 500


# ─── InternVLA-M1 Policy Wrapper ──────────────────────────────────────────────

class InternVLA_M1_policy:
    """Policy wrapper for InternVLA-M1."""

    def __init__(self, model_path: str, device: str = "cuda"):
        from InternVLA.model.framework.share_tools import read_model_config
        log_message(f"Loading InternVLA-M1 from {model_path}")
        self.model = InternVLA_M1.from_pretrained(model_path)
        self.model = self.model.to(device).eval()
        self.device = device

        _, norm_stats    = read_model_config(model_path)
        action_stats     = norm_stats["franka"]["action"]
        self.action_mask = np.array(action_stats["mask"], dtype=bool)
        self.action_high = np.array(action_stats["max"],  dtype=np.float32)
        self.action_low  = np.array(action_stats["min"],  dtype=np.float32)

        model_config, _  = read_model_config(model_path)
        self.chunk_size  = (
            model_config["framework"]["action_model"]["future_action_window_size"] + 1
        )
        log_message(f"InternVLA-M1 loaded. chunk_size={self.chunk_size}")

    def predict(self, agentview_img: np.ndarray, wrist_img: np.ndarray,
                instruction: str, cfg_scale: float = 1.5,
                num_ddim_steps: int = 10) -> np.ndarray:
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

        normalized = np.clip(pred["normalized_actions"][0], -1, 1)  # [T, 7]
        normalized[:, 6] = np.where(normalized[:, 6] < 0.5, 0.0, 1.0)

        actions = np.where(
            self.action_mask,
            0.5 * (normalized + 1) * (self.action_high - self.action_low) + self.action_low,
            normalized,
        )
        return actions  # [T, 7]


# ─── Observation & Action Helpers ─────────────────────────────────────────────

def get_obs_internvla(obs):
    """Returns (agentview, wrist) as uint8 RGB arrays at 224x224."""
    agentview = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
    wrist      = np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1])
    agentview  = cv2.resize(agentview, (224, 224))
    wrist      = cv2.resize(wrist,     (224, 224))
    return agentview.astype(np.uint8), wrist.astype(np.uint8)


def binarize_gripper_for_robosuite(gripper_val: float) -> float:
    """Converts unnormalized gripper {0.0, 1.0} to robosuite delta_qpos."""
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


# ─── Custom Task Loading ──────────────────────────────────────────────────────

def load_custom_tasks(comp_level: str = "l1"):
    """Build Task objects and load init_states for each task_comp task."""
    bddl_dir = os.path.join(get_libero_path("bddl_files"), "libero_goal")
    init_dir  = os.path.join(get_libero_path("init_states"), "libero_goal")

    custom_tasks = []
    for task_def in TASK_COMP_REGISTRY[comp_level]:
        bddl_filename = task_def["bddl_file"]
        init_from     = task_def["init_states_from"]

        bddl_path = os.path.join(bddl_dir, bddl_filename)
        assert os.path.exists(bddl_path), f"BDDL file not found: {bddl_path}"

        task_description = extract_command_from_bddl(bddl_path)
        assert task_description is not None, f"Could not extract language from {bddl_path}"

        task_name = bddl_filename.replace(".bddl", "")
        task = Task(
            name=task_name,
            language=task_description,
            problem="Libero",
            problem_folder="libero_goal",
            bddl_file=bddl_filename,
            init_states_file=f"{init_from}.pruned_init",
        )

        init_states_path = os.path.join(init_dir, f"{init_from}.pruned_init")
        assert os.path.exists(init_states_path), f"Init states not found: {init_states_path}"
        init_states = torch.load(init_states_path, weights_only=False)

        custom_tasks.append({
            "task":             task,
            "init_states":      init_states,
            "task_description": task_description,
            "bddl_path":        bddl_path,
        })

    return custom_tasks


def create_env_from_bddl(bddl_path, resolution=256):
    """Create LIBERO environment directly from a BDDL file path."""
    env_args = {
        "bddl_file_name": bddl_path,
        "camera_heights": resolution,
        "camera_widths":  resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env


# ─── Logging Setup ────────────────────────────────────────────────────────────

def setup_logging(cfg):
    run_id = f"EVAL-task_comp_{cfg.comp_level}-internvla_m1-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to: {local_log_filepath}")
    return log_file, local_log_filepath, run_id


# ─── Episode Runner ───────────────────────────────────────────────────────────

def run_episode(cfg, env, task_description: str, policy: InternVLA_M1_policy,
                initial_state=None, log_file=None):
    """Run a single episode with InternVLA-M1."""
    env.reset()

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.reset()

    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action("tiny_vla"))

    chunk_size    = policy.chunk_size
    query_freq    = chunk_size
    image_list    = []
    action_list   = []
    state_list    = []
    success       = False
    current_chunk = None

    try:
        for t in range(TASK_MAX_STEPS):
            agentview, wrist = get_obs_internvla(obs)
            image_list.append(agentview.copy())
            state_list.append(extract_eef_state(obs).copy())

            if t % query_freq == 0:
                current_chunk = policy.predict(agentview, wrist, task_description)

            action_raw  = current_chunk[t % chunk_size]
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


# ─── Task Runner ──────────────────────────────────────────────────────────────

def run_custom_task(cfg, task_info, task_idx, num_tasks, policy,
                    log_file, total_episodes=0, total_successes=0):
    """Run evaluation for a single task_comp task with InternVLA-M1."""
    task             = task_info["task"]
    init_states      = task_info["init_states"]
    task_description = task_info["task_description"]
    bddl_path        = task_info["bddl_path"]

    env = create_env_from_bddl(bddl_path, resolution=cfg.env_img_res)

    log_message("=" * 80, log_file)
    log_message(f"TASK {task_idx + 1}/{num_tasks} (Task Composition {cfg.comp_level.upper()})", log_file)
    log_message(f"BDDL:      {task.bddl_file}", log_file)
    log_message(f"Command:   {task_description}", log_file)
    log_message(f"Init from: {task.init_states_file}", log_file)
    log_message("=" * 80, log_file)

    task_episodes = task_successes = 0

    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        initial_state = init_states[episode_idx]

        success, replay_traj = run_episode(
            cfg, env, task_description, policy, initial_state, log_file
        )

        task_episodes   += 1
        total_episodes  += 1
        if success:
            task_successes  += 1
            total_successes += 1

        # Save rollout video
        rollout_dir = (
            f"/mnt/beegfs/a.cardamone7/outputs/rollouts/libero_goal/"
            f"task_composition/internvla_m1/task_comp_{cfg.comp_level}/run_{cfg.run_number}"
        )
        os.makedirs(rollout_dir, exist_ok=True)
        processed_desc = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
        mp4_path = (
            f"{rollout_dir}/{DATE_TIME}"
            f"--episode={total_episodes}--success={success}--task={processed_desc}.mp4"
        )

        video_writer = imageio.get_writer(mp4_path, fps=30)
        for img in replay_traj["images"]:
            video_writer.append_data(img)
        video_writer.close()
        log_message(f"Saved rollout MP4 at {mp4_path}", log_file)

        npy_path = mp4_path.replace(".mp4", ".npy")
        np.save(npy_path, replay_traj)
        log_message(f"Saved trajectory at {npy_path}", log_file)

        log_message(f"Success: {success}", log_file)
        log_message(
            f"# episodes: {total_episodes} | "
            f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)",
            log_file,
        )

    task_sr = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
    log_message(f"Task SR: {task_sr:.4f} ({task_sr * 100:.1f}%)", log_file)

    try:
        env.close()
        log_message("Environment closed successfully", log_file)
    except Exception as e:
        log_message(f"Warning: Error closing environment: {e}", log_file)
    gc.collect()

    return total_episodes, total_successes, task_description, task_sr, task_episodes, task_successes


# ─── Results Table ────────────────────────────────────────────────────────────

def print_results_table(task_results, all_results, comp_level: str):
    print("\n" + "=" * 100)
    print(f"TASK COMPOSITION {comp_level.upper()} (InternVLA-M1) - RESULTS TABLE")
    print("=" * 100)
    print(f"{'Task':<60} | {'Success Rate':>12} | {'Successes/Total':>15}")
    print("-" * 100)

    for task_name, result in task_results.items():
        sr   = result["success_rate"]
        eps  = result["episodes"]
        succ = result.get("successes", int(round(sr * eps)))
        print(f"{task_name:<60} | {sr:>11.1%} | {succ:>6}/{eps:<8}")

    print("-" * 100)
    overall_sr   = all_results["success_rate"]
    overall_succ = all_results["total_successes"]
    overall_eps  = all_results["total_episodes"]
    print(f"{'OVERALL':<60} | {overall_sr:>11.1%} | {overall_succ:>6}/{overall_eps:<8}")
    print("=" * 100)


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class GenerateConfig:
    # Model
    model_path: str = "/home/A.CARDAMONE7/checkpoints/InternVLA-M1-LIBERO-Goal"

    # LIBERO environment
    num_steps_wait:      int = 10
    num_trials_per_task: int = 50
    env_img_res:         int = 256

    # Task composition
    comp_level: str = "l1"
    task_start: int = 0
    task_end:   int = -1   # -1 means all tasks

    # Logging
    run_id_note:   Optional[str] = None
    local_log_dir: str           = "./experiments/logs"
    summary_file:  Optional[str] = None

    seed:       int  = 7
    run_number: int  = 0
    debug:      bool = False


# ─── Main ─────────────────────────────────────────────────────────────────────

@draccus.wrap()
def eval_task_comp(cfg: GenerateConfig) -> float:
    """Evaluate InternVLA-M1 on task composition scenarios."""
    set_seed_everywhere(cfg.seed)

    policy = InternVLA_M1_policy(cfg.model_path)

    all_custom_tasks = load_custom_tasks(cfg.comp_level)
    total_num_tasks  = len(all_custom_tasks)

    task_end     = cfg.task_end if cfg.task_end >= 0 else total_num_tasks
    custom_tasks = all_custom_tasks[cfg.task_start:task_end]
    num_tasks    = len(custom_tasks)

    log_message(f"Loaded {total_num_tasks} task_comp_{cfg.comp_level} tasks", None)
    log_message(f"Running subset [{cfg.task_start}:{task_end}] ({num_tasks} tasks)", None)
    for i, ct in enumerate(custom_tasks):
        log_message(f"  [{cfg.task_start + i}] {ct['task_description']}", None)

    log_file, local_log_filepath, run_id = setup_logging(cfg)

    log_message("=" * 80, log_file)
    log_message(f"TASK COMPOSITION {cfg.comp_level.upper()} EVALUATION (InternVLA-M1)", log_file)
    log_message(f"Model:             {cfg.model_path}", log_file)
    log_message(f"Seed:              {cfg.seed}", log_file)
    log_message(f"Trials per task:   {cfg.num_trials_per_task}", log_file)
    log_message(f"Tasks:             {num_tasks} (subset [{cfg.task_start}:{task_end}] of {total_num_tasks})", log_file)
    log_message("=" * 80, log_file)

    total_episodes = total_successes = 0
    task_results   = {}

    for task_idx in tqdm.tqdm(range(num_tasks), desc=f"Task Comp {cfg.comp_level.upper()}"):
        total_episodes, total_successes, task_name, task_sr, task_eps, task_succ = run_custom_task(
            cfg, custom_tasks[task_idx], task_idx, num_tasks, policy,
            log_file, total_episodes, total_successes,
        )
        task_results[task_name] = {
            "success_rate": task_sr,
            "episodes":     task_eps,
            "successes":    task_succ,
        }

    final_sr   = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    all_results = {
        "success_rate":    final_sr,
        "total_episodes":  total_episodes,
        "total_successes": total_successes,
    }

    log_message("=" * 80, log_file)
    log_message(f"FINAL RESULTS — TASK COMPOSITION {cfg.comp_level.upper()} (InternVLA-M1):", log_file)
    log_message(f"Total episodes:  {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall SR:      {final_sr:.4f} ({final_sr * 100:.1f}%)", log_file)
    log_message("=" * 80, log_file)

    if cfg.summary_file:
        with open(cfg.summary_file, "w") as f:
            json.dump({"task_results": task_results, "overall": all_results}, f, indent=2)

    log_file.close()

    print_results_table(task_results, all_results, cfg.comp_level)
    return final_sr


if __name__ == "__main__":
    eval_task_comp()
