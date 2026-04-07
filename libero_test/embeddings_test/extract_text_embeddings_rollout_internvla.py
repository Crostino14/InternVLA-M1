"""
Extract text-backbone embeddings from InternVLA-M1 on LIBERO tasks.

This mirrors the OpenVLA rollout embedding workflow:
- For each task and command level (default/l1/l2/l3), run N rollouts
- Extract text-conditioned embeddings from the Qwen2.5-VL backbone
- Support first-step-only mode or full-rollout mode
- Save per-rollout and mean embeddings for downstream similarity analysis
"""

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


CURRENT_FILE = Path(__file__).resolve()
LIBERO_TEST_ROOT = CURRENT_FILE.parents[1]
PROJECT_ROOT = CURRENT_FILE.parents[2]
ROBOSUITE_TEST_ROOT = CURRENT_FILE.parents[3]
LIBERO_ROOT = ROBOSUITE_TEST_ROOT / "LIBERO"

sys.path.insert(0, str(LIBERO_TEST_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LIBERO_ROOT))

from InternVLA.model.framework.M1 import InternVLA_M1
from InternVLA.model.framework.share_tools import read_mode_config
from InternVLA.model.modules.vlm.QWen2_5 import IMAGE_TOKEN_INDEX, VIDEO_TOKEN_INDEX
from libero.libero import benchmark
from utils.libero_utils import get_libero_dummy_action, get_libero_env


TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

SPATIAL_COT_PROMPT = (
    "Your task is {instruction}. First, identify key objects and their positions using spatial relations: "
    "left/right of, in front/behind, next to, above/below. Then execute the action."
)


@dataclass
class EmbeddingConfig:
    model_path: str
    task_suite_name: str = "libero_goal"
    command_levels: Tuple[str, ...] = ("default", "l1", "l2", "l3")
    output_dir: str = "/mnt/beegfs/a.cardamone7/outputs/embeddings/internvla"
    env_img_res: int = 256
    num_steps_wait: int = 10
    num_rollouts_per_task: int = 10
    first_step_only: bool = False
    seed: int = 0
    use_cot: bool = False
    cot_prompt: str = SPATIAL_COT_PROMPT
    cfg_scale: float = 1.5
    num_ddim_steps: int = 10
    task_range: str = "0-9"


def parse_task_range(task_range: str, max_tasks: int) -> List[int]:
    start, end = [int(x) for x in task_range.split("-")]
    start = max(0, start)
    end = min(max_tasks - 1, end)
    if end < start:
        return []
    return list(range(start, end + 1))


def get_obs_internvla(obs: Dict) -> Tuple[np.ndarray, np.ndarray]:
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    agentview = cv2.resize(agentview, (224, 224))
    wrist = cv2.resize(wrist, (224, 224))
    return agentview.astype(np.uint8), wrist.astype(np.uint8)


def binarize_gripper_for_robosuite(gripper_val: float) -> float:
    return 1.0 - 2.0 * float(gripper_val > 0.5)


def _migrate_legacy_config_keys(cfg) -> None:
    """Backfill known legacy typo keys in old checkpoint configs."""
    legacy_to_new = [
        ("framework.layer_qformer.ouptput_dim", "framework.layer_qformer.output_dim"),
        ("framework.action_model.ouptput_dim", "framework.action_model.output_dim"),
    ]

    for legacy_key, new_key in legacy_to_new:
        old_val = OmegaConf.select(cfg, legacy_key)
        new_val = OmegaConf.select(cfg, new_key)

        if new_val is None and old_val is not None:
            OmegaConf.update(cfg, new_key, old_val, force_add=True)
            print(f"Config compatibility: mapped '{legacy_key}' -> '{new_key}' ({old_val})")

    # Conservative fallback for very old/partial configs.
    if OmegaConf.select(cfg, "framework.layer_qformer.output_dim") is None:
        OmegaConf.update(cfg, "framework.layer_qformer.output_dim", 768, force_add=True)
        print("Config compatibility: fallback set framework.layer_qformer.output_dim=768")


class InternVLATextEmbeddingExtractor:
    def __init__(self, cfg: EmbeddingConfig, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = device

        run_dir = os.path.dirname(os.path.dirname(cfg.model_path))
        config_path = os.path.join(run_dir, "config.yaml")
        omega_cfg = OmegaConf.load(config_path)
        _migrate_legacy_config_keys(omega_cfg)

        self.model = InternVLA_M1(config=omega_cfg)
        state_dict = torch.load(cfg.model_path, map_location="cpu")
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: missing keys in checkpoint (first 10): {missing[:10]}")
        if unexpected:
            print(f"Warning: unexpected keys in checkpoint (first 10): {unexpected[:10]}")

        self.model = self.model.to(self.device).eval()

        model_config, norm_stats = read_mode_config(cfg.model_path)
        embodiment_key = "franka"
        if embodiment_key not in norm_stats:
            if "new_embodiment" in norm_stats:
                embodiment_key = "new_embodiment"
            else:
                embodiment_key = list(norm_stats.keys())[0]
                print(
                    "Warning: 'franka' not found in norm_stats. "
                    f"Using fallback embodiment '{embodiment_key}'."
                )

        action_stats = norm_stats[embodiment_key]["action"]
        self.action_mask = np.array(action_stats["mask"], dtype=bool)
        self.action_high = np.array(action_stats["max"], dtype=np.float32)
        self.action_low = np.array(action_stats["min"], dtype=np.float32)

        self.chunk_size = model_config["framework"]["action_model"]["future_action_window_size"] + 1

    def _format_instruction(self, instruction: str) -> str:
        if self.cfg.use_cot:
            return self.cfg.cot_prompt.replace("{instruction}", instruction)
        return instruction

    def _pool_text_tokens(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if input_ids is None or attention_mask is None:
            return hidden_states.mean(dim=1)

        text_mask = attention_mask.bool()
        text_mask = text_mask & input_ids.ne(IMAGE_TOKEN_INDEX)
        text_mask = text_mask & input_ids.ne(VIDEO_TOKEN_INDEX)

        if int(text_mask.sum().item()) == 0:
            text_mask = attention_mask.bool()

        mask = text_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled

    @torch.inference_mode()
    def extract_text_embedding(
        self,
        agentview_img: np.ndarray,
        wrist_img: np.ndarray,
        instruction: str,
    ) -> Tuple[np.ndarray, str]:
        formatted_instruction = self._format_instruction(instruction)

        view1 = Image.fromarray(agentview_img)
        view2 = Image.fromarray(wrist_img)

        qwen_inputs = self.model.qwen_vl_interface.build_qwenvl_inputs(
            images=[[view1, view2]],
            instructions=[formatted_instruction],
        )

        outputs = self.model.qwen_vl_interface(
            **qwen_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden = outputs.hidden_states[-1]
        pooled = self._pool_text_tokens(
            hidden_states=last_hidden,
            input_ids=qwen_inputs.get("input_ids"),
            attention_mask=qwen_inputs.get("attention_mask"),
        )
        embedding = pooled[0].detach().cpu().float().numpy()
        return embedding, formatted_instruction

    @torch.inference_mode()
    def predict_action_chunk(
        self,
        agentview_img: np.ndarray,
        wrist_img: np.ndarray,
        instruction: str,
    ) -> Tuple[np.ndarray, str]:
        formatted_instruction = self._format_instruction(instruction)

        view1 = Image.fromarray(agentview_img)
        view2 = Image.fromarray(wrist_img)

        pred = self.model.predict_action(
            batch_images=[[view1, view2]],
            instructions=[formatted_instruction],
            cfg_scale=self.cfg.cfg_scale,
            use_ddim=True,
            num_ddim_steps=self.cfg.num_ddim_steps,
        )

        normalized = np.clip(pred["normalized_actions"][0], -1, 1)
        normalized[:, 6] = np.where(normalized[:, 6] < 0.5, 0.0, 1.0)

        actions = np.where(
            self.action_mask,
            0.5 * (normalized + 1.0) * (self.action_high - self.action_low) + self.action_low,
            normalized,
        )
        return actions, formatted_instruction


def extract_first_step_embedding(
    cfg: EmbeddingConfig,
    env,
    extractor: InternVLATextEmbeddingExtractor,
    task_instruction: str,
    initial_state=None,
) -> Tuple[np.ndarray, str]:
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.reset()

    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action("tiny_vla"))

    agentview, wrist = get_obs_internvla(obs)
    return extractor.extract_text_embedding(agentview, wrist, task_instruction)


def run_single_episode(
    cfg: EmbeddingConfig,
    env,
    extractor: InternVLATextEmbeddingExtractor,
    task_instruction: str,
    initial_state=None,
    max_steps: int = 300,
):
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.reset()

    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action("tiny_vla"))

    embeddings = []
    success = False
    chunk_actions = None
    instruction_used = None

    try:
        for t in range(max_steps):
            agentview, wrist = get_obs_internvla(obs)

            emb, instruction_used = extractor.extract_text_embedding(agentview, wrist, task_instruction)
            embeddings.append(emb)

            if t % extractor.chunk_size == 0:
                chunk_actions, _ = extractor.predict_action_chunk(agentview, wrist, task_instruction)

            action_raw = chunk_actions[t % extractor.chunk_size]
            gripper_env = binarize_gripper_for_robosuite(action_raw[6])
            env_action = np.concatenate([action_raw[:6], [gripper_env]])

            obs, _, done, _ = env.step(env_action.tolist())
            if done:
                success = True
                break

    except Exception as exc:
        print(f"      Episode error: {exc}")

    return embeddings, success, len(embeddings), instruction_used


def extract_embeddings_rollout(cfg: EmbeddingConfig):
    extractor = InternVLATextEmbeddingExtractor(cfg=cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()

    task_ids = parse_task_range(cfg.task_range, task_suite.n_tasks)
    max_steps = TASK_MAX_STEPS.get(cfg.task_suite_name, 300)
    mode_str = "FIRST STEP ONLY" if cfg.first_step_only else "FULL ROLLOUT"

    print("\n" + "=" * 90)
    print(f"INTERNVLA-M1 TEXT EMBEDDING EXTRACTION - {mode_str}")
    print("=" * 90)
    print(f"Model path: {cfg.model_path}")
    print(f"Task suite: {cfg.task_suite_name}")
    print(f"Task IDs: {task_ids}")
    print(f"Command levels: {cfg.command_levels}")
    print(f"Rollouts per task: {cfg.num_rollouts_per_task}")
    print(f"Use CoT prompt: {cfg.use_cot}")
    print(f"CFG scale: {cfg.cfg_scale} | DDIM steps: {cfg.num_ddim_steps}")
    if not cfg.first_step_only:
        print(f"Max steps per episode: {max_steps}")
    print("=" * 90)

    all_embeddings: Dict[str, Dict] = {}

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        task_name = getattr(task, "name", str(task))
        initial_states = task_suite.get_task_init_states(task_id)

        print("\n" + "-" * 90)
        print(f"Task {task_id + 1}/{task_suite.n_tasks}: {task_name}")
        print("-" * 90)

        for level in cfg.command_levels:
            print(f"  Level {level.upper()}:")

            rollout_embeddings: List[np.ndarray] = []
            rollout_all_embeddings: List[np.ndarray] = []
            rollout_successes: List[bool] = []
            successes = 0
            total_steps = 0
            command_text = None
            instruction_used = None

            for rollout_idx in range(cfg.num_rollouts_per_task):
                try:
                    env, task_description, _ = get_libero_env(
                        task,
                        model_family="InternVLA-M1",
                        change_command=(level != "default"),
                        command_level=level if level != "default" else None,
                        resolution=cfg.env_img_res,
                    )
                    env.seed(cfg.seed + rollout_idx)
                    command_text = task_description

                    init_state = initial_states[rollout_idx % len(initial_states)]

                    if cfg.first_step_only:
                        emb, instruction_used = extract_first_step_embedding(
                            cfg=cfg,
                            env=env,
                            extractor=extractor,
                            task_instruction=task_description,
                            initial_state=init_state,
                        )
                        rollout_embeddings.append(emb)
                        print(f"    Rollout {rollout_idx + 1:02d}/{cfg.num_rollouts_per_task}: ok (1 step)")
                    else:
                        episode_embeddings, success, steps, instruction_used = run_single_episode(
                            cfg=cfg,
                            env=env,
                            extractor=extractor,
                            task_instruction=task_description,
                            initial_state=init_state,
                            max_steps=max_steps,
                        )

                        if episode_embeddings:
                            arr = np.stack(episode_embeddings, axis=0)
                            rollout_all_embeddings.append(arr)
                            rollout_embeddings.append(arr.mean(axis=0))

                        rollout_successes.append(bool(success))
                        successes += int(success)
                        total_steps += int(steps)
                        status = "ok" if success else "fail"
                        print(
                            f"    Rollout {rollout_idx + 1:02d}/{cfg.num_rollouts_per_task}: "
                            f"{status} ({steps} steps)"
                        )

                except Exception as exc:
                    print(f"    Rollout {rollout_idx + 1:02d}: error - {exc}")
                finally:
                    try:
                        env.close()
                    except Exception:
                        pass

            if not rollout_embeddings:
                print("    No embeddings extracted for this level.")
                continue

            per_rollout = np.stack(rollout_embeddings, axis=0)
            mean_embedding = per_rollout.mean(axis=0)

            key = f"task_{task_id:02d}_{level}"
            all_embeddings[key] = {
                "task_id": task_id,
                "task_name": task_name,
                "command_level": level,
                "command_text": command_text,
                "instruction_used": instruction_used,
                "embedding": mean_embedding,
                "embedding_per_rollout": per_rollout,
                "num_rollouts": len(rollout_embeddings),
                "first_step_only": cfg.first_step_only,
                "use_cot": cfg.use_cot,
                "cfg_scale": cfg.cfg_scale,
                "num_ddim_steps": cfg.num_ddim_steps,
            }

            if not cfg.first_step_only:
                all_embeddings[key]["embedding_all_steps"] = np.concatenate(rollout_all_embeddings, axis=0)
                all_embeddings[key]["rollout_successes"] = rollout_successes
                all_embeddings[key]["num_successes"] = successes
                all_embeddings[key]["total_steps"] = total_steps
                all_embeddings[key]["success_rate"] = successes / max(len(rollout_embeddings), 1)

                print(
                    f"    Summary: {successes}/{len(rollout_embeddings)} success | "
                    f"steps={total_steps} | emb_shape={mean_embedding.shape}"
                )
            else:
                print(f"    Summary: {len(rollout_embeddings)} embeddings | emb_shape={mean_embedding.shape}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    mode_suffix = "first_step" if cfg.first_step_only else "full"
    output_path = os.path.join(
        cfg.output_dir,
        (
            f"internvla_rollout_embeddings_{cfg.task_suite_name}_"
            f"{'_'.join(cfg.command_levels)}_{mode_suffix}_r{cfg.num_rollouts_per_task}.pkl"
        ),
    )

    with open(output_path, "wb") as f:
        pickle.dump(all_embeddings, f)

    print("\n" + "=" * 90)
    print("EXTRACTION COMPLETE")
    print("=" * 90)
    print(f"Total entries: {len(all_embeddings)}")
    if all_embeddings:
        first_key = next(iter(all_embeddings.keys()))
        print(f"Embedding shape: {all_embeddings[first_key]['embedding'].shape}")
    print(f"Output file: {output_path}")

    return all_embeddings, output_path


def parse_args() -> EmbeddingConfig:
    parser = argparse.ArgumentParser(
        description="Extract InternVLA-M1 text backbone embeddings on LIBERO rollouts"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to InternVLA checkpoint (.pt)")
    parser.add_argument("--task_suite", type=str, default="libero_goal", help="LIBERO suite name")
    parser.add_argument(
        "--command_levels",
        type=str,
        nargs="+",
        default=["default", "l1", "l2", "l3"],
        help="Command levels to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/beegfs/a.cardamone7/outputs/embeddings/internvla",
        help="Directory where pickle output is saved",
    )
    parser.add_argument("--resolution", type=int, default=256, help="Environment camera resolution")
    parser.add_argument("--num_steps_wait", type=int, default=10, help="Warmup no-op steps")
    parser.add_argument("--num_rollouts", type=int, default=10, help="Rollouts per task")
    parser.add_argument("--seed", type=int, default=0, help="Random seed base")
    parser.add_argument("--task_range", type=str, default="0-9", help="Task range, e.g. 0-9")
    parser.add_argument("--first_step_only", action="store_true", help="Extract only first observation embedding")
    parser.add_argument("--use_cot", action="store_true", help="Apply spatial CoT prompt before extraction")
    parser.add_argument("--cfg_scale", type=float, default=1.5, help="Classifier-free guidance scale")
    parser.add_argument("--num_ddim_steps", type=int, default=10, help="DDIM sampling steps")

    args = parser.parse_args()

    command_levels = tuple(level.strip().lower() for level in args.command_levels)

    return EmbeddingConfig(
        model_path=args.model_path,
        task_suite_name=args.task_suite,
        command_levels=command_levels,
        output_dir=args.output_dir,
        env_img_res=args.resolution,
        num_steps_wait=args.num_steps_wait,
        num_rollouts_per_task=args.num_rollouts,
        first_step_only=bool(args.first_step_only),
        seed=args.seed,
        use_cot=bool(args.use_cot),
        cfg_scale=args.cfg_scale,
        num_ddim_steps=args.num_ddim_steps,
        task_range=args.task_range,
    )


if __name__ == "__main__":
    config = parse_args()
    extract_embeddings_rollout(config)
