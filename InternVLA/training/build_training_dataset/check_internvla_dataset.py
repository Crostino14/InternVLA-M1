#!/usr/bin/env python3
"""
InternVLA Dataset & Training Config Validator
=============================================
Checks all critical points before launching a fine-tuning run:
  1. dataset_statistics.json  — presence + correct embodiment key
  2. LeRobot dataset keys     — action_dim, obs keys, language_instruction
  3. data_mix registry        — entry in the dataloader dict
  4. YAML config alignment    — action_dim / window_size / hidden_dim
  5. Checkpoint compatibility — missing/unexpected keys
  6. Freeze modules           — sanity check on trainable params

Usage:
  python check_internvla_dataset.py \
      --dataset_path /mnt/beegfs/a.cardamone7/datasets/lerobot_libero_goal_l3 \
      --config_yaml  InternVLA/config/training/internvla_co_train_libero_l3.yaml \
      --checkpoint   /mnt/beegfs/a.cardamone7/checkpoints/.../pytorch_model.pt \
      --data_mix     libero_goal_l3_finetune \
      --embodiment   new_embodiment \
      --action_dim   7 \
      --future_window 7 \
      --past_window   1
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ──────────────────────────────────────────────
#  Colour helpers
# ──────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"

def ok(msg):    print(f"  {GREEN}✔{RESET}  {msg}")
def warn(msg):  print(f"  {YELLOW}⚠{RESET}  {msg}")
def fail(msg):  print(f"  {RED}✘{RESET}  {msg}")
def info(msg):  print(f"  {CYAN}ℹ{RESET}  {msg}")
def section(title):
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")


# ──────────────────────────────────────────────
#  CHECK 1 — dataset_statistics.json
# ──────────────────────────────────────────────
def check_dataset_statistics(dataset_path: Path, embodiment: str) -> dict | None:
    section("CHECK 1 · dataset_statistics.json")
    stats_path = dataset_path / "dataset_statistics.json"

    if not stats_path.exists():
        # Try meta/ subfolder (LeRobot v2 layout)
        stats_path = dataset_path / "meta" / "stats.json"

    if not stats_path.exists():
        fail(f"File not found in {dataset_path} (tried root and meta/)")
        warn("Regenerate it with the snippet shown at the end of this report.")
        return None

    ok(f"Found: {stats_path}")

    with open(stats_path) as f:
        stats = json.load(f)

    info(f"Top-level keys: {list(stats.keys())}")

    if embodiment not in stats:
        fail(f"Embodiment key '{embodiment}' NOT in stats. Available: {list(stats.keys())}")
        warn("Rename the top-level key to match the embodiment tag used during training.")
        return None

    ok(f"Embodiment key '{embodiment}' found.")
    emb = stats[embodiment]

    required_sub = ["action", "observation.state"]
    for sub in required_sub:
        if sub not in emb:
            warn(f"Sub-key '{sub}' missing under '{embodiment}'.")
        else:
            action_stats = emb[sub]
            for stat in ["mean", "std", "min", "max"]:
                if stat not in action_stats:
                    warn(f"  '{sub}.{stat}' missing.")
                else:
                    vals = action_stats[stat]
                    ok(f"  {sub}.{stat} → dim={len(vals)}  sample={[round(v,4) for v in vals[:3]]}…")

    return stats


# ──────────────────────────────────────────────
#  CHECK 2 — LeRobot dataset structure
# ──────────────────────────────────────────────
REQUIRED_KEYS = [
    "action",
    "observation.state",
    "language_instruction",
]
CAMERA_KEY_CANDIDATES = [
    "observation.images.image",
    "observation.image",
    "observation.images.front_img_1",
]

def check_lerobot_dataset(dataset_path: Path, action_dim: int):
    section("CHECK 2 · LeRobot dataset keys & action_dim")
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        warn("lerobot package not importable — skipping dataset key checks.")
        warn("Make sure your conda env is active: conda activate internvla-m1")
        return

    try:
        ds = LeRobotDataset(str(dataset_path))
    except Exception as e:
        fail(f"Cannot load dataset: {e}")
        return

    ok(f"Dataset loaded — {len(ds)} frames, {ds.num_episodes} episodes.")
    sample = ds[0]
    found_keys = list(sample.keys())
    info(f"Sample keys: {found_keys}")

    for k in REQUIRED_KEYS:
        if k not in found_keys:
            fail(f"Required key '{k}' NOT found in sample.")
        else:
            ok(f"Key '{k}' present.")

    # action dim
    if "action" in sample:
        act = sample["action"]
        dim = act.shape[-1] if hasattr(act, "shape") else len(act)
        if dim != action_dim:
            fail(f"action_dim mismatch: dataset has {dim}, config expects {action_dim}")
        else:
            ok(f"action_dim={dim} ✔")

    # camera key
    cam_found = [k for k in CAMERA_KEY_CANDIDATES if k in found_keys]
    if not cam_found:
        fail(f"No camera key found. Tried: {CAMERA_KEY_CANDIDATES}")
        warn("Update CAMERA_KEY_CANDIDATES in this script or the collate_fn.")
    else:
        ok(f"Camera key: '{cam_found[0]}'")
        img = sample[cam_found[0]]
        shape = img.shape if hasattr(img, "shape") else "?"
        info(f"  Image shape: {shape}  (expected [3, H, W] or [H, W, 3])")

    # episodes per task (approximate)
    try:
        tasks = set(ds.meta.tasks.values()) if hasattr(ds.meta, "tasks") else set()
        n_tasks = len(tasks) if tasks else "?"
        info(f"Unique tasks: {n_tasks}")
    except Exception:
        pass


# ──────────────────────────────────────────────
#  CHECK 3 — data_mix registry
# ──────────────────────────────────────────────
DATA_MIX_REGISTRY_PATHS = [
    "InternVLA/dataloader/lerobot_datasets.py",
    "InternVLA/dataloader/gr00t_lerobot/lerobot_datasets.py",
    "InternVLA/dataloader/__init__.py",
]

def check_data_mix_registry(data_mix: str, internvla_root: Path):
    section("CHECK 3 · data_mix registry")
    found_in = []
    for rel_path in DATA_MIX_REGISTRY_PATHS:
        p = internvla_root / rel_path
        if not p.exists():
            continue
        text = p.read_text(errors="replace")
        if data_mix in text:
            found_in.append(str(p))

    if not found_in:
        fail(f"data_mix '{data_mix}' not found in any known registry file.")
        info("Searched paths:")
        for rp in DATA_MIX_REGISTRY_PATHS:
            info(f"  {internvla_root / rp}  (exists={( internvla_root / rp).exists()})")
        warn("Add an entry for this data_mix in the appropriate registry dict.")
    else:
        for f in found_in:
            ok(f"Found '{data_mix}' in: {f}")


# ──────────────────────────────────────────────
#  CHECK 4 — YAML config alignment
# ──────────────────────────────────────────────
def check_yaml_config(config_yaml: Path, action_dim: int, future_window: int, past_window: int):
    section("CHECK 4 · YAML config alignment")
    if not config_yaml.exists():
        warn(f"Config file not found: {config_yaml}")
        return

    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config_yaml)
    except ImportError:
        import yaml
        with open(config_yaml) as f:
            cfg_dict = yaml.safe_load(f)
        class _Wrap:
            def __init__(self, d): self._d = d
            def __getattr__(self, k):
                v = self._d.get(k)
                return _Wrap(v) if isinstance(v, dict) else v
        cfg = _Wrap(cfg_dict)

    checks = [
        ("framework.action_model.action_dim",               action_dim,    "action_dim"),
        ("framework.action_model.future_action_window_size", future_window, "future_action_window_size"),
        ("framework.action_model.past_action_window_size",   past_window,   "past_action_window_size"),
    ]

    def _nested_get(obj, path):
        parts = path.split(".")
        cur = obj
        for p in parts:
            try:
                cur = getattr(cur, p) if not isinstance(cur, dict) else cur[p]
            except Exception:
                return None
        return cur

    for path, expected, label in checks:
        val = _nested_get(cfg, path)
        if val is None:
            warn(f"{label}: key '{path}' not found in YAML.")
        elif int(val) != int(expected):
            fail(f"{label} mismatch: YAML={val}, CLI arg={expected}")
        else:
            ok(f"{label}={val} matches CLI arg.")

    # Freeze modules
    freeze = _nested_get(cfg, "trainer.freeze_modules")
    if freeze is None or str(freeze).lower() in ("none", "null", ""):
        warn("trainer.freeze_modules is None — entire model will be updated. "
             "Consider freezing 'qwenvl' to prevent catastrophic forgetting.")
    else:
        ok(f"freeze_modules='{freeze}'")
        info("Trainable: QFormer projector + DiT action head.")

    # chunk size consistency
    fw = _nested_get(cfg, "framework.action_model.future_action_window_size")
    if fw is not None:
        chunk = int(fw) + 1
        info(f"Derived chunk_size = future_action_window_size+1 = {chunk}")


# ──────────────────────────────────────────────
#  CHECK 5 — checkpoint compatibility
# ──────────────────────────────────────────────
def check_checkpoint(checkpoint: Path, config_yaml: Path, internvla_root: Path):
    section("CHECK 5 · Checkpoint key compatibility")
    if not checkpoint.exists():
        warn(f"Checkpoint not found: {checkpoint}")
        return

    try:
        import torch
        ckpt = torch.load(checkpoint, map_location="cpu")
        ckpt_keys = set(ckpt.keys())
        ok(f"Checkpoint loaded — {len(ckpt_keys)} top-level parameter tensors.")

        # Heuristic checks
        action_head_keys = [k for k in ckpt_keys if "action" in k.lower() or "dit" in k.lower()]
        projector_keys   = [k for k in ckpt_keys if "qform" in k.lower() or "projector" in k.lower()]

        info(f"Action head params : {len(action_head_keys)}")
        info(f"QFormer/projector  : {len(projector_keys)}")

        if not action_head_keys:
            warn("No action head keys found — checkpoint may lack the DiT head. "
                 "Verify it's an InternVLA checkpoint, not a bare QwenVL checkpoint.")

        # Try actual model load if framework is importable
        try:
            sys.path.insert(0, str(internvla_root))
            from omegaconf import OmegaConf
            from InternVLA.model.framework import build_framework
            cfg = OmegaConf.load(config_yaml)
            model = build_framework(cfg)
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            if missing:
                warn(f"Missing keys ({len(missing)}): {missing[:5]}{'…' if len(missing)>5 else ''}")
            else:
                ok("No missing keys — checkpoint fully covers model.")
            if unexpected:
                warn(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'…' if len(unexpected)>5 else ''}")
            else:
                ok("No unexpected keys.")
        except Exception as e:
            info(f"Full model load skipped ({e}). Key counts above are heuristic.")

    except Exception as e:
        fail(f"Cannot load checkpoint: {e}")


# ──────────────────────────────────────────────
#  REGENERATE SNIPPET
# ──────────────────────────────────────────────
def print_regen_snippet(dataset_path: Path, embodiment: str):
    section("SNIPPET · Regenerate dataset_statistics.json")
    snippet = f"""
import json
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("{dataset_path}")
raw_stats = ds.meta.stats   # dict: action / observation.state → mean, std, min, max

wrapped = {{ "{embodiment}": raw_stats }}

out_path = "{dataset_path}/dataset_statistics.json"
with open(out_path, "w") as f:
    json.dump(wrapped, f, indent=2)

print(f"Saved to {{out_path}}")
"""
    print(snippet)


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Validate InternVLA dataset & training config before launching a run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_path",   required=True,
                        help="Root dir of the LeRobot dataset "
                             "(contains meta/ or dataset_statistics.json)")
    parser.add_argument("--config_yaml",    default="",
                        help="Path to the OmegaConf YAML training config.")
    parser.add_argument("--checkpoint",     default="",
                        help="Path to pytorch_model.pt checkpoint to validate.")
    parser.add_argument("--internvla_root", default=".",
                        help="Root of the InternVLA repo "
                             "(for import and registry search).")
    parser.add_argument("--data_mix",       default="libero_goal_l3_finetune",
                        help="data_mix string used in the training script.")
    parser.add_argument("--embodiment",     default="new_embodiment",
                        help="Embodiment tag key expected in dataset_statistics.json.")
    parser.add_argument("--action_dim",     type=int, default=7,
                        help="Action dimensionality "
                             "(e.g. 7 for [x,y,z,roll,pitch,yaw,gripper]).")
    parser.add_argument("--future_window",  type=int, default=7,
                        help="future_action_window_size (chunk = this + 1).")
    parser.add_argument("--past_window",    type=int, default=1,
                        help="past_action_window_size in config.")
    args = parser.parse_args()

    dataset_path   = Path(args.dataset_path)
    config_yaml    = Path(args.config_yaml)    if args.config_yaml    else None
    checkpoint     = Path(args.checkpoint)     if args.checkpoint     else None
    internvla_root = Path(args.internvla_root)

    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  InternVLA Dataset & Config Validator{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")
    info(f"dataset_path  : {dataset_path}")
    info(f"embodiment    : {args.embodiment}")
    info(f"data_mix      : {args.data_mix}")
    info(f"action_dim    : {args.action_dim}")
    info(f"future_window : {args.future_window}  → chunk_size={args.future_window+1}")
    info(f"config_yaml   : {config_yaml or '(not provided)'}")
    info(f"checkpoint    : {checkpoint or '(not provided)'}")

    # Run all checks
    check_dataset_statistics(dataset_path, args.embodiment)
    check_lerobot_dataset(dataset_path, args.action_dim)
    check_data_mix_registry(args.data_mix, internvla_root)

    if config_yaml and config_yaml.exists():
        check_yaml_config(config_yaml, args.action_dim, args.future_window, args.past_window)
    else:
        section("CHECK 4 · YAML config alignment")
        warn("--config_yaml not provided or file not found — skipping.")

    if checkpoint and checkpoint.exists():
        check_checkpoint(checkpoint, config_yaml, internvla_root)
    else:
        section("CHECK 5 · Checkpoint key compatibility")
        warn("--checkpoint not provided or file not found — skipping.")

    print_regen_snippet(dataset_path, args.embodiment)

    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  Validation complete.{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}\n")


if __name__ == "__main__":
    main()