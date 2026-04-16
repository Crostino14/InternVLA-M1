"""Create trajectory-density heatmaps from InternVLA-M1 rollout files.

This script reads rollout `.npy` artifacts, extracts end-effector positions,
and builds per-task plus combined heatmaps. It is used for qualitative failure
analysis in syntactic and task-level generalization experiments.
"""

import argparse
import glob
import os
import pickle as pkl
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


TABLE_SIZE = (1.0, 1.0)  # height (y), width (x) in meters
DEFAULT_STATE_KEYS = ["states", "ee_states", "robot_states", "qpos"]


def infer_command_level_from_path(path):
    """Infer evaluation level from a rollout directory path.

    Args:
        path (str): Rollout directory path.

    Returns:
        str: Level label such as `DEFAULT`, `L1`, `L2`, `L3`,
        `TASK_COMP_L1`, or `TASK_COMP_L2`.
    """
    normalized_path = path.lower()
    if "task_comp_l1" in normalized_path:
        return "TASK_COMP_L1"
    if "task_comp_l2" in normalized_path:
        return "TASK_COMP_L2"
    if "command_l1" in normalized_path or "test_l1" in normalized_path:
        return "L1"
    if "command_l2" in normalized_path or "test_l2" in normalized_path:
        return "L2"
    if "command_l3" in normalized_path or "test_l3" in normalized_path:
        return "L3"
    if "ablation" in normalized_path:
        return "ABLATION"
    return "DEFAULT"


def get_rollout_paths(base_path, model_prefix):
    """Return rollout subpaths for direct or multi-level layouts.

    Args:
        base_path (str): Root rollout path.
        model_prefix (str): Display prefix for plot titles.

    Returns:
        list[tuple[str, str]]: `(path, model_label)` entries to process.
    """
    run_folders_direct = glob.glob(os.path.join(base_path, "run_*"))
    if run_folders_direct:
        return [(base_path, model_prefix)]

    normalized_path = base_path.lower()
    if "task_comp" in normalized_path or "task_composition" in normalized_path:
        return [
            (os.path.join(base_path, "task_comp_l1"), f"{model_prefix} (Task Comp L1)"),
            (os.path.join(base_path, "task_comp_l2"), f"{model_prefix} (Task Comp L2)"),
        ]

    return [
        (os.path.join(base_path, "default"), f"{model_prefix} (Default)"),
        (os.path.join(base_path, "test_l1"), f"{model_prefix} (L1)"),
        (os.path.join(base_path, "test_l2"), f"{model_prefix} (L2)"),
        (os.path.join(base_path, "test_l3"), f"{model_prefix} (L3)"),
        (os.path.join(base_path, "command_ablation"), f"{model_prefix} (Ablation)"),
    ]


def parse_episode_id(path):
    """Extract rollout episode id from filename.

    Args:
        path (str): Rollout file path.

    Returns:
        int: Episode number if found, otherwise a large fallback key.
    """
    match = re.search(r"episode=(\d+)", os.path.basename(path))
    if match:
        return int(match.group(1))
    return 10**9


def pick_state_array(rollout_data, state_keys):
    """Select the first available state array from preferred keys.

    Args:
        rollout_data (dict): Loaded rollout dictionary from `.npy`.
        state_keys (list[str]): Priority order of state keys.

    Returns:
        tuple[np.ndarray | None, str | None]: Selected array and key, or
        `(None, None)` when no candidate is found.
    """
    for key in state_keys:
        if key in rollout_data:
            return np.array(rollout_data[key]), key
    return None, None


def compute_heatmap_data(task_distribution):
    """Convert trajectories to a cropped density grid.

    Args:
        task_distribution (list[np.ndarray]): List of rollout trajectories with
            xyz positions in meters.

    Returns:
        np.ndarray: Cropped density heatmap for plotting.
    """
    px_resolution = 0.5  # cm per pixel
    table_size_cm = np.array(TABLE_SIZE) * 100
    table_size_px = (table_size_cm / px_resolution).astype(np.int32)
    table_map = np.zeros((table_size_px[0], table_size_px[1]), dtype=np.float32)

    for trajectory in task_distribution:
        trajectory = np.array(trajectory)[:, :2]
        px_traj = (trajectory * 100 / px_resolution).astype(np.int32)
        px_traj[:, 0] = table_map.shape[0] // 2 + px_traj[:, 0]
        px_traj[:, 1] = table_map.shape[1] // 2 + px_traj[:, 1]
        px_traj = px_traj[
            (px_traj[:, 0] >= 0)
            & (px_traj[:, 0] < table_map.shape[0])
            & (px_traj[:, 1] >= 0)
            & (px_traj[:, 1] < table_map.shape[1])
        ]
        for x, y in px_traj:
            table_map[x, y] += 1

    y_min, y_max = -45, 20
    x_min, x_max = -35, 35
    y_min_px = int((y_min + table_size_cm[0] / 2) / px_resolution)
    y_max_px = int((y_max + table_size_cm[0] / 2) / px_resolution)
    x_min_px = int((x_min + table_size_cm[1] / 2) / px_resolution)
    x_max_px = int((x_max + table_size_cm[1] / 2) / px_resolution)

    return table_map[y_min_px:y_max_px, x_min_px:x_max_px]


def heat_map(task_distribution, task_path, task_name, command_level="DEFAULT"):
    """Create and save one heatmap image for a single task.

    Args:
        task_distribution (list[np.ndarray]): Per-rollout xyz trajectories.
        task_path (str): Output folder for saved image.
        task_name (str): Task command label.
        command_level (str): Level label shown in the title.

    Returns:
        np.ndarray: Cropped heatmap array used for plotting.
    """
    cropped_map = compute_heatmap_data(task_distribution)

    y_min, y_max = -45, 20
    x_min, x_max = -35, 35
    px_resolution = 0.5
    task_title = task_name.replace("_", " ").title()

    fig, ax = plt.subplots(figsize=(10, 14))
    plt.title(f"[{command_level}] \"{task_title}\"")
    plt.xlabel("Y Axis (cm)")
    plt.ylabel("X Axis (cm)")

    vmax = np.max(cropped_map) if np.max(cropped_map) > 0 else 1
    norm = mcolors.LogNorm(vmin=1, vmax=vmax)
    im = ax.imshow(cropped_map, cmap="plasma", origin="upper", norm=norm)
    ax.invert_xaxis()

    ticks_x = np.arange(0, cropped_map.shape[1], int(10 / px_resolution))
    ticks_y = np.arange(0, cropped_map.shape[0], int(10 / px_resolution))
    tick_labels_x = np.arange(x_min, x_max, 10)
    tick_labels_y = np.arange(y_min, y_max, 10)

    plt.xticks(ticks_x, tick_labels_x)
    plt.yticks(ticks_y, tick_labels_y)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Trajectory Density (log scale)")

    os.makedirs(task_path, exist_ok=True)
    save_path = os.path.join(task_path, f"{task_name}_heatmap.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Saved heatmap to {save_path}")
    return cropped_map


def create_combined_heatmap(heatmaps_data, task_path, model_name, command_level):
    """Create one combined image with all task heatmaps.

    Args:
        heatmaps_data (dict[str, np.ndarray]): Task name to heatmap array.
        task_path (str): Output folder for saved image.
        model_name (str): Model name shown in figure title.
        command_level (str): Level label shown in figure title.

    Returns:
        None: Writes image file to disk.
    """
    if not heatmaps_data:
        print("WARNING: No heatmaps to combine")
        return

    n_tasks = len(heatmaps_data)
    task_names = list(heatmaps_data.keys())

    positive_max = [np.max(hm) for hm in heatmaps_data.values() if np.max(hm) > 0]
    global_vmax = max(positive_max) if positive_max else 1

    fig_width = 5 * n_tasks + 1
    fig, axes = plt.subplots(1, n_tasks, figsize=(fig_width, 8))
    if n_tasks == 1:
        axes = [axes]

    norm = mcolors.LogNorm(vmin=1, vmax=global_vmax)

    y_min, y_max = -45, 20
    x_min, x_max = -35, 35
    px_resolution = 0.5

    for idx, (ax, task_name) in enumerate(zip(axes, task_names)):
        cropped_map = heatmaps_data[task_name]
        task_title = task_name.replace("_", " ").title()

        im = ax.imshow(cropped_map, cmap="plasma", origin="upper", norm=norm)
        ax.invert_xaxis()
        ax.set_title(f'"{task_title}"', fontsize=9, wrap=True)

        ticks_x = np.arange(0, cropped_map.shape[1], int(10 / px_resolution))
        ticks_y = np.arange(0, cropped_map.shape[0], int(10 / px_resolution))
        tick_labels_x = np.arange(x_min, x_max, 10)
        tick_labels_y = np.arange(y_min, y_max, 10)

        ax.set_xticks(ticks_x)
        ax.set_xticklabels(tick_labels_x, fontsize=7)
        ax.set_yticks(ticks_y)
        ax.set_yticklabels(tick_labels_y, fontsize=7)

        if idx == 0:
            ax.set_ylabel("X Axis (cm)", fontsize=9)
        ax.set_xlabel("Y Axis (cm)", fontsize=9)

    fig.suptitle(f"{model_name} - {command_level}", fontsize=12)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Trajectory Density (log scale)", fontsize=10)

    plt.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.08, wspace=0.15)

    os.makedirs(task_path, exist_ok=True)
    save_path = os.path.join(task_path, f"combined_heatmaps_{command_level.lower()}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Saved combined heatmap to {save_path}")


def process_rollout_folder(test_path, model_name, dataset_config=None, state_keys=None):
    """Process one rollout folder and generate task heatmaps.

    Args:
        test_path (str): Folder containing `run_*` rollout directories.
        model_name (str): Display name used in plot titles.
        dataset_config (dict | None): Optional dataset stats for
            de-normalizing states.
        state_keys (list[str] | None): Priority list for state-array keys.

    Returns:
        None: Writes per-task and combined heatmap images.
    """
    command_level = infer_command_level_from_path(test_path)
    state_keys = state_keys or DEFAULT_STATE_KEYS

    print(f"\n{'=' * 80}")
    print(f"Processing: {test_path}")
    print(f"Model: {model_name}")
    print(f"Command level: {command_level}")
    print(f"State keys priority: {state_keys}")
    print(f"{'=' * 80}\n")

    run_folders = glob.glob(os.path.join(test_path, "run_*"))
    if not run_folders:
        print(f"WARNING: No run_* folders found in {test_path}")
        return

    tasks_trajectories = {}
    skipped_missing_state = 0

    for run in run_folders:
        trajectories_npy = glob.glob(os.path.join(run, "*.npy"))
        trajectories_npy.sort(key=parse_episode_id)

        for trajectory_npy in trajectories_npy:
            data = np.load(trajectory_npy, allow_pickle=True).item()
            print(f"Analyzing {os.path.basename(trajectory_npy)}...")

            task_name = data.get("task_command", "unknown_task")
            if task_name not in tasks_trajectories:
                tasks_trajectories[task_name] = []

            states, used_key = pick_state_array(data, state_keys)
            if states is None:
                skipped_missing_state += 1
                print(
                    f"  WARNING: no state array found (keys checked: {state_keys}) in {os.path.basename(trajectory_npy)}"
                )
                print("  Skipping this file...")
                continue

            if states.ndim == 1:
                print(f"  WARNING: state array has shape {states.shape}, expected [T, D]. Skipping...")
                continue

            states_xyz = states[:, :3]

            if dataset_config is not None and "qpos_mean" in dataset_config and "qpos_std" in dataset_config:
                qpos_mean = np.array(dataset_config["qpos_mean"])[:3]
                qpos_std = np.array(dataset_config["qpos_std"])[:3]
                states_xyz = (states_xyz * qpos_std) + qpos_mean

            print(f"  Using state key: {used_key} | trajectory length: {len(states_xyz)}")
            tasks_trajectories[task_name].append(states_xyz)

    if tasks_trajectories:
        heatmaps_data = {}
        for task_name, episodes in tasks_trajectories.items():
            if not episodes:
                continue
            print(f"Creating heatmap for task: {task_name} with {len(episodes)} episodes")
            cropped_map = heat_map(episodes, test_path, task_name, command_level)
            heatmaps_data[task_name] = cropped_map

        print("\nCreating combined heatmap...")
        create_combined_heatmap(heatmaps_data, test_path, model_name, command_level)

        if skipped_missing_state > 0:
            print(f"WARNING: skipped {skipped_missing_state} rollout files due to missing state arrays")
    else:
        print(f"WARNING: No valid trajectories found in {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create heatmaps for InternVLA-M1 rollout trajectories")
    parser.add_argument(
        "--test_path",
        type=str,
        default="/mnt/beegfs/a.cardamone7/outputs/rollouts/libero_goal/task_composition/internvla_m1",
        help="Path to rollout folder (direct run_* or parent folder with multiple configs)",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="",
        help="Optional path to dataset_stats.pkl for state denormalization",
    )
    parser.add_argument(
        "--state_keys",
        type=str,
        default=",".join(DEFAULT_STATE_KEYS),
        help="Comma-separated key priority for state arrays in rollout npy",
    )
    args = parser.parse_args()

    if args.dataset_config and os.path.exists(args.dataset_config):
        dataset_config = pkl.load(open(args.dataset_config, "rb"))
        print(f"Loaded dataset config from: {args.dataset_config}")
    else:
        dataset_config = None
        if args.dataset_config:
            print("WARNING: Dataset config not found; states are assumed already denormalized")

    state_keys = [k.strip() for k in args.state_keys.split(",") if k.strip()]

    for path, model_name in get_rollout_paths(args.test_path, "InternVLA-M1"):
        if os.path.exists(path):
            process_rollout_folder(path, model_name, dataset_config=dataset_config, state_keys=state_keys)
        else:
            print(f"WARNING: Path does not exist: {path}")

    print("\n" + "=" * 80)
    print("HEATMAP GENERATION COMPLETE")
    print("=" * 80)
