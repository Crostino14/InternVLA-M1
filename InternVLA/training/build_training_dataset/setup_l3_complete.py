#!/usr/bin/env python3
"""
Complete L3 dataset setup: copy data files and link video HDF5 files.
"""

import json
import os
import shutil
from pathlib import Path

SOURCE_DATASET = Path("/mnt/beegfs/a.cardamone7/datasets/lerobot_libero_goal")
DEST_DATASET = Path("/mnt/beegfs/a.cardamone7/datasets/lerobot_libero_goal_l3")


def copy_episode_data(variation_dir):
    """Copy parquet files for episodes referenced in episodes.jsonl to the variation directory."""
    
    # Read episodes.jsonl to get episode indices
    episodes_file = variation_dir / "meta" / "episodes.jsonl"
    if not episodes_file.exists():
        return 0
    
    episodes = []
    with open(episodes_file, 'r') as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    
    # Copy parquet files for each episode
    source_data_dir = SOURCE_DATASET / "data" / "chunk-000"
    dest_data_dir = variation_dir / "data" / "chunk-000"
    dest_data_dir.mkdir(parents=True, exist_ok=True)
    
    if not source_data_dir.exists():
        print(f"  Warning: Source data directory not found: {source_data_dir}")
        return 0
    
    copied = 0
    for ep in episodes:
        ep_idx = ep["episode_index"]
        parquet_file = f"episode_{ep_idx:06d}.parquet"
        source_file = source_data_dir / parquet_file
        dest_file = dest_data_dir / parquet_file
        
        if source_file.exists() and not dest_file.exists():
            try:
                shutil.copy2(source_file, dest_file)
                copied += 1
            except Exception as e:
                print(f"    Error copying {parquet_file}: {e}")
    
    return copied


def link_video_hdf5(variation_dir):
    """Create symlinks to HDF5 video files."""
    
    source_videos_dir = SOURCE_DATASET / "videos" / "chunk-000"
    dest_videos_dir = variation_dir / "videos" / "chunk-000"
    dest_videos_dir.mkdir(parents=True, exist_ok=True)
    
    if not source_videos_dir.exists():
        return False
    
    # Link HDF5 files
    for hdf5_file in source_videos_dir.glob("*"):
        dest_link = dest_videos_dir / hdf5_file.name
        
        if not dest_link.exists():
            try:
                os.symlink(hdf5_file, dest_link)
            except Exception as e:
                print(f"    Error linking {hdf5_file.name}: {e}")
                return False
    
    return True


def update_tasks_metadata(variation_dir):
    """Create/update tasks.jsonl with proper task information."""
    
    tasks_file = variation_dir / "meta" / "tasks.jsonl"
    
    # Read original episodes to get original task names
    episodes_file = variation_dir / "meta" / "episodes.jsonl"
    if not episodes_file.exists():
        return
    
    episodes = []
    with open(episodes_file, 'r') as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    
    if not episodes:
        return
    
    # Get unique tasks in this variation
    unique_tasks = set()
    for ep in episodes:
        for task in ep.get("tasks", []):
            unique_tasks.add(task)
    
    # Write tasks.jsonl
    with open(tasks_file, 'w') as f:
        for task_idx, task_name in enumerate(sorted(unique_tasks)):
            task_entry = {
                "task_index": task_idx,
                "task": task_name
            }
            f.write(json.dumps(task_entry) + '\n')


def update_info_json(variation_dir):
    """Create/update info.json with proper metadata."""
    
    info_file = variation_dir / "meta" / "info.json"
    
    # Read source info.json as template
    source_info_file = SOURCE_DATASET / "meta" / "info.json"
    if source_info_file.exists():
        with open(source_info_file, 'r') as f:
            info = json.load(f)
    else:
        info = {}
    
    # Update info with variation-specific data
    info["name"] = variation_dir.name
    
    # Write updated info.json
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)


def setup_all_variations():
    """Setup all L3 variations with data files and video links."""
    
    print("=" * 70)
    print("Setting up L3 Dataset: Copy Data & Link Videos")
    print("=" * 70)
    
    variation_dirs = sorted([d for d in DEST_DATASET.iterdir() if d.is_dir()])
    print(f"\nFound {len(variation_dirs)} L3 variations")
    
    total_copied = 0
    total_linked = 0
    total_failed = 0
    
    for i, variation_dir in enumerate(variation_dirs, 1):
        variation_name = variation_dir.name
        
        # Copy episode data files
        copied = copy_episode_data(variation_dir)
        total_copied += copied
        
        # Link video HDF5 files
        linked = link_video_hdf5(variation_dir)
        if linked:
            total_linked += 1
        else:
            total_failed += 1
        
        # Update metadata
        update_tasks_metadata(variation_dir)
        update_info_json(variation_dir)
        
        if i % 5 == 0 or i == len(variation_dirs):
            print(f"[{i:2d}/{len(variation_dirs)}] {variation_name:<55} ({copied:3d} files)")
    
    print("\n" + "=" * 70)
    print(f"✓ Setup complete!")
    print(f"  - Data files copied: {total_copied}")
    print(f"  - Video links created: {total_linked}/{len(variation_dirs)}")
    if total_failed > 0:
        print(f"  - Failed: {total_failed}")
    print("=" * 70)


if __name__ == "__main__":
    setup_all_variations()
