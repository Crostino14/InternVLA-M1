#!/usr/bin/env python3
"""
Copy ONLY the video data for relevant episodes from HDF5 to each L3 variation.
Each variation gets its own complete HDF5 files with only its episodes.
"""

import json
import h5py
import os
from pathlib import Path
import shutil

SOURCE_DATASET = Path("/mnt/beegfs/a.cardamone7/datasets/lerobot_libero_goal")
DEST_DATASET = Path("/mnt/beegfs/a.cardamone7/datasets/lerobot_libero_goal_l3")


def get_episode_indices(variation_dir):
    """Get list of episode indices for a variation from episodes.jsonl."""
    episodes_file = variation_dir / "meta" / "episodes.jsonl"
    
    if not episodes_file.exists():
        return []
    
    episode_indices = []
    with open(episodes_file, 'r') as f:
        for line in f:
            ep = json.loads(line.strip())
            episode_indices.append(ep["episode_index"])
    
    return sorted(episode_indices)


def copy_hdf5_episodes(src_hdf5, dest_hdf5, episode_indices):
    """Copy only specified episodes from source HDF5 to destination HDF5."""
    
    try:
        with h5py.File(src_hdf5, 'r') as src_file:
            with h5py.File(dest_hdf5, 'w') as dest_file:
                # Copy structure and copy only relevant episodes
                for key in src_file.keys():
                    if key.startswith('episode_'):
                        # Extract episode number from key like "episode_000005"
                        ep_num = int(key.split('_')[1])
                        
                        if ep_num in episode_indices:
                            # Copy entire episode group
                            src_file.copy(src_file[key], dest_file, name=key)
                
                # Copy attributes if any
                for attr in src_file.attrs:
                    dest_file.attrs[attr] = src_file.attrs[attr]
        
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


def setup_videos_for_variation(variation_dir):
    """Extract and copy video HDF5 files for this variation."""
    
    # Get episode indices for this variation
    episode_indices = get_episode_indices(variation_dir)
    if not episode_indices:
        return 0
    
    source_videos_dir = SOURCE_DATASET / "videos" / "chunk-000"
    dest_videos_dir = variation_dir / "videos" / "chunk-000"
    dest_videos_dir.mkdir(parents=True, exist_ok=True)
    
    if not source_videos_dir.exists():
        print(f"  Warning: Source videos dir not found: {source_videos_dir}")
        return 0
    
    # Copy each HDF5 video file with filtered episodes
    copied = 0
    hdf5_files = list(source_videos_dir.glob("*.hdf5")) + list(source_videos_dir.glob("*.h5"))
    
    if not hdf5_files:
        # Try without extension pattern
        hdf5_files = [f for f in source_videos_dir.iterdir() 
                      if f.is_file() and (f.suffix == '.hdf5' or f.suffix == '.h5' or 'image' in f.name)]
    
    for src_hdf5 in hdf5_files:
        dest_hdf5 = dest_videos_dir / src_hdf5.name
        
        # Copy HDF5 with only relevant episodes
        if copy_hdf5_episodes(src_hdf5, dest_hdf5, episode_indices):
            copied += 1
    
    return copied


def setup_all_variations():
    """Setup all L3 variations with copied video data (no symlinks)."""
    
    print("=" * 80)
    print("L3 Dataset Setup: Copy REAL video files (HDF5) - Only relevant episodes")
    print("=" * 80)
    
    variation_dirs = sorted([d for d in DEST_DATASET.iterdir() if d.is_dir()])
    print(f"\nSetup {len(variation_dirs)} L3 variations\n")
    
    total_copied = 0
    
    for i, variation_dir in enumerate(variation_dirs, 1):
        variation_name = variation_dir.name
        
        # Remove old symlinks first if they exist
        dest_videos = variation_dir / "videos" / "chunk-000"
        if dest_videos.exists():
            for f in dest_videos.glob("*"):
                try:
                    os.remove(f)
                except:
                    pass
        
        # Get episodes for this variation
        episodes = get_episode_indices(variation_dir)
        
        # Copy HDF5 files with filtered episodes
        copied = setup_videos_for_variation(variation_dir)
        total_copied += copied
        
        status = "✓" if copied > 0 else "✗"
        print(f"[{i:2d}/{len(variation_dirs)}] {status} {variation_name:<60} {len(episodes):3d} episodes, {copied} HDF5 files copied")
    
    print("\n" + "=" * 80)
    print(f"✓ Video setup complete! {total_copied} HDF5 files created across all variations")
    print("=" * 80)


if __name__ == "__main__":
    setup_all_variations()
