#!/usr/bin/env python3
"""
Copy video HDF5 files (observation.images.*) with ONLY the relevant episodes for each L3 variation.
Each variation directory gets filtered HDF5 files containing only its episodes.
"""

import json
import h5py
from pathlib import Path

SOURCE_DATASET = Path("/mnt/beegfs/a.cardamone7/datasets/lerobot_libero_goal")
DEST_DATASET = Path("/mnt/beegfs/a.cardamone7/datasets/lerobot_libero_goal_l3")


def get_episode_indices_for_variation(variation_dir):
    """Get set of episode indices for a variation."""
    episodes_file = variation_dir / "meta" / "episodes.jsonl"
    
    episode_indices = set()
    with open(episodes_file, 'r') as f:
        for line in f:
            ep = json.loads(line.strip())
            episode_indices.add(ep["episode_index"])
    
    return episode_indices


def filter_hdf5_by_episodes(src_file, dest_file, episode_indices):
    """Copy HDF5 file keeping only specific episodes."""
    
    try:
        with h5py.File(src_file, 'r') as src:
            with h5py.File(dest_file, 'w') as dst:
                # Copy only episodes in episode_indices
                for key in src.keys():
                    # Keys are like "episode_000005"
                    if key.startswith('episode_'):
                        ep_num = int(key.split('_')[1])
                        
                        if ep_num in episode_indices:
                            # Copy this episode's data
                            src.copy(src[key], dst, name=key)
                
                # Copy root attributes if any
                for attr in src.attrs:
                    dst.attrs[attr] = src.attrs[attr]
        
        return True
    except Exception as e:
        print(f"      ERROR in {src_file.name}: {e}")
        return False


def setup_video_files_for_variation(variation_dir):
    """Create filtered HDF5 video files for one variation."""
    
    # Get episode indices
    episode_indices = get_episode_indices_for_variation(variation_dir)
    
    source_chunk_dir = SOURCE_DATASET / "videos" / "chunk-000"
    dest_chunk_dir = variation_dir / "videos" / "chunk-000"
    dest_chunk_dir.mkdir(parents=True, exist_ok=True)
    
    # Video file types to copy
    video_files = ["observation.images.image", "observation.images.wrist_image"]
    
    copied_count = 0
    
    for video_file in video_files:
        src_hdf5 = source_chunk_dir / video_file
        dst_hdf5 = dest_chunk_dir / video_file
        
        if not src_hdf5.exists():
            print(f"      WARNING: {src_hdf5.name} not found")
            continue
        
        # Filter and copy
        if filter_hdf5_by_episodes(src_hdf5, dst_hdf5, episode_indices):
            copied_count += 1
    
    return copied_count, len(episode_indices)


def main():
    print("=" * 100)
    print("L3 Dataset Video Setup: Copy Filtered HDF5 Files (Only Relevant Episodes)")
    print("=" * 100)
    
    variation_dirs = sorted([d for d in DEST_DATASET.iterdir() if d.is_dir()])
    print(f"\nProcessing {len(variation_dirs)} L3 variations...\n")
    
    total_variations = 0
    total_episodes_per_variation = []
    
    for i, variation_dir in enumerate(variation_dirs, 1):
        var_name = variation_dir.name
        
        # Setup video files for this variation
        copied, ep_count = setup_video_files_for_variation(variation_dir)
        total_episodes_per_variation.append(ep_count)
        
        status = "✓" if copied == 2 else "⚠"
        print(f"[{i:2d}/{len(variation_dirs)}] {status} {var_name:<65} {ep_count:3d} episodes → {copied} HDF5 files")
        total_variations += 1
    
    print("\n" + "=" * 100)
    print(f"✓ Complete! Processed {total_variations} variations")
    print(f"  Episodes distribution: min={min(total_episodes_per_variation)}, max={max(total_episodes_per_variation)}, avg={sum(total_episodes_per_variation)//len(total_episodes_per_variation)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
