#!/usr/bin/env python3
"""
Copy video files from source dataset to L3 variation directories.
"""

import json
import os
import shutil
from pathlib import Path

SOURCE_DATASET = Path("/mnt/beegfs/a.cardamone7/datasets/lerobot_libero_goal")
DEST_DATASET = Path("/mnt/beegfs/a.cardamone7/datasets/lerobot_libero_goal_l3")


def copy_videos_for_variation(variation_dir):
    """Copy video files referenced in episodes.jsonl to the variation directory."""
    
    # Read episodes.jsonl to get episode indices
    episodes_file = variation_dir / "meta" / "episodes.jsonl"
    if not episodes_file.exists():
        print(f"  Warning: {episodes_file} not found")
        return 0
    
    episodes = []
    with open(episodes_file, 'r') as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    
    # Find and copy video files
    source_videos_dir = SOURCE_DATASET / "videos"
    dest_videos_dir = variation_dir / "videos"
    
    if not source_videos_dir.exists():
        print(f"  Warning: Source videos directory not found: {source_videos_dir}")
        return 0
    
    copied = 0
    skipped = 0
    
    for ep in episodes:
        ep_idx = ep["episode_index"]
        
        # Try different video formats
        for ext in ['mp4', 'avi', 'mov', 'mkv']:
            source_video = source_videos_dir / f"episode_{ep_idx}.{ext}"
            
            if source_video.exists():
                dest_video = dest_videos_dir / source_video.name
                
                # Skip if already exists
                if dest_video.exists():
                    skipped += 1
                    break
                
                try:
                    shutil.copy2(source_video, dest_video)
                    copied += 1
                except Exception as e:
                    print(f"    Error copying {source_video.name}: {e}")
                break
        else:
            # No video file found for this episode
            pass
    
    return copied


def copy_all_videos():
    """Copy videos for all L3 variations."""
    
    print("Copying video files to L3 variations...")
    print("=" * 70)
    
    total_copied = 0
    variation_dirs = sorted([d for d in DEST_DATASET.iterdir() if d.is_dir()])
    
    for i, variation_dir in enumerate(variation_dirs, 1):
        variation_name = variation_dir.name
        copied = copy_videos_for_variation(variation_dir)
        total_copied += copied
        
        if i % 3 == 0:  # Print progress every 3 variations
            print(f"[{i}/{len(variation_dirs)}] {variation_name}: {copied} videos")
    
    print("=" * 70)
    print(f"✓ Total videos copied: {total_copied}")


if __name__ == "__main__":
    copy_all_videos()
