#!/usr/bin/env python3
"""
Copy video MP4 files - only relevant episodes for each L3 variation.
Structure:
- observation.images.image/episode_XXXXX.mp4
- observation.images.wrist_image/episode_XXXXX.mp4
"""

import json
import shutil
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


def copy_videos_for_variation(variation_dir):
    """Copy video files for only this variation's episodes."""
    
    # Get episode indices
    episode_indices = get_episode_indices_for_variation(variation_dir)
    
    source_chunk_dir = SOURCE_DATASET / "videos" / "chunk-000"
    dest_chunk_dir = variation_dir / "videos" / "chunk-000"
    dest_chunk_dir.mkdir(parents=True, exist_ok=True)
    
    # Video types
    video_types = ["observation.images.image", "observation.images.wrist_image"]
    
    total_copied = 0
    
    for video_type in video_types:
        source_type_dir = source_chunk_dir / video_type
        dest_type_dir = dest_chunk_dir / video_type
        dest_type_dir.mkdir(parents=True, exist_ok=True)
        
        if not source_type_dir.exists():
            print(f"      WARNING: {source_type_dir} not found")
            continue
        
        # Copy only relevant episode videos
        for ep_idx in episode_indices:
            video_file = f"episode_{ep_idx:06d}.mp4"
            source_video = source_type_dir / video_file
            dest_video = dest_type_dir / video_file
            
            if source_video.exists():
                try:
                    shutil.copy2(source_video, dest_video)
                    total_copied += 1
                except Exception as e:
                    print(f"      ERROR copying {video_file}: {e}")
            else:
                print(f"      WARNING: {source_video} not found")
    
    return total_copied, len(episode_indices)


def main():
    print("=" * 100)
    print("L3 Dataset Video Setup: Copy MP4 Files - Only Relevant Episodes per Variation")
    print("=" * 100)
    
    variation_dirs = sorted([d for d in DEST_DATASET.iterdir() if d.is_dir()])
    print(f"\nProcessing {len(variation_dirs)} L3 variations...\n")
    
    total_videos_copied = 0
    
    for i, variation_dir in enumerate(variation_dirs, 1):
        var_name = variation_dir.name
        
        # Copy videos for this variation
        copied, ep_count = copy_videos_for_variation(variation_dir)
        total_videos_copied += copied
        
        # Each episode should have 2 videos (image + wrist_image)
        expected_videos = ep_count * 2
        status = "✓" if copied == expected_videos else "⚠"
        
        print(f"[{i:2d}/{len(variation_dirs)}] {status} {var_name:<65} {ep_count:3d} episodes → {copied:4d}/{expected_videos:4d} videos")
    
    print("\n" + "=" * 100)
    print(f"✓ Complete! Copied {total_videos_copied} total videos across all variations")
    print("=" * 100)


if __name__ == "__main__":
    main()
