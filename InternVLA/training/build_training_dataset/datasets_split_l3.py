#!/usr/bin/env python3
"""
Script to organize LIBERO-Goal L3 dataset into 30 task variation directories.
Each L3 variation gets its own directory with filtered episodes and videos.
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

# Mapping of base LIBERO-Goal task names to their L3 syntactic variations
TASK_TO_L3_VARIATIONS = {
    "open the middle drawer of the cabinet": [
        "open_the_middle_layer_of_the_drawer_syn_l3_v1",
        "open_the_middle_layer_of_the_drawer_syn_l3_v2",
    ],
    "put the bowl on the stove": [
        "put_the_bowl_on_the_stove_syn_l3_v1",
        "put_the_bowl_on_the_stove_syn_l3_v2",
        "put_the_bowl_on_the_stove_syn_l3_v3",
    ],
    "put the wine bottle on top of the cabinet": [
        "put_the_wine_bottle_on_the_top_of_the_cabinet_syn_l3_v1",
        "put_the_wine_bottle_on_the_top_of_the_cabinet_syn_l3_v2",
        "put_the_wine_bottle_on_the_top_of_the_cabinet_syn_l3_v3",
    ],
    "open the top drawer and put the bowl inside": [
        "open_the_top_drawer_and_put_the_bowl_inside_syn_l3_v1",
        "open_the_top_drawer_and_put_the_bowl_inside_syn_l3_v2",
        "open_the_top_drawer_and_put_the_bowl_inside_syn_l3_v3",
    ],
    "put the bowl on top of the cabinet": [
        "put_the_bowl_on_the_top_of_the_cabinet_syn_l3_v1",
        "put_the_bowl_on_the_top_of_the_cabinet_syn_l3_v2",
        "put_the_bowl_on_the_top_of_the_cabinet_syn_l3_v3",
    ],
    "push the plate to the front of the stove": [
        "push_the_plate_to_the_front_of_the_stove_syn_l3_v1",
        "push_the_plate_to_the_front_of_the_stove_syn_l3_v2",
        "push_the_plate_to_the_front_of_the_stove_syn_l3_v3",
    ],
    "put the cream cheese in the bowl": [
        "put_the_cream_cheese_on_the_bowl_syn_l3_v1",
        "put_the_cream_cheese_on_the_bowl_syn_l3_v2",
    ],
    "turn on the stove": [
        "turn_on_the_stove_syn_l3_v1",
        "turn_on_the_stove_syn_l3_v2",
    ],
    "put the bowl on the plate": [
        "put_the_bowl_on_the_plate_syn_l3_v1",
        "put_the_bowl_on_the_plate_syn_l3_v2",
        "put_the_bowl_on_the_plate_syn_l3_v3",
    ],
    "put the wine bottle on the rack": [
        "put_the_wine_bottle_on_the_rack_syn_l3_v1",
        "put_the_wine_bottle_on_the_rack_syn_l3_v2",
        "put_the_wine_bottle_on_the_rack_syn_l3_v3",
    ],
}

# Source and destination paths
SOURCE_DATASET = Path("/mnt/beegfs/a.cardamone7/datasets/lerobot_libero_goal")
DEST_DATASET = Path("/mnt/beegfs/a.cardamone7/datasets/lerobot_libero_goal_l3")


def load_episodes(dataset_path):
    """Load episodes.jsonl from dataset."""
    episodes_file = dataset_path / "meta" / "episodes.jsonl"
    episodes = []
    with open(episodes_file, 'r') as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    return episodes


def get_task_episodes(episodes, task_name):
    """Filter episodes for a specific task."""
    return [ep for ep in episodes if task_name in ep["tasks"]]


def create_l3_directories(source_dataset, dest_dataset, task_to_variations, episodes):
    """Create directory structure and populate with filtered episodes for each L3 variation."""
    
    # Ensure destination root exists
    dest_dataset.mkdir(parents=True, exist_ok=True)
    
    # For each base task and its L3 variations
    for base_task, variations in task_to_variations.items():
        print(f"\nProcessing base task: {base_task}")
        
        # Get all episodes for this base task
        task_episodes = get_task_episodes(episodes, base_task)
        print(f"  Found {len(task_episodes)} episodes for base task")
        
        # Each L3 variation gets the same episodes (syntactic variation, not data variation)
        # In practice, you'd split or filter based on actual L3 variant characteristics
        for variant in variations:
            variant_dir = dest_dataset / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (variant_dir / "data").mkdir(exist_ok=True)
            (variant_dir / "meta").mkdir(exist_ok=True)
            (variant_dir / "videos").mkdir(exist_ok=True)
            
            # Write filtered episodes.jsonl
            episodes_file = variant_dir / "meta" / "episodes.jsonl"
            with open(episodes_file, 'w') as f:
                for episode in task_episodes:
                    f.write(json.dumps(episode) + '\n')
            
            print(f"  Created {variant}: {episodes_file} with {len(task_episodes)} episodes")
            
            # Copy or symlink videos
            source_videos_dir = source_dataset / "videos"
            dest_videos_dir = variant_dir / "videos"
            
            if source_videos_dir.exists():
                # Get video filenames for these episodes
                for ep in task_episodes:
                    ep_idx = ep["episode_index"]
                    video_file = f"episode_{ep_idx}.mp4"
                    source_video = source_videos_dir / video_file
                    
                    # Try common video formats
                    for ext in ['mp4', 'avi', 'mov', 'mkv']:
                        source_video = source_videos_dir / f"episode_{ep_idx}.{ext}"
                        if source_video.exists():
                            dest_video = dest_videos_dir / source_video.name
                            # Symlink to save space
                            if not dest_video.exists():
                                try:
                                    os.symlink(source_video, dest_video)
                                except Exception as e:
                                    print(f"    Warning: Could not symlink {source_video.name}: {e}")
                            break


def create_metadata_files(dest_dataset):
    """Create tasks.jsonl and info.json for each L3 variation directory."""
    
    for task_dir in dest_dataset.iterdir():
        if task_dir.is_dir():
            # Create tasks.jsonl (one entry per unique task)
            tasks_file = task_dir / "meta" / "tasks.jsonl"
            task_name = task_dir.name.replace("_syn_l3_v", " ").rsplit(' ', 1)[0].replace('_', ' ')
            
            task_entry = {
                "task_index": 0,
                "task": task_name
            }
            
            with open(tasks_file, 'w') as f:
                f.write(json.dumps(task_entry) + '\n')
            
            # Create info.json
            info_file = task_dir / "meta" / "info.json"
            info = {
                "name": task_dir.name,
                "description": f"L3 syntactic variation: {task_name}",
                "fps": 30,
                "features": ["observation.images", "observation.state", "action"]
            }
            
            with open(info_file, 'w') as f:
                json.dump(info, f, indent=2)


def main():
    print("=" * 70)
    print("LIBERO-Goal L3 Dataset Organization Script")
    print("=" * 70)
    
    # Verify source dataset exists
    if not SOURCE_DATASET.exists():
        print(f"ERROR: Source dataset not found at {SOURCE_DATASET}")
        return
    
    print(f"\nSource dataset: {SOURCE_DATASET}")
    print(f"Destination: {DEST_DATASET}")
    
    # Load episodes from source
    print("\nLoading source episodes...")
    episodes = load_episodes(SOURCE_DATASET)
    print(f"Loaded {len(episodes)} total episodes")
    
    # Count episodes per task
    print("\nEpisodes per task:")
    task_counts = defaultdict(int)
    for ep in episodes:
        for task in ep["tasks"]:
            task_counts[task] += 1
    
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")
    
    # Create directory structure
    print("\n" + "=" * 70)
    print("Creating L3 variation directories...")
    print("=" * 70)
    create_l3_directories(SOURCE_DATASET, DEST_DATASET, TASK_TO_L3_VARIATIONS, episodes)
    
    # Create metadata files
    print("\n" + "=" * 70)
    print("Creating metadata files...")
    print("=" * 70)
    create_metadata_files(DEST_DATASET)
    
    print("\n" + "=" * 70)
    print("✓ Dataset organization complete!")
    print("=" * 70)
    print(f"\nCreated {len(TASK_TO_L3_VARIATIONS) * sum(1 for v in TASK_TO_L3_VARIATIONS.values())} L3 task variations")
    print(f"Location: {DEST_DATASET}")


if __name__ == "__main__":
    main()
