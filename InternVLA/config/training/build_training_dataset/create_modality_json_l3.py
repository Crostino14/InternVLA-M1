#!/usr/bin/env python3
"""
Generate modality.json for each L3 task variation.
Based on LIBERO Goal action and observation structure.
"""

import json
from pathlib import Path

DEST_DATASET = Path("/mnt/beegfs/a.cardamone7/datasets/lerobot_libero_goal_l3")

# ModaLity metadata template for LIBERO Goal (7-dim action, state, and video)
MODALITY_TEMPLATE = {
    "state": {
        "observation.position": {
            "start": 0,
            "end": 7,
            "dtype": "float64",
            "absolute": True,
            "rotation_type": None,
            "original_key": "observation.state"
        },
        "observation.velocity": {
            "start": 7,
            "end": 14,
            "dtype": "float64",
            "absolute": True,
            "rotation_type": None,
            "original_key": "observation.state"
        },
        "observation.effort": {
            "start": 14,
            "end": 21,
            "dtype": "float64",
            "absolute": False,
            "rotation_type": None,
            "original_key": "observation.state"
        },
        "observation.gripper": {
            "start": 21,
            "end": 22,
            "dtype": "float64",
            "absolute": False,
            "rotation_type": None,
            "original_key": "observation.state"
        }
    },
    "action": {
        "action": {
            "start": 0,
            "end": 7,
            "dtype": "float64",
            "absolute": False,
            "rotation_type": None,
            "original_key": "action"
        }
    },
    "video": {
        "observation.images.image": {
            "start": 0,
            "end": 1,
            "dtype": "video",
            "absolute": True,
            "rotation_type": None,
            "original_key": "observation.images.image"
        },
        "observation.images.wrist_image": {
            "start": 1,
            "end": 2,
            "dtype": "video",
            "absolute": True,
            "rotation_type": None,
            "original_key": "observation.images.wrist_image"
        }
    }
}


def create_modality_json_for_variations():
    """Create modality.json for each L3 variation."""
    
    print("=" * 80)
    print("Creating meta/modality.json for all L3 variations")
    print("=" * 80)
    
    variation_dirs = sorted([d for d in DEST_DATASET.iterdir() if d.is_dir()])
    created = 0
    
    for i, variation_dir in enumerate(variation_dirs, 1):
        meta_dir = variation_dir / "meta"
        modality_file = meta_dir / "modality.json"
        
        # Create modality.json
        with open(modality_file, 'w') as f:
            json.dump(MODALITY_TEMPLATE, f, indent=2)
        
        created += 1
        if i % 5 == 0 or i == len(variation_dirs):
            print(f"[{i:2d}/{len(variation_dirs)}] Created modality.json for {variation_dir.name}")
    
    print("\n" + "=" * 80)
    print(f"✓ Created {created} modality.json files")
    print("=" * 80)


if __name__ == "__main__":
    create_modality_json_for_variations()
