import glob
import os
from pathlib import Path
from typing import Sequence
from omegaconf import OmegaConf

from InternVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from InternVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from InternVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from InternVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag

def make_libero_l3_mixture(data_root_dir: str) -> list:
    """
    Generates the dataset mixture for LIBERO Goal L3 fine-tuning.
    Scans the data_root_dir for subdirectories corresponding to each L3 task variation.
    Each variation should have its own directory with the correct videos for that variation.
    """
    root = Path(data_root_dir)
    mixture_spec = []
    
    # Verify that the root directory exists
    if not root.exists():
        print(f"Warning: Data root directory {data_root_dir} does not exist.")
        return mixture_spec
    
    # Expected L3 task variations (ensuring each maps to its own dataset directory)
    expected_l3_variations = [
        # Task 1: "Open the middle layer of the drawer" (2 variations)
        "open_the_middle_layer_of_the_drawer_syn_l3_v1",
        "open_the_middle_layer_of_the_drawer_syn_l3_v2",
        
        # Task 2: "Put the bowl on the stove" (3 variations)
        "put_the_bowl_on_the_stove_syn_l3_v1",
        "put_the_bowl_on_the_stove_syn_l3_v2",
        "put_the_bowl_on_the_stove_syn_l3_v3",
        
        # Task 3: "Put the wine bottle on the top of the cabinet" (3 variations)
        "put_the_wine_bottle_on_the_top_of_the_cabinet_syn_l3_v1",
        "put_the_wine_bottle_on_the_top_of_the_cabinet_syn_l3_v2",
        "put_the_wine_bottle_on_the_top_of_the_cabinet_syn_l3_v3",
        
        # Task 4: "Open the top drawer and put the bowl inside" (3 variations)
        "open_the_top_drawer_and_put_the_bowl_inside_syn_l3_v1",
        "open_the_top_drawer_and_put_the_bowl_inside_syn_l3_v2",
        "open_the_top_drawer_and_put_the_bowl_inside_syn_l3_v3",
        
        # Task 5: "Put the bowl on the top of the cabinet" (3 variations)
        "put_the_bowl_on_the_top_of_the_cabinet_syn_l3_v1",
        "put_the_bowl_on_the_top_of_the_cabinet_syn_l3_v2",
        "put_the_bowl_on_the_top_of_the_cabinet_syn_l3_v3",
        
        # Task 6: "Push the plate to the front of the stove" (3 variations)
        "push_the_plate_to_the_front_of_the_stove_syn_l3_v1",
        "push_the_plate_to_the_front_of_the_stove_syn_l3_v2",
        "push_the_plate_to_the_front_of_the_stove_syn_l3_v3",
        
        # Task 7: "Put the cream cheese on the bowl" (2 variations)
        "put_the_cream_cheese_on_the_bowl_syn_l3_v1",
        "put_the_cream_cheese_on_the_bowl_syn_l3_v2",
        
        # Task 8: "Turn on the stove" (2 variations)
        "turn_on_the_stove_syn_l3_v1",
        "turn_on_the_stove_syn_l3_v2",
        
        # Task 9: "Put the bowl on the plate" (3 variations)
        "put_the_bowl_on_the_plate_syn_l3_v1",
        "put_the_bowl_on_the_plate_syn_l3_v2",
        "put_the_bowl_on_the_plate_syn_l3_v3",
        
        # Task 10: "Put the wine bottle on the rack" (3 variations)
        "put_the_wine_bottle_on_the_rack_syn_l3_v1",
        "put_the_wine_bottle_on_the_rack_syn_l3_v2",
        "put_the_wine_bottle_on_the_rack_syn_l3_v3",
    ]
    
    # Check which variations exist as directories in data_root_dir
    for variation_name in expected_l3_variations:
        variation_path = root / variation_name
        if variation_path.is_dir():
            mixture_spec.append((variation_name, 1.0, "libero_goal_l3_finetune"))
        else:
            print(f"Warning: Expected directory {variation_path} not found. Each L3 variation should have its own directory.")
    
    print(f"Found {len(mixture_spec)} L3 task variation directories in {data_root_dir}.")
    return mixture_spec

def collate_fn(batch):
    """Collate function for LeRobot datasets. Filters out None values."""
    # Filter out any None values (which might occur if a sample fails to load)
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        print("WARNING: All items in batch are None!")
        return None
    return batch

def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    max_episodes_per_task: int = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param delete_pause_frame: Whether to delete pause frames.
    :param max_episodes_per_task: Max episodes to load per task.
    :return: A LeRobotSingleDataset object.
    """
    
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    
    # Pass max_episodes_per_task to the dataset constructor
    # NOTE: This requires LeRobotSingleDataset's __init__ to be modified to accept this argument.
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend="torchvision_av",
        delete_pause_frame=delete_pause_frame,
        max_episodes=max_episodes_per_task, # Assuming LeRobotSingleDataset accepts `max_episodes`
    )

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    delete_pause_frame: bool = True,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    max_episodes_per_task = None

    if data_mix == "libero_goal_l3_finetune":
        max_episodes_per_task = getattr(data_cfg, "max_episodes_per_task", None)
        if max_episodes_per_task is None:
            raise ValueError("`max_episodes_per_task` must be set in the config for `libero_goal_l3_finetune`")
        
        print(f"Building `libero_goal_l3_finetune` mixture with max {max_episodes_per_task} episodes per task.")
        mixture_spec = make_libero_l3_mixture(data_root_dir)
    else:
        mixture_spec = DATASET_NAMED_MIXTURES[data_mix]

    return LeRobotMixtureDataset(
        data_root_dir=data_root_dir,
        mixture_spec=mixture_spec,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        delete_pause_frame=delete_pause_frame,
        max_episodes_per_task=max_episodes_per_task,
    )

if __name__ == "__main__":
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./InternVLA/config/training/internvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    if os.environ.get("DEBUG", None):
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    vla_dataset_cfg = cfg.datasets.vla_data
    dataset = get_vla_dataset(
        data_cfg=vla_dataset_cfg,
        seed=cfg.seed,
    )
    
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=16, # For Debug
        collate_fn=collate_fn,
    )

    from tqdm import tqdm
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        print(batch)
        pass