from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import tyro

from dex_retargeting.constants import RobotName, HandType
from dex_retargeting.retargeting_config import RetargetingConfig
from hand_robot_viewer import RobotHandDatasetSAPIENViewer
from hand_viewer import HandDatasetSAPIENViewer

from Franka_RL.dataset import DataFactory
from Franka_RL.robots import DexHandFactory

# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_


def mano2dexhand(dataset: str, data_root: Path, robots: Optional[List[RobotName]], fps: int):
    for robot in robots:
        dexhand = DexHandFactory.create_robot(robot)
        dataset = DataFactory.create_data(dataset, data_path=data_root, dexhand=dexhand)
        viewer = RobotHandDatasetSAPIENViewer(list([robot]), HandType.right if dexhand.side == "right" else HandType.left, headless=True, use_ray_tracing=True)

        # Data ID, feel free to change it to visualize different trajectory
        data_id = 4

        sampled_data = dataset[data_id]
        for key, value in sampled_data.items():
            if "pose" not in key:
                print(f"{key}: {value}")
        viewer.load_object_hand(sampled_data)
        viewer.render_dexycb_data(sampled_data, fps)


def main(dataset: str, dataset_dir: str, robots: Optional[List[RobotName]] = None, fps: int = 10):
    """
    Render the human and robot trajectories for grasping object inside DexYCB dataset.
    The human trajectory is visualized as provided, while the robot trajectory is generated from position retargeting

    Args:
        dexycb_dir: Data root path to the dexycb dataset
        robots: The names of robots to render, if None, render human hand trajectory only
        fps: frequency to render hand-object trajectory

    """
    data_root = Path(dataset_dir).absolute()
    if not data_root.exists():
        raise ValueError(f"Path to DexYCB dir: {data_root} does not exist.")
    else:
        print(f"Using DexYCB dir: {data_root}")

    mano2dexhand(dataset, data_root, robots, fps)


if __name__ == "__main__":
    tyro.cli(main)
