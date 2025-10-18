from .base import DexHand

import os
from .factory import RobotFactory

current_dir = os.path.dirname(os.path.abspath(__file__))
base_package = __name__

RobotFactory.auto_register_robots(current_dir, base_package)
