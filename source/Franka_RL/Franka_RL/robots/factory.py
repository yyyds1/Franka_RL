import os
import importlib
from .base import DexHand


class RobotFactory:
    _registry = {}

    @classmethod
    def register(cls, robot_type: str, hand_class):
        """Register a new hand type."""
        cls._registry[robot_type] = hand_class

    @classmethod
    def create_robot(cls, robot_type: str, *args, **kwargs) -> DexHand:
        # assert side in ["left", "right"], f"Invalid side '{side}', must be 'left' or 'right'."
        # """Create a hand instance by type."""
        # dexhand_type += "_rh" if side == "right" else "_lh"
        if robot_type not in cls._registry:
            raise ValueError(f"Hand type '{robot_type}' not registered.")
        return cls._registry[robot_type](*args, **kwargs)

    @classmethod
    def auto_register_robots(cls, directory: str, base_package: str):
        """Automatically import all hand modules in the directory."""
        for filename in os.listdir(directory):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = f"{base_package}.{filename[:-3]}"
                importlib.import_module(module_name)
