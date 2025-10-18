import os
import importlib
from .base import DemoData


class DataFactory:
    _registry = {}

    @classmethod
    def register(cls, data_type: str, data_class):
        """Register a new data type."""
        cls._registry[data_type] = data_class

    @classmethod
    def create_data(cls, data_type: str, *args, **kwargs) -> DemoData:
        # assert side in ["left", "right"], f"Invalid side '{side}', must be 'left' or 'right'."
        # """Create a data instance by type."""
        # data_type += "_rh" if side == "right" else "_lh"
        if data_type not in cls._registry:
            raise ValueError(f"Data type '{data_type}' not registered.")
        return cls._registry[data_type](*args, **kwargs)

    @classmethod
    def auto_register_data(cls, directory: str, base_package: str):
        """Automatically import all data modules in the directory."""
        for filename in os.listdir(directory):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = f"{base_package}.{filename[:-3]}"
                importlib.import_module(module_name)

    @staticmethod
    def dataset_type(index):
        """We use the index to get the dataset type"""
        """Define your own dataset type and index format here"""

        if type(index) == str and index.endswith("M"):
            # !!! The right hand mirrored dataset comes from the original left hand dataset
            is_mirrored = True
            index = index[:-1]
        else:
            is_mirrored = False

        if type(index) == str and "@" in index:
            dtype = "oakink2"
        elif type(index) == str and index.startswith("g"):
            dtype = "grabdemo"
        elif type(index) == str and index.startswith("v"):
            dtype = "visionpro"
        else:
            dtype = "favor"

        if is_mirrored:
            dtype += "_mirrored"
        return dtype
