from .base import DemoData

import os
from .factory import DataFactory

current_dir = os.path.dirname(os.path.abspath(__file__))
base_package = __name__

DataFactory.auto_register_data(current_dir, base_package)
