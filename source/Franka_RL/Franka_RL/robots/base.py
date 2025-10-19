from abc import ABC, abstractmethod
import os
import numpy as np


class DexHand(ABC):
    def __init__(self):
        self._usd_path = None
        self.side = None
        self.name = None
        self.body_names = None
        self.dof_names = None
        self.wrist_name = None
        self.contact_body_names = None
        self.init_state = None
        self.actuators = None
        self.dof_limit = None
        self.weight_idx = None
        self.retargeting_cfg = None

    
    @property
    def n_dofs(self):
        return len(self.dof_names)

    @property
    def n_bodies(self):
        return len(self.body_names)