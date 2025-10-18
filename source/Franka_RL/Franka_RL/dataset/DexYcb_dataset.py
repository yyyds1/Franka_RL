from .base import DexhandData
from .decorators import register_dataset

@register_dataset("DexYcb")
class DexYcbDataset(DexhandData):
    def __init__(
        self, 
        *, 
        data_path: str = '', 
        device = 'cpu', 
        dexhand=None,
        **kwargs,
    ):
        super().__init__(data_path=data_path, device=device, dexhand=dexhand)