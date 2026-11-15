"""Crop-only model lifecycle, with identical inference calls to the live service."""
import gc
from pathlib import Path
import threading
import numpy as np
import torch

RUN_LOCK = threading.Lock()


class CropService:
    _engine = None

    def __init__(self):
        from .crop.master_head_engine_v100 import MasterHeadEngineV100
        from .paths import asset_path
        if CropService._engine is None:
            CropService._engine = MasterHeadEngineV100(models_dir=asset_path("yoloface_8n.onnx").parent)
        self.engine = CropService._engine

    def activate(self, device):
        if device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but unavailable')
            try:
                import comfy.model_management as mm
            except ImportError:
                mm = None
            if mm is not None:
                mm.unload_all_models()
                mm.soft_empty_cache()
        self.engine.compute_device = torch.device(device)
        self.engine.birefnet.to(device=device, dtype=torch.float32)

    def process(self, rgb):
        return self.engine.process(np.ascontiguousarray(rgb[..., ::-1]))

    def offload(self):
        self.engine.birefnet.to('cpu')
        self.engine.compute_device = torch.device('cpu')
        self.engine.last_state = None
        self.engine.hood_mask = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
