"""Isolated commercial nodes; do not shadow upstream node class names."""
import json
import re
import threading
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from .reference_chroma_light_node import ReferenceChromaLightTest

_LOCK = threading.RLock()

class SaoChepReferenceChroma(ReferenceChromaLightTest):
    OUTPUT_NODE = False

    def apply_chroma_test(self, images, reference_image, **kwargs):
        # YuNet singleton has mutable input size. Serialize calls within this process.
        with _LOCK:
            if len(images) == 0 or len(reference_image) != 1:
                raise ValueError('Expected non-empty video and exactly one reference image')
            return super().apply_chroma_test(images, reference_image, **kwargs)

class SaoChepDeliveryResize:
    CATEGORY = 'SaoChep/Commercial'
    FUNCTION = 'resize'
    RETURN_TYPES = ('IMAGE',)

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',), 'width': ('INT', {'default': 720, 'min': 64, 'max': 2160, 'step': 2}),
                             'height': ('INT', {'default': 1280, 'min': 64, 'max': 3840, 'step': 2})}}

    def resize(self, images, width, height):
        if images.shape[1:3] == (height, width):
            return (images,)
        if abs(images.shape[2] / images.shape[1] - width / height) > 0.002:
            raise ValueError('Delivery resize must preserve aspect ratio')
        frames = []
        for frame in images:
            # Match delivered H.264 8-bit pipeline; only spatial resampling, no color grade.
            u8 = (frame.detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
            resized = Image.fromarray(u8).resize((width, height), Image.Resampling.LANCZOS)
            frames.append(torch.from_numpy(np.asarray(resized).copy()).float() / 255)
        return (torch.stack(frames),)

class SaoChepSaveReport:
    CATEGORY = 'SaoChep/Commercial'
    FUNCTION = 'save'
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'text': ('STRING', {'forceInput': True}),
                             'filename_prefix': ('STRING', {'default': 'saochep/report'})}}

    def save(self, text, filename_prefix):
        import folder_paths
        if not re.fullmatch(r'[A-Za-z0-9_/-]+', filename_prefix) or '..' in filename_prefix:
            raise ValueError('Unsafe report prefix')
        root = Path(folder_paths.get_output_directory()).resolve()
        path = (root / (filename_prefix + '.json')).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Report path escapes output directory')
        data = json.loads(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Job-specific prefix, exclusive create: never overwrite another customer's report.
        with path.open('x', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {'ui': {'text': [text], 'files': [{'filename': path.name,
                    'subfolder': str(path.parent.relative_to(root)).replace('\\', '/'), 'type': 'output'}]}}

NODE_CLASS_MAPPINGS = {
    "ReferenceChromaLightTest": ReferenceChromaLightTest,
    "SaoChepReferenceChroma": SaoChepReferenceChroma,
    "SaoChepDeliveryResize": SaoChepDeliveryResize,
    "SaoChepSaveReport": SaoChepSaveReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReferenceChromaLightTest": "Reference Chroma Light Test (Optimized V2.2)",
    "SaoChepReferenceChroma": "SaoChep — Reference color (opt-in)",
    "SaoChepDeliveryResize": "SaoChep — Delivery resolution",
    "SaoChepSaveReport": "SaoChep — Save color telemetry",
}
