"""
SaoChepCommercial - Custom Nodes for ComfyUI Commercial Production
Core active nodes (ReferenceChromaLightTest, SaoChepCommercialExport) are imported directly;
legacy/experimental nodes are guarded with optional import blocks (Audit V3 Section 13).
"""
import json
import re
import threading
from pathlib import Path
import torch
from PIL import Image
import numpy as np

from .reference_chroma_light_node import ReferenceChromaLightTest
from .commercial_naturalize_export_node import SaoChepCommercialExport
from .adaptive_clarity_enhancer_node import SaoChepAdaptiveClarityEnhancer
from .shot_geometry_node import SaoChepShotGeometry
from .ocular_steering_node import SaoChepOcularSteering

_LOCK = threading.RLock()


class SaoChepReferenceChroma(ReferenceChromaLightTest):
    OUTPUT_NODE = False

    def apply_chroma_test(self, images, reference_image, **kwargs):
        with _LOCK:
            if len(images) == 0 or len(reference_image) != 1:
                raise ValueError("Expected non-empty video and exactly one reference image")
            return super().apply_chroma_test(images, reference_image, **kwargs)


class SaoChepDeliveryResize:
    CATEGORY = "SaoChep/Commercial"
    FUNCTION = "resize"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "width": ("INT", {"default": 720, "min": 64, "max": 2160, "step": 2}),
                "height": ("INT", {"default": 1280, "min": 64, "max": 3840, "step": 2}),
            }
        }

    def resize(self, images, width, height):
        if images.shape[1:3] == (height, width):
            return (images,)
        if abs(images.shape[2] / images.shape[1] - width / height) > 0.002:
            raise ValueError("Delivery resize must preserve aspect ratio")
        frames = []
        for frame in images:
            u8 = (frame.detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
            resized = Image.fromarray(u8).resize((width, height), Image.Resampling.LANCZOS)
            frames.append(torch.from_numpy(np.asarray(resized).copy()).float() / 255)
        return (torch.stack(frames),)


class SaoChepSaveReport:
    CATEGORY = "SaoChep/Commercial"
    FUNCTION = "save"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "saochep/report"}),
            }
        }

    def save(self, text, filename_prefix):
        import folder_paths
        if not re.fullmatch(r"[A-Za-z0-9_/-]+", filename_prefix) or ".." in filename_prefix:
            raise ValueError("Unsafe report prefix")
        root = Path(folder_paths.get_output_directory()).resolve()
        path = (root / (filename_prefix + ".json")).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Report path escapes output directory")
        data = json.loads(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {
            "ui": {
                "text": [text],
                "files": [
                    {
                        "filename": path.name,
                        "subfolder": str(path.parent.relative_to(root)).replace(chr(92), "/"),
                        "type": "output",
                    }
                ],
            }
        }


NODE_CLASS_MAPPINGS = {
    "ReferenceChromaLightTest": ReferenceChromaLightTest,
    "SaoChepCommercialExport": SaoChepCommercialExport,
    "SaoChepAdaptiveClarityEnhancer": SaoChepAdaptiveClarityEnhancer,
    "SaoChepReferenceChroma": SaoChepReferenceChroma,
    "SaoChepDeliveryResize": SaoChepDeliveryResize,
    "SaoChepSaveReport": SaoChepSaveReport,
    "SaoChepShotGeometry": SaoChepShotGeometry,
    "SaoChepOcularSteering": SaoChepOcularSteering,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReferenceChromaLightTest": "Reference Chroma Light Test (Color Restore Node 9000)",
    "SaoChepCommercialExport": "SaoChep Commercial Master Export (100% On-Flow Naturalize Master)",
    "SaoChepAdaptiveClarityEnhancer": "SaoChep Adaptive Clarity Enhancer (ByteDance CapCut On-Flow Node 2704)",
    "SaoChepReferenceChroma": "SaoChep — Reference color (opt-in)",
    "SaoChepDeliveryResize": "SaoChep — Delivery resize",
    "SaoChepSaveReport": "SaoChep — Save report",
    "SaoChepShotGeometry": "SaoChep Shot Geometry & Floor Inpaint (On-Flow Preflight Node 9010)",
    "SaoChepOcularSteering": "SaoChep Ocular Steering & Anti-Glare (On-Flow Gaze Lock Node 9020)",
}

# Guarded optional imports for experimental/legacy nodes (Audit V3 Section 13)
try:
    from .auto_reference_grade_node import AutoReferenceGrade
    NODE_CLASS_MAPPINGS["AutoReferenceGrade"] = AutoReferenceGrade
    NODE_DISPLAY_NAME_MAPPINGS["AutoReferenceGrade"] = "Auto Reference Grade (Commercial Grade Node 9003)"
except Exception:
    pass

try:
    from .adaptive_face_restore_node import SaoChepAdaptiveFaceRestore
    NODE_CLASS_MAPPINGS["SaoChepAdaptiveFaceRestore"] = SaoChepAdaptiveFaceRestore
    NODE_DISPLAY_NAME_MAPPINGS["SaoChepAdaptiveFaceRestore"] = "SaoChep Adaptive Face Restore"
except Exception:
    pass

try:
    from .smart_crop_node import SaoChepSmartFaceCrop
    NODE_CLASS_MAPPINGS["SaoChepSmartFaceCrop"] = SaoChepSmartFaceCrop
    NODE_DISPLAY_NAME_MAPPINGS["SaoChepSmartFaceCrop"] = "SaoChep Smart Face Crop 9:16"
except Exception:
    pass

try:
    from .vision_prompt_node import SaoChepVisionPrompt
    NODE_CLASS_MAPPINGS["SaoChepVisionPrompt"] = SaoChepVisionPrompt
    NODE_DISPLAY_NAME_MAPPINGS["SaoChepVisionPrompt"] = "SaoChep Vision Prompt"
except Exception:
    pass

try:
    from .semantic_observer_node import SaoChepSemanticObserver
    NODE_CLASS_MAPPINGS["SaoChepSemanticObserver"] = SaoChepSemanticObserver
    NODE_DISPLAY_NAME_MAPPINGS["SaoChepSemanticObserver"] = "SaoChep Semantic Observer (VLM Google Drive)"
except Exception:
    pass

try:
    from .multi_char_identity_lock_node import SaoChepMultiCharIdentityLock
    NODE_CLASS_MAPPINGS["SaoChepMultiCharIdentityLock"] = SaoChepMultiCharIdentityLock
    NODE_DISPLAY_NAME_MAPPINGS["SaoChepMultiCharIdentityLock"] = "SaoChep Multi-Character Identity Lock (On-Flow)"
except Exception:
    pass

try:
    from .adaptive_scene_router_node import SaoChepAdaptiveSceneRouter, SaoChepAdaptiveTimelineMerger
    NODE_CLASS_MAPPINGS["SaoChepAdaptiveSceneRouter"] = SaoChepAdaptiveSceneRouter
    NODE_CLASS_MAPPINGS["SaoChepAdaptiveTimelineMerger"] = SaoChepAdaptiveTimelineMerger
    NODE_DISPLAY_NAME_MAPPINGS["SaoChepAdaptiveSceneRouter"] = "SaoChep Adaptive Scene Router (100% On-Flow Node 2700)"
    NODE_DISPLAY_NAME_MAPPINGS["SaoChepAdaptiveTimelineMerger"] = "SaoChep Adaptive Timeline Merger (100% On-Flow Node 2701)"
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


