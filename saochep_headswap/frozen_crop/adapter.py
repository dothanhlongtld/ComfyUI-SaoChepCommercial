"""Stable Comfy ports plus native, transparent head-only output."""
import hashlib
import json
from pathlib import Path
import uuid

import cv2
import numpy as np
from PIL import Image
import torch

from .crop_pipeline_v101 import reference_without_neck
from .legacy_crop import legacy_reference_layout
from .runtime import CropService, RUN_LOCK


def native_rgba(result):
    rgba = result['head_rgba']
    visible = rgba[..., 3] != 0
    y = np.flatnonzero(visible.any(axis=1))
    x = np.flatnonzero(visible.any(axis=0))
    if x.size == 0:
        raise ValueError('Crop output is empty')
    # The legacy bbox is inclusive, so retain the last visible row/column.
    crop = rgba[y[0]:y[-1]+1, x[0]:x[-1]+1].copy()
    crop[crop[..., 3] == 0, :3] = 0
    return crop


class SaoChepCropHeadHandSafe:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            'required': {'image': ('IMAGE',), 'hood_mask': ('MASK',), 'hand_mask': ('MASK',),
                         'hand_landmarks_json': ('STRING', {'forceInput': True})},
            'optional': {
                'clip_bg_mode': (['black', 'midgray'], {'default': 'black'}),
                'target_width': ('INT', {'default': 576, 'min': 256, 'max': 2048}),
                'target_height': ('INT', {'default': 1024, 'min': 256, 'max': 2048}),
                'matting_device': (['auto', 'cuda', 'cpu'], {'default': 'auto'}),
                'max_pixels': ('INT', {'default': 6000000, 'min': 65536, 'max': 24000000}),
                'min_side': ('INT', {'default': 64, 'min': 32, 'max': 512}),
                'subject': (['human', 'legacy_animal_fallback'], {'default': 'human'}),
                'save_diagnostics': ('BOOLEAN', {'default': False}),
                'driving_images': ('IMAGE',),
                'layout_mode': (['legacy', 'driving_head'], {'default': 'legacy'}),
            },
        }
        return inputs

    RETURN_TYPES = ('IMAGE', 'IMAGE', 'MASK', 'IMAGE', 'MASK', 'STRING', 'IMAGE')
    RETURN_NAMES = ('clip_rgb', 'reference_rgb_hq', 'reference_alpha_hq', 'tight_crop_rgb_hq',
                    'tight_crop_alpha_hq', 'report_json', 'native_crop_rgba')
    FUNCTION = 'isolate'
    CATEGORY = 'SaoChep/Crop Head'

    def isolate(self, image, hood_mask, hand_mask, hand_landmarks_json, clip_bg_mode='black', target_width=576, target_height=1024,
                matting_device='auto', max_pixels=6000000, min_side=64, subject='human',
                save_diagnostics=False, driving_images=None, layout_mode='legacy'):
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
            raise ValueError('Expected one RGB image. Submit each image as a separate job.')
        height, width = image.shape[1:3]
        if min(height, width) < min_side or height*width > max_pixels:
            raise ValueError(f'Image size {width}x{height} exceeds the configured input limits')
        if any(tuple(mask.shape) != (1, height, width) for mask in (hood_mask, hand_mask)):
            raise ValueError('Hood and hand masks must each have one plane matching the input image')
        if not all(torch.isfinite(value).all() for value in (image, hood_mask, hand_mask)):
            raise ValueError('Input image or hood mask contains non-finite values')
        device = matting_device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rgb = np.rint(image[0].detach().cpu().numpy().clip(0, 1)*255).astype(np.uint8)
        hood = hood_mask[0].detach().float().cpu().numpy().clip(0, 1)
        hand = hand_mask[0].detach().float().cpu().numpy().clip(0, 1)
        guides = json.loads(hand_landmarks_json)
        canvas_geometry = None
        with RUN_LOCK:
            service = CropService()
            try:
                service.activate(device)
                service.engine.hood_mask = hood
                result = service.engine.process(np.ascontiguousarray(rgb[..., ::-1]),
                                                allow_animal_fallback=subject == 'legacy_animal_fallback')
                result, removed = reference_without_neck(result, rgb, service.engine, hand_mask=hand, hand_guides=guides)
                if layout_mode == 'driving_head' and driving_images is not None:
                    from .outfit_guard import detect_geometry
                    driver = np.rint(driving_images[0].detach().cpu().numpy().clip(0, 1)*255).astype(np.uint8)
                    if driver.shape[:2] != (target_height, target_width):
                        driver = cv2.resize(driver, (target_width, target_height), interpolation=cv2.INTER_AREA)
                    canvas_geometry = detect_geometry(driver, service.engine.yolo_sess)
            finally:
                service.offload()

        rgba = native_rgba(result)
        outputs = list(legacy_reference_layout(result, target_width, target_height, clip_bg_mode))
        if layout_mode == 'driving_head' and canvas_geometry is not None:
            from .core import alignment_transform
            driver_info = {'geometry': canvas_geometry, 'head_bbox': canvas_geometry['box']}
            transform = alignment_transform(result, driver_info)
            matted = np.rint(result['head_fg_rgb'].astype(np.float32)*result['head_alpha'][..., None]).astype(np.uint8)
            outputs[1] = cv2.warpAffine(matted, transform, (target_width, target_height), flags=cv2.INTER_LINEAR, borderValue=0)
            outputs[2] = cv2.warpAffine(result['head_alpha'], transform, (target_width, target_height), flags=cv2.INTER_LINEAR, borderValue=0)

        report = {
            'release': 'v101-r4-handsafe', 'species': result['species'], 'device': device,
            'input_size': [width, height], 'output_size': list(rgba.shape[1::-1]),
            'head_bbox': result['head_bbox'], 'neck_pixels_removed': removed,
            'diagnostics': result.get('diagnostics', {}),
            'reference_canvas_geometry': canvas_geometry,
            'layout_mode': layout_mode,
            'rgba_sha256': hashlib.sha256(rgba.tobytes()).hexdigest(),
        }
        if save_diagnostics:
            import folder_paths
            directory = Path(folder_paths.get_output_directory())/'saochep_crop_diagnostics'
            directory.mkdir(parents=True, exist_ok=True)
            filename = uuid.uuid4().hex+'.png'
            Image.fromarray(result['head_rgba']).save(directory/filename)
            report['diagnostic_path'] = f'saochep_crop_diagnostics/{filename}'
        tensors = tuple(torch.from_numpy(value.astype(np.float32)/(255. if i in (0, 1, 3) else 1.))[None]
                        for i, value in enumerate(outputs))
        return (*tensors, json.dumps(report, ensure_ascii=False), torch.from_numpy(rgba.astype(np.float32)/255.)[None])
