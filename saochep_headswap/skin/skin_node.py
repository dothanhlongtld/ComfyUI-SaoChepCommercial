"""CPU face parsing and driving-skin color matching with existing ONNX models."""
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .aligned_parse import aligned_parse
from .driving_skin_color import skin_statistics, driving_offsets, apply_skin_offset, bounded_skin_offset


class AGDrivingSkinColor:
    _models = None

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "driving_images": ("IMAGE",),
                             "actor_hint": ("MASK",), "fps": ("INT", {"default": 24, "min": 1, "max": 60})},
                "optional":{"color_mode":(["conservative", "full_transfer"],{"default":"conservative"})}}

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "restore"
    CATEGORY = "SaoChep/Whole head"

    @classmethod
    def models(cls):
        if cls._models is None:
            import folder_paths
            import onnxruntime as ort
            from ..frozen_crop.paths import model_roots
            folders = [Path(__file__).parent, *model_roots(), Path(folder_paths.models_dir)/"detection"]
            def find(*names):
                paths = [folder/name for folder in folders for name in names]
                path = next((path for path in paths if path.is_file()), None)
                if path is None:
                    raise RuntimeError("Required existing model missing: "+", ".join(names)+
                                       "\nSearched:\n"+"\n".join(str(path) for path in paths))
                return str(path)
            options = ort.SessionOptions()
            options.intra_op_num_threads = min(4, os.cpu_count() or 1)
            parser = ort.InferenceSession(find("bisenet_face_parsing.onnx", "bisenet.onnx"), options,
                                          providers=["CPUExecutionProvider"])
            detector = cv2.FaceDetectorYN.create(find("face_detection_yunet_2023mar.onnx"), "", (576, 1024), score_threshold=.45)
            cls._models = detector, parser
        return cls._models

    def parse(self, rgb, hint):
        detector, session = self.models()
        detector.setInputSize((rgb.shape[1], rgb.shape[0]))
        _, faces = detector.detect(np.ascontiguousarray(rgb[..., ::-1]))
        if faces is None:
            return None
        def overlap(face):
            x, y, w, h = face[:4]
            left, top = max(0, int(x)), max(0, int(y))
            patch = hint[top:min(len(hint), int(y+h)), left:min(hint.shape[1], int(x+w))]
            return float(patch.mean()) if patch.size else 0.
        ranked = sorted(((overlap(face), face) for face in faces), key=lambda item: item[0], reverse=True)
        if ranked[0][0] < .15:
            return None
        face = ranked[0][1]
        parts = aligned_parse(rgb, face[4:14].reshape(5, 2), face[2], face[3], session,
            np.array([.485, .456, .406], np.float32), np.array([.229, .224, .225], np.float32))
        skin = (parts["labels"] == 1).astype(np.float32)
        permitted = np.isin(parts["labels"], [1, 7, 8, 10]).astype(np.uint8)
        weight = np.clip(cv2.distanceTransform(permitted, cv2.DIST_L2, 3)/3, 0, 1)
        return skin_statistics(rgb, skin), weight

    def restore(self, images, driving_images, actor_hint, fps=24, color_mode='conservative'):
        import torch
        if images.shape != driving_images.shape or actor_hint.shape != images.shape[:3]:
            raise ValueError("Skin restore inputs must share frame timing and dimensions")
        source_stats, candidate_stats, weights = [], [], []
        for source, candidate, hint in zip(driving_images, images, actor_hint):
            original, generated = [np.rint(image.detach().cpu().numpy().clip(0, 1)*255).astype(np.uint8)
                                   for image in (source, candidate)]
            hint = hint.detach().cpu().numpy()
            src, gen = self.parse(original, hint), self.parse(generated, hint)
            source_stats.append(src[0] if src is not None else None)
            candidate_stats.append(gen[0] if gen is not None else None)
            weights.append((gen[1] if gen is not None else np.zeros(hint.shape, np.float32)).astype(np.float16))
        offsets, report = driving_offsets(source_stats, candidate_stats, [0, len(images)], fps=fps)
        if color_mode not in ('conservative','full_transfer'):
            raise ValueError('Unknown skin color mode')
        if color_mode=='conservative':
            offsets=np.stack([bounded_skin_offset(delta) for delta in offsets])
        output = torch.empty_like(images, device="cpu")
        for i, image in enumerate(images):
            rgb = np.rint(image.detach().cpu().numpy().clip(0, 1)*255).astype(np.uint8)
            weight = weights[i].astype(np.float32)
            if color_mode=='conservative':
                weight=cv2.GaussianBlur(weight,(0,0),2)*weight
            corrected = apply_skin_offset(rgb, weight, weight > 0, offsets[i])
            output[i] = torch.from_numpy(corrected.astype(np.float32)/255)
        report.update(target="driving_actor_skin", color_mode=color_mode, frame_count=len(images),
                      lab_offsets=offsets.tolist(), state="review_required" if report["invalid_frames"] or report["clamped_frames"] else "candidate_unverified",
                      outside_skin_pixels_preserved_before_encoding=True)
        return output, json.dumps(report)
