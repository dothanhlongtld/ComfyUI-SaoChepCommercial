"""Independent clothing/skin evidence and pose-aware face boundary for photos."""
from pathlib import Path

import cv2
import numpy as np


class PhotoEvidence:
    def __init__(self, models):
        import mediapipe as mp
        if not hasattr(mp.tasks.vision, 'FaceLandmarksConnections'):
            raise RuntimeError('MediaPipe Tasks face-connection API is required')
        self.mp = mp
        models = Path(models)
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(
            mp.tasks.vision.ImageSegmenterOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(models/'multi.tflite')),
                output_confidence_masks=True))
        self.strand_segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(
            mp.tasks.vision.ImageSegmenterOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(models/'hair.tflite')),
                output_confidence_masks=True))
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(models/'face_landmarker.task')),
                num_faces=1))

    def predict_strands(self, rgb):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        masks = self.strand_segmenter.segment(image).confidence_masks
        if len(masks) != 2:
            raise ValueError('Expected two classes in the hair-specific model')
        return masks[1].numpy_view().squeeze().copy()

    def predict(self, rgb, geometry):
        mp = self.mp
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        probabilities = [mask.numpy_view().squeeze().copy()
                         for mask in self.segmenter.segment(image).confidence_masks]
        if len(probabilities) != 6:
            raise ValueError('Expected the six-class selfie model')
        width, height = float(geometry['fw']), float(geometry['fh'])
        points = np.asarray(geometry['landmarks'], np.float32)
        eyes = points[:2].mean(axis=0)
        left, top = max(0, int(eyes[0]-width)), max(0, int(eyes[1]-height*.65))
        right, bottom = min(rgb.shape[1], int(eyes[0]+width)), min(rgb.shape[0], int(eyes[1]+height*1.35))
        outline = None
        crop = rgb[top:bottom, left:right]
        if crop.size:
            result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
            if result.face_landmarks:
                mesh = np.array([[left+p.x*(right-left), top+p.y*(bottom-top)]
                                 for p in result.face_landmarks[0]], np.float32)
                mesh_eyes = np.stack([(mesh[33]+mesh[133])/2, (mesh[362]+mesh[263])/2])
                alignment_error = min(np.linalg.norm(mesh_eyes-points[:2], axis=1).mean(),
                                      np.linalg.norm(mesh_eyes[::-1]-points[:2], axis=1).mean())
                if alignment_error < width*.15:
                    ids = [edge.start for edge in mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL]
                    mask = np.zeros(rgb.shape[:2], np.uint8)
                    cv2.fillPoly(mask, [np.rint(mesh[ids]).astype(np.int32)], 1)
                    outline = mask > 0
        return probabilities, outline
