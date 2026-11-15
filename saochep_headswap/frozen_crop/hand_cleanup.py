"""Find hands by anatomy, then remove only SAM masks anchored to those hands."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch


class HandLandmarks:
    def __init__(self, model_path):
        import mediapipe as mp
        self.mp = mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            num_hands=4, min_hand_detection_confidence=.3, min_hand_presence_confidence=.35,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def locate(self, rgb):
        height, width = rgb.shape[:2]
        frame = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        detected = self.detector.detect(frame)
        guides = []
        for landmarks in detected.hand_landmarks:
            points = np.array([[p.x*width, p.y*height] for p in landmarks], np.float32)
            if points.shape != (21, 2) or not np.isfinite(points).all():
                continue
            left, top = points.min(axis=0)
            right, bottom = points.max(axis=0)
            pad = max(right-left, bottom-top)*.16
            left, top = max(0., left-pad), max(0., top-pad)
            right, bottom = min(float(width), right+pad), min(float(height), bottom+pad)
            if right-left < 3 or bottom-top < 3:
                continue
            box = {'x': float(left), 'y': float(top), 'width': float(right-left), 'height': float(bottom-top)}
            guides.append({'box': box, 'points': points.tolist()})
        return guides


class SaoChepHandGuides:
    _model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'max_pixels': ('INT', {'default': 6000000, 'min': 65536, 'max': 24000000})}}

    RETURN_TYPES = ('BOUNDING_BOX', 'STRING')
    RETURN_NAMES = ('hand_boxes', 'hand_landmarks_json')
    FUNCTION = 'locate'
    CATEGORY = 'SaoChep/Crop Head'

    def locate(self, image, max_pixels):
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
            raise ValueError('Expected one RGB image for hand detection')
        height, width = image.shape[1:3]
        if height*width > max_pixels or not torch.isfinite(image).all():
            raise ValueError('Input exceeds hand detection size limit or contains non-finite values')
        if self.__class__._model is None:
            from .paths import asset_path
            self.__class__._model = HandLandmarks(asset_path('hand_landmarker.task'))
        rgb = np.rint(image[0].detach().cpu().numpy().clip(0, 1)*255).astype(np.uint8)
        guides = self._model.locate(rgb)
        return [guide['box'] for guide in guides], json.dumps(guides)


def remove_confirmed_hands(alpha, geometry, state, proposed, guides):
    """Keep accessories unless independently verified hand evidence conflicts."""
    height, width = alpha.shape
    if proposed.shape != alpha.shape:
        raise ValueError('Hand mask must match image size')
    if not guides:
        return alpha, {'hand_status': 'no_anatomical_detection', 'hand_pixels_removed': 0}

    visible = alpha > .05
    remove = np.zeros(alpha.shape, bool)
    face_width = float(geometry['fw'])
    face_area = max(1., face_width*float(geometry['fh']))
    for guide in guides:
        points = np.rint(guide['points']).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width-1)
        points[:, 1] = np.clip(points[:, 1], 0, height-1)
        box = guide['box']
        left, top = max(0, int(box['x'])), max(0, int(box['y']))
        right = min(width, int(np.ceil(box['x']+box['width'])))
        bottom = min(height, int(np.ceil(box['y']+box['height'])))
        region = np.s_[top:bottom, left:right]
        proposal = proposed[region] > .5
        if not proposal.size:
            continue
        _, labels = cv2.connectedComponents(proposal.astype(np.uint8), connectivity=8)
        in_box = (points[:, 0] >= left) & (points[:, 0] < right) & (points[:, 1] >= top) & (points[:, 1] < bottom)
        x, y = points[in_box].T
        anchor_labels = labels[y-top, x-left]
        counts = np.bincount(anchor_labels, minlength=1)
        accepted = np.flatnonzero(counts >= 4)
        confirmed = np.isin(labels, accepted[accepted != 0])
        # Anatomical landmarks near the actual crop cannot be silently
        # ignored when the guided segmenter fails to supply a coherent mask.
        if not confirmed.any() and np.count_nonzero(visible[y, x]) >= 3:
            raise ValueError('HAND_MASK_UNCERTAIN: use a reference with the hand away from the head')
        remove[region] |= confirmed

    affected = remove & (alpha > 0)
    if not affected.any():
        return alpha, {'hand_status': 'hands_outside_crop', 'hand_pixels_removed': 0}

    # A hand can legitimately be in front of the face or an item. Refuse a
    # conflicted crop instead of feeding Wan a face/item with missing pixels.
    radius = max(1, round(face_width*.075))
    feature_zone = np.zeros(alpha.shape, np.uint8)
    for x, y in np.rint(geometry['landmarks']).astype(int):
        cv2.circle(feature_zone, (int(x), int(y)), radius, 1, -1)
    face = state['aligned']['face'] > .75
    if (affected & (feature_zone > 0)).any() or np.count_nonzero(affected & face & visible) > face_area*.018:
        raise ValueError('HAND_OCCLUDES_FACE: use a reference without a hand over the eyes, nose or mouth')
    items = state['owned_headwear'] | np.isin(state['aligned']['labels'], [6, 9])
    if np.count_nonzero(affected & items & visible) > max(3, face_area*.0004):
        raise ValueError('HAND_OVERLAPS_HEAD_ITEM: use a reference where the item is unobstructed')
    # Protect recognized item pixels even along a one-pixel mask overlap.
    affected &= ~items
    result = alpha.copy()
    result[affected] = 0
    return result, {'hand_status': 'removed', 'hand_pixels_removed': int(affected.sum()),
                    'hand_hair_overlap_pixels': int(np.count_nonzero(affected & (state['aligned']['hair'] > .5)))}
