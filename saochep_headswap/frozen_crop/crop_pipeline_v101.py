"""Photo-only v101 pipeline, with shared model sessions and explicit invariants."""
from pathlib import Path

import cv2
import numpy as np

from .crop_anatomy import PhotoEvidence
from .hand_cleanup import remove_confirmed_hands
from .crop_jaw_boundary import regularize_jaw
from .crop_neck_ownership import strip_neck
from .crop_semantic_consensus import refine, headwear_mask
from .crop_trimap_matte import TrimapMatte

_evidence = None
_matting = None


def reference_without_neck(result, rgb, engine, *, hand_mask=None, hand_guides=None):
    global _evidence, _matting
    if result.get('geometry') is None:
        clean = dict(result)
        clean['diagnostics'] = {**result.get('diagnostics', {}), 'v101_branch': 'animal_original_v100'}
        return clean, 0
    from .paths import model_directory
    models = model_directory()
    if _evidence is None:
        _evidence = PhotoEvidence(models)
    if _matting is None:
        _matting = TrimapMatte(models/'vitmatte_small')
    state = engine.last_state
    if state is None:
        raise RuntimeError('Missing v100 segmentation observations')
    try:
        state = dict(state)
        interior = (state['aligned']['face'] > .9).astype(np.uint8)
        radius = max(1, round(float(state['fw'])*.03))
        interior = cv2.erode(interior, np.ones((radius*2+1,)*2, np.uint8)) > 0
        interior &= state['solid_interior_core']
        state['identity_core'] = interior
        probabilities, outline = _evidence.predict(rgb, result['geometry'])
        state['face_outline'] = outline
        state['hood_mask'] = getattr(engine, 'hood_mask', None)
        state['clothing_evidence'] = probabilities[4]
        items = headwear_mask(state['aligned'], state['p_full'], result['head_alpha'], state, rgb)
        state['owned_headwear'] = items
        alpha, neck_count = strip_neck(result['head_alpha'], state['aligned'], result['geometry'], items)
        alpha, foreground, report = refine(rgb, alpha, result['head_fg_rgb'], state, probabilities)
        alpha, detail = regularize_jaw(alpha, state, probabilities)
        report.update(detail)
        alpha, foreground, detail = _matting.refine(rgb, alpha, foreground, state, probabilities,
                                                   device=engine.compute_device.type,
                                                   strand_predictor=_evidence.predict_strands)
        report.update(detail)
        if not np.isfinite(alpha).all() or alpha.min() < 0 or alpha.max() > 1:
            raise RuntimeError('Invalid crop alpha')
        if not np.array_equal(alpha[interior], result['head_alpha'][interior]):
            raise RuntimeError('Face-interior alpha changed; candidate rejected')
        if not np.array_equal(foreground[interior], result['head_fg_rgb'][interior]):
            raise RuntimeError('Face-interior colors changed; candidate rejected')
        if not np.array_equal(alpha[items], result['head_alpha'][items]):
            raise RuntimeError('Headwear alpha changed; candidate rejected')
        if hand_mask is not None:
            alpha, hand_report = remove_confirmed_hands(alpha, result["geometry"], state, hand_mask, hand_guides)
            report.update(hand_report)
        yy, xx = np.nonzero(alpha > .05)
        if not len(yy):
            raise RuntimeError('Empty head crop')
        clean = dict(result)
        clean['head_alpha'] = alpha
        clean['head_fg_rgb'] = foreground
        clean['head_rgba'] = np.dstack([foreground, np.rint(alpha*255).astype(np.uint8)])
        clean['head_bbox'] = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
        clean['diagnostics'] = {**result.get('diagnostics', {}), **report,
            'v101_branch': 'extended_hair_trimap' if report['trimap_applied'] else 'compact_head',
            'face_interior_unchanged': True, 'owned_headwear_pixels': int(items.sum()),
            'reference_hands_reconstructed': False}
        return clean, neck_count
    finally:
        engine.last_state = None
        engine.hood_mask = None
