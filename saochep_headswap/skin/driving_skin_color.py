"""Prototype color transfer from driving skin to generated skin.

Inputs must already have actor-specific skin masks and aligned timestamps.
This module does not detect faces or certify the supplied masks.
"""
import cv2
import numpy as np


def bounded_skin_offset(offset, max_lightness=4., max_chroma=3.):
    """Limit correction magnitude while preserving the measured Lab direction."""
    delta=np.asarray(offset,np.float32)
    if delta.shape!=(3,) or not np.isfinite(delta).all():
        raise ValueError('Expected a finite Lab offset')
    if max_lightness<=0 or max_chroma<=0:
        raise ValueError('Color correction bounds must be positive')
    strength=min(1.,max_lightness/max(abs(float(delta[0])),1e-6),
                 max_chroma/max(float(np.linalg.norm(delta[1:])),1e-6))
    return delta*strength


def skin_statistics(rgb, skin, minimum_pixels=64):
    if rgb.shape[:2] != skin.shape or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB and a matching 2D skin mask")
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    pixels = lab[np.asarray(skin) >= .8]
    if len(pixels) < minimum_pixels:
        return None
    low, high = np.percentile(pixels[:, 0], [20, 80])
    core = pixels[(pixels[:, 0] >= low) & (pixels[:, 0] <= high)]
    if len(core) < minimum_pixels // 2:
        return None
    return np.median(core, axis=0)


def driving_offsets(source_statistics, candidate_statistics, boundaries, fps=24,
                    max_lightness=30., max_chroma=25., smoothing_seconds=.12):
    """Estimate paired corrections and smooth them separately within each shot.

    Invalid measurements stay invalid. The caller must flag these frames for
    recovery before publishing. Limits are provisional backtest parameters.
    """
    count = len(source_statistics)
    if count != len(candidate_statistics) or fps <= 0:
        raise ValueError("Statistics must be paired with a positive frame rate")
    if not boundaries or boundaries[0] != 0 or boundaries[-1] != count or any(
            b <= a for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("Shot boundaries must partition the statistics")
    limits = np.array([max_lightness, max_chroma, max_chroma], dtype=np.float32)
    if (limits < 0).any() or smoothing_seconds <= 0:
        raise ValueError("Correction limits and smoothing duration are invalid")
    offsets = np.zeros((count, 3), np.float32)
    valid = np.zeros(count, bool)
    clamped = np.zeros(count, bool)
    for i, (source, candidate) in enumerate(zip(source_statistics, candidate_statistics)):
        if source is None or candidate is None:
            continue
        delta = np.asarray(source, np.float32) - np.asarray(candidate, np.float32)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            continue
        offsets[i] = np.clip(delta, -limits, limits)
        clamped[i] = bool((np.abs(delta) > limits).any())
        valid[i] = True
    smoothed = offsets.copy()
    weight = 1 - np.exp(-1 / (fps * smoothing_seconds))
    for start, stop in zip(boundaries, boundaries[1:]):
        previous = None
        for i in range(start, stop):
            if not valid[i]:
                previous = None
                continue
            window = offsets[max(start, i-2):min(stop, i+3)]
            good = valid[max(start, i-2):min(stop, i+3)]
            estimate = np.median(window[good], axis=0)
            previous = estimate if previous is None else previous + weight * (estimate - previous)
            smoothed[i] = previous
    return smoothed, {"invalid_frames": np.flatnonzero(~valid).tolist(),
                      "clamped_frames": np.flatnonzero(clamped).tolist()}


def apply_skin_offset(rgb, skin_weight, permitted_head, offset):
    """Preserve texture with a Lab translation and freeze every exterior pixel."""
    if rgb.dtype != np.uint8 or skin_weight.shape != rgb.shape[:2] or permitted_head.shape != skin_weight.shape:
        raise ValueError("Expected uint8 RGB and aligned skin/head masks")
    delta = np.asarray(offset, np.float32)
    if delta.shape != (3,) or not np.isfinite(delta).all():
        raise ValueError("Expected a finite Lab offset")
    weight = np.clip(np.asarray(skin_weight, np.float32), 0, 1)
    weight = np.where(np.asarray(permitted_head) > 0, weight, 0)
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    lab += weight[..., None] * delta
    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    result = np.rint(np.clip(corrected, 0, 1) * 255).astype(np.uint8)
    result[weight == 0] = rgb[weight == 0]
    return result
