"""Reference-alignment anchor ranking, ported from HSBackfaceHeadMaskTrial."""
import numpy as np


def choose_index(boxes, frontal, sharpness, repaired):
    boxes = np.asarray(boxes, dtype=np.float64)
    valid = np.isfinite(boxes).all(1) & (boxes[:, 2:4] > 0).all(1)
    valid &= np.isfinite(frontal) & np.isfinite(sharpness) & ~np.asarray(repaired, bool)
    if not valid.any():
        raise ValueError("No reliable reference alignment frame")
    sizes = np.sqrt(boxes[:, 2] * boxes[:, 3])
    typical_size = np.median(sizes[valid])
    centers = boxes[:, :2] + boxes[:, 2:4] / 2
    movement = np.zeros(len(boxes))
    if len(boxes) > 1:
        differences = np.linalg.norm(np.diff(centers, axis=0), axis=1) / typical_size
        movement[1:] += differences
        movement[:-1] += differences
    indices = np.flatnonzero(valid)
    criteria = [np.abs(np.log(sizes[indices] / typical_size)),
                np.asarray(frontal)[indices], movement[indices],
                -np.asarray(sharpness)[indices]]
    total = np.zeros(len(indices))
    for values in criteria:
        total += np.array([np.count_nonzero(values < value) for value in values])
    scores = np.full(len(boxes), -np.inf)
    scores[indices] = -total
    return int(np.argmax(scores))
