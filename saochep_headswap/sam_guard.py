import numpy as np


def transition_metrics(before, after, *, motion_compensated=False):
    """Return legacy continuity measurements plus the selected overlap metric."""
    if type(motion_compensated) is not bool:
        raise TypeError("motion_compensated must be a bool")
    if before.ndim != 2 or after.ndim != 2:
        raise TypeError("SAM head masks must be 2D arrays")
    if before.shape != after.shape:
        raise TypeError("SAM head masks must have matching shapes")
    if not before.any() or not after.any():
        raise ValueError('Empty SAM head')

    def box(mask):
        y, x = np.where(mask)
        return np.array([x.min(), y.min(), x.max() + 1, y.max() + 1], float)

    a, b = box(before), box(after)
    ratio = after.sum() / before.sum()
    delta = (b[:2] + b[2:]) / 2 - (a[:2] + a[2:]) / 2
    movement = np.linalg.norm(delta)
    denominator = min(before.sum(), after.sum())
    raw_overlap = np.count_nonzero(before & after) / denominator
    compensated_overlap = raw_overlap
    if motion_compensated:
        dx, dy = np.rint(delta).astype(int)
        height, width = before.shape
        bx0, by0 = max(0, -dx), max(0, -dy)
        bx1, by1 = min(width, width - dx), min(height, height - dy)
        if bx1 > bx0 and by1 > by0:
            ax0, ay0 = bx0 + dx, by0 + dy
            intersection = np.count_nonzero(
                before[by0:by1, bx0:bx1] & after[ay0:ay0 + by1-by0, ax0:ax0 + bx1-bx0])
            compensated_overlap = intersection / denominator
        else:
            compensated_overlap = 0.0
    return ratio, movement, float(max(a[2:] - a[:2])), raw_overlap, compensated_overlap


def validate_transition(before, after, *, motion_compensated=False):
    ratio, movement, extent, _, overlap = transition_metrics(
        before, after, motion_compensated=motion_compensated)
    if not .5 <= ratio <= 2 or movement > extent or overlap < .25:
        raise ValueError('Uncertain SAM head continuity')
