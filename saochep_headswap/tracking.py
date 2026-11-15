import numpy as np


def plausible(previous, current):
    if not np.isfinite(current).all() or (current[2:4] <= 0).any():
        return False
    ratios = current[2:4] / previous[2:4]
    movement = np.linalg.norm((current[:2]+current[2:4]/2) - (previous[:2]+previous[2:4]/2))
    return bool(np.all((ratios >= .5) & (ratios <= 2.)) and movement <= 2*max(previous[2:4]))


def continuity(previous, current):
    movement = np.linalg.norm((current[:2]+current[2:4]/2) - (previous[:2]+previous[2:4]/2))
    return movement/max(previous[2:4]) + np.abs(np.log(current[2:4]/previous[2:4])).sum()


def repair_gaps(faces, max_gap):
    result = list(faces)
    repaired = []
    index = 0
    while index < len(result):
        if result[index] is not None:
            index += 1
            continue
        start = index
        while index < len(result) and result[index] is None:
            index += 1
        length = index-start
        if start == 0 or index == len(result) or length > max_gap:
            raise ValueError(f'Face tracking unavailable at frames {start}..{index-1}')
        before, after = result[start-1], result[index]
        if not plausible(before, after):
            raise ValueError(f'Unconfirmed identity across tracking gap at {start}')
        for offset in range(length):
            fraction = (offset+1)/(length+1)
            result[start+offset] = before*(1-fraction)+after*fraction
            repaired.append(start+offset)
    return result, repaired
