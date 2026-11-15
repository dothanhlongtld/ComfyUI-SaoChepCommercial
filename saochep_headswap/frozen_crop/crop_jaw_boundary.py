"""Regularize small skin-only stair steps on the lateral lower-jaw boundary."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter


def regularize_jaw(alpha, state, probabilities):
    parsed = state['aligned']
    points = np.asarray(state['landmarks'], np.float32)
    eyes = points[:2].mean(axis=0)
    direction = points[1]-points[0]
    direction /= max(np.linalg.norm(direction), 1.)
    down = np.array([-direction[1], direction[0]])
    if down[1] < 0:
        down *= -1
    rotation = np.array([[direction[0], direction[1], eyes[0]-np.dot(direction, eyes)],
                         [down[0], down[1], eyes[1]-np.dot(down, eyes)]], np.float32)
    h, w = alpha.shape
    warp = lambda value: cv2.warpAffine(value.astype(np.float32), rotation, (w, h), flags=cv2.INTER_NEAREST)
    aligned = warp(alpha)
    face_width, face_height = float(state['fw']), float(state['fh'])
    mouth = rotation[:, :2] @ points[3:5].mean(axis=0)+rotation[:, 2]
    yy, xx = np.indices(alpha.shape)
    # Operate on predicted skin only. Hats, ears and dark facial hair keep
    # their original alpha. The central chin is outside the editable region.
    skin = ((parsed['face']+parsed['neck']) > .7) & ((probabilities[2]+probabilities[3]) > .8)
    skin &= (parsed['hair'] < .1) & (probabilities[1] < .15) & (parsed['hat'] < .1)
    skin &= ~np.isin(parsed['labels'], [7, 8, 9])
    skin &= ~state.get('identity_core', np.zeros(alpha.shape, bool))
    skin &= ~state.get('owned_headwear', np.zeros(alpha.shape, bool))
    visible = (aligned > .5) & (warp(skin) > .5)
    visible &= (yy > mouth[1]+face_height*.08) & (yy < mouth[1]+face_height*.45)
    columns = np.flatnonzero(visible.any(axis=0))
    if len(columns) < 9:
        return alpha, {'jaw_step_pixels': 0}
    lower = np.full(w, np.nan, np.float32)
    for x in columns:
        lower[x] = np.flatnonzero(visible[:, x])[-1]
    values = np.interp(np.arange(w), columns, lower[columns])
    window = max(5, round(face_width*.10)|1)
    window = min(window, len(columns) if len(columns)%2 else len(columns)-1)
    smooth = savgol_filter(values, window, 2)
    excess = values-smooth
    lateral = (np.abs(np.arange(w)-mouth[0]) > face_width*.25)
    lateral &= np.abs(np.arange(w)-mouth[0]) < face_width*.65
    affected = lateral & (excess > max(.6, face_height*.001)) & (excess < face_height*.05)
    inside_distance = distance_transform_edt(aligned > .05)
    editable = (warp(skin) > .5) & affected[None, :]
    editable &= (yy > mouth[1]+face_height*.08) & (inside_distance < max(1.5, face_width*.018))
    ramp = np.clip((smooth[None, :]+1.0-yy)/1.5, 0, 1)
    reduction = np.where(editable, np.maximum(0, aligned-ramp), 0).astype(np.float32)
    restored = cv2.warpAffine(reduction, cv2.invertAffineTransform(rotation), (w, h), flags=cv2.INTER_LINEAR)
    restored *= skin
    output = np.clip(alpha-restored, 0, 1)
    return output, {'jaw_step_pixels': int(np.count_nonzero(output != alpha))}
