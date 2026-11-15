import cv2
import numpy as np


def measure_frame(image, face, hand_polygons, repaired=False):
    height, width = image.shape[:2]
    box = np.asarray(face[:4], float)
    eyes = np.asarray(face[4:8]).reshape(2, 2)
    mouth = np.asarray(face[10:14]).reshape(2, 2)
    direction = eyes[1] - eyes[0]
    span = float(np.linalg.norm(direction))
    if span <= np.finfo(float).eps:
        raise ValueError('Eye landmarks overlap')
    horizontal = direction / span
    eye_center = eyes.mean(0)
    nose_offset = face[8:10] - eye_center
    x, y, w, h = box
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(width, int(x+w)), min(height, int(y+h))
    face_region = np.zeros((height, width), np.uint8)
    cv2.ellipse(face_region, (round(x+w/2), round(y+h/2)),
                (max(1, round(w/2)), max(1, round(h/2))), 0, 0, 360, 1, -1)
    hands = np.zeros_like(face_region)
    for polygon in hand_polygons:
        cv2.fillConvexPoly(hands, cv2.convexHull(np.asarray(polygon, np.int32)), 1)
    overlap = float(np.count_nonzero(hands & face_region) / max(1, face_region.sum()))
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError('Face outside frame')
    gray = cv2.cvtColor(cv2.resize(crop, (128,128)), cv2.COLOR_RGB2GRAY)
    return {'box': box.tolist(), 'yaw_proxy': abs(float(nose_offset @ horizontal))/span,
            'roll': abs(float(np.arctan2(direction[1], direction[0]))),
            'pitch_proxy': float(np.linalg.norm(mouth.mean(0)-eye_center))/span,
            'hand_overlap': overlap,
            'sharpness': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            'repaired': bool(repaired)}


def choose_ranked_frame(metrics):
    boxes = np.asarray([row['box'] for row in metrics], float)
    valid = np.isfinite(boxes).all(1) & (boxes[:, 2:4] > 0).all(1)
    valid &= np.array([not row['repaired'] and not row.get('excluded', False) for row in metrics])
    if not valid.any():
        raise ValueError('No observed face suitable for alignment')
    overlap = np.asarray([row['hand_overlap'] for row in metrics])
    # Prefer an unoccluded face whenever the detector finds one in the segment.
    unoccluded = valid & (overlap <= .02)
    if unoccluded.any():
        valid = unoccluded
    indices = np.flatnonzero(valid)
    centers = boxes[:, :2] + boxes[:, 2:4]/2
    sizes = np.sqrt(boxes[:, 2]*boxes[:, 3])
    typical_center = np.median(centers[indices], axis=0)
    typical_size = float(np.median(sizes[indices]))
    representative = np.linalg.norm(centers-typical_center, axis=1)/typical_size
    representative += np.abs(np.log(sizes/typical_size))
    movement = np.zeros(len(metrics))
    if len(metrics) > 1:
        speed = np.linalg.norm(np.diff(centers, axis=0), axis=1)/typical_size
        movement[1:] += speed
        movement[:-1] += speed
    pitch = np.array([row['pitch_proxy'] for row in metrics])
    criteria = [overlap, np.array([row['yaw_proxy'] for row in metrics]),
                np.array([row['roll'] for row in metrics]),
                np.abs(pitch-np.median(pitch[indices])), representative, movement,
                -np.array([row['sharpness'] for row in metrics])]
    score = np.zeros(len(indices))
    for values in criteria:
        values = values[indices]
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite frame quality measurement')
        ranks = np.array([np.count_nonzero(values < value) for value in values], float)
        score += ranks/max(1, len(indices)-1)
    selected = int(indices[np.argmin(score)])
    return selected, {'selected_frame': selected, 'candidate_count': len(metrics),
                      'valid_candidates': len(indices), 'selected_metrics': metrics[selected],
                      'metrics': metrics, 'selection': 'relative ranks across visible frames'}


def choose_frame(metrics):
    """Prefer the initial pose when its face is usable and temporally consistent."""
    ranked, report = choose_ranked_frame(metrics)
    first = metrics[0]
    observed = [item for item in metrics if not item['repaired'] and not item.get('excluded', False)
                and np.isfinite(item['box']).all() and min(item['box'][2:4]) > 0]
    accepted = False
    checks = {}
    if not first['repaired'] and not first.get('excluded', False) and observed:
        width, height = first['box'][2:4]
        typical_size = np.median([np.sqrt(item['box'][2]*item['box'][3]) for item in observed])
        typical_sharpness = np.median([item['sharpness'] for item in observed])
        size_ratio = np.sqrt(width*height)/typical_size
        # These dimensionless gates are provisional and require visual regression.
        checks = {
            'unoccluded': first['hand_overlap'] <= .02,
            'face_scale_consistent': .7 <= size_ratio <= 1.4,
            'pose_usable': first['yaw_proxy'] <= .55 and first['roll'] <= np.deg2rad(25),
            'sharpness_usable': first['sharpness'] >= .65*typical_sharpness,
        }
        checks = {key: bool(value) for key, value in checks.items()}
        accepted = all(checks.values())
    selected = 0 if accepted else ranked
    report.update(selected_frame=selected, selected_metrics=metrics[selected],
                  selection='usable_first_frame' if accepted else 'relative_quality_rank',
                  initial_frame_checks=checks, fallback_ranked_frame=ranked)
    return selected, report
