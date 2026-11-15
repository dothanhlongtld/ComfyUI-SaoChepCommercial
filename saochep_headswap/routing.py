"""Geometry uses landmark coordinates and semantic probabilities, never media IDs."""
import numpy as np


def eye_centers(points):
    points = np.asarray(points)
    return np.array([points[[33,133]].mean(0), points[[263,362]].mean(0)])


def select_translation(reference_points, driving_points, neck_probability,
                       face_probability, *, confidence=.5, min_neck_fraction=.1):
    height,width = neck_probability.shape
    source_eyes,target_eyes = eye_centers(reference_points),eye_centers(driving_points)
    if not np.isfinite(source_eyes).all() or not np.isfinite(target_eyes).all():
        raise ValueError('Eyes unavailable: cannot align automatically')
    for point in target_eyes:
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            raise ValueError('Driving eyes outside canvas')
    delta = target_eyes[1]-target_eyes[0]
    span = float(np.linalg.norm(delta))
    if span <= np.finfo(np.float32).eps:
        raise ValueError('Degenerate eye geometry')
    horizontal=delta/span
    vertical=np.array([-horizontal[1],horizontal[0]])
    if vertical[1] < 0: vertical=-vertical
    chin=np.asarray(driving_points[152])
    source_chin=np.asarray(reference_points[152])
    yy,xx=np.indices((height,width),dtype=np.float32)
    dx,dy=xx-chin[0],yy-chin[1]
    u=dx*horizontal[0]+dy*horizontal[1]
    v=dx*vertical[0]+dy*vertical[1]
    # Dimensionless bands scale with the eye span; none are pixel offsets.
    band=(np.abs(u)<=.3*span)&(v>=0)&(v<=.5*span)
    jaw=(np.abs(u)<=.2*span)&(v>=-.2*span)&(v<0)
    neck_fraction=float(np.mean(neck_probability[band]>=confidence)) if band.any() else 0.
    jaw_fraction=float(np.mean(face_probability[jaw]>=confidence)) if jaw.any() else 0.
    valid_chin=np.isfinite(chin).all() and np.isfinite(source_chin).all()
    valid_chin=valid_chin and 0<=chin[0]<width and 0<=chin[1]<height
    # Reference is already a cropped head with the neck removed; its neck is irrelevant.
    use_chin=bool(valid_chin and neck_fraction>=min_neck_fraction and jaw_fraction>=min_neck_fraction)
    source=source_chin if use_chin else source_eyes.mean(0)
    target=chin if use_chin else target_eyes.mean(0)
    return target-source, {'branch':'chin' if use_chin else 'eyes',
        'reason':'visible_neck_and_jaw' if use_chin else 'neck_or_jaw_not_confirmed',
        'neck_fraction':neck_fraction,'jaw_fraction':jaw_fraction,
        'source_anchor':source.tolist(),'target_anchor':target.tolist(),
        'eye_span':span,'confidence_threshold':confidence,
        'coverage_threshold':min_neck_fraction}
