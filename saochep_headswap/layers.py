"""Head-only RGB compositing. No generated body pixel may fill disocclusion.

Confidence here means geometric/photometric agreement, not a calibrated model
probability. Unobserved areas fail closed instead of accepting Wan's outfit.
"""
from dataclasses import dataclass

import cv2
import numpy as np


class QualityFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class Policy:
    neighbors: int = 6
    flow_side: int = 384
    flow_fb_pixels: float = 1.5
    plate_agreement: float = 14.
    minimum_observations: int = 2
    minimum_observed_fraction: float = .995
    maximum_small_hole_pixels: int = 8
    spatial_boundary_pixels: float = 2.


def validate_mask(mask, shape):
    value = np.asarray(mask, np.float32)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError('Mask dimensions or finite-value check failed')
    if value.min() < 0 or value.max() > 1:
        raise ValueError('Mask must be in [0,1]')
    return value


def compact_source_head(head, face_core):
    """Drop detached below-jaw pieces only on confirmed bare, short-haired heads."""
    core = face_core > .5
    rows = np.flatnonzero(core.any(axis=1))
    if not rows.size:
        return head.copy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats((head > .5).astype(np.uint8), connectivity=8)
    anchored = np.unique(labels[core & (labels > 0)])
    if not anchored.size:
        return head.copy()
    # Keep all components near the jaw, including fine detached edge pixels.
    clearance = max(2, round((rows[-1]-rows[0]+1)*.02))
    remove = np.zeros(count, dtype=bool)
    remove[1:] = stats[1:, cv2.CC_STAT_TOP] > rows[-1] + clearance
    remove[anchored] = False
    return np.where(remove[labels], 0, head)


def component_family(seed, candidates, scale):
    """Associate connected head parts; never use a fixed SAM object index."""
    seed = np.asarray(seed) > .5
    if seed.sum() < 16:
        raise QualityFailure('head_face_seed_missing')
    radius = max(2, round(scale * .12))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
    accepted = seed.copy()
    pieces = []
    for candidate in candidates:
        count, labels, stats, _ = cv2.connectedComponentsWithStats((candidate > .5).astype(np.uint8))
        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]
            if max(3, seed.sum()*.001) <= area <= seed.sum()*25:
                pieces.append(labels == label)
    # Iteration associates glasses -> cap -> scalp fragments without a huge hull.
    for _ in range(3):
        near = cv2.dilate(accepted.astype(np.uint8), kernel) > 0
        added = False
        for index in range(len(pieces)-1, -1, -1):
            if np.count_nonzero(pieces[index] & near) >= 2:
                accepted |= pieces.pop(index)
                added = True
        if not added:
            break
    return accepted


def anatomical_face_guard(face_box, shape):
    box = np.asarray(face_box, np.float32)
    if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= 0).any():
        raise ValueError('Expected a detected face bounding box')
    x, y, width, height = box
    guard = np.zeros(shape, np.uint8)
    # The upper head plane under bangs is opaque too. Its interior cannot use
    # background alpha just because a strand was labeled separately from skin.
    cv2.ellipse(guard, (round(x+width*.5), round(y+height*.36)),
                (max(1, round(width*.42)), max(1, round(height*.56))), 0, 0, 360, 1, -1)
    # Lower jaw ownership continues to come from parsing; this guard repairs
    # forehead/temple gaps without promoting detected neck into the head.
    guard[max(0, int(y+height*.78)):] = 0
    return guard > 0


def face_ownership(labels, hair, hat, face_box=None):
    """Repair enclosed face holes without growing the jaw into real neck skin.

    A morphology operation is geometry, not a new semantic prediction. Explicit
    exterior neck/clothing labels must remain negative after closing the face.
    """
    face = np.isin(labels, list(range(1, 14)))
    flood = np.pad(face.astype(np.uint8), 1)
    cv2.floodFill(flood, None, (0, 0), 1)
    enclosed = flood[1:-1, 1:-1] == 0
    # The winning neck/garment class owns the pixel. A weaker hair score must
    # not override that winner and reintroduce a strip of neck or dark cloth.
    negative = np.isin(labels, [14, 15, 16])
    # A tiny mislabeled enclosed island is inside the face, unlike the neck
    # connected to exterior background. Keep eyes/cheeks opaque in either case.
    negative[enclosed] = False
    sealed = face | enclosed
    if face_box is not None:
        interior = anatomical_face_guard(face_box, labels.shape)
        sealed |= interior
        negative[interior] = False
    closed = cv2.morphologyEx(sealed.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    core = (closed | sealed) & ~negative
    if (core & negative).any():
        raise AssertionError('Face and neck ownership must be disjoint')
    return core, negative


def worn_hood_region(hood, head, geometry):
    """Limit a dedicated hood segmentation to the attached head region."""
    x, y, width, height = geometry['face_box']
    rows, cols = np.ogrid[:head.shape[0], :head.shape[1]]
    envelope = ((rows >= y-height*.85) & (rows <= y+height*1.08)
                & (cols >= x-width*.65) & (cols <= x+width*1.65))
    seed = (hood > .8) & envelope
    anchor = seed & (head > .5) & (rows < y+height*.35)
    if not anchor.any():
        return np.zeros_like(seed)
    _, labels = cv2.connectedComponents(seed.astype(np.uint8), connectivity=8)
    attached = np.zeros(int(labels.max())+1, dtype=bool)
    attached[labels[anchor]] = True
    seed = attached[labels]
    # Preserve trim missed by the hood tracker only next to confident hood
    # pixels, within the existing head segmentation and the face envelope.
    radius = max(1, min(6, round(width*.045)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    candidate = (cv2.dilate(seed.astype(np.uint8), kernel) > 0) & envelope & (head > .5)
    if not candidate.any():
        return np.zeros_like(candidate)
    return candidate


def edge_alpha(rgb, binary, face_core):
    """Local guided boundary; opaque face core, no full-face smoothing."""
    mask = (np.asarray(binary) > .5).astype(np.float32)
    guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    box = lambda image: cv2.boxFilter(image, -1, (7, 7), borderType=cv2.BORDER_REFLECT)
    mean_i, mean_p = box(guide), box(mask)
    variance = box(guide*guide)-mean_i*mean_i
    covariance = box(guide*mask)-mean_i*mean_p
    a = covariance/(variance+.001)
    b = mean_p-a*mean_i
    guided = np.clip(box(a)*guide+box(b), 0, 1)
    kernel = np.ones((3, 3), np.uint8)
    inside = cv2.erode(mask.astype(np.uint8), kernel) > 0
    outside = cv2.dilate(mask.astype(np.uint8), kernel) == 0
    guided[inside | (face_core > .5)] = 1
    guided[outside & (face_core <= .5)] = 0
    return guided.astype(np.float32)


def stable_boundaries(frames, alphas, cores):
    """Motion-compensated smoothing only inside each current uncertain edge."""
    result = []
    empty = np.zeros(alphas[0].shape, bool)
    policy = Policy()
    for index, alpha in enumerate(alphas):
        boundary = (alpha > .001) & (alpha < .999) & (cores[index] < .5)
        value = alpha.copy()
        weighted = alpha.copy()
        weights = np.ones(alpha.shape, np.float32)
        for neighbor in (index-1, index+1):
            if not 0 <= neighbor < len(frames):
                continue
            flow, good = flow_pair(frames[index], frames[neighbor], empty, empty, policy)
            observation = _remap(alphas[neighbor], flow)
            usable = good & boundary & (np.abs(observation-alpha) < .35)
            weighted[usable] += observation[usable]*.35
            weights[usable] += .35
        value[boundary] = (weighted/weights)[boundary]
        value[cores[index] > .5] = 1
        result.append(value)
    return result


def matte_head(rgb, binary, face_core, alpha_hint=None):
    """Estimate straight foreground at the narrow uncertain boundary.

    This is a local two-color matte, not a replacement for a learned video
    matting model. Low-contrast edges fall back to the guided silhouette.
    """
    alpha = edge_alpha(rgb, binary, face_core)
    foreground = rgb.copy()
    inside = cv2.erode((binary > .5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    outside = cv2.dilate((binary > .5).astype(np.uint8), np.ones((3, 3), np.uint8)) == 0
    boundary = (~inside) & (~outside) & (face_core < .5)
    if not inside.any() or not outside.any() or not boundary.any():
        return foreground, alpha
    def nearest_colors(known):
        _, labels = cv2.distanceTransformWithLabels((~known).astype(np.uint8), cv2.DIST_L2, 5,
                                                   labelType=cv2.DIST_LABEL_PIXEL)
        lookup = np.zeros((int(labels.max())+1, 3), np.float32)
        lookup[labels[known]] = rgb[known]
        return lookup[labels[boundary]]
    f, b = nearest_colors(inside), nearest_colors(outside)
    c = rgb[boundary].astype(np.float32)
    vector = f-b
    denominator = (vector*vector).sum(axis=1)
    estimate = np.clip(((c-b)*vector).sum(axis=1)/np.maximum(denominator, 1), 0, 1)
    reliable = denominator > 20**2
    a = alpha[boundary]
    a[reliable] = estimate[reliable]
    if alpha_hint is not None:
        a = .75*a+.25*alpha_hint[boundary]
    alpha[boundary] = a
    straight = (c-(1-a[:, None])*b)/np.maximum(a[:, None], .2)
    straight[a < .2] = f[a < .2]
    foreground[boundary] = np.rint(straight).clip(0, 255).astype(np.uint8)
    return foreground, alpha


def _remap(image, flow, interpolation=cv2.INTER_LINEAR):
    yy, xx = np.indices(flow.shape[:2], np.float32)
    return cv2.remap(image, xx+flow[..., 0], yy+flow[..., 1], interpolation,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def flow_pair(target, neighbor, target_invalid, neighbor_invalid, policy):
    """Estimate motion on observed pixels, extend into the removed head region."""
    h, w = target.shape[:2]
    scale = min(1., policy.flow_side/max(h, w))
    size = (max(16, round(w*scale)), max(16, round(h*scale)))
    gray = lambda rgb: cv2.cvtColor(cv2.resize(rgb, size), cv2.COLOR_RGB2GRAY)
    a, b = gray(target), gray(neighbor)
    bad_a = cv2.resize(target_invalid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    bad_b = cv2.resize(neighbor_invalid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    if bad_a.all() or bad_b.all():
        raise QualityFailure('no_observed_motion_context')
    engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    ab, ba = engine.calc(a, b, None), engine.calc(b, a, None)
    # Flow estimated on hair is not motion evidence for the body behind it.
    # Replace that flow using only surrounding observed motion.
    for vector, bad in ((ab, bad_a), (ba, bad_b)):
        if bad.any():
            for channel in range(2):
                vector[..., channel] = cv2.inpaint(vector[..., channel], bad, 3, cv2.INPAINT_NS)
    ab = cv2.resize(ab, (w, h))/scale
    ba = cv2.resize(ba, (w, h))/scale
    inverse = _remap(ba, ab)
    error = np.linalg.norm(ab+inverse, axis=2)
    visible = _remap((~neighbor_invalid).astype(np.float32), ab) > .999
    yy, xx = np.indices((h, w), np.float32)
    valid = visible & (error < policy.flow_fb_pixels/scale)
    valid &= (xx+ab[..., 0] >= 1) & (xx+ab[..., 0] < w-2)
    valid &= (yy+ab[..., 1] >= 1) & (yy+ab[..., 1] < h-2)
    return ab, valid


def source_plate(frames, old_masks, occluders, frame, needed, policy=Policy(), strict=True):
    """Use two agreeing, actually visible driving observations per filled pixel."""
    target = np.asarray(frames[frame], np.uint8)
    plate = target.copy()
    count_needed = int(np.count_nonzero(needed))
    if not count_needed:
        return plate, np.ones(needed.shape, np.float32), {'needed_pixels': 0, 'unresolved_pixels': 0}
    n = len(frames)
    # Spread neighbors over the shot, rather than requiring adjacent hair to move.
    candidates = sorted(set(np.linspace(0, n-1, min(n, policy.neighbors+1)).round().astype(int))-{frame},
                        key=lambda index: abs(index-frame))[:policy.neighbors]
    samples, validity = [], []
    for index in candidates:
        invalid = (old_masks[index] > .01) | (occluders[index] > .01)
        flow, good = flow_pair(target, np.asarray(frames[index], np.uint8),
                               (old_masks[frame] > .01) | (occluders[frame] > .01), invalid, policy)
        sample = _remap(np.asarray(frames[index], np.uint8), flow).astype(np.float32)
        # Retain only requested reveal pixels; storing every warped full RGB
        # frame made additional source observations unnecessarily expensive.
        samples.append(sample[needed])
        validity.append(good[needed])
    confidence = np.ones(needed.shape, np.float32)
    confidence[needed] = 0
    if samples:
        # Work on needed pixels only; avoid a K x full-resolution RGB allocation.
        values = np.stack(samples)
        valid = np.stack(validity)
        # A deterministic medoid avoids NaN median warnings for unseen pixels.
        distances = np.linalg.norm(values[:, None]-values[None, :], axis=3)
        compatible = (distances <= policy.plate_agreement) & valid[:, None] & valid[None, :]
        votes = compatible.sum(axis=1)
        winner = votes.argmax(axis=0)
        columns = np.arange(len(winner))
        evidence = votes[winner, columns] >= policy.minimum_observations
        chosen = values[winner, columns]
        plate[needed] = np.rint(chosen).clip(0, 255).astype(np.uint8)
        confidence[needed] = evidence.astype(np.float32)
    unresolved = needed & (confidence == 0)
    # A thin silhouette boundary can be interpolated from the driving frame's
    # adjacent visible colors. This cannot fill a large hidden patch of shirt.
    distance = cv2.distanceTransform((old_masks[frame] > .01).astype(np.uint8), cv2.DIST_L2, 3)
    boundary_holes = unresolved & (distance <= policy.spatial_boundary_pixels)
    spatial_count = 0
    if boundary_holes.any():
        invalid = ((old_masks[frame] > .01) | (occluders[frame] > .01)).astype(np.uint8)
        first = cv2.inpaint(target, invalid, 3, cv2.INPAINT_TELEA)
        second = cv2.inpaint(target, invalid, 3, cv2.INPAINT_NS)
        agreement = np.max(np.abs(first.astype(np.float32)-second.astype(np.float32)), axis=2) <= 10
        visible = (1-invalid).astype(np.float32)
        side = 2*int(np.ceil(policy.spatial_boundary_pixels))+5
        box = lambda value: cv2.boxFilter(value, -1, (side, side), normalize=False)
        count = box(visible)
        values = target.astype(np.float32)
        mean = box(values*visible[..., None])/np.maximum(count[..., None], 1)
        variance = box(values*values*visible[..., None])/np.maximum(count[..., None], 1)-mean*mean
        smooth_context = np.sqrt(np.maximum(variance, 0)).max(axis=2) <= 18
        # Broad textured garments and unobserved shapes are not reconstructed
        # by this fallback; it only extends sufficiently observed smooth color.
        accepted = boundary_holes & agreement & smooth_context & (count >= 9)
        interpolation = np.rint((first.astype(np.float32)+second.astype(np.float32))*.5).astype(np.uint8)
        plate[accepted] = interpolation[accepted]
        confidence[accepted] = 1
        spatial_count = int(accepted.sum())
        unresolved = needed & (confidence == 0)
    unresolved_count = int(unresolved.sum())
    allowed_holes = min(policy.maximum_small_hole_pixels,
                        int(count_needed*(1-policy.minimum_observed_fraction)))
    if unresolved_count > allowed_holes and strict:
        raise QualityFailure(f'source_plate_unobserved: frame={frame}, unresolved={unresolved_count}, needed={count_needed}')
    if unresolved_count and unresolved_count <= allowed_holes:
        # Tiny interpolation holes only, source RGB exclusively; never Wan RGB.
        plate = cv2.inpaint(plate, unresolved.astype(np.uint8), 2, cv2.INPAINT_TELEA)
        confidence[unresolved] = 1
    return plate, confidence, {'needed_pixels': count_needed, 'unresolved_pixels': unresolved_count,
                               'source_boundary_interpolated_pixels': spatial_count,
                               'observations_tested': len(samples), 'provenance': 'driving_video_only'}


def verified_background(plate, confidence, source, candidate, old, alpha, body, occluder,
                        background, source_core, generated_core, source_background=None, source_body=None):
    """Allow scene RGB in an old-head hole, never a generated garment/neck.

    Only above both lower-face planes, with positive background segmentation,
    an explicit body/occluder veto, and agreement with nearby source scenery.
    This is an additional heuristic; it does not certify unseen scene texture.
    """
    shape = alpha.shape
    rows = np.indices(shape)[0]
    old_y, old_x = np.where(source_core > .5)
    new_y, new_x = np.where(generated_core > .5)
    if not len(old_y) or not len(new_y):
        return plate, confidence, 0
    face_height = min(old_y.max()-old_y.min()+1, new_y.max()-new_y.min()+1)
    face_width = min(old_x.max()-old_x.min()+1, new_x.max()-new_x.min()+1)
    cutoff = int(min(old_y.max(), new_y.max())-face_height*.15)
    upper = rows < cutoff
    known = (background > .8) & (alpha < .001) & (body < .1) & (occluder < .01) & upper
    source_visible = (old < .01) & (occluder < .01) & upper
    if source_background is not None:
        source_visible &= source_background > .8
    if not known.any() or not source_visible.any():
        return plate, confidence, 0
    def nearest(image, visible):
        distance, labels = cv2.distanceTransformWithLabels((~visible).astype(np.uint8), cv2.DIST_L2, 5,
                                                          labelType=cv2.DIST_LABEL_PIXEL)
        lookup = np.zeros((int(labels.max())+1, 3), np.uint8)
        lookup[labels[visible]] = image[visible]
        return lookup[labels], distance
    rgb, distance = nearest(candidate, known)
    observed, source_distance = nearest(source, source_visible)
    delta = np.max(np.abs(rgb.astype(np.float32)-observed.astype(np.float32)), axis=2)
    needed = old*(1-alpha)*(1-occluder) > .02
    permitted = needed & (confidence < .99) & upper & (body < .1) & (occluder < .01)
    if source_body is not None:
        permitted &= source_body < .1
    permitted &= (distance <= max(3, min(24, face_width*.12)))
    permitted &= source_distance <= max(16, min(64, face_width*.35))
    # When an independent source-garment veto is supplied, scene RGB above the
    # lower face may reconstruct genuinely occluded scenery. Color agreement
    # remains mandatory for the weaker, geometry-only caller.
    if source_body is None:
        permitted &= delta <= 25
    result, certainty = plate.copy(), confidence.copy()
    result[permitted] = rgb[permitted]
    certainty[permitted] = 1
    return result, certainty, int(permitted.sum())


def driving_skin_transition(plate, confidence, source, old, alpha, source_skin, source_neck,
                            generated_neck, source_body, occluder, geometry):
    """Fill a narrow old-jaw/new-neck gap using driving-neck colors only."""
    box = np.asarray(geometry['face_box'], np.float32)
    width, height = float(box[2]), float(box[3])
    rows = np.indices(old.shape)[0]
    below_mouth = rows >= float(geometry['mouth_y'])+height*.05
    within_jaw = rows <= float(geometry['mouth_y'])+height*.45
    distance = cv2.distanceTransform((old > .01).astype(np.uint8), cv2.DIST_L2, 3)
    needed = old*(1-alpha)*(1-occluder) > .02
    target = needed & (confidence < .99) & below_mouth & within_jaw
    target &= (source_skin > .8) & (generated_neck > .5) & (source_body < .1) & (occluder < .01)
    target &= distance <= max(2, min(12, width*.08))
    known = (source_neck > .8) & (old < .01) & (source_body < .1) & (occluder < .01)
    known &= cv2.erode((source_neck > .8).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if not target.any() or known.sum() < 16:
        return plate, confidence, 0
    weight = cv2.GaussianBlur(known.astype(np.float32), (5, 5), 1.)
    color = cv2.GaussianBlur(source.astype(np.float32)*known[..., None], (5, 5), 1.)
    color /= np.maximum(weight[..., None], 1e-6)
    distance, labels = cv2.distanceTransformWithLabels((~known).astype(np.uint8), cv2.DIST_L2, 5,
                                                       labelType=cv2.DIST_LABEL_PIXEL)
    lookup = np.zeros((int(labels.max())+1, 3), np.float32)
    lookup[labels[known]] = color[known]
    target &= distance <= max(8, width*.5)
    result, certainty = plate.copy(), confidence.copy()
    result[target] = np.rint(lookup[labels[target]]).clip(0, 255).astype(np.uint8)
    certainty[target] = 1
    return result, certainty, int(target.sum())


def observed_driving_garment(source_body, source_forbidden, source_skin, source_core, generated_body):
    """Recover original garment pixels misassigned to the driving head mask."""
    source_cloth = source_body > .5
    observed = source_cloth & (source_forbidden > .5)
    face_rows = np.flatnonzero(np.any(source_core > .5, axis=1))
    if face_rows.size:
        below_face = np.arange(source_core.shape[0])[:, None] > face_rows[-1]
        interior = cv2.erode(source_cloth.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        observed |= below_face & interior & (generated_body > .8)
    return observed & (source_skin < .5) & (source_core < .5)


def extend_matte_plate(plate, confidence, old, alpha, core, body, source_body, occluder):
    """Extend already resolved plate colors across a thin, partial hair edge.

    Opaque old-head pixels covered by the new head are NOT plate evidence.
    Uncovered regions, face holes and garment conflicts cannot use this path.
    """
    needed = old*(1-alpha)*(1-occluder) > .02
    known = ((needed & (confidence >= .99)) | (old < .001)) & (occluder < .01)
    # A transparent strand needs the same underlying plate at alpha .45 as at
    # .55. Opacity is not background confidence; the 2px evidence distance and
    # semantic ownership vetoes below provide the spatial guard.
    target = needed & (confidence < .99) & (alpha > .02) & (alpha < .999)
    target &= (core < .5) & (body < .1) & (source_body < .1) & (occluder < .01)
    if not known.any() or not target.any():
        return plate, confidence, 0
    distance, labels = cv2.distanceTransformWithLabels((~known).astype(np.uint8), cv2.DIST_L2, 5,
                                                       labelType=cv2.DIST_LABEL_PIXEL)
    lookup = np.zeros((int(labels.max())+1, 3), np.uint8)
    lookup[labels[known]] = plate[known]
    target &= distance <= 2
    result, certainty = plate.copy(), confidence.copy()
    result[target] = lookup[labels[target]]
    certainty[target] = 1
    return result, certainty, int(target.sum())


def smooth_source_background(plate, confidence, source, old, alpha, core, body, source_body,
                             occluder, source_background, max_distance=16):
    """Extrapolate only a locally smooth SOURCE background at partial hair edges.

    Two interleaved source sample sets must independently predict the same RGB
    plane. This is bounded interpolation evidence, not recovered hidden texture.
    """
    needed=old*(1-alpha)*(1-occluder)>.02
    target=needed&(confidence<.99)&(alpha>.02)&(alpha<.999)&(core<.5)
    target&=(body<.1)&(source_body<.1)&(occluder<.01)
    known=(source_background>.8)&(old<.01)&(source_body<.1)&(occluder<.01)
    distance=cv2.distanceTransform((~known).astype(np.uint8),cv2.DIST_L2,5)
    target&=distance<=max_distance
    result,certainty=plate.copy(),confidence.copy();accepted=0
    radius=max(8,int(max_distance*1.5));h,w=old.shape
    for y,x in zip(*np.where(target)):
        y1,y2=max(0,y-radius),min(h,y+radius+1)
        x1,x2=max(0,x-radius),min(w,x+radius+1)
        yy,xx=np.where(known[y1:y2,x1:x2]);yy+=y1;xx+=x1
        if len(xx)<32:continue
        matrix=np.column_stack([np.ones(len(xx)),(xx-x)/radius,(yy-y)/radius])
        values=source[yy,xx].astype(np.float32);predictions=[]
        for parity in (0,1):
            subset=((xx+yy)%2)==parity
            if subset.sum()<16:break
            coef,_,rank,_=np.linalg.lstsq(matrix[subset],values[subset],rcond=None)
            if rank<3:break
            residual=np.linalg.norm(matrix[subset]@coef-values[subset],axis=1)
            if np.percentile(residual,90)>6:break
            predictions.append(coef[0])
        if len(predictions)!=2 or np.max(np.abs(predictions[0]-predictions[1]))>3:continue
        prediction=np.mean(predictions,axis=0)
        if np.any(prediction<values.min(0)-6) or np.any(prediction>values.max(0)+6):continue
        result[y,x]=np.rint(prediction).clip(0,255).astype(np.uint8)
        certainty[y,x]=1;accepted+=1
    return result,certainty,accepted


def additional_source_observations(plate, confidence, frames, old, occluders, frame,
                                   alpha, core, body, source_body, boundary_limit):
    """Escalate source-only evidence for remaining reveal pixels, never the whole shot."""
    needed=old[frame]*(1-alpha)*(1-occluders[frame])>.02
    result,certainty=plate.copy(),confidence.copy()
    attempts=[]
    for neighbors in sorted(set(min(len(frames)-1,n) for n in (12,24,48,72))):
        missing=needed&(certainty<.99)
        if not missing.any():break
        if neighbors<2:continue
        candidate,valid,_=source_plate(frames,old,occluders,frame,missing,
            Policy(neighbors=neighbors,spatial_boundary_pixels=boundary_limit),strict=False)
        accepted=missing&(valid>=.99)
        result[accepted]=candidate[accepted];certainty[accepted]=1
        result,certainty,extended=extend_matte_plate(result,certainty,old[frame],alpha,
            core,body,source_body,occluders[frame])
        attempts.append({'source_observations':neighbors,'observed_pixels':int(accepted.sum()),
                         'extended_pixels':extended,'remaining':int((needed&(certainty<.99)).sum())})
    return result,certainty,attempts


def compose_frame(source, generated_head, alpha, old, plate, confidence, occluder, forbidden, core):
    h, w = source.shape[:2]
    if any(frame.dtype != np.uint8 or frame.shape != source.shape for frame in (source, generated_head, plate)):
        raise ValueError('RGB frames must share uint8 geometry')
    for mask in (alpha, old, confidence, occluder, forbidden, core):
        validate_mask(mask, (h, w))
    if np.any((core > .5) & (alpha < .98)):
        raise QualityFailure('face_core_transparency')
    if np.any((alpha > .02) & (forbidden > .5) & (occluder < .99)):
        raise QualityFailure('generated_body_or_neck_in_head_matte')
    needed = old*(1-alpha)*(1-occluder) > .02
    if np.any(needed & (confidence < .99)):
        raise QualityFailure('source_plate_confidence_missing')
    base = source.astype(np.float32)*(1-old[..., None])+plate.astype(np.float32)*old[..., None]
    value = generated_head.astype(np.float32)*alpha[..., None]+base*(1-alpha[..., None])
    value = source.astype(np.float32)*occluder[..., None]+value*(1-occluder[..., None])
    result = np.rint(value).clip(0, 255).astype(np.uint8)
    permitted = ((old > 0) | (alpha > 0)) & (occluder < 1)
    result[~permitted] = source[~permitted]
    return result
