"""Evidence-gated photo cleanup. All distances scale with detected face size."""
import cv2
import numpy as np
from scipy import ndimage


def headwear_mask(parsed, full_labels, alpha=None, state=None, rgb=None):
    """A coarse hat label needs a connected, independently confirmed head item."""
    if state is not None and state.get('owned_headwear') is not None:
        return state['owned_headwear']
    confirmed = parsed['hat'] > .5
    components, count = ndimage.label(full_labels == 18)
    overlap = np.bincount(components[confirmed].ravel(), minlength=count+1)
    selected = np.flatnonzero(overlap >= 8)
    selected = selected[selected != 0]
    result = confirmed | np.isin(components, selected)
    if state is not None and alpha is not None:
        points = np.asarray(state['landmarks'], np.float32)
        direction = points[1]-points[0]
        direction /= max(float(np.linalg.norm(direction)), 1.)
        down = np.array([-direction[1], direction[0]])
        if down[1] < 0:
            down *= -1
        eyes = points[:2].mean(axis=0)
        mouth = points[3:5].mean(axis=0)
        yy, xx = np.indices(alpha.shape, np.float32)
        side = (xx-eyes[0])*direction[0]+(yy-eyes[1])*direction[1]
        depth = (xx-mouth[0])*down[0]+(yy-mouth[1])*down[1]
        on_head = (np.abs(side) < float(state['fw'])*1.4) & (depth < float(state['fh'])*.35)
        # A hood lining is cloth as a material, but belongs to the head item.
        # Its connected fabric can cross hat/cloth class boundaries. Require
        # an independently confirmed headwear seed and the original head matte.
        fabric = (parsed['hat'] > .25) | (parsed['cloth'] > .35) | (full_labels == 18)
        fabric &= (alpha > .05) & on_head
        labels, _ = ndimage.label(fabric)
        ids = np.unique(labels[confirmed & fabric])
        ids = ids[ids != 0]
        result |= np.isin(labels, ids) & fabric
        if rgb is not None:
            # Hair can occlude the connection between two visible pieces of
            # the same hood. Confirm the disconnected fabric by its material
            # color, within the existing head extent and near the head only.
            seed = confirmed & (alpha > .8) & on_head
            if seed.sum() >= 8:
                lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
                prototype = np.median(lab[seed], axis=0)
                same_material = np.linalg.norm(lab-prototype, axis=2) < 22
                result |= fabric & same_material
        role = state.get('hood_mask')
        if role is not None:
            material = state.get('clothing_evidence', np.zeros(alpha.shape, np.float32))
            role = np.asarray(role) > .5
            if role.shape != alpha.shape:
                raise ValueError('Headwear role mask must match photo geometry')
            anchor = role & (alpha > .1) & (depth < -float(state['fh'])*.25)
            anchor &= (material > .3) & ((parsed['cloth'] > .3) | (full_labels == 18))
            if anchor.sum() >= max(8, float(state['fw'])*float(state['fh'])*.001):
                result |= role & on_head & (alpha > .05)
    return result


def refine(rgb, alpha, foreground, state, probabilities):
    parsed = state['aligned']
    subject = state['subj_alpha']
    width = float(state['fw'])
    height = float(state['fh'])
    points = np.asarray(state['landmarks'], np.float32)
    horizontal = points[1]-points[0]
    horizontal /= max(float(np.linalg.norm(horizontal)), 1.)
    down = np.array([-horizontal[1], horizontal[0]])
    if down[1] < 0:
        down *= -1
    yy, xx = np.indices(alpha.shape, np.float32)
    mouth = points[3:5].mean(axis=0)
    mouth_distance = (xx-mouth[0])*down[0]+(yy-mouth[1])*down[1]
    below_mouth = mouth_distance > height*.12
    hair, body, face, clothing = (probabilities[index] for index in (1, 2, 3, 4))
    # Facial interiors and detected ears have explicit ownership. Clothing
    # cleanup cannot punch through them, even when another model disagrees.
    face_guard = ((parsed['face'] > .8) & (face > .6)) | state['solid_interior_core']
    ears = np.isin(parsed['labels'], [7, 8, 9]) & (parsed['face'] > .5)
    headwear = headwear_mask(parsed, state['p_full'], alpha, state, rgb)
    protected = face_guard | ears | headwear
    identity_core = state.get('identity_core', np.zeros(alpha.shape, bool))
    protected |= identity_core
    cloth_vote = (parsed['cloth'] > .55) | np.isin(state['p_full'], [15, 16])
    cloth_remove = (clothing > .75) & cloth_vote & (hair < .25) & ~protected
    neck_vote = (parsed['neck'] > .2) | (state['p_full'] == 14)
    neck_remove = (body > .65) & neck_vote & (hair < .25) & below_mouth & ~protected
    outline = state.get('face_outline')
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    # The legacy mouth+0.1*face_height cutoff leaves two rectangular neck
    # ledges when the jaw rises at its sides. Use local neck evidence there,
    # while keeping the central chin/beard independent of that cleanup.
    lateral_jaw = np.abs((xx-mouth[0])*horizontal[0]+(yy-mouth[1])*horizontal[1]) > width*.25
    skin_seed = (parsed['face'] > .8) & (face > .8) & (mouth_distance < 0) & (alpha > .8)
    if skin_seed.sum() >= 30:
        skin_color = np.median(lab[skin_seed], axis=0)
        skin_match = (np.linalg.norm(lab[..., 1:]-skin_color[1:], axis=2) < 18)
        skin_match &= lab[..., 0] > skin_color[0]-60
        neck_ledges = lateral_jaw & (mouth_distance > 0) & skin_match
        neck_ledges &= (parsed['neck'] > .65) & (state['p_full'] == 14)
        neck_ledges &= (hair < .25) & (parsed['hair'] < .25) & ~protected
        neck_remove |= neck_ledges
    if outline is not None:
        # Override an old parser's face-to-neck leak only when independent
        # anatomy and a strong body-skin prediction both support the cut.
        anatomical_neck = (~outline) & below_mouth & (body > .8) & (hair < .15)
        anatomical_neck &= ~ears & ~headwear & ~identity_core
        neck_remove |= anatomical_neck
        protected &= ~anatomical_neck
    # A parser can call a bright shirt fragment "hair". Compare it with
    # observed garment colors outside the extracted head, using verified
    # scalp colors as a veto. No background color is used as a garment seed.
    scalp = (parsed['hair'] > .7) & (hair > .7) & (subject > .95) & ~below_mouth
    eyes = points[:2].mean(axis=0)
    lateral = (xx-eyes[0])*horizontal[0]+(yy-eyes[1])*horizontal[1]
    vertical = (xx-eyes[0])*down[0]+(yy-eyes[1])*down[1]
    scalp &= (np.abs(lateral) < width*.5) & (vertical > -height*.7) & (vertical < -height*.05)
    scalp &= ~headwear
    garment_seed = np.isin(state['p_full'], [15, 16]) & (clothing > .7)
    garment_seed &= (alpha < .05) & (subject > .95) & below_mouth
    color_remove = np.zeros(alpha.shape, bool)
    state['garment_color_ambiguity'] = np.zeros(alpha.shape, bool)
    if scalp.sum() >= 30 and garment_seed.sum() >= 30:
        def centers(mask):
            pixels = lab[mask]
            pixels = pixels[::max(1, len(pixels)//3000)].copy()
            cv2.setRNGSeed(101)
            return cv2.kmeans(pixels, min(3, len(pixels)), None,
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, .2),
                1, cv2.KMEANS_PP_CENTERS)[2]
        hair_centers, cloth_centers = centers(scalp), centers(garment_seed)
        hair_distance = np.minimum.reduce([np.linalg.norm(lab-center, axis=2) for center in hair_centers])
        cloth_distance = np.minimum.reduce([np.linalg.norm(lab-center, axis=2) for center in cloth_centers])
        possible = (cloth_distance < 15) & (hair_distance > 30) & (cloth_distance < hair_distance*.45)
        possible &= (subject > .95) & below_mouth
        # A bright textile seen through dark strands is not safe known
        # foreground. Let the matting model resolve it; this is uncertainty,
        # never a direct instruction to erase hair of an unusual color.
        state['garment_color_ambiguity'] = possible & (alpha > .02) & ~protected
        components, _ = ndimage.label(possible)
        connected = np.unique(components[garment_seed & possible])
        connected = connected[connected != 0]
        color_remove = np.isin(components, connected) & possible & ~protected
        color_remove &= (parsed['hair'] < .85) & (hair < .98)
    out = alpha.copy()
    removed = (cloth_remove | neck_remove | color_remove) & (out > 0)
    out[removed] = 0
    # Repair concave notches and internal gaps only with independent hair
    # evidence, an opaque subject matte and a nearby existing hair boundary.
    support = out > .5
    radius = max(1, round(width*.10))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*radius+1, 2*radius+1))
    enclosed = ndimage.binary_fill_holes(cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE, kernel))
    hair_vote = (parsed['hair'] > .65) | (state['prob_hair'] > .8)
    repair = enclosed & (out < .98) & (subject > .97) & (hair > .65) & hair_vote
    near = ndimage.distance_transform_edt(~support) <= width*.12
    repair &= near & ~protected & ~cloth_remove & ~neck_remove & ~color_remove
    out[repair] = subject[repair]
    fg = foreground.copy()
    fg[repair] = rgb[repair]
    return out.astype(np.float32), fg, {
        'consensus_clothing_removed': int(np.count_nonzero(cloth_remove & (alpha > 0))),
        'consensus_neck_removed': int(np.count_nonzero(neck_remove & (alpha > 0))),
        'connected_garment_color_removed': int(np.count_nonzero(color_remove & (alpha > 0))),
        'consensus_hair_restored': int(np.count_nonzero(repair)),
        'protected_pixels_changed': int(np.count_nonzero(protected & (out != alpha))),
    }
