"""Estimate hair-versus-garment alpha with an automatically generated trimap."""
import cv2
import numpy as np
from scipy import ndimage
if __package__:
    from .crop_semantic_consensus import headwear_mask
else:
    from crop_semantic_consensus import headwear_mask


def make_trimap(alpha, state, probabilities, rgb=None, body_zone=None):
    parsed = state['aligned']
    face = (parsed['face'] > .6) | state['solid_interior_core']
    items = headwear_mask(parsed, state['p_full'], alpha, state, rgb)
    protected = face | items
    radius = max(1, round(float(state['fw'])*.035))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    support = alpha > .05
    inner = cv2.erode(support.astype(np.uint8), kernel) > 0
    outer = cv2.dilate(support.astype(np.uint8), kernel) > 0
    hair = probabilities[1]
    garment = (probabilities[4] > .85) & (hair < .2)
    garment &= np.isin(state['p_full'], [15, 16]) | (parsed['cloth'] > .6)
    ambiguity = state.get('garment_color_ambiguity', np.zeros(alpha.shape, bool))
    if body_zone is not None:
        body, clothing = probabilities[2], probabilities[4]
        clothing_dominates = (clothing > .55) & (clothing > hair+.2) & (clothing > body)
        skin_dominates = (body > .45) & (body > hair) & (parsed['neck'] > .1)
        ambiguity = ambiguity | (body_zone & (clothing_dominates | skin_dominates))
    foreground = (inner & ~garment & ~ambiguity) | (protected & (alpha > .98))
    background = (~outer | garment) & ~foreground
    trimap = np.full(alpha.shape, 128, np.uint8)
    trimap[background] = 0
    trimap[foreground] = 255
    possible_hair = support & ((parsed['hair'] > .2) | (state['prob_hair'] > .3) | (hair > .4))
    domain = cv2.dilate(possible_hair.astype(np.uint8), kernel) > 0
    domain &= ~protected
    return trimap, domain, protected


def separate_hair_over_body(rgb, alpha, foreground, state, probabilities, editable):
    """Remove nearby sweater/skin color from mixed pixels below the face."""
    hair, skin, face, cloth = (probabilities[index] for index in (1, 2, 3, 4))
    parsed = state['aligned']
    editable = editable & (alpha > .02) & (face < .25)
    cloth_mix = editable & (cloth > .26) & (cloth > skin) & (hair < .7)
    skin_mix = editable & (skin > .34) & (hair < .64) & (parsed['neck'] > .08)
    hair_seed = (alpha > .93) & (hair > .75) & (face < .25)
    hair_seed &= (parsed['hair'] > .6) | (state['prob_hair'] > .6)
    if not (cloth_mix | skin_mix).any() or not hair_seed.any():
        return alpha, foreground, 0

    width = float(state['fw'])
    image = rgb.astype(np.float32)
    brightness = image.mean(axis=2)
    radius = max(1, round(width*.045))
    neighborhood = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,)*2)
    low_hair = cv2.erode(np.where(hair_seed, brightness, 255), neighborhood)
    # A bright garment thread can itself be mislabelled confident hair.
    # Use the nearby strand, away from outliers in the hair seed color.
    hair_seed &= brightness <= low_hair+22
    distance, nearest = ndimage.distance_transform_edt(~hair_seed, return_indices=True)
    corrected_alpha = alpha.copy()
    corrected_color = foreground.copy()
    changed = 0
    strand_evidence = state.get('strand_evidence')
    for mixed, background in ((cloth_mix, cloth), (skin_mix, skin)):
        nearby_agreement = np.zeros(alpha.shape, dtype=bool)
        if strand_evidence is not None:
            confirmed = mixed & (strand_evidence < .42) & (background > .48)
            radius = max(1, round(width*.022))
            neighborhood = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,)*2)
            nearby_agreement = cv2.dilate(confirmed.astype(np.uint8), neighborhood) > 0
            mixed |= editable & nearby_agreement & (background > .21) & (strand_evidence < .84)
        mixed &= distance < width*.19
        source = (alpha < .03) & (background > .78) & (hair < .15) & (face < .15)
        if not mixed.any() or not source.any():
            continue
        # Average nearby confirmed samples so the color of a single knit
        # stitch or shadow does not become the estimated layer underneath.
        sigma = max(1., width*.014)
        source_weight = source.astype(np.float32)
        density = cv2.GaussianBlur(source_weight, (0, 0), sigma)
        samples = cv2.GaussianBlur(image*source_weight[..., None], (0, 0), sigma)
        samples /= np.maximum(density[..., None], 1e-4)
        radius = max(1, round(width*.055))
        neighborhood = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,)*2)
        bright_background = cv2.dilate(np.where(source, brightness, 0), neighborhood)
        bg_distance, bg_index = ndimage.distance_transform_edt(~source, return_indices=True)
        ys, xs = np.nonzero(mixed & (bg_distance < width*.24))
        original = image[ys, xs]
        by, bx = bg_index[:, ys, xs]
        behind = samples[by, bx]
        strand = image[nearest[0, ys, xs], nearest[1, ys, xs]]
        direction = strand-behind
        contrast = np.linalg.norm(direction, axis=1)
        fraction = np.sum((original-behind)*direction, axis=1)/np.maximum(contrast**2, 1)
        fitted = behind + fraction[:, None]*direction
        error = np.linalg.norm(original-fitted, axis=1)
        # The original pixel must lie between the measured layers. Reject
        # low contrast, color mismatches and ambiguous near-solid strands.
        confidence = np.clip((contrast-45)/40, 0, 1)*np.clip((32-error)/16, 0, 1)
        layer_vote = np.clip((background[ys, xs]-hair[ys, xs]+.06)/.34, 0, 1)
        # Compare highlights against the observed brightness range of the
        # same local material, not only its average color.
        within_background_range = brightness[ys, xs] <= bright_background[by, bx]+6
        bright_match = (fraction < .28) & (error < 15) & within_background_range
        bright_match &= (contrast > 75) & (bg_distance[ys, xs] < width*.11)
        bright_match &= (distance[ys, xs] < width*.1)
        layer_vote = np.maximum(layer_vote, bright_match.astype(np.float32))
        shade_match = np.maximum(np.clip((fraction+.24)/.16, 0, 1), bright_match)
        confidence *= layer_vote*shade_match
        confidence *= np.clip((.88-fraction)/.16, 0, 1)
        coverage = np.clip(fraction, 0, 1)
        material_cut = np.zeros_like(coverage)
        if strand_evidence is not None:
            strand_score = strand_evidence[ys, xs]
            # A second hair-specific prediction breaks a tie between hair
            # and underlying material. Its low score alone cannot cut hair.
            low_strand = np.clip((.58-strand_score)/.28, 0, 1)
            material_vote = np.clip((background[ys, xs]-.24)/.15, 0, 1)
            same_layer = (error < 27) & (contrast > 72) & (fraction < .86)
            same_layer &= (bg_distance[ys, xs] < width*.16) & (distance[ys, xs] < width*.12)
            extra = low_strand*material_vote*same_layer
            # At the edge of an agreed mixed patch, a strong color fit can
            # remove the underlying color even when the mask is uncertain.
            continuing_mix = nearby_agreement[ys, xs] & (error < 12) & (contrast > 90)
            continuing_mix &= (bg_distance[ys, xs] < width*.12) & (fraction < .64)
            edge_mix = np.clip((.64-fraction)/.24, 0, 1)*continuing_mix
            extra = np.maximum(extra, edge_mix)
            matte_limit = np.clip((strand_score-.24)/.42, 0, 1)
            coverage -= extra*np.maximum(coverage-matte_limit, 0)
            confidence = np.maximum(confidence, extra)
            # Agreement between the classifiers can reject exposed body
            # material even when its local color sample is unreliable.
            material_cut = np.clip((background[ys, xs]-.45)/.15, 0, 1)
            material_cut *= np.clip((.42-strand_score)/.17, 0, 1)
            material_cut *= (hair[ys, xs] < .42) & (bg_distance[ys, xs] < width*.13)
        weight = confidence[:, None]
        old_alpha = alpha[ys, xs]
        new_alpha = old_alpha - confidence*np.maximum(old_alpha-coverage, 0)
        new_alpha = np.minimum(new_alpha, old_alpha*(1-material_cut))
        new_color = np.rint(foreground[ys, xs]*(1-weight)+strand*weight).clip(0, 255).astype(np.uint8)
        # Keep the visible contribution monotone even for unusual colors.
        safe = np.all(new_color*new_alpha[:, None] <= foreground[ys, xs]*old_alpha[:, None], axis=1)
        ys, xs = ys[safe], xs[safe]
        corrected_alpha[ys, xs] = new_alpha[safe]
        corrected_color[ys, xs] = new_color[safe]
        changed += int(np.count_nonzero(np.maximum(confidence[safe], material_cut[safe]) > .05))
    return corrected_alpha, corrected_color, changed


class TrimapMatte:
    def __init__(self, model_path):
        from transformers import VitMatteImageProcessor, VitMatteForImageMatting
        self.processor = VitMatteImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = VitMatteForImageMatting.from_pretrained(model_path, local_files_only=True).float().eval()

    def refine(self, rgb, alpha, foreground, state, probabilities, device='cpu', strand_predictor=None):
        import torch
        points = np.asarray(state['landmarks'], np.float32)
        horizontal = points[1]-points[0]
        horizontal /= max(float(np.linalg.norm(horizontal)), 1.)
        down = np.array([-horizontal[1], horizontal[0]])
        if down[1] < 0:
            down *= -1
        mouth = points[3:5].mean(axis=0)
        yy, xx = np.indices(alpha.shape, np.float32)
        depth = (xx-mouth[0])*down[0]+(yy-mouth[1])*down[1]
        extended = (alpha > .1) & ((state['aligned']['hair'] > .5) | (probabilities[1] > .7))
        extended &= depth > float(state['fh'])*.55
        if extended.sum() < float(state['fw'])*float(state['fh'])*.03:
            return alpha, foreground, {'trimap_applied': False, 'hair_alpha_changed': 0}
        body_zone = depth > float(state['fh'])*.12
        trimap, domain, protected = make_trimap(alpha, state, probabilities, rgb, body_zone)
        h, w = alpha.shape
        scale = min(1., 1024/max(h, w))
        size = (max(32, round(w*scale)), max(32, round(h*scale)))
        image = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        guide = cv2.resize(trimap, size, interpolation=cv2.INTER_NEAREST)
        self.model.to(device)
        try:
            inputs = self.processor(images=image, trimaps=guide, return_tensors='pt')
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                prediction = self.model(**inputs).alphas[0, 0].float().cpu().numpy()
            prediction = prediction[:size[1], :size[0]]
            prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_LINEAR).clip(0, 1)
        finally:
            self.model.to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        prediction[trimap == 0] = 0
        prediction[trimap == 255] = 1
        output = alpha.copy()
        # Consensus owns recoveries. Matting can refine this support inward,
        # but cannot pull bright source background back into the crop.
        editable = domain & (alpha > 0) & (trimap != 255)
        face = state['aligned']['face'] > .35
        outline = state.get('face_outline')
        if outline is not None:
            face |= outline
        radius = max(1, round(float(state['fw'])*.018))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        face_margin = cv2.dilate(face.astype(np.uint8), kernel) > 0
        editable &= ~face_margin
        # A soft edge may be refined. Opaque hair and hair across the face
        # must not be darkened just because matting disagrees with the crop.
        edge_weight = np.clip((.95-alpha)/.25, 0, 1)
        hair, body, clothing = (probabilities[index] for index in (1, 2, 4))
        clothing_weight = np.clip((clothing-.5)/.35, 0, 1)
        clothing_weight *= body_zone & (clothing > hair) & (clothing > body)
        weight = np.maximum(edge_weight, clothing_weight)
        output[editable] -= np.maximum(alpha[editable]-prediction[editable], 0)*weight[editable]

        # The body-skin model can win at the thin end of a hair lock even
        # when the face parser calls the mixture hair. Limit only that mixed
        # boundary, using the neck vote and the independent face outline.
        skin_edge = editable & body_zone & (alpha < .85)
        skin_edge &= (body > .45) & (body > hair) & (state['aligned']['neck'] > .1)
        hair_share = hair[skin_edge]/(hair[skin_edge]+body[skin_edge])
        output[skin_edge] = np.minimum(output[skin_edge], alpha[skin_edge]*hair_share)
        body_boundary = domain & ~face_margin & body_zone
        if strand_predictor is not None and body_boundary.any():
            state = {**state, 'strand_evidence': strand_predictor(rgb)}
        output, foreground, unmixed = separate_hair_over_body(
            rgb, output, foreground, state, probabilities, body_boundary)
        assert np.array_equal(output[protected], alpha[protected])
        return output, foreground, {'trimap_applied': True, 'trimap_unknown_pixels': int(np.count_nonzero(trimap == 128)),
                           'body_skin_edge_pixels': int(np.count_nonzero(skin_edge)),
                           'body_color_unmixed_pixels': unmixed,
                           'hair_alpha_changed': int(np.count_nonzero(output != alpha))}
