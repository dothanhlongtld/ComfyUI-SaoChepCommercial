"""Long-hair refinement, shared by cached backtests and on-flow inference.

No filename routing or image-specific coordinates. Uncertain garment/skin is
background for the hair graph cut; the original face and micro-neck are merged
afterwards. This is a candidate, not a certified semantic garment detector.
"""
import cv2
import numpy as np
from scipy import ndimage


def refine_long_hair(img_rgb, state):
    h, w = img_rgb.shape[:2]
    def val(k): return np.asarray(state[k])
    subject = val('subj_alpha')
    ph, pc = val('prob_hair'), val('prob_cloth')
    pn, pf = val('prob_neck'), val('prob_face')
    labels = val('p_full')
    chin, fw, fh, cx = (float(val(k)) for k in ('anatomical_chin_y','fw','fh','cx'))
    yy, xx = np.ogrid[:h, :w]
    below = yy > chin
    # Restrict optimization to a head-scaled region, preserving asymmetric hair.
    domain = (subject > .08) & (abs(xx-cx) < 1.55*fw) & (yy < chin+3.2*fh)
    scalp = (ph > .75) & (subject > .85) & (yy < chin-.18*fh)
    if scalp.sum() < 40:
        return val('baseline_alpha').copy(), val('baseline_rgb').copy(), {'refinement': 'insufficient_hair_seeds'}
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    skin_seed = (pf > .85) & (labels == 1) & (abs(xx-cx) < .35*fw) & (yy < chin-.2*fh)
    skin = np.zeros((h,w), bool)
    if skin_seed.sum() > 30:
        sp = lab[skin_seed]
        med = np.median(sp, axis=0)
        scale = np.maximum(np.std(sp,axis=0), [18,5,5])
        sd = np.sqrt(np.sum(((lab-med)/scale)**2 * [0.25,1,1],axis=2))
        # Skin is a veto only where it is also supported by the semantic parser.
        skin = below & (sd < 2.5) & ((pf+pn > .15) | (ph < .20))
    garment = (pc > .35) & below
    torso_sample = (subject > .90) & (abs(xx-cx)<.28*fw) & (yy>chin+.6*fh) & (yy<chin+1.25*fh)
    if torso_sample.sum()>40:
        gm = np.median(lab[torso_sample],axis=0)
        hm = np.median(lab[scalp],axis=0)
        if np.linalg.norm(gm[1:]-hm[1:])>22:
            dg=np.linalg.norm(lab[...,1:]-gm[1:],axis=2)
            dh=np.linalg.norm(lab[...,1:]-hm[1:],axis=2)
            garment |= below & (dg<12) & (dh>22)
    face_neck = np.isin(labels, list(range(1,16))) & (ph < .30)
    negative = (~domain) | skin | garment | face_neck
    seed = scalp & (~negative)
    gc = np.full((h,w), cv2.GC_PR_BGD, np.uint8)
    gc[domain & (ph > .18)] = cv2.GC_PR_FGD
    gc[negative] = cv2.GC_BGD
    gc[seed] = cv2.GC_FGD
    # Reduced optimization grid limits CPU cost without distorting aspect ratio.
    scale = min(1., 900./max(h,w))
    size = (max(1,round(w*scale)), max(1,round(h*scale)))
    small = cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)
    gs = cv2.resize(gc, size, interpolation=cv2.INTER_NEAREST)
    cv2.setRNGSeed(96)
    cv2.grabCut(small, gs, None, np.zeros((1,65),np.float64), np.zeros((1,65),np.float64), 5, cv2.GC_INIT_WITH_MASK)
    hair = cv2.resize(np.isin(gs,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)>0
    hair &= domain & (~skin)
    # Keep only components anchored to genuine scalp hair, including strands
    # separated by an occluding hand when the parser supplies strong evidence.
    _, cc = cv2.connectedComponents(hair.astype(np.uint8))
    anchors = np.unique(cc[seed | ((ph > .85) & below & (~negative))])
    hair &= np.isin(cc, anchors[anchors>0])
    # Face, ears, glasses and micro-neck come from the existing protected core.
    sanctuary = val('head_sanctuary').astype(bool)
    semantic_face = np.isin(labels,range(1,14)) & (subject > .25) & (yy < chin+.12*fh)
    _, face_cc = cv2.connectedComponents(semantic_face.astype(np.uint8))
    core = val('solid_interior_core').astype(bool)
    face_ids = np.unique(face_cc[core & semantic_face])
    face = semantic_face & np.isin(face_cc,face_ids[face_ids>0])
    face |= core | (sanctuary & (pf > .12) & (yy <= chin) & np.isin(labels,range(1,14)))
    face = ndimage.binary_fill_holes(face)
    hat = sanctuary & (labels == 18) & (yy <= chin)
    # Follow the parsed jaw rather than a horizontal chin line. A rolled head
    # otherwise loses its chin and leaves isolated strips of neck.
    face_distance = ndimage.distance_transform_edt(~face)
    neck = (face_distance <= .065*fh) & (face_distance > 0) & (pn > .35) & (subject > .80) & (abs(xx-cx)<.38*fw)
    support = hair | face | hat | neck
    support |= (val('baseline_alpha')>.005) & (yy<float(val('my'))) & (abs(xx-cx)<.9*fw) & (pc<.25) & (labels!=14)
    alpha = subject * support.astype(np.float32)
    alpha[core & support] = 1.
    # Retain v9.5's existing soft neck termination rather than growing the neck.
    old = val('baseline_alpha')
    alpha[neck] = subject[neck] * np.clip((.065*fh-face_distance[neck])/max(1.,.015*fh),0,1)
    # Feather inward only: never blur garment colors into the extracted hair.
    dist = cv2.distanceTransform(support.astype(np.uint8),cv2.DIST_L2,3)
    alpha *= np.clip(dist / max(1., .006*fw),0,1)
    fg = img_rgb.copy()
    # Foreground colors at a matte edge must come from hair, never nearby skin
    # or another object. This prevents colored-background halos on flyaways.
    clean_hair = hair & (subject > .97) & (dist > max(2.,.012*fw))
    fringe = hair & (alpha > .002) & (subject < .98)
    if clean_hair.any() and fringe.any():
        _, idx = ndimage.distance_transform_edt(~clean_hair,return_indices=True)
        near = img_rgb[idx[0],idx[1]].astype(np.float32)
        pull = np.clip((.98-subject)/.75,0,.95)[...,None]
        corrected = img_rgb*(1-pull)+near*pull
        fg[fringe] = corrected.astype(np.uint8)[fringe]
    return alpha.astype(np.float32), fg, {'refinement': 'seeded_hair_graphcut', 'hair_pixels': int(hair.sum()), 'skin_veto_pixels':int(skin.sum())}
