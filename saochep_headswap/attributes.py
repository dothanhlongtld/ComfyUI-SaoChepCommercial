"""Reference-only semantic hints; no filename, ethnicity or gender captions."""
import numpy as np


def reference_attributes(labels, alpha):
    if labels.shape != alpha.shape:
        raise ValueError('Reference parsing and alpha must share geometry')
    visible = alpha > .5
    face_area = int(np.count_nonzero(np.isin(labels, range(1, 14)) & visible))
    hat_area = int(np.count_nonzero((labels == 18) & visible))
    glasses_area = int(np.count_nonzero((labels == 6) & visible))
    return {'hat': hat_area >= max(64, face_area*.15),
            'glasses': glasses_area >= max(16, face_area*.002),
            'face_pixels': face_area, 'hat_pixels': hat_area, 'glasses_pixels': glasses_area,
            'method': 'bisenet_reference_labels_inside_accepted_crop',
            'scope': 'Semantic evidence, not a guarantee of accessory identity'}


def accessory_prompt(attributes):
    extra = []
    if attributes.get('hat'):
        extra.append('The reference is wearing a hat. Keep that same hat on the head throughout the video; '
                     'match its shape, material and color from the reference image.')
    if attributes.get('glasses'):
        extra.append('Retain the same reference eyeglasses, with their frame shape and placement.')
    return ' '.join(extra)
