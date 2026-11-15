"""Head-only semantic ownership without the legacy horizontal neck allowance."""
import numpy as np


def strip_neck(alpha, parsed, geometry, headwear=None):
    points = np.asarray(geometry['landmarks'], np.float32)
    direction = points[1]-points[0]
    direction /= max(float(np.linalg.norm(direction)), 1.)
    down = np.array([-direction[1], direction[0]])
    if down[1] < 0:
        down *= -1
    mouth = points[3:5].mean(axis=0)
    yy, xx = np.indices(alpha.shape, np.float32)
    lower = (xx-mouth[0])*down[0]+(yy-mouth[1])*down[1] > 0
    neck = (parsed['neck'] > .18) & (parsed['neck'] > parsed['face']) & (parsed['neck'] > parsed['hair'])
    keep_face = parsed['face'] >= .5
    keep_hair = (parsed['hair'] >= .35) & (parsed['hair'] >= parsed['neck'])
    keep_items = parsed['hat'] >= .5
    if headwear is not None:
        keep_items |= headwear
    remove = lower & neck & ~keep_face & ~keep_hair & ~keep_items
    output = alpha.copy()
    output[remove] = 0
    return output, int(np.count_nonzero(remove & (alpha > 0)))
