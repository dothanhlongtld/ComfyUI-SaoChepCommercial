import numpy as np

def validate_transition(before, after):
    if not before.any() or not after.any():raise ValueError('Empty SAM head')
    def box(m):
        y,x=np.where(m);return np.array([x.min(),y.min(),x.max()+1,y.max()+1],float)
    a,b=box(before),box(after)
    ratio=after.sum()/before.sum()
    movement=np.linalg.norm((a[:2]+a[2:])/2-(b[:2]+b[2:])/2)
    overlap=np.count_nonzero(before&after)/min(before.sum(),after.sum())
    if not .5<=ratio<=2 or movement>max(a[2:]-a[:2]) or overlap<.25:
        raise ValueError('Uncertain SAM head continuity')
