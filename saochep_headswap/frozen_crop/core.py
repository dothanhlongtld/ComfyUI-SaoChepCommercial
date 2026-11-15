import cv2
import numpy as np

def alignment_transform(reference,driver):
    rg=reference.get('geometry');dg=driver.get('geometry')
    if dg is None:raise ValueError('Driving face not detected: cannot bound head-only replacement')
    if rg is not None:
        a=np.asarray(rg['landmarks'],np.float32);b=np.asarray(dg['landmarks'],np.float32)
        transform,_=cv2.estimateAffinePartial2D(a,b,method=cv2.LMEDS)
    else:
        # Animal head references still need a human driver with detectable pose.
        x1,y1,x2,y2=reference['head_bbox'];dx1,dy1,dx2,dy2=driver['head_bbox']
        scale=min((dx2-dx1+1)/max(1,x2-x1+1),(dy2-dy1+1)/max(1,y2-y1+1))
        transform=np.array([[scale,0,(dx1+dx2)/2-scale*(x1+x2)/2],
                            [0,scale,dy2-scale*y2]],np.float32)
    if transform is None or not np.isfinite(transform).all():raise ValueError('Invalid head alignment')
    return transform

