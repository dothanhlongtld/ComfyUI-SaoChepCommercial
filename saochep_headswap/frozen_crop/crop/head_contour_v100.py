import cv2
import numpy as np
from scipy import ndimage


def refine_contour(rgb,state,alpha,foreground,long_hair=False):
    out=alpha.copy();fg=foreground.copy();p=state['aligned'];sub=state['subj_alpha']
    fw,fh,cx,chin,my=[float(state[k]) for k in ['fw','fh','cx','anatomical_chin_y','my']]
    h,w=alpha.shape;yy,xx=np.ogrid[:h,:w]
    restored=removed=0
    if long_hair:return out,fg,{'cheek_restored':0,'shoulder_spur_removed':0}
    # Repair a concave bite at the cheek/hair junction using original skin
    # colors. A head-side gap must not be mistaken for the exterior matte.
    face=(p['face']>.7)&(sub>.95)
    if face.any():
        radius=max(2,round(.07*fw));kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,)*2)
        closed=cv2.morphologyEx((out>.5).astype(np.uint8),cv2.MORPH_CLOSE,kernel)>0
        dist,idx=ndimage.distance_transform_edt(~face,return_indices=True)
        lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
        near=lab[idx[0],idx[1]]
        delta=np.linalg.norm(lab-near,axis=2)
        restore=closed&(out<.98)&(sub>.98)&(dist<.08*fw)&(delta<23)&(p['cloth']<.15)&(p['neck']<.2)
        restore&=(yy>my-.35*fh)&(yy<chin)&((p['face']>.08)|(state['prob_face']>.2))
        # A cast shadow below the cheek is often labelled as neck by both
        # passes. Recover only a narrow submandibular skin band from source.
        jaw_band=(dist<.09*fw)&(out<.98)&(sub>.98)&(delta<45)&(p['cloth']<.10)
        jaw_band&=(p['neck']>.5)&(yy>my-.2*fh)&(yy<chin-.05*fh)
        restore|=jaw_band
        if restore.sum()>max(30,.009*fw*fh):
            out[restore]=1;fg[restore]=rgb[restore];restored=int(restore.sum())
    # Detect a single deep side tress. Clothing stripes attached to it extend
    # sideways across the shoulder; hair continuation supplies the boundary.
    left=[];right=[];heights=[]
    for y in range(max(0,int(chin+.45*fh)),min(h,int(chin+1.25*fh))):
        xs=np.where(out[y]>.8)[0]
        if len(xs)>.12*fw:left.append(int(xs.min()));right.append(int(xs.max()));heights.append(y)
    if len(left)>.25*fh:
        xl,xr=np.percentile(left,25),np.percentile(right,75)
        if abs((xl+xr)/2-cx)>.32*fw and (xr-xl)>.15*fw:
            # Scope to collar/shoulder height, retain the tress itself.
            left_fit=np.polyfit(heights,left,1);right_fit=np.polyfit(heights,right,1)
            lane_left=np.clip(left_fit[0]*yy+left_fit[1],xl-.15*fw,xl+.15*fw)
            lane_right=np.clip(right_fit[0]*yy+right_fit[1],xr-.15*fw,xr+.15*fw)
            spur=(out>.005)&((xx<lane_left-.015*fw)|(xx>lane_right-.03*fw))&(yy>my-.05*fh)&(yy<chin+.35*fh)
            spur&=(p['face']<.1)&(p['neck']<.2)
            removed=int(spur.sum());out[spur]=0
    return out.astype(np.float32),fg,{'cheek_restored':restored,'shoulder_spur_removed':removed}
