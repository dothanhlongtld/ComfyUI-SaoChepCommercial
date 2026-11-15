"""Semantic + local photometric boundary correction, no case-name routing."""
import cv2
import numpy as np
from scipy import ndimage


def correct_boundary(rgb,state,alpha,foreground,long_hair=False):
    p=state['aligned'];sub=state['subj_alpha'];h,w=alpha.shape
    fw,fh,cx,chin,my=[float(state[k]) for k in ['fw','fh','cx','anatomical_chin_y','my']]
    yy,xx=np.ogrid[:h,:w];out=alpha.copy();fg=foreground.copy()
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
    face=(p['face']>.75)&(sub>.9)
    removed=0;lane_removed=0
    # A background pixel adjacent to skin is not hair just because it remains
    # connected to the subject. Require both low head confidence and local
    # color disagreement before deleting it. Limit this to a thin cheek band.
    if face.any() and not long_hair:
        distance,idx=ndimage.distance_transform_edt(~face,return_indices=True)
        near=lab[idx[0],idx[1]]
        color_delta=np.linalg.norm(lab-near,axis=2)
        suspect=(out>.005)&(p['face']<.35)&(p['hair']<.12)&(p['hat']<.1)&(p['neck']<.2)
        suspect&=(distance<.09*fw)&(yy>my-.65*fh)&(yy<chin)&(color_delta>25)
        # Ignore sparse soft-alpha speckles; operate only when a visible
        # foreground patch actually exists outside the semantic face.
        visible=int(np.count_nonzero(suspect&(out>.5)))
        if visible>max(600,.0003*fw*fh):
            removed=int(suspect.sum());out[suspect]=0
    # The long side tress in a non-long-hair route can carry a collar bridge.
    # Estimate its lane from the continuation below the collar, not a funnel
    # centered on the face. This preserves asymmetric hair and removes the
    # attached bridge protruding toward the neck.
    deep=(out>.8)&(yy>chin+.45*fh)&(yy<chin+1.25*fh)
    if not long_hair and np.count_nonzero(deep)>.18*fw*fh:
        centers=[];left=[];right=[]
        for y in range(max(0,int(chin+.45*fh)),min(h,int(chin+1.25*fh))):
            row=(out[y]>.8).astype(np.uint8)
            n,cc,stats,_=cv2.connectedComponentsWithStats(row[None,:])
            if n>1:
                i=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);xs=np.where(cc[0]==i)[0]
                if len(xs)>.12*fw:left.append(xs.min());right.append(xs.max());centers.append((xs.min()+xs.max())/2)
        if len(left)>.25*fh:
            xl,xr=float(np.percentile(left,25)),float(np.percentile(right,75))
            lane_center=float(np.median(centers))
            # Only a single side tress separated from the face center.
            if abs(lane_center-cx)>.32*fw:
                core=deep&(xx>=xl+.04*fw)&(xx<=xr-.04*fw)
                if core.sum()>30:
                    hair_lab=np.median(lab[core],axis=0)
                    margin=-.02*fw
                    outside=(xx<xl-margin)|(xx>xr+margin)
                    dark=lab[...,0]<hair_lab[0]*.95
                    bridge=(out>.005)&outside&dark&(yy>my+.05*fh)&(yy<chin+.35*fh)&(p['face']<.15)&(p['neck']<.25)
                    lane_removed=int(bridge.sum());out[bridge]=0
                    # Close narrow cracks inside the head above the collar;
                    # background gaps below the jaw remain transparent.
                    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
                    solid=cv2.morphologyEx((out>.5).astype(np.uint8),cv2.MORPH_CLOSE,k)>0
                    restore=solid&(out<.9)&(sub>.95)&(yy<chin)&((p['face']+p['hair'])>.55)
                    out[restore]=sub[restore];fg[restore]=rgb[restore]
    recovered=0
    # A significant number of enclosed hair gaps signals destructive matting,
    # rather than normal isolated flyaway strands. Recover nearby hair only
    # when two parsers, the subject matte and scalp color all agree.
    if not long_hair:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(max(3,round(.05*fw))|1,)*2)
        closed=cv2.morphologyEx((alpha>.5).astype(np.uint8),cv2.MORPH_CLOSE,k)>0
        gaps=closed&(alpha<.9)&(p['hair']>.5)&(state['prob_hair']>.5)&(sub>.98)&(yy>my)&(yy<chin+.8*fh)
        if gaps.sum()>.01*fw*fh:
            scalp=(state['prob_hair']>.65)&(sub>.95)&(yy<my)
            if scalp.sum()>30:
                median=np.median(lab[scalp],axis=0)
                chroma=np.linalg.norm(lab[...,1:]-median[1:],axis=2)
                near=ndimage.distance_transform_edt(~((alpha>.8)&(p['hair']>.5)))<.08*fw
                recover=(out<.98)&(p['hair']>.5)&(state['prob_hair']>.5)&(sub>.98)&(chroma<12)&near
                recover&=(yy>my)&(yy<chin+.8*fh)&(abs(xx-cx)>.27*fw)
                recovered=int(recover.sum());out[recover]=sub[recover];fg[recover]=rgb[recover]
    return out.astype(np.float32),fg,{'cheek_foreign_removed':removed,'collar_bridge_removed':lane_removed,'hair_recovered':recovered}
