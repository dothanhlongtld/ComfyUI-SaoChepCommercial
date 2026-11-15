import cv2
import numpy as np
from scipy import ndimage


def finish_head(rgb,state,alpha,foreground,long_hair=False):
    p=state['aligned'];sub=state['subj_alpha']
    fw,fh,cx,chin,my=[float(state[k]) for k in ['fw','fh','cx','anatomical_chin_y','my']]
    h,w=alpha.shape;yy,xx=np.ogrid[:h,:w]
    out=alpha.copy();fg=foreground.copy()
    # Restrict short-neck repairs to outputs where the existing neck is absent.
    face=(p['face']>.6)&(sub>.95)
    count,cc,stats,_=cv2.connectedComponentsWithStats(face.astype(np.uint8))
    if count>1:
        main=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);face=cc==main
    distance=ndimage.distance_transform_edt(~face)
    neck=(p['neck']>.65)&(abs(xx-cx)<.32*fw)&(distance<.095*fh)&(sub>.95)
    existing=np.count_nonzero(neck&(out>.8))
    neck_added=0
    if not long_hair and neck.sum()>max(50,.008*fw*fh) and existing/max(1,neck.sum())<.25:
        # Close the face support first: a garment veto may have grazed a cheek.
        restore=(p['face']>.45)&(sub>.95)&(p['cloth']<.1)
        out[restore]=1;fg[restore]=rgb[restore]
        a=np.clip((.095*fh-distance)/max(2,.018*fh),0,1)
        neck_added=int(np.count_nonzero(neck&(a>out)))
        out[neck]=np.maximum(out[neck],a[neck]);fg[neck]=rgb[neck]
    # Only semantic garment regions below the jaw, with neither pass supplying
    # credible hair, may cut into a dark hair silhouette.
    bad=(p['cloth']>.6)&(p['hair']<.10)&(state['prob_hair']<.25)&(yy>my)&(p['face']<.1)&(p['neck']<.2)
    removed=int(np.count_nonzero(bad&(out>.1)))
    if not long_hair and removed>max(600,.02*fw*fh):out[bad]=0
    else:removed=0
    # A dark collar can be labelled as hair by both parsers. Only shave narrow
    # protrusions on a long side tress when their color differs from its core.
    tress=(out>.8)&(yy>chin+.25*fh)
    if not long_hair and np.count_nonzero(tress&(yy>chin+fh))>.15*fw*fh:
        lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
        r=max(2,round(.07*fw));k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,)*2)
        opened=cv2.morphologyEx((out>.5).astype(np.uint8),cv2.MORPH_OPEN,k)>0
        clean=cv2.erode(tress.astype(np.uint8),np.ones((5,5),np.uint8))>0
        if clean.sum()>20:
            median=np.median(lab[clean],axis=0)
            spur=(~opened)&(out>.05)&(yy>my)&(yy<chin+.35*fh)&(lab[...,0]<median[0]*.95)&(np.linalg.norm(lab-median,axis=2)>8)
            spur |= (out>.05)&(yy>chin-.10*fh)&(yy<chin+.30*fh)&(lab[...,0]<median[0]*.55)&(p['face']<.1)&(p['neck']<.2)
            out[spur]=0;removed+=int(spur.sum())
    # Fill ragged internal hair holes only when both the image matte and the
    # aligned parser identify actual hair, never by dilation onto a garment.
    r=max(1,round(.025*fw));kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
    closed=cv2.morphologyEx((out>.5).astype(np.uint8),cv2.MORPH_CLOSE,kernel)>0
    add=closed&(out<.9)&(p['hair']>.5)&(state['prob_hair']>.5)&(sub>.98)&(yy>my)&(yy<chin+.8*fh)
    hair_added=int(add.sum())
    if not long_hair and hair_added>max(30,.002*fw*fh):out[add]=sub[add];fg[add]=rgb[add]
    else:hair_added=0
    # Remove only a thin unsupported fringe next to a short-haired face.
    # The face/ear sanctuary is never eroded by this cleanup.
    rim=(out>.005)&(p['face']<.12)&(p['hair']<.12)&(p['hat']<.1)&(p['neck']<.15)&(yy>my-.25*fh)&(yy<chin)
    distance=cv2.distanceTransform((out>.05).astype(np.uint8),cv2.DIST_L2,3)
    shave=rim&(distance<max(1.5,.008*fw))
    shaved=0
    if not long_hair and np.count_nonzero(shave)>max(200,.0005*fw*fh):
        out[shave]=0;shaved=int(shave.sum())
    return out.astype(np.float32),fg,{'neck_added_pixels':neck_added,'garment_pixels_removed':removed,'hair_holes_restored':hair_added,'unsupported_rim_removed':shaved}
