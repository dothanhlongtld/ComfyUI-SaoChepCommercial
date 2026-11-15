"""Preserve head interiors; remove evidence-backed foreign objects at edges.

All geometry is in face-relative units. Filenames and acceptance-case lists are
used only by the backtest, never by this production module.
"""
import cv2
import numpy as np
from scipy import ndimage


def preserve_head(rgb,state,alpha,foreground,long_hair=False):
    parsed=state['aligned'];subj=state['subj_alpha']
    fw,fh,cx,my,chin=[float(state[k]) for k in ['fw','fh','cx','my','anatomical_chin_y']]
    h,w=alpha.shape;yy,xx=np.ogrid[:h,:w]
    labels=parsed['labels'];face_prob=parsed['face'];hair_prob=parsed['hair']
    # High-resolution, roll-aligned face labels now include the ears and jaw.
    face_raw=(face_prob>.28)&(subj>.80)
    # A few pixels of skin-colored clothing can be labelled as face. Keep the
    # principal face component plus nearby ears, not distant neck/body islands.
    count,cc,stats,_=cv2.connectedComponentsWithStats(face_raw.astype(np.uint8))
    if count>1:
        overlaps=np.bincount(cc[state['solid_interior_core']].ravel(),minlength=count);overlaps[0]=0
        main=int(np.argmax(overlaps)) if overlaps.max()>0 else 1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        principal=cc==main
        near=ndimage.distance_transform_edt(~principal)<.23*fw
        face_raw &= principal | (near & np.isin(labels,[7,8,9]) & (yy<=chin))
    radius=max(1,round(.012*fw));kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,)*2)
    face=cv2.morphologyEx(face_raw.astype(np.uint8),cv2.MORPH_CLOSE,kernel)>0
    face=ndimage.binary_fill_holes(face)
    face=cv2.dilate(face.astype(np.uint8),kernel)>0
    face &= (subj>.85)&(parsed['cloth']<.50)
    # Restoring a solid face must not mean restoring a long neck. Parsing's
    # neck class is excluded from this protected face support.
    face &= (parsed['neck']<.50)
    core=cv2.erode(((face_prob>.60)&(subj>.95)).astype(np.uint8),np.ones((3,3),np.uint8))>0
    deficit=int(np.count_nonzero(core&(alpha<.5)))
    threshold=max(20,round(.0004*fw*fh))
    repair_face=long_hair or deficit>threshold
    out=alpha.copy();fg=foreground.copy()
    if repair_face:
        # Closing face/tress seams may restore hair pixels beside the cheek;
        # it does not change the waterfall hair farther from the face.
        closed=cv2.morphologyEx((out>.5).astype(np.uint8),cv2.MORPH_CLOSE,kernel)>0
        closed=ndimage.binary_fill_holes(closed)
        seam=closed & (cv2.dilate(face.astype(np.uint8),kernel)>0)&(subj>.90)
        solid=face|seam
        out[solid]=1.
        fg[solid]=rgb[solid]
        # Build a short, continuous neck attached to the parsed jaw, including
        # a lifted chin. Distance is measured from the jaw, not a screen row.
        distance=ndimage.distance_transform_edt(~face_raw)
        neck=(parsed['neck']>.50)&(distance<=.075*fh)&(subj>.9)&(abs(xx-cx)<.45*fw)
        neck_alpha=np.clip((.075*fh-distance)/max(1,.015*fh),0,1).astype(np.float32)
        out[neck]=np.maximum(out[neck],neck_alpha[neck])
        fg[neck]=rgb[neck]
    protect=cv2.dilate(face.astype(np.uint8),kernel)>0
    protect |= state['solid_interior_core'] & np.isin(state['p_full'],[1,2,3,4,5,6,10,11,12,13])
    # Retain the user's good hair extraction. Only positively separated
    # garment colors, or low-confidence non-head objects, may be removed.
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
    scalp=(state['prob_hair']>.7)&(subj>.8)&(yy<my)
    garment=(state['prob_cloth']>.8)&(subj>.9)&(yy>my)
    veto=np.zeros((h,w),bool)
    hm=np.median(lab[scalp],axis=0) if scalp.sum()>30 else None
    if scalp.sum()>30 and garment.sum()>30:
        samples=lab[garment][::max(1,int(garment.sum()/4000))]
        cv2.setRNGSeed(97)
        _,_,centers=cv2.kmeans(samples.astype(np.float32),min(3,len(samples)),None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,.2),1,cv2.KMEANS_PP_CENTERS)
        for g in centers:
            if np.linalg.norm(g[1:]-hm[1:])<22:continue
            dg=np.linalg.norm(lab[...,1:]-g[1:],axis=2)
            dh=np.linalg.norm(lab[...,1:]-hm[1:],axis=2)
            veto|=(yy>my)&(dg<15)&(dh>23)&(dg<dh*.55)&(~protect)
    landmarks=np.asarray(state['landmarks'],float)
    eye_y=float(np.mean(landmarks[:2,1]))
    if not long_hair:
        confidence=np.maximum.reduce([hair_prob,face_prob,parsed['hat']])
        foreign=(confidence<.12)&(state['prob_hair']<.18)&(~protect)&(yy>eye_y-.4*fh)&(yy<chin+.05*fh)
        foreign|=(parsed['cloth']>.65)&(~protect)&(yy>my)&(yy<chin+.05*fh)
        # Preserve a legitimate neck stub even when the human parser calls it
        # background; never infer a garment merely from its dark color.
        foreign &= (abs(xx-cx)>.45*fw)
        foreign &= ~((abs(xx-cx)<.28*fw)&(yy>my)&(parsed['neck']>.3))
        # Skin-like fingers touching the temple can be mislabelled as hair.
        # Their luminance and chroma disagree with the actual scalp prototype.
        primary_foreign=int(np.count_nonzero(foreign&(alpha>.1)))
        if scalp.sum()>30 and primary_foreign>max(500,.03*fw*fh):
            skin_seed=(face_prob>.85)&(labels==1)&(subj>.95)
            if skin_seed.sum()>30:
                skin_med=np.median(lab[skin_seed],axis=0)
                dh=np.linalg.norm(lab-hm,axis=2)
                ds=np.linalg.norm(lab-skin_med,axis=2)
                foreign|=(dh>45)&(ds<42)&(~protect)&(abs(xx-cx)>.52*fw)&(yy>eye_y-.40*fh)&(yy<chin)
                # A temple occluder may itself be parsed as an ear. Permit
                # removal only at the outside of the skull and only after a
                # substantial foreign-object region was independently found.
                foreign|=(dh>45)&(ds<42)&(abs(xx-cx)>.50*fw)&(yy>eye_y-.40*fh)&(yy<eye_y+.05*fh)
        if np.count_nonzero(foreign&(alpha>.1))>max(30,.001*fw*fh):veto|=foreign
    removed=int(np.count_nonzero(veto&(out>.005)))
    if not long_hair and removed<max(200,.005*fw*fh):veto[:]=False;removed=0
    if not long_hair and removed>0:
        # Once an independently detected foreign object exists, finish the
        # cut all the way to its attachment near the jaw. A fixed lateral
        # cutoff would leave a loop of shoulder or car headrest behind.
        veto|=(parsed['cloth']>.55)&(~protect)&(yy>my)
        veto|=(confidence<.10)&(state['prob_hair']<.15)&(~protect)&(yy>eye_y-.2*fh)&(yy<chin)
        removed=int(np.count_nonzero(veto&(out>.005)))
    out[veto]=0
    # Remove detached debris left by an occluder cut, while preserving small
    # hair strands supported by either parser and earrings near the face.
    candidate=(out>.05).astype(np.uint8)
    _,cc=cv2.connectedComponents(candidate)
    ids=np.unique(cc[(face_prob>.5)&(out>.5)])
    detached=(~np.isin(cc,ids[ids>0]))&(out>.005)&(~protect)&(hair_prob<.3)&(state['prob_hair']<.3)
    if long_hair or removed==0:detached[:]=False
    out[detached]=0
    if removed>0:
        # Garment cleanup cannot punch new holes in the aligned face.
        out[face]=np.maximum(out[face],alpha[face])
    fg,spill_count=remove_chroma_spill(rgb,fg,subj,out)
    # Never alter an approved-looking image merely because its full-frame
    # RGB contains arbitrary values in fully transparent pixels.
    report={'face_core_missing_before':deficit,'face_repair_threshold':threshold,
            'face_repaired':bool(repair_face),'foreign_removed_pixels':removed,
            'debris_removed_pixels':int(np.count_nonzero(detached)),
            'chroma_spill_corrected_pixels':spill_count,
            'protected_face_nonopaque_pixels':int(np.count_nonzero(core&(out<.99)))}
    return out.astype(np.float32),fg,report


def remove_chroma_spill(rgb,foreground,subject,alpha):
    """Remove saturated, spatially coherent background chroma at matte edges.

    Alpha and interior colors are immutable here. Neutral or varied background
    colors do not trigger the correction, avoiding global recoloring of hair.
    """
    h,w=alpha.shape
    ys,xs=np.where(alpha>.1)
    if not len(ys):return foreground,0
    x1,x2=max(0,int(xs.min())-8),min(w,int(xs.max())+9)
    y1,y2=max(0,int(ys.min())-8),min(h,int(ys.max())+9)
    # Estimate background only in the vicinity of the extracted head.
    area=np.zeros((h,w),bool);area[y1:y2,x1:x2]=True
    bg_mask=(subject<.01)&area
    if bg_mask.sum()<30:return foreground,0
    pixels=rgb[bg_mask].astype(np.float32)
    median=np.median(pixels,axis=0)
    chroma=median-median.mean()
    # A dominant colored backdrop (such as blue sky) is measurable evidence
    # for spill. Mixed indoor scenery is too ambiguous for this operation.
    if np.linalg.norm(chroma)<35 or np.median(np.linalg.norm(pixels-median,axis=1))>18:return foreground,0
    clean=(subject>.99)&(alpha>.99)
    clean=cv2.erode(clean.astype(np.uint8),np.ones((3,3),np.uint8))>0
    if not clean.any():return foreground,0
    _,bi=ndimage.distance_transform_edt(~bg_mask,return_indices=True)
    _,fi=ndimage.distance_transform_edt(~clean,return_indices=True)
    c=rgb.astype(np.float32);b=rgb[bi[0],bi[1]].astype(np.float32);f=rgb[fi[0],fi[1]].astype(np.float32)
    cc=c-c.mean(2,keepdims=True);bc=b-b.mean(2,keepdims=True);fc=f-f.mean(2,keepdims=True)
    direction=bc-fc
    amount=np.sum((cc-fc)*direction,axis=2)/(np.sum(direction*direction,axis=2)+1e-5)
    rim=(alpha>.005)&(subject<.97)&(subject>.05)&(~clean)&(amount>.10)
    out=foreground.copy()
    corrected=np.clip(c-np.clip(amount,0,.85)[...,None]*direction,0,255).astype(np.uint8)
    out[rim]=corrected[rim]
    return out,int(np.count_nonzero(rim))


def extend_animal_neck(subject,alpha):
    ys,xs=np.where(alpha>.05)
    if not len(ys):return alpha,{'animal_neck':'empty'}
    top,bottom=int(ys.min()),int(ys.max());height=bottom-top+1
    widths=np.sum(subject>.5,axis=1).astype(float)
    before=np.median(widths[max(top,bottom-round(height*.15)):bottom+1])
    after=np.percentile(widths[bottom+1:min(len(widths),bottom+1+round(height*.12))],85)
    head_width=int(xs.max()-xs.min()+1)
    if before/max(head_width,1)>.55 or after/max(before,1)<1.17:return alpha,{'animal_neck':'already_present','neck_rows_added':0}
    stub=max(3,round(height*.055));stop=min(len(widths)-1,bottom+stub)
    for y in range(bottom+1,stop+1):
        if widths[y]>before*1.30:stop=max(bottom+1,y-1);break
    yy=np.arange(alpha.shape[0])[:,None]
    band=(yy>=bottom-round(height*.04))&(yy<=stop)
    add=subject*np.clip((stop+1-yy)/max(2,stub*.15),0,1)
    out=np.maximum(alpha,np.where(band,add,0)).astype(np.float32)
    return out,{'animal_neck':'extended_before_shoulder_flare','neck_rows_added':int(stop-bottom)}
