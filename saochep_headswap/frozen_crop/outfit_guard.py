import cv2
import numpy as np

def detect_geometry(rgb,session):
    height,width=rgb.shape[:2]
    bgr=np.ascontiguousarray(rgb[...,::-1])
    blob=np.transpose(cv2.resize(bgr,(640,640)).astype(np.float32)/255.,(2,0,1))[None]
    rows=session.run(None,{session.get_inputs()[0].name:blob})[0][0].T
    valid=np.where(rows[:,4]>=.4)[0]
    if not len(valid):raise RuntimeError('Human face lost; cannot safely compose head-only output')
    row=rows[valid[np.argmax(rows[valid,4])]]
    cx,cy,fw,fh=row[:4]*np.array([width/640,height/640,width/640,height/640])
    points=[[float(row[5+i*3]*width/640),float(row[6+i*3]*height/640)] for i in range(5)]
    return {'landmarks':points,'fw':float(fw),'fh':float(fh),'cx':float(cx),
        'box':[float(cx-fw/2),float(cy-fh/2),float(cx+fw/2),float(cy+fh/2)]}

