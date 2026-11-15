"""Small CPU parsing pass; reused ONNX session, no additional GPU model."""
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from .frozen_crop.paths import asset_path


@lru_cache(maxsize=1)
def session():
    model=asset_path('bisenet_face_parsing.onnx')
    options=ort.SessionOptions()
    options.intra_op_num_threads=2
    return ort.InferenceSession(str(model),options,providers=['CPUExecutionProvider'])


def parse_driving(rgb, points):
    height,width=rgb.shape[:2]
    eyes=np.array([points[[33,133]].mean(0),points[[263,362]].mean(0)])
    center=eyes.mean(0)
    span=float(np.linalg.norm(eyes[1]-eyes[0]))
    if not np.isfinite(span) or span<=0:raise ValueError('Invalid parsing geometry')
    # The ROI includes face + possible neck; ratios do not depend on image size.
    left=max(0,int(center[0]-2*span));right=min(width,int(center[0]+2*span))
    top=max(0,int(center[1]-span));bottom=min(height,int(center[1]+3*span))
    if left>=right or top>=bottom:raise ValueError('Empty face parsing ROI')
    crop=cv2.resize(rgb[top:bottom,left:right],(512,512))
    blob=((crop-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32))
    model=session()
    logits=model.run(None,{model.get_inputs()[0].name:blob.transpose(2,0,1)[None].astype(np.float32)})[0][0]
    exp=np.exp(logits-logits.max(axis=0,keepdims=True))
    prob=exp/exp.sum(axis=0,keepdims=True)
    def project(value):
        result=np.zeros((height,width),np.float32)
        result[top:bottom,left:right]=cv2.resize(value,(right-left,bottom-top))
        return result
    return project(prob[14]),project(prob[1:14].sum(0))
