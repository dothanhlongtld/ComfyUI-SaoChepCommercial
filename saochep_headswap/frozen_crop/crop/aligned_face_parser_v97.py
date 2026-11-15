import cv2,numpy as np


def aligned_parse(rgb,landmarks,fw,fh,session,mean,std):
    landmarks=np.asarray(landmarks,float)
    e1,e2=landmarks[0],landmarks[1]
    if e1[0]>e2[0]:e1,e2=e2,e1
    u=(e2-e1)/max(1,np.linalg.norm(e2-e1));v=np.array([-u[1],u[0]])
    center=(e1+e2)/2+v*.22*fh
    span=max(fw,fh)*1.9;scale=512./span
    matrix=np.array([[scale*u[0],scale*u[1],256-scale*np.dot(center,u)],
                     [scale*v[0],scale*v[1],256-scale*np.dot(center,v)]],np.float32)
    crop=cv2.warpAffine(rgb,matrix,(512,512),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    blob=np.transpose((crop.astype(np.float32)/255.-mean)/std,(2,0,1))[None]
    logits=session.run(None,{session.get_inputs()[0].name:blob})[0][0]
    exp=np.exp(logits-logits.max(0,keepdims=True));prob=exp/(exp.sum(0,keepdims=True)+1e-6)
    inverse=cv2.invertAffineTransform(matrix);h,w=rgb.shape[:2]
    project=lambda a:cv2.warpAffine(a,inverse,(w,h),flags=cv2.INTER_LINEAR,borderValue=0)
    return {'face':project(prob[1:14].sum(0)),'hair':project(prob[17]),'cloth':project(prob[15:17].sum(0)),
            'neck':project(prob[14]),'hat':project(prob[18]),
            'labels':cv2.warpAffine(prob.argmax(0).astype(np.uint8),inverse,(w,h),flags=cv2.INTER_NEAREST,borderValue=0)}
