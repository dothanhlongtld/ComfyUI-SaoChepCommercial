import json, uuid, hashlib
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from .runtime import CropService, RUN_LOCK

def legacy_reference_layout(result,target_width=576,target_height=1024,clip_bg_mode='black'):
    alpha=result['head_alpha'];foreground=result['head_fg_rgb'];height,width=alpha.shape
    x1,y1,x2,y2=map(int,result['head_bbox'])
    center_x=(x1+x2)//2;center_y=(y1+y2)//2
    span=max(x2-x1,y2-y1,100);size=int(span*1.35)
    left=max(0,center_x-size//2);top=max(0,center_y-size//2)
    right=min(width,left+size);bottom=min(height,top+size)
    background=128. if clip_bg_mode=='midgray' else 0.
    def composite(rgb,a):return (rgb.astype(np.float32)*a[...,None]+background*(1-a[...,None])).astype(np.uint8)
    clip=cv2.resize(composite(foreground[top:bottom,left:right],alpha[top:bottom,left:right]),(512,512),interpolation=cv2.INTER_LANCZOS4)
    scale=min(target_width/width,target_height/height)
    nw,nh=max(1,round(width*scale)),max(1,round(height*scale))
    resized_rgb=cv2.resize(foreground,(nw,nh),interpolation=cv2.INTER_LANCZOS4)
    resized_alpha=cv2.resize(alpha,(nw,nh),interpolation=cv2.INTER_LINEAR)
    canvas=np.full((target_height,target_width,3),background,np.uint8)
    canvas_alpha=np.zeros((target_height,target_width),np.float32)
    px=(target_width-nw)//2;py=(target_height-nh)//2
    canvas[py:py+nh,px:px+nw]=composite(resized_rgb,resized_alpha)
    canvas_alpha[py:py+nh,px:px+nw]=resized_alpha
    tight_rgb=foreground[y1:y2,x1:x2];tight_alpha=alpha[y1:y2,x1:x2]
    if not tight_rgb.size:tight_rgb=foreground;tight_alpha=alpha
    tight=cv2.resize(composite(tight_rgb,tight_alpha),(512,512),interpolation=cv2.INTER_LANCZOS4)
    tight_alpha=cv2.resize(tight_alpha,(512,512),interpolation=cv2.INTER_LINEAR)
    return clip,canvas,canvas_alpha,tight,tight_alpha


from .crop_pipeline_v101 import reference_without_neck


class SaoChepHeadIsolationV100Legacy:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'image':('IMAGE',),'hood_mask':('MASK',)},'optional':{
            'clip_bg_mode':(['black','midgray'],{'default':'black'}),
            'lambda_strength':('FLOAT',{'default':1.,'min':1.,'max':1.}),
            'target_width':('INT',{'default':576,'min':256,'max':2048,'step':32}),
            'target_height':('INT',{'default':1024,'min':256,'max':2048,'step':32}),
            'matting_device':(['cuda','cpu'],{'default':'cuda'}),
            'driving_images':('IMAGE',),
            'layout_mode':(['legacy','driving_head'],{'default':'legacy'}),
            'neck_mode':(['short_neck','head_only'],{'default':'head_only'})}}
    RETURN_TYPES=('IMAGE','IMAGE','MASK','IMAGE','MASK','STRING')
    RETURN_NAMES=('clip_rgb','reference_rgb_hq','reference_alpha_hq','tight_crop_rgb_hq','tight_crop_alpha_hq','report_json')
    FUNCTION='isolate';CATEGORY='SaoChep/Clean Crop v101'
    def isolate(self,image,clip_bg_mode='black',lambda_strength=1.,target_width=576,target_height=1024,matting_device='cuda',driving_images=None,layout_mode='legacy',neck_mode='head_only',hood_mask=None):
        import folder_paths
        if len(image)!=1:raise ValueError('One reference per headswap job')
        if lambda_strength!=1.:raise ValueError('Crop v100 fidelity requires lambda_strength=1')
        if not RUN_LOCK.acquire(blocking=False):raise RuntimeError('Another SaoChep crop/headswap job is active')
        service=None
        try:
            service=CropService();service.activate(matting_device)
            rgb=np.rint(image[0].detach().cpu().numpy().clip(0,1)*255).astype(np.uint8)
            service.engine.hood_mask = None if hood_mask is None else hood_mask[0].detach().cpu().numpy()
            result=service.process(rgb)
            removed_neck=0
            if neck_mode=='head_only':result,removed_neck=reference_without_neck(result,rgb,service.engine)
            outputs=list(legacy_reference_layout(result,target_width,target_height,clip_bg_mode))
            reference_geometry=result.get('geometry');canvas_geometry=None
            if layout_mode=='driving_head':
                if driving_images is None:raise ValueError('Driving frames needed for reference layout')
                from .outfit_guard import detect_geometry
                from .core import alignment_transform
                driver=np.rint(driving_images[0].detach().cpu().numpy().clip(0,1)*255).astype(np.uint8)
                if driver.shape[:2]!=(target_height,target_width):raise ValueError('Driver/reference canvas size mismatch')
                canvas_geometry=detect_geometry(driver,service.engine.yolo_sess)
                driver_info={'geometry':canvas_geometry,'head_bbox':canvas_geometry['box']}
                transform=alignment_transform(result,driver_info)
                matted=np.rint(result['head_fg_rgb'].astype(np.float32)*result['head_alpha'][...,None]).astype(np.uint8)
                outputs[1]=cv2.warpAffine(matted,transform,(target_width,target_height),flags=cv2.INTER_LINEAR,borderValue=0)
                outputs[2]=cv2.warpAffine(result['head_alpha'],transform,(target_width,target_height),flags=cv2.INTER_LINEAR,borderValue=0)
            report={'engine':'v101 R4 candidate over frozen v100 photo crop','adapter':'bf348e66 archived v47 six-port formatting',
                'species':result['species'],'head_bbox':result['head_bbox'],'orig_size':[rgb.shape[1],rgb.shape[0]],
                'target_size':[target_width,target_height],'diagnostics':result.get('diagnostics',{}),
                'layout_mode':layout_mode,'neck_mode':neck_mode,'neck_pixels_removed':removed_neck,'reference_canvas_geometry':canvas_geometry,
                'head_extent_per_face_width':float((result['head_bbox'][3]-result['head_bbox'][1]+1)/max(1,reference_geometry['fw'])) if reference_geometry else None,
                'rgba_sha256':hashlib.sha256(result['head_rgba'].tobytes()).hexdigest(),
                'crop_core_sha256':hashlib.sha256((Path(__file__).parent/'crop/master_head_engine_v100.py').read_bytes()).hexdigest(),
                'note':'No automatic removal/reconstruction of hands in reference; accepted crop output is used as-is.'}
            directory=Path(folder_paths.get_output_directory())/'saochep_legacy_crop';directory.mkdir(parents=True,exist_ok=True)
            token=uuid.uuid4().hex[:12]
            for suffix,value in [('clip',outputs[0]),('reference',outputs[1]),('alpha',np.rint(outputs[2]*255).astype(np.uint8)),('rgba',result['head_rgba'])]:
                Image.fromarray(value).save(directory/f'{token}_{suffix}.png')
            report['diagnostic_prefix']='saochep_legacy_crop/'+token
            tensors=tuple(torch.from_numpy(value.astype(np.float32)/(255. if i in (0,1,3) else 1.))[None] for i,value in enumerate(outputs))
            return (*tensors,json.dumps(report,ensure_ascii=False))
        finally:
            try:
                if service is not None:service.offload()
            finally:RUN_LOCK.release()

