"""Reachable CUSTOM-NODE adapters for the supplied head-swap graph.

Algorithms are vendored under this package; graph/job lifecycle remains with the caller.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent


def _verify_crop_sources(package_root=HERE):
    manifest_path = package_root / "frozen_crop" / "frozen_crop_sha256.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        path = package_root / "frozen_crop" / relative
        if not path.is_file():
            raise RuntimeError(f"Required head-swap crop source/asset missing: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Head-swap frozen crop fingerprint mismatch: {relative}")


def _rgb(image):
    import numpy as np
    return np.rint(image.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)


def _colored(mask):
    import torch
    output = torch.ones((*mask.shape, 3), dtype=torch.float32, device=mask.device)
    output[..., :2] = 1 - mask[..., None]
    return output


def _parse_selected_face(image, face, parser, aligned_parse):
    """Parse the exact face already selected by temporal/actor tracking."""
    import numpy as np
    return aligned_parse(image, face[4:14].reshape(5, 2), face[2], face[3], parser,
                         np.array([.485, .456, .406], np.float32),
                         np.array([.229, .224, .225], np.float32))


def _parse_driver_semantics(driver_float, landmarks):
    """Bundle semantic parsing expects normalized float RGB in [0, 1]."""
    from .adaptive_semantic import parse_driving
    return parse_driving(driver_float, landmarks)


def _resize_alignment_crop(crop):
    """Upscale a detection crop without changing its aspect ratio."""
    import cv2
    scale = max(1., 512 / max(crop.shape[:2]))
    return cv2.resize(crop, None, fx=scale, fy=scale)


def _choose_face(image, hint=None, previous=None):
    import numpy as np
    from .skin.skin_node import AGDrivingSkinColor
    from .skin.aligned_parse import aligned_parse
    from .layers import QualityFailure
    detector, parser = AGDrivingSkinColor.models()
    detector.setInputSize((image.shape[1], image.shape[0]))
    _, faces = detector.detect(np.ascontiguousarray(image[..., ::-1]))
    if faces is None:
        raise QualityFailure("face_not_detected")
    def score(face):
        x,y,w,h=face[:4]
        if hint is not None:
            patch=hint[max(0,int(y)):min(len(hint),int(y+h)),max(0,int(x)):min(hint.shape[1],int(x+w))]
            return float(patch.mean()) if patch.size else 0.
        center=np.array([x+w*.5,y+h*.5])
        if previous is not None:
            old=previous[:2]+previous[2:4]*.5
            return float(np.exp(-np.linalg.norm(center-old)/max(previous[2:4]))*np.sqrt(w*h))
        frame_center=np.array([image.shape[1]*.5,image.shape[0]*.4])
        return float(np.sqrt(w*h)/(1+np.linalg.norm(center-frame_center)/max(image.shape[:2])))
    ranked=sorted(faces,key=score,reverse=True)
    if len(ranked)>1 and score(ranked[1])>=score(ranked[0])*.85:
        raise QualityFailure("ambiguous_driving_actor")
    face=ranked[0]
    if hint is not None and score(face)<.1:
        raise QualityFailure("generated_face_outside_driving_head")
    parts=aligned_parse(image,face[4:14].reshape(5,2),face[2],face[3],parser,
                        np.array([.485,.456,.406],np.float32),np.array([.229,.224,.225],np.float32))
    return face,parts


class HSCommercialResolutionCrop:
    """Reference head isolation/layout on the driving canvas (not a job runner)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "driving_images": ("IMAGE",),
            "hood_mask": ("MASK",), "hand_mask": ("MASK",),
            "hand_landmarks_json": ("STRING", {"forceInput": True})}}
    RETURN_TYPES=("IMAGE","IMAGE","MASK","IMAGE","MASK","STRING","IMAGE")
    RETURN_NAMES=("clip_rgb","reference_rgb_hq","reference_alpha_hq","tight_crop_rgb_hq",
                  "tight_crop_alpha_hq","report_json","native_crop_rgba")
    FUNCTION="crop"
    CATEGORY="SaoChep/HeadSwap"

    @classmethod
    def validate_dependencies(cls):
        """Check import/API availability without model loading, downloads or GPU use."""
        import importlib
        import mediapipe as mp
        import onnxruntime as ort
        import timm.layers as timm_layers
        import kornia.filters as kornia_filters
        import transformers
        import cv2  # noqa: F401
        import torch  # noqa: F401
        from PIL import Image  # noqa: F401
        tasks = getattr(mp, "tasks", None)
        vision = getattr(tasks, "vision", None)
        if tasks is None or vision is None:
            raise RuntimeError("MediaPipe Tasks API (mp.tasks.vision) is unavailable")
        required_mp_tasks = ("BaseOptions",)
        required_mp_vision = ("FaceLandmarker", "FaceLandmarkerOptions", "HandLandmarker",
                              "HandLandmarkerOptions", "ImageSegmenter", "ImageSegmenterOptions")
        missing_mp = [f"tasks.{name}" for name in required_mp_tasks if not hasattr(tasks, name)]
        missing_mp += [f"tasks.vision.{name}" for name in required_mp_vision
                       if not hasattr(vision, name)]
        if missing_mp:
            raise RuntimeError(f"MediaPipe Tasks API missing: {missing_mp}")
        for name in ("FaceLandmarker", "HandLandmarker", "ImageSegmenter"):
            if not callable(getattr(getattr(vision, name), "create_from_options", None)):
                raise RuntimeError(f"MediaPipe Tasks {name}.create_from_options API is unavailable")
        if not callable(getattr(getattr(cv2, "FaceDetectorYN", None), "create", None)):
            raise RuntimeError("OpenCV FaceDetectorYN.create API is unavailable")
        cv2_required = ("FaceDetectorYN", "warpAffine", "resize", "cvtColor", "distanceTransform", "fillPoly")
        missing_cv2 = [name for name in cv2_required if not hasattr(cv2, name)]
        if missing_cv2:
            raise RuntimeError(f"OpenCV API missing: {missing_cv2}")
        for module, names, label in (
            (ort, ("SessionOptions", "InferenceSession"), "ONNX Runtime"),
            (timm_layers, ("DropPath", "to_2tuple", "trunc_normal_"), "timm.layers"),
            (kornia_filters, ("laplacian",), "kornia.filters"),
            (transformers, ("AutoModelForImageSegmentation", "VitMatteImageProcessor", "VitMatteForImageMatting"), "Transformers"),
        ):
            absent = [name for name in names if not hasattr(module, name)]
            if absent:
                raise RuntimeError(f"{label} API missing: {absent}")
        for name in ("AutoModelForImageSegmentation", "VitMatteImageProcessor", "VitMatteForImageMatting"):
            if not callable(getattr(getattr(transformers, name), "from_pretrained", None)):
                raise RuntimeError(f"Transformers {name}.from_pretrained API is unavailable")
        local_model = importlib.import_module(
            f"{__package__}.frozen_crop.models.birefnet_lite.birefnet")
        if not hasattr(local_model, "BiRefNet"):
            raise RuntimeError("Vendored BiRefNet implementation is not importable")
        crop_engine = importlib.import_module(
            f"{__package__}.frozen_crop.crop.master_head_engine_v100")
        importlib.import_module(f"{__package__}.frozen_crop.crop_pipeline_v101")
        if not hasattr(crop_engine, "MasterHeadEngineV100"):
            raise RuntimeError("Vendored crop engine is not importable")
        return {"mediapipe_tasks": required_mp_tasks + required_mp_vision, "onnxruntime": ("SessionOptions", "InferenceSession"),
                "transformers": ("AutoModelForImageSegmentation", "VitMatteImageProcessor", "VitMatteForImageMatting"),
                "timm": ("DropPath", "to_2tuple", "trunc_normal_"), "kornia": ("laplacian",),
                "local_birefnet": "imported; model weights not loaded",
                "local_crop_engine": "imported; model weights not loaded"}

    def crop(self,image,driving_images,hood_mask,hand_mask,hand_landmarks_json):
        if image.ndim!=4 or image.shape[0]!=1 or image.shape[-1]!=3: raise ValueError("Expected exactly one RGB reference image")
        if driving_images.ndim!=4 or not len(driving_images): raise ValueError("Expected non-empty driving IMAGE sequence")
        height,width=driving_images.shape[1:3]
        _verify_crop_sources()
        from .frozen_crop.adapter import SaoChepCropHeadHandSafe
        output=SaoChepCropHeadHandSafe().isolate(image=image,hood_mask=hood_mask,hand_mask=hand_mask,
            hand_landmarks_json=hand_landmarks_json,driving_images=driving_images[:1],layout_mode="driving_head",
            clip_bg_mode="black",target_width=int(width),target_height=int(height),matting_device="auto",
            max_pixels=6_000_000,min_side=64,subject="human",save_diagnostics=False)
        report=json.loads(output[5]); report["release"]="SaoChepCommercial-headswap-adapter-v1"
        report["frozen_crop_manifest_sha256"]=hashlib.sha256((HERE/"frozen_crop"/"frozen_crop_sha256.json").read_bytes()).hexdigest()
        return (*output[:5],json.dumps(report),output[6])


class HSFinalReferenceMask:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"mask":("MASK",)}}
    RETURN_TYPES=("IMAGE",); FUNCTION="make"; CATEGORY="SaoChep/HeadSwap"
    def make(self,mask):
        import torch
        result=torch.zeros((*mask.shape,3),dtype=torch.float32,device=mask.device)
        result[...,2]=(mask>.5).to(torch.float32)
        return (result,)


class HSFinal20261109HandGuides:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"image":("IMAGE",),"max_pixels":("INT",{"default":6000000,"min":65536,"max":24000000})}}
    RETURN_TYPES=("BOUNDING_BOX","STRING"); RETURN_NAMES=("hand_boxes","hand_landmarks_json")
    FUNCTION="locate"; CATEGORY="SaoChep/HeadSwap"
    def locate(self,image,max_pixels=6_000_000):
        from .frozen_crop.hand_cleanup import SaoChepHandGuides
        if image.ndim!=4 or image.shape[0]!=1: raise ValueError("Hand guide input requires one reference image")
        return SaoChepHandGuides().locate(image=image,max_pixels=max_pixels)


class HSFirstFrameAnchorTrial:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"images":("IMAGE",),"geometry":("STRING",)}}
    RETURN_TYPES=("IMAGE","STRING","STRING"); FUNCTION="select"; CATEGORY="SaoChep/HeadSwap"
    def select(self,images,geometry):
        import cv2, mediapipe as mp, numpy as np
        from .skin.skin_node import AGDrivingSkinColor
        from .selection import measure_frame,choose_frame
        from .frozen_crop.paths import asset_path
        records=json.loads(geometry)
        if len(records)!=len(images) or not len(images): raise ValueError("Geometry timing mismatch or empty driving sequence")
        detector,_=AGDrivingSkinColor.models()
        options=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(asset_path("hand_landmarker.task"))),num_hands=2,min_hand_detection_confidence=.5)
        metrics=[]
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as hands:
            for tensor,record in zip(images,records):
                rgb=_rgb(tensor); detector.setInputSize((rgb.shape[1],rgb.shape[0])); _,faces=detector.detect(np.ascontiguousarray(rgb[...,::-1]))
                if faces is None:
                    metrics.append({'box':record['face_box'],'yaw_proxy':0.,'roll':0.,'pitch_proxy':0.,'hand_overlap':1.,'sharpness':0.,'repaired':False,'excluded':True}); continue
                box=np.asarray(record['face_box']); face=min(faces,key=lambda value:float(np.linalg.norm(value[:4]-box)))
                scale=min(1.,640/max(rgb.shape[:2])); small=cv2.resize(rgb,None,fx=scale,fy=scale)
                found=hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(small)))
                polygons=[np.array([[p.x*rgb.shape[1],p.y*rgb.shape[0]] for p in hand]) for hand in found.hand_landmarks]
                metrics.append(measure_frame(rgb,face,polygons,record.get('temporal_repaired',False)))
        index,report=choose_frame(metrics); report['hand_detection_is_proxy']=True
        return images[index:index+1],json.dumps(report),json.dumps([records[index]])


class HSStableAlignmentFrame:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"images":("IMAGE",),"geometry":("STRING",)}}
    RETURN_TYPES=("IMAGE","STRING","STRING"); FUNCTION="select"; CATEGORY="SaoChep/HeadSwap"
    def select(self,images,geometry):
        import cv2,numpy as np
        from .skin.skin_node import AGDrivingSkinColor
        from .stable_anchor import choose_index
        records=json.loads(geometry)
        if len(records)!=len(images): raise ValueError("Geometry timing mismatch")
        detector,_=AGDrivingSkinColor.models(); boxes=[]; frontal=[]; sharpness=[]; repaired=[]
        for tensor,record in zip(images,records):
            rgb=_rgb(tensor); box=np.asarray(record['face_box'],float); boxes.append(box); repaired.append(record.get('temporal_repaired',False))
            detector.setInputSize((rgb.shape[1],rgb.shape[0])); _,candidates=detector.detect(np.ascontiguousarray(rgb[...,::-1]))
            if candidates is None: frontal.append(np.nan); sharpness.append(np.nan); continue
            face=min(candidates,key=lambda f:float(np.linalg.norm(f[:4]-box))); eyes=face[4:8].reshape(2,2)
            span=max(float(np.linalg.norm(eyes[1]-eyes[0])),np.finfo(float).eps); frontal.append(float(np.linalg.norm(face[8:10]-eyes.mean(0))/span))
            x,y,w,h=box; crop=rgb[max(0,int(y)):min(len(rgb),int(y+h)),max(0,int(x)):min(rgb.shape[1],int(x+w))]
            sharpness.append(float(cv2.Laplacian(cv2.cvtColor(cv2.resize(crop,(128,128)),cv2.COLOR_RGB2GRAY),cv2.CV_64F).var()) if crop.size else np.nan)
        index=choose_index(boxes,frontal,sharpness,repaired)
        return images[index:index+1],json.dumps({'selected_frame':index,'face_box':records[index]['face_box'],'semantic_occlusion_verified':False}),json.dumps([records[index]])


class HSAdaptiveChinEyeR2:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"reference":("IMAGE",),"alpha":("MASK",),"driving":("IMAGE",),"confidence":("FLOAT",{"default":.5,"min":.1,"max":.99}),"min_neck_fraction":("FLOAT",{"default":.1,"min":.01,"max":1.})},"optional":{"driving_geometry":("STRING",{"forceInput":True})}}
    RETURN_TYPES=("IMAGE","MASK","STRING"); FUNCTION="align"; CATEGORY="SaoChep/HeadSwap"
    def align(self,reference,alpha,driving,confidence=.5,min_neck_fraction=.1,driving_geometry=None):
        import cv2,mediapipe as mp,numpy as np,torch
        from .routing import select_translation
        if len(reference)!=1 or len(alpha)!=1 or len(driving)!=1: raise ValueError("One reference, alpha, and driving anchor required")
        ref=reference[0].detach().float().cpu().numpy(); mask=alpha[0].detach().float().cpu().numpy(); drv=driving[0].detach().float().cpu().numpy()
        if ref.shape[:2]!=drv.shape[:2] or mask.shape!=ref.shape[:2]: raise ValueError("Reference/mask/driver canvas mismatch")
        from .frozen_crop.paths import asset_path
        options=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(asset_path("face_landmarker.task"))),num_faces=2,min_face_detection_confidence=.5)
        h,w=mask.shape
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            def detect(rgb,bounds):
                x0,y0,x1,y1=bounds; crop=np.rint(np.clip(rgb[y0:y1,x0:x1],0,1)*255).astype(np.uint8)
                if crop.size==0: raise ValueError("Empty face ROI")
                crop=_resize_alignment_crop(crop)
                result=landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(crop)))
                if len(result.face_landmarks)!=1: raise ValueError("Reliable single face required for chin alignment")
                return np.array([[p.x*(x1-x0)+x0,p.y*(y1-y0)+y0] for p in result.face_landmarks[0]],np.float32)
            yy,xx=np.where(mask>.5)
            if not len(xx): raise ValueError("Empty reference alpha")
            ref_points=detect(ref,(int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1)); method="full_frame"
            if driving_geometry:
                x,y,fw,fh=json.loads(driving_geometry)[0]['face_box']; bounds=(max(0,int(x-fw*.5)),max(0,int(y-fh*.5)),min(w,int(x+fw*1.5)),min(h,int(y+fh*1.5)))
                try: drv_points=detect(drv,bounds); method="tracked_face_roi"
                except ValueError: drv_points=detect(drv,(0,0,w,h))
            else: drv_points=detect(drv,(0,0,w,h))
        neck,face=_parse_driver_semantics(drv,drv_points)
        offset,route=select_translation(ref_points,drv_points,neck,face,confidence=confidence,min_neck_fraction=min_neck_fraction)
        matrix=np.array([[1,0,offset[0]],[0,1,offset[1]]],np.float32)
        return torch.from_numpy(cv2.warpAffine(ref,matrix,(w,h),flags=cv2.INTER_LINEAR))[None].to(reference.device),torch.from_numpy(cv2.warpAffine(mask,matrix,(w,h),flags=cv2.INTER_LINEAR))[None].to(alpha.device),json.dumps({**route,'driving_detection':method,'mode':'translation_only','scale_correction':1.0,'translation':offset.tolist(),'occlusion_verified':False})


class HSBackfaceHeadMaskTrial:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"images":("IMAGE",),"track_data":("SAM3_TRACK_DATA",)}}
    RETURN_TYPES=("MASK","IMAGE","MASK","MASK","MASK","MASK","MASK","STRING")
    RETURN_NAMES=("head","pose_mask","face_core","forbidden","headwear","skin","neck","geometry")
    FUNCTION="assemble"; CATEGORY="SaoChep/HeadSwap"
    def assemble(self,images,track_data,actor_hint=None):
        import cv2,numpy as np,torch
        from .tracking import plausible,continuity,repair_gaps
        from .layers import QualityFailure
        from .skin.skin_node import AGDrivingSkinColor
        from .skin.aligned_parse import aligned_parse
        from .layers import face_ownership,component_family
        detector,parser=AGDrivingSkinColor.models()
        faces=[]; last=None
        for tensor in images:
            rgb=_rgb(tensor); detector.setInputSize((rgb.shape[1],rgb.shape[0])); _,candidates=detector.detect(np.ascontiguousarray(rgb[...,::-1])); chosen=None
            if candidates is not None:
                if last is None:
                    try: chosen,_=_choose_face(rgb)
                    except QualityFailure: pass
                else:
                    valid=[f for f in candidates if plausible(last,f)]
                    if valid: chosen=min(valid,key=lambda f:continuity(last,f))
                    else:
                        try: chosen,_=_choose_face(rgb)
                        except QualityFailure: pass
            faces.append(chosen)
            if chosen is not None: last=chosen
        # Match the source algorithm: discard isolated scale outliers while
        # retaining a sustained camera/actor scale transition.
        for index in range(1, len(faces)-1):
            before, current, after = faces[index-1:index+2]
            if (before is not None and current is not None and after is not None
                    and plausible(before, after) and not plausible(before, current)
                    and not plausible(current, after)):
                faces[index] = None
        tail=len(faces)
        while tail and faces[tail-1] is None: tail-=1
        if not tail: raise QualityFailure('no_confirmed_face_for_actor_association')
        faces,repaired=repair_gaps(faces[:tail],max_gap=3); faces+= [None]*(len(images)-tail)
        from comfy.ldm.sam3.tracker import unpack_masks
        packed=track_data.get('packed_masks')
        if packed is None or len(packed)!=len(images): raise QualityFailure('head_parts_timing_mismatch')
        h,w=images.shape[1:3]; heads=[];cores=[];forbidden=[];items=[];skins=[];necks=[];geometries=[];owned=[];last_head=None
        for i,tensor in enumerate(images):
            rgb=_rgb(tensor); face=faces[i]
            if face is None:
                if last_head is None or not owned: raise QualityFailure('no_confirmed_sam_head_ownership')
                raw=unpack_masks(packed[i:i+1].cpu())[0].numpy(); candidates=[cv2.resize(raw[j].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0 for j in owned if j<len(raw)]
                if not candidates: raise QualityFailure('sam_actor_track_missing')
                head=np.logical_or.reduce(candidates); from .sam_guard import validate_transition; validate_transition(last_head,head)
                empty=np.zeros((h,w),np.float32); heads.append(head.astype(np.float32))
                for target in (cores,forbidden,items,skins,necks): target.append(empty.copy())
                geometries.append({'face_box':faces[tail-1][:4].astype(float).tolist(),'face_visible':False,'temporal_repaired':True,'tracking_source':'sam_confirmed_actor','mouth_y':None,'hat_pixels':0,'glasses_pixels':0}); last_head=head; continue
            parsed=_parse_selected_face(rgb,face,parser,aligned_parse)
            labels=parsed['labels']; core,veto=face_ownership(labels,parsed['hair'],parsed['hat'],face[:4])
            if core.sum()<32: raise QualityFailure(f'face_parsing_failed:{i}')
            raw=unpack_masks(packed[i:i+1].cpu())[0].numpy(); parts=[cv2.resize(p.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST) for p in raw]
            parts.extend([(parsed['hair']>.5).astype(np.uint8),(parsed['hat']>.5).astype(np.uint8)])
            head=component_family(core,parts,min(face[2:4])); head[veto]=False;head[core]=True
            owned=[j for j,p in enumerate(parts[:len(raw)]) if np.count_nonzero((p>0)&head)/max(1,np.count_nonzero(p))>.5 and np.count_nonzero((p>0)&head)/max(1,np.count_nonzero(head))>.01]
            last_head=head; heads.append(head.astype(np.float32));cores.append(core.astype(np.float32));forbidden.append(veto.astype(np.float32))
            items.append((np.isin(labels,[6,18])&head).astype(np.float32));skins.append(np.isin(labels,[1,14]).astype(np.float32));necks.append((labels==14).astype(np.float32))
            geometries.append({'face_box':face[:4].astype(float).tolist(),'temporal_repaired':i in repaired,'mouth_y':float(face[10:14].reshape(2,2)[:,1].mean()),'hat_pixels':int(np.count_nonzero((labels==18)&head)),'glasses_pixels':int(np.count_nonzero((labels==6)&head))})
        mask=torch.from_numpy(np.stack(heads)); return mask,_colored(mask),torch.from_numpy(np.stack(cores)),torch.from_numpy(np.stack(forbidden)),torch.from_numpy(np.stack(items)),torch.from_numpy(np.stack(skins)),torch.from_numpy(np.stack(necks)),json.dumps(geometries)


class HSFrameSequenceFit:
    """Align conditioning IMAGE frames while preserving requested output duration.

    SCAIL receives the original requested frame count separately. Conditioning is
    padded/truncated to its smallest valid 4n+1 sequence; padding is capped at
    four repeated tail frames. The original source/audio stream is untouched.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",),
            "target_frames": ("INT", {"default": 72, "min": 1})}}
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "aligned_frames")
    FUNCTION = "fit"
    CATEGORY = "SaoChep/HeadSwap"

    def fit(self, images, target_frames):
        import torch
        if getattr(images, "ndim", None) != 4 or len(images) == 0 or min(images.shape[1:]) <= 0:
            raise ValueError("Expected non-empty IMAGE sequence [frames,height,width,channels]")
        if type(target_frames) is not int or target_frames < 1:
            raise ValueError("target_frames must be a positive requested SCAIL frame count")
        aligned_frames = max(5, target_frames + ((1 - target_frames) % 4))
        if not bool(torch.isfinite(images).all()):
            raise ValueError("IMAGE sequence contains non-finite values")
        if len(images) < aligned_frames:
            padding_count = aligned_frames - len(images)
            if padding_count > 4:
                raise ValueError(
                    f"Source conditioning is {padding_count} frames short; maximum tail padding is four")
            padding = images[-1:].expand((padding_count,) + tuple(images.shape[1:]))
            images = torch.cat((images, padding), dim=0)
        else:
            images = images[:aligned_frames]
        return images, aligned_frames


class HSFinal20261109Contracts:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"driving_images": ("IMAGE",), "pose_mask": ("IMAGE",),
            "reference_image": ("IMAGE",), "reference_alpha": ("MASK",),
            "reference_mask": ("IMAGE",), "crop_report": ("STRING",), "policy_report": ("STRING",)}}
    RETURN_TYPES=("IMAGE","IMAGE","IMAGE","STRING")
    FUNCTION="check"; CATEGORY="SaoChep/HeadSwap"
    def check(self,driving_images,pose_mask,reference_image,reference_alpha,reference_mask,crop_report,policy_report):
        import torch
        if driving_images.shape!=pose_mask.shape or len(driving_images)<5 or (len(driving_images)-1)%4:
            raise ValueError("Driving/mask timing or 4*n+1 geometry mismatch")
        if reference_image.shape!=reference_mask.shape or reference_alpha.shape!=reference_image.shape[:3]:
            raise ValueError("Reference/mask canvas mismatch")
        if tuple(reference_image.shape[1:3])!=tuple(driving_images.shape[1:3]): raise ValueError("Driving/reference dimensions differ")
        for tensor in (driving_images,pose_mask,reference_image,reference_alpha,reference_mask):
            if not bool(torch.isfinite(tensor).all()): raise ValueError("Non-finite workflow input")
        expected=reference_alpha>.5
        actual=(reference_mask[...,2]>.9)&(reference_mask[...,:2].max(dim=-1).values<.1)
        if not torch.equal(actual.cpu(),expected.cpu()): raise ValueError("Reference identity mask does not match crop alpha")
        head_pixels=(pose_mask[...,2]>.9)&(pose_mask[...,:2].max(dim=-1).values<.1)
        areas=head_pixels.float().mean(dim=(1,2))
        if bool((areas<=0).any()) or bool((areas>.45).any()): raise ValueError("Empty or excessive head mask")
        report={"release":"SaoChepCommercial-headswap-adapter-v1","frames":len(driving_images),
            "mask_area_min":float(areas.min()),"mask_area_max":float(areas.max()),
            "crop":json.loads(crop_report),"policy":json.loads(policy_report),
            "quality_status":"visual_regression_pending"}
        return driving_images,pose_mask,reference_image,json.dumps(report)


# Every new registry ID is explicitly namespaced to avoid collisions with bundle/upstream nodes.
NODE_CLASS_MAPPINGS={
    "SaoChepHeadSwapCrop":HSCommercialResolutionCrop,
    "SaoChepHeadSwapReferenceMask":HSFinalReferenceMask,
    "SaoChepHeadSwapHandGuides":HSFinal20261109HandGuides,
    "SaoChepHeadSwapAnchor":HSFirstFrameAnchorTrial,
    "SaoChepHeadSwapStableAnchor":HSStableAlignmentFrame,
    "SaoChepHeadSwapChinEyeAlign":HSAdaptiveChinEyeR2,
    "SaoChepHeadSwapBackfaceMask":HSBackfaceHeadMaskTrial,
    "SaoChepHeadSwapContracts":HSFinal20261109Contracts,
    "SaoChepHeadSwapFrameFit":HSFrameSequenceFit,
}
