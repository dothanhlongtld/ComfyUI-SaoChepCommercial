"""Reachable CUSTOM-NODE adapters for the supplied head-swap graph.

Algorithms are vendored under this package; graph/job lifecycle remains with the caller.
"""
import hashlib
import json
import math
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


def _alignment_landmarks(rgb, box, yunet_face, landmarker, mp):
    """Measure one face; try bounded crop sizes only around the tracked face."""
    import numpy as np
    from .routing import eye_centers

    if yunet_face is None:
        return None
    height, width = rgb.shape[:2]
    x, y, fw, fh = box
    expected = np.asarray(yunet_face[4:14], dtype=np.float32).reshape(5, 2).copy()
    expected[:2] = expected[:2][np.argsort(expected[:2, 0])]
    expected[3:] = expected[3:][np.argsort(expected[3:, 0])]
    axis_scale = np.array([fw, fh], dtype=np.float32)
    for pad in (.5, None, .25, 1.):
        x0, y0, x1, y1 = (0, 0, width, height) if pad is None else (
            max(0, int(x - fw * pad)), max(0, int(y - fh * pad)),
            min(width, int(x + fw * (1 + pad))), min(height, int(y + fh * (1 + pad))),
        )
        crop = rgb[y0:y1, x0:x1]
        if not crop.size:
            continue
        crop = _resize_alignment_crop(crop)
        faces = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                          data=np.ascontiguousarray(crop))).face_landmarks
        if len(faces) != 1 or len(faces[0]) < 363:
            continue
        points = np.array([[p.x * (x1 - x0) + x0, p.y * (y1 - y0) + y0]
                           for p in faces[0]], dtype=np.float32)
        if not np.isfinite(points).all():
            continue
        eyes = eye_centers(points)
        mouth = points[[61, 291]]
        observed = np.vstack((eyes[np.argsort(eyes[:, 0])], points[1], mouth[np.argsort(mouth[:, 0])]))
        residual = np.linalg.norm((observed - expected) / axis_scale, axis=1)
        if (np.linalg.norm(eyes[0] - eyes[1]) > np.finfo(np.float32).eps
                and np.all(eyes >= [x, y]) and np.all(eyes < [x + fw, y + fh])
                and np.max(residual) <= .25 and np.sqrt(np.mean(residual ** 2)) <= .16):
            return points, 'full_frame' if pad is None else 'tracked_face_roi'
    return None


def _balance_aligned_chin(reference_points, driving_points, offset):
    """Bound the eye/chin compromise along the face axis without scaling the head."""
    import numpy as np
    from .routing import eye_centers

    reference_eyes, driving_eyes = eye_centers(reference_points), eye_centers(driving_points)
    reference_delta = reference_eyes[1] - reference_eyes[0]
    driving_delta = driving_eyes[1] - driving_eyes[0]
    span = float(np.linalg.norm(driving_delta))
    reference_span = float(np.linalg.norm(reference_delta))
    if not np.isfinite(span) or span <= 0 or not .7 * span <= reference_span <= 1.4 * span:
        return offset, 0.
    vertical = np.array([-driving_delta[1], driving_delta[0]]) / span
    if vertical[1] < 0:
        vertical = -vertical
    depth = float((driving_points[152] - driving_eyes.mean(0)) @ vertical)
    reference_depth = float((reference_points[152] - reference_eyes.mean(0)) @ vertical)
    # Avoid compensating for different roll, strong foreshortening, or a bad mesh.
    if (abs(float(reference_delta @ vertical)) > .25 * reference_span
            or not .5 * span <= depth <= 3 * span
            or not .7 * depth <= reference_depth <= 1.4 * depth):
        return offset, 0.
    residual = reference_depth - depth
    # Leave small differences alone and cap the movement of the confirmed chin.
    excess = max(0., abs(residual) - .04 * depth)
    adjustment = float(np.copysign(min(.5 * excess, .08 * depth), residual))
    return offset + vertical * adjustment, adjustment


def _alignment_face_options(mp, asset_path):
    return mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(asset_path('face_landmarker.task'))),
        num_faces=2, min_face_detection_confidence=.5,
    )


def _validated_actor_mask(actor_mask, images_shape):
    import numpy as np
    if (getattr(actor_mask, "ndim", None) != 3 or actor_mask.shape[0] != images_shape[0]
            or actor_mask.shape[1:3] != images_shape[1:3]):
        raise ValueError("Actor mask timing or geometry mismatch")
    values = actor_mask.detach().cpu().numpy() if hasattr(actor_mask, "detach") else np.asarray(actor_mask)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Actor mask values must be finite and in [0,1]")
    return values


def _filter_faces_by_mask(faces, mask):
    import numpy as np
    selected = []
    height, width = mask.shape
    for face in faces if faces is not None else ():
        x, y, face_width, face_height = map(float, face[:4])
        x0, y0 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
        x1, y1 = min(width, int(np.ceil(x + face_width))), min(height, int(np.ceil(y + face_height)))
        if x1 <= x0 or y1 <= y0:
            continue
        patch = mask[y0:y1, x0:x1]
        if patch.size and np.count_nonzero(patch > .1) / patch.size >= .5:
            selected.append(face)
    return selected


def _rank_face_candidates(image, faces, hint=None, previous=None):
    import numpy as np
    from .layers import QualityFailure
    if faces is None or len(faces) == 0:
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
    return face


def _choose_face(image, hint=None, previous=None):
    import numpy as np
    from .skin.skin_node import AGDrivingSkinColor
    from .skin.aligned_parse import aligned_parse
    detector, parser = AGDrivingSkinColor.models()
    detector.setInputSize((image.shape[1], image.shape[0]))
    _, faces = detector.detect(np.ascontiguousarray(image[..., ::-1]))
    face = _rank_face_candidates(image, faces, hint=hint, previous=previous)
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


def _observed_face_geometry(record):
    import numpy as np
    if not isinstance(record, dict) or record.get("face_visible") is False:
        return None
    try:
        box = np.asarray(record.get("face_box"), dtype=float)
    except (TypeError, ValueError):
        return None
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:4] <= 0):
        return None
    vector = record.get("yunet_face")
    if vector is not None:
        try:
            vector = np.asarray(vector, dtype=float)
        except (TypeError, ValueError):
            return None
        if vector.shape != (15,) or not np.isfinite(vector).all() or not np.allclose(vector[:4], box, rtol=0, atol=1e-4):
            return None
    return box, vector


class HSFirstFrameAnchorTrial:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"images":("IMAGE",),"geometry":("STRING",)}}
    RETURN_TYPES=("IMAGE","STRING","STRING"); FUNCTION="select"; CATEGORY="SaoChep/HeadSwap"
    def select(self,images,geometry):
        import cv2, mediapipe as mp, numpy as np
        from .skin.skin_node import AGDrivingSkinColor
        from .selection import measure_frame,choose_frame,choose_ranked_frame
        from .frozen_crop.paths import asset_path
        records=json.loads(geometry)
        if len(records)!=len(images) or not len(images): raise ValueError("Geometry timing mismatch or empty driving sequence")
        detector=None; metrics=[]; frame_indices=[]
        options=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(asset_path("hand_landmarker.task"))),num_hands=2,min_hand_detection_confidence=.5)
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as hands:
            for frame_index,(tensor,record) in enumerate(zip(images,records)):
                observed=_observed_face_geometry(record)
                if observed is None: continue
                box,face_vector=observed
                rgb=_rgb(tensor)
                if face_vector is not None:
                    face=face_vector
                else:
                    if detector is None: detector,_=AGDrivingSkinColor.models()
                    detector.setInputSize((rgb.shape[1],rgb.shape[0])); _,faces=detector.detect(np.ascontiguousarray(rgb[...,::-1]))
                    if faces is None: continue
                    face=min(faces,key=lambda value:float(np.linalg.norm(value[:4]-box)))
                    if face.shape != (15,) or not np.isfinite(face).all(): continue
                scale=min(1.,640/max(rgb.shape[:2])); small=cv2.resize(rgb,None,fx=scale,fy=scale)
                found=hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(small)))
                polygons=[np.array([[p.x*rgb.shape[1],p.y*rgb.shape[0]] for p in hand]) for hand in found.hand_landmarks]
                try:
                    metric=measure_frame(rgb,face,polygons,record.get('temporal_repaired',False))
                except ValueError:
                    continue
                metrics.append(metric); frame_indices.append(frame_index)
        if not metrics: raise ValueError("No valid observed face geometry")
        if frame_indices[0] == 0:
            index,report=choose_frame(metrics)
        else:
            index,report=choose_ranked_frame(metrics)
            report.update(selection="relative_quality_rank",fallback_ranked_frame=index)
        if records[frame_indices[index]].get('tracking_source') == 'sam_confirmed_actor':
            # Keep the current candidate if chin/eye landmarks are available. A
            # profile can lose the mesh entirely, even on a credible YuNet face.
            eligible = [i for i, metric in enumerate(metrics) if not metric['repaired']
                        and records[frame_indices[i]].get('tracking_source') == 'sam_confirmed_actor'
                        and metric['hand_overlap'] <= max(.02, metrics[index]['hand_overlap'])]
            spread = np.linspace(0, len(eligible) - 1, min(6, len(eligible))).round().astype(int)
            order = dict.fromkeys([index, report.get('fallback_ranked_frame', index),
                                   *(eligible[i] for i in spread)])
            with mp.tasks.vision.FaceLandmarker.create_from_options(_alignment_face_options(mp, asset_path)) as mesh:
                for candidate in order:
                    if candidate not in eligible:
                        continue
                    frame = frame_indices[candidate]
                    box, face = _observed_face_geometry(records[frame])
                    if _alignment_landmarks(_rgb(images[frame]), box, face, mesh, mp) is not None:
                        if candidate != index:
                            index = candidate
                            report.update(selection='tracked_mesh_retry', selected_metrics=metrics[index])
                        break
        original_index=frame_indices[index]
        report["selected_frame"]=original_index
        if "fallback_ranked_frame" in report:
            report["fallback_ranked_frame"]=frame_indices[report["fallback_ranked_frame"]]
        report["candidate_frame_indices"]=frame_indices
        report['hand_detection_is_proxy']=True
        return images[original_index:original_index+1],json.dumps(report),json.dumps([records[original_index]])


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
        detector=None; boxes=[]; frontal=[]; sharpness=[]; repaired=[]; frame_indices=[]
        for frame_index,(tensor,record) in enumerate(zip(images,records)):
            observed=_observed_face_geometry(record)
            if observed is None: continue
            box,face_vector=observed; rgb=_rgb(tensor)
            if face_vector is not None:
                face=face_vector
            else:
                if detector is None: detector,_=AGDrivingSkinColor.models()
                detector.setInputSize((rgb.shape[1],rgb.shape[0])); _,candidates=detector.detect(np.ascontiguousarray(rgb[...,::-1]))
                if candidates is None: continue
                face=min(candidates,key=lambda f:float(np.linalg.norm(f[:4]-box)))
                if face.shape != (15,) or not np.isfinite(face).all(): continue
            eyes=face[4:8].reshape(2,2)
            span=max(float(np.linalg.norm(eyes[1]-eyes[0])),np.finfo(float).eps)
            frontal.append(float(np.linalg.norm(face[8:10]-eyes.mean(0))/span))
            x,y,fw,fh=box; crop=rgb[max(0,int(y)):min(len(rgb),int(y+fh)),max(0,int(x)):min(rgb.shape[1],int(x+fw))]
            if not crop.size:
                frontal.pop(); continue
            sharpness.append(float(cv2.Laplacian(cv2.cvtColor(cv2.resize(crop,(128,128)),cv2.COLOR_RGB2GRAY),cv2.CV_64F).var()))
            boxes.append(box); repaired.append(record.get('temporal_repaired',False)); frame_indices.append(frame_index)
        if not boxes: raise ValueError("No valid observed face geometry")
        index=choose_index(boxes,frontal,sharpness,repaired)
        original_index=frame_indices[index]
        report={'selected_frame':original_index,'face_box':records[original_index]['face_box'],'semantic_occlusion_verified':False,
                'candidate_frame_indices':frame_indices}
        return images[original_index:original_index+1],json.dumps(report),json.dumps([records[original_index]])


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
        actor_alignment = False
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
                record = json.loads(driving_geometry)[0]
                x,y,fw,fh=record['face_box']
                actor_alignment = record.get('tracking_source') == 'sam_confirmed_actor'
                if actor_alignment:
                    observed = _observed_face_geometry(record)
                    mesh = (_alignment_landmarks(_rgb(driving[0]), *observed, landmarker, mp)
                            if observed is not None else None)
                    if mesh is None:
                        raise ValueError('Reliable single face required for chin alignment')
                    drv_points, method = mesh
                else:
                    bounds=(max(0,int(x-fw*.5)),max(0,int(y-fh*.5)),min(w,int(x+fw*1.5)),min(h,int(y+fh*1.5)))
                    try: drv_points=detect(drv,bounds); method="tracked_face_roi"
                    except ValueError: drv_points=detect(drv,(0,0,w,h))
            else: drv_points=detect(drv,(0,0,w,h))
        neck,face=_parse_driver_semantics(drv,drv_points)
        offset,route=select_translation(ref_points,drv_points,neck,face,confidence=confidence,min_neck_fraction=min_neck_fraction)
        if actor_alignment and route['branch'] == 'chin':
            offset, adjustment = _balance_aligned_chin(ref_points, drv_points, offset)
            route['chin_balance_adjustment'] = adjustment
        matrix=np.array([[1,0,offset[0]],[0,1,offset[1]]],np.float32)
        return torch.from_numpy(cv2.warpAffine(ref,matrix,(w,h),flags=cv2.INTER_LINEAR))[None].to(reference.device),torch.from_numpy(cv2.warpAffine(mask,matrix,(w,h),flags=cv2.INTER_LINEAR))[None].to(alpha.device),json.dumps({**route,'driving_detection':method,'mode':'translation_only','scale_correction':1.0,'translation':offset.tolist(),'occlusion_verified':False})


class HSBackfaceHeadMaskTrial:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"images":("IMAGE",),"track_data":("SAM3_TRACK_DATA",)},"optional":{"actor_mask":("MASK",)}}
    RETURN_TYPES=("MASK","IMAGE","MASK","MASK","MASK","MASK","MASK","STRING")
    RETURN_NAMES=("head","pose_mask","face_core","forbidden","headwear","skin","neck","geometry")
    FUNCTION="assemble"; CATEGORY="SaoChep/HeadSwap"
    def assemble(self,images,track_data,actor_mask=None):
        import cv2,numpy as np,torch
        from .tracking import plausible,continuity,preserve_confirmed_gaps
        from .layers import QualityFailure
        from .skin.skin_node import AGDrivingSkinColor
        from .skin.aligned_parse import aligned_parse
        from .layers import face_ownership,component_family
        actor_mask_np = _validated_actor_mask(actor_mask, images.shape) if actor_mask is not None else None
        detector,parser=AGDrivingSkinColor.models()
        if actor_mask_np is not None:
            packed=track_data.get('packed_masks')
            if packed is None or len(packed)!=len(images):
                raise QualityFailure('actor_mask_sam_timing_mismatch')
            return self._assemble_with_actor_mask(images, actor_mask_np, packed, detector, parser, cv2, np, torch,
                                                  QualityFailure, face_ownership, component_family, aligned_parse)
        faces=[]; last=None
        for frame_index, tensor in enumerate(images):
            rgb=_rgb(tensor); detector.setInputSize((rgb.shape[1],rgb.shape[0])); _,candidates=detector.detect(np.ascontiguousarray(rgb[...,::-1])); chosen=None
            if candidates is not None:
                if actor_mask is not None:
                    hint=actor_mask_np[frame_index]
                    candidates=_filter_faces_by_mask(candidates, hint)
                if last is None:
                    try:
                        chosen = _rank_face_candidates(rgb, candidates, hint=hint if actor_mask is not None else None)
                    except QualityFailure: pass
                else:
                    valid=[f for f in candidates if plausible(last,f)]
                    if valid: chosen=min(valid,key=lambda f:continuity(last,f))
                    # No continuity match means absence/occlusion or actor
                    # ambiguity. Never reset association to a newly ranked face.
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
        try:
            faces,_=preserve_confirmed_gaps(faces[:tail],max_gap=3)
        except ValueError as error:
            raise QualityFailure(f'face_gap_unconfirmed:{error}') from error
        # Interpolated boxes validate the gap endpoints only; never treat them as
        # observed landmarks or parse a fabricated face. SAM ownership below is
        # the only permitted source for these missing-face frames.
        faces+= [None]*(len(images)-tail)
        from comfy.ldm.sam3.tracker import unpack_masks
        packed=track_data.get('packed_masks')
        if packed is None or len(packed)!=len(images): raise QualityFailure('head_parts_timing_mismatch')
        h,w=images.shape[1:3]; frame_count=len(images)
        heads=np.empty((frame_count,h,w),np.float32);cores=np.empty_like(heads);forbidden=np.empty_like(heads)
        items=np.empty_like(heads);skins=np.empty_like(heads);necks=np.empty_like(heads)
        geometries=[];owned=[];last_head=None;last_confirmed_face=None
        for i,tensor in enumerate(images):
            rgb=_rgb(tensor); face=faces[i]
            if face is None:
                if last_head is None or not owned: raise QualityFailure('no_confirmed_sam_head_ownership')
                raw=unpack_masks(packed[i:i+1].cpu())[0].numpy(); candidates=[cv2.resize(raw[j].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0 for j in owned if j<len(raw)]
                if not candidates: raise QualityFailure('sam_actor_track_missing')
                head=np.logical_or.reduce(candidates); from .sam_guard import validate_transition; validate_transition(last_head,head)
                heads[i]=head
                for target in (cores,forbidden,items,skins,necks): target[i].fill(0)
                geometries.append({'face_box':last_confirmed_face[:4].astype(float).tolist(),'face_visible':False,'temporal_repaired':True,'tracking_source':'sam_confirmed_actor','mouth_y':None,'hat_pixels':0,'glasses_pixels':0}); last_head=head; continue
            parsed=_parse_selected_face(rgb,face,parser,aligned_parse)
            labels=parsed['labels']; core,veto=face_ownership(labels,parsed['hair'],parsed['hat'],face[:4])
            if core.sum()<32: raise QualityFailure(f'face_parsing_failed:{i}')
            raw=unpack_masks(packed[i:i+1].cpu())[0].numpy(); parts=[cv2.resize(p.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST) for p in raw]
            parts.extend([(parsed['hair']>.5).astype(np.uint8),(parsed['hat']>.5).astype(np.uint8)])
            head=component_family(core,parts,min(face[2:4])); head[veto]=False;head[core]=True
            owned=[j for j,p in enumerate(parts[:len(raw)]) if np.count_nonzero((p>0)&head)/max(1,np.count_nonzero(p))>.5 and np.count_nonzero((p>0)&head)/max(1,np.count_nonzero(head))>.01]
            last_head=head;last_confirmed_face=face; heads[i]=head;cores[i]=core;forbidden[i]=veto
            items[i]=np.isin(labels,[6,18])&head;skins[i]=np.isin(labels,[1,14]);necks[i]=labels==14
            geometries.append({'face_box':face[:4].astype(float).tolist(),'yunet_face':face.astype(float).tolist(),'face_visible':True,'temporal_repaired':False,'mouth_y':float(face[10:14].reshape(2,2)[:,1].mean()),'hat_pixels':int(np.count_nonzero((labels==18)&head)),'glasses_pixels':int(np.count_nonzero((labels==6)&head))})
        mask=torch.from_numpy(heads); return mask,_colored(mask),torch.from_numpy(cores),torch.from_numpy(forbidden),torch.from_numpy(items),torch.from_numpy(skins),torch.from_numpy(necks),json.dumps(geometries)

    def _assemble_with_actor_mask(self, images, actor_masks, packed, detector, parser, cv2, np, torch,
                                  QualityFailure, face_ownership, component_family, aligned_parse):
        """Bidirectionally own hidden head frames from an observed face anchor."""
        from comfy.ldm.sam3.tracker import unpack_masks
        from .tracking import plausible
        from .sam_guard import validate_transition
        frame_count, height, width = images.shape[:3]
        if np.any(np.count_nonzero(actor_masks > .1, axis=(1, 2)) == 0):
            raise QualityFailure("selected_body_track_empty_frame")

        detected = [None] * frame_count
        for index, tensor in enumerate(images):
            rgb = _rgb(tensor)
            detector.setInputSize((width, height))
            _, faces = detector.detect(np.ascontiguousarray(rgb[..., ::-1]))
            credible = _credible_face_detections(faces, width, height)
            eligible = _filter_faces_by_mask([face for _, face in credible], actor_masks[index])
            if eligible:
                try:
                    detected[index] = _rank_face_candidates(rgb, eligible, hint=actor_masks[index])
                except QualityFailure as error:
                    raise QualityFailure(f"actor_face_ambiguous:{index}:{error}") from error

        observed = [i for i, face in enumerate(detected) if face is not None]
        if not observed:
            raise QualityFailure("no_observed_face_anchor_for_actor_track")
        # Bound anchor validation work: inspect at most eight observed frames, in
        # descending face-area order. Each frame is decoded/parsed only for its
        # candidate evaluation; the first semantically and spatially owned anchor wins.
        anchor_candidates = sorted(observed, key=lambda i: float(detected[i][2] * detected[i][3]), reverse=True)[:8]
        anchor = anchor_candidates[0]
        associated = [None] * frame_count

        track_count = None
        mask_geometry = None
        def sam_parts(frame_index):
            nonlocal track_count, mask_geometry
            raw = unpack_masks(packed[frame_index:frame_index + 1].cpu())[0]
            if hasattr(raw, "cpu"):
                raw = raw.cpu().numpy()
            raw = np.asarray(raw)
            if raw.ndim != 3 or raw.shape[0] > 64 or min(raw.shape[1:]) < 1:
                raise QualityFailure(f"actor_sam_masks_invalid:{frame_index}")
            if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1):
                raise QualityFailure(f"actor_sam_masks_invalid:{frame_index}")
            if track_count is None:
                track_count, mask_geometry = raw.shape[0], raw.shape[1:]
            elif raw.shape[0] != track_count or raw.shape[1:] != mask_geometry:
                raise QualityFailure(f"actor_sam_track_dimensions_changed:{frame_index}")
            return [cv2.resize((part > .5).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
                    for part in raw]

        heads = np.zeros((frame_count, height, width), np.float32)
        cores = np.zeros_like(heads); forbidden = np.zeros_like(heads)
        items = np.zeros_like(heads); skins = np.zeros_like(heads); necks = np.zeros_like(heads)
        geometries = [None] * frame_count
        owned_indices = None

        def process(frame_index, previous_head=None, anchor_frame=False):
            nonlocal owned_indices
            rgb = _rgb(images[frame_index])
            tracks = accepted[7] if anchor_frame else sam_parts(frame_index)
            face = associated[frame_index]
            semantic = None
            if face is not None:
                if anchor_frame:
                    semantic, core, veto, head, owned_indices = accepted[2:7]
                else:
                    semantic = _parse_selected_face(rgb, face, parser, aligned_parse)
                    labels = semantic["labels"]
                    core, veto = face_ownership(labels, semantic["hair"], semantic["hat"], face[:4])
                    if core.sum() < 32:
                        raise QualityFailure(f"face_parsing_failed:{frame_index}")
                    owned_parts = [tracks[j] for j in owned_indices if j < len(tracks)]
                    if len(owned_parts) != len(owned_indices):
                        raise QualityFailure(f"sam_actor_track_missing:{frame_index}")
                    if not any(np.count_nonzero(part & core) / max(1, np.count_nonzero(core)) > .01 for part in owned_parts):
                        raise QualityFailure(f"visible_face_not_owned_by_sam_track:{frame_index}")
                    head = component_family(core, owned_parts + [(semantic["hair"] > .5), (semantic["hat"] > .5)], min(face[2:4]))
                    head[veto] = False; head[core] = True
            else:
                if not owned_indices:
                    raise QualityFailure(f"no_confirmed_sam_head_ownership:{frame_index}")
                owned_parts = [tracks[j] for j in owned_indices if j < len(tracks)]
                if len(owned_parts) != len(owned_indices) or not any(part.any() for part in owned_parts):
                    raise QualityFailure(f"sam_actor_track_missing:{frame_index}")
                head = np.logical_or.reduce(owned_parts)
                core = veto = None
            body = actor_masks[frame_index] > .1
            head_area = np.count_nonzero(head)
            intersection = np.count_nonzero(head & body)
            ownership_ratio = intersection / head_area if head_area else 0.0
            recovered = False
            if head_area < 1 or ownership_ratio < .5:
                # A face-present non-anchor semantic expansion can include background.
                # Recover only from the anchor-confirmed SAM tracks; never re-anchor.
                if face is not None and not anchor_frame and owned_indices:
                    candidate = np.logical_or.reduce(owned_parts)
                    candidate_area = np.count_nonzero(candidate)
                    candidate_intersection = np.count_nonzero(candidate & body)
                    candidate_ratio = candidate_intersection / candidate_area if candidate_area else 0.0
                    recovery_accepted = candidate_area > 0 and candidate_ratio >= .5
                    continuity_error = None
                    if recovery_accepted and previous_head is not None:
                        try:
                            validate_transition(previous_head, candidate, motion_compensated=True)
                        except ValueError as error:
                            recovery_accepted = False
                            continuity_error = error
                    if continuity_error is not None:
                        raise QualityFailure(f"sam_head_continuity_unconfirmed:{frame_index}:{continuity_error}") from continuity_error
                    if recovery_accepted:
                        head = candidate
                        recovered = True
                if not recovered:
                    raise QualityFailure(f"sam_head_not_owned_by_selected_body:{frame_index}")
            if recovered:
                face = semantic = core = veto = None
            elif previous_head is not None:
                try:
                    validate_transition(previous_head, head, motion_compensated=True)
                except ValueError as error:
                    raise QualityFailure(f"sam_head_continuity_unconfirmed:{frame_index}:{error}") from error
            heads[frame_index] = head
            if face is not None:
                cores[frame_index] = core; forbidden[frame_index] = veto
                labels = semantic["labels"]
                items[frame_index] = np.isin(labels, [6, 18]) & head
                skins[frame_index] = np.isin(labels, [1, 14]); necks[frame_index] = labels == 14
                geometries[frame_index] = {"face_box": face[:4].astype(float).tolist(), "yunet_face": face.astype(float).tolist(),
                    "face_visible": True, "temporal_repaired": False, "tracking_source": "sam_confirmed_actor",
                    "mouth_y": float(face[10:14].reshape(2, 2)[:, 1].mean()),
                    "hat_pixels": int(np.count_nonzero((labels == 18) & head)), "glasses_pixels": int(np.count_nonzero((labels == 6) & head))}
            else:
                geometries[frame_index] = {"face_box": None, "face_visible": False, "temporal_repaired": True,
                    "tracking_source": "sam_confirmed_actor", "mouth_y": None, "hat_pixels": 0, "glasses_pixels": 0}
            return head

        # Evaluate candidates without touching output arrays or shared ownership state.
        # Only explicit semantic/ownership quality failures advance to another frame;
        # malformed masks, parser exceptions, ambiguity, and schema errors propagate.
        anchor_failure = None
        accepted = None
        for candidate_frame in anchor_candidates:
            face = detected[candidate_frame]
            rgb = _rgb(images[candidate_frame])
            tracks = sam_parts(candidate_frame)
            semantic = _parse_selected_face(rgb, face, parser, aligned_parse)
            labels = semantic["labels"]
            core, veto = face_ownership(labels, semantic["hair"], semantic["hat"], face[:4])
            if core.sum() < 32:
                anchor_failure = QualityFailure(f"face_parsing_failed:{candidate_frame}")
                continue
            parts = tracks + [(semantic["hair"] > .5), (semantic["hat"] > .5)]
            head = component_family(core, parts, min(face[2:4]))
            head[veto] = False; head[core] = True
            candidate_owned = []
            head_area = int(np.sum(head))
            for track_index, part in enumerate(tracks):
                part_area = int(np.sum(part))
                intersection_area = int(np.sum(part & head))
                if (intersection_area / max(1, part_area) > .5
                        and intersection_area / max(1, head_area) > .01):
                    candidate_owned.append(track_index)
            body = actor_masks[candidate_frame] > .1
            overlap = int(np.sum(head & body)) / head_area if head_area else 0.0
            if not candidate_owned:
                anchor_failure = QualityFailure(f"no_sam_head_owned_by_anchor:{candidate_frame}")
            elif overlap < .5:
                anchor_failure = QualityFailure(f"sam_head_not_owned_by_selected_body:{candidate_frame}")
            else:
                accepted = (candidate_frame, face, semantic, core, veto, head, candidate_owned, tracks)
            if accepted is not None:
                anchor = candidate_frame
                associated = [None] * frame_count
                associated[anchor] = face
                # Rebuild the face association outward from the accepted fixed anchor.
                for direction in (1, -1):
                    last = face
                    index = anchor + direction
                    while 0 <= index < frame_count:
                        next_face = detected[index]
                        if next_face is not None:
                            if not plausible(last, next_face):
                                raise QualityFailure(f"actor_face_untrusted:{index}")
                            associated[index] = next_face
                            last = next_face
                        index += direction
                # Only the accepted candidate's semantic masks/head are retained.
                owned_indices = candidate_owned
                break
        if accepted is None:
            if anchor_failure is not None:
                raise anchor_failure
            raise QualityFailure(f"sam_head_not_owned_by_selected_body:{anchor}")
        anchor_head = process(anchor, anchor_frame=True)
        for direction in (1, -1):
            previous = anchor_head
            index = anchor + direction
            while 0 <= index < frame_count:
                previous = process(index, previous_head=previous)
                index += direction
        mask = torch.from_numpy(heads)
        return (mask, _colored(mask), torch.from_numpy(cores), torch.from_numpy(forbidden),
                torch.from_numpy(items), torch.from_numpy(skins), torch.from_numpy(necks), json.dumps(geometries))


def _target_report(text):
    if not isinstance(text, str) or len(text) > 4096:
        raise ValueError("Target report exceeds bounded JSON limits")
    try:
        report = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid target report JSON") from error
    required = {"version", "decision", "reason", "frame_count", "target_index"}
    interval_keys = {"active_start_frame", "active_end_frame"}
    if (not isinstance(report, dict) or not required.issubset(report)
            or set(report) - required not in (set(), interval_keys)):
        raise ValueError("Invalid target report schema")
    if type(report["version"]) is not int or report["version"] != 1 or type(report["frame_count"]) is not int or report["frame_count"] < 1:
        raise ValueError("Invalid target report version or frame count")
    if report["decision"] not in ("render", "native"):
        raise ValueError("Invalid target decision")
    if report["reason"] not in ("TARGET_CONFIRMED", "TARGET_ABSENT", "TARGET_AMBIGUOUS", "TARGET_UNCERTAIN"):
        raise ValueError("Invalid target reason")
    has_start = "active_start_frame" in report
    has_end = "active_end_frame" in report
    if has_start != has_end:
        raise ValueError("Active interval bounds must be provided together")
    if has_start:
        start, end = report["active_start_frame"], report["active_end_frame"]
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= report["frame_count"]):
            raise ValueError("Invalid active interval bounds")
        if report["decision"] != "render":
            raise ValueError("Active interval bounds require a render target")
        if report["reason"] == "TARGET_CONFIRMED" and end - start < 3:
            raise ValueError("Confirmed active interval is too short")
    index = report["target_index"]
    if index is not None and (type(index) is not int or not 0 <= index < 64):
        raise ValueError("Invalid target track index")
    if report["reason"] == "TARGET_CONFIRMED":
        if report["decision"] != "render" or index is None:
            raise ValueError("Inconsistent target decision")
    elif report["decision"] != "native" or index is not None:
        raise ValueError("Inconsistent target decision")
    return report


def _credible_face_detections(faces, width, height):
    """Return confidence- and geometry-credible YuNet faces."""
    import numpy as np
    credible = []
    if faces is not None:
        faces = np.asarray(faces)
        if faces.ndim != 2 or faces.shape[1] < 15 or not np.isfinite(faces).all():
            raise ValueError("Malformed YuNet detections")
        for face in faces:
            x, y, fw, fh = face[:4]
            if face[14] < .9:
                continue
            if (fw <= 0 or fh <= 0 or x < 0 or y < 0 or x+fw > width or y+fh > height):
                continue
            points = face[4:14].reshape(5, 2)
            if (np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width)
                    or np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height)):
                continue
            credible.append((points, face.copy()))
    return credible


def _identity_observations(image_tensor, packed_mask, detector, width, height, track_count,
                           mask_geometry, cv2, np, unpack_masks, *, allow_single_face_overlap=False):
    """Validate one frame and return BGR plus uniquely SAM-owned credible faces."""
    image = np.asarray(image_tensor.cpu() if hasattr(image_tensor, "cpu") else image_tensor)
    if image.shape != (height, width, 3) or not np.isfinite(image).all():
        raise ValueError("Invalid target-selection image frame")
    bgr = np.ascontiguousarray(np.rint(np.clip(image[..., ::-1], 0, 1) * 255).astype(np.uint8))
    detector.setInputSize((width, height))
    _, faces = detector.detect(bgr)
    credible = _credible_face_detections(faces, width, height)
    if not credible:
        return bgr, {}
    raw = unpack_masks(packed_mask.cpu())[0]
    if hasattr(raw, "cpu"):
        raw = raw.cpu().numpy()
    raw = np.asarray(raw)
    if (raw.ndim != 3 or raw.shape[0] != track_count or raw.shape[1:] != mask_geometry
            or not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1)):
        raise ValueError("SAM person-track masks changed or became invalid during decode")
    owners = []
    for points, face in credible:
        xi = np.clip(points[:, 0].astype(int), 0, width-1)
        yi = np.clip(points[:, 1].astype(int), 0, height-1)
        tracks = []
        for track in range(track_count):
            resized = cv2.resize((raw[track] > .5).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            if np.mean(resized[yi, xi]) >= .8 or (
                allow_single_face_overlap and track_count == 1 and len(credible) == 1
                and _filter_faces_by_mask([face], resized)
            ):
                tracks.append(track)
        owners.append((tracks[0] if len(tracks) == 1 else None, face))
    counts = {track: sum(owner == track for owner, _ in owners)
              for track, _ in owners if track is not None}
    return bgr, {track: next(face for owner, face in owners if owner == track)
                 for track, occurrences in counts.items() if occurrences == 1}


def _consecutive_face_runs(frames):
    runs = []
    for frame in frames:
        if not runs or frame > runs[-1][-1] + 1:
            runs.append([frame])
        else:
            runs[-1].append(frame)
    return runs


class HSHeadSwapTarget:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "track_data": ("SAM3_TRACK_DATA",)},
                "optional": {"selection_policy": (["legacy_stable_body", "face_visible_actor"],),
                             "identity_lock": ("BOOLEAN", {"default": False}),
                             "identity_attempt_prefix": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("actor_mask", "target_report")
    FUNCTION = "select"
    CATEGORY = "SaoChep/HeadSwap"

    @classmethod
    def IS_CHANGED(cls, identity_lock=False, **_kwargs):
        # Identity templates are attempt-local mutable state; never reuse a cached
        # decision after a first scene seeds or changes the source template.
        return float("nan") if identity_lock else None

    def select(self, images, track_data, selection_policy="legacy_stable_body", identity_lock=False, identity_attempt_prefix=""):
        import cv2, numpy as np, torch, re
        if type(identity_lock) is not bool:
            raise ValueError("identity_lock must be boolean")
        if identity_lock and not re.fullmatch(r"aiworkergrid/[0-9a-f]{32}", identity_attempt_prefix):
            raise ValueError("Invalid attempt-owned identity prefix")
        if identity_lock and selection_policy != "face_visible_actor":
            raise ValueError("Identity lock requires face_visible_actor selection")
        if identity_lock and not isinstance(identity_attempt_prefix, str):
            raise ValueError("Identity attempt prefix must be a string")
        from comfy.ldm.sam3.tracker import unpack_masks
        if selection_policy not in ("legacy_stable_body", "face_visible_actor"):
            raise ValueError("Unknown target selection policy")
        count, height, width = images.shape[:3]
        if count < 1 or height < 1 or width < 1:
            raise ValueError("Expected non-empty person-track frames")
        if (isinstance(track_data, dict) and "packed_masks" in track_data
                and track_data["packed_masks"] is None
                and type(track_data.get("n_frames")) is int and track_data["n_frames"] == count
                and type(track_data.get("scores")) is list and not track_data["scores"]):
            output = torch.zeros((1, height, width), dtype=torch.float32).expand((count, height, width))
            report = {"version": 1, "decision": "native", "reason": "TARGET_ABSENT",
                      "frame_count": count, "target_index": None}
            return output, json.dumps(report, separators=(",", ":"))
        packed = track_data.get("packed_masks")
        if packed is None or len(packed) != count:
            raise ValueError("SAM person-track timing mismatch")
        if getattr(packed, "ndim", None) != 4 or packed.shape[0] != count:
            raise ValueError("SAM person-track tensor must be [frames,tracks,height,width]")
        max_tracks = 64
        track_count = None
        mask_geometry = None
        areas = None
        for frame in range(count):
            raw = unpack_masks(packed[frame:frame + 1].cpu())[0]
            if hasattr(raw, "cpu"):
                raw = raw.cpu().numpy()
            raw = np.asarray(raw)
            if raw.ndim != 3 or raw.shape[0] > max_tracks or raw.shape[1] < 1 or raw.shape[2] < 1:
                raise ValueError("SAM person-track mask dimensions exceed supported bounds")
            if track_count is None:
                track_count = raw.shape[0]
                mask_geometry = raw.shape[1:]
                areas = np.zeros((count, track_count), dtype=np.float64)
            elif raw.shape[0] != track_count or raw.shape[1:] != mask_geometry:
                raise ValueError("SAM person-track dimensions changed between frames")
            if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1):
                raise ValueError("SAM person-track masks must be finite and in [0,1]")
            for index, mask in enumerate(raw):
                areas[frame, index] = float(np.count_nonzero(mask > .5))
        track_count = track_count or 0
        if selection_policy == "face_visible_actor":
            return self._select_face_visible(images, packed, track_count, mask_geometry, areas, height, width, count, torch, cv2, np, unpack_masks,
                                             identity_lock=identity_lock, identity_attempt_prefix=identity_attempt_prefix)
        # Short clips (<3 frames) deliberately route native: there is too little
        # temporal evidence to distinguish a principal from a transient entrant.
        reason, decision, selected = "TARGET_ABSENT", "native", None
        if track_count and np.any(areas > 0):
            coverage = np.count_nonzero(areas > 0, axis=0) / count
            # A trusted principal must be near-complete. Within the trusted
            # group, tracks within five coverage points compete by stable area;
            # this rejects brief entrants without making one missing mask frame
            # alone disqualify an otherwise continuous shot principal.
            near_complete = np.flatnonzero((areas > 0).any(axis=0) & (coverage >= .9)) if count >= 3 else np.array([], dtype=int)
            if count < 3:
                reason = "TARGET_UNCERTAIN"
            elif not len(near_complete):
                reason = "TARGET_UNCERTAIN"
            else:
                best_coverage = float(np.max(coverage[near_complete]))
                eligible = near_complete[coverage[near_complete] >= best_coverage - .05]
                stable_area = np.array([np.median(areas[areas[:, i] > 0, i]) for i in eligible])
                ranked_order = np.argsort(-stable_area, kind="stable")
                ranked = eligible[ranked_order]
                ranked_area = stable_area[ranked_order]
                if len(ranked) > 1 and ranked_area[1] >= ranked_area[0] * .95:
                    reason = "TARGET_AMBIGUOUS"
                else:
                    selected = int(ranked[0]); reason = "TARGET_CONFIRMED"; decision = "render"
                    if np.any(areas[:, selected] <= 0):
                        selected = None; reason = "TARGET_UNCERTAIN"; decision = "native"
        if selected is None:
            output = torch.zeros((1, height, width), dtype=torch.float32).expand((count, height, width))
        else:
            output = np.empty((count, height, width), dtype=np.float32)
            for frame in range(count):
                raw = unpack_masks(packed[frame:frame + 1].cpu())[0]
                if hasattr(raw, "cpu"):
                    raw = raw.cpu().numpy()
                raw = np.asarray(raw)
                if (raw.ndim != 3 or raw.shape[0] != track_count or raw.shape[1:] != mask_geometry
                        or not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1)):
                    raise ValueError("SAM person-track masks changed or became invalid during decode")
                output[frame] = cv2.resize((raw[selected] > .5).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        report = {"version": 1, "decision": decision, "reason": reason, "frame_count": count, "target_index": selected}
        return torch.from_numpy(output), json.dumps(report, separators=(",", ":"))


    @staticmethod
    def _select_face_visible(images, packed, track_count, mask_geometry, areas, height, width, count, torch, cv2, np, unpack_masks,
                             identity_lock=False, identity_attempt_prefix=""):
        """Choose a body track from unique, sustained high-confidence face ownership."""
        evidence = [[] for _ in range(track_count)]
        identity_match_frames = [[] for _ in range(track_count)]
        recognizer = template_file = templates = None
        identity_unmatched = False
        if identity_lock:
            import folder_paths
            from .identity import template_path, load_templates, create_recognizer, embed, match_counts
            template_file = template_path(folder_paths.get_output_directory(), identity_attempt_prefix)
            templates = load_templates(template_file)
            recognizer = create_recognizer(Path(folder_paths.models_dir) / "sface" / "face_recognition_sface_2021dec.onnx")
        if count >= 3 and track_count:
            from .skin.skin_node import AGDrivingSkinColor
            detector, _ = AGDrivingSkinColor.models()
            for frame in range(count):
                bgr, owned = _identity_observations(
                    images[frame], packed[frame:frame+1], detector, width, height,
                    track_count, mask_geometry, cv2, np, unpack_masks)
                for track, face in owned.items():
                    evidence[track].append(frame)
                    if identity_lock and templates is not None:
                        feature = embed(recognizer, bgr, face)
                        if feature is not None and match_counts([feature], templates)[0]:
                            identity_match_frames[track].append(frame)
        # A profile can place landmarks along the edge of an otherwise overlapping
        # mask. Retry only sparse strict ownership on a continuous, single track,
        # with an existing actor template and the same face-box gate as head assembly.
        if (templates is not None and track_count == 1 and evidence[0]
                and max(map(len, _consecutive_face_runs(evidence[0]))) < 3
                and np.all(areas[:, 0] > 0)):
            retry_frames = []
            retry_matches = []
            for frame in range(count):
                bgr, owned = _identity_observations(
                    images[frame], packed[frame:frame+1], detector, width, height,
                    track_count, mask_geometry, cv2, np, unpack_masks, allow_single_face_overlap=True)
                if 0 in owned:
                    retry_frames.append(frame)
                    feature = embed(recognizer, bgr, owned[0])
                    if feature is not None and match_counts([feature], templates)[0]:
                        retry_matches.append(frame)
            evidence[0], identity_match_frames[0] = retry_frames, retry_matches
        candidates = []
        for track, frames in enumerate(evidence):
            best = max(_consecutive_face_runs(frames), key=len, default=[])
            if len(best) >= 3:
                candidates.append((track, best))
        candidates.sort(key=lambda item: (-len(item[1]), item[1][0], item[0]))
        selected = candidates[0][0] if candidates else None
        ambiguous = len(candidates) > 1 and len(candidates[1][1]) >= len(candidates[0][1]) * .95
        if identity_lock:
            from .identity import MIN_MATCHES, save_templates_once
            if templates is not None:
                eligible = [track for track, run in candidates
                            if len(set(run).intersection(identity_match_frames[track])) >= MIN_MATCHES]
                ambiguous = len(eligible) > 1
                identity_unmatched = not eligible and not ambiguous
                selected = eligible[0] if len(eligible) == 1 else None
            elif selected is not None and not ambiguous:
                from .identity import IdentityError
                selected_run = next(run for track, run in candidates if track == selected)
                sample_positions = np.linspace(0, len(selected_run) - 1,
                                               num=min(5, len(selected_run))).round().astype(int)
                seed = []
                for position in sample_positions:
                    frame = selected_run[int(position)]
                    bgr, owned = _identity_observations(
                        images[frame], packed[frame:frame+1], detector, width, height,
                        track_count, mask_geometry, cv2, np, unpack_masks)
                    if selected not in owned:
                        raise IdentityError("Source actor observations changed during identity seeding")
                    feature = embed(recognizer, bgr, owned[selected])
                    if feature is not None:
                        seed.append(feature)
                if len(seed) < 3:
                    selected = None
                else:
                    save_templates_once(template_file, seed)
            else:
                selected = None
        if ambiguous:
            selected = None
        start = end = None
        if selected is not None:
            # Preserve the existing best-run ranking for identity, but begin
            # only at reliable face evidence. Earlier body-only entry stays native.
            run = next(run for run in _consecutive_face_runs(evidence[selected]) if len(run) >= 3)
            start, end = run[0], run[-1] + 1
            if np.any(areas[start:end, selected] <= 0):
                selected = None
            else:
                while end < count and areas[end, selected] > 0:
                    end += 1
        if selected is None:
            # No masks at all is absence; tracks without reliable face evidence
            # (or competing face-supported tracks) are uncertainty.
            reason = ("TARGET_ABSENT" if identity_unmatched or not np.any(areas > 0)
                      else "TARGET_AMBIGUOUS" if ambiguous else "TARGET_UNCERTAIN")
            output = torch.zeros((1, height, width), dtype=torch.float32).expand((count, height, width))
            if reason in ("TARGET_UNCERTAIN", "TARGET_AMBIGUOUS"):
                summaries = []
                for track, frames in enumerate(evidence[:8]):
                    runs = []
                    for frame in frames:
                        if not runs or frame > runs[-1][-1] + 1: runs.append([frame])
                        else: runs[-1].append(frame)
                    summaries.append(f"{track}:{int(np.count_nonzero(areas[:, track]))}:{len(frames)}:{max(map(len, runs), default=0)}:{float(np.median(areas[areas[:, track] > 0, track])) if np.any(areas[:, track] > 0) else 0:.1f}")
            report = {"version":1,"decision":"native","reason":reason,"frame_count":count,"target_index":None}
        else:
            output = np.zeros((count, height, width), dtype=np.float32)
            for frame in range(start, end):
                raw = unpack_masks(packed[frame:frame+1].cpu())[0]
                if hasattr(raw, "cpu"):
                    raw = raw.cpu().numpy()
                raw = np.asarray(raw)
                if (raw.ndim != 3 or raw.shape[0] != track_count or raw.shape[1:] != mask_geometry
                        or not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1)):
                    raise ValueError("SAM person-track masks changed or became invalid during decode")
                output[frame] = cv2.resize((raw[selected] > .5).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            report = {"version":1,"decision":"render","reason":"TARGET_CONFIRMED","frame_count":count,"target_index":selected,
                      "active_start_frame":start,"active_end_frame":end}
        mask = output if selected is None else torch.from_numpy(output)
        return mask, json.dumps(report, separators=(",", ":"))


class HSHeadSwapActiveSlice:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "actor_mask": ("MASK",), "target_report": ("STRING",)}}
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("active_images", "active_actor_mask", "active_target_report")
    FUNCTION = "slice"
    CATEGORY = "SaoChep/HeadSwap"

    def slice(self, images, actor_mask, target_report):
        import numpy as np
        report = _target_report(target_report)
        count = getattr(images, "shape", (0,))[0]
        if (type(count) is not int or count < 1 or report["frame_count"] != count
                or report["decision"] != "render" or report["target_index"] is None):
            raise ValueError("Active slice requires a confirmed full-scene target report")
        if (getattr(actor_mask, "ndim", None) != 3 or actor_mask.shape[0] != count
                or tuple(actor_mask.shape[1:]) != tuple(images.shape[1:3])):
            raise ValueError("Actor mask timing or geometry does not match source images")
        masks = np.asarray(actor_mask.cpu() if hasattr(actor_mask, "cpu") else actor_mask)
        if not np.isfinite(masks).all() or np.any(masks < 0) or np.any(masks > 1):
            raise ValueError("Actor masks must be finite and in [0,1]")
        active = np.any(masks > .5, axis=(1, 2))
        indices = np.flatnonzero(active)
        if not len(indices):
            raise ValueError("Confirmed actor mask is empty")
        start, end = int(indices[0]), int(indices[-1]) + 1
        if not active[start:end].all():
            raise ValueError("Actor mask active interval contains empty frames")
        if ("active_start_frame" in report
                and (report["active_start_frame"], report["active_end_frame"]) != (start, end)):
            raise ValueError("Actor mask occupancy does not match reported active interval")
        localized = {"version":1,"decision":"render","reason":"TARGET_CONFIRMED",
                     "frame_count":end-start,"target_index":report["target_index"]}
        if start != 0 or end != count:
            localized.update(active_start_frame=0, active_end_frame=end-start)
        return (images[start:end], actor_mask[start:end], json.dumps(localized, separators=(",", ":")))


class HSHeadSwapPreflight:
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "check"
    CATEGORY = "SaoChep/HeadSwap"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"target_report": ("STRING",), "native_width": ("INT",), "native_height": ("INT",), "delivery_width": ("INT",), "delivery_height": ("INT",)},
                "optional": {"prepared_report": ("STRING",), "reference_mask": ("IMAGE",), "delivery_scale": ("INT", {"default": 1, "min": 1, "max": 2, "step": 1})}}
    def check(self, target_report, native_width, native_height, delivery_width, delivery_height, prepared_report=None, reference_mask=None, delivery_scale=1):
        import numpy as np
        target = _target_report(target_report)
        dims = (native_width, native_height, delivery_width, delivery_height)
        if any(type(value) is not int or not 1 <= value <= 16384 for value in dims):
            raise ValueError("Dimensions must be integers in [1,16384]")
        if type(delivery_scale) is not int or delivery_scale not in (1, 2):
            raise ValueError("delivery_scale must be 1 or 2")
        final_delivery_width = delivery_width * delivery_scale
        final_delivery_height = delivery_height * delivery_scale
        if final_delivery_width > 16384 or final_delivery_height > 16384:
            raise ValueError("Scaled delivery dimensions exceed 16384")
        has_prepared_report = prepared_report is not None
        has_reference_mask = reference_mask is not None
        if has_prepared_report != has_reference_mask:
            raise ValueError("Prepared report and reference mask must be provided together")
        prepared = False
        if has_prepared_report:
            if target["decision"] != "render":
                raise ValueError("Native target decision cannot claim prepared generation")
            if not isinstance(prepared_report, str) or len(prepared_report) > 16384:
                raise ValueError("Prepared report exceeds bounded JSON limits")
            try:
                parsed_prepared = json.loads(prepared_report)
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError("Invalid prepared report JSON") from error
            if not isinstance(parsed_prepared, dict):
                raise ValueError("Prepared report must be an object")
            mode = parsed_prepared.get("mode")
            detection = parsed_prepared.get("driving_detection")
            scale = parsed_prepared.get("scale_correction")
            translation = parsed_prepared.get("translation")
            if mode != "translation_only" or detection not in ("tracked_face_roi", "full_frame"):
                raise ValueError("Prepared report lacks recognized alignment evidence")
            if type(scale) not in (int, float) or not math.isfinite(scale) or scale != 1.0:
                raise ValueError("Prepared report has invalid scale correction")
            if (not isinstance(translation, list) or len(translation) != 2
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in translation)):
                raise ValueError("Prepared report has invalid translation")
            if (getattr(reference_mask, "ndim", None) != 4
                    or tuple(reference_mask.shape) != (1, native_height, native_width, 3)):
                raise ValueError("Reference mask must have exact [1,H,W,3] native geometry")
            mask = reference_mask.detach().cpu().numpy() if hasattr(reference_mask, "detach") else np.asarray(reference_mask)
            if not np.isfinite(mask).all() or np.any(mask < 0) or np.any(mask > 1):
                raise ValueError("Reference mask values must be finite and in [0,1]")
            blue = mask[0, ..., 2]
            if (np.any(np.abs(mask[0, ..., :2]) > 1e-6)
                    or np.any(~(np.isclose(blue, 0.0, atol=1e-6) | np.isclose(blue, 1.0, atol=1e-6)))
                    or not np.any(np.isclose(blue, 1.0, atol=1e-6))):
                raise ValueError("Reference mask does not match the black/blue binary encoding")
            prepared = True
        report = {"version": 1, "decision": target["decision"], "reason": target["reason"], "frame_count": target["frame_count"], "target_index": target["target_index"], "native_width": native_width, "native_height": native_height, "delivery_width": final_delivery_width, "delivery_height": final_delivery_height, "prepared": prepared}
        for key in ("active_start_frame", "active_end_frame"):
            if key in target:
                report[key] = target[key]
        return {"ui": {"headswap_preflight": [report]}, "result": ()}


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
    "SaoChepHeadSwapTarget":HSHeadSwapTarget,
    "SaoChepHeadSwapActiveSlice":HSHeadSwapActiveSlice,
    "SaoChepHeadSwapPreflight":HSHeadSwapPreflight,
    "SaoChepHeadSwapFrameFit":HSFrameSequenceFit,
}
