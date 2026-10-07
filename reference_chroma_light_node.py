# -*- coding: utf-8 -*-
"""
ReferenceChromaLightTest ComfyUI Node (Production V3.0 - Audited & Memory-Optimized)
Part of ComfyUI-OpticalStudioEnhance / SaoChepCommercial

Audited V3 Upgrades (GEMINI_HANDOVER_SCR_SMT_FINAL_ENGINE_AUDIT_V3.md):
- Section 7: Removed per-chunk torch.cuda.synchronize() in production (sync only when debug_enabled=True).
- Section 8: Cached CIELAB M and M_inv matrices per (device, dtype) in _CIELAB_CONST_CACHE.
- Section 9: Preallocated output tensor on CPU (corrected_images[i:j].copy_(chunk_cpu)) eliminating
  list accumulation + torch.cat memory peak for 20s/30s Pro/Pro++ jobs.
- Section 10: Adaptive GPU batch size with automatic CUDA OOM half-batch fallback.
- Section 11: Thread-local OpenCV YuNet FaceDetectorYN instances (threading.local()) for concurrency safety.
- Section 12: Removed dead/unreachable MediaPipe auto-download and initialization code.
"""

import os
import sys
import json
import time
import threading
import cv2
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False


_CIELAB_CONST_CACHE = {}
_THREAD_LOCAL = threading.local()
_YUNET_LOCK = threading.RLock()


def _get_cielab_matrices(device, dtype):
    key = (str(device), str(dtype))
    if key not in _CIELAB_CONST_CACHE:
        M = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=dtype, device=device)
        M_inv = torch.tensor([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252]
        ], dtype=dtype, device=device)
        _CIELAB_CONST_CACHE[key] = (M, M_inv)
    return _CIELAB_CONST_CACHE[key]


def rgb_srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    srgb_c = np.clip(srgb, 0.0, 1.0)
    return np.where(
        srgb_c <= 0.04045,
        srgb_c / 12.92,
        np.power((srgb_c + 0.055) / 1.055, 2.4)
    ).astype(np.float32)


def rgb_linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    lin_c = np.clip(linear, 0.0, 1.0)
    return np.where(
        lin_c <= 0.0031308,
        lin_c * 12.92,
        1.055 * np.power(np.maximum(lin_c, 1e-8), 1.0 / 2.4) - 0.055
    ).astype(np.float32)


def rgb_to_cielab_float32(srgb: np.ndarray) -> np.ndarray:
    """Standard D65 CIE XYZ -> CIELAB float32: L in [0..100], a, b signed"""
    rgb_lin = rgb_srgb_to_linear(srgb)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=np.float32)

    xyz = np.matmul(rgb_lin, M.T)
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    xr = xyz[..., 0] / Xn
    yr = xyz[..., 1] / Yn
    zr = xyz[..., 2] / Zn

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    fx = np.where(xr > eps, np.cbrt(np.maximum(xr, 1e-8)), (kappa * xr + 16.0) / 116.0)
    fy = np.where(yr > eps, np.cbrt(np.maximum(yr, 1e-8)), (kappa * yr + 16.0) / 116.0)
    fz = np.where(zr > eps, np.cbrt(np.maximum(zr, 1e-8)), (kappa * zr + 16.0) / 116.0)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return np.stack([L, a, b], axis=-1).astype(np.float32)


def transform_cielab_batch_np(batch_rgb: np.ndarray, offset_a: float, offset_b: float):
    lab = rgb_to_cielab_float32(batch_rgb)
    L = lab[..., 0]
    a = lab[..., 1] + np.float32(offset_a)
    b = lab[..., 2] + np.float32(offset_b)

    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    xr = np.where(fx ** 3 > eps, fx ** 3, (116.0 * fx - 16.0) / kappa)
    yr = np.where(L > kappa * eps, ((L + 16.0) / 116.0) ** 3, L / kappa)
    zr = np.where(fz ** 3 > eps, fz ** 3, (116.0 * fz - 16.0) / kappa)

    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    xyz = np.stack([xr * Xn, yr * Yn, zr * Zn], axis=-1)

    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ], dtype=np.float32)

    rgb_lin = np.matmul(xyz, M_inv.T)
    rgb_raw = rgb_linear_to_srgb(rgb_lin)

    oof_mask = np.any((rgb_lin < 0.0) | (rgb_lin > 1.0), axis=-1)
    oof_ratio = float(np.sum(oof_mask)) / float(max(1, oof_mask.size))
    rgb_clipped = np.clip(rgb_raw, 0.0, 1.0).astype(np.float32)

    return rgb_clipped, oof_ratio


def transform_cielab_batch_torch(
    batch_tensor: "torch.Tensor",
    offset_a: float,
    offset_b: float,
    device: str = "cuda",
    sync_cuda: bool = False,
):
    orig_device = batch_tensor.device
    inp = batch_tensor.to(device=device, dtype=torch.float32, non_blocking=True)
    M, M_inv = _get_cielab_matrices(inp.device, inp.dtype)

    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    inp_c = torch.clamp(inp, 0.0, 1.0)
    rgb_lin = torch.where(inp_c <= 0.04045, inp_c / 12.92, torch.pow((inp_c + 0.055) / 1.055, 2.4))
    xyz = torch.matmul(rgb_lin, M.t())

    xr = xyz[..., 0] / Xn
    yr = xyz[..., 1] / Yn
    zr = xyz[..., 2] / Zn

    fx = torch.where(xr > eps, torch.pow(torch.clamp(xr, min=1e-8), 1.0 / 3.0), (kappa * xr + 16.0) / 116.0)
    fy = torch.where(yr > eps, torch.pow(torch.clamp(yr, min=1e-8), 1.0 / 3.0), (kappa * yr + 16.0) / 116.0)
    fz = torch.where(zr > eps, torch.pow(torch.clamp(zr, min=1e-8), 1.0 / 3.0), (kappa * zr + 16.0) / 116.0)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy) + float(offset_a)
    b = 200.0 * (fy - fz) + float(offset_b)

    fy2 = (L + 16.0) / 116.0
    fx2 = fy2 + a / 500.0
    fz2 = fy2 - b / 200.0

    xr2 = torch.where(fx2 ** 3 > eps, fx2 ** 3, (116.0 * fx2 - 16.0) / kappa)
    yr2 = torch.where(L > kappa * eps, torch.pow((L + 16.0) / 116.0, 3.0), L / kappa)
    zr2 = torch.where(fz2 ** 3 > eps, fz2 ** 3, (116.0 * fz2 - 16.0) / kappa)

    xyz2 = torch.stack([xr2 * Xn, yr2 * Yn, zr2 * Zn], dim=-1)
    rgb_lin2 = torch.matmul(xyz2, M_inv.t())

    lin_c = torch.clamp(rgb_lin2, 0.0, 1.0)
    rgb_raw = torch.where(lin_c <= 0.0031308, lin_c * 12.92, 1.055 * torch.pow(torch.clamp(lin_c, min=1e-8), 1.0 / 2.4) - 0.055)

    oof_mask = torch.any((rgb_lin2 < 0.0) | (rgb_lin2 > 1.0), dim=-1)
    oof_ratio = float(torch.count_nonzero(oof_mask).item()) / float(max(1, oof_mask.numel()))
    rgb_clipped = torch.clamp(rgb_raw, 0.0, 1.0)

    if sync_cuda and str(device).startswith("cuda"):
        torch.cuda.synchronize()

    return rgb_clipped.to(orig_device, non_blocking=True), oof_ratio


_YUNET_DIAGNOSTIC = {
    "searched_paths": [],
    "model_path_found": None,
    "init_success": False,
    "init_error": None
}


def get_cached_yunet():
    """
    Thread-local OpenCV YuNet ONNX detector cache (Section 11).
    Prevents race conditions on setInputSize((w, h)) when concurrent jobs execute.
    """
    detector = getattr(_THREAD_LOCAL, "yunet_detector", None)
    if detector is not None:
        return detector, 0.0

    with _YUNET_LOCK:
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "face_detection_yunet_2023mar.onnx"),
            os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx"),
            os.path.join(os.path.dirname(__file__), "..", "models", "face_detection_yunet_2023mar.onnx"),
        ]
        if HAS_FOLDER_PATHS:
            possible_paths.append(os.path.join(folder_paths.models_dir, "detection", "face_detection_yunet_2023mar.onnx"))
            possible_paths.append(os.path.join(folder_paths.models_dir, "face_detection_yunet_2023mar.onnx"))

        _YUNET_DIAGNOSTIC["searched_paths"] = []
        model_path = None
        for p in possible_paths:
            exists = os.path.exists(p)
            size = os.path.getsize(p) if exists else 0
            _YUNET_DIAGNOSTIC["searched_paths"].append({"path": p, "exists": exists, "size_bytes": size})
            if exists and size > 200000 and model_path is None:
                model_path = p

        _YUNET_DIAGNOSTIC["model_path_found"] = model_path
        if not model_path:
            _YUNET_DIAGNOSTIC["init_error"] = "YuNet ONNX model file not found in any searched paths"
            return None, 0.0

        t0 = time.time()
        try:
            try:
                detector = cv2.FaceDetectorYN.create(model_path, "", (300, 300), score_threshold=0.5, nms_threshold=0.3)
            except Exception:
                import tempfile, shutil
                temp_p = os.path.join(tempfile.gettempdir(), "face_detection_yunet_2023mar.onnx")
                if not os.path.exists(temp_p):
                    shutil.copyfile(model_path, temp_p)
                detector = cv2.FaceDetectorYN.create(temp_p, "", (300, 300), score_threshold=0.5, nms_threshold=0.3)
            _THREAD_LOCAL.yunet_detector = detector
            init_ms = (time.time() - t0) * 1000.0
            _YUNET_DIAGNOSTIC["init_success"] = True
            return detector, init_ms
        except Exception as e:
            _YUNET_DIAGNOSTIC["init_success"] = False
            _YUNET_DIAGNOSTIC["init_error"] = f"{type(e).__name__}: {str(e)}"
            sys.stderr.write(f"[ReferenceChromaLightTest] Failed to load YuNet: {e}\n")
            return None, 0.0


class ReferenceChromaLightTest:
    CATEGORY = "OpticalStudioEnhance"
    FUNCTION = "apply_chroma_test"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("graded_images", "verification_preview", "analysis_report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Batch video frames from WAN [B, H, W, C]"}),
                "reference_image": ("IMAGE", {"tooltip": "Single reference photo R [1, H, W, C]"}),
                "strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Interpolation strength from video median to reference cheek"}),
                "max_ab_offset": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Maximum clamp magnitude for delta a* and delta b*"}),
                "sample_count": ("INT", {"default": 9, "min": 3, "max": 25, "step": 1, "tooltip": "Number of uniformly spaced sample frames to evaluate"}),
                "min_valid_samples": ("INT", {"default": 3, "min": 1, "max": 15, "step": 1, "tooltip": "Minimum valid face sample frames required"}),
                "min_face_width_px": ("INT", {"default": 64, "min": 32, "max": 256, "step": 8, "tooltip": "Rejection threshold for face width"}),
                "min_patch_pixels": ("INT", {"default": 32, "min": 8, "max": 128, "step": 4, "tooltip": "Rejection threshold for valid cheek patch pixels"}),
                "bypass_offset_threshold": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.02, "tooltip": "Bypass threshold (Section 3)"}),
                "processing_backend": (["auto", "gpu", "cpu_fast", "baseline"], {"default": "auto", "tooltip": "Processing backend"}),
                "batch_frames": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1, "tooltip": "Frames per vector batch"}),
                "debug_enabled": ("BOOLEAN", {"default": False, "tooltip": "Export cheek polygon overlay masks"}),
                "benchmark_run_id": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1, "tooltip": "Cache busting identifier"}),
                "enabled": ("BOOLEAN", {"default": False, "tooltip": "Master enable toggle"}),
            },
            "optional": {
                "branch": (["CHROMA_LIGHT", "WARM_HALF"], {"default": "CHROMA_LIGHT", "tooltip": "Branch B: CHROMA_LIGHT"}),
                "test_mode": (["auto_end_to_end", "TRANSFORM_ONLY_TEST"], {"default": "auto_end_to_end"})
            }
        }

    def _sample_cheek_color(self, img_rgb_u8: np.ndarray, min_face_w: int, min_patch_px: int) -> dict:
        h, w = img_rgb_u8.shape[:2]
        yunet, _ = get_cached_yunet()
        if yunet is None:
            return {"valid": False, "reason": "NO_YUNET_AVAILABLE", "engine": "none"}

        try:
            with _YUNET_LOCK:
                yunet.setInputSize((w, h))
                img_bgr = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2BGR)
                _, faces = yunet.detect(img_bgr)
            if faces is None or len(faces) == 0:
                return {"valid": False, "reason": "NO_FACE_DETECTED", "engine": "yunet_onnx"}

            if len(faces) > 1:
                return {"valid": False, "reason": "MULTIPLE_FACES", "engine": "yunet_onnx"}
            best_face = faces[0]

            bbox = best_face[0:4]
            face_w = float(bbox[2])
            if face_w < min_face_w:
                return {"valid": False, "reason": f"FACE_TOO_SMALL_{face_w:.1f}px", "engine": "yunet_onnx"}

            reye = best_face[4:6]
            leye = best_face[6:8]
            nose = best_face[8:10]
            rmouth = best_face[10:12]
            lmouth = best_face[12:14]

            ipd = float(np.linalg.norm(leye - reye))
            c_r = np.array([reye[0] - 0.40 * (nose[0] - reye[0]), reye[1] + 0.60 * (rmouth[1] - reye[1])])
            c_l = np.array([leye[0] + 0.40 * (leye[0] - nose[0]), leye[1] + 0.60 * (lmouth[1] - leye[1])])
            radius = max(4, int(round(ipd * 0.12)))

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (int(round(c_r[0])), int(round(c_r[1]))), radius, 255, -1)
            cv2.circle(mask, (int(round(c_l[0])), int(round(c_l[1]))), radius, 255, -1)
            patch_pixels = int(np.sum(mask > 0))
            if patch_pixels < min_patch_px:
                return {"valid": False, "reason": "CHEEKS_INSUFFICIENT_PATCH_PIXELS", "engine": "yunet_onnx"}

            img_rgb_f = img_rgb_u8.astype(np.float32) / 255.0
            lab = rgb_to_cielab_float32(img_rgb_f)

            pixels_L = lab[..., 0][mask > 0]
            pixels_a = lab[..., 1][mask > 0]
            pixels_b = lab[..., 2][mask > 0]

            p20, p80 = np.percentile(pixels_L, [20, 80])
            core_mask = (pixels_L >= p20) & (pixels_L <= p80)

            valid_a = pixels_a[core_mask]
            valid_b = pixels_b[core_mask]
            valid_L = pixels_L[core_mask]

            if len(valid_a) < max(4, min_patch_px // 2):
                return {"valid": False, "reason": "CHEEKS_INSUFFICIENT_CORE_PIXELS", "engine": "yunet_onnx"}

            mean_a = float(np.median(valid_a))
            mean_b = float(np.median(valid_b))
            mean_L = float(np.median(valid_L))

            theta = np.linspace(0, 2 * np.pi, 16)
            r_poly = np.stack([c_r[0] + radius * np.cos(theta), c_r[1] + radius * np.sin(theta)], axis=1).astype(np.int32)
            l_poly = np.stack([c_l[0] + radius * np.cos(theta), c_l[1] + radius * np.sin(theta)], axis=1).astype(np.int32)

            return {
                "valid": True,
                "engine": "yunet_onnx_fallback",
                "face_width_px": round(float(face_w), 1),
                "cheeks_used": ["right", "left"],
                "patch_pixels": patch_pixels,
                "core_pixels": int(len(valid_a)),
                "a": round(mean_a, 3),
                "b": round(mean_b, 3),
                "L": round(mean_L, 3),
                "r_poly": r_poly,
                "l_poly": l_poly
            }
        except Exception as e:
            return {"valid": False, "reason": f"EXCEPTION_{str(e)}", "engine": "yunet_onnx"}

    def _select_adaptive_batch_frames(self, requested_batch: int, frame_h: int, frame_w: int) -> int:
        if not (HAS_TORCH and torch.cuda.is_available()):
            return max(1, int(requested_batch))
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            bytes_per_frame = max(1, frame_h * frame_w * 3 * 4 * 6)  # ~6 intermediate float32 tensors per frame
            safe_max = max(4, min(64, int(free_bytes // (bytes_per_frame * 2))))
            return max(4, min(int(requested_batch), safe_max))
        except Exception:
            return max(1, int(requested_batch))

    def apply_chroma_test(self, images, reference_image, strength=0.35, max_ab_offset=2.0,
                          sample_count=9, min_valid_samples=3, min_face_width_px=64,
                          min_patch_pixels=32, bypass_offset_threshold=0.20,
                          processing_backend="auto", batch_frames=16, debug_enabled=False,
                          benchmark_run_id=0, enabled=False, branch="CHROMA_LIGHT",
                          test_mode="auto_end_to_end"):
        if branch != "CHROMA_LIGHT" or test_mode not in ("auto_end_to_end", "TRANSFORM_ONLY_TEST"):
            raise ValueError("Commercial node supports automatic CHROMA_LIGHT only")
        t_node_start = time.time()

        t_model_init_ms = 0.0
        t_ref_landmark_ms = 0.0
        t_sample_landmark_ms = 0.0
        t_sampling_stats_ms = 0.0
        t_device_transfer_ms = 0.0
        t_color_transform_ms = 0.0

        env_diagnostic = {
            "python_version": sys.version,
            "yunet_diagnostic": _YUNET_DIAGNOSTIC,
            "torch_status": {
                "has_torch": HAS_TORCH,
                "torch_version": getattr(torch, "__version__", None) if HAS_TORCH else None,
                "cuda_available": bool(torch.cuda.is_available()) if HAS_TORCH else False,
                "cuda_device_name": str(torch.cuda.get_device_name(0)) if (HAS_TORCH and torch.cuda.is_available()) else None
            }
        }

        if not enabled:
            t_total_ms = (time.time() - t_node_start) * 1000.0
            rep = {
                "status": "BYPASS_DISABLED",
                "reason": "Node disabled by user",
                "timings": {"total_node_ms": round(t_total_ms, 2)},
                "diagnostic": env_diagnostic
            }
            return (images, images[:1], json.dumps(rep, indent=2))

        if test_mode == "TRANSFORM_ONLY_TEST":
            offset_a = 0.0987
            offset_b = 1.1109
            max_offset_magnitude = 1.1109
            clamped = False
            raw_offset_a = offset_a
            raw_offset_b = offset_b
            delta_a = offset_a / max(1e-6, strength)
            delta_b = offset_b / max(1e-6, strength)
            ref_sample = {"L": 75.11, "a": 16.603, "b": 13.485, "engine": "benchmark_fixed"}
            med_L, med_a, med_b = 74.535, 16.321, 10.311
            valid_vid_samples = [1] * 9
            status_label = "TRANSFORM_ONLY_TEST"
            landmark_engine = "benchmark_fixed"
        else:
            _, t_yn_init = get_cached_yunet()
            t_model_init_ms = t_yn_init

            t_ref_start = time.time()
            ref_np = reference_image[0].cpu().numpy() if hasattr(reference_image, "cpu") else reference_image[0]
            ref_u8 = np.clip(ref_np * 255.0, 0, 255).astype(np.uint8)
            ref_sample = self._sample_cheek_color(ref_u8, min_face_width_px, min_patch_pixels)
            t_ref_landmark_ms = (time.time() - t_ref_start) * 1000.0

            if not ref_sample["valid"]:
                t_total_ms = (time.time() - t_node_start) * 1000.0
                rep = {
                    "status": "BYPASS_NO_REF_FACE",
                    "reason": f"Reference face invalid: {ref_sample.get('reason')}",
                    "engine_used": ref_sample.get("engine", "none"),
                    "timings": {
                        "model_initialization_ms": round(t_model_init_ms, 2),
                        "reference_landmark_ms": round(t_ref_landmark_ms, 2),
                        "total_node_ms": round(t_total_ms, 2)
                    },
                    "diagnostic": env_diagnostic
                }
                return (images, images[:1], json.dumps(rep, indent=2))

            landmark_engine = ref_sample.get("engine", "unknown")

            t_sample_start = time.time()
            num_frames = images.shape[0]
            sample_indices = sorted(set(int(round((0.10 + i * (0.80 / max(1, sample_count - 1))) * (num_frames - 1))) for i in range(sample_count)))
            video_samples = []

            for idx in sample_indices:
                f_np = images[idx].cpu().numpy() if hasattr(images, "cpu") else images[idx]
                f_u8 = np.clip(f_np * 255.0, 0, 255).astype(np.uint8)
                res = self._sample_cheek_color(f_u8, min_face_width_px, min_patch_pixels)
                res["frame_idx"] = idx
                video_samples.append(res)

            t_sample_landmark_ms = (time.time() - t_sample_start) * 1000.0

            t_stats_start = time.time()
            valid_vid_samples = [s for s in video_samples if s["valid"]]
            if len(valid_vid_samples) < min_valid_samples:
                t_total_ms = (time.time() - t_node_start) * 1000.0
                rep = {
                    "status": "BYPASS_INSUFFICIENT_VALID_SAMPLES",
                    "reason": f"Found {len(valid_vid_samples)} valid samples, required {min_valid_samples}",
                    "engine_used": landmark_engine,
                    "timings": {
                        "model_initialization_ms": round(t_model_init_ms, 2),
                        "reference_landmark_ms": round(t_ref_landmark_ms, 2),
                        "video_sample_landmark_ms": round(t_sample_landmark_ms, 2),
                        "total_node_ms": round(t_total_ms, 2)
                    },
                    "diagnostic": env_diagnostic
                }
                return (images, images[:1], json.dumps(rep, indent=2))

            med_L = float(np.median([s["L"] for s in valid_vid_samples]))
            med_a = float(np.median([s["a"] for s in valid_vid_samples]))
            med_b = float(np.median([s["b"] for s in valid_vid_samples]))

            delta_a = ref_sample["a"] - med_a
            delta_b = ref_sample["b"] - med_b

            raw_offset_a = strength * delta_a
            raw_offset_b = strength * delta_b

            offset_a = np.clip(raw_offset_a, -max_ab_offset, max_ab_offset)
            offset_b = np.clip(raw_offset_b, -max_ab_offset, max_ab_offset)
            clamped = (offset_a != raw_offset_a) or (offset_b != raw_offset_b)

            max_offset_magnitude = max(abs(float(offset_a)), abs(float(offset_b)))
            t_sampling_stats_ms = (time.time() - t_stats_start) * 1000.0

            if max_offset_magnitude <= bypass_offset_threshold:
                t_total_ms = (time.time() - t_node_start) * 1000.0
                telemetry = {
                    "status": "BYPASS_SMALL_OFFSET",
                    "branch": branch,
                    "landmark_engine": landmark_engine,
                    "backend_used": "bypass",
                    "device_actual": "None (Zero Copy Pass-Through)",
                    "bypass_triggered": True,
                    "bypass_threshold": float(bypass_offset_threshold),
                    "max_offset_magnitude": round(max_offset_magnitude, 4),
                    "reference_cheek": {"L": round(ref_sample["L"], 3), "a": round(ref_sample["a"], 3), "b": round(ref_sample["b"], 3)},
                    "video_median_cheek": {"L": round(med_L, 3), "a": round(med_a, 3), "b": round(med_b, 3), "valid_samples": len(valid_vid_samples)},
                    "deltas": {"delta_a": round(delta_a, 3), "delta_b": round(delta_b, 3)},
                    "offsets": {
                        "strength": strength,
                        "applied_offset_a": round(float(offset_a), 4),
                        "applied_offset_b": round(float(offset_b), 4),
                        "clamped": bool(clamped)
                    },
                    "out_of_gamut": {"mean_oof_ratio_percent": 0.0},
                    "timings": {
                        "model_initialization_ms": round(t_model_init_ms, 2),
                        "reference_landmark_ms": round(t_ref_landmark_ms, 2),
                        "video_sample_landmark_ms": round(t_sample_landmark_ms, 2),
                        "sampling_statistics_ms": round(t_sampling_stats_ms, 2),
                        "device_transfer_ms": 0.0,
                        "color_transform_ms": 0.0,
                        "total_node_ms": round(t_total_ms, 2),
                        "encode_ms": 0.0
                    },
                    "benchmark_run_id": benchmark_run_id,
                    "diagnostic": env_diagnostic
                }
                return (images, images[:1], json.dumps(telemetry, indent=2))

            status_label = "SUCCESS_APPLIED"

        use_gpu = False
        if processing_backend in ["auto", "gpu"]:
            if HAS_TORCH and torch.cuda.is_available():
                use_gpu = True
                backend_used = "gpu"
            else:
                if processing_backend == "gpu":
                    raise RuntimeError("GPU requested but CUDA unavailable; refusing silent CPU fallback")
                backend_used = "cpu_fast"
        elif processing_backend == "cpu_fast":
            backend_used = "cpu_fast"
        else:
            backend_used = "baseline"

        b, fh, fw = images.shape[0], images.shape[1], images.shape[2]
        oof_ratios = []
        fresh_parity_benchmark = None

        # Preallocate destination buffer on CPU once (Section 9: avoids out_chunks + torch.cat memory spike)
        is_tensor_input = HAS_TORCH and isinstance(images, torch.Tensor)
        if is_tensor_input:
            corrected_images = torch.empty_like(images, device="cpu")
        else:
            corrected_images = np.empty_like(images)

        if use_gpu and HAS_TORCH:
            actual_device = torch.cuda.get_device_name(0)
            eff_batch = self._select_adaptive_batch_frames(batch_frames, fh, fw)
            i = 0
            while i < b:
                cur_step = min(eff_batch, b - i)
                chunk = images[i : i + cur_step]
                if not isinstance(chunk, torch.Tensor):
                    chunk = torch.from_numpy(chunk)
                try:
                    t_dt_sub1 = time.time()
                    chunk_cuda = chunk.to("cuda", non_blocking=True)
                    if debug_enabled:
                        torch.cuda.synchronize()
                    t_device_transfer_ms += (time.time() - t_dt_sub1) * 1000.0

                    t_col_sub = time.time()
                    chunk_res, oof = transform_cielab_batch_torch(
                        chunk_cuda, offset_a, offset_b, device="cuda", sync_cuda=bool(debug_enabled)
                    )
                    t_color_transform_ms += (time.time() - t_col_sub) * 1000.0

                    oof_ratios.append(oof)
                    t_dt_sub2 = time.time()
                    chunk_cpu = chunk_res.cpu()
                    if debug_enabled:
                        torch.cuda.synchronize()
                    t_device_transfer_ms += (time.time() - t_dt_sub2) * 1000.0

                    if is_tensor_input:
                        corrected_images[i : i + cur_step].copy_(chunk_cpu)
                    else:
                        corrected_images[i : i + cur_step] = chunk_cpu.numpy()

                    del chunk_cuda, chunk_res, chunk_cpu
                    i += cur_step
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and eff_batch > 2:
                        torch.cuda.empty_cache()
                        eff_batch = max(2, eff_batch // 2)
                        continue
                    raise

            if not debug_enabled:
                torch.cuda.synchronize()

            if debug_enabled:
                first_len = min(batch_frames, b)
                parity_chunk_gpu = corrected_images[0:first_len].numpy() if is_tensor_input else corrected_images[0:first_len]
                first_chunk_cpu_input = images[0:first_len].cpu().numpy() if hasattr(images, "cpu") else np.asarray(images[0:first_len])
                parity_chunk_cpu, _ = transform_cielab_batch_np(first_chunk_cpu_input, offset_a, offset_b)
                parity_diff = np.abs(parity_chunk_gpu - parity_chunk_cpu)
                parity_mae = float(np.mean(parity_diff))
                parity_max = float(np.max(parity_diff))
                fresh_parity_benchmark = {
                    "tested_frames": int(first_len),
                    "applied_offsets": {"offset_a": round(float(offset_a), 4), "offset_b": round(float(offset_b), 4)},
                    "srgb_threshold_fixed": True,
                    "rgb_float_mae": round(parity_mae, 8),
                    "rgb_float_max_diff": round(parity_max, 8),
                    "parity_verified": bool(parity_mae <= 0.1 / 255 and parity_max <= 1.0 / 255)
                }

        elif backend_used == "cpu_fast":
            actual_device = "CPU (Vectorized NumPy)"
            for i in range(0, b, batch_frames):
                j = min(i + batch_frames, b)
                chunk = images[i:j]
                chunk_np = chunk.cpu().numpy() if (HAS_TORCH and isinstance(chunk, torch.Tensor)) else np.asarray(chunk)
                t_col_sub = time.time()
                chunk_res, oof = transform_cielab_batch_np(chunk_np, offset_a, offset_b)
                t_color_transform_ms += (time.time() - t_col_sub) * 1000.0
                oof_ratios.append(oof)
                if is_tensor_input:
                    corrected_images[i:j].copy_(torch.from_numpy(chunk_res))
                else:
                    corrected_images[i:j] = chunk_res

        else:
            actual_device = "CPU (Baseline Python Loop)"
            for i in range(b):
                f_np = images[i].cpu().numpy() if (HAS_TORCH and hasattr(images, "cpu")) else images[i]
                t_col_sub = time.time()
                f_res, oof = transform_cielab_batch_np(f_np[None, ...], offset_a, offset_b)
                t_color_transform_ms += (time.time() - t_col_sub) * 1000.0
                oof_ratios.append(oof)
                if is_tensor_input:
                    corrected_images[i].copy_(torch.from_numpy(f_res[0]))
                else:
                    corrected_images[i] = f_res[0]

        mean_oof = float(np.mean(oof_ratios)) if oof_ratios else 0.0
        t_total_ms = (time.time() - t_node_start) * 1000.0

        telemetry = {
            "status": status_label,
            "branch": branch,
            "landmark_engine": landmark_engine,
            "backend_used": backend_used,
            "device_actual": actual_device,
            "bypass_triggered": False,
            "bypass_threshold": float(bypass_offset_threshold),
            "max_offset_magnitude": round(max_offset_magnitude, 4),
            "reference_cheek": {"L": round(ref_sample["L"], 3), "a": round(ref_sample["a"], 3), "b": round(ref_sample["b"], 3)},
            "video_median_cheek": {"L": round(med_L, 3), "a": round(med_a, 3), "b": round(med_b, 3), "valid_samples": len(valid_vid_samples)},
            "deltas": {"delta_a": round(delta_a, 3), "delta_b": round(delta_b, 3)},
            "offsets": {
                "strength": strength,
                "max_ab_offset": max_ab_offset,
                "raw_offset_a": round(raw_offset_a, 4),
                "raw_offset_b": round(raw_offset_b, 4),
                "applied_offset_a": round(float(offset_a), 4),
                "applied_offset_b": round(float(offset_b), 4),
                "clamped": bool(clamped)
            },
            "out_of_gamut": {"mean_oof_ratio_percent": round(mean_oof * 100.0, 4)},
            "cpu_gpu_parity_fresh": fresh_parity_benchmark,
            "timings": {
                "model_initialization_ms": round(t_model_init_ms, 2),
                "reference_landmark_ms": round(t_ref_landmark_ms, 2),
                "video_sample_landmark_ms": round(t_sample_landmark_ms, 2),
                "sampling_statistics_ms": round(t_sampling_stats_ms, 2),
                "device_transfer_ms": round(t_device_transfer_ms, 2),
                "color_transform_ms": round(t_color_transform_ms, 2),
                "total_node_ms": round(t_total_ms, 2),
                "encode_ms": 0.0
            },
            "benchmark_run_id": benchmark_run_id,
            "diagnostic": env_diagnostic
        }

        return (corrected_images, corrected_images[:1], json.dumps(telemetry, indent=2))


NODE_CLASS_MAPPINGS = {
    "ReferenceChromaLightTest": ReferenceChromaLightTest
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReferenceChromaLightTest": "Reference Chroma Light Test (Optimized V3.0)"
}
