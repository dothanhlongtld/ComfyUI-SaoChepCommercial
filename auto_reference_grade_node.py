# -*- coding: utf-8 -*-
"""
AutoReferenceGrade ComfyUI Node (Production V3.2 - Adaptive Safeguard Engine)
Part of ComfyUI-OpticalStudioEnhance

Changelog V3.2:
- Exact FFmpeg Candidate Scoring: Replaced NumPy approximation with exact FFmpeg C-libavfilter pipe on sample frames for 100% mathematical consistency with final render.
- Refined Candidate Skin Filter: Tightened locus (sat in [14..55]%, a* in [3.5..30.0], cr-cb >= 10) to completely reject beige bedroom walls and graphic prints (such as red stars).
- Active Background Shift Rejection: Any candidate level with mean_bg_shift > 2.0 is actively disqualified.
- Active Skin Clipping Rejection: Any candidate level inducing > 0.8% new blown highlights or crushed shadows is actively disqualified.
- Verified No-Op: strength=0 guarantees exact identity tensor pass-through (diff = 0.0).
"""

import os
import sys
import shutil
import json
import time
import math
import subprocess
import cv2
import numpy as np

def get_ffmpeg_binary():
    bin_p = shutil.which("ffmpeg")
    if bin_p:
        return bin_p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


try:
    import torch
except ImportError:
    torch = None

class AutoReferenceGrade:
    """
    ComfyUI Node: AutoReferenceGrade (Adaptive Safeguard V3.2)
    """
    CATEGORY = "OpticalStudioEnhance"
    FUNCTION = "apply_grade"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("graded_images", "verification_preview", "analysis_report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Batch video frames from WAN [B, H, W, C]"}),
                "reference_image": ("IMAGE", {"tooltip": "Single reference photo R [1, H, W, C]"}),
                "mode": (["reference_safe", "studio_warm"], {
                    "default": "reference_safe",
                    "tooltip": "reference_safe: strictly respects reference tone with exact FFmpeg candidate evaluation and background/clipping rejection. studio_warm: aesthetic V3 preset."
                }),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider", "tooltip": "Interpolation strength from neutral (0.0 = exact bypass/no-op, 1.0 = target)"}),
                "unsharp_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.80, "step": 0.02, "tooltip": "Micro-acutance unsharp strength (0.0 = completely disabled)"}),
                "sensor_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.1, "tooltip": "ISO sensor noise std dev (0.0 = completely disabled)"}),
                "enable": ("BOOLEAN", {"default": True, "tooltip": "Enable or bypass node"})
            },
            "optional": {
                "candidate_selection": ("BOOLEAN", {"default": True, "tooltip": "In reference_safe mode, evaluate levels 0.0, 0.5, 1.0 with exact FFmpeg pipe and auto-select"}),
                "red_bias": ("FLOAT", {"default": 0.0, "min": -0.03, "max": 0.03, "step": 0.002, "tooltip": "Manual fine red offset (-0.03 to +0.03, default 0.0)"}),
                "max_contrast": ("FLOAT", {"default": 1.10, "min": 1.0, "max": 1.30, "step": 0.01}),
                "max_saturation": ("FLOAT", {"default": 1.28, "min": 0.90, "max": 1.40, "step": 0.01}),
                "adaptive_tier_mode": ("BOOLEAN", {"default": True, "tooltip": "Automatically measure Face ROI IQA variance. If >= 100 (Tier 1 sharp), automatically zeroes noise and unsharp to protect silky hair."})
            }
        }

    def _analyze_candidate_skin(self, img_rgb_u8: np.ndarray) -> dict:
        """
        Refined Candidate Color Filter in YCrCb and Lab color spaces.
        Tightened to exclude beige/wood walls (low sat, low a*) and graphic garment prints (hyper sat/red).
        """
        h, w = img_rgb_u8.shape[:2]
        y1, y2 = int(h * 0.12), int(h * 0.65)
        x1, x2 = int(w * 0.18), int(w * 0.82)
        roi = img_rgb_u8[y1:y2, x1:x2]
        n_roi_pixels = (y2 - y1) * (x2 - x1)

        ycrcb = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb)
        lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB).astype(np.float32)
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV).astype(np.float32)

        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        L = lab[:, :, 0] * 100.0 / 255.0
        a = lab[:, :, 1] - 128.0
        b = lab[:, :, 2] - 128.0
        sat = hsv[:, :, 1] / 255.0 * 100.0

        # Refined Candidate Human Skin Locus:
        # 1. Cr in [133..180], Cb in [77..135], (Cr - Cb) >= 10
        # 2. a* in [3.5..30.0] (rejects beige/grey walls a* < 3.5 AND hyper-red garment prints a* > 30)
        # 3. Saturation in [14.0..55.0]% (rejects low-saturation walls < 14% AND cartoon prints > 55%)
        # 4. L* in [28.0..88.0] (rejects black hair < 28 and blown white shirts > 88)
        roi_mask = (
            (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 135) &
            (cr.astype(int) - cb.astype(int) >= 10) &
            (a >= 3.5) & (a <= 30.0) &
            (sat >= 14.0) & (sat <= 55.0) &
            (L >= 28.0) & (L <= 88.0)
        )
        sample_pixel_count = int(np.sum(roi_mask))
        sample_fraction = float(sample_pixel_count) / float(max(1, n_roi_pixels))

        full_mask = np.zeros((h, w), dtype=bool)
        full_mask[y1:y2, x1:x2] = roi_mask

        is_valid = (sample_pixel_count >= 300) and (sample_fraction >= 0.015)

        if is_valid:
            skin_L = L[roi_mask]
            skin_a = a[roi_mask]
            skin_b = b[roi_mask]
            skin_sat = sat[roi_mask]

            return {
                "valid": True,
                "reason": "Sufficient candidate skin pixels within refined locus",
                "sample_pixel_count": sample_pixel_count,
                "sample_fraction": round(sample_fraction, 4),
                "sampling_mask": full_mask,
                "roi_box": (y1, y2, x1, x2),
                "L_med": float(np.median(skin_L)),
                "L_p95": float(np.percentile(skin_L, 95)),
                "a_med": float(np.median(skin_a)),
                "b_med": float(np.median(skin_b)),
                "sat_med": float(np.median(skin_sat)),
                "skin_rgb": [round(float(x), 1) for x in np.median(roi[roi_mask], axis=0)]
            }
        else:
            return {
                "valid": False,
                "reason": f"Insufficient candidate skin samples (count={sample_pixel_count} < 300 or fraction={sample_fraction:.3f} < 0.015)",
                "sample_pixel_count": sample_pixel_count,
                "sample_fraction": round(sample_fraction, 4),
                "sampling_mask": full_mask,
                "roi_box": (y1, y2, x1, x2),
                "L_med": None,
                "L_p95": None,
                "a_med": None,
                "b_med": None,
                "sat_med": None,
                "skin_rgb": None
            }

    def _interpolate_params(self, s: float, contrast_target: float, brightness_target: float,
                            saturation_target: float, rm_target: float, rs_target: float, rh_target: float,
                            gm_target: float, gs_target: float, bm_target: float, bs_target: float,
                            unsharp_target: float, noise_target: float) -> dict:
        """Linear interpolation from neutral state (s in [0, 1])"""
        s = float(np.clip(s, 0.0, 1.0))
        return {
            "contrast": 1.0 + s * (contrast_target - 1.0),
            "saturation": 1.0 + s * (saturation_target - 1.0),
            "brightness": s * brightness_target,
            "rm": s * rm_target,
            "rs": s * rs_target,
            "rh": s * rh_target,
            "gm": s * gm_target,
            "gs": s * gs_target,
            "bm": s * bm_target,
            "bs": s * bs_target,
            "unsharp": s * unsharp_target,
            "noise": s * noise_target
        }

    def _build_ffmpeg_filter_string(self, p: dict) -> str:
        """Constructs FFmpeg filter chain from parameter dictionary"""
        filter_parts = []
        if round(p["contrast"], 3) != 1.0 or round(p["brightness"], 3) != 0.0 or round(p["saturation"], 3) != 1.0:
            filter_parts.append(f"eq=contrast={p['contrast']:.3f}:brightness={p['brightness']:.3f}:saturation={p['saturation']:.3f}")

        if (abs(p["rm"]) > 1e-4 or abs(p["rs"]) > 1e-4 or abs(p["rh"]) > 1e-4 or
            abs(p["gm"]) > 1e-4 or abs(p["gs"]) > 1e-4 or
            abs(p["bm"]) > 1e-4 or abs(p["bs"]) > 1e-4):
            cb_str = f"rs={p['rs']:.3f}:rm={p['rm']:.3f}:rh={p['rh']:.3f}:gs={p['gs']:.3f}:gm={p['gm']:.3f}:bs={p['bs']:.3f}:bm={p['bm']:.3f}"
            filter_parts.append(f"colorbalance={cb_str}")

        if p["unsharp"] > 0.005:
            filter_parts.append(f"unsharp=3:3:{p['unsharp']:.3f}:3:3:0.0")

        if p["noise"] > 0.05:
            filter_parts.append(f"noise=c0s={p['noise']:.1f}:c1s={p['noise']*0.55:.1f}:c2s={p['noise']*0.55:.1f}:allf=t+u")

        return ",".join(filter_parts)

    def _run_ffmpeg_pipe(self, frames_rgb_u8: np.ndarray, vf_str: str) -> tuple:
        """Runs exact in-memory FFmpeg C-libavfilter pipe"""
        if not vf_str or vf_str.strip() == "":
            return frames_rgb_u8.copy(), 0.0, None

        b, h, w, c = frames_rgb_u8.shape
        ffmpeg_bin = get_ffmpeg_binary()
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
            "-r", "24", "-i", "pipe:0",
            "-vf", vf_str,
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
        ]

        t0 = time.time()
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            raw_bytes = frames_rgb_u8.tobytes()
            try:
                out_bytes, err = p.communicate(raw_bytes, timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.communicate()
                return None, time.time() - t0, "FFmpeg pipe timed out after 30s"
            elapsed = time.time() - t0
            if p.returncode == 0 and len(out_bytes) == len(raw_bytes):
                res = np.frombuffer(out_bytes, dtype=np.uint8).reshape((b, h, w, c))
                return res, elapsed, None
            else:
                err_msg = err.decode('utf-8', errors='ignore') if err else f"returncode {p.returncode}"
                return None, elapsed, err_msg
        except Exception as ex:
            return None, time.time() - t0, str(ex)

    def apply_grade(self, images, reference_image, mode="reference_safe", strength=1.0,
                    unsharp_amount=0.0, sensor_noise=0.0, enable=True,
                    candidate_selection=True, red_bias=0.0, max_contrast=1.10, max_saturation=1.28,
                    adaptive_tier_mode=True):
        t_start = time.time()

        is_torch = torch is not None and isinstance(images, torch.Tensor)
        device = images.device if is_torch else "cpu"

        # Requirement 2: Strength 0 or disabled returns exact input tensor with 0 processing
        if not enable or strength <= 0.0:
            empty_preview = images[0:1] if (is_torch or isinstance(images, np.ndarray)) else images
            report = json.dumps({
                "status": "BYPASS",
                "backend": "none",
                "mode": mode,
                "reason": "Node disabled or strength is 0.0",
                "total_time_ms": round((time.time() - t_start) * 1000.0, 2)
            }, indent=2)
            return (images, empty_preview, report)

        # 1. Extract ONLY single reference frame & 9 sample frames first (avoids ~8-15 GB full-batch RAM spike on bypass!)
        b = int(images.shape[0])
        if is_torch:
            ref_t = reference_image[0] if reference_image.ndim == 4 else reference_image
            ref_frame = ref_t.detach().cpu().numpy() if ref_t.dtype == torch.uint8 else (ref_t.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        else:
            ref_arr = np.asarray(reference_image)
            ref_arr = ref_arr[0] if ref_arr.ndim == 4 else ref_arr
            ref_frame = ref_arr if ref_arr.dtype == np.uint8 else np.clip(ref_arr * 255.0, 0, 255).astype(np.uint8)

        h, w, c = int(images.shape[1]), int(images.shape[2]), int(images.shape[3])

        def _extract_frame_u8(idx: int) -> np.ndarray:
            f = images[idx]
            if is_torch:
                arr_f = f.detach().cpu().numpy()
                return arr_f if f.dtype == torch.uint8 else (arr_f * 255.0).clip(0, 255).astype(np.uint8)
            arr_f = np.asarray(f)
            return arr_f if arr_f.dtype == np.uint8 else np.clip(arr_f * 255.0, 0, 255).astype(np.uint8)

        # Adaptive Tier 1 Check: Auto-protect silky hair if reference is sharp
        if adaptive_tier_mode:
            ref_h, ref_w = ref_frame.shape[:2]
            face_roi = ref_frame[int(ref_h * 0.08):int(ref_h * 0.35), int(ref_w * 0.30):int(ref_w * 0.70)]
            if face_roi.size > 0:
                face_lap = float(cv2.Laplacian(cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
            else:
                face_lap = float(cv2.Laplacian(cv2.cvtColor(ref_frame, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())

            if face_lap >= 100.0:
                sensor_noise = 0.0
                unsharp_amount = 0.0
                print(f"[AutoReferenceGrade] TIER 1 Sharp Reference (Face IQA={face_lap:.1f} >= 100). Auto-zeroed sensor_noise & unsharp for silky hair!")

        # 2. Analyze Reference Image (without stretching or distorting aspect ratio)
        ref_analysis = self._analyze_candidate_skin(ref_frame)
        if not ref_analysis["valid"] and mode == "reference_safe":
            empty_preview = images[0:1]
            report = json.dumps({
                "status": "BYPASS",
                "backend": "none",
                "mode": mode,
                "reason": f"Reference image candidate skin analysis invalid: {ref_analysis['reason']}",
                "total_time_ms": round((time.time() - t_start) * 1000.0, 2)
            }, indent=2)
            return (images, empty_preview, report)

        # 3. Multi-frame Sampling across Clip (9 evenly spaced frames)
        sample_indices = np.unique(np.linspace(0, b - 1, min(9, b), dtype=int))
        sampled_frames_dict = {int(idx): _extract_frame_u8(int(idx)) for idx in sample_indices}
        frame_samples = []
        for idx in sample_indices:
            analysis = self._analyze_candidate_skin(sampled_frames_dict[int(idx)])
            frame_samples.append((int(idx), analysis))

        valid_samples = [s for s in frame_samples if s[1]["valid"]]
        valid_count = len(valid_samples)

        # Requirement 5: Need at least 5 valid frames
        if valid_count < 5 and mode == "reference_safe":
            empty_preview = images[0:1]
            report = json.dumps({
                "status": "BYPASS",
                "backend": "none",
                "mode": mode,
                "reason": f"Insufficient valid candidate skin frames ({valid_count}/9 valid, minimum 5 required)",
                "sample_indices": [int(x) for x in sample_indices],
                "valid_count": valid_count,
                "total_time_ms": round((time.time() - t_start) * 1000.0, 2)
            }, indent=2)
            return (images, empty_preview, report)

        # Median aggregate of valid frames
        if valid_count > 0:
            wan_L_med = float(np.median([s[1]["L_med"] for s in valid_samples]))
            wan_a_med = float(np.median([s[1]["a_med"] for s in valid_samples]))
            wan_b_med = float(np.median([s[1]["b_med"] for s in valid_samples]))
            wan_sat_med = float(np.median([s[1]["sat_med"] for s in valid_samples]))
            wan_L_p95 = float(np.median([s[1]["L_p95"] for s in valid_samples]))
        else:
            wan_L_med, wan_a_med, wan_b_med, wan_sat_med, wan_L_p95 = 65.0, 12.0, 8.0, 22.0, 80.0

        ref_a = ref_analysis["a_med"] if ref_analysis["valid"] else 12.0
        ref_b = ref_analysis["b_med"] if ref_analysis["valid"] else 8.0
        ref_sat = ref_analysis["sat_med"] if ref_analysis["valid"] else 22.0

        delta_a = ref_a - wan_a_med
        delta_b = ref_b - wan_b_med
        sat_ratio = ref_sat / max(wan_sat_med, 1.0)

        # 4. Mode-specific Target Derivation
        if mode == "reference_safe":
            contrast_target = 1.0
            brightness_target = 0.0
            unsharp_target = 0.0
            noise_target = 0.0

            # Delta mapping within [-0.03, 0.03]
            rm_target = float(np.clip(delta_a * 0.0035 + red_bias, -0.03, 0.03))
            bm_target = float(np.clip(-delta_b * 0.0035, -0.03, 0.03))
            gm_target = float(np.clip(-delta_a * 0.0008, -0.01, 0.01))
            rs_target = round(rm_target * 0.3, 3)
            rh_target = round(rm_target * 0.3, 3)
            bs_target = round(bm_target * 0.3, 3)
            gs_target = round(gm_target * 0.3, 3)

            # Saturation within [0.90, 1.10]
            saturation_target = float(np.clip(sat_ratio, 0.90, 1.10))

        else: # mode == "studio_warm"
            rm_target = float(np.clip(
                0.022 + red_bias + max(0.0, delta_a) * 0.006 + max(0.0, (ref_a - 10.0)) * 0.0035 - max(0.0, -delta_a) * 0.005,
                0.010, 0.045
            ))
            bm_target = float(np.clip(-0.010 - (delta_b / 15.0) * 0.010, -0.022, 0.002))
            rs_target = round(rm_target * 0.40, 3)
            rh_target = round(rm_target * 0.35, 3)
            bs_target = round(bm_target * 0.50, 3)
            gm_target = float(np.clip(delta_b * 0.03, -0.006, 0.006))
            gs_target = round(gm_target * 0.40, 3)

            saturation_target = float(np.clip(
                1.18 + max(0.0, (sat_ratio - 1.0)) * 0.25 + max(0.0, (ref_a - 10.0)) * 0.02,
                1.15, max_saturation
            ))

            if wan_L_p95 >= 81.0:
                contrast_target = 1.010
                brightness_target = -0.005
            elif wan_L_p95 <= 68.0:
                contrast_target = 1.065
                brightness_target = 0.012
            else:
                contrast_target = 1.040
                brightness_target = 0.005
            contrast_target = float(np.clip(contrast_target, 1.00, max_contrast))

            unsharp_target = unsharp_amount
            noise_target = sensor_noise

        # 5. Exact FFmpeg Candidate Scoring & Real Background/Clipping Rejection
        level_scores = {}
        selected_level = 1.0
        level_decision_reason = "Manual strength applied"

        if mode == "reference_safe" and candidate_selection and len(valid_samples) >= 5:
            # Gather sampled frames into array
            valid_indices = [s[0] for s in valid_samples]
            valid_masks = [s[1]["sampling_mask"] for s in valid_samples]
            sample_frames_stack = np.stack([sampled_frames_dict[int(idx)] for idx in valid_indices], axis=0)
            n_samples = len(valid_indices)

            candidates = [0.0, 0.5, 1.0]
            MAX_BG_SHIFT = 2.0
            MAX_NEW_CLIPPING_RATIO = 0.008 # max 0.8% new clipped pixels on skin

            for lvl in candidates:
                p_lvl = self._interpolate_params(
                    lvl, contrast_target, brightness_target, saturation_target,
                    rm_target, rs_target, rh_target, gm_target, gs_target, bm_target, bs_target,
                    unsharp_target, noise_target
                )
                vf_lvl = self._build_ffmpeg_filter_string(p_lvl)

                # Execute EXACT FFmpeg on the sampled batch (bit-exact output matching)
                pipe_out, pipe_t, pipe_err = self._run_ffmpeg_pipe(sample_frames_stack, vf_lvl)
                if pipe_out is None:
                    # FFmpeg failure on sample frames
                    level_scores[str(lvl)] = {
                        "disqualified": True,
                        "reject_reason": f"FFmpeg error: {pipe_err}"
                    }
                    continue

                chroma_dists = []
                bg_shifts = []
                new_clipping_counts = []
                total_skin_counts = []

                for j in range(n_samples):
                    f_orig = sample_frames_stack[j]
                    f_trans = pipe_out[j]
                    mask = valid_masks[j]

                    # 1. Chroma distance under input mask
                    lab_trans = cv2.cvtColor(f_trans, cv2.COLOR_RGB2LAB).astype(np.float32)
                    a_trans = lab_trans[:, :, 1] - 128.0
                    b_trans = lab_trans[:, :, 2] - 128.0
                    a_val = float(np.median(a_trans[mask]))
                    b_val = float(np.median(b_trans[mask]))
                    dist = math.sqrt((a_val - ref_a) ** 2 + (b_val - ref_b) ** 2)
                    chroma_dists.append(dist)

                    # 2. Background shift check (non-skin pixels in upper half)
                    bg_mask = (~mask) & (np.arange(h)[:, None] < int(h * 0.5))
                    if np.sum(bg_mask) > 1000:
                        bg_diff = float(np.mean(np.abs(f_trans[bg_mask].astype(float) - f_orig[bg_mask].astype(float))))
                        bg_shifts.append(bg_diff)

                    # 3. Clipping check on skin
                    skin_orig = f_orig[mask]
                    skin_trans = f_trans[mask]
                    n_skin = len(skin_orig)
                    total_skin_counts.append(n_skin)
                    if n_skin > 0:
                        orig_clipped = np.sum((np.max(skin_orig, axis=1) >= 253) | (np.min(skin_orig, axis=1) <= 5))
                        trans_clipped = np.sum((np.max(skin_trans, axis=1) >= 253) | (np.min(skin_trans, axis=1) <= 5))
                        new_clipping_counts.append(max(0, trans_clipped - orig_clipped))

                mean_dist = float(np.mean(chroma_dists))
                mean_bg = float(np.mean(bg_shifts)) if bg_shifts else 0.0
                tot_skin = sum(total_skin_counts)
                clip_ratio = (sum(new_clipping_counts) / tot_skin) if tot_skin > 0 else 0.0

                disqualified = False
                reject_reason = "PASS"

                # Real rejection criteria:
                if lvl > 0.0:
                    if mean_bg > MAX_BG_SHIFT:
                        disqualified = True
                        reject_reason = f"Background shift {mean_bg:.2f} exceeds safety threshold {MAX_BG_SHIFT}"
                    elif clip_ratio > MAX_NEW_CLIPPING_RATIO:
                        disqualified = True
                        reject_reason = f"New skin clipping ratio {clip_ratio*100:.2f}% exceeds threshold {MAX_NEW_CLIPPING_RATIO*100:.1f}%"

                level_scores[str(lvl)] = {
                    "mean_chroma_dist": round(mean_dist, 3),
                    "mean_bg_shift": round(mean_bg, 2),
                    "skin_clipping_ratio": round(clip_ratio, 4),
                    "disqualified": disqualified,
                    "rejection_status": reject_reason
                }

            # Decision Logic with Active Rejection & Safeguards:
            d0 = level_scores["0.0"]["mean_chroma_dist"]
            s05 = level_scores.get("0.5", {})
            s10 = level_scores.get("1.0", {})

            pass_05 = (not s05.get("disqualified", True)) and ((d0 - s05.get("mean_chroma_dist", 999.0)) >= 0.30)
            pass_10 = (not s10.get("disqualified", True)) and ((d0 - s10.get("mean_chroma_dist", 999.0)) >= 0.30)

            if not pass_05 and not pass_10:
                selected_level = 0.0
                reasons = []
                if s05.get("disqualified"): reasons.append(f"0.5 rejected ({s05.get('rejection_status')})")
                if s10.get("disqualified"): reasons.append(f"1.0 rejected ({s10.get('rejection_status')})")
                if not reasons: reasons.append("No level improved chroma by >= 0.30")
                level_decision_reason = f"Fallback to 0.0 (Bypass): {'; '.join(reasons)}"
            elif pass_05 and not pass_10:
                selected_level = 0.5
                level_decision_reason = f"Level 0.5 selected (Level 1.0 was disqualified: {s10.get('rejection_status')})"
            elif pass_10 and not pass_05:
                selected_level = 1.0
                level_decision_reason = f"Level 1.0 selected (Level 0.5 was disqualified: {s05.get('rejection_status')})"
            else:
                # Both passed! Compare distances
                d05 = s05["mean_chroma_dist"]
                d10 = s10["mean_chroma_dist"]
                if abs(d05 - d10) <= 0.25:
                    selected_level = 0.5
                    level_decision_reason = f"Levels 0.5 and 1.0 yield equivalent chroma (|{d05:.2f} - {d10:.2f}| <= 0.25) -> Lower level 0.5 selected for safety"
                elif d10 < d05 - 0.25:
                    selected_level = 1.0
                    level_decision_reason = f"Level 1.0 yields best improvement (dist={d10:.2f} vs {d05:.2f}) -> Level 1.0 selected"
                else:
                    selected_level = 0.5
                    level_decision_reason = f"Level 0.5 yields best balance (dist={d05:.2f}) -> Level 0.5 selected"

        # Effective strength computation
        effective_s = float(np.clip(strength * selected_level, 0.0, 1.0))

        if effective_s <= 0.0:
            empty_preview = images[0:1]
            report = json.dumps({
                "status": "BYPASS",
                "backend": "none",
                "mode": mode,
                "reason": f"Candidate verification chose level 0.0: {level_decision_reason}",
                "level_scores": level_scores,
                "selected_level": selected_level,
                "total_time_ms": round((time.time() - t_start) * 1000.0, 2)
            }, indent=2)
            return (images, empty_preview, report)

        # Final interpolated parameters
        final_params = self._interpolate_params(
            effective_s, contrast_target, brightness_target, saturation_target,
            rm_target, rs_target, rh_target, gm_target, gs_target, bm_target, bs_target,
            unsharp_target, noise_target
        )

        vf = self._build_ffmpeg_filter_string(final_params)

        # 6. Execute FFmpeg In-Memory Pipe in 32-Frame Chunks (prevents ~15 GB RAM spike on 15s/20s videos)
        chunk_size = 32
        t_ffmpeg = 0.0
        ffmpeg_error = None
        backend = "ffmpeg"
        out_tensor = torch.empty_like(images) if is_torch else np.empty_like(images, dtype=np.float32)
        center_idx = b // 2
        center_orig = _extract_frame_u8(center_idx)
        center_proc = center_orig.copy()

        for start_i in range(0, b, chunk_size):
            end_i = min(b, start_i + chunk_size)
            chunk_t = images[start_i:end_i]
            if is_torch:
                chunk_np = chunk_t.detach().cpu().numpy() if chunk_t.dtype == torch.uint8 else (chunk_t.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr_c = np.asarray(chunk_t)
                chunk_np = arr_c if arr_c.dtype == np.uint8 else np.clip(arr_c * 255.0, 0, 255).astype(np.uint8)

            proc_chunk, t_sub, err_sub = self._run_ffmpeg_pipe(chunk_np, vf)
            t_ffmpeg += t_sub
            if proc_chunk is None:
                ffmpeg_error = err_sub
                break

            if start_i <= center_idx < end_i:
                center_proc = proc_chunk[center_idx - start_i].copy()

            if is_torch:
                out_tensor[start_i:end_i] = torch.from_numpy(proc_chunk.astype(np.float32) / 255.0).to(device)
            else:
                out_tensor[start_i:end_i] = proc_chunk.astype(np.float32) / 255.0

        # Requirement 7: Transparent error reporting (no silent python substitution reported as ffmpeg)
        if ffmpeg_error is not None:
            if mode == "reference_safe":
                empty_preview = images[0:1]
                report = json.dumps({
                    "status": "ERROR_BYPASS",
                    "backend": "none",
                    "error": ffmpeg_error,
                    "applied_filter": vf,
                    "total_time_ms": round((time.time() - t_start) * 1000.0, 2)
                }, indent=2)
                return (images, empty_preview, report)
            else:
                backend = "error_passthrough"
                out_tensor = images

        # 7. Build 4-Panel Verification & Candidate Mask Inspection Overlay Preview
        center_analysis = self._analyze_candidate_skin(center_orig)

        h_prev, w_prev = 512, 288
        p_ref = cv2.resize(ref_frame, (w_prev, h_prev))
        p_raw = cv2.resize(center_orig, (w_prev, h_prev))
        p_out = cv2.resize(center_proc, (w_prev, h_prev))

        # Build mask overlay on raw frame
        overlay_raw = center_orig.copy()
        if center_analysis["sampling_mask"] is not None:
            c_mask = center_analysis["sampling_mask"]
            overlay_raw[c_mask] = (overlay_raw[c_mask] * 0.4 + np.array([0, 255, 100]) * 0.6).astype(np.uint8)
            y1, y2, x1, x2 = center_analysis["roi_box"]
            cv2.rectangle(overlay_raw, (x1, y1), (x2, y2), (255, 255, 0), 2)

        p_mask = cv2.resize(overlay_raw, (w_prev, h_prev))

        composite = np.hstack([p_ref, p_raw, p_mask, p_out])
        cv2.putText(composite, "1. REF", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(composite, "2. RAW", (w_prev + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
        cv2.putText(composite, "3. CANDIDATE MASK", (2 * w_prev + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(composite, f"4. OUT ({mode})", (3 * w_prev + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)

        preview_np = np.expand_dims(composite.astype(np.float32) / 255.0, axis=0)
        preview_tensor = torch.from_numpy(preview_np).to(device) if is_torch else preview_np

        total_time_ms = round((time.time() - t_start) * 1000.0, 2)
        ffmpeg_time_ms = round(t_ffmpeg * 1000.0, 2)

        report_dict = {
            "status": "SUCCESS",
            "version": "V3.2 Production (Active Rejection & Exact FFmpeg Evaluation)",
            "mode": mode,
            "backend": backend,
            "effective_strength": round(effective_s, 3),
            "candidate_selection": {
                "enabled": candidate_selection and mode == "reference_safe",
                "selected_level": selected_level,
                "decision_reason": level_decision_reason,
                "candidate_scores": level_scores
            },
            "reference_skin_status": {
                "valid": ref_analysis["valid"],
                "reason": ref_analysis["reason"],
                "sample_pixel_count": ref_analysis["sample_pixel_count"],
                "sample_fraction": ref_analysis["sample_fraction"],
                "chroma_a": round(ref_a, 2),
                "chroma_b": round(ref_b, 2),
                "saturation": round(ref_sat, 1)
            },
            "video_sampling_status": {
                "total_frames": b,
                "sampled_frames_count": len(sample_indices),
                "valid_frames_count": valid_count,
                "median_L": round(wan_L_med, 2),
                "median_a": round(wan_a_med, 2),
                "median_b": round(wan_b_med, 2),
                "median_saturation": round(wan_sat_med, 1),
                "median_L_p95": round(wan_L_p95, 2)
            },
            "chroma_delta": {
                "delta_a": round(delta_a, 2),
                "delta_b": round(delta_b, 2),
                "sat_ratio": round(sat_ratio, 3)
            },
            "applied_filter_params": {
                "contrast": round(final_params["contrast"], 3),
                "brightness": round(final_params["brightness"], 3),
                "saturation": round(final_params["saturation"], 3),
                "rm": round(final_params["rm"], 3),
                "rs": round(final_params["rs"], 3),
                "rh": round(final_params["rh"], 3),
                "gm": round(final_params["gm"], 3),
                "gs": round(final_params["gs"], 3),
                "bm": round(final_params["bm"], 3),
                "bs": round(final_params["bs"], 3),
                "unsharp": round(final_params["unsharp"], 3),
                "noise": round(final_params["noise"], 2)
            },
            "ffmpeg_filter_string": vf,
            "timing": {
                "ffmpeg_time_ms": ffmpeg_time_ms,
                "total_node_time_ms": total_time_ms
            }
        }

        return (out_tensor, preview_tensor, json.dumps(report_dict, indent=2))

NODE_CLASS_MAPPINGS = {
    "AutoReferenceGrade": AutoReferenceGrade
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoReferenceGrade": "Auto Reference Grade (Adaptive Safeguard V3.2)"
}
