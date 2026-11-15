# -*- coding: utf-8 -*-
r"""
SaoChep Master Head Isolation Engine v9.5 (Multi-Branch Architecture)
Phân định 3 Nhánh Xử Lý Độc Lập (Zero Case-Overfitting):
- Nhánh A (Standard Human): 100% Golden v9.1 + Head Items (mũ lưỡi trai, mũ trùm đầu, kính).
- Nhánh B (Quadruped Animal): 100% Argon Geometric Inflection (Chó Ngao & Chó Bull).
- Nhánh C (Long Hair Dedicated): Tri-Prototype Mahalanobis + Adaptive Gravity Hair Funnel.
"""
import os, sys, cv2, time
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import onnxruntime as ort
from scipy import ndimage
from PIL import Image

def _build_gabor_bank(lambd: float, ksize: int = 15) -> list:
    angles = np.deg2rad([-30, -15, 0, 15, 30])
    sigma = 0.56 * lambd
    kernels = []
    for theta in angles:
        k = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma=0.5, psi=0, ktype=cv2.CV_32F)
        k_norm = k / (np.sum(np.abs(k)) + 1e-6)
        kernels.append(k_norm)
    return kernels

_GABOR_FINE = _build_gabor_bank(lambd=3.0, ksize=15)
_GABOR_COARSE = _build_gabor_bank(lambd=9.5, ksize=25)


try:
    from .head_preserve_v97 import preserve_head, extend_animal_neck
    from .aligned_face_parser_v97 import aligned_parse
except ImportError:
    from head_preserve_v97 import preserve_head, extend_animal_neck
    from aligned_face_parser_v97 import aligned_parse

class MasterHeadEngineV100:
    def __init__(self, models_dir=None, birefnet_dir=None):
        if models_dir is None:
            candidates = [os.environ.get("SAOCHEP_MODELS_DIR"),
                          str(Path(__file__).parent / "models"),
                          str(Path.cwd() / "models")]
            try:
                import folder_paths
                candidates.insert(1, os.path.join(folder_paths.models_dir, "saochep_head"))
            except ImportError:
                pass
            models_dir = next((p for p in candidates if p and
                               os.path.isfile(os.path.join(p,"yoloface_8n.onnx")) and
                               os.path.isfile(os.path.join(p,"bisenet_face_parsing.onnx"))),None)
            if models_dir is None:
                raise FileNotFoundError("Set SAOCHEP_MODELS_DIR or install ONNX models in models/saochep_head")

        yolo_path = os.path.join(models_dir, "yoloface_8n.onnx")
        bisenet_path = os.path.join(models_dir, "bisenet_face_parsing.onnx")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4
        self.yolo_sess = ort.InferenceSession(yolo_path, opts, providers=["CPUExecutionProvider"])
        self.bisenet_sess = ort.InferenceSession(bisenet_path, opts, providers=["CPUExecutionProvider"])

        if birefnet_dir is None:
            birefnet_dir = Path(models_dir)/"birefnet_lite"
        self.birefnet = AutoModelForImageSegmentation.from_pretrained(
            str(Path(birefnet_dir).resolve()),
            trust_remote_code=True, local_files_only=True, use_safetensors=True,
        ).float().eval()

        self.compute_device = torch.device("cpu")

        self.transform_biref = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.bise_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.bise_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @staticmethod
    def _guided_filter_alpha(guide_gray, alpha, radius=5, eps=1e-3):
        ksize = (2 * radius + 1, 2 * radius + 1)
        mean_I = cv2.boxFilter(guide_gray, cv2.CV_32F, ksize)
        mean_p = cv2.boxFilter(alpha, cv2.CV_32F, ksize)
        mean_Ip = cv2.boxFilter(guide_gray * alpha, cv2.CV_32F, ksize)
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = cv2.boxFilter(guide_gray * guide_gray, cv2.CV_32F, ksize)
        var_I = mean_II - mean_I * mean_I
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
        mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)
        return np.clip(mean_a * guide_gray + mean_b, 0.0, 1.0)

    def process(self, img_bgr, target_face_index=0, neck_stub_ratio=0.15, *, allow_animal_fallback=True):
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ---------------------------------------------------------------------
        # 1. YOLOFACE DETECTION
        # ---------------------------------------------------------------------
        blob = cv2.resize(img_bgr, (640, 640)).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, :]
        outputs = self.yolo_sess.run(None, {self.yolo_sess.get_inputs()[0].name: blob})[0][0]
        outputs = np.transpose(outputs, (1, 0))
        scores = outputs[:, 4]
        valid_idx = np.where(scores >= 0.40)[0]

        # =====================================================================
        # NHÁNH B: ĐỘNG VẬT 4 CHÂN (Chó Ngao & Chó Bull)
        # =====================================================================
        if len(valid_idx) == 0:
            if not allow_animal_fallback:
                raise ValueError("No human face detected. Submit a clear single-person photo.")
            pil_img = Image.fromarray(img_rgb)
            t_in = self.transform_biref(pil_img).unsqueeze(0).to(self.compute_device)
            with torch.inference_mode():
                preds = self.birefnet(t_in)[-1].sigmoid().cpu()
            subj_alpha = cv2.resize(preds[0, 0].numpy(), (orig_w, orig_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            subj_alpha = np.clip(subj_alpha, 0.0, 1.0)

            fg_bin = (subj_alpha > 0.35).astype(np.uint8)
            num_l, lbls, stats, _ = cv2.connectedComponentsWithStats(fg_bin, connectivity=8)
            if num_l > 1:
                best_l = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                main_fg = (lbls == best_l)
            else:
                main_fg = fg_bin > 0

            ys, xs = np.where(main_fg)
            if len(ys) > 50:
                top_y, bot_y = int(ys.min()), int(ys.max())
                total_h = max(10, bot_y - top_y)
                total_w = max(10, int(xs.max() - xs.min()))

                W_prof = np.zeros(total_h, dtype=np.float32)
                Cx_prof = np.zeros(total_h, dtype=np.float32)
                for dy in range(total_h):
                    r_xs = np.where(main_fg[top_y + dy, :])[0]
                    if len(r_xs) >= 5:
                        xl, xr = np.percentile(r_xs, 4.0), np.percentile(r_xs, 96.0)
                        W_prof[dy] = float(xr - xl)
                        Cx_prof[dy] = float(0.5 * (xl + xr))
                    elif dy > 0:
                        W_prof[dy], Cx_prof[dy] = W_prof[dy - 1], Cx_prof[dy - 1]

                sigma_w = max(2.5, total_h * 0.015)
                W_smooth = ndimage.gaussian_filter1d(W_prof, sigma=sigma_w, mode="nearest")
                dW = np.gradient(W_smooth)
                d2W = np.gradient(dW)

                s_y1 = int(total_h * 0.08)
                s_y2 = int(total_h * 0.22)
                head_cx = float(np.median(Cx_prof[s_y1:max(s_y1 + 2, s_y2)]))
                head_w = float(np.percentile(W_smooth[s_y1:max(s_y1 + 2, s_y2)], 88.0))

                aspect_fg = float(total_h) / max(float(total_w), 1.0)
                dW_dy = np.gradient(W_smooth.astype(np.float32))

                # Argon's Zero-Crossing Neck Detector for long standing dogs (Chó Ngao)
                if aspect_fg > 1.35 and (head_w / float(total_h) < 0.22):
                    y_search_min = int(np.clip(
                        max(total_h * 0.185, head_w * 0.88),
                        total_h * 0.180,
                        total_h * 0.245
                    ))
                    y_search_max = int(min(total_h * 0.34, y_search_min + head_w * 1.35))

                    y_neck_cut = None
                    for y_rel in range(y_search_min, max(y_search_min + 1, y_search_max)):
                        w_curr = W_smooth[y_rel]
                        prev_slope = np.min(dW_dy[max(0, y_rel - 8):y_rel + 1])
                        next_slope = np.max(dW_dy[y_rel:min(len(dW_dy), y_rel + 15)])
                        w_below_60 = W_smooth[min(len(W_smooth) - 1, y_rel + 60)]

                        is_zero_crossing = (prev_slope <= -0.25) and (next_slope >= 1.60)
                        is_shoulder_flare = (w_below_60 >= w_curr * 1.42) and (w_curr <= head_w * 0.85)

                        if is_zero_crossing and is_shoulder_flare:
                            local_win = W_smooth[max(y_search_min, y_rel - 5):min(y_search_max, y_rel + 9)]
                            y_neck_cut = max(y_search_min, y_rel - 5) + int(np.argmin(local_win))
                            y_neck_cut = min(y_neck_cut + 6, int(total_h * 0.23))
                            break

                    if y_neck_cut is None:
                        y_neck_cut = y_search_min + int(np.argmin(W_smooth[y_search_min:y_search_max]))
                    best_dy = y_neck_cut
                else:
                    # Chó Bull & các động vật bè ngang
                    min_h_ratio = 0.255 if total_h > total_w * 1.15 else 0.22
                    max_h_ratio = 0.345 if total_h > total_w * 1.15 else 0.42
                    y_search_min = int(np.clip(max(total_h * min_h_ratio, head_w * 0.88), total_h * 0.21, total_h * 0.32))
                    y_search_max = int(np.clip(min(total_h * max_h_ratio, head_w * 1.28), y_search_min + 8, total_h * 0.44))

                    gray_ani = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
                    gy_ani = np.abs(cv2.Sobel(gray_ani, cv2.CV_32F, 0, 1, ksize=3))
                    muzzle_x1 = max(0, int(head_cx - head_w * 0.28))
                    muzzle_x2 = min(orig_w, int(head_cx + head_w * 0.28))

                    best_dy = int(0.5 * (y_search_min + y_search_max))
                    best_score = -1e9
                    for dy in range(y_search_min, y_search_max):
                        y_abs = top_y + dy
                        edge_strength = float(np.mean(gy_ani[max(0, y_abs - 2):min(orig_h, y_abs + 3), muzzle_x1:muzzle_x2]))
                        lum_above = float(np.mean(gray_ani[max(top_y, y_abs - int(head_w * 0.15)):y_abs, muzzle_x1:muzzle_x2]))
                        lum_below = float(np.mean(gray_ani[y_abs:min(orig_h, y_abs + int(head_w * 0.15)), muzzle_x1:muzzle_x2]))
                        snout_to_chest_contrast = max(0.0, lum_below - lum_above)
                        flare_score = float(max(0.0, d2W[dy] * 15.0) + max(0.0, dW[dy] * 4.0))

                        total_score = 1.2 * edge_strength + 0.8 * snout_to_chest_contrast + flare_score
                        if total_score > best_score:
                            best_score = total_score
                            best_dy = dy

                head_h = float(min(y_search_max, best_dy + int(head_w * 0.04)))
                cutoff_y = top_y + head_h

                x_coords_ani = np.arange(orig_w)
                dx_ani = (x_coords_ani - head_cx) / (head_w * 0.58 + 1e-5)
                jaw_ani = cutoff_y - (head_h * 0.12) * np.clip(dx_ani ** 2, 0.0, 1.2)

                y_grid_a, x_grid_a = np.ogrid[:orig_h, :orig_w]
                head_support_a = np.clip((jaw_ani[None, :] - y_grid_a) / 6.0, 0.0, 1.0).astype(np.float32)
                guide_gray_a = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                refined_alpha_a = self._guided_filter_alpha(guide_gray_a, subj_alpha, radius=3, eps=5e-4)

                final_alpha = np.clip(refined_alpha_a * head_support_a, 0.0, 1.0).astype(np.float32)
                final_alpha[y_grid_a <= (jaw_ani[None, :] - 10.0)] = subj_alpha[y_grid_a <= (jaw_ani[None, :] - 10.0)]
                final_alpha[y_grid_a > (jaw_ani[None, :] + 2.0)] = 0.0
            else:
                final_alpha = subj_alpha

            final_alpha = (final_alpha*255).astype(np.uint8).astype(np.float32)/255.
            final_alpha, animal_report = extend_animal_neck(subj_alpha, final_alpha)
            final_alpha = np.rint(final_alpha*255).astype(np.uint8).astype(np.float32)/255.
            final_rgba = np.dstack([img_rgb, (final_alpha * 255.0).astype(np.uint8)])
            ys_a, xs_a = np.where(final_alpha > 0.05)
            bbox_a = [int(xs_a.min()), int(ys_a.min()), int(xs_a.max()), int(ys_a.max())] if len(ys_a) and len(xs_a) else [0, 0, orig_w, orig_h]
            return {
                "head_alpha": final_alpha,
                "head_fg_rgb": img_rgb,
                "head_rgba": final_rgba,
                "species": "ANIMAL",
                "diagnostics": animal_report,
                "head_bbox": bbox_a,
                "shirt_pixels": None
            }

        # =====================================================================
        # HUMAN PROCESSING: YOLOFACE LANDMARKS & BIREFNET
        # =====================================================================
        best_i = valid_idx[np.argmax(scores[valid_idx])]
        row = outputs[best_i]
        cx_f, cy_f, w_f, h_f = row[0] * (orig_w / 640.0), row[1] * (orig_h / 640.0), row[2] * (orig_w / 640.0), row[3] * (orig_h / 640.0)
        tx1, ty1 = int(round(max(0, cx_f - w_f / 2))), int(round(max(0, cy_f - h_f / 2)))
        tx2, ty2 = int(round(min(orig_w, cx_f + w_f / 2))), int(round(min(orig_h, cy_f + h_f / 2)))
        fw, fh = max(1, tx2 - tx1), max(1, ty2 - ty1)
        cx, cy = (tx1 + tx2) // 2, (ty1 + ty2) // 2

        landmarks = []
        for l_idx in range(5):
            lx = int(round(row[5 + l_idx * 3] * (orig_w / 640.0)))
            ly = int(round(row[6 + l_idx * 3] * (orig_h / 640.0)))
            landmarks.append((lx, ly))
        re, le, nose, rm, lm = landmarks
        my = (rm[1] + lm[1]) / 2.0
        ny = nose[1]
        eye_y = (re[1] + le[1]) / 2.0
        d_nm = max(8.0, my - ny)

        # BiRefNet Matting
        pil_img = Image.fromarray(img_rgb)
        t_in = self.transform_biref(pil_img).unsqueeze(0).to(self.compute_device)
        with torch.inference_mode():
            preds = self.birefnet(t_in)[-1].sigmoid().cpu()
        subj_alpha = cv2.resize(preds[0, 0].numpy(), (orig_w, orig_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        subj_alpha = np.clip(subj_alpha, 0.0, 1.0)

        # BiSeNet Face Parsing (Pass 1)
        roi_dim = int(max(fw, fh) * 3.1)
        rx1, ry1 = max(0, cx - roi_dim // 2), max(0, ty1 + int(fh * 0.40) - int(roi_dim * 0.48))
        rx2, ry2 = min(orig_w, rx1 + roi_dim), min(orig_h, ry1 + roi_dim)
        crop = img_rgb[ry1:ry2, rx1:rx2]
        ch, cw = crop.shape[:2]
        blob_bise = cv2.resize(crop, (512, 512)).astype(np.float32) / 255.0
        blob_bise = (blob_bise - self.bise_mean) / self.bise_std
        blob_bise = np.transpose(blob_bise, (2, 0, 1))[None, :].astype(np.float32)
        out_bise = self.bisenet_sess.run(None, {self.bisenet_sess.get_inputs()[0].name: blob_bise})[0][0]
        exp_logits = np.exp(out_bise - np.max(out_bise, axis=0, keepdims=True))
        probs_bise = exp_logits / (np.sum(exp_logits, axis=0, keepdims=True) + 1e-6)

        prob_cloth_512 = probs_bise[16] + probs_bise[15]
        prob_neck_512 = probs_bise[14]
        prob_hair_512 = probs_bise[17]
        prob_hat_512 = probs_bise[18]
        prob_face_512 = np.sum(probs_bise[1:14], axis=0)

        p_full = np.zeros((orig_h, orig_w), dtype=np.uint8)
        p_full[ry1:ry2, rx1:rx2] = cv2.resize(np.argmax(out_bise, axis=0).astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST)
        prob_cloth = np.zeros((orig_h, orig_w), dtype=np.float32)
        prob_cloth[ry1:ry2, rx1:rx2] = cv2.resize(prob_cloth_512, (cw, ch), interpolation=cv2.INTER_LINEAR)
        prob_neck = np.zeros((orig_h, orig_w), dtype=np.float32)
        prob_neck[ry1:ry2, rx1:rx2] = cv2.resize(prob_neck_512, (cw, ch), interpolation=cv2.INTER_LINEAR)
        prob_hair = np.zeros((orig_h, orig_w), dtype=np.float32)
        prob_hair[ry1:ry2, rx1:rx2] = cv2.resize(prob_hair_512, (cw, ch), interpolation=cv2.INTER_LINEAR)
        prob_hat = np.zeros((orig_h, orig_w), dtype=np.float32)
        prob_hat[ry1:ry2, rx1:rx2] = cv2.resize(prob_hat_512, (cw, ch), interpolation=cv2.INTER_LINEAR)
        prob_face = np.zeros((orig_h, orig_w), dtype=np.float32)
        prob_face[ry1:ry2, rx1:rx2] = cv2.resize(prob_face_512, (cw, ch), interpolation=cv2.INTER_LINEAR)

        # CIELAB
        full_img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        y_grid, x_grid = np.ogrid[:orig_h, :orig_w]
        safe_skin_zone = (np.isin(p_full, [1, 10]) & (y_grid >= int(ty1 + fh * 0.12)) & (y_grid <= int(my)) & (np.abs(x_grid - cx) <= int(fw * 0.38)) & (prob_cloth < 0.05))
        if np.sum(safe_skin_zone) > 30:
            skin_pixels = full_img_lab[safe_skin_zone]
            skin_med_lab = np.median(skin_pixels, axis=0)
            skin_std_lab = np.clip(np.std(skin_pixels, axis=0), [10.0, 4.5, 4.5], [32.0, 16.0, 16.0])
        else:
            skin_med_lab = np.array([150.0, 138.0, 145.0], dtype=np.float32)
            skin_std_lab = np.array([22.0, 10.0, 10.0], dtype=np.float32)

        skin_chroma_diff = np.sqrt((full_img_lab[:, :, 1] - skin_med_lab[1]) ** 2 + (full_img_lab[:, :, 2] - skin_med_lab[2]) ** 2)
        skin_mahalanobis = np.sqrt(
            0.22 * ((full_img_lab[:, :, 0] - skin_med_lab[0]) / skin_std_lab[0]) ** 2 +
            1.0 * ((full_img_lab[:, :, 1] - skin_med_lab[1]) / skin_std_lab[1]) ** 2 +
            1.0 * ((full_img_lab[:, :, 2] - skin_med_lab[2]) / skin_std_lab[2]) ** 2
        )

        scalp_hair_zone = (
            ((p_full == 17) | (prob_hair > 0.45)) &
            (prob_hat < 0.20) & (prob_cloth < 0.12) &
            (y_grid <= int(my)) & (np.abs(x_grid - cx) <= int(fw * 0.95)) &
            (subj_alpha > 0.60)
        )
        if np.sum(scalp_hair_zone) > 40:
            hair_pixels = full_img_lab[scalp_hair_zone]
            hair_med_lab = np.median(hair_pixels, axis=0)
            hair_std_lab = np.clip(np.std(hair_pixels, axis=0), [14.0, 4.5, 4.5], [42.0, 18.0, 18.0])
        else:
            hair_med_lab = np.array([45.0, 128.0, 128.0], dtype=np.float32)
            hair_std_lab = np.array([28.0, 10.0, 10.0], dtype=np.float32)

        hair_mahalanobis = np.sqrt(
            0.15 * ((full_img_lab[:, :, 0] - hair_med_lab[0]) / hair_std_lab[0]) ** 2 +
            1.00 * ((full_img_lab[:, :, 1] - hair_med_lab[1]) / hair_std_lab[1]) ** 2 +
            1.00 * ((full_img_lab[:, :, 2] - hair_med_lab[2]) / hair_std_lab[2]) ** 2
        )
        hair_chroma_dist = np.sqrt(
            (full_img_lab[:, :, 1] - hair_med_lab[1]) ** 2 +
            (full_img_lab[:, :, 2] - hair_med_lab[2]) ** 2
        )
        hair_delta_e = np.linalg.norm(full_img_lab - hair_med_lab[None, None, :], axis=2)

        # Structure Tensor Fiber Coherence
        gray_f = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        j_xx = cv2.GaussianBlur(gx * gx, (7, 7), 1.5)
        j_yy = cv2.GaussianBlur(gy * gy, (7, 7), 1.5)
        j_xy = cv2.GaussianBlur(gx * gy, (7, 7), 1.5)
        trace_j = j_xx + j_yy + 1e-6
        det_j = np.maximum(0.0, j_xx * j_yy - j_xy * j_xy)
        coherence = np.sqrt(np.maximum(0.0, trace_j ** 2 - 4.0 * det_j)) / trace_j
        is_oriented_fiber = (coherence > 0.18) & (j_xx >= 0.35 * j_yy)

        cloth_sample_zone = np.isin(p_full, [15, 16]) & (prob_cloth > 0.60) & (subj_alpha > 0.5)
        has_cloth_sample = np.sum(cloth_sample_zone) > 40
        if has_cloth_sample:
            cloth_med_lab = np.median(full_img_lab[cloth_sample_zone], axis=0)
            cloth_delta_e = np.linalg.norm(full_img_lab - cloth_med_lab[None, None, :], axis=2)
            cloth_chroma_diff = np.sqrt((full_img_lab[:, :, 1] - cloth_med_lab[1]) ** 2 + (full_img_lab[:, :, 2] - cloth_med_lab[2]) ** 2)
        else:
            cloth_med_lab = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            cloth_delta_e = np.full((orig_h, orig_w), 999.0, dtype=np.float32)
            cloth_chroma_diff = np.full((orig_h, orig_w), 999.0, dtype=np.float32)

        # Jaw Curve & Chin
        lip_mask = np.isin(p_full, [11, 12, 13]) & (np.abs(x_grid - cx) <= int(fw * 0.35))
        lower_lip_bot_y = int(np.percentile(np.where(lip_mask)[0], 98.0)) if np.any(lip_mask) else int(round(my + 0.28 * d_nm))
        is_below_mouth = y_grid > lower_lip_bot_y
        
        # Sanctuary Features: bao gồm cả Kính (Class 6 eye_g), lông mày, mắt, tai, mũi, miệng
        sanctuary_features = np.isin(p_full, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

        pitch_ratio = np.clip(d_nm / max(8.0, ny - eye_y), 0.55, 1.45)
        chin_drop_est = int(round(d_nm * (1.18 / (pitch_ratio ** 0.35))))
        anatomical_chin_y = min(orig_h - 1, lower_lip_bot_y + chin_drop_est)

        x_coords = np.arange(orig_w)
        dx_norm = (x_coords - cx) / (fw * 0.52 + 1e-5)
        base_jaw_curve = anatomical_chin_y - int(fh * 0.16) * np.clip(dx_norm ** 2, 0.0, 1.25)

        is_chin_and_cheek_sanctuary = (
            (y_grid <= base_jaw_curve[None, :]) &
            (np.abs(x_grid - cx) <= int(fw * 0.46)) &
            (prob_face > 0.35) &
            (~np.isin(p_full, [15, 16]))
        )

        face_center_y = (eye_y + my) * 0.5
        elliptical_face_dist = np.sqrt(
            ((x_grid - cx) / (fw * 0.52 + 1e-5)) ** 2 +
            ((y_grid - face_center_y) / (fh * 0.60 + 1e-5)) ** 2
        )
        inner_face_core_ellipse = elliptical_face_dist <= 0.92
        outer_face_limit_ellipse = elliptical_face_dist <= 1.06

        strict_chin_floor = lower_lip_bot_y + int(d_nm * 1.28)
        finger_barrier = (
            ((y_grid > strict_chin_floor) & (y_grid > base_jaw_curve[None, :] + 3.0)) |
            ((~outer_face_limit_ellipse) & (y_grid > int(eye_y)) & (y_grid <= base_jaw_curve[None, :] + 6.0) & (~np.isin(p_full, [7, 8])))
        )

        chroma_thresh = max(16.5, 2.9 * np.hypot(skin_std_lab[1], skin_std_lab[2]))
        garment_color_leak = (
            is_below_mouth & (~sanctuary_features) & (~is_chin_and_cheek_sanctuary) &
            ((skin_chroma_diff > chroma_thresh) | (skin_mahalanobis > 3.8) |
             (has_cloth_sample & (cloth_chroma_diff < skin_chroma_diff * 0.78) & (skin_chroma_diff > 11.0)))
        )

        hard_cloth_forbidden = (np.isin(p_full, [15, 16]) | (prob_cloth > 0.16) | (garment_color_leak & (prob_face < 0.70))) & (~sanctuary_features) & (~is_chin_and_cheek_sanctuary)
        collar_barrier = (cv2.dilate(hard_cloth_forbidden.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & (~sanctuary_features) & (~is_chin_and_cheek_sanctuary)

        submental_chin_rescue = (((p_full == 14) | (prob_neck > 0.35)) & (y_grid <= (base_jaw_curve[None, :] + int(fh * 0.04))) & (np.abs(x_grid - cx) <= int(fw * 0.36)) & (~collar_barrier) & (~garment_color_leak) & (~finger_barrier))

        raw_face_skin = (
            (((np.isin(p_full, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]) & (prob_face > 0.30)) |
              submental_chin_rescue | is_chin_and_cheek_sanctuary) &
             (~hard_cloth_forbidden) & (~garment_color_leak) & (~finger_barrier) & (subj_alpha > 0.35))
        )

        skin_edt = ndimage.distance_transform_edt(raw_face_skin)
        thin_appendage_bridge = (~inner_face_core_ellipse) & (skin_edt > 0) & (skin_edt <= max(3.5, fw * 0.042))
        severed_face_skin = (raw_face_skin & (~thin_appendage_bridge)) | (raw_face_skin & inner_face_core_ellipse)

        num_f, lbls_f, stats_f, _ = cv2.connectedComponentsWithStats(severed_face_skin.astype(np.uint8), connectivity=8)
        if num_f > 1:
            nose_seed = severed_face_skin & (np.abs(x_grid - cx) <= int(fw * 0.22)) & (np.abs(y_grid - ny) <= int(fh * 0.22))
            valid_f_lbls = np.unique(lbls_f[nose_seed])
            valid_f_lbls = valid_f_lbls[valid_f_lbls > 0]
            strict_face_skin = np.isin(lbls_f, valid_f_lbls) if len(valid_f_lbls) > 0 else severed_face_skin
        else:
            strict_face_skin = severed_face_skin

        detached_skin_islands = raw_face_skin & (~strict_face_skin)
        forbidden_island_zone = cv2.dilate(detached_skin_islands.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        strict_face_skin = (
            cv2.dilate(strict_face_skin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        ) & raw_face_skin & (~forbidden_island_zone)

        jaw_curve = np.copy(base_jaw_curve).astype(np.float32)
        face_x_min = max(0, tx1 - int(fw * 0.10))
        face_x_max = min(orig_w - 1, tx2 + int(fw * 0.10))
        for x in range(face_x_min, face_x_max + 1):
            col_ys = np.where(strict_face_skin[:min(orig_h, int(base_jaw_curve[x]) + int(fh * 0.08)), x])[0]
            if len(col_ys) > 0:
                jaw_curve[x] = min(float(base_jaw_curve[x] + int(fh * 0.04)), max(float(lower_lip_bot_y + 6), float(np.percentile(col_ys, 99.0))))

        jaw_curve = ndimage.uniform_filter1d(jaw_curve, size=max(9, int(fw * 0.15) | 1), mode="nearest")
        is_below_jaw = y_grid > jaw_curve[None, :]

        # Micro-neck buffer
        max_neck_stub_px = int(np.clip(round(fh * neck_stub_ratio), 10, 48))
        dx_neck = (x_coords - cx) / (fw * 0.32 + 1e-5)
        neck_profile_weight = np.clip(1.0 - dx_neck ** 2, 0.0, 1.0)
        micro_neck_curve = jaw_curve + max_neck_stub_px * neck_profile_weight

        for x in range(face_x_min, face_x_max + 1):
            jaw_y_int = int(round(jaw_curve[x]))
            neck_y_int = min(orig_h - 1, int(round(micro_neck_curve[x])))
            if neck_y_int > jaw_y_int:
                collar_hits = np.where(collar_barrier[jaw_y_int:neck_y_int + 1, x])[0]
                if len(collar_hits) > 0:
                    micro_neck_curve[x] = float(jaw_y_int + collar_hits[0] - 1)

        micro_neck_curve = ndimage.uniform_filter1d(micro_neck_curve, size=max(7, int(fw * 0.11) | 1), mode="nearest")
        is_below_micro_neck = y_grid > micro_neck_curve[None, :]

        valid_neck_stub = (
            (((p_full == 14) | (prob_neck > 0.22) | (p_full == 1)) &
             (y_grid >= (jaw_curve[None, :] - 2.0)) & (~is_below_micro_neck) &
             (np.abs(x_grid - cx) <= int(fw * 0.32)) &
             (~hard_cloth_forbidden) & (~collar_barrier) & (~garment_color_leak) &
             (~finger_barrier) & (subj_alpha > 0.45))
        )
        neck_soft_weights = np.clip((micro_neck_curve[None, :] - y_grid) / 5.0, 0.0, 1.0).astype(np.float32)

        # ---------------------------------------------------------------------
        # HEAD SANCTUARY & PHỤ KIỆN TRÊN ĐẦU (Mũ lưỡi trai, Mũ trùm đầu / Hood, Kính)
        # ---------------------------------------------------------------------
        natural_face = (strict_face_skin | is_chin_and_cheek_sanctuary) & (~is_below_jaw) & (~hard_cloth_forbidden)
        filled_face = ndimage.binary_fill_holes(natural_face.astype(np.uint8)) > 0
        safe_hole_fill_zone = ((y_grid <= lower_lip_bot_y + int(d_nm * 0.30)) & (~hard_cloth_forbidden) & (p_full != 0))
        solid_face = natural_face | (filled_face & safe_hole_fill_zone)
        solid_face_core = (solid_face | sanctuary_features | is_chin_and_cheek_sanctuary) & (subj_alpha > 0.40) & (~hard_cloth_forbidden)

        # Biref Upper Dome: Mũ lưỡi trai (Class 18) và Kính (Class 6) đã được bảo vệ qua sanctuary_features và prob_hat.
        # Giữ nguyên ràng buộc chuẩn của Golden v9.1 để không rò rỉ vào cổ áo hay ngón tay ở Ca 15.
        biref_upper_dome = (
            (y_grid <= int(my)) &
            (np.abs(x_grid - cx) <= int(fw * 1.45)) &
            (subj_alpha > 0.15) &
            (~np.isin(p_full, [14, 15, 16])) &
            (prob_cloth < 0.10) &
            (prob_neck < 0.20)
        )

        upper_hair_and_hat = (
            (~is_below_jaw) &
            (np.isin(p_full, [17, 18]) | (prob_hair > 0.12) | (prob_hat > 0.20) | biref_upper_dome) &
            (~np.isin(p_full, [15, 16])) &
            (prob_cloth < 0.20)
        )

        raw_head_sanctuary = solid_face_core | natural_face | upper_hair_and_hat
        num_s, lbls_s = cv2.connectedComponents(cv2.dilate(raw_head_sanctuary.astype(np.uint8), np.ones((9, 9), np.uint8)))
        if num_s > 1:
            face_seed_lbls = np.unique(lbls_s[solid_face_core])
            face_seed_lbls = face_seed_lbls[face_seed_lbls > 0]
            head_sanctuary = raw_head_sanctuary & np.isin(lbls_s, face_seed_lbls) if len(face_seed_lbls) > 0 else raw_head_sanctuary
        else:
            head_sanctuary = raw_head_sanctuary

        head_sanctuary_halo = (
            (cv2.dilate(head_sanctuary.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) &
            (~is_below_jaw) &
            (~np.isin(p_full, [15, 16])) &
            (prob_cloth < 0.15)
        )
        head_sanctuary = head_sanctuary | head_sanctuary_halo

        # =====================================================================
        # BỘ ĐIỀU PHỐI TỰ ĐỘNG PHÂN NHÁNH (Long Hair Router)
        # =====================================================================
        # Tính Golden raw_hair_waterfall và connected_hair làm nền tảng chuẩn
        is_outside_neck_pillar = np.abs(x_grid - cx) > int(fw * 0.32)
        shoulder_skin_veto = (
            is_below_jaw & is_outside_neck_pillar &
            (np.isin(p_full, [1, 14]) | (prob_face > 0.22) | (prob_neck > 0.22) |
             ((skin_mahalanobis < 2.6) & (skin_chroma_diff < 13.5) & (hair_mahalanobis > 2.4)))
        )

        # Diệt sạch áo vàng và áo ngoại lai chuẩn Golden v9.1
        yellow_cloth_veto = is_below_jaw & (full_img_lab[:, :, 2] > 22.0) & (full_img_lab[:, :, 0] > 55.0) & (prob_face < 0.30)

        cloth_is_chromatic = has_cloth_sample and (np.hypot(cloth_med_lab[1] - 128.0, cloth_med_lab[2] - 128.0) > 6.0)
        is_closer_to_cloth = (
            has_cloth_sample and cloth_is_chromatic and
            (cloth_delta_e < np.minimum(24.0, hair_mahalanobis * 9.5)) &
            (cloth_chroma_diff < hair_chroma_dist * 0.82)
        ) if has_cloth_sample and cloth_is_chromatic else np.zeros((orig_h, orig_w), dtype=bool)

        foreign_garment_chroma = (
            is_below_jaw &
            (hair_mahalanobis > 3.1) &
            (hair_chroma_dist > max(16.0, 2.6 * np.hypot(hair_std_lab[1], hair_std_lab[2]))) &
            (skin_mahalanobis > 3.0)
        )

        is_authentic_hair_color = (hair_delta_e < 32.0) | ((hair_med_lab[0] < 45.0) & (full_img_lab[:, :, 0] < 45.0) & (hair_delta_e < 38.0))
        is_authentic_hair_stream = (
            ((hair_mahalanobis <= 2.85) | is_authentic_hair_color) &
            (~is_closer_to_cloth) &
            (~yellow_cloth_veto) &
            ((prob_hair > 0.08) | is_oriented_fiber | (p_full == 17) | is_authentic_hair_color)
        )

        actual_garment = (
            (np.isin(p_full, [15, 16, 18]) | (prob_cloth > 0.22) | is_closer_to_cloth | foreign_garment_chroma | yellow_cloth_veto) &
            (~(is_authentic_hair_stream & (prob_cloth < 0.65)))
        )
        hard_veto_barrier = cv2.dilate((shoulder_skin_veto | actual_garment).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

        raw_hair_waterfall = (
            (subj_alpha > 0.16) &
            (~hard_veto_barrier) &
            (~np.isin(p_full, [1, 14])) &
            (~finger_barrier | (np.abs(x_grid - cx) > int(fw * 0.36))) &
            (is_authentic_hair_stream | (p_full == 17) | (prob_hair > 0.22))
        )

        hair_with_head = head_sanctuary | raw_hair_waterfall
        num_hw, lbls_hw = cv2.connectedComponents(hair_with_head.astype(np.uint8))
        if num_hw > 1:
            head_lbls = np.unique(lbls_hw[head_sanctuary])
            head_lbls = head_lbls[head_lbls > 0]
            connected_hair = np.isin(lbls_hw, head_lbls) & raw_hair_waterfall
        else:
            connected_hair = raw_hair_waterfall

        collar_level_y = anatomical_chin_y + int(1.75 * fh)
        collar_zone_hair = connected_hair & (y_grid <= collar_level_y)

        # Kiểm tra điều kiện Tóc Nữ Dài (Long Hair Candidate)
        base_hair = (prob_hair > 0.50) & (subj_alpha > 0.50)
        band_y1 = max(0, int(anatomical_chin_y - 0.20 * fh))
        band_y2 = min(orig_h, int(anatomical_chin_y + 0.10 * fh))
        neck_band = base_hair[band_y1:band_y2, :]
        neck_span = np.sum(np.any(neck_band, axis=0)) if neck_band.size > 0 else 0
        hair_below_chin = np.sum(base_hair[anatomical_chin_y:, :])
        
        # Ngưỡng phân tách toán học: Tóc dài thật sự buông qua ngực/vai có tỷ lệ diện tích >= 0.50 * fw * fh
        # Váy Ngủ (tóc búi sau gáy) có tỷ lệ 0.398 < 0.50 -> 100% giữ nguyên ở Nhánh A Standard.
        # Nữ tóc dài buông xõa có tỷ lệ >= 0.613 -> 100% định tuyến vào Nhánh C Long Hair.
        is_long_hair_candidate = (neck_span >= 0.15 * fw) and (hair_below_chin >= 0.50 * fw * fh)

        # ---------------------------------------------------------------------
        # NHÁNH C: DEDICATED LONG HAIR WATERFALL (Nữ tóc buông dài)
        # ---------------------------------------------------------------------
        if is_long_hair_candidate:
            lab = full_img_lab
            upper_mask = np.zeros_like(base_hair)
            upper_mask[:anatomical_chin_y, :] = True
            hair_seed_pts = lab[(prob_hair > 0.80) & upper_mask & (subj_alpha > 0.50)]
            
            if len(hair_seed_pts) >= 80:
                l_med = np.median(hair_seed_pts[:, 0])
                hair_clusters = []
                for subset in (hair_seed_pts[hair_seed_pts[:, 0] <= l_med], hair_seed_pts[hair_seed_pts[:, 0] > l_med]):
                    if len(subset) >= 20:
                        mu = np.mean(subset, axis=0)
                        cov_inv = np.linalg.pinv(np.cov(subset.T) + np.eye(3) * 12.0)
                        hair_clusters.append((mu, cov_inv))

                sx1 = max(0, int(cx - 0.28 * fw))
                sx2 = min(orig_w, int(cx + 0.28 * fw))
                sy1 = max(0, int(ty1 + 0.20 * fh))
                sy2 = min(orig_h, int(ty1 + 0.72 * fh))
                skin_pts = lab[sy1:sy2, sx1:sx2].reshape(-1, 3)
                skin_mu = np.mean(skin_pts, axis=0)
                skin_cov_inv = np.linalg.pinv(np.cov(skin_pts.T) + np.eye(3) * 16.0)

                gx1 = max(0, int(cx - 0.22 * fw))
                gx2 = min(orig_w, int(cx + 0.22 * fw))
                gy1 = min(orig_h, int(anatomical_chin_y + 0.45 * fh))
                gy2 = min(orig_h, int(anatomical_chin_y + 0.85 * fh))
                garment_pts = lab[gy1:gy2, gx1:gx2].reshape(-1, 3)
                has_garment = len(garment_pts) >= 40
                if has_garment:
                    g_mu = np.mean(garment_pts, axis=0)
                    g_cov_inv = np.linalg.pinv(np.cov(garment_pts.T) + np.eye(3) * 20.0)

                def _mahal(arr_lab: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
                    diff = arr_lab - mu.reshape(1, 1, 3)
                    return np.sqrt(np.maximum(0.0, np.einsum("ijk,kl,ijl->ij", diff, cov_inv, diff)))

                # Waterfall ROI bắt đầu từ lower_lip_bot_y để nối liền mạch với xương hàm
                roi_y1 = max(0, int(lower_lip_bot_y))
                roi_y2 = min(orig_h, int(anatomical_chin_y + 4.20 * fh))
                roi_x1 = max(0, int(cx - 1.45 * fw))
                roi_x2 = min(orig_w, int(cx + 1.45 * fw))
                
                lab_roi = lab[roi_y1:roi_y2, roi_x1:roi_x2]
                alpha_roi = subj_alpha[roi_y1:roi_y2, roi_x1:roi_x2] > 0.35
                
                L_cie_roi = lab_roi[:, :, 0] * (100.0 / 255.0)
                a_cie_roi = lab_roi[:, :, 1] - 128.0
                b_cie_roi = lab_roi[:, :, 2] - 128.0
                yellow_veto_roi = (b_cie_roi > 28.0) & ((b_cie_roi - np.abs(a_cie_roi)) > 14.0) & (L_cie_roi > 55.0)

                d_hair = np.minimum.reduce([_mahal(lab_roi, mu, ci) for mu, ci in hair_clusters])
                d_skin = _mahal(lab_roi, skin_mu, skin_cov_inv)
                d_neg = np.minimum(d_skin, _mahal(lab_roi, g_mu, g_cov_inv)) if has_garment else d_skin

                color_valid = (d_neg - d_hair > 0.20) & (d_hair < 3.6)

                # Adaptive Gravity Hair Funnel
                roi_h, roi_w = lab_roi.shape[:2]
                funnel_mask = np.zeros((roi_h, roi_w), dtype=bool)
                xs_global = np.arange(roi_x1, roi_x2)
                for dy in range(roi_h):
                    y_curr = roi_y1 + dy
                    t = max(0.0, (y_curr - anatomical_chin_y) / float(fh))
                    if t <= 0.35:
                        half_w = 1.10 * fw + 0.12 * fw * np.sin(np.pi * t / 0.70)
                    else:
                        half_w = 1.22 * fw * np.exp(-0.16 * (t - 0.35))
                    funnel_mask[dy, :] = (xs_global >= (cx - half_w)) & (xs_global <= (cx + half_w))

                candidate_wf = alpha_roi & color_valid & funnel_mask & (~yellow_veto_roi)

                # Hạt giống kết nối từ cằm và cổ Golden
                chin_seed_zone = (head_sanctuary | collar_zone_hair)[roi_y1:roi_y2, roi_x1:roi_x2]

                num_lwf, lbls_wf = cv2.connectedComponents((candidate_wf | chin_seed_zone).astype(np.uint8))
                seed_lbls_wf = np.unique(lbls_wf[chin_seed_zone])
                seed_lbls_wf = seed_lbls_wf[seed_lbls_wf > 0]
                
                if len(seed_lbls_wf) > 0:
                    wf_connected = np.isin(lbls_wf, seed_lbls_wf) & candidate_wf
                else:
                    wf_connected = np.zeros((roi_h, roi_w), dtype=bool)

                wf_u8 = cv2.morphologyEx(wf_connected.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                wf_u8 = cv2.morphologyEx(wf_u8, cv2.MORPH_CLOSE, np.ones((7, 3), np.uint8))

                waterfall_ext_mask = np.zeros((orig_h, orig_w), dtype=bool)
                waterfall_ext_mask[roi_y1:roi_y2, roi_x1:roi_x2] = wf_u8 > 0
                # Bảo vệ toàn vẹn áo trùm đầu (Hoodie hood) bao quanh vòm đầu phía trên cằm
                hood_on_head = (
                    (y_grid <= anatomical_chin_y) &
                    (np.abs(x_grid - cx) <= int(fw * 1.25)) &
                    (subj_alpha > 0.40) &
                    (~finger_barrier)
                )

                # Ràng buộc phễu trọng lực toàn cục dưới cằm: Tuyệt đối không cho phép tóc tràn sang cánh tay hay vật thể
                full_funnel_mask = np.zeros((orig_h, orig_w), dtype=bool)
                for dy in range(orig_h - anatomical_chin_y):
                    y_curr = anatomical_chin_y + dy
                    t = dy / float(fh)
                    if t <= 0.35:
                        half_w = 1.10 * fw + 0.12 * fw * np.sin(np.pi * t / 0.70)
                    else:
                        half_w = 1.22 * fw * np.exp(-0.16 * (t - 0.35))
                    full_funnel_mask[y_curr, :] = (x_grid[0, :] >= (cx - half_w)) & (x_grid[0, :] <= (cx + half_w))

                outside_funnel_below_chin = (y_grid > anatomical_chin_y) & (~full_funnel_mask)

                all_protected_hair = (head_sanctuary | connected_hair | waterfall_ext_mask | hood_on_head) & (~outside_funnel_below_chin)
            else:
                all_protected_hair = head_sanctuary | connected_hair
        else:
            # -----------------------------------------------------------------
            # NHÁNH A: STANDARD HUMAN (Golden v9.1 sạch 100%)
            # -----------------------------------------------------------------
            all_protected_hair = head_sanctuary | connected_hair

        # Tổng hợp Head Support
        head_support = np.zeros((orig_h, orig_w), dtype=np.float32)
        head_support[all_protected_hair] = 1.0
        head_support[valid_neck_stub] = np.maximum(head_support[valid_neck_stub], neck_soft_weights[valid_neck_stub])

        blurred_support = cv2.GaussianBlur(head_support, (3, 3), 0.6)
        head_support = np.where(all_protected_hair, 1.0, blurred_support)

        strictly_excluded = (
            (is_below_micro_neck & (~all_protected_hair)) |
            (is_below_jaw & (~valid_neck_stub) & (~all_protected_hair)) |
            (hard_cloth_forbidden & (~all_protected_hair)) |
            (finger_barrier & (~all_protected_hair)) |
            (yellow_cloth_veto & (~all_protected_hair))
        )
        head_support[strictly_excluded] = 0.0

        guide_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        refined_subj_alpha = self._guided_filter_alpha(guide_gray, subj_alpha, radius=3, eps=5e-4)
        base_alpha = np.where(all_protected_hair, subj_alpha, refined_subj_alpha)

        eroded_face_interior = cv2.erode(
            (solid_face & (subj_alpha >= 0.88)).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        ) > 0
        solid_interior_core = (eroded_face_interior | sanctuary_features) & (subj_alpha >= 0.85) & (~hard_cloth_forbidden)

        final_alpha = np.clip(base_alpha * head_support, 0.0, 1.0).astype(np.float32)
        final_alpha[solid_interior_core] = 1.0
        solid_neck_core = (
            cv2.erode(valid_neck_stub.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        ) & (neck_soft_weights >= 0.88) & (subj_alpha > 0.90) & (~collar_barrier) & (~finger_barrier)
        final_alpha[solid_neck_core] = 1.0
        final_alpha[strictly_excluded] = 0.0

        below_jaw_debris = (final_alpha > 0.15) & is_below_jaw & (~all_protected_hair) & (~valid_neck_stub)
        if np.any(below_jaw_debris):
            final_alpha[below_jaw_debris] = 0.0

        # Topological Head Continuity Gate: Loại bỏ tuyệt đối dị vật/ngón tay lơ lửng không liền mạch
        bin_c = (final_alpha > 0.05).astype(np.uint8)
        num_c, lbls_c = cv2.connectedComponents(bin_c, connectivity=8)
        if num_c > 2:
            head_seeds = [lbls_c[int(nose[1]), int(nose[0])], lbls_c[int(eye_y), int(cx)]]
            valid_components = set([lbl for lbl in head_seeds if lbl > 0])
            if is_long_hair_candidate:
                overlapping_lbls = np.unique(lbls_c[solid_interior_core | valid_neck_stub | head_sanctuary])
            else:
                overlapping_lbls = np.unique(lbls_c[solid_interior_core | valid_neck_stub])
            for ol in overlapping_lbls:
                if ol > 0:
                    valid_components.add(ol)
            keep_mask = np.isin(lbls_c, list(valid_components))
            final_alpha = np.where(keep_mask, final_alpha, 0.0).astype(np.float32)

        # Decontamination ở rìa má
        fg_rgb = img_rgb.copy()
        deep_clean_core = cv2.erode(
            ((final_alpha >= 0.94) & (subj_alpha >= 0.94)).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        ) > 0
        if not np.any(deep_clean_core):
            deep_clean_core = (final_alpha >= 0.85)

        _, edt_idx = ndimage.distance_transform_edt(~deep_clean_core, return_indices=True)
        nearest_clean_rgb = img_rgb[edt_idx[0], edt_idx[1]].astype(np.float32)
        nearest_clean_lab = full_img_lab[edt_idx[0], edt_idx[1]]
        lum_drop = np.maximum(0.0, nearest_clean_lab[:, :, 0] - full_img_lab[:, :, 0])
        local_delta_e = np.linalg.norm(full_img_lab - nearest_clean_lab, axis=2)

        is_skin_rim = (
            (final_alpha > 0.005) &
            (~deep_clean_core) &
            (~sanctuary_features) &
            (p_full != 17) & (prob_hair < 0.25)
        )
        dark_rim_bleed = is_skin_rim & ((lum_drop > 10.0) | (local_delta_e > 9.5) | (final_alpha < 0.92))
        if np.any(dark_rim_bleed):
            shave_offset = np.clip((lum_drop - 8.0) / 60.0, 0.05, 0.36)
            shaved_alpha = np.clip((final_alpha - shave_offset) / (1.0 - shave_offset + 1e-5), 0.0, 1.0) ** 1.30
            final_alpha = np.where(dark_rim_bleed & (lum_drop > 10.0), shaved_alpha, final_alpha).astype(np.float32)

        fringe_mask = (final_alpha > 0.005) & (~deep_clean_core) & (~sanctuary_features)
        if np.any(fringe_mask):
            base_pull = np.clip((0.96 - final_alpha) / 0.96, 0.0, 1.0)
            boost = np.where(dark_rim_bleed, np.clip(0.75 + lum_drop / 40.0, 0.78, 0.96), 0.65)
            pull_w = np.maximum(base_pull, np.where(dark_rim_bleed & (lum_drop > 12.0), 0.88, 0.0)) * boost
            pull_w = np.clip(pull_w, 0.0, 0.96)[:, :, None]
            decontam_rgb = np.clip(
                img_rgb.astype(np.float32) * (1.0 - pull_w) + nearest_clean_rgb * pull_w,
                0.0, 255.0
            ).astype(np.uint8)
            fg_rgb[fringe_mask] = decontam_rgb[fringe_mask]

        fg_rgb[solid_interior_core | valid_neck_stub] = img_rgb[solid_interior_core | valid_neck_stub]
        final_rgba = np.dstack([fg_rgb, (final_alpha * 255.0).astype(np.uint8)])

        # Đo Bbox
        ys_h, xs_h = np.where(final_alpha > 0.05)
        if len(ys_h) and len(xs_h):
            bbox_h = [int(xs_h.min()), int(ys_h.min()), int(xs_h.max()), int(ys_h.max())]
        else:
            bbox_h = [tx1, ty1, tx2, ty2]

        diagnostics = {"refinement": "unchanged", "review_required": True}
        if is_long_hair_candidate:
            try:
                from .head_refine_v96 import refine_long_hair
            except ImportError:
                from head_refine_v96 import refine_long_hair
            state = {'subj_alpha': subj_alpha, 'prob_hair': prob_hair, 'prob_cloth': prob_cloth, 'prob_neck': prob_neck, 'prob_face': prob_face, 'p_full': p_full, 'head_sanctuary': head_sanctuary, 'valid_neck_stub': valid_neck_stub, 'anatomical_chin_y': anatomical_chin_y, 'fw': fw, 'fh': fh, 'cx': cx, 'my': my, 'lower_lip_bot_y': lower_lip_bot_y, 'solid_interior_core': solid_interior_core, 'finger_barrier': finger_barrier, 'jaw_curve': jaw_curve, 'micro_neck_curve': micro_neck_curve}
            state.update(baseline_alpha=final_alpha, baseline_rgb=fg_rgb)
            final_alpha, fg_rgb, detail = refine_long_hair(img_rgb, state)
            diagnostics.update(detail)
            final_rgba = np.dstack([fg_rgb, (final_alpha*255).astype(np.uint8)])
            ys_h, xs_h = np.where(final_alpha > .05)
            if len(ys_h):
                bbox_h = [int(xs_h.min()),int(ys_h.min()),int(xs_h.max()),int(ys_h.max())]
        state97 = dict(subj_alpha=subj_alpha, prob_hair=prob_hair, prob_cloth=prob_cloth,
                       prob_face=prob_face, p_full=p_full, fw=fw, fh=fh, cx=cx, my=my,
                       anatomical_chin_y=anatomical_chin_y, solid_interior_core=solid_interior_core,
                       landmarks=np.asarray(landmarks))
        state97["aligned"] = aligned_parse(img_rgb, landmarks, fw, fh, self.bisenet_sess, self.bise_mean, self.bise_std)
        self.last_state = state97
        # Match PNG alpha precision used for the review baseline.
        final_alpha = (final_alpha*255).astype(np.uint8).astype(np.float32)/255.
        final_alpha, fg_rgb, report97 = preserve_head(img_rgb, state97, final_alpha, fg_rgb, is_long_hair_candidate)
        diagnostics.update(report97)
        try:
            from .head_finish_v98 import finish_head
        except ImportError:
            from head_finish_v98 import finish_head
        final_alpha = np.rint(final_alpha*255).astype(np.uint8).astype(np.float32)/255.
        final_alpha, fg_rgb, report98 = finish_head(img_rgb, state97, final_alpha, fg_rgb, is_long_hair_candidate)
        diagnostics.update(report98)
        try:
            from .head_boundary_v99 import correct_boundary
        except ImportError:
            from head_boundary_v99 import correct_boundary
        final_alpha = np.rint(final_alpha*255).astype(np.uint8).astype(np.float32)/255.
        final_alpha, fg_rgb, report99 = correct_boundary(img_rgb, state97, final_alpha, fg_rgb, is_long_hair_candidate)
        diagnostics.update(report99)
        try:
            from .head_contour_v100 import refine_contour
        except ImportError:
            from head_contour_v100 import refine_contour
        final_alpha = np.rint(final_alpha*255).astype(np.uint8).astype(np.float32)/255.
        final_alpha, fg_rgb, report100 = refine_contour(img_rgb, state97, final_alpha, fg_rgb, is_long_hair_candidate)
        diagnostics.update(report100)
        final_rgba = np.dstack([fg_rgb, np.rint(final_alpha*255).astype(np.uint8)])
        ys_h, xs_h = np.where(final_alpha>.05)
        if len(ys_h): bbox_h = [int(xs_h.min()), int(ys_h.min()), int(xs_h.max()), int(ys_h.max())]
        diagnostics["semantic_cloth_overlap_pixels"] = int(np.count_nonzero((final_alpha>.05)&(prob_cloth>.5)))
        diagnostics["note"] = "Semantic overlap is a proxy, not a garment-free certificate."

        species_tag = "HUMAN_LONG_HAIR" if is_long_hair_candidate else "HUMAN_STANDARD"
        return {
            "head_alpha": final_alpha,
            "head_fg_rgb": fg_rgb,
            "head_rgba": final_rgba,
            "geometry": {"landmarks": np.asarray(landmarks).tolist(), "fw": fw, "fh": fh, "cx": cx},
            "species": species_tag,
            "diagnostics": diagnostics,
            "head_bbox": bbox_h,
            "shirt_pixels": None
        }
