# -*- coding: utf-8 -*-
"""
SaoChepAdaptiveClarityEnhancer - 100% PyTorch CUDA On-Flow
ByteDance CapCut Post-Processing Engine for ComfyUI
Ref: HANDOVER_ADAPTIVE_CLARITY_ENHANCER.md
"""
import math
import torch
import torch.nn.functional as F

class SaoChepAdaptiveClarityEnhancer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),  # PyTorch Tensor: [B, H, W, C] in range [0.0, 1.0]
            },
            "optional": {
                "adaptive_mode": ("BOOLEAN", {"default": True}),
                "blend_opacity": ("FLOAT", {"default": 0.31, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sharpen_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "clarity_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "SaoChep/Commercial"

    def build_tone_lut_tensor(self, device):
        # 1D LUT 256 levels per ByteDance CapCut Shader specification
        x = torch.linspace(0.0, 1.0, 256, device=device)
        
        # 1. Brightness (+46%, param = 0.3 * 0.45785877 = 0.13735763 -> p = 1.686788)
        p_bright = 1.0 + (0.3 * 0.45785877) * 5.0
        bright = 1.0 - torch.pow(1.0 - x, p_bright)
        
        # 2. Shadow (-100% UI, p_s = -1.0 -> a_s = 1.900)
        p_s = -1.0
        a_s = 1.0 - 0.503 * p_s + 0.183 * (p_s ** 2) - 0.147 * (p_s ** 3) + 0.067 * (p_s ** 4)
        shadow = torch.pow(bright, a_s) + (a_s - 1.0) * (torch.pow(bright, 2) - torch.pow(bright, 3))
        
        # 3. Contrast Sigmoid (Pivot = 0.435)
        param_c = 0.29840547 * 0.6 + 1.0
        pivot = 0.435
        a = torch.exp(torch.tensor(8.33 * param_c - 12.16, device=device)) + 5.82 * param_c - 1.72
        
        def sig(val):
            s = 1.0 / (1.0 + torch.exp(-a * (val - pivot)))
            y = s + pivot - 0.5
            k = a * s * (1.0 - s)
            return y, k

        leftVal, leftSlope = sig(torch.tensor(0.0, device=device))
        rightVal, rightSlope = sig(torch.tensor(1.0, device=device))
        pVal, pSlope = sig(torch.tensor(pivot, device=device))

        leftDiff = 0.0 - leftVal
        rightDiff = 1.0 - rightVal
        leftSlopeDiff = pSlope - leftSlope
        rightSlopeDiff = pSlope - rightSlope

        y, k = sig(shadow)
        scale_left = (pSlope - k) / (leftSlopeDiff + 1e-7)
        scale_right = (pSlope - k) / (rightSlopeDiff + 1e-7)

        lut = torch.where(shadow <= pivot, y + (scale_left ** 2) * leftDiff, y + (scale_right ** 2) * rightDiff)
        return torch.clamp(lut, 0.0, 1.0)

    def _process_chunk(self, chunk, tone_lut, adaptive_mode, blend_opacity, sharpen_strength, clarity_strength):
        # 1. Apply Tone Curve through LUT
        idx = torch.clamp((chunk * 255.0).round().long(), 0, 255)
        layer_tone = tone_lut[idx]

        # 2. Extract YCbCr channels for Base and Tone
        R_base, G_base, B_base = chunk[..., 0], chunk[..., 1], chunk[..., 2]
        Y_base  = 0.299 * R_base + 0.587 * G_base + 0.114 * B_base
        Cb_base = -0.168736 * R_base - 0.331264 * G_base + 0.5 * B_base
        Cr_base = 0.5 * R_base - 0.418688 * G_base - 0.081312 * B_base

        R_tone, G_tone, B_tone = layer_tone[..., 0], layer_tone[..., 1], layer_tone[..., 2]
        Y_tone  = 0.299 * R_tone + 0.587 * G_tone + 0.114 * B_tone

        # 3. Directional Sharpening (4-way Laplacian per CapCut fshader.frag)
        if sharpen_strength > 0.001:
            Y_pad = F.pad(Y_tone.unsqueeze(1), (1, 1, 1, 1), mode='replicate').squeeze(1)
            p4 = Y_pad[:, 1:-1, :-2]
            p6 = Y_pad[:, 1:-1, 2:]
            p2 = Y_pad[:, :-2, 1:-1]
            p8 = Y_pad[:, 2:, 1:-1]
            p1 = Y_pad[:, :-2, :-2]
            p9 = Y_pad[:, 2:, 2:]
            p3 = Y_pad[:, :-2, 2:]
            p7 = Y_pad[:, 2:, :-2]

            f1 = 2.0 * Y_tone - p4 - p6
            f2 = 2.0 * Y_tone - p2 - p8
            f3 = 2.0 * Y_tone - p3 - p7
            f4 = 2.0 * Y_tone - p1 - p9

            abs1, abs2, abs3, abs4 = torch.abs(f1), torch.abs(f2), torch.abs(f3), torch.abs(f4)
            m = torch.maximum(torch.maximum(abs1, abs2), torch.maximum(abs3, abs4))
            best_f = torch.where(m == abs1, f1, torch.where(m == abs2, f2, torch.where(m == abs3, f3, f4)))
            sharpen_term = sharpen_strength * best_f
        else:
            sharpen_term = 0.0

        # 4. Multi-scale Clarity (fine vs med structure contrast)
        Y_4d = Y_tone.unsqueeze(1)  # [B, 1, H, W]
        blur_fine = F.avg_pool2d(Y_4d, kernel_size=5, stride=1, padding=2)
        blur_med  = F.avg_pool2d(Y_4d, kernel_size=13, stride=1, padding=6)
        clarity = (blur_fine - blur_med).squeeze(1)

        # 5. Scene-Adaptive Scaling & Dynamic Router
        eff_sharpen = sharpen_strength
        eff_clarity = clarity_strength
        eff_blend   = blend_opacity

        if adaptive_mode:
            # Measure global average scene luminance to detect low-key scenes
            scene_luma = Y_base.mean().item()
            if scene_luma < 0.35:
                # Scale back enhancement on low-key scenes to prevent noise amplification
                scene_scale = max(0.35, scene_luma / 0.35)
                eff_sharpen = sharpen_strength * scene_scale
                eff_clarity = clarity_strength * scene_scale
                eff_blend   = blend_opacity * scene_scale

            # Recompute directional sharpening with scene-scaled factor if adjusted
            if eff_sharpen != sharpen_strength and sharpen_strength > 0.001:
                sharpen_term = eff_sharpen * best_f

            Y_enhanced = torch.clamp(Y_tone + sharpen_term + eff_clarity * clarity, 0.0, 1.0)

            # Shadow Toe Guard (Y < 0.20): protect shadow regions from luminance collapse (< 5% max drop)
            Y_mult_candidate = Y_base * (1.0 - eff_blend) + (Y_base * Y_enhanced) * eff_blend
            # Enforce max 5% drop limit on shadow luminance
            min_allowed_Y = Y_base * 0.95
            Y_guarded = torch.maximum(Y_mult_candidate, min_allowed_Y)

            # Smoothstep transition for shadow toe [0.0, 0.20]
            t_toe = torch.clamp(Y_base / 0.20, 0.0, 1.0)
            toe_weight = t_toe * t_toe * (3.0 - 2.0 * t_toe)
            Y_final_luma = Y_base * (1.0 - toe_weight) + Y_guarded * toe_weight

            # Chroma Decoupling: Lock original Cb, Cr to eliminate color shift and saturation blowout
            R_out = torch.clamp(Y_final_luma + 1.402 * Cr_base, 0.0, 1.0)
            G_out = torch.clamp(Y_final_luma - 0.344136 * Cb_base - 0.714136 * Cr_base, 0.0, 1.0)
            B_out = torch.clamp(Y_final_luma + 1.772 * Cb_base, 0.0, 1.0)
            return torch.stack([R_out, G_out, B_out], dim=-1)
        else:
            # Legacy Mode A: RGB Multiply (when adaptive_mode is explicitly False)
            Y_enhanced = torch.clamp(Y_tone + sharpen_term + eff_clarity * clarity, 0.0, 1.0)
            Cb_tone = -0.168736 * R_tone - 0.331264 * G_tone + 0.5 * B_tone
            Cr_tone = 0.5 * R_tone - 0.418688 * G_tone - 0.081312 * B_tone
            R_lay = torch.clamp(Y_enhanced + 1.402 * Cr_tone, 0.0, 1.0)
            G_lay = torch.clamp(Y_enhanced - 0.344136 * Cb_tone - 0.714136 * Cr_tone, 0.0, 1.0)
            B_lay = torch.clamp(Y_enhanced + 1.772 * Cb_tone, 0.0, 1.0)
            layer_overlay = torch.stack([R_lay, G_lay, B_lay], dim=-1)
            multiply = chunk * layer_overlay
            return torch.clamp(chunk * (1.0 - blend_opacity) + multiply * blend_opacity, 0.0, 1.0)

    def _temporal_head_stabilize(self, images: torch.Tensor) -> torch.Tensor:
        """
        Suppresses initial 0-2s VAE causal boundary shift, luminance drop, and color fluctuation.
        Anchors the head frames to the video's stable steady-state baseline using
        a smooth cosine decay ramp (< 0.005s on GPU).
        """
        B = images.shape[0]
        if B < 30:
            return images

        # 1. Spatial frame means [B, 3]
        frame_means = images.mean(dim=(1, 2))  # [B, 3]

        # 2. Head length: first ~48 frames (2.0s @ 24fps) or up to 20% of video
        head_len = min(60, max(24, int(B * 0.20)))

        # Stable baseline: rolling median of steady-state frames right after the head
        stable_window = frame_means[head_len : min(head_len + 48, B)]
        if len(stable_window) == 0:
            return images
        stable_head_ref = torch.median(stable_window, dim=0).values  # [3]

        # 3. Smooth temporal correction across frames 0 to head_len
        for t in range(head_len):
            f_mean = frame_means[t]
            delta = f_mean - stable_head_ref  # [3] (R, G, B)

            # Only correct if significant luminance/chroma discrepancy (> 0.008, ~2/255)
            if torch.max(torch.abs(delta)) > 0.008:
                # Smooth cosine decay from 1.0 down to 0.0
                weight = 0.5 * (1.0 + torch.cos(torch.tensor(3.141592653589793 * t / float(head_len), device=images.device, dtype=images.dtype)))
                correction = delta * weight
                images[t] = torch.clamp(images[t] - correction.view(1, 1, 3), 0.0, 1.0)

        return images

    def _temporal_boundary_smooth(self, images: torch.Tensor) -> torch.Tensor:
        """
        Pure Causal VAE Boundary Purge (Rule 28 Single-Pass Length Gate):
        When length = N + 1 (e.g. 241 frames for a 240 frame video),
        Wan 2.1 frame 0 contains the causal lattice boundary artifact.
        Slicing images = images[1:] completely purges the artifact and leaves exactly N pristine frames.
        """
        B = images.shape[0]
        if B % 4 == 1 and B > 16:
            return images[1:]
        return images

    def _temporal_color_stabilize(self, images: torch.Tensor) -> torch.Tensor:
        """
        Suppresses VAE tail-chunk chromatic drift (Cyan/Magenta/Cooling shift)
        across temporal boundaries with smooth cosine ease-in (< 0.005s on GPU).
        Eliminates step discontinuity at second 8 (tail_start).
        """
        B = images.shape[0]
        if B < 60:
            return images

        # 1. Spatial frame means [B, 3] in [0.0, 1.0]
        frame_means = images.mean(dim=(1, 2))  # [B, 3]

        # 2. Steady-state baseline: rolling window after head stabilization (frames head_len..min(B, head_len+48))
        head_len = min(60, max(24, int(B * 0.20)))
        steady_end = min(B, max(head_len + 48, int(B * 0.50)))
        stable_window = frame_means[head_len : steady_end]
        if len(stable_window) == 0:
            return images
        stable_baseline = torch.median(stable_window, dim=0).values  # [3]

        tail_start = steady_end
        if tail_start >= B - 8:
            return images

        # 3. Low-pass smoothed macro trajectory over the tail to avoid per-frame flicker
        kernel_size = min(31, max(5, ((B - tail_start) // 4) * 2 + 1))
        pad = kernel_size // 2
        padded = F.pad(
            frame_means.unsqueeze(0).transpose(1, 2), (pad, pad), mode="replicate"
        ).transpose(1, 2).squeeze(0)
        smoothed_means = torch.empty_like(frame_means)
        for i in range(B):
            smoothed_means[i] = padded[i : i + kernel_size].mean(dim=0)

        # 4. Smooth cosine ease-in transition across ramp_len frames
        ramp_len = max(12, int((B - tail_start) * 0.35))
        pi_val = 3.141592653589793

        for t in range(tail_start, B):
            delta = smoothed_means[t] - stable_baseline  # [3] (R, G, B)
            # Decompose into luminance drift vs pure chromatic drift
            # Y = 0.299 R + 0.587 G + 0.114 B
            y_delta = 0.299 * delta[0] + 0.587 * delta[1] + 0.114 * delta[2]
            chroma_delta = delta - y_delta
            # Correct 100% of chromatic drift and 30% of luminance drift (to preserve legitimate lighting progression)
            target_correction = chroma_delta + 0.3 * y_delta

            progress = min(1.0, float(t - tail_start) / float(ramp_len))
            weight = 0.5 * (1.0 - math.cos(pi_val * progress))

            correction = target_correction * weight
            if torch.max(torch.abs(correction)) > 0.004:
                images[t] = torch.clamp(images[t] - correction.view(1, 1, 3), 0.0, 1.0)

        return images

    def process(self, images, adaptive_mode=True, blend_opacity=0.31, sharpen_strength=1.0, clarity_strength=0.4):
        orig_device = images.device
        calc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Keep temporal operations global while using one CPU working/output buffer.
        images = self._temporal_boundary_smooth(images)
        out = torch.empty(images.shape, dtype=images.dtype, device="cpu")
        out.copy_(images)
        self._temporal_head_stabilize(out)
        self._temporal_color_stabilize(out)

        tone_lut = self.build_tone_lut_tensor(calc_device)
        for start in range(0, out.shape[0], 32):
            end = min(start + 32, out.shape[0])
            sub = out[start:end].to(calc_device)
            processed = self._process_chunk(
                sub, tone_lut, adaptive_mode, blend_opacity,
                sharpen_strength, clarity_strength,
            )
            out[start:end].copy_(processed.to("cpu"))
            del sub, processed

        return (out.to(orig_device),)


NODE_CLASS_MAPPINGS = {
    "SaoChepAdaptiveClarityEnhancer": SaoChepAdaptiveClarityEnhancer
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaoChepAdaptiveClarityEnhancer": "SaoChep Adaptive Clarity Enhancer (ByteDance CapCut On-Flow)"
}
