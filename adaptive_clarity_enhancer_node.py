# -*- coding: utf-8 -*-
"""
SaoChepAdaptiveClarityEnhancer - 100% PyTorch CUDA On-Flow
ByteDance CapCut Post-Processing Engine for ComfyUI
Ref: HANDOVER_ADAPTIVE_CLARITY_ENHANCER.md
"""
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
        p4 = torch.roll(Y_tone, shifts=1, dims=2)
        p6 = torch.roll(Y_tone, shifts=-1, dims=2)
        p2 = torch.roll(Y_tone, shifts=1, dims=1)
        p8 = torch.roll(Y_tone, shifts=-1, dims=1)
        p1 = torch.roll(p4, shifts=1, dims=1)
        p9 = torch.roll(p6, shifts=-1, dims=1)
        p3 = torch.roll(p6, shifts=1, dims=1)
        p7 = torch.roll(p4, shifts=-1, dims=1)

        f1 = 2.0 * Y_tone - p4 - p6
        f2 = 2.0 * Y_tone - p2 - p8
        f3 = 2.0 * Y_tone - p3 - p7
        f4 = 2.0 * Y_tone - p1 - p9

        abs1, abs2, abs3, abs4 = torch.abs(f1), torch.abs(f2), torch.abs(f3), torch.abs(f4)
        m = torch.maximum(torch.maximum(abs1, abs2), torch.maximum(abs3, abs4))
        best_f = torch.where(m == abs1, f1, torch.where(m == abs2, f2, torch.where(m == abs3, f3, f4)))

        # 4. Multi-scale Clarity (fine vs med structure contrast)
        Y_4d = Y_tone.unsqueeze(1)  # [B, 1, H, W]
        blur_fine = F.avg_pool2d(Y_4d, kernel_size=5, stride=1, padding=2)
        blur_med  = F.avg_pool2d(Y_4d, kernel_size=13, stride=1, padding=6)
        clarity = (blur_fine - blur_med).squeeze(1)

        Y_enhanced = torch.clamp(Y_tone + sharpen_strength * best_f + clarity_strength * clarity, 0.0, 1.0)

        # 5. Adaptive Dynamic Router
        use_luma_protection = False
        if adaptive_mode:
            max_c = torch.max(chunk, dim=-1)[0]
            min_c = torch.min(chunk, dim=-1)[0]
            sat_mean = ((max_c - min_c) / (max_c + 1e-5)).mean().item()
            # If saturation is high (>= 0.28) -> Already saturated, lock chroma
            if sat_mean >= 0.28:
                use_luma_protection = True

        if use_luma_protection:
            # Mode B: Luma-Protected Multiply (Zero Color Shift, Zero Color Burn)
            Y_mult = Y_base * (1.0 - blend_opacity) + (Y_base * Y_enhanced) * blend_opacity
            R_out = torch.clamp(Y_mult + 1.402 * Cr_base, 0.0, 1.0)
            G_out = torch.clamp(Y_mult - 0.344136 * Cb_base - 0.714136 * Cr_base, 0.0, 1.0)
            B_out = torch.clamp(Y_mult + 1.772 * Cb_base, 0.0, 1.0)
            return torch.stack([R_out, G_out, B_out], dim=-1)
        else:
            # Mode A: RGB Multiply (Cures milky/washed-out haze)
            Cb_tone = -0.168736 * R_tone - 0.331264 * G_tone + 0.5 * B_tone
            Cr_tone = 0.5 * R_tone - 0.418688 * G_tone - 0.081312 * B_tone
            R_lay = torch.clamp(Y_enhanced + 1.402 * Cr_tone, 0.0, 1.0)
            G_lay = torch.clamp(Y_enhanced - 0.344136 * Cb_tone - 0.714136 * Cr_tone, 0.0, 1.0)
            B_lay = torch.clamp(Y_enhanced + 1.772 * Cb_tone, 0.0, 1.0)
            layer_overlay = torch.stack([R_lay, G_lay, B_lay], dim=-1)
            multiply = chunk * layer_overlay
            return torch.clamp(chunk * (1.0 - blend_opacity) + multiply * blend_opacity, 0.0, 1.0)

    def process(self, images, adaptive_mode=True, blend_opacity=0.31, sharpen_strength=1.0, clarity_strength=0.4):
        B = images.shape[0]
        device = images.device
        tone_lut = self.build_tone_lut_tensor(device)

        # Chunk processing to safeguard VRAM on long sequences (1,000-case standard)
        chunk_size = 32
        if B > chunk_size:
            out_chunks = []
            for i in range(0, B, chunk_size):
                sub = images[i : i + chunk_size]
                out_chunks.append(self._process_chunk(sub, tone_lut, adaptive_mode, blend_opacity, sharpen_strength, clarity_strength))
            return (torch.cat(out_chunks, dim=0),)
        else:
            return (self._process_chunk(images, tone_lut, adaptive_mode, blend_opacity, sharpen_strength, clarity_strength),)


NODE_CLASS_MAPPINGS = {
    "SaoChepAdaptiveClarityEnhancer": SaoChepAdaptiveClarityEnhancer
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaoChepAdaptiveClarityEnhancer": "SaoChep Adaptive Clarity Enhancer (ByteDance CapCut On-Flow)"
}
