# -*- coding: utf-8 -*-
"""
SaoChepDynamicBackgroundComposite - 100% On-Flow Dynamic Background Preservation Node
======================================================================================
Composites rendered Wan 2.1 AI character onto clean original camera background
using soft-feathered alpha blending.

Features:
- Seamless 2-4px Gaussian feathering on character mask edges to eliminate hard halos.
- Robust resolution matching between character layer and driving background plate.
- Preserves 100% background dynamics (moving cars, shaking leaves, camera pans).
"""

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None


class SaoChepDynamicBackgroundComposite:
    CATEGORY = "SaoChep/Commercial"
    FUNCTION = "composite"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_images": ("IMAGE", {"tooltip": "Original camera background video frames [B, H, W, C]"}),
                "character_images": ("IMAGE", {"tooltip": "Rendered AI character video frames [B, H, W, C]"}),
                "character_mask": ("MASK", {"tooltip": "Character alpha mask [B, H, W] or [1, H, W]"}),
            },
            "optional": {
                "feather_pixels": ("INT", {"default": 3, "min": 0, "max": 16, "step": 1, "tooltip": "Gaussian blur edge feathering radius (px)"}),
                "blend_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Overall blending opacity"}),
            }
        }

    def _feather_mask_torch(self, mask_4d: "torch.Tensor", feather_radius: int) -> "torch.Tensor":
        """Applies separable Gaussian blur on PyTorch mask tensor [B, 1, H, W]."""
        if feather_radius <= 0 or F is None:
            return mask_4d

        ksize = feather_radius * 2 + 1
        sigma = max(0.5, float(feather_radius) / 2.0)
        coords = torch.arange(ksize, dtype=mask_4d.dtype, device=mask_4d.device) - feather_radius
        g_1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        g_1d = g_1d / g_1d.sum()

        kernel_x = g_1d.view(1, 1, 1, ksize)
        kernel_y = g_1d.view(1, 1, ksize, 1)

        pad = feather_radius
        padded = F.pad(mask_4d, (pad, pad, pad, pad), mode="replicate")
        blurred_x = F.conv2d(padded, kernel_x)
        blurred_xy = F.conv2d(blurred_x, kernel_y)
        return torch.clamp(blurred_xy, 0.0, 1.0)

    def composite(self, background_images, character_images, character_mask,
                  feather_pixels=3, blend_opacity=1.0):
        if background_images is None or character_images is None:
            return (character_images or background_images, "EMPTY_INPUTS")

        is_torch = torch is not None and isinstance(character_images, torch.Tensor)

        if not is_torch:
            # Fallback for non-torch environments
            return (character_images, "BYPASS_NON_TORCH_INPUT")

        device = character_images.device
        bg = background_images.to(device)
        char = character_images

        num_frames = min(bg.shape[0], char.shape[0])
        bg = bg[:num_frames]
        char = char[:num_frames]

        H_bg, W_bg = bg.shape[1], bg.shape[2]
        H_ch, W_ch = char.shape[1], char.shape[2]

        # Match dimensions if necessary
        if (H_ch, W_ch) != (H_bg, W_bg):
            # Resize character to match background plate [B, C, H, W]
            char_perm = char.permute(0, 3, 1, 2)
            char_perm = F.interpolate(char_perm, size=(H_bg, W_bg), mode="bilinear", align_corners=False)
            char = char_perm.permute(0, 2, 3, 1)

        # Prepare character mask [B, 1, H, W]
        mask = character_mask.to(device)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)

        # Expand mask across temporal frames if single mask given
        if mask.shape[0] == 1 and num_frames > 1:
            mask = mask.expand(num_frames, -1, -1, -1)
        else:
            mask = mask[:num_frames]

        # Match mask spatial resolution to background
        if mask.shape[2:] != (H_bg, W_bg):
            mask = F.interpolate(mask, size=(H_bg, W_bg), mode="bilinear", align_corners=False)

        # Apply soft feathering to eliminate edge artifacts
        if feather_pixels > 0:
            feathered_mask = self._feather_mask_torch(mask, feather_pixels)
        else:
            feathered_mask = torch.clamp(mask, 0.0, 1.0)

        # Modulate by opacity
        alpha = (feathered_mask * float(blend_opacity)).permute(0, 2, 3, 1)  # [B, H, W, 1]

        # Alpha composite: char * alpha + bg * (1 - alpha)
        composited = char * alpha + bg * (1.0 - alpha)
        composited = torch.clamp(composited, 0.0, 1.0)

        report = f"COMPOSITED_DYNAMIC_BG: {num_frames} frames ({W_bg}x{H_bg}), feather={feather_pixels}px, opacity={blend_opacity:.2f}"
        return (composited, report)
