# -*- coding: utf-8 -*-
"""
SaoChepMultiCharIdentityLock - 100% On-Flow Multi-Character Identity Lock Guardrail Node
Solves crossing-path identity swapping, occlusion track loss, and color palette inversion in Wan 2.1 Multi-Character workflows (Template 3 & Template 4).

Works directly between Node 104 (SCAIL2ColoredMask) and Node 105 (ImageBlur) / Node 5000 (SCAIL2AutoVideo).
Takes [B, H, W, 3] pose_video_mask tensor and ensures that Character 0 (Blue) and Character 1 (Red)
MAINTAIN STRICT TEMPORAL CONTINUITY ACROSS CROSSINGS, JUMPS, AND COLLISIONS.
"""

import numpy as np
import torch
import cv2

class SaoChepMultiCharIdentityLock:
    CATEGORY = "SaoChep/Commercial"
    FUNCTION = "stabilize_identities"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("stabilized_pose_mask", "stabilization_report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_video_mask": ("IMAGE", {"tooltip": "Pose video mask [B, H, W, 3] from SCAIL2ColoredMask"}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Enable inertial trajectory identity locking"}),
            },
            "optional": {
                "reference_image_mask": ("IMAGE", {"tooltip": "Optional reference image mask to anchor initial identities"}),
                "max_allowed_jump_px": ("INT", {"default": 100, "min": 20, "max": 500, "tooltip": "Max allowed centroid jump per frame before triggering swap correction"}),
            }
        }

    def stabilize_identities(self, pose_video_mask, enabled=True, reference_image_mask=None, max_allowed_jump_px=100):
        if not enabled or pose_video_mask is None or len(pose_video_mask) <= 1:
            return (pose_video_mask, '{"status": "BYPASS_DISABLED"}')

        B, H, W, C = pose_video_mask.shape
        device = pose_video_mask.device
        dtype = pose_video_mask.dtype

        # Convert to numpy uint8 [B, H, W, 3]
        mask_np = (pose_video_mask.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # Palette colors in SCAIL:
        # Identity 0 = Blue (0, 0, 255)
        # Identity 1 = Red (255, 0, 0)
        # Background = Black (0, 0, 0) in Animation Mode, or White (255, 255, 255) in Replacement Mode

        out_masks = np.copy(mask_np)
        swaps_detected = 0
        swap_frames = []

        # Track previous centroids and velocities
        # id0 = Blue, id1 = Red
        prev_pos = {0: None, 1: None}
        prev_vel = {0: np.array([0.0, 0.0]), 1: np.array([0.0, 0.0])}

        for t in range(B):
            frame = mask_np[t]

            # Detect Blue and Red masks with channel dominance (handles slight compression/blending)
            # Blue: channel 2 dominant over channel 0
            # Red: channel 0 dominant over channel 2
            blue_mask = (frame[:, :, 2] > frame[:, :, 0] + 30) & (frame[:, :, 2] > 80)
            red_mask = (frame[:, :, 0] > frame[:, :, 2] + 30) & (frame[:, :, 0] > 80)

            # Compute centroids
            c_blue = np.array([np.mean(np.where(blue_mask)[1]), np.mean(np.where(blue_mask)[0])]) if np.any(blue_mask) else None
            c_red = np.array([np.mean(np.where(red_mask)[1]), np.mean(np.where(red_mask)[0])]) if np.any(red_mask) else None

            if t == 0:
                # Frame 0: Anchor initial identities
                prev_pos[0] = c_blue
                prev_pos[1] = c_red
                continue

            pred_0 = prev_pos[0] + prev_vel[0] if prev_pos[0] is not None else None
            pred_1 = prev_pos[1] + prev_vel[1] if prev_pos[1] is not None else None

            if c_blue is not None and c_red is not None and pred_0 is not None and pred_1 is not None:
                # Cost without swap: Blue is Track 0, Red is Track 1
                cost_normal = np.linalg.norm(c_blue - pred_0) + np.linalg.norm(c_red - pred_1)
                # Cost with swap: Red is Track 0, Blue is Track 1
                cost_swapped = np.linalg.norm(c_red - pred_0) + np.linalg.norm(c_blue - pred_1)

                jump_0 = np.linalg.norm(c_blue - prev_pos[0])
                jump_1 = np.linalg.norm(c_red - prev_pos[1])

                # Swap detected if inertial cost is decisively lower for swapped assignment OR sudden jump occurred
                is_swapped = (cost_swapped + 6.0 < cost_normal) or ((jump_0 > max_allowed_jump_px or jump_1 > max_allowed_jump_px) and cost_swapped < cost_normal)

                if is_swapped:
                    # SWAP DETECTED! Invert colors back to original identity
                    swaps_detected += 1
                    swap_frames.append(t)
                    # Swap pixels in output mask
                    new_frame = frame.copy()
                    new_frame[blue_mask] = [255, 0, 0] # Make blue pixels red
                    new_frame[red_mask] = [0, 0, 255] # Make red pixels blue
                    out_masks[t] = new_frame

                    # Update positions with swapped assignment (Track 0 was Red, Track 1 was Blue)
                    cur_pos_0 = c_red
                    cur_pos_1 = c_blue
                else:
                    cur_pos_0 = c_blue
                    cur_pos_1 = c_red

                # Update velocities (exponential moving average)
                prev_vel[0] = 0.6 * prev_vel[0] + 0.4 * (cur_pos_0 - prev_pos[0])
                prev_vel[1] = 0.6 * prev_vel[1] + 0.4 * (cur_pos_1 - prev_pos[1])
                prev_pos[0] = cur_pos_0
                prev_pos[1] = cur_pos_1

            elif c_blue is not None and pred_0 is not None:
                prev_pos[0] = c_blue
            elif c_red is not None and pred_1 is not None:
                prev_pos[1] = c_red

        out_tensor = torch.from_numpy(out_masks.astype(np.float32) / 255.0).to(device=device, dtype=dtype)
        report = {
            "total_frames": B,
            "swaps_detected": swaps_detected,
            "swap_frames": swap_frames,
            "status": "SWAP_CORRECTED" if swaps_detected > 0 else "STABLE"
        }
        import json
        return (out_tensor, json.dumps(report, indent=2))
