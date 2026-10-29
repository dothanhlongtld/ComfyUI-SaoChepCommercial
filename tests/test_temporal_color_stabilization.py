"""CPU-safe tests for temporal color stabilization and reference chroma drift compensation."""

import importlib.util
import math
import unittest
from pathlib import Path

MODULE_CLARITY_PATH = Path(__file__).resolve().parents[1] / "adaptive_clarity_enhancer_node.py"
MODULE_CHROMA_PATH = Path(__file__).resolve().parents[1] / "reference_chroma_light_node.py"


class TemporalColorStabilizeMathTests(unittest.TestCase):
    def test_smooth_ease_in_weight_boundary_continuity(self):
        """Verify ease-in weight starts exactly at 0.0, reaches 1.0, and has zero boundary jump."""
        ramp_len = 24
        pi_val = 3.141592653589793
        for step in range(ramp_len + 10):
            progress = min(1.0, float(step) / float(ramp_len))
            weight = 0.5 * (1.0 - math.cos(pi_val * progress))
            if step == 0:
                self.assertAlmostEqual(weight, 0.0, places=7)
            elif step >= ramp_len:
                self.assertAlmostEqual(weight, 1.0, places=7)
            else:
                self.assertGreater(weight, 0.0)
                self.assertLess(weight, 1.0)

    def test_chroma_luminance_decomposition(self):
        """Verify chromatic drift decomposition preserves luminance while extracting color cast."""
        # Simulated cold blue/cyan drift: Red down, Green/Blue up
        delta = [0.0, 0.03, 0.05]
        # Y = 0.299 R + 0.587 G + 0.114 B
        y_delta = 0.299 * delta[0] + 0.587 * delta[1] + 0.114 * delta[2]
        chroma_delta = [delta[c] - y_delta for c in range(3)]
        target_correction = [chroma_delta[c] + 0.3 * y_delta for c in range(3)]

        # Target correction must prioritize reducing B and G over R
        self.assertGreater(target_correction[2], target_correction[0])
        self.assertGreater(target_correction[1], target_correction[0])

    def test_pure_python_stabilization_simulation(self):
        """Simulate 120-frame sequence drift correction in pure python."""
        B = 120
        # First 60 frames: R=0.60, G=0.50, B=0.45 (warmth R-B = 0.15)
        # Last 60 frames: R=0.48, G=0.54, B=0.58 (warmth R-B = -0.10, cold cyan)
        frame_means = [[0.60, 0.50, 0.45] for _ in range(60)] + [[0.48, 0.54, 0.58] for _ in range(60)]

        head_len = min(60, max(24, int(B * 0.20)))  # 24
        steady_end = min(B, max(head_len + 48, int(B * 0.50)))  # 72
        stable_window = frame_means[head_len:steady_end]

        stable_baseline = [
            sorted([f[c] for f in stable_window])[len(stable_window) // 2]
            for c in range(3)
        ]
        self.assertAlmostEqual(stable_baseline[0], 0.60, places=4)
        self.assertAlmostEqual(stable_baseline[2], 0.45, places=4)

        tail_start = steady_end
        kernel_size = min(31, max(5, ((B - tail_start) // 4) * 2 + 1))
        pad = kernel_size // 2
        padded = [frame_means[0]] * pad + frame_means + [frame_means[-1]] * pad
        smoothed = []
        for i in range(B):
            window = padded[i : i + kernel_size]
            smoothed.append([sum(w[c] for w in window) / kernel_size for c in range(3)])

        ramp_len = max(12, int((B - tail_start) * 0.35))
        pi_val = 3.141592653589793

        corrected_tail = []
        for t in range(tail_start, B):
            delta = [smoothed[t][c] - stable_baseline[c] for c in range(3)]
            y_delta = 0.299 * delta[0] + 0.587 * delta[1] + 0.114 * delta[2]
            chroma_delta = [delta[c] - y_delta for c in range(3)]
            target_correction = [chroma_delta[c] + 0.3 * y_delta for c in range(3)]

            progress = min(1.0, float(t - tail_start) / float(ramp_len))
            weight = 0.5 * (1.0 - math.cos(pi_val * progress))

            corr_frame = [
                min(1.0, max(0.0, frame_means[t][c] - target_correction[c] * weight))
                for c in range(3)
            ]
            corrected_tail.append(corr_frame)

        # Before correction, final tail frame was -0.10 (cold)
        self.assertAlmostEqual(frame_means[-1][0] - frame_means[-1][2], -0.10, places=2)
        # After correction, final tail frame warmth is restored to ~0.15 (warm, matching first half)
        final_corr_warmth = corrected_tail[-1][0] - corrected_tail[-1][2]
        self.assertGreater(final_corr_warmth, 0.10)
        self.assertAlmostEqual(final_corr_warmth, 0.15, delta=0.03)


class TemporalColorStabilizeTorchTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Requires PyTorch")
    def test_tail_chromatic_drift_is_stabilized_to_baseline(self):
        import torch

        spec = importlib.util.spec_from_file_location("clarity_node", MODULE_CLARITY_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        node = mod.SaoChepAdaptiveClarityEnhancer()

        # 120 frames: first 60 frames warm (R > B), last 60 frames cold (B > R)
        B, H, W = 120, 8, 8
        images = torch.full((B, H, W, 3), 0.5, dtype=torch.float32)

        images[:60, :, :, 0] = 0.60
        images[:60, :, :, 1] = 0.50
        images[:60, :, :, 2] = 0.45

        images[60:, :, :, 0] = 0.48
        images[60:, :, :, 1] = 0.54
        images[60:, :, :, 2] = 0.58

        orig_tail_warmth = (images[-1, :, :, 0] - images[-1, :, :, 2]).mean().item()
        self.assertLess(orig_tail_warmth, 0.0)

        out = node._temporal_color_stabilize(images.clone())

        corr_tail_warmth = (out[-1, :, :, 0] - out[-1, :, :, 2]).mean().item()
        self.assertGreater(corr_tail_warmth, 0.08)
        self.assertAlmostEqual(corr_tail_warmth, 0.15, delta=0.03)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Requires PyTorch")
    def test_short_sequence_is_untouched(self):
        import torch

        spec = importlib.util.spec_from_file_location("clarity_node", MODULE_CLARITY_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        node = mod.SaoChepAdaptiveClarityEnhancer()

        short_imgs = torch.rand(45, 8, 8, 3, dtype=torch.float32)
        out = node._temporal_color_stabilize(short_imgs.clone())
        torch.testing.assert_close(out, short_imgs)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Requires PyTorch")
    def test_uniform_sequence_has_no_drift_correction(self):
        import torch

        spec = importlib.util.spec_from_file_location("clarity_node", MODULE_CLARITY_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        node = mod.SaoChepAdaptiveClarityEnhancer()

        uniform_imgs = torch.full((100, 8, 8, 3), 0.52, dtype=torch.float32)
        out = node._temporal_color_stabilize(uniform_imgs.clone())
        torch.testing.assert_close(out, uniform_imgs)


@unittest.skipUnless(
    all(importlib.util.find_spec(m) is not None for m in ("numpy", "cv2")),
    "Requires NumPy and OpenCV",
)
class ReferenceChromaBroadcastingTests(unittest.TestCase):
    def test_transform_cielab_batch_np_supports_per_frame_offsets(self):
        import numpy as np

        spec = importlib.util.spec_from_file_location("chroma_node", MODULE_CHROMA_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        batch = np.full((4, 8, 8, 3), 0.5, dtype=np.float32)

        out_scalar, _ = mod.transform_cielab_batch_np(batch, 1.5, -1.5)
        self.assertEqual(out_scalar.shape, (4, 8, 8, 3))

        offsets_a = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)[:, None, None]
        offsets_b = np.array([-0.5, -1.0, -1.5, -2.0], dtype=np.float32)[:, None, None]
        out_array, _ = mod.transform_cielab_batch_np(batch, offsets_a, offsets_b)

        self.assertEqual(out_array.shape, (4, 8, 8, 3))
        np.testing.assert_allclose(out_array[2], out_scalar[2], rtol=1e-5, atol=1e-5)

    def test_transform_only_test_mode_executes_without_unbound_locals(self):
        import numpy as np

        spec = importlib.util.spec_from_file_location("chroma_node", MODULE_CHROMA_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        node = mod.ReferenceChromaLightTest()

        dummy_imgs = np.full((4, 16, 16, 3), 0.5, dtype=np.float32)
        ref_img = np.full((1, 16, 16, 3), 0.6, dtype=np.float32)

        graded, preview, report_json = node.apply_chroma_test(
            dummy_imgs,
            ref_img,
            enabled=True,
            test_mode="TRANSFORM_ONLY_TEST",
            processing_backend="cpu_fast",
        )
        self.assertEqual(graded.shape, (4, 16, 16, 3))


if __name__ == "__main__":
    unittest.main()
