"""CPU-safe regressions for bounded adaptive-clarity processing."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Requires ComfyUI's PyTorch dependency")
import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "adaptive_clarity_enhancer_node.py"
spec = importlib.util.spec_from_file_location(
    "adaptive_clarity_under_test", MODULE_PATH
)
clarity_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clarity_module)
SaoChepAdaptiveClarityEnhancer = clarity_module.SaoChepAdaptiveClarityEnhancer


def legacy_process(
    node,
    images,
    adaptive_mode=True,
    blend_opacity=0.31,
    sharpen_strength=1.0,
    clarity_strength=0.4,
):
    """Pre-fix process orchestration, retained here as a small CPU baseline."""
    calc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    working = images.to(calc_device)
    working = node._temporal_boundary_smooth(working)
    working = node._temporal_head_stabilize(working)
    working = node._temporal_color_stabilize(working)
    tone_lut = node.build_tone_lut_tensor(calc_device)
    groups = [
        node._process_chunk(
            working[i : i + 32],
            tone_lut,
            adaptive_mode,
            blend_opacity,
            sharpen_strength,
            clarity_strength,
        )
        for i in range(0, working.shape[0], 32)
    ]
    if not groups:
        return (torch.empty_like(working),)
    return (torch.cat(groups, dim=0).to(images.device),)


class AdaptiveClarityMemoryTests(unittest.TestCase):
    def setUp(self):
        cpu_only = patch.object(torch.cuda, "is_available", return_value=False)
        cpu_only.start()
        self.addCleanup(cpu_only.stop)
        self.node = SaoChepAdaptiveClarityEnhancer()

    def test_matches_legacy_cpu_processing_and_preserves_input(self):
        for frames in (9, 29, 33, 41, 301):
            for saturated in (False, True):
                for adaptive in (False, True):
                    with self.subTest(
                        frames=frames, saturated=saturated, adaptive=adaptive
                    ):
                        images = torch.rand(frames, 8, 8, 3, dtype=torch.float32)
                        if not saturated:
                            images = (
                                images.mean(-1, keepdim=True).expand_as(images).clone()
                            )
                        original = images.clone()
                        expected = legacy_process(
                            self.node, images.clone(), adaptive_mode=adaptive
                        )[0]
                        actual = self.node.process(images, adaptive_mode=adaptive)[0]
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(images, original, rtol=0, atol=0)
                        self.assertEqual(actual.dtype, torch.float32)
                        self.assertEqual(
                            actual.shape[0],
                            frames - (1 if frames % 4 == 1 and frames > 16 else 0),
                        )

    def test_preserves_global_head_stabilization_for_long_video(self):
        # The legacy baseline uses head_len=60 at 300 frames after boundary trim.
        images = torch.rand(301, 8, 8, 3)
        expected = legacy_process(self.node, images.clone())[0]
        actual = self.node.process(images)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_empty_input_returns_empty_cpu_tensor(self):
        images = torch.empty(0, 8, 8, 3, dtype=torch.float32)
        result = self.node.process(images)[0]
        self.assertEqual(result.shape, images.shape)
        self.assertEqual(result.device.type, "cpu")

    def test_gpu_path_transfers_only_fixed_groups_and_keeps_results_on_cpu(self):
        images = torch.rand(66, 8, 8, 3, dtype=torch.float32)
        transfers = []
        groups = []
        real_to = torch.Tensor.to

        def simulated_to(tensor, *args, **kwargs):
            device = kwargs.get("device", args[0] if args else None)
            if str(device) == "cuda":
                transfers.append(tuple(tensor.shape))
                return tensor  # Simulate device transfer without requiring CUDA.
            return real_to(tensor, *args, **kwargs)

        def process_group(
            chunk,
            tone_lut,
            adaptive_mode,
            blend_opacity,
            sharpen_strength,
            clarity_strength,
        ):
            groups.append((tuple(chunk.shape), adaptive_mode, chunk.dtype))
            return chunk.clone()

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.Tensor, "to", simulated_to),
            patch.object(
                self.node, "build_tone_lut_tensor", return_value=torch.zeros(256)
            ),
            patch.object(self.node, "_process_chunk", side_effect=process_group),
        ):
            result = self.node.process(images, adaptive_mode=False)[0]

        self.assertEqual(transfers, [(32, 8, 8, 3), (32, 8, 8, 3), (2, 8, 8, 3)])
        self.assertEqual([group[0][0] for group in groups], [32, 32, 2])
        self.assertTrue(all(not shape or shape[0] <= 32 for shape, _, _ in groups))
        self.assertTrue(all(not adaptive for _, adaptive, _ in groups))
        self.assertTrue(all(dtype == torch.float32 for _, _, dtype in groups))
        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(result.shape, images.shape)


if __name__ == "__main__":
    unittest.main()
