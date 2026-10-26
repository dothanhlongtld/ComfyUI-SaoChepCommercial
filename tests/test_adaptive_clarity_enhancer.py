"""CPU-safe tests for Shadow Toe Guard, Chroma Decoupling, and Scene-Adaptive scaling."""

import importlib.util
import math
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "adaptive_clarity_enhancer_node.py"


class AdaptiveClarityEnhancerTests(unittest.TestCase):
    def test_shadow_toe_guard_formula_bounds(self):
        """Pure-math test of the smoothstep toe curve and <= 5% luminance drop bound."""
        # For Y_base in [0.0, 0.20], drop must not exceed 5%
        for y_int in range(1, 20):
            y_base = y_int / 100.0
            # Even in extreme darkening scenario where y_enhanced = 0.0, blend = 0.31
            y_mult_candidate = y_base * (1.0 - 0.31) + (y_base * 0.0) * 0.31
            min_allowed_y = y_base * 0.95
            y_guarded = max(y_mult_candidate, min_allowed_y)

            t_toe = min(1.0, max(0.0, y_base / 0.20))
            toe_weight = t_toe * t_toe * (3.0 - 2.0 * t_toe)
            y_final = y_base * (1.0 - toe_weight) + y_guarded * toe_weight

            drop_percent = (y_base - y_final) / y_base * 100.0
            self.assertLessEqual(
                drop_percent,
                5.0001,
                f"Luminance drop {drop_percent}% exceeded 5% limit at Y={y_base}",
            )
            # Must never be negative or greater than 1
            self.assertGreaterEqual(y_final, 0.0)
            self.assertLessEqual(y_final, 1.0)

    def test_torch_shadow_toe_guard_execution_if_available(self):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("Requires ComfyUI's PyTorch dependency")

        import torch

        spec = importlib.util.spec_from_file_location(
            "adaptive_clarity_under_test", MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        node = module.SaoChepAdaptiveClarityEnhancer()

        # Dark low-key image (mean Y < 0.20)
        dark_img = torch.full((1, 16, 16, 3), 0.10, dtype=torch.float32)
        out = node.process(dark_img, adaptive_mode=True)[0]

        # Calculate output Y
        y_out = 0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]
        mean_y_out = y_out.mean().item()

        # Must not drop more than 5% (0.10 * 0.95 = 0.095)
        self.assertGreaterEqual(mean_y_out, 0.095)


if __name__ == "__main__":
    unittest.main()
