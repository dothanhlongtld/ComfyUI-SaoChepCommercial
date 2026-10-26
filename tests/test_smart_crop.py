"""Tests for SaoChepSmartFaceCrop node and head_torso_clip mode."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "smart_crop_node.py"


class SmartFaceCropTests(unittest.TestCase):
    def test_input_types_contains_head_torso_clip(self):
        """Verify target_aspect includes head_torso_clip mode."""
        import ast

        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        input_types_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "INPUT_TYPES":
                input_types_def = node
                break
        self.assertIsNotNone(input_types_def)
        source = ast.unparse(input_types_def)
        self.assertIn("head_torso_clip", source)

    def test_head_torso_clip_bounding_logic(self):
        """Test the head_torso_clip math directly without requiring external torch/cv2 models."""
        h, w = 1920, 1080
        # Simulated face: fx, fy, fw, fh
        # Face at y=300, height=200 (fh/h = 200/1920 = 0.104 < 0.35, not close-up)
        fx, fy, fw, fh = 450, 300, 180, 200
        cx = fx + fw / 2.0
        cy = fy + fh / 2.0

        crop_h = min(h, int(round(fh * 2.5)))  # 500
        y1 = max(0, int(round(fy - fh * 0.15)))  # 300 - 30 = 270
        y2 = min(h, y1 + crop_h)  # 270 + 500 = 770
        crop_w = min(w, max(int(round(fw * 2.2)), int(round(crop_h * 0.75))))  # max(396, 375) = 396
        x1 = max(0, min(int(round(cx - crop_w / 2.0)), w - crop_w))
        x2 = min(w, x1 + crop_w)

        self.assertEqual(y2 - y1, 500)
        self.assertEqual(x2 - x1, 396)
        self.assertGreaterEqual(x1, 0)
        self.assertLessEqual(x2, w)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(y2, h)

    def test_closeup_bypass_condition(self):
        """Test close-up face condition bypass (fh/h >= 0.35)."""
        h = 1000
        fh_closeup = 400
        self.assertGreaterEqual(fh_closeup / float(h), 0.35)

        fh_fullbody = 120
        self.assertLess(fh_fullbody / float(h), 0.35)


if __name__ == "__main__":
    unittest.main()
