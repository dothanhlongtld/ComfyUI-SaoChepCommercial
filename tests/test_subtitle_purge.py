"""CPU-safe tests for SaoChepSubtitlePurge node and Zero-Copy Bypass."""

import ast
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "subtitle_purge_node.py"


class SubtitlePurgeTests(unittest.TestCase):
    def test_input_types_and_interface(self):
        """Verify INPUT_TYPES structure and return types via AST."""
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        input_types_found = False
        return_types_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "RETURN_TYPES":
                        return_types_found = True
            elif isinstance(node, ast.FunctionDef) and node.name == "INPUT_TYPES":
                input_types_found = True

        self.assertTrue(input_types_found)
        self.assertTrue(return_types_found)

    def test_zero_copy_bypass_when_disabled(self):
        """Bypass mode must return the exact object with 0 overhead and 0 alteration."""
        import sys

        sys.path.insert(0, str(MODULE_PATH.parent))
        from subtitle_purge_node import SaoChepSubtitlePurge

        node = SaoChepSubtitlePurge()
        dummy_input = ["fake_frame_reference"]
        out_images, report = node.purge_subtitles(dummy_input, enabled=False)

        # Exact identity verification (Zero-Copy)
        self.assertIs(out_images, dummy_input)
        self.assertEqual(report, "BYPASS_DISABLED")

    def test_mocked_execution_clean_bypass_prevents_name_error(self):
        """Simulate present cv2/numpy dependencies to test execution path and ensure no NameError."""
        import sys
        sys.path.insert(0, str(MODULE_PATH.parent))
        import subtitle_purge_node
        from subtitle_purge_node import SaoChepSubtitlePurge

        class FakeFrame:
            shape = (1000, 1000, 3)

        fake_np = MagicMock()
        fake_np.uint8 = "uint8"
        fake_np.asarray.side_effect = lambda x, dtype=None: x
        fake_cv2 = MagicMock()
        fake_cv2.cvtColor.return_value = MagicMock()

        node = SaoChepSubtitlePurge()
        fake_images = [FakeFrame(), FakeFrame()]

        with patch.object(subtitle_purge_node, "np", fake_np), \
             patch.object(subtitle_purge_node, "cv2", fake_cv2), \
             patch.object(node, "_detect_text_in_strip", return_value=(False, None)):
            out_images, report = node.purge_subtitles(fake_images, enabled=True)
            self.assertIs(out_images, fake_images)
            self.assertIn("BYPASS_CLEAN_NO_SUBTITLE_DETECTED", report)

    def test_clean_video_bypass(self):
        """Uniform / clean bottom zone must trigger zero-copy bypass without altering input."""
        import importlib.util

        if importlib.util.find_spec("numpy") is None or importlib.util.find_spec("cv2") is None:
            raise unittest.SkipTest("Requires numpy and cv2")
        import numpy as np
        import sys

        sys.path.insert(0, str(MODULE_PATH.parent))
        from subtitle_purge_node import SaoChepSubtitlePurge

        node = SaoChepSubtitlePurge()
        clean_img = [np.full((100, 100, 3), 128, dtype=np.uint8)]
        out_images, report = node.purge_subtitles(clean_img, enabled=True, bottom_zone_start=0.75)

        if "OPENCV_UNAVAILABLE" not in report:
            self.assertIs(out_images, clean_img)
            self.assertIn("BYPASS_CLEAN_NO_SUBTITLE_DETECTED", report)


if __name__ == "__main__":
    unittest.main()
