"""Run with: python -m unittest discover -s tests -v."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import commercial_naturalize_export_node as export


class HandoffRegressionTests(unittest.TestCase):
    @unittest.skipUnless(
        all(importlib.util.find_spec(name) for name in ("torch", "cv2", "numpy", "PIL")),
        "Requires ComfyUI's tensor/image dependencies",
    )
    def test_crossfade_preserves_resized_originals(self):
        import torch
        from adaptive_scene_router_node import SaoChepAdaptiveTimelineMerger

        routing = json.dumps({
            "is_pure_actor": False,
            "actor_indices": [1, 2, 3, 4],
            "shots": [{"type": "AI_ACTOR", "start_f": 1, "end_f": 5}],
        })
        for original_size in (2, 4):
            for blend_frames in (0, 1):
                with self.subTest(original_size=original_size, blend_frames=blend_frames):
                    original = torch.zeros((6, original_size, original_size, 3))
                    rendered = torch.ones((4, 4, 4, 3))
                    merged, _ = SaoChepAdaptiveTimelineMerger().merge_timeline(
                        original, rendered, routing, blend_frames=blend_frames,
                    )
                    expected = torch.tensor([
                        0, 0.5 if blend_frames else 1, 1, 1,
                        0.5 if blend_frames else 1, 0,
                    ], dtype=merged.dtype)[:, None, None, None].expand(6, 4, 4, 3)
                    torch.testing.assert_close(merged, expected)
                    torch.testing.assert_close(original, torch.zeros_like(original))
                    torch.testing.assert_close(rendered, torch.ones_like(rendered))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Requires FFmpeg")
    def test_export_preserves_full_timeline_and_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i", "testsrc2=size=32x32:rate=8:duration=1",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
            ], check=True, capture_output=True, timeout=30)
            folders = types.SimpleNamespace(get_output_directory=lambda: tmp)
            with patch.object(export, "folder_paths", folders):
                for fps, frame_count in ((0, 8), (8, 8), (16, 16)):
                    with self.subTest(output_fps=fps):
                        result = export.SaoChepCommercialExport().export_master(
                            str(source), output_fps=fps, preset="veryfast",
                            cleanup_intermediate=False,
                        )
                        output = result["result"][0][1][0]
                        probe = subprocess.run([
                            "ffprobe", "-v", "error", "-count_frames", "-show_streams",
                            "-of", "json", output,
                        ], check=True, capture_output=True, text=True, timeout=30)
                        streams = json.loads(probe.stdout)["streams"]
                        video = next(s for s in streams if s["codec_type"] == "video")
                        audio = next(s for s in streams if s["codec_type"] == "audio")
                        self.assertEqual(int(video["nb_read_frames"]), frame_count)
                        self.assertAlmostEqual(float(video["duration"]), 1.0, places=2)
                        self.assertAlmostEqual(float(audio["duration"]), 1.0, places=2)
                        self.assertAlmostEqual(float(video["start_time"]), 0.0, places=2)
                        self.assertTrue(source.exists())


if __name__ == "__main__":
    unittest.main()
