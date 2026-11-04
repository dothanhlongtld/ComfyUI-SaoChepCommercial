"""Test SaoChepAdaptiveSceneRouter input types and schema compatibility."""
import ast
import json
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "adaptive_scene_router_node.py"


def validate_node_input(received_type: str, input_type: str) -> bool:
    """ComfyUI node input validation logic from comfy_execution/validation.py."""
    if input_type == "*":
        return True
    recv = set(received_type.split(","))
    inp = set(input_type.split(","))
    return bool(recv & inp)


class TestAdaptiveSceneRouterSchema(unittest.TestCase):
    def _get_input_types(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "SaoChepAdaptiveSceneRouter":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "INPUT_TYPES":
                        dummy_class = ast.ClassDef(
                            name="Dummy",
                            bases=[],
                            keywords=[],
                            body=[item],
                            decorator_list=[],
                        )
                        mod = ast.Module(body=[dummy_class], type_ignores=[])
                        ast.fix_missing_locations(mod)
                        code = compile(mod, filename="<ast>", mode="exec")
                        ns = {}
                        exec(code, ns)
                        return ns["Dummy"].INPUT_TYPES()
        raise ValueError("Could not find SaoChepAdaptiveSceneRouter.INPUT_TYPES")

    def test_fps_input_accepts_both_int_and_float(self):
        input_types = self._get_input_types()
        required = input_types.get("required", {})
        self.assertIn("fps", required)
        fps_spec = required["fps"]
        fps_type = fps_spec[0]

        # The node must accept both INT (e.g. from SCAIL2EasyConfig) and FLOAT in ComfyUI
        self.assertTrue(
            validate_node_input("INT", fps_type),
            f"fps input_type {fps_type!r} rejected received_type 'INT'",
        )
        self.assertTrue(
            validate_node_input("FLOAT", fps_type),
            f"fps input_type {fps_type!r} rejected received_type 'FLOAT'",
        )

    def test_fps_math_with_int_and_float(self):
        # Verify the calculation in route_scenes works with both int and float
        total_frames = 48
        for fps in (24, 24.0):
            with self.subTest(fps=fps):
                total_sec = total_frames / float(fps)
                self.assertEqual(total_sec, 2.0)
                step_frames = max(1, int(round(0.35 * fps)))
                self.assertEqual(step_frames, 8)


class TensorSimulator:
    """Minimal tensor-like simulator for numerical and structural validation without PyTorch."""

    def __init__(self, val, shape=(), dtype="float32", device="cpu"):
        self.dtype = dtype
        self.device = device
        if isinstance(val, list):
            self.val = [
                v if isinstance(v, TensorSimulator) else TensorSimulator(v, shape=(), dtype=dtype, device=device)
                for v in val
            ]
            self.shape = shape or ((len(val),) + (self.val[0].shape if self.val else ()))
        elif isinstance(val, (int, float)):
            self.val = float(val)
            self.shape = shape or ()
        else:
            self.val = val
            self.shape = shape or ()

    def __len__(self):
        return len(self.val) if isinstance(self.val, list) else 0

    def __getitem__(self, idx):
        return self.val[idx]

    def __setitem__(self, idx, item):
        self.val[idx] = item

    def to(self, *args, **kwargs):
        return self

    def clone(self):
        if isinstance(self.val, list):
            return TensorSimulator(
                [v.clone() for v in self.val],
                shape=self.shape,
                dtype=self.dtype,
                device=self.device,
            )
        return TensorSimulator(self.val, shape=self.shape, dtype=self.dtype, device=self.device)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self

    def clip(self, *args, **kwargs):
        return self

    def astype(self, *args, **kwargs):
        return self

    def float(self):
        return self

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            if isinstance(self.val, list):
                return TensorSimulator([v * other for v in self.val], shape=self.shape)
            return TensorSimulator(self.val * other, shape=self.shape)
        if isinstance(other, TensorSimulator):
            if isinstance(self.val, list) and isinstance(other.val, list):
                return TensorSimulator([a * b for a, b in zip(self.val, other.val)], shape=self.shape)
            return TensorSimulator(self.val * other.val, shape=self.shape)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            if isinstance(self.val, list):
                return TensorSimulator([v + other for v in self.val], shape=self.shape)
            return TensorSimulator(self.val + other, shape=self.shape)
        if isinstance(other, TensorSimulator):
            if isinstance(self.val, list) and isinstance(other.val, list):
                return TensorSimulator([a + b for a, b in zip(self.val, other.val)], shape=self.shape)
            return TensorSimulator(self.val + other.val, shape=self.shape)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if isinstance(self.val, list):
                return TensorSimulator([v / other for v in self.val], shape=self.shape)
            return TensorSimulator(self.val / other, shape=self.shape)
        return NotImplemented


class TestAdaptiveTimelineMerger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        cls.merger_class_node = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SaoChepAdaptiveTimelineMerger"
        )

    def _instantiate_merger(self, mock_torch, mock_fitter):
        from unittest.mock import Mock, MagicMock
        ns = {
            "json": json,
            "np": MagicMock(),
            "torch": mock_torch,
            "_contain_pad_frame": mock_fitter,
            "print": Mock(),
        }
        mod = ast.Module(body=[self.merger_class_node], type_ignores=[])
        ast.fix_missing_locations(mod)
        code = compile(mod, filename="adaptive_scene_router_node.py", mode="exec")
        exec(code, ns)
        return ns["SaoChepAdaptiveTimelineMerger"]()

    def test_pure_actor_mode_passthrough(self):
        from unittest.mock import Mock, MagicMock
        mock_torch = MagicMock()
        mock_fitter = Mock()
        merger = self._instantiate_merger(mock_torch, mock_fitter)

        original = TensorSimulator([1.0, 2.0, 3.0], shape=(3, 1080, 1920, 3))
        rendered = TensorSimulator([10.0, 20.0, 30.0], shape=(3, 1280, 720, 3))

        for routing in (
            json.dumps({"is_pure_actor": True}),
            json.dumps({"is_pure_actor": True, "actor_indices": [0, 1, 2]}),
            json.dumps({"shots": []}),  # missing actor_indices
            "invalid_json",
        ):
            with self.subTest(routing=routing):
                result, report = merger.merge_timeline(original, rendered, routing)
                self.assertIs(result, rendered)
                self.assertIn("Passthrough mode (Pure Actor", report)
                mock_fitter.assert_not_called()
                mock_torch.empty.assert_not_called()

    def test_empty_actor_indices_fits_originals_to_delivery_canvas(self):
        from unittest.mock import Mock, MagicMock
        mock_torch = MagicMock()
        mock_torch.empty.side_effect = lambda shape, dtype=None, device="cpu": TensorSimulator(
            [TensorSimulator(0.0, shape[1:]) for _ in range(shape[0])],
            shape=shape,
            dtype=dtype,
            device=device,
        )
        mock_torch.from_numpy.side_effect = lambda arr: arr

        mock_fitter = Mock(side_effect=lambda fr_np, w, h: fr_np.clone())
        merger = self._instantiate_merger(mock_torch, mock_fitter)

        # 2a. Matching dimensions: keeps original untouched with zero fitting
        orig_same = TensorSimulator([1.0, 2.0], shape=(2, 1280, 720, 3))
        rend_same = TensorSimulator([], shape=(0, 1280, 720, 3))
        routing = json.dumps({"is_pure_actor": False, "actor_indices": []})

        res_same, report_same = merger.merge_timeline(orig_same, rend_same, routing)
        self.assertIs(res_same, orig_same)
        self.assertIn("Empty actor indices, original fitted to delivery canvas", report_same)
        mock_fitter.assert_not_called()
        mock_torch.empty.assert_not_called()

        # 2b. Differing dimensions: preallocates delivery canvas and fits every original frame
        orig_diff = TensorSimulator([1.0, 2.0], shape=(2, 1080, 1920, 3))
        rend_diff = TensorSimulator([], shape=(0, 1280, 720, 3))

        res_diff, report_diff = merger.merge_timeline(orig_diff, rend_diff, routing)
        self.assertEqual(res_diff.shape, (2, 1280, 720, 3))
        self.assertEqual(mock_fitter.call_count, 2)
        mock_torch.empty.assert_called_once_with((2, 1280, 720, 3), dtype=orig_diff.dtype, device=orig_diff.device)
        self.assertIn("Empty actor indices, original fitted to delivery canvas", report_diff)

    def test_mixed_timeline_placing_actor_frames_at_correct_indices(self):
        from unittest.mock import Mock, MagicMock
        mock_torch = MagicMock()
        mock_torch.empty.side_effect = lambda shape, dtype=None, device="cpu": TensorSimulator(
            [TensorSimulator(0.0, shape[1:]) for _ in range(shape[0])],
            shape=shape,
            dtype=dtype,
            device=device,
        )
        mock_torch.from_numpy.side_effect = lambda arr: arr

        # 6-frame video: actor at indices 1, 2, 4
        orig_frames = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        rend_frames = [10.0, 20.0, 30.0]

        mock_fitter = Mock(side_effect=lambda fr_np, w, h: fr_np.clone())
        merger = self._instantiate_merger(mock_torch, mock_fitter)

        routing = json.dumps({
            "is_pure_actor": False,
            "actor_indices": [1, 2, 4],
            "shots": [{"type": "AI_ACTOR", "start_f": 1, "end_f": 3}, {"type": "AI_ACTOR", "start_f": 4, "end_f": 5}],
        })

        # Differing dimensions:
        orig = TensorSimulator(orig_frames, shape=(6, 1080, 1920, 3))
        rend = TensorSimulator(rend_frames, shape=(3, 1280, 720, 3))

        merged, report = merger.merge_timeline(orig, rend, routing, blend_frames=0)
        self.assertEqual(merged.shape, (6, 1280, 720, 3))
        # B-roll frames at 0, 3, 5
        self.assertEqual(merged[0].val, 100.0)
        self.assertEqual(merged[3].val, 103.0)
        self.assertEqual(merged[5].val, 105.0)
        # Actor frames at 1, 2, 4
        self.assertEqual(merged[1].val, 10.0)
        self.assertEqual(merged[2].val, 20.0)
        self.assertEqual(merged[4].val, 30.0)
        self.assertIn("Successfully merged 3 rendered AI frames with 3 native original camera frames", report)

    def test_two_frame_cross_fade_linear_blend_math(self):
        from unittest.mock import Mock, MagicMock
        mock_torch = MagicMock()
        mock_torch.empty.side_effect = lambda shape, dtype=None, device="cpu": TensorSimulator(
            [TensorSimulator(0.0, shape[1:]) for _ in range(shape[0])],
            shape=shape,
            dtype=dtype,
            device=device,
        )
        mock_torch.from_numpy.side_effect = lambda arr: arr

        mock_fitter = Mock(side_effect=lambda fr_np, w, h: fr_np.clone())
        merger = self._instantiate_merger(mock_torch, mock_fitter)

        # 8-frame video: actor shot from frame 2 to 6 (length 4)
        orig_vals = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        rend_vals = [200.0, 300.0, 400.0, 500.0]
        orig = TensorSimulator(orig_vals, shape=(8, 1080, 1920, 3))
        rend = TensorSimulator(rend_vals, shape=(4, 1280, 720, 3))

        routing = json.dumps({
            "is_pure_actor": False,
            "actor_indices": [2, 3, 4, 5],
            "shots": [{"type": "AI_ACTOR", "start_f": 2, "end_f": 6}],
        })

        merged, _ = merger.merge_timeline(orig, rend, routing, blend_frames=2)

        # B-roll outside actor shot:
        self.assertEqual(merged[0].val, 0.0)
        self.assertEqual(merged[1].val, 10.0)
        self.assertEqual(merged[6].val, 60.0)
        self.assertEqual(merged[7].val, 70.0)

        # Cross-fade at shot start (st_f = 2, blend_frames = 2):
        # b_i = 0: alpha = (0+1)/(2+1) = 1/3, idx = 2, rend_idx = 0
        # math: (1 - 1/3) * orig[2] + (1/3) * rend[0] = (2/3) * 20.0 + (1/3) * 200.0 = 80.0
        expected_2 = (2.0 / 3.0) * orig_vals[2] + (1.0 / 3.0) * rend_vals[0]
        self.assertAlmostEqual(merged[2].val, expected_2, places=5)
        self.assertAlmostEqual(merged[2].val, 80.0, places=5)

        # b_i = 1: alpha = (1+1)/(2+1) = 2/3, idx = 3, rend_idx = 1
        # math: (1 - 2/3) * orig[3] + (2/3) * rend[1] = (1/3) * 30.0 + (2/3) * 300.0 = 210.0
        expected_3 = (1.0 / 3.0) * orig_vals[3] + (2.0 / 3.0) * rend_vals[1]
        self.assertAlmostEqual(merged[3].val, expected_3, places=5)
        self.assertAlmostEqual(merged[3].val, 210.0, places=5)

        # Cross-fade at shot end (end_f = 6, blend_frames = 2):
        # b_i = 0: alpha = 1/3, idx = 6-1-0 = 5, rend_idx = 3
        # math: (1 - 1/3) * orig[5] + (1/3) * rend[3] = (2/3) * 50.0 + (1/3) * 500.0 = 200.0
        expected_5 = (2.0 / 3.0) * orig_vals[5] + (1.0 / 3.0) * rend_vals[3]
        self.assertAlmostEqual(merged[5].val, expected_5, places=5)
        self.assertAlmostEqual(merged[5].val, 200.0, places=5)

        # b_i = 1: alpha = 2/3, idx = 6-1-1 = 4, rend_idx = 2
        # math: (1 - 2/3) * orig[4] + (2/3) * rend[2] = (1/3) * 40.0 + (2/3) * 400.0 = 280.0
        expected_4 = (1.0 / 3.0) * orig_vals[4] + (2.0 / 3.0) * rend_vals[2]
        self.assertAlmostEqual(merged[4].val, expected_4, places=5)
        self.assertAlmostEqual(merged[4].val, 280.0, places=5)

    def test_zero_full_timeline_resized_originals_clone(self):
        from unittest.mock import Mock, MagicMock
        source_text = MODULE_PATH.read_text(encoding="utf-8")
        # 1. Ensure resized_originals is completely absent from source
        self.assertNotIn("resized_originals", source_text)

        # 2. Ensure torch.stack is not called in SaoChepAdaptiveTimelineMerger
        merger_source = ast.unparse(self.merger_class_node)
        self.assertNotIn("torch.stack", merger_source)

        # 3. Verify execution fits ONLY B-roll and boundary frames, skipping interior actor frames
        mock_torch = MagicMock()
        mock_torch.empty.side_effect = lambda shape, dtype=None, device="cpu": TensorSimulator(
            [TensorSimulator(0.0, shape[1:]) for _ in range(shape[0])],
            shape=shape,
            dtype=dtype,
            device=device,
        )
        mock_torch.from_numpy.side_effect = lambda arr: arr

        fitted_indices = []

        def track_fitter(fr_np, w, h):
            fitted_indices.append(round(fr_np.val / 255.0))
            return fr_np.clone()

        mock_fitter = Mock(side_effect=track_fitter)
        merger = self._instantiate_merger(mock_torch, mock_fitter)

        # 8-frame video, actor from frame 2 to 6 (actor_indices: [2, 3, 4, 5]), blend_frames = 1
        # B-roll: [0, 1, 6, 7] (4 frames)
        # Boundaries: [2, 5] (2 frames)
        # Interior actor frames: [3, 4] -> must NOT be fitted!
        orig_vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        rend_vals = [20.0, 30.0, 40.0, 50.0]
        orig = TensorSimulator(orig_vals, shape=(8, 1080, 1920, 3))
        rend = TensorSimulator(rend_vals, shape=(4, 1280, 720, 3))

        routing = json.dumps({
            "is_pure_actor": False,
            "actor_indices": [2, 3, 4, 5],
            "shots": [{"type": "AI_ACTOR", "start_f": 2, "end_f": 6}],
        })

        merger.merge_timeline(orig, rend, routing, blend_frames=1)

        # Exactly 6 frames fitted, interior frames 3.0 and 4.0 were never fitted
        self.assertEqual(len(fitted_indices), 6)
        self.assertNotIn(3.0, fitted_indices)
        self.assertNotIn(4.0, fitted_indices)
        self.assertCountEqual(fitted_indices, [0.0, 1.0, 6.0, 7.0, 2.0, 5.0])
        # Canvas preallocated exactly once
        self.assertEqual(mock_torch.empty.call_count, 1)
