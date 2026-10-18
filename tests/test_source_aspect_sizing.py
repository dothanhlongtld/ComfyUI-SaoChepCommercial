"""CPU-only checks for source sizing and B-roll fit geometry."""

import ast
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from source_aspect_sizing_node import (
    SaoChepSourceAspectSizing,
    delivery_dimensions,
    native_dimensions,
)


class FakeImageBatch:
    ndim = 4

    def __init__(self, shape):
        self.shape = shape


class SourceAspectSizingTests(unittest.TestCase):
    def test_native_budgets_alignment_and_bounds(self):
        for budget in (576 * 1024, 640 * 1152, 704 * 1280):
            for width, height in (
                (1024, 1024),
                (1920, 1080),
                (720, 1280),
                (4, 1),
                (1, 4),
            ):
                with self.subTest(budget=budget, source=(width, height)):
                    result = native_dimensions(width, height, budget)
                    self.assertTrue(
                        all(256 <= d <= 1280 and d % 32 == 0 for d in result)
                    )

    def test_native_budget_is_independent_of_input_resolution(self):
        for budget in (576 * 1024, 640 * 1152, 704 * 1280):
            for width, height in ((1920, 1080), (1080, 1920), (480, 480), (1280, 960)):
                with self.subTest(budget=budget, source=(width, height)):
                    native = native_dimensions(width, height, budget)
                    self.assertEqual(
                        native, native_dimensions(width * 2, height * 2, budget)
                    )
                    self.assertLess(abs(native[0] * native[1] / budget - 1), 0.06)
        self.assertEqual(native_dimensions(4000, 1000, 704 * 1280), (1280, 320))
        self.assertEqual(native_dimensions(1000, 4000, 704 * 1280), (320, 1280))

    def test_node_uses_supplied_source_shape(self):
        node = SaoChepSourceAspectSizing()
        self.assertEqual(
            node.size(FakeImageBatch((1, 480, 480, 3)), 576 * 1024, 720),
            (768, 768, 720, 720),
        )
        self.assertEqual(
            node.size(FakeImageBatch((240, 1440, 2560, 3)), 576 * 1024, 720),
            (1024, 576, 1280, 720),
        )

    def test_delivery_short_side_ratio_and_even_dimensions(self):
        for short in (720, 1080):
            for width, height in (
                (1024, 1024),
                (1920, 1080),
                (720, 1280),
                (10000, 1),
                (1, 10000),
            ):
                out_w, out_h = delivery_dimensions(width, height, short)
                self.assertEqual(min(out_w, out_h), short)
                self.assertEqual(out_w % 2, 0)
                self.assertEqual(out_h % 2, 0)
                self.assertAlmostEqual(out_w / out_h, width / height, delta=0.005)

    def test_rejects_invalid_dimensions_budget_and_short_side(self):
        for args in ((0, 10, 100), (10, 0, 100), (10, 10, 0)):
            with self.assertRaises(ValueError):
                native_dimensions(*args)
        for short in (0, 721):
            with self.assertRaises(ValueError):
                delivery_dimensions(10, 10, short)
        node = SaoChepSourceAspectSizing()
        for source in (
            FakeImageBatch((0, 10, 10, 3)),
            FakeImageBatch((1, 0, 10, 3)),
            FakeImageBatch((10, 10, 3)),
        ):
            with self.assertRaises(ValueError):
                node.size(source, 100, 720)

    def test_registration_import_precedes_mapping_and_has_display_name(self):
        tree = ast.parse((ROOT / "__init__.py").read_text())
        names = [
            node.lineno
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "source_aspect_sizing_node"
        ]
        mapping = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "NODE_CLASS_MAPPINGS"
                for t in node.targets
            )
        )
        display = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "NODE_DISPLAY_NAME_MAPPINGS"
                for t in node.targets
            )
        )
        self.assertTrue(names and names[0] < mapping.lineno)
        self.assertIn("SaoChepSourceAspectSizing", ast.unparse(mapping.value))
        self.assertIn("SaoChepSourceAspectSizing", ast.unparse(display.value))

    def test_shot_geometry_does_not_stretch_thin_references(self):
        tree = ast.parse((ROOT / "shot_geometry_node.py").read_text())
        geometry = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SaoChepShotGeometry"
        )
        apply = next(
            node
            for node in geometry.body
            if isinstance(node, ast.FunctionDef) and node.name == "apply_geometry"
        )
        assignments = [
            node
            for node in apply.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id in ("scale", "scaled_w", "scaled_h")
                for t in node.targets
            )
        ]
        module = ast.Module(body=assignments, type_ignores=[])
        for source, target, expected in (
            ((10000, 1), (1280, 256), (1280, 1)),
            ((16, 10000), (576, 1024), (2, 1024)),
            ((480, 480), (576, 1024), (576, 576)),
        ):
            ns = {
                "orig_w": source[0],
                "orig_h": source[1],
                "target_width": target[0],
                "target_height": target[1],
                "scale": 2.0,
            }
            exec(compile(module, "shot_geometry_node.py", "exec"), ns)  # noqa: S102 -- owned sizing assignments only
            self.assertEqual((ns["scaled_w"], ns["scaled_h"]), expected)

    def test_timeline_branches_keep_the_delivery_canvas(self):
        tree = ast.parse((ROOT / "adaptive_scene_router_node.py").read_text())
        merger_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "SaoChepAdaptiveTimelineMerger"
        )
        torch = MagicMock()
        fitter = Mock()
        ns = {
            "json": json,
            "np": MagicMock(),
            "torch": torch,
            "_contain_pad_frame": fitter,
            "print": Mock(),
        }
        module = ast.Module(body=[merger_class], type_ignores=[])
        exec(compile(module, "adaptive_scene_router_node.py", "exec"), ns)  # noqa: S102 -- owned merger AST only
        merger = ns["SaoChepAdaptiveTimelineMerger"]()
        original = MagicMock()
        original.shape = (2, 1080, 1920, 3)
        original.__iter__.side_effect = lambda: iter([MagicMock(), MagicMock()])
        rendered = MagicMock()
        rendered.shape = (2, 1280, 720, 3)
        fitted = torch.stack.return_value.to.return_value
        fitted.shape = (2, 1280, 720, 3)

        # Explicit empty-index metadata fits every original to delivery size.
        result, _ = merger.merge_timeline(
            original,
            rendered,
            json.dumps({"is_pure_actor": False, "actor_indices": []}),
        )
        self.assertIs(result, fitted)
        self.assertEqual(fitter.call_count, 2)
        self.assertTrue(
            all(call.args[1:] == (720, 1280) for call in fitter.call_args_list)
        )

        # Mixed routing places generated frames on the fitted originals.
        result, _ = merger.merge_timeline(
            original,
            rendered,
            json.dumps({"is_pure_actor": False, "actor_indices": [0, 1]}),
            blend_frames=0,
        )
        self.assertIs(result, fitted)
        self.assertEqual(fitted.__setitem__.call_count, 2)

        # Pure-actor output already has delivery dimensions and needs no fitting.
        fitter.reset_mock()
        result, _ = merger.merge_timeline(
            original, rendered, json.dumps({"is_pure_actor": True})
        )
        self.assertIs(result, rendered)
        fitter.assert_not_called()

        # Already aligned empty-index metadata keeps original pixels untouched.
        original.shape = rendered.shape
        result, _ = merger.merge_timeline(
            original,
            rendered,
            json.dumps({"is_pure_actor": False, "actor_indices": []}),
        )
        self.assertIs(result, original)
        fitter.assert_not_called()

    def test_contain_fit_geometry_preserves_shape_and_centers_padding(self):
        tree = ast.parse((ROOT / "adaptive_scene_router_node.py").read_text())
        funcs = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("_contain_geometry", "_contain_pad_frame")
        ]
        module = ast.Module(body=funcs, type_ignores=[])

        class FakeFrame:
            shape = (1080, 1920, 3)

        class FakeCV2:
            INTER_LANCZOS4 = 1
            BORDER_CONSTANT = 2

            @staticmethod
            def resize(frame, size, interpolation):
                self.assertEqual(size, (720, 405))
                return ("resized", size)

            @staticmethod
            def copyMakeBorder(frame, top, bottom, left, right, border, value):
                self.assertEqual(frame, ("resized", (720, 405)))
                self.assertEqual((top, bottom, left, right), (437, 438, 0, 0))
                self.assertEqual(border, FakeCV2.BORDER_CONSTANT)
                self.assertEqual(value, (0, 0, 0))
                return ("canvas", 720, 1280)

        ns = {"cv2": FakeCV2}
        exec(compile(module, "adaptive_scene_router_node.py", "exec"), ns)  # noqa: S102 -- owned helper AST only
        self.assertEqual(
            ns["_contain_pad_frame"](FakeFrame(), 720, 1280), ("canvas", 720, 1280)
        )
        fit = ns["_contain_geometry"]
        rw, rh, left, top = fit(1080, 1920, 1280, 720)
        self.assertEqual((rw, rh), (405, 720))
        self.assertEqual((left, top), ((1280 - 405) // 2, 0))


if __name__ == "__main__":
    unittest.main()
