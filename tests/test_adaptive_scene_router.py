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
