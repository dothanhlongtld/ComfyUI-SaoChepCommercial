"""CPU-safe tests for SaoChepDynamicBackgroundComposite node."""

import ast
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "dynamic_background_composite_node.py"


class DynamicBackgroundCompositeTests(unittest.TestCase):
    def test_node_interface_and_return_names(self):
        """Verify node structure and return types via AST."""
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

    def test_feather_profile_smoothness(self):
        """Test Gaussian feathering math properties on step boundary."""
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("Requires torch for tensor feather test")

        import torch
        import sys

        sys.path.insert(0, str(MODULE_PATH.parent))
        from dynamic_background_composite_node import SaoChepDynamicBackgroundComposite

        node = SaoChepDynamicBackgroundComposite()
        # Step mask [1, 1, 10, 10]: left half 0, right half 1
        mask = torch.zeros((1, 1, 10, 10), dtype=torch.float32)
        mask[:, :, :, 5:] = 1.0

        feathered = node._feather_mask_torch(mask, feather_radius=2)
        # Boundary columns (index 4 and 5) should be softly interpolated between 0 and 1
        val_4 = feathered[0, 0, 5, 4].item()
        val_5 = feathered[0, 0, 5, 5].item()
        self.assertGreater(val_4, 0.0)
        self.assertLess(val_4, 1.0)
        self.assertGreater(val_5, 0.0)
        self.assertLess(val_5, 1.0)


if __name__ == "__main__":
    unittest.main()
