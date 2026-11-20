import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "comfy_custom_nodes.SaoChepCommercial.saochep_headswap"
for name in ("comfy_custom_nodes", "comfy_custom_nodes.SaoChepCommercial"):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    module = sys.modules[name]
    module.__path__ = [str(ROOT.parent)] if name == "comfy_custom_nodes" else [str(ROOT)]
spec = importlib.util.spec_from_file_location(
    PACKAGE, ROOT / "saochep_headswap" / "__init__.py",
    submodule_search_locations=[str(ROOT / "saochep_headswap")])
module = importlib.util.module_from_spec(spec)
sys.modules[PACKAGE] = module
spec.loader.exec_module(module)
guard = __import__(PACKAGE + ".sam_guard", fromlist=["validate_transition"])


def shifted_masks(shape=(40, 60), offset=8):
    before = np.zeros(shape, bool)
    after = np.zeros(shape, bool)
    before[10:20, 10:20] = True
    after[10:20, 10 + offset:20 + offset] = True
    return before, after


def test_compensated_translation_is_opt_in():
    before, after = shifted_masks()
    with pytest.raises(ValueError, match="Uncertain SAM head continuity"):
        guard.validate_transition(before, after)
    guard.validate_transition(before, after, motion_compensated=True)


def test_compensation_does_not_relax_bbox_movement_limit():
    before, after = shifted_masks(offset=11)
    with pytest.raises(ValueError, match="Uncertain SAM head continuity"):
        guard.validate_transition(before, after, motion_compensated=True)


def test_compensation_preserves_area_and_empty_guards():
    before, after = shifted_masks()
    after[:] = False
    after[10:20, 9:31] = True
    with pytest.raises(ValueError, match="Uncertain SAM head continuity"):
        guard.validate_transition(before, after, motion_compensated=True)
    with pytest.raises(ValueError, match="Empty SAM head"):
        guard.validate_transition(np.zeros_like(before), after, motion_compensated=True)


def test_compensation_rejects_low_shape_overlap():
    before = np.zeros((50, 60), bool)
    after = np.zeros_like(before)
    before[10:20, 10:20] = True
    after[10:12, 18:28] = True
    with pytest.raises(ValueError, match="Uncertain SAM head continuity"):
        guard.validate_transition(before, after, motion_compensated=True)


def test_clipped_translation_uses_original_area_denominator():
    before = np.zeros((30, 30), bool)
    after = np.zeros_like(before)
    before[8:18, 0:10] = True
    after[8:18, 0:5] = True
    # Rounded bbox-center delta=-2 clips two columns; overlap is computed using
    # the un-clipped minimum original area (the smaller after mask here).
    metrics = guard.transition_metrics(before, after, motion_compensated=True)
    assert metrics[4] == pytest.approx(1.0)
    guard.validate_transition(before, after, motion_compensated=True)


@pytest.mark.parametrize("motion_compensated", [False, True])
def test_broadcastable_shape_mismatch_is_rejected_in_both_modes(motion_compensated):
    before = np.zeros((1, 20), bool)
    after = np.zeros((10, 20), bool)
    before[0, 4:8] = True
    after[0, 4:8] = True
    with pytest.raises(TypeError, match="matching shapes"):
        guard.validate_transition(before, after, motion_compensated=motion_compensated)


@pytest.mark.parametrize("shape", [(20,), (2, 10, 10)])
def test_non_2d_masks_are_rejected(shape):
    before = np.ones(shape, bool)
    after = np.ones(shape, bool)
    with pytest.raises(TypeError, match="must be 2D"):
        guard.validate_transition(before, after)


def test_motion_compensated_mode_requires_actual_bool():
    before, after = shifted_masks()
    with pytest.raises(TypeError, match="must be a bool"):
        guard.validate_transition(before, after, motion_compensated=1)
