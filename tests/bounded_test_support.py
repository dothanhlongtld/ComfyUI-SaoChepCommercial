"""Isolated canonical-module loader for CPU unit checks (not ComfyUI startup)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

PREFIX = "aiworkergrid/" + "a" * 32


@pytest.fixture
def frames_module():
    repository = Path(__file__).resolve().parents[1]
    package = types.ModuleType("saochep_bounded_under_test")
    package.__path__ = [str(repository)]
    sys.modules[package.__name__] = package
    loaded: list[Any] = []
    try:
        for name in ("frame_store", "bounded_video_nodes"):
            qualified = f"{package.__name__}.{name}"
            spec = importlib.util.spec_from_file_location(
                qualified, repository / f"{name}.py"
            )
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified] = module
            spec.loader.exec_module(module)
            loaded.append(module)
        store, nodes = loaded
        nodes.os, nodes.shutil = store.os, store.shutil
        yield nodes
    finally:
        for module in loaded:
            sys.modules.pop(module.__name__, None)
        sys.modules.pop(package.__name__, None)


def make_store(module, root: Path, frames=None):
    if frames is None:
        frames = np.arange(35 * 2 * 3 * 3, dtype=np.float32).reshape(35, 2, 3, 3)
    with module.FrameStoreWriter(root, PREFIX, "generated") as writer:
        writer.append(frames)
        store = writer.finish()
    return store, frames
