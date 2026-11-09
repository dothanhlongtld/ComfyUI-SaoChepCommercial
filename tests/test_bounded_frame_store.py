"""CPU-safe lossless storage checks for the bounded worker nodes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "saochep_frame_store", Path(__file__).parents[1] / "frame_store.py"
)
assert SPEC and SPEC.loader
frame_store = importlib.util.module_from_spec(SPEC)
import sys

sys.modules[SPEC.name] = frame_store
SPEC.loader.exec_module(frame_store)

PREFIX = "aiworkergrid/" + "a" * 32


def make_store(tmp_path, array):
    with frame_store.FrameStoreWriter(tmp_path, PREFIX, "test") as writer:
        writer.append(array)
        return writer.finish()


def test_lossless_values_dtype_order_and_partial_cross_chunk_range(tmp_path):
    values = np.arange(35 * 2 * 3 * 3, dtype=np.float32).reshape(35, 2, 3, 3)
    store = make_store(tmp_path, values)
    assert tuple(chunk.count for chunk in store.chunks) == (16, 16, 3)
    actual = store.read(14, 5)
    assert actual.dtype == values.dtype
    np.testing.assert_array_equal(actual, values[14:19])
    np.testing.assert_array_equal(np.concatenate(list(store.iter_batches())), values)


def test_multiple_numeric_dtypes_preserved(tmp_path):
    for dtype in (np.float16, np.float32, np.float64, np.uint8, np.int16):
        values = np.arange(2 * 2 * 3 * 3, dtype=dtype).reshape(2, 2, 3, 3)
        store = make_store(tmp_path, values)
        actual = store.read(0, 2)
        assert actual.dtype == values.dtype
        np.testing.assert_array_equal(actual, values)
        store.discard()


def test_corruption_is_reported_without_returning_data(tmp_path):
    values = np.zeros((2, 2, 3, 3), dtype=np.float32)
    store = make_store(tmp_path, values)
    chunk = store._directory() / store.chunks[0].name
    chunk.write_bytes(chunk.read_bytes()[:-1])
    with pytest.raises(frame_store.FrameStoreError):
        store.read(0, 1)


def test_invalid_ranges_are_rejected(tmp_path):
    store = make_store(tmp_path, np.zeros((3, 2, 3, 3), dtype=np.float32))
    for start, count in ((-1, 1), (0, 17), (3, 1), (0, 0), (True, 1)):
        with pytest.raises(frame_store.FrameStoreError):
            store.read(start, count)
