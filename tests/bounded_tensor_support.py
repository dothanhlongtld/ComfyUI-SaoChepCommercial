"""NumPy tensor facade and fixtures; not CUDA/PyTorch numerical proof."""

from __future__ import annotations

import types
from unittest.mock import Mock

import numpy as np
import pytest


class TensorDouble:
    """NumPy-backed test double for control flow only; never claimed as Torch proof."""

    def __init__(self, values):
        self.array = np.asarray(values)

    def __array__(self, dtype=None, copy=None):
        return (
            np.array(self.array, dtype=dtype, copy=copy)
            if copy is not None
            else np.asarray(self.array, dtype=dtype)
        )

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def device(self):
        return "cpu"

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return TensorDouble(self.array[index])

    def __setitem__(self, index, value):
        self.array[index] = np.asarray(value)

    def __add__(self, other):
        return TensorDouble(self.array + np.asarray(other))

    __radd__ = __add__

    def __sub__(self, other):
        return TensorDouble(self.array - np.asarray(other))

    def __rsub__(self, other):
        return TensorDouble(np.asarray(other) - self.array)

    def __mul__(self, other):
        return TensorDouble(self.array * np.asarray(other, dtype=self.array.dtype))

    __rmul__ = __mul__

    def cpu(self):
        return self

    detach = cpu

    def clone(self):
        return TensorDouble(self.array.copy())

    def numpy(self):
        return self.array

    def item(self):
        return self.array.item()

    def to(self, *_args, dtype=None, **_kwargs):
        return TensorDouble(
            self.array.astype(dtype) if dtype is not None else self.array
        )

    def view(self, *shape):
        return TensorDouble(self.array.reshape(*shape))

    def mean(self, dim):
        return TensorDouble(self.array.mean(axis=dim))

    def unsqueeze(self, dim):
        return TensorDouble(np.expand_dims(self.array, dim))

    def squeeze(self, dim):
        return TensorDouble(np.squeeze(self.array, axis=dim))

    def transpose(self, first, second):
        return TensorDouble(np.swapaxes(self.array, first, second))


def lower_median(values, dim):
    array = np.asarray(values)
    middle = (array.shape[dim] - 1) // 2
    return types.SimpleNamespace(
        values=TensorDouble(
            np.take(np.partition(array, middle, axis=dim), middle, axis=dim)
        )
    )


@pytest.fixture
def memory(frames_module, monkeypatch):
    torch = types.SimpleNamespace(
        from_numpy=TensorDouble,
        empty=lambda shape, dtype=None, **_kw: TensorDouble(
            np.empty(shape, dtype=dtype)
        ),
        empty_like=lambda value: TensorDouble(np.empty_like(np.asarray(value))),
        median=lower_median,
        abs=lambda value: TensorDouble(np.abs(np.asarray(value))),
        max=lambda value: np.max(np.asarray(value)),
        cos=lambda value: TensorDouble(np.cos(np.asarray(value))),
        tensor=lambda value, dtype=None, **_kw: TensorDouble(
            np.asarray(value, dtype=dtype)
        ),
        clamp=lambda value, minimum, maximum: TensorDouble(
            np.clip(np.asarray(value), minimum, maximum)
        ),
        device=lambda value: value,
        cuda=types.SimpleNamespace(is_available=lambda: True, empty_cache=Mock()),
        float32=np.float32,
        nn=types.SimpleNamespace(
            functional=types.SimpleNamespace(
                pad=lambda value, pad, **_kw: TensorDouble(
                    np.pad(np.asarray(value), ((0, 0), (0, 0), pad), mode="edge")
                )
            )
        ),
    )
    monkeypatch.setattr(frames_module, "_torch", lambda: torch)
    monkeypatch.setattr(frames_module, "_interrupt", Mock())
    return frames_module


def all_pixels(store):
    # Full-video materialization is a tiny test oracle, never worker code.
    return np.concatenate(list(store.iter_batches()))
