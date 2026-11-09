"""Exercise the real clarity shader with NumPy arithmetic, not an identity shader.

This proves grouping, scalar routing and formulas on CPU; it is not a claim about
PyTorch CUDA reduction order, kernels or whole-catalog visual acceptance.
"""

from __future__ import annotations

import ast
import math
import types
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from bounded_tensor_support import TensorDouble, all_pixels
from bounded_test_support import make_store


class NumericTensor(TensorDouble):
    def __getitem__(self, index):
        if isinstance(index, TensorDouble):
            index = index.array
        return NumericTensor(self.array[index])

    def __add__(self, other):
        return NumericTensor(self.array + np.asarray(other, dtype=self.array.dtype))

    __radd__ = __add__

    def __sub__(self, other):
        return NumericTensor(self.array - np.asarray(other, dtype=self.array.dtype))

    def __rsub__(self, other):
        return NumericTensor(np.asarray(other, dtype=self.array.dtype) - self.array)

    def __mul__(self, other):
        return NumericTensor(self.array * np.asarray(other, dtype=self.array.dtype))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return NumericTensor(self.array / np.asarray(other, dtype=self.array.dtype))

    def __rtruediv__(self, other):
        return NumericTensor(np.asarray(other, dtype=self.array.dtype) / self.array)

    def __pow__(self, power):
        return NumericTensor(self.array**power)

    def __neg__(self):
        return NumericTensor(-self.array)

    def __eq__(self, other):
        return self.array == np.asarray(other)

    def __le__(self, other):
        return self.array <= np.asarray(other)

    def round(self):
        return NumericTensor(np.round(self.array))

    def long(self):
        return NumericTensor(self.array.astype(np.int64))

    def unsqueeze(self, dim):
        return NumericTensor(np.expand_dims(self.array, dim))

    def squeeze(self, dim):
        return NumericTensor(np.squeeze(self.array, axis=dim))

    def mean(self, dim=None):
        return NumericTensor(self.array.mean(axis=dim))


def shader_functions():
    source = (
        Path(__file__).parents[1] / "adaptive_clarity_enhancer_node.py"
    ).read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
    selected: list[ast.stmt] = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_process_chunk", "build_tone_lut_tensor"}
    ]
    assert len(selected) == 2

    def pad(value, padding, mode):
        assert mode == "replicate"
        left, right, top, bottom = padding
        return NumericTensor(
            np.pad(
                np.asarray(value),
                ((0, 0), (0, 0), (top, bottom), (left, right)),
                mode="edge",
            )
        )

    def pool(value, kernel_size, stride, padding):
        assert stride == 1
        values = np.pad(
            np.asarray(value), ((0, 0), (0, 0), (padding, padding), (padding, padding))
        )
        # NumPy 2.2's stubs omit the documented tuple-axis overload.
        windows = cast(Any, np.lib.stride_tricks.sliding_window_view)(
            values, (kernel_size, kernel_size), axis=(-2, -1)
        )
        return NumericTensor(windows.mean(axis=(-2, -1)))

    torch = types.SimpleNamespace(
        clamp=lambda value, low, high: NumericTensor(
            np.clip(np.asarray(value), low, high)
        ),
        abs=lambda value: NumericTensor(np.abs(np.asarray(value))),
        maximum=lambda left, right: NumericTensor(
            np.maximum(np.asarray(left), np.asarray(right))
        ),
        where=lambda cond, yes, no: NumericTensor(
            np.where(cond, np.asarray(yes), np.asarray(no))
        ),
        stack=lambda values, dim: NumericTensor(
            np.stack([np.asarray(v) for v in values], axis=dim)
        ),
        linspace=lambda start, end, steps, **_kw: NumericTensor(
            np.linspace(start, end, steps, dtype=np.float32)
        ),
        pow=lambda value, power: NumericTensor(
            np.power(np.asarray(value), np.asarray(power))
        ),
        exp=lambda value: NumericTensor(np.exp(np.asarray(value))),
        tensor=lambda value, **_kw: NumericTensor(np.asarray(value, dtype=np.float32)),
    )
    namespace = {"torch": torch, "F": types.SimpleNamespace(pad=pad, avg_pool2d=pool)}
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(Path(__file__)), "exec"), namespace)  # noqa: S102 - trusted local shader under test
    return namespace["_process_chunk"], namespace["build_tone_lut_tensor"]


def temporal_oracle(pixels):
    out = (
        pixels[1:].copy()
        if len(pixels) % 4 == 1 and len(pixels) > 16
        else pixels.copy()
    )
    count = len(out)
    if count >= 30:
        means = out.mean(axis=(1, 2))
        head = min(60, max(24, int(count * 0.20)))
        window = means[head : min(head + 48, count)]
        if len(window):
            middle = (len(window) - 1) // 2
            baseline = np.partition(window, middle, axis=0)[middle]
            for index in range(head):
                delta = means[index] - baseline
                if np.max(np.abs(delta)) > 0.008:
                    weight = np.float32(0.5) * (
                        np.float32(1) + np.cos(np.float32(math.pi * index / head))
                    )
                    out[index] = np.clip(out[index] - delta * weight, 0, 1)
    if count >= 60:
        means = out.mean(axis=(1, 2))
        head = min(60, max(24, int(count * 0.20)))
        end = min(count, max(head + 48, int(count * 0.50)))
        window = means[head:end]
        if len(window) and end < count - 8:
            middle = (len(window) - 1) // 2
            baseline = np.partition(window, middle, axis=0)[middle]
            kernel = min(31, max(5, ((count - end) // 4) * 2 + 1))
            padded = np.pad(means, ((kernel // 2, kernel // 2), (0, 0)), mode="edge")
            smoothed = np.asarray(
                [padded[i : i + kernel].mean(axis=0) for i in range(count)]
            )
            ramp = max(12, int((count - end) * 0.35))
            for index in range(end, count):
                delta = smoothed[index] - baseline
                y = 0.299 * delta[0] + 0.587 * delta[1] + 0.114 * delta[2]
                correction = (delta - y + 0.3 * y) * (
                    0.5 * (1 - math.cos(math.pi * min(1.0, (index - end) / ramp)))
                )
                if np.max(np.abs(correction)) > 0.004:
                    out[index] = np.clip(out[index] - correction, 0, 1)
    return out


@pytest.mark.parametrize("count", [17, 33, 64, 66, 161])
def test_original_32_group_shader_matches_bounded_after_temporal_corrections(
    memory, tmp_path, count
):
    rng = np.random.default_rng(10)
    frames = rng.uniform(0.05, 0.2, (count, 5, 6, 3)).astype(np.float32)
    frames[16:32] += np.float32(0.40)
    frames[48:64] += np.float32(0.30)
    frames[-8:] = np.minimum(frames[-8:] + np.float32(0.25), 1)
    source, _ = make_store(memory, tmp_path, frames)
    # Non-divisor budget exercises group32 alignment independently of chunk files.
    memory._layout.__globals__["MAX_BATCH_PIXELS"] = 5 * 6 * 7
    shader, build_lut = shader_functions()
    calls = []

    class Clarity:
        def build_tone_lut_tensor(self, device):
            return build_lut(self, device)

        def _process_chunk(self, batch, lut, *settings, scene_luma=None):
            calls.append((len(batch), scene_luma))
            return TensorDouble(
                np.asarray(
                    shader(
                        self,
                        NumericTensor(np.asarray(batch)),
                        lut,
                        *settings,
                        scene_luma=scene_luma,
                    )
                )
            )

    node = Clarity()
    memory._BINDINGS["clarity"] = Clarity
    actual = all_pixels(memory.AWGMemoryClarity().enhance(source)[0])
    corrected = temporal_oracle(frames)
    lut = node.build_tone_lut_tensor("cpu")
    expected = np.concatenate(
        [
            np.asarray(
                shader(
                    node,
                    NumericTensor(corrected[i : i + 32]),
                    lut,
                    True,
                    0.31,
                    1.0,
                    0.4,
                )
            )
            for i in range(0, len(corrected), 32)
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    assert max(n for n, _ in calls) <= 7
    assert sum(n for n, _ in calls) == len(corrected)
    offset = 0
    for batch_count, luma in calls:
        assert offset // 32 == (offset + batch_count - 1) // 32
        group = corrected[
            (offset // 32) * 32 : min((offset // 32 + 1) * 32, len(corrected))
        ]
        y = 0.299 * group[..., 0] + 0.587 * group[..., 1] + 0.114 * group[..., 2]
        assert luma == pytest.approx(float(y.mean()), abs=2e-7)
        offset += batch_count


def test_shader_original_group_is_not_equivalent_to_naive_16_frame_calls():
    shader, build_lut = shader_functions()
    rng = np.random.default_rng(11)
    frames = rng.uniform(0.05, 0.15, (32, 8, 8, 3)).astype(np.float32)
    frames[16:] += np.float32(0.40)
    lut = build_lut(None, "cpu")
    original = np.asarray(
        shader(None, NumericTensor(frames), lut, True, 0.31, 1.0, 0.4)
    )
    naive = np.concatenate(
        [
            np.asarray(
                shader(
                    None, NumericTensor(frames[i : i + 16]), lut, True, 0.31, 1.0, 0.4
                )
            )
            for i in (0, 16)
        ]
    )
    assert np.max(np.abs(original - naive)) > 1e-5
    y = 0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]
    correct = np.concatenate(
        [
            np.asarray(
                shader(
                    None,
                    NumericTensor(frames[i : i + 16]),
                    lut,
                    True,
                    0.31,
                    1.0,
                    0.4,
                    scene_luma=float(y.mean()),
                )
            )
            for i in (0, 16)
        ]
    )
    np.testing.assert_allclose(correct, original, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("luma", [0.0, 0.1225, 0.34999, 0.35, 0.35001, 0.8])
@pytest.mark.parametrize("adaptive", [False, True])
def test_shader_override_keeps_thresholds_and_default_path(luma, adaptive):
    shader, build_lut = shader_functions()
    frames = np.full((4, 5, 6, 3), luma, dtype=np.float32)
    lut = build_lut(None, "cpu")
    y = 0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]
    default = shader(None, NumericTensor(frames), lut, adaptive, 0.31, 1.0, 0.4)
    explicit = shader(
        None,
        NumericTensor(frames),
        lut,
        adaptive,
        0.31,
        1.0,
        0.4,
        scene_luma=float(y.mean()),
    )
    np.testing.assert_array_equal(np.asarray(default), np.asarray(explicit))
