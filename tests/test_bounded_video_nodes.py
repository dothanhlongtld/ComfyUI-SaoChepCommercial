"""CPU bounded-node calculations/control flow; not Torch/GPU parity proof."""

from __future__ import annotations

import inspect
import json
import math
import types
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from bounded_tensor_support import TensorDouble, all_pixels
from bounded_test_support import PREFIX, make_store


@pytest.mark.parametrize(
    "frames,pose_frames,plan",
    [
        (1, 1, [(0, 1)]),
        (4, 4, [(0, 1)]),
        (5, 5, [(0, 5)]),
        (85, 85, [(0, 85)]),
        (86, 86, [(0, 85)]),
        (89, 89, [(0, 85), (85, 9)]),
        (120, 120, [(0, 85), (85, 37)]),
        (120, 45, [(0, 45)]),
        (0, 5, [(0, 5)]),
    ],
)
def test_generation_uses_plan_conditioning_anchor_and_exact_frozen_tail(
    memory, tmp_path, monkeypatch, frames, pose_frames, plan
):
    condition_calls, sampling_calls = [], []
    planner = Mock(return_value=plan)

    def condition(*args, **kwargs):
        previous = kwargs["previous_frames"]
        previous_values = None if previous is None else previous.numpy().copy()
        condition_calls.append((args, previous_values))
        start = args[10] - (min(len(previous), args[11]) if previous is not None else 0)
        return (
            "conditioned positive",
            "conditioned negative",
            {"length": args[5], "start": start},
        )

    def sample(*args):
        sampling_calls.append(args)
        return types.SimpleNamespace(result=(args[-1],))

    class Decode:
        def decode(self, _vae, latent):
            values = np.arange(
                latent["start"], latent["start"] + latent["length"], dtype=np.float32
            )
            return (
                TensorDouble(
                    np.broadcast_to(
                        values[:, None, None, None], (len(values), 1, 1, 3)
                    ).copy()
                ),
            )

    memory._BINDINGS.update(
        scail=types.SimpleNamespace(_plan=planner),
        condition=types.SimpleNamespace(execute=condition),
        sampler=types.SimpleNamespace(execute=sample),
        decode=Decode,
    )
    original_import = memory.importlib.import_module
    monkeypatch.setattr(
        memory.importlib,
        "import_module",
        lambda name: (
            types.SimpleNamespace(get_output_directory=lambda: str(tmp_path))
            if name == "folder_paths"
            else original_import(name)
        ),
    )
    inputs = {
        "frames": frames,
        "video_frames": pose_frames,
        "pose_video": TensorDouble(np.zeros((pose_frames, 1, 1, 3), dtype=np.float32)),
        "length": 85,
        "previous_frame_count": 5,
        "positive": "text positive",
        "negative": "text negative",
        "vae": "vae",
        "width": 576,
        "height": 1024,
        "replacement_mode": True,
        "reference_image": "reference",
        "clip_vision_output": "vision",
        "pose_video_mask": "mask",
        "reference_image_mask": "reference mask",
        "model": "model",
        "seed": 42,
        "cfg": 1.0,
        "sampler": "euler",
        "sigmas": "sigmas",
    }
    result = memory.AWGMemoryGenerate().generate(PREFIX, **inputs)[0]
    expected_count = frames if frames > 0 else pose_frames
    planner.assert_called_once_with(expected_count, pose_frames, 85, 5, False)
    produced = plan[0][1] + sum(length - 5 for _, length in plan[1:])
    expected = np.minimum(np.arange(expected_count), produced - 1)
    np.testing.assert_array_equal(all_pixels(result)[:, 0, 0, 0], expected)
    for index, (args, previous) in enumerate(condition_calls):
        assert args[0:2] == ("text positive", "text negative")
        assert args[6:10] == (1, 1.0, 0.0, 1.0)
        if index:
            assert previous is not None
            assert len(previous) == 5
            np.testing.assert_array_equal(
                previous[:, 0, 0, 0], np.arange(args[10] - 5, args[10])
            )
        else:
            assert previous is None
    for args in sampling_calls:
        assert args[1:6] == (
            True,
            42,
            1.0,
            "conditioned positive",
            "conditioned negative",
        )
    assert all(chunk.count <= 16 for chunk in result.chunks)


def test_frame_sequence_reads_each_chunk_once_and_closes(memory, tmp_path, monkeypatch):
    source, values = make_store(memory, tmp_path)
    original_read = memory.FrameStore.read
    calls = []

    def counted(self, start, count):
        calls.append((start, count))
        return original_read(self, start, count)

    monkeypatch.setattr(memory.FrameStore, "read", counted)
    sequence = memory.FrameSequence(source)
    metadata_frame = sequence[0]
    assert not np.shares_memory(metadata_frame.numpy(), sequence._batch)
    np.testing.assert_array_equal(
        np.stack([value.numpy() for value in sequence]), values
    )
    assert calls == [(0, 16), (16, 16), (32, 3)]
    sequence.close()
    assert sequence._batch is None
    with pytest.raises(memory.FrameStoreError, match="closed"):
        sequence[0]


def sample_node(reference_a, sample_as):
    results = [{"valid": True, "a": reference_a, "b": 0, "L": 50}] + [
        {"valid": True, "a": value, "b": 0, "L": 50} for value in sample_as
    ]
    return types.SimpleNamespace(
        _sample_cheek_color=Mock(side_effect=results),
        _select_adaptive_batch_frames=lambda requested, _height, _width: requested,
    )


def test_chroma_preserves_global_bypass_even_with_large_drift(memory, tmp_path):
    source, _ = make_store(memory, tmp_path)
    node = sample_node(0, [-2, 0, 2])
    offsets = memory._chroma_offsets(
        source,
        TensorDouble(np.full((1, 2, 3, 3), 0.5, dtype=np.float32)),
        node,
        {"sample_count": 3, "strength": 1.0},
    )
    assert offsets is None
    assert node._sample_cheek_color.call_count == 4
    # Reference conversion clips/truncates 127.5 to 127, not round-to-128.
    assert node._sample_cheek_color.call_args_list[0].args[0][0, 0, 0] == 127


def test_chroma_sample_indices_short_clip_validity_and_interpolation(memory, tmp_path):
    source, _ = make_store(memory, tmp_path)
    node = sample_node(3, [-2, 0, 2])
    offsets = memory._chroma_offsets(
        source,
        TensorDouble(np.zeros((1, 2, 3, 3), dtype=np.float32)),
        node,
        {"sample_count": 3, "strength": 1.0, "max_ab_offset": 2.0},
    )
    expected_indices = [3, 17, 31]
    expected = np.interp(np.arange(35), expected_indices, [2, 2, 1]).astype(np.float32)
    np.testing.assert_array_equal(offsets[0], expected)
    tiny, _ = make_store(memory, tmp_path, np.zeros((2, 2, 3, 3), dtype=np.float32))
    tiny_node = sample_node(3, [0, 0])
    assert (
        memory._chroma_offsets(
            tiny, TensorDouble(np.zeros((1, 2, 3, 3))), tiny_node, {}
        )
        is None
    )
    assert (
        tiny_node._sample_cheek_color.call_count == 3
    )  # Nine requests deduplicate to two frames.


def test_chroma_writes_bounded_groups_and_reclaims_after_success(memory, tmp_path):
    source, frames = make_store(
        memory, tmp_path, np.full((35, 2, 3, 3), 0.25, dtype=np.float32)
    )
    calls = []
    node = sample_node(3, [0] * 9)

    def transform(batch, a, b, **kwargs):
        calls.append((len(batch), a, b, kwargs))
        return batch, 0

    memory._BINDINGS.update(
        chroma=lambda: node,
        chroma_module=types.SimpleNamespace(transform_cielab_batch_torch=transform),
    )
    result = memory.AWGMemoryChroma().correct(
        source,
        TensorDouble(np.zeros((1, 2, 3, 3))),
        enabled=True,
        processing_backend="gpu",
    )[0]
    np.testing.assert_array_equal(all_pixels(result), frames)
    assert [entry[0] for entry in calls] == [16, 16, 3]
    assert not (source.output_root / source.prefix / source.directory_name).exists()


def test_resize_reuses_upstream_settings_with_bounded_input(memory, tmp_path):
    source, frames = make_store(memory, tmp_path)
    calls = []

    class Resize:
        def resize(self, image, **settings):
            calls.append((len(image), settings))
            return (image, 3, 2, None)

    memory._BINDINGS["resize"] = Resize
    settings = {
        "width": 3,
        "height": 2,
        "keep_proportion": "pad",
        "upscale_method": "bicubic",
        "divisible_by": 2,
        "pad_color": "0, 0, 0",
        "crop_position": "center",
        "device": "gpu",
        "unique_id": "9001",
    }
    result = memory.AWGMemoryResize().resize(source, **settings)[0]
    np.testing.assert_array_equal(all_pixels(result), frames)
    assert [count for count, _ in calls] == [16, 16, 3]
    assert all(actual == settings for _, actual in calls)


def test_clarity_uses_lower_median(memory):
    means = TensorDouble(np.arange(30 * 3, dtype=np.float32).reshape(30, 3))
    head, reference = memory._head_plan(means)
    assert head == 24
    np.testing.assert_array_equal(
        reference.numpy(), means.numpy()[26]
    )  # Lower middle of six, not average.


@pytest.mark.parametrize("count", [1, 24, 29, 30, 60, 85, 120])
def test_clarity_boundary_head_then_recomputed_means_and_bounded_lut(
    memory, tmp_path, monkeypatch, count
):
    frames = np.linspace(0.0, 1.0, count * 2 * 3 * 3, dtype=np.float32).reshape(
        count, 2, 3, 3
    )
    source, _ = make_store(memory, tmp_path, frames)
    calls: list[tuple[Any, ...]] = []

    class Clarity:
        def build_tone_lut_tensor(self, device):
            calls.append(("lut", device))
            return "lut"

        def _process_chunk(self, batch, lut, *settings, scene_luma=None):
            calls.append(("chunk", len(batch), lut, settings, scene_luma))
            return batch

    memory._BINDINGS["clarity"] = Clarity
    observed_means = []
    original_tail = memory._tail_plan

    def record_tail(means):
        observed_means.append(means.numpy().copy())
        return original_tail(means)

    monkeypatch.setattr(memory, "_tail_plan", record_tail)
    result = memory.AWGMemoryClarity().enhance(source)[0]
    crop = 1 if count % 4 == 1 and count > 16 else 0
    expected = frames[crop:].copy()
    size = len(expected)
    means_before = expected.mean(axis=(1, 2))
    if size >= 30:
        head = min(60, max(24, int(size * 0.20)))
        window = means_before[head : min(head + 48, size)]
        reference = np.partition(window, (len(window) - 1) // 2, axis=0)[
            (len(window) - 1) // 2
        ]
        for index in range(head):
            delta = means_before[index] - reference
            if np.max(np.abs(delta)) > 0.008:
                weight = 0.5 * (1 + np.cos(np.float32(math.pi * index / head)))
                expected[index] = np.clip(expected[index] - delta * weight, 0, 1)
    np.testing.assert_allclose(observed_means[0], expected.mean(axis=(1, 2)), atol=1e-6)
    if size >= 30:
        assert not np.array_equal(observed_means[0][:24], means_before[:24])
    assert len(result) == count - crop
    assert max(entry[1] for entry in calls if entry[0] == "chunk") <= 16
    assert all(
        entry[3] == (True, 0.31, 1.0, 0.4) for entry in calls if entry[0] == "chunk"
    )
    assert all(entry[4] is not None for entry in calls if entry[0] == "chunk")
    # This asserts control-flow/statistics under a NumPy test double, not actual Torch parity.
    if size >= 60:
        head = min(60, max(24, int(size * 0.20)))
        end = min(size, max(head + 48, int(size * 0.50)))
        if end < size - 8:
            corrected = expected.mean(axis=(1, 2))
            window = corrected[head:end]
            reference = np.partition(window, (len(window) - 1) // 2, axis=0)[
                (len(window) - 1) // 2
            ]
            kernel = min(31, max(5, ((size - end) // 4) * 2 + 1))
            padded = np.pad(
                corrected, ((kernel // 2, kernel // 2), (0, 0)), mode="edge"
            )
            smoothed = np.asarray(
                [padded[index : index + kernel].mean(axis=0) for index in range(size)]
            )
            ramp = max(12, int((size - end) * 0.35))
            for index in range(end, size):
                delta = smoothed[index] - reference
                luminance = 0.299 * delta[0] + 0.587 * delta[1] + 0.114 * delta[2]
                correction = (delta - luminance + 0.3 * luminance) * (
                    0.5 * (1 - math.cos(math.pi * min(1.0, (index - end) / ramp)))
                )
                if np.max(np.abs(correction)) > 0.004:
                    expected[index] = np.clip(expected[index] - correction, 0, 1)
    np.testing.assert_allclose(all_pixels(result), expected, atol=1e-6)


def test_encoding_reuses_vhs_contract_and_reclaims_only_on_success(memory, tmp_path):
    source, frames = make_store(memory, tmp_path)
    expected = {"ui": {"gifs": []}, "result": ((True, ["raw.mp4"]),)}
    received = []

    class VHS:
        def combine_video(self, images, **settings):
            received.append(settings)
            np.testing.assert_array_equal(
                np.stack([image.numpy() for image in images]), frames
            )
            return expected

    memory._BINDINGS["vhs"] = VHS
    settings = {
        "format": "video/h264-mp4",
        "loop_count": 0,
        "pingpong": False,
        "save_output": True,
        "frame_rate": 24.0,
        "crf": 20,
        "pix_fmt": "yuv420p",
        "save_metadata": False,
        "trim_to_audio": False,
        "filename_prefix": PREFIX + "/raw",
        "audio": "audio",
        "prompt": {"2702": "prepared graph"},
        "extra_pnginfo": {"workflow": {"extra": {}}},
        "unique_id": "2702",
        "manual_format_widgets": None,
    }
    assert memory.AWGMemoryEncode().encode(source, **settings) is expected
    assert received == [settings]
    assert not (source.output_root / source.prefix / source.directory_name).exists()


@pytest.mark.parametrize(
    "failure", [RuntimeError("encoder failed"), KeyboardInterrupt()]
)
def test_encoding_failure_closes_sequence_and_retains_stage(memory, tmp_path, failure):
    source, _ = make_store(memory, tmp_path)
    sequences = []

    class VHS:
        def combine_video(self, images, **_settings):
            sequences.append(images)
            images[0]
            raise failure

    memory._BINDINGS["vhs"] = VHS
    with pytest.raises(memory.AWGEncodingFailure, match="AWG_ENCODING_FAILURE"):
        memory.AWGMemoryEncode().encode(
            source,
            format="video/h264-mp4",
            loop_count=0,
            pingpong=False,
            save_output=True,
            filename_prefix=PREFIX + "/raw",
        )
    assert sequences[0]._closed and sequences[0]._batch is None
    assert (source.output_root / source.prefix / source.directory_name).exists()


def verified_helper(value=1):
    return value + 1


def test_helper_compatibility_rejects_signature_or_behavior_changes(memory):
    expected = {
        "parameters": ["value:POSITIONAL_OR_KEYWORD:1"],
        "sha256": memory._source_fingerprint(inspect.getsource(verified_helper)),
    }
    memory._verify_symbol("helper", verified_helper, expected)
    with pytest.raises(memory.FrameStoreError, match="Unsupported"):
        memory._verify_symbol(
            "helper", verified_helper, {**expected, "sha256": "0" * 64}
        )
    with pytest.raises(memory.FrameStoreError, match="Unsupported"):
        memory._verify_symbol(
            "helper", verified_helper, {**expected, "parameters": ["different"]}
        )


def test_cached_file_handles_are_never_reused(memory):
    for node in memory.NODE_CLASS_MAPPINGS.values():
        assert math.isnan(node.IS_CHANGED())
    assert (
        memory.AWGMemoryGenerate.validate_dependencies is memory.validate_dependencies
    )


def test_stage_schemas_use_handles_and_keep_resize_image_name(memory):
    schemas: dict[str, Any] = {
        "scail": {
            "required": {"frames": ("INT",)},
            "optional": {"anchor_frames": ("IMAGE",)},
        },
        "chroma": {"required": {"images": ("IMAGE",), "reference_image": ("IMAGE",)}},
        "resize": {
            "required": {"image": ("IMAGE",)},
            "optional": {"mask": ("MASK",), "device": (["cpu", "gpu"],)},
        },
        "clarity": {"required": {"images": ("IMAGE",)}, "optional": {}},
        "vhs": {
            "required": {"images": ("IMAGE",), "format": (["other"],)},
            "optional": {"vae": ("VAE",), "meta_batch": ("VHS_BatchManager",)},
        },
    }
    memory._BINDINGS.update(
        {
            name: types.SimpleNamespace(INPUT_TYPES=lambda schema=schema: schema)
            for name, schema in schemas.items()
        }
    )
    assert memory.AWGMemoryGenerate.INPUT_TYPES()["required"]["attempt_prefix"] == (
        "STRING",
    )
    assert memory.AWGMemoryChroma.INPUT_TYPES()["required"]["images"] == ("AWG_FRAMES",)
    resize = memory.AWGMemoryResize.INPUT_TYPES()
    assert resize["required"]["image"] == ("AWG_FRAMES",)
    assert {
        "width",
        "height",
        "upscale_method",
        "keep_proportion",
        "pad_color",
        "crop_position",
        "divisible_by",
    } <= resize["required"].keys()
    assert "mask" not in resize["optional"]
    assert memory.AWGMemoryClarity.INPUT_TYPES()["required"]["images"] == (
        "AWG_FRAMES",
    )
    encode = memory.AWGMemoryEncode.INPUT_TYPES()
    assert encode["required"]["format"] == (["video/h264-mp4"],)
    assert "audio" in encode["optional"]
    assert not {"images", "latents", "meta_batch", "vae"} & encode["optional"].keys()
    assert schemas["resize"]["required"]["image"] == ("IMAGE",)  # No upstream mutation.


def test_readiness_checks_all_adapted_symbols_and_rejects_drift_without_fallback(
    memory, tmp_path, monkeypatch
):
    owners = {
        name.split(".", 1)[0]: types.SimpleNamespace() for name in memory._COMPATIBILITY
    }
    for name in memory._COMPATIBILITY:
        owner, attribute = name.split(".", 1)
        setattr(owners[owner], attribute, Mock())
    for name in ("scail_module", "chroma", "chroma_module", "clarity"):
        owners.setdefault(name, types.SimpleNamespace())
    owners["clarity"].process = lambda images: None
    owners["clarity"]._process_chunk = lambda chunk, scene_luma=None: None
    owners["scail_module"].WanSCAILToVideo = owners["condition"]
    owners["scail_module"].SamplerCustom = owners["sampler"]
    owners["scail_module"].VAEDecode = owners["decode"]
    owners["vhs_module"].base_formats_dir = str(tmp_path)
    (tmp_path / "h264-mp4.json").write_text(json.dumps(memory._H264_FORMAT))
    registry = {
        "SCAIL2AutoVideo": owners["scail"],
        "ReferenceChromaLightTest": owners["chroma"],
        "ImageResizeKJv2": owners["resize"],
        "SaoChepAdaptiveClarityEnhancer": owners["clarity"],
        "VHS_VideoCombine": owners["vhs"],
    }

    def module_for(owner):
        for name in ("scail", "chroma", "vhs"):
            if owner is owners[name]:
                return owners[f"{name}_module"]
        raise AssertionError("unexpected module owner")

    monkeypatch.setattr(memory.inspect, "getmodule", module_for)
    verify = Mock()
    monkeypatch.setattr(memory, "_verify_symbol", verify)
    memory.validate_dependencies(registry)
    assert {call.args[0] for call in verify.call_args_list} == set(
        memory._COMPATIBILITY
    )
    assert memory._binding("scail") is owners["scail"]
    (tmp_path / "h264-mp4.json").write_text(
        json.dumps({**memory._H264_FORMAT, "pre_pass": ["unsafe"]})
    )
    with pytest.raises(memory.FrameStoreError, match="Unsupported H.264"):
        memory.validate_dependencies(registry)
    assert not memory._BINDINGS
    # Schema inspection stays available even without optional execution plugins;
    # execution must still refuse the failed readiness binding.
    assert "model" in memory.AWGMemoryGenerate.INPUT_TYPES()["required"]
    with pytest.raises(memory.FrameStoreError, match="not been validated"):
        memory._binding("scail")


def test_incompatible_helper_stops_readiness_and_clears_previous_bindings(memory):
    memory._BINDINGS["old"] = "must not survive validation failure"
    with pytest.raises(memory.FrameStoreError, match="unavailable"):
        memory.validate_dependencies({})
    assert not memory._BINDINGS


def test_sequence_interrupts_before_read_and_encoder_rejects_other_output_prefix(
    memory, tmp_path, monkeypatch
):
    source, _ = make_store(memory, tmp_path)
    sequence = memory.FrameSequence(source)
    monkeypatch.setattr(
        memory, "_interrupt", Mock(side_effect=InterruptedError("lost authority"))
    )
    with pytest.raises(InterruptedError):
        sequence[0]
    assert sequence._batch is None
    monkeypatch.setattr(memory, "_interrupt", Mock())
    vhs = Mock()
    memory._BINDINGS["vhs"] = vhs
    with pytest.raises(memory.AWGEncodingFailure):
        memory.AWGMemoryEncode().encode(
            source,
            format="video/h264-mp4",
            loop_count=0,
            pingpong=False,
            save_output=True,
            filename_prefix="other-attempt/raw",
        )
    vhs.assert_not_called()
    assert (source.output_root / source.prefix / source.directory_name).exists()


def test_interrupted_stage_writer_closes_before_unwinding(memory, tmp_path):
    writer = memory.FrameStoreWriter(tmp_path, PREFIX, "interrupt")
    check = Mock(side_effect=[None, InterruptedError("lost authority")])
    with pytest.raises(InterruptedError), writer:
        writer.append(np.zeros((17, 2, 3, 3), dtype=np.float32), check=check)
    assert writer.closed and len(writer.chunks) == 1
    with pytest.raises(memory.FrameStoreError, match="closed"):
        writer.finish()


def test_resize_batch_budget_accounts_for_delivery_canvas(memory, tmp_path):
    source, _ = make_store(memory, tmp_path)
    groups = []

    class Resize:
        def resize(self, image, **_settings):
            groups.append(len(image))
            return (image,)

    memory._BINDINGS["resize"] = Resize
    # The fake helper avoids allocating this canvas; only batch-budget routing is proved.
    memory.AWGMemoryResize().resize(source, width=4096, height=2160)
    assert max(groups) == 3 and sum(groups) == 35


def test_encoder_cannot_select_tensor_latent_or_meta_batch_path(memory, tmp_path):
    source, _ = make_store(memory, tmp_path)
    node = Mock()
    memory._BINDINGS["vhs"] = node
    for name in ("latents", "vae", "meta_batch"):
        with pytest.raises(memory.AWGEncodingFailure):
            memory.AWGMemoryEncode().encode(
                source,
                format="video/h264-mp4",
                loop_count=0,
                pingpong=False,
                save_output=True,
                filename_prefix=PREFIX + "/raw",
                **{name: object()},
            )
    node.assert_not_called()


def test_fingerprint_is_stable_across_supported_python_ast_empty_fields(memory):
    # Fixed value independently checked on Python 3.12 and 3.14, including bytes/None.
    assert (
        memory._source_fingerprint('def helper(x=1):\n    return [x, b"", None]\n')
        == "b80f8dd5afb0b792562fd9d288e050b164450f0605e7ec73ceb6a1ed0abfd213"
    )


def test_chroma_preserves_oom_group_reduction_without_skipped_frames(memory, tmp_path):
    source, frames = make_store(
        memory, tmp_path, np.full((35, 2, 3, 3), 0.25, dtype=np.float32)
    )
    node = sample_node(3, [0] * 9)
    groups = []

    def transform(batch, _a, _b, **_kwargs):
        groups.append(len(batch))
        if len(groups) == 1:
            raise RuntimeError("CUDA out of memory")
        return batch, 0

    memory._BINDINGS.update(
        chroma=lambda: node,
        chroma_module=types.SimpleNamespace(transform_cielab_batch_torch=transform),
    )
    result = memory.AWGMemoryChroma().correct(
        source,
        TensorDouble(np.zeros((1, 2, 3, 3))),
        enabled=True,
        processing_backend="gpu",
    )[0]
    assert groups == [16, 8, 8, 8, 8, 3]
    memory._torch().cuda.empty_cache.assert_called_once()
    np.testing.assert_array_equal(all_pixels(result), frames)


def test_helper_default_mutation_is_rejected_even_with_unchanged_source(
    memory, monkeypatch
):
    expected = {
        "parameters": ["value:POSITIONAL_OR_KEYWORD:1"],
        "sha256": memory._callable_fingerprint(verified_helper),
    }
    monkeypatch.setattr(verified_helper, "__defaults__", (2,))
    assert memory._callable_fingerprint(verified_helper) == expected["sha256"]
    with pytest.raises(memory.FrameStoreError, match="Unsupported"):
        memory._verify_symbol("helper", verified_helper, expected)
