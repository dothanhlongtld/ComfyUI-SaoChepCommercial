"""CPU-only proof of the upstream-model RelayVSR bridge contract."""


def _mock_torch(monkeypatch, relay, np):
    from contextlib import contextmanager

    class Torch:
        uint8 = np.uint8

        @staticmethod
        def from_numpy(value):
            return value

        @staticmethod
        @contextmanager
        def inference_mode():
            yield

    monkeypatch.setattr(
        relay.importlib,
        "import_module",
        lambda name: Torch if name == "torch" else None,
    )


def test_relay_schema_and_registration(memory):
    node = memory.NODE_CLASS_MAPPINGS["AWGMemoryRelayVSR"]
    assert node.INPUT_TYPES()["optional"]["allow_x4_downsample"] == (
        "BOOLEAN",
        {"default": False},
    )
    assert node.INPUT_TYPES()["required"] == {
        "images": ("AWG_FRAMES",),
        "model": ("RELAYVSR_MODEL",),
    }
    assert node.RETURN_TYPES == ("AWG_FRAMES",)
    assert node.FUNCTION == "upscale"


def test_upstream_handle_streaming_restore_and_ownership(memory, tmp_path, monkeypatch):
    import importlib

    import numpy as np
    from bounded_test_support import make_store

    data = np.linspace(0, 1, 19 * 2 * 3 * 3, dtype=np.float32).reshape(19, 2, 3, 3)
    source, _ = make_store(memory, tmp_path, data)
    relay = importlib.import_module("saochep_bounded_under_test.relayvsr_node")
    monkeypatch.setattr(relay, "_interrupt", lambda: None)
    _mock_torch(monkeypatch, relay, np)
    calls = []

    class Engine:
        def restore(self, frames, **kwargs):
            calls.append(kwargs)
            for frame in frames:
                assert isinstance(frame, np.ndarray)
                yield np.repeat(np.repeat(frame.transpose(1, 2, 0), 4, 0), 4, 1)

    class Model:
        def ensure_loaded(self):
            calls.append("load")
            return Engine()

        def free(self):
            raise AssertionError("bridge must leave lifecycle to ComfyUI")

    result = memory.AWGMemoryRelayVSR().upscale(source, Model())[0]
    output = np.concatenate([result.read(0, 16), result.read(16, 3)])
    assert output.shape == (19, 8, 12, 3)
    assert output.dtype == np.float32
    np.testing.assert_allclose(output[:, 0, 0], np.rint(data[:, 0, 0] * 255) / 255)
    assert calls == [
        "load",
        {"seed": 42, "sr_ratio": 4, "chunk_len": 16, "check": relay._interrupt},
    ]
    assert not (source.output_root / source.prefix / source.directory_name).exists()


def test_bicubic_float_overshoot_is_clamped_at_uint8_boundary(
    memory, tmp_path, monkeypatch
):
    import importlib

    import numpy as np
    from bounded_test_support import make_store

    # Observed on the L4 reproduction after bicubic delivery resize:
    # finite float32 pixels ranged from -0.0337227 to 1.0344917.
    data = np.array(
        [[[[-0.0337227, 0.5, 1.0344917], [0.0, 1.0, 0.25]]]], dtype=np.float32
    )
    source, _ = make_store(memory, tmp_path, data)
    relay = importlib.import_module("saochep_bounded_under_test.relayvsr_node")
    _mock_torch(monkeypatch, relay, np)
    monkeypatch.setattr(relay, "_interrupt", lambda: None)

    frames = list(relay._input_frames(source))
    expected = np.array([[[0, 0]], [[128, 255]], [[255, 64]]], dtype=np.uint8)
    np.testing.assert_array_equal(frames[0], expected)
    np.testing.assert_array_equal(source.read(0, 1), data)


def test_nonfinite_and_nonimage_integer_inputs_remain_rejected(
    memory, tmp_path, monkeypatch
):
    import importlib

    import numpy as np
    import pytest
    from bounded_test_support import make_store

    relay = importlib.import_module("saochep_bounded_under_test.relayvsr_node")
    _mock_torch(monkeypatch, relay, np)
    monkeypatch.setattr(relay, "_interrupt", lambda: None)
    for bad in (np.nan, np.inf, -np.inf):
        source, _ = make_store(
            memory, tmp_path, np.full((1, 1, 1, 3), bad, dtype=np.float32)
        )
        with pytest.raises(memory.FrameStoreError, match="finite float RGB"):
            list(relay._input_frames(source))
    source, _ = make_store(memory, tmp_path, np.ones((1, 1, 1, 3), dtype=np.int16))
    with pytest.raises(memory.FrameStoreError, match="finite float RGB"):
        list(relay._input_frames(source))


def test_interrupt_closes_streams_writer_and_preserves_primary_error(
    memory, tmp_path, monkeypatch
):
    import importlib

    import numpy as np
    import pytest
    from bounded_test_support import make_store

    source, _ = make_store(memory, tmp_path, np.zeros((2, 2, 3, 3), dtype=np.uint8))
    relay = importlib.import_module("saochep_bounded_under_test.relayvsr_node")
    _mock_torch(monkeypatch, relay, np)
    state = {"checks": 0}

    def interrupt():
        state["checks"] += 1
        if state["checks"] == 5:
            raise InterruptedError("cancelled")

    monkeypatch.setattr(relay, "_interrupt", interrupt)
    closed = []

    class Engine:
        def restore(self, frames, **_kwargs):
            try:
                next(frames)
                yield np.zeros((8, 12, 3), dtype=np.uint8)
                next(frames)
            finally:
                closed.append("restore")
                raise RuntimeError("close failed")

    class Model:
        def ensure_loaded(self):
            return Engine()

    with pytest.raises(InterruptedError, match="cancelled"):
        memory.AWGMemoryRelayVSR().upscale(source, Model())
    assert "restore" in closed
    assert (source.output_root / source.prefix / source.directory_name).exists()
    partial = list((source.output_root / source.prefix).glob("frames-relayvsr-*"))
    assert len(partial) == 1


def test_invalid_output_shape_dtype_or_count_preserves_source(
    memory, tmp_path, monkeypatch
):
    import importlib

    import numpy as np
    import pytest
    from bounded_test_support import make_store

    relay = importlib.import_module("saochep_bounded_under_test.relayvsr_node")
    _mock_torch(monkeypatch, relay, np)
    monkeypatch.setattr(relay, "_interrupt", lambda: None)

    class Model:
        def __init__(self, output):
            self.output = output

        def ensure_loaded(self):
            return self

        def restore(self, frames, **_kwargs):
            next(frames)
            yield from self.output

    cases = [
        ([np.zeros((8, 11, 3), dtype=np.uint8)], "exact 4x"),
        ([np.zeros((8, 12, 3), dtype=np.float32)], "exact 4x"),
        ([], "frame count"),
    ]
    for output, error in cases:
        source, _ = make_store(memory, tmp_path, np.zeros((1, 2, 3, 3), dtype=np.uint8))
        with pytest.raises(memory.FrameStoreError, match=error):
            memory.AWGMemoryRelayVSR().upscale(source, Model(output))
        assert (source.output_root / source.prefix / source.directory_name).exists()


def test_native_x2_scale_and_default_x4_are_backward_compatible(
    memory, tmp_path, monkeypatch
):
    import importlib
    from types import SimpleNamespace

    import numpy as np
    from bounded_test_support import make_store

    relay = importlib.import_module("saochep_bounded_under_test.relayvsr_node")
    _mock_torch(monkeypatch, relay, np)
    monkeypatch.setattr(relay, "_interrupt", lambda: None)
    calls = []

    class Engine:
        geometry = SimpleNamespace(scale=2)

        def restore(self, frames, **kwargs):
            calls.append(kwargs)
            for frame in frames:
                yield np.repeat(np.repeat(frame.transpose(1, 2, 0), 2, 0), 2, 1)

    source, _ = make_store(memory, tmp_path, np.ones((3, 2, 3, 3), dtype=np.float32))
    result = relay.AWGMemoryRelayVSR().upscale(
        source, SimpleNamespace(ensure_loaded=lambda: Engine()), sr_ratio=2
    )[0]
    assert result.shape == (4, 6, 3)
    assert len(result) == 3
    assert calls[0]["sr_ratio"] == 2
    assert relay.AWGMemoryRelayVSR.INPUT_TYPES()["optional"]["sr_ratio"] == (
        "INT",
        {"default": 4, "min": 2, "max": 4, "step": 2},
    )


def test_x4_downsample_requires_opt_in_and_preserves_requested_ratio(
    memory, tmp_path, monkeypatch
):
    import importlib
    from types import SimpleNamespace

    import numpy as np
    import pytest
    from bounded_test_support import make_store

    relay = importlib.import_module("saochep_bounded_under_test.relayvsr_node")
    _mock_torch(monkeypatch, relay, np)
    monkeypatch.setattr(relay, "_interrupt", lambda: None)
    calls = []

    class Engine:
        geometry = SimpleNamespace(scale=4)

        def restore(self, frames, **kwargs):
            calls.append(kwargs)
            for frame in frames:
                yield np.repeat(np.repeat(frame.transpose(1, 2, 0), 2, 0), 2, 1)

    model = SimpleNamespace(ensure_loaded=lambda: Engine())
    source, _ = make_store(memory, tmp_path, np.ones((2, 2, 3, 3), dtype=np.uint8))
    with pytest.raises(memory.FrameStoreError, match="native X2"):
        relay.AWGMemoryRelayVSR().upscale(source, model, sr_ratio=2)
    assert (source.output_root / source.prefix / source.directory_name).exists()

    result = relay.AWGMemoryRelayVSR().upscale(
        source, model, sr_ratio=2, allow_x4_downsample=True
    )[0]
    assert result.shape == (4, 6, 3)
    assert len(result) == 2
    assert calls[0]["sr_ratio"] == 2


def test_unknown_native_geometry_is_rejected(memory, tmp_path, monkeypatch):
    import importlib
    from types import SimpleNamespace

    import numpy as np
    import pytest
    from bounded_test_support import make_store

    relay = importlib.import_module("saochep_bounded_under_test.relayvsr_node")
    _mock_torch(monkeypatch, relay, np)
    monkeypatch.setattr(relay, "_interrupt", lambda: None)
    for geometry in (None, SimpleNamespace(scale=3)):
        source, _ = make_store(memory, tmp_path, np.ones((1, 2, 3, 3), dtype=np.uint8))
        model = SimpleNamespace(
            ensure_loaded=lambda geometry=geometry: SimpleNamespace(geometry=geometry)
        )
        with pytest.raises(
            memory.FrameStoreError, match="native X2 checkpoints or opt-in"
        ):
            relay.AWGMemoryRelayVSR().upscale(
                source, model, sr_ratio=2, allow_x4_downsample=True
            )
        assert (source.output_root / source.prefix / source.directory_name).exists()


def test_x2_route_refuses_x4_checkpoints_and_invalid_scales(
    memory, tmp_path, monkeypatch
):
    import importlib
    from types import SimpleNamespace

    import numpy as np
    import pytest
    from bounded_test_support import make_store

    relay = importlib.import_module("saochep_bounded_under_test.relayvsr_node")
    _mock_torch(monkeypatch, relay, np)
    monkeypatch.setattr(relay, "_interrupt", lambda: None)
    engine = SimpleNamespace(geometry=SimpleNamespace(scale=4))
    model = SimpleNamespace(ensure_loaded=lambda: engine)
    source, _ = make_store(memory, tmp_path, np.ones((1, 2, 3, 3), dtype=np.uint8))
    for scale in (True, 0, 3, 8):
        with pytest.raises(memory.FrameStoreError, match="scale must be"):
            relay.AWGMemoryRelayVSR().upscale(source, model, sr_ratio=scale)
    with pytest.raises(memory.FrameStoreError, match="native X2 checkpoints"):
        relay.AWGMemoryRelayVSR().upscale(source, model, sr_ratio=2)
    assert len(source.read(0, 1)) == 1
