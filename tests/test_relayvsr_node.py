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
