"""CPU-only Relay node contract checks; no Relay imports or model assets required."""

import math
from contextlib import contextmanager


def test_relay_node_schema_and_registration(memory):
    node = memory.NODE_CLASS_MAPPINGS["AWGMemoryRelayVSR"]
    required = node.INPUT_TYPES()["required"]
    assert set(required) == {"images", "wan_checkpoint", "flash_checkpoint"}
    assert required["images"] == ("AWG_FRAMES",)
    assert required["wan_checkpoint"][1]["default"] == "wan.safetensors"
    assert required["flash_checkpoint"][1]["default"] == "flash.safetensors"
    assert node.RETURN_TYPES == ("AWG_FRAMES",)
    assert node.FUNCTION == "upscale"
    assert math.isnan(node.IS_CHANGED())


def _setup_paths(monkeypatch, relay, tmp_path, events=None):
    import types
    import numpy as np

    def imported(name):
        if name == "folder_paths":
            return types.SimpleNamespace(
                base_path=str(tmp_path), get_folder_paths=lambda _kind: [str(tmp_path)]
            )
        if name == "torch":

            class Torch:
                device = staticmethod(lambda value: value)

                @staticmethod
                @contextmanager
                def inference_mode():
                    yield

                @staticmethod
                def isfinite(value):
                    return types.SimpleNamespace(
                        all=lambda: types.SimpleNamespace(
                            item=lambda: np.isfinite(value.array).all()
                        )
                    )

            return Torch
        return types.SimpleNamespace(
            unload_all_models=lambda: events.append("unload")
            if events is not None
            else None
        )

    monkeypatch.setattr(relay.importlib, "import_module", imported)
    (tmp_path / "relayvsr").mkdir(exist_ok=True)
    (tmp_path / "wan.safetensors").touch()
    (tmp_path / "flash.safetensors").touch()


def test_float_store_conversion_is_one_continuous_multichunk_stream(
    memory, tmp_path, monkeypatch
):
    import importlib
    import numpy as np
    import types
    from bounded_test_support import make_store

    frames = np.arange(19 * 3, dtype=np.float32).reshape(19, 1, 1, 3) / 56
    source, _ = make_store(memory, tmp_path, frames)
    relay = importlib.import_module("saochep_bounded_under_test.relayvsr_node")
    events = []
    _setup_paths(monkeypatch, relay, tmp_path, events)

    def interrupt():
        events.append("interrupt")

    monkeypatch.setattr(relay, "_interrupt", interrupt)
    calls, stream_calls = [], []

    class Tensor:
        def __init__(self, array):
            self.array, self.shape = np.asarray(array), np.asarray(array).shape

        def __getitem__(self, key):
            return Tensor(self.array[key])

        def add(self, value):
            return Tensor(self.array + value)

        def mul_(self, value):
            self.array *= value
            return self

        def clamp_(self, low, high):
            self.array = self.array.clip(low, high)
            return self

        def to(self, dtype):
            assert FakeTorch.active
            return Tensor(self.array.astype(dtype))

        def permute(self, *dims):
            return Tensor(self.array.transpose(dims))

        def contiguous(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

        def all(self):
            return self

        def item(self):
            return bool(np.isfinite(self.array).all())

    class FakeTorch:
        uint8, float32, active = np.uint8, np.float32, False

        @staticmethod
        @contextmanager
        def inference_mode():
            FakeTorch.active = True
            try:
                yield
            finally:
                FakeTorch.active = False

        @staticmethod
        def from_numpy(array):
            calls.append(array.copy())
            return Tensor(array)

        @staticmethod
        def isfinite(value):
            return Tensor(np.isfinite(value.array))

    class Packet:
        def __init__(self, index, rgb, source_time):
            self.index, self.rgb, self.source_time = index, rgb, source_time

    class Stream:
        def frames(self, packets):
            stream_calls.append(True)
            expected = 0
            for packet in packets:
                assert FakeTorch.active and packet.index == expected
                expected += 1
                rgb = packet.rgb.array.astype(np.float32) / 255
                output = np.repeat(np.repeat(rgb * 2 - 1, 4, axis=1), 4, axis=2)[None]
                yield types.SimpleNamespace(index=packet.index, rgb=Tensor(output))
            assert expected == len(frames)

    def load(*_args):
        events.append("load")
        return FakeTorch, Packet, Stream(), (object(),)

    monkeypatch.setattr(relay, "_load_relay", load)
    result = memory.AWGMemoryRelayVSR().upscale(source)[0]
    output = np.concatenate((result.read(0, 16), result.read(16, 3)))
    assert output.shape == (19, 4, 4, 3)
    assert output.dtype == frames.dtype
    assert len(calls) == 19 and stream_calls == [True]
    assert events[:4] == ["interrupt", "unload", "interrupt", "load"]
    np.testing.assert_allclose(
        output[:, 0, 0], np.rint(frames[:, 0, 0] * 255) / 255, atol=1e-6
    )
    assert not (source.output_root / source.prefix / source.directory_name).exists()


def test_relay_failure_preserves_source(memory, tmp_path, monkeypatch):
    import importlib
    import numpy as np
    import pytest
    from bounded_test_support import make_store

    source, _ = make_store(memory, tmp_path, np.zeros((2, 2, 3, 3), dtype=np.uint8))
    relay = importlib.import_module("saochep_bounded_under_test.relayvsr_node")
    _setup_paths(monkeypatch, relay, tmp_path)
    monkeypatch.setattr(relay, "_interrupt", lambda: None)
    monkeypatch.setattr(
        relay,
        "_load_relay",
        lambda *_: (_ for _ in ()).throw(RuntimeError("load failed")),
    )
    with pytest.raises(RuntimeError, match="load failed"):
        memory.AWGMemoryRelayVSR().upscale(source)
    assert (source.output_root / source.prefix / source.directory_name).exists()


def test_relay_interruption_before_model_load_preserves_source(
    memory, tmp_path, monkeypatch
):
    import importlib
    import numpy as np
    import pytest
    from bounded_test_support import make_store

    source, _ = make_store(memory, tmp_path, np.zeros((1, 2, 3, 3), dtype=np.uint8))
    relay = importlib.import_module("saochep_bounded_under_test.relayvsr_node")
    _setup_paths(monkeypatch, relay, tmp_path)
    monkeypatch.setattr(
        relay,
        "_interrupt",
        lambda: (_ for _ in ()).throw(InterruptedError("cancelled")),
    )
    loaded = []
    monkeypatch.setattr(relay, "_load_relay", lambda *_: loaded.append(True))
    with pytest.raises(InterruptedError, match="cancelled"):
        memory.AWGMemoryRelayVSR().upscale(source)
    assert (
        not loaded
        and (source.output_root / source.prefix / source.directory_name).exists()
    )


class _MockWriter:
    """Delegates to the real bounded writer while exposing closure state."""

    def __init__(self, module, *args):
        self.inner = module._test_real_writer(*args)
        self.closed = False
        self.finished = False
        self.chunks = self.inner.chunks

    def append(self, *args, **kwargs):
        return self.inner.append(*args, **kwargs)

    def finish(self):
        self.finished = True
        return self.inner.finish()

    def close(self):
        self.closed = True
        self.inner.close()


def test_nonfinite_relay_output_fails_closed_and_closes_stream(
    memory, tmp_path, monkeypatch
):
    import importlib
    import numpy as np
    import types
    import pytest

    from bounded_test_support import make_store

    source, _ = make_store(memory, tmp_path, np.zeros((1, 2, 3, 3), dtype=np.uint8))
    relay = importlib.import_module("saochep_bounded_under_test.relayvsr_node")
    _setup_paths(monkeypatch, relay, tmp_path)
    monkeypatch.setattr(relay, "_interrupt", lambda: None)
    writers = []
    monkeypatch.setattr(
        relay, "_test_real_writer", relay.FrameStoreWriter, raising=False
    )

    class Writer(_MockWriter):
        def __init__(self, *args):
            super().__init__(relay, *args)
            writers.append(self)

    monkeypatch.setattr(relay, "FrameStoreWriter", Writer)
    closed = []

    class Tensor:
        def __init__(self, array):
            self.array = np.asarray(array)
            self.shape = self.array.shape

        def __getitem__(self, key):
            return Tensor(self.array[key])

        def all(self):
            return Tensor(np.asarray(self.array.all()))

        def item(self):
            return bool(self.array)

    class Torch:
        uint8 = np.uint8

        @staticmethod
        def device(value):
            return value

        @staticmethod
        def from_numpy(array):
            return Tensor(array)

        @staticmethod
        def isfinite(value):
            return Tensor(np.isfinite(value.array))

        @staticmethod
        @contextmanager
        def inference_mode():
            yield

    class Packet:
        def __init__(self, index, rgb, source_time):
            self.index, self.rgb, self.source_time = index, rgb, source_time

    class Stream:
        def frames(self, packets):
            try:
                packet = next(packets)
                invalid = np.zeros((1, 3, 8, 12), dtype=np.float32)
                invalid[0, 0, 0, 0] = np.nan
                invalid[0, 1, 0, 0] = np.inf
                invalid[0, 2, 0, 0] = -np.inf
                yield types.SimpleNamespace(index=packet.index, rgb=Tensor(invalid))
            finally:
                closed.append("stream")

    import weakref

    model_refs = []

    class Model:
        pass

    def load(*_args):
        models = (Model(), Model(), Model())
        model_refs.extend(weakref.ref(model) for model in models)
        return Torch, Packet, Stream(), models

    monkeypatch.setattr(relay, "_load_relay", load)
    with pytest.raises(memory.FrameStoreError, match="non-finite"):
        memory.AWGMemoryRelayVSR().upscale(source)
    assert closed == ["stream"]
    assert writers[0].closed and not writers[0].finished and not writers[0].chunks
    assert (source.output_root / source.prefix / source.directory_name).exists()
    import gc

    gc.collect()
    assert all(reference() is None for reference in model_refs)


def test_interrupt_after_partial_output_closes_both_generators(
    memory, tmp_path, monkeypatch
):
    import importlib
    import numpy as np
    import types
    import pytest

    from bounded_test_support import make_store

    source, _ = make_store(memory, tmp_path, np.zeros((2, 2, 3, 3), dtype=np.uint8))
    relay = importlib.import_module("saochep_bounded_under_test.relayvsr_node")
    _setup_paths(monkeypatch, relay, tmp_path)
    writers, closed = [], []
    monkeypatch.setattr(
        relay, "_test_real_writer", relay.FrameStoreWriter, raising=False
    )
    state = {"written": False}

    def interrupt():
        if state["written"]:
            raise InterruptedError("cancelled after output")

    monkeypatch.setattr(relay, "_interrupt", interrupt)

    class Writer(_MockWriter):
        def __init__(self, *args):
            super().__init__(relay, *args)
            writers.append(self)

        def append(self, *args, **kwargs):
            result = self.inner.append(*args, **kwargs)
            state["written"] = True
            return result

    monkeypatch.setattr(relay, "FrameStoreWriter", Writer)

    class Tensor:
        def __init__(self, array):
            self.array, self.shape = np.asarray(array), np.asarray(array).shape

        def __getitem__(self, key):
            return Tensor(self.array[key])

        def add(self, value):
            return Tensor(self.array + value)

        def mul_(self, value):
            self.array *= value
            return self

        def clamp_(self, low, high):
            self.array = self.array.clip(low, high)
            return self

        def to(self, dtype):
            return Tensor(self.array.astype(dtype))

        def permute(self, *dims):
            return Tensor(self.array.transpose(dims))

        def contiguous(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

        def all(self):
            return self

        def item(self):
            return bool(np.isfinite(self.array).all())

    class Torch:
        uint8 = np.uint8
        float32 = np.float32

        @staticmethod
        def device(value):
            return value

        @staticmethod
        def from_numpy(array):
            return Tensor(array)

        @staticmethod
        def isfinite(value):
            return Tensor(np.isfinite(value.array))

        @staticmethod
        @contextmanager
        def inference_mode():
            yield

    class Packet:
        def __init__(self, index, rgb, source_time):
            self.index, self.rgb, self.source_time = index, rgb, source_time

    def packet_source(*_args):
        try:
            yield Packet(0, Tensor(np.zeros((3, 2, 3), dtype=np.uint8)), 0.0)
            yield Packet(1, Tensor(np.zeros((3, 2, 3), dtype=np.uint8)), 1 / 24)
        finally:
            closed.append("input")

    monkeypatch.setattr(relay, "_input_packets", packet_source)

    class Stream:
        def frames(self, packets):
            try:
                next(packets)
                yield types.SimpleNamespace(
                    index=0, rgb=Tensor(np.zeros((1, 3, 8, 12), dtype=np.float32))
                )
                yield types.SimpleNamespace(
                    index=1, rgb=Tensor(np.zeros((1, 3, 8, 12), dtype=np.float32))
                )
            finally:
                closed.append("stream")

    monkeypatch.setattr(
        relay, "_load_relay", lambda *_: (Torch, Packet, Stream(), (object(),))
    )
    with pytest.raises(InterruptedError, match="cancelled after output"):
        memory.AWGMemoryRelayVSR().upscale(source)
    assert set(closed) == {"stream", "input"}
    assert writers[0].closed and not writers[0].finished and len(writers[0].chunks) == 1
    assert (source.output_root / source.prefix / source.directory_name).exists()
