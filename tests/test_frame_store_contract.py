"""CPU-only frame-store contract; no ComfyUI/Torch imports or live processes."""

from __future__ import annotations

import dataclasses
import errno
import types

import numpy as np
import pytest
from bounded_test_support import PREFIX, make_store


def test_frame_store_round_trip_and_partial_cross_chunk_range(frames_module, tmp_path):
    store, frames = make_store(frames_module, tmp_path)
    assert tuple(chunk.count for chunk in store.chunks) == (16, 16, 3)
    assert len(store) == 35
    np.testing.assert_array_equal(store.read(14, 5), frames[14:19])
    result = np.concatenate(
        list(store.iter_batches())
    )  # Test oracle only, never worker code.
    np.testing.assert_array_equal(result, frames)
    assert result.dtype == frames.dtype
    assert store.fps == 24
    with pytest.raises(dataclasses.FrozenInstanceError):
        store.fps = 30


@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.float64, np.uint8, np.int16]
)
def test_lossless_dtype_preservation(frames_module, tmp_path, dtype):
    values = np.arange(2 * 2 * 3 * 3, dtype=dtype).reshape(2, 2, 3, 3)
    store, _ = make_store(frames_module, tmp_path, values)
    result = store.read(0, 2)
    assert result.dtype == values.dtype
    np.testing.assert_array_equal(result, values)


def test_frame_store_preserves_float_bit_patterns_and_fortran_input(
    frames_module, tmp_path
):
    frames = np.asfortranarray(np.zeros((2, 2, 3, 3), dtype=np.float32))
    frames[0, 0, 0] = [np.nan, np.inf, -0.0]
    store, _ = make_store(frames_module, tmp_path, frames)
    result = store.read(0, 2)
    np.testing.assert_array_equal(result.view(np.uint32), frames.copy().view(np.uint32))


def test_read_ranges_remain_bounded(frames_module, tmp_path):
    store, _ = make_store(frames_module, tmp_path)
    for start, count in ((-1, 1), (0, 17), (34, 2), (0, 0), (True, 1)):
        with pytest.raises(frames_module.FrameStoreError):
            store.read(start, count)


def test_writer_refuses_layout_change_objects_and_empty_finish(frames_module, tmp_path):
    with frames_module.FrameStoreWriter(tmp_path, PREFIX, "test") as writer:
        with pytest.raises(frames_module.FrameStoreError):
            writer.finish()
        with pytest.raises(frames_module.FrameStoreError):
            writer.append(np.empty((1, 2, 3, 3), dtype=object))
        writer.append(np.zeros((1, 2, 3, 3), dtype=np.float32))
        with pytest.raises(frames_module.FrameStoreError):
            writer.append(np.zeros((1, 2, 4, 3), dtype=np.float32))
        writer.finish()
        with pytest.raises(frames_module.FrameStoreError):
            writer.append(np.zeros((1, 2, 3, 3), dtype=np.float32))


@pytest.mark.parametrize(
    "prefix", ["../outside", "aiworkergrid/../outside", "aiworkergrid/ABC"]
)
def test_writer_refuses_unsafe_attempt_prefix(frames_module, tmp_path, prefix):
    with pytest.raises(frames_module.FrameStoreError):
        frames_module.FrameStoreWriter(tmp_path, prefix, "test")


def test_store_refuses_symlinked_parent_and_chunk(frames_module, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "aiworkergrid").symlink_to(outside, target_is_directory=True)
    with pytest.raises(frames_module.FrameStoreError):
        frames_module.FrameStoreWriter(tmp_path, PREFIX, "test")
    (tmp_path / "aiworkergrid").unlink()
    store, _ = make_store(frames_module, tmp_path)
    chunk_path = store._directory() / store.chunks[0].name
    original = outside / "untouched.npy"
    chunk_path.rename(original)
    chunk_path.symlink_to(original)
    with pytest.raises(frames_module.FrameStoreError):
        store.read(0, 1)
    with pytest.raises(frames_module.FrameStoreError):
        store.discard()
    assert original.exists()


@pytest.mark.parametrize("damage", ["truncate", "payload", "trailing"])
def test_corrupt_chunks_fail_clearly(frames_module, tmp_path, damage):
    store, _ = make_store(frames_module, tmp_path)
    path = store._directory() / store.chunks[0].name
    payload = bytearray(path.read_bytes())
    if damage == "truncate":
        payload = payload[:-1]
    elif damage == "payload":
        payload[-1] ^= 1
    else:
        payload.extend(b"extra")
    path.write_bytes(payload)
    with pytest.raises(frames_module.FrameStoreError, match="checksum"):
        store.read(0, 1)


def test_disk_exhaustion_closes_writer_without_publishing_handle(
    frames_module, tmp_path, monkeypatch
):
    writer = frames_module.FrameStoreWriter(tmp_path, PREFIX, "test")
    monkeypatch.setattr(
        frames_module.shutil, "disk_usage", lambda _path: types.SimpleNamespace(free=0)
    )
    with pytest.raises(
        frames_module.FrameStoreError, match="Insufficient disk"
    ) as caught:
        writer.append(np.zeros((1, 2, 3, 3), dtype=np.float32))
    assert caught.value.__cause__.errno == errno.ENOSPC
    assert writer.closed and not writer.chunks
    with pytest.raises(frames_module.FrameStoreError):
        writer.finish()


def test_store_discard_is_attempt_confined_and_stale_handle_fails(
    frames_module, tmp_path
):
    store, _ = make_store(frames_module, tmp_path)
    other = tmp_path / "aiworkergrid" / ("b" * 32)
    other.mkdir()
    (other / "keep").write_text("other attempt")
    store.discard()
    assert (other / "keep").read_text() == "other attempt"
    with pytest.raises(frames_module.FrameStoreError):
        store.read(0, 1)


def test_iterator_checks_interruption_before_read(frames_module, tmp_path):
    store, _ = make_store(frames_module, tmp_path)
    with pytest.raises(InterruptedError):
        next(store.iter_batches(lambda: (_ for _ in ()).throw(InterruptedError())))


def test_write_failure_closes_descriptor_and_keeps_partial_unpublished(
    frames_module, tmp_path, monkeypatch
):
    writer = frames_module.FrameStoreWriter(tmp_path, PREFIX, "test")
    handles = []

    def fail_write(stream, _batch, **_kwargs):
        handles.append(stream)
        stream.write(b"partial")
        raise OSError(errno.ENOSPC, "disk became full")

    monkeypatch.setattr(frames_module.np, "save", fail_write)
    with pytest.raises(frames_module.FrameStoreError, match="Insufficient disk"):
        writer.append(np.zeros((1, 2, 3, 3), dtype=np.float32))
    assert writer.closed and handles[0].closed and not writer.chunks
    assert not list(writer.directory.glob("*.npy"))
    assert list(writer.directory.glob("*.partial"))


def test_chunk_reader_closes_file_before_returning_frames(
    frames_module, tmp_path, monkeypatch
):
    store, _ = make_store(frames_module, tmp_path)
    handles = []
    original_fdopen = frames_module.os.fdopen

    def track_fdopen(*args, **kwargs):
        stream = original_fdopen(*args, **kwargs)
        handles.append(stream)
        return stream

    monkeypatch.setattr(frames_module.os, "fdopen", track_fdopen)
    batch = next(store.iter_batches())
    assert isinstance(batch, np.ndarray) and not isinstance(batch, np.memmap)
    assert handles and all(stream.closed for stream in handles)
