from __future__ import annotations

import errno
import hashlib
import math
import os
import re
import shutil
import stat
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Self, cast

import numpy as np
from numpy.typing import NDArray

MAX_BATCH_FRAMES = 16
MAX_BATCH_PIXELS = 1920 * 1080 * MAX_BATCH_FRAMES
_PREFIX = re.compile(r"aiworkergrid/[0-9a-f]{32}")
_STAGE = re.compile(r"[a-z][a-z0-9_-]{0,47}")


class FrameStoreError(RuntimeError):
    """Lossless intermediate storage failed; never substitute lower-quality data."""


def _checked_directory(path: Path, *, create: bool = False) -> Path:
    path = path.absolute()
    # Check every component, not just the leaf: resolve() alone hides symlinks.
    for part in (path, *path.parents):
        if part.is_symlink():
            raise FrameStoreError("Frame-store paths cannot contain symlinks.")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir() or path.resolve() != path:
        raise FrameStoreError("Frame-store directory is unavailable or unsafe.")
    return path


def _attempt_directory(output_root: Path, prefix: str, *, create: bool = False) -> Path:
    if not _PREFIX.fullmatch(prefix):
        raise FrameStoreError("Invalid attempt-owned frame-store prefix.")
    root = _checked_directory(output_root)
    return _checked_directory(root / prefix, create=create)


def _layout(
    shape: tuple[int, int, int], dtype: str
) -> tuple[np.dtype[np.generic], int]:
    if (
        len(shape) != 3
        or any(type(size) is not int or size <= 0 for size in shape)
        or shape[2] not in (3, 4)
    ):
        raise FrameStoreError(
            "Frames must have positive HWC dimensions and RGB/RGBA channels."
        )
    try:
        parsed: np.dtype[np.generic] = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise FrameStoreError("Invalid frame-store dtype.") from exc
    if parsed.kind not in "buif" or parsed.hasobject or parsed.fields is not None:
        raise FrameStoreError(
            "Only plain numeric, non-object frame dtypes are supported."
        )
    pixels = shape[0] * shape[1]
    if pixels > MAX_BATCH_PIXELS:
        raise FrameStoreError("One frame exceeds the bounded processing pixel budget.")
    return parsed, max(1, min(MAX_BATCH_FRAMES, MAX_BATCH_PIXELS // pixels))


@dataclass(frozen=True)
class FrameChunk:
    name: str
    count: int
    sha256: str


@dataclass(frozen=True)
class FrameStore:
    output_root: Path
    prefix: str
    directory_name: str
    shape: tuple[int, int, int]
    dtype: str
    fps: float
    chunks: tuple[FrameChunk, ...]

    def __post_init__(self) -> None:
        _, limit = _layout(self.shape, self.dtype)
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise FrameStoreError("Frame-store FPS must be positive and finite.")
        if not re.fullmatch(
            r"frames-[a-z][a-z0-9_-]{0,47}-[0-9a-f]{32}", self.directory_name
        ):
            raise FrameStoreError("Invalid frame-store directory name.")
        if type(self.shape) is not tuple or type(self.chunks) is not tuple:
            raise FrameStoreError("Frame-store metadata must be immutable tuples.")
        if not self.chunks or any(
            chunk.name != f"{index:08d}.npy"
            or type(chunk.count) is not int
            or not 0 < chunk.count <= limit
            or not re.fullmatch(r"[0-9a-f]{64}", chunk.sha256)
            for index, chunk in enumerate(self.chunks)
        ):
            raise FrameStoreError("Invalid or empty ordered frame-store manifest.")
        self._directory()

    def _directory(self) -> Path:
        return _checked_directory(
            _attempt_directory(self.output_root, self.prefix) / self.directory_name
        )

    @property
    def batch_frames(self) -> int:
        return _layout(self.shape, self.dtype)[1]

    def __len__(self) -> int:
        return sum(chunk.count for chunk in self.chunks)

    def _read_chunk(self, chunk: FrameChunk) -> NDArray[np.generic]:
        path = self._directory() / chunk.name
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise FrameStoreError("Frame chunk is not a regular file.")
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != chunk.sha256:
                    raise FrameStoreError(
                        "Frame-chunk checksum does not match its manifest."
                    )
                stream.seek(0)
                # NumPy 2.2 exposes these public readers with incomplete stubs.
                read_magic = cast(
                    Callable[[BinaryIO], tuple[int, int]], np.lib.format.read_magic
                )
                version = read_magic(stream)
                if version not in ((1, 0), (2, 0)):
                    raise FrameStoreError("Unsupported frame-chunk format.")
                header_reader = cast(
                    Callable[..., tuple[tuple[int, ...], bool, np.dtype[np.generic]]],
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0,
                )
                shape, _, dtype = header_reader(stream, max_header_size=1024)
                if shape != (chunk.count, *self.shape) or dtype != np.dtype(self.dtype):
                    raise FrameStoreError(
                        "Frame-chunk shape or dtype does not match its manifest."
                    )
                expected_size = stream.tell() + math.prod(shape) * dtype.itemsize
                if os.fstat(stream.fileno()).st_size != expected_size:
                    raise FrameStoreError(
                        "Frame chunk is truncated or contains trailing data."
                    )
                stream.seek(0)
                # No mmap: the file is closed before data reaches the consumer.
                return cast(
                    NDArray[np.generic],
                    np.load(stream, allow_pickle=False, max_header_size=1024),
                )
        except (OSError, ValueError, EOFError) as exc:
            raise FrameStoreError("Cannot read a complete owned frame chunk.") from exc

    def read(self, start: int, count: int) -> NDArray[np.generic]:
        if (
            type(start) is not int
            or type(count) is not int
            or start < 0
            or not 0 < count <= self.batch_frames
            or start + count > len(self)
        ):
            raise FrameStoreError(
                "Frame range is outside the store or processing budget."
            )
        destination: NDArray[np.generic] = np.empty(
            (count, *self.shape), dtype=np.dtype(self.dtype)
        )
        offset = written = 0
        for chunk in self.chunks:
            end = offset + chunk.count
            if start < end and start + count > offset:
                source = self._read_chunk(chunk)
                first = max(start, offset) - offset
                take = min(start + count, end) - (offset + first)
                destination[written : written + take] = source[first : first + take]
                written += take
                del source
            offset = end
            if written == count:
                break
        return destination

    def iter_batches(
        self, check: Callable[[], None] = lambda: None
    ) -> Iterator[NDArray[np.generic]]:
        for start in range(0, len(self), self.batch_frames):
            check()
            batch = self.read(start, min(self.batch_frames, len(self) - start))
            yield batch
            del batch

    def discard(self) -> None:
        # Caller must be the final consumer. No readers/mappings survive read().
        directory = self._directory()
        expected = {chunk.name for chunk in self.chunks}
        if {path.name for path in directory.iterdir()} != expected:
            raise FrameStoreError("Refusing to delete unexpected frame-store files.")
        for chunk in self.chunks:
            path = directory / chunk.name
            if path.is_symlink() or not path.is_file():
                raise FrameStoreError("Refusing to delete an unsafe frame chunk.")
        for chunk in self.chunks:
            (directory / chunk.name).unlink()
        directory.rmdir()


class FrameStoreWriter:
    """One stage owns one writer; finish transfers ownership to an immutable handle."""

    def __init__(self, output_root: Path, prefix: str, stage: str, fps: float = 24.0):
        if not _STAGE.fullmatch(stage) or not math.isfinite(fps) or fps <= 0:
            raise FrameStoreError("Invalid frame-store stage or FPS.")
        self.output_root = _checked_directory(output_root)
        self.prefix, self.fps = prefix, fps
        attempt = _attempt_directory(self.output_root, prefix, create=True)
        self.directory_name = f"frames-{stage}-{uuid.uuid4().hex}"
        self.directory = attempt / self.directory_name
        self.directory.mkdir(mode=0o700)
        self.shape: tuple[int, int, int] | None = None
        self.dtype: str | None = None
        self.chunks: list[FrameChunk] = []
        self.closed = False

    def append(
        self, frames: NDArray[np.generic], check: Callable[[], None] = lambda: None
    ) -> None:
        if self.closed:
            raise FrameStoreError("Frame-store writer is closed.")
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[0] < 1:
            raise FrameStoreError("Expected a nonempty batch of HWC frames.")
        shape = (int(frames.shape[1]), int(frames.shape[2]), int(frames.shape[3]))
        dtype, limit = _layout(shape, frames.dtype.str)
        if self.shape is not None and (shape != self.shape or dtype.str != self.dtype):
            raise FrameStoreError("Frame layout cannot change within a store.")
        self.shape, self.dtype = shape, dtype.str
        _checked_directory(self.directory)
        for start in range(0, len(frames), limit):
            check()
            batch = frames[start : start + limit]
            name = f"{len(self.chunks):08d}.npy"
            temporary = self.directory / f".{name}.partial"
            try:
                required = batch.nbytes + 4096
                if shutil.disk_usage(self.directory).free < required:
                    raise OSError(
                        errno.ENOSPC, "Insufficient disk for lossless frame chunk"
                    )
                descriptor = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                )
                with os.fdopen(descriptor, "wb") as stream:
                    np.save(stream, batch, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                with temporary.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                os.replace(temporary, self.directory / name)
                self.chunks.append(FrameChunk(name, len(batch), digest))
            except OSError as exc:
                self.close()
                message = (
                    "Insufficient disk for lossless frame storage."
                    if exc.errno == errno.ENOSPC
                    else "Cannot write a complete owned frame chunk."
                )
                raise FrameStoreError(message) from exc

    def finish(self) -> FrameStore:
        if self.closed or self.shape is None or self.dtype is None:
            raise FrameStoreError("Cannot finish a closed or empty frame-store writer.")
        result = FrameStore(
            self.output_root,
            self.prefix,
            self.directory_name,
            self.shape,
            self.dtype,
            self.fps,
            tuple(self.chunks),
        )
        self.closed = True
        return result

    def close(self) -> None:
        # Files are opened only inside append(), whose context has already closed.
        # Leave incomplete data for attempt cleanup after execution has stopped.
        self.closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
