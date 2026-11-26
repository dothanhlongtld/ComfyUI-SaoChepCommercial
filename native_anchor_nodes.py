"""Comfy nodes for lossless native-frame continuation and Head Swap overlap validation."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path
from typing import Any

from .frame_store import FrameStoreWriter

ANCHOR_COUNT = 5
MAX_ANCHOR_BYTES = 48 * 1024**2


def anchor_byte_count(width: int, height: int) -> int:
    size = ANCHOR_COUNT * width * height * 3 * 4
    if min(width, height) <= 0 or size > MAX_ANCHOR_BYTES:
        raise ValueError("Unsupported native anchor geometry.")
    return size


def anchor_path(root: Path, prefix: str, part: int) -> Path:
    if not re.fullmatch(r"aiworkergrid/[a-f0-9]{32}", prefix) or not 0 <= part < 2:
        raise ValueError("Invalid native anchor owner.")
    directory = root / prefix
    if directory.resolve(strict=True) != directory.absolute() or not directory.is_dir():
        raise ValueError("Unsafe native anchor directory.")
    return directory / f"anchor-{part}.f32"


def headswap_anchor_scope(
    lineage: str, pose: Any, mask: Any, reference: Any, reference_mask: Any
) -> str:
    """Bind the anchor to the exact prepared inputs at the overlapping five frames."""
    if not re.fullmatch(r"[a-f0-9]{64}", lineage):
        raise ValueError("Invalid Head Swap shot/actor identity.")
    if (
        len(pose) != ANCHOR_COUNT
        or pose.shape != mask.shape
        or len(reference) != 1
        or reference.shape != reference_mask.shape
    ):
        raise ValueError("Invalid Head Swap overlap or reference geometry.")
    h = hashlib.sha256(b"AWG-HEAD-SWAP-ANCHOR-1" + bytes.fromhex(lineage))
    for tensor in (pose, mask, reference, reference_mask):
        array = (
            tensor.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False)
        )
        h.update(repr(tuple(array.shape)).encode() + b"\0")
        h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def anchor_scope_bytes(scope: str | None) -> bytes:
    if scope is None:
        return b""
    if not re.fullmatch(r"[a-f0-9]{64}", scope):
        raise ValueError("Invalid native anchor scope.")
    return bytes.fromhex(scope)


class _UncachedNativeNode:
    @classmethod
    def IS_CHANGED(cls, **_inputs: Any) -> float:
        # Native nodes participate in an ephemeral FrameStore pipeline. Never
        # reuse a consumed handle or vertices derived from an old checkpoint.
        return float("nan")


class AWGLoadNativeAnchor(_UncachedNativeNode):
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "AIWorkerGrid"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "optional": {"scope": ("STRING",)},
            "required": {
                key: (kind,)
                for key, kind in (
                    ("prefix", "STRING"),
                    ("part", "INT"),
                    ("width", "INT"),
                    ("height", "INT"),
                )
            },
        }

    def load(
        self, prefix: str, part: int, width: int, height: int, scope: str | None = None
    ) -> tuple[Any]:
        import folder_paths
        import numpy as np
        import torch

        path = anchor_path(Path(folder_paths.get_output_directory()), prefix, part)
        scope_prefix = anchor_scope_bytes(scope)
        expected = anchor_byte_count(width, height) + len(scope_prefix)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != expected:
                raise ValueError("Invalid native anchor file.")
            data = stream.read(expected + 1)
        if len(data) != expected:
            raise ValueError("Invalid native anchor length.")
        if data[: len(scope_prefix)] != scope_prefix:
            raise ValueError(
                "Head Swap anchor does not match prepared scene/actor overlap."
            )
        array = (
            np.frombuffer(data, dtype="<f4", offset=len(scope_prefix))
            .reshape(ANCHOR_COUNT, height, width, 3)
            .copy()
        )
        if not np.isfinite(array).all():
            raise ValueError("Non-finite native anchor pixels.")
        return (torch.from_numpy(array),)


class AWGOwnNativeFrames(_UncachedNativeNode):
    RETURN_TYPES = ("AWG_FRAMES",)
    FUNCTION = "keep"
    CATEGORY = "AIWorkerGrid"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "optional": {"scope": ("STRING",)},
            "required": {
                key: (kind,)
                for key, kind in (
                    ("frames", "AWG_FRAMES"),
                    ("owned", "INT"),
                    ("generated", "INT"),
                    ("prefix", "STRING"),
                    ("part", "INT"),
                    ("save", "BOOLEAN"),
                    ("width", "INT"),
                    ("height", "INT"),
                )
            },
        }

    def keep(
        self,
        frames: Any,
        owned: int,
        generated: int,
        prefix: str,
        part: int,
        save: bool,
        width: int,
        height: int,
        scope: str | None = None,
    ) -> tuple[Any]:
        import numpy as np
        from comfy.model_management import (
            throw_exception_if_processing_interrupted as check,
        )

        if (
            len(frames) != generated
            or not (0 < owned <= generated <= owned + 3 or (owned, generated) == (1, 5))
            or frames.fps != 24
            or frames.shape != (height, width, 3)
        ):
            raise ValueError("Invalid bounded native SCAIL output.")
        if save:
            expected = anchor_byte_count(width, height)
            check()
            tail = np.ascontiguousarray(frames.read(owned - ANCHOR_COUNT, ANCHOR_COUNT))
            if (
                tail.dtype != np.dtype("<f4")
                or tail.nbytes != expected
                or not np.isfinite(tail).all()
            ):
                raise ValueError("Native anchor is not lossless float32 RGB.")
            path = anchor_path(frames.output_root, prefix, part)
            fd = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(fd, "wb") as stream:
                stream.write(anchor_scope_bytes(scope))
                stream.write(memoryview(tail).cast("B"))
        if owned == generated:
            return (frames,)
        with FrameStoreWriter(
            frames.output_root, frames.prefix, "owned-native", frames.fps
        ) as writer:
            for start in range(0, owned, 4):
                check()
                writer.append(frames.read(start, min(4, owned - start)), check=check)
            result = writer.finish()
        frames.discard()
        return (result,)


class AWGSliceHeadSwapConditioning(_UncachedNativeNode):
    """Slice pose and mask together after full active-shot preparation."""

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    FUNCTION = "slice"
    CATEGORY = "AIWorkerGrid"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "pose": ("IMAGE",),
                "mask": ("IMAGE",),
                "reference": ("IMAGE",),
                "reference_mask": ("IMAGE",),
                "lineage": ("STRING",),
                "use_anchor": ("BOOLEAN",),
                "save_anchor": ("BOOLEAN",),
                "owned": ("INT",),
                **{
                    name: ("INT",)
                    for name in ("active_frames", "start", "stop", "padding")
                },
            }
        }

    def slice(
        self,
        pose: Any,
        mask: Any,
        active_frames: int,
        start: int,
        stop: int,
        padding: int,
        reference: Any,
        reference_mask: Any,
        lineage: str,
        use_anchor: bool,
        save_anchor: bool,
        owned: int,
    ) -> tuple[Any, Any, str, str]:
        import torch

        if (
            pose.ndim != 4
            or pose.shape != mask.shape
            or not 0 < active_frames <= len(pose)
            or not 0 <= start < stop <= active_frames
            or not 0 <= padding <= 4
            or (stop - start + padding - 1) % 4
            or (padding and stop != active_frames)
        ):
            raise ValueError("Invalid Head Swap paired conditioning window.")
        owned_end = start + (ANCHOR_COUNT if use_anchor else 0) + owned
        if owned <= 0 or owned_end > stop or (save_anchor and owned < ANCHOR_COUNT):
            raise ValueError("Invalid Head Swap owned conditioning range.")

        def scope(lo: int, hi: int) -> str:
            return headswap_anchor_scope(
                lineage, pose[lo:hi], mask[lo:hi], reference, reference_mask
            )

        previous = scope(start, start + ANCHOR_COUNT) if use_anchor else ""
        following = scope(owned_end - ANCHOR_COUNT, owned_end) if save_anchor else ""
        a, b = pose[start:stop], mask[start:stop]
        if padding:
            a = torch.cat((a, a[-1:].expand((padding, *a.shape[1:]))))
            b = torch.cat((b, b[-1:].expand((padding, *b.shape[1:]))))
        return a, b, previous, following
