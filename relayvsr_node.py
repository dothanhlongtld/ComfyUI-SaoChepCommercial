"""Bounded bridge to the upstream ComfyUI RelayVSR model loader."""

from __future__ import annotations

import importlib
import sys
import threading
from typing import Any, Iterator

import numpy as np

from .frame_store import FrameStore, FrameStoreError, FrameStoreWriter

_RESTORE_LOCK = threading.Lock()


def _interrupt() -> None:
    importlib.import_module(
        "comfy.model_management"
    ).throw_exception_if_processing_interrupted()


def _input_frames(images: FrameStore) -> Iterator[Any]:
    torch = importlib.import_module("torch")
    for batch in images.iter_batches(check=_interrupt):
        try:
            for frame in batch:
                _interrupt()
                if frame.shape != images.shape or frame.shape[2] != 3:
                    raise FrameStoreError("RelayVSR accepts RGB frames only.")
                if frame.dtype == np.uint8:
                    rgb = frame
                elif frame.dtype.kind == "f" and np.isfinite(frame).all():
                    # Bicubic/lanczos IMAGE resize may overshoot [0,1]. Match
                    # upstream RelayVSR's pixel conversion at this boundary;
                    # clip before multiplication to avoid float overflow.
                    rgb = np.rint(frame.clip(0, 1) * 255).astype(np.uint8)
                else:
                    raise FrameStoreError("Frames must be uint8 or finite float RGB.")
                yield torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        finally:
            del batch


class AWGMemoryRelayVSR:
    CATEGORY = "AIWorkerGrid/Memory"
    RETURN_TYPES = ("AWG_FRAMES",)
    FUNCTION = "upscale"

    @classmethod
    def IS_CHANGED(cls, **_kwargs: Any) -> float:
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "images": ("AWG_FRAMES",),
                "model": ("RELAYVSR_MODEL",),
            }
        }

    def upscale(self, images: FrameStore, model: Any) -> tuple[FrameStore]:
        if not isinstance(images, FrameStore):
            raise FrameStoreError("RelayVSR requires an AWG frame-store handle.")
        if not callable(getattr(model, "ensure_loaded", None)):
            raise FrameStoreError("Connect the upstream RelayVSR model loader.")
        output_shape = (images.shape[0] * 4, images.shape[1] * 4, images.shape[2])
        writer = FrameStoreWriter(
            images.output_root, images.prefix, "relayvsr", images.fps
        )
        restore = iterator = None
        with _RESTORE_LOCK:
            try:
                _interrupt()
                torch = importlib.import_module("torch")
                with torch.inference_mode():
                    engine = model.ensure_loaded()
                    _interrupt()
                    iterator = _input_frames(images)
                    restore = engine.restore(
                        iterator,
                        seed=42,
                        sr_ratio=4,
                        chunk_len=16,
                        check=_interrupt,
                    )
                    expected = 0
                    for output in restore:
                        _interrupt()
                        array = np.asarray(output)
                        if array.dtype != np.uint8 or array.shape != output_shape:
                            raise FrameStoreError(
                                "RelayVSR output must be exact 4x HWC uint8 RGB."
                            )
                        if np.dtype(images.dtype).kind == "f":
                            array = array.astype(np.float32) / 255.0
                            array = array.astype(np.dtype(images.dtype), copy=False)
                        writer.append(array[None, ...], check=_interrupt)
                        expected += 1
                _interrupt()
                if expected != len(images):
                    raise FrameStoreError(
                        "RelayVSR output frame count does not match input."
                    )
                result = writer.finish()
            finally:
                primary_error = sys.exc_info()[0] is not None
                cleanup_errors = []
                for stream in (restore, iterator):
                    try:
                        close = getattr(stream, "close", None)
                        if close is not None:
                            close()
                    except BaseException as exc:
                        cleanup_errors.append(exc)
                try:
                    writer.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
                if cleanup_errors and not primary_error:
                    raise cleanup_errors[0]
        images.discard()
        return (result,)
