"""Bounded AWG frame-store adapter for the immutable RelayVSR source checkout."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .frame_store import FrameStore, FrameStoreError, FrameStoreWriter


def _interrupt() -> None:
    importlib.import_module(
        "comfy.model_management"
    ).throw_exception_if_processing_interrupted()


@contextmanager
def _relay_import_path(root: Path) -> Iterator[None]:
    """Expose only the provisioned source during imports and remove new module cache entries."""
    resolved = str(root.resolve())
    existing = {
        name: module
        for name, module in sys.modules.items()
        if name == "refvsr_inference" or name.startswith("refvsr_inference.")
    }
    for module in existing.values():
        origin = getattr(module, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(root.resolve()):
            raise FrameStoreError("A different RelayVSR package is already loaded.")
    sys.path.insert(0, resolved)
    try:
        yield
    finally:
        try:
            sys.path.remove(resolved)
        except ValueError:
            pass
        for name in tuple(sys.modules):
            if (
                name == "refvsr_inference" or name.startswith("refvsr_inference.")
            ) and name not in existing:
                sys.modules.pop(name, None)


def _load_relay(source_root: Path, wan_checkpoint: Path, flash_checkpoint: Path):
    with _relay_import_path(source_root):
        torch = importlib.import_module("torch")
        checkpoints = importlib.import_module("refvsr_inference.checkpoints")
        pipeline_module = importlib.import_module(
            "refvsr_inference.models.wan.sampling"
        )
        stream_module = importlib.import_module("refvsr_inference.runtime.streaming")
        device = torch.device("cuda")
        dtype = torch.bfloat16
        wan, flash = checkpoints.load_checkpoints(
            wan_checkpoint, flash_checkpoint, device=device, dtype=dtype
        )
        sampler = pipeline_module.WanVSRPipeline(
            scheduler_config={"num_train_timesteps": 1000, "shift": 5.0},
            sampling={
                "execution_mode": "streaming",
                "num_inference_steps": 1,
                "timesteps": [1000],
                "guidance_scale": 1.0,
                "seed": 42,
                "kv_cache_window_size": wan.temporal_context_frames,
            },
        )
        sampler.prepare_streaming(
            model=wan,
            prompt_embeds=wan.prompt_embeds,
            device=device,
            compute_dtype=dtype,
        )
        stream = stream_module.CollaborativeFrameStream(
            wan_model=wan,
            flash_model=flash,
            wan_pipeline=sampler,
            prompt_embeds=wan.prompt_embeds,
            device=device,
            compute_dtype=dtype,
            window_size=16,
            seed=42,
        )
        return torch, stream_module.InputFrame, stream, (wan, flash, sampler)


def _input_packets(
    images: FrameStore, limit: int, torch: Any, input_frame: Any
) -> Iterator[Any]:
    for start in range(0, len(images), limit):
        _interrupt()
        batch = images.read(start, min(limit, len(images) - start))
        try:
            for offset, frame in enumerate(batch):
                _interrupt()
                if frame.shape != images.shape or frame.shape[2] != 3:
                    raise FrameStoreError("RelayVSR accepts RGB frames only.")
                rgb = frame
                if rgb.dtype != np.uint8:
                    if (
                        rgb.dtype.kind != "f"
                        or not np.isfinite(rgb).all()
                        or rgb.min() < 0
                        or rgb.max() > 1
                    ):
                        raise FrameStoreError(
                            "Float frames must be finite RGB in [0,1]."
                        )
                    rgb = np.rint(rgb * 255).clip(0, 255).astype(np.uint8)
                chw = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
                yield input_frame(start + offset, chw, (start + offset) / images.fps)
        finally:
            del batch


def _checkpoints() -> list[str]:
    try:
        folder_paths = importlib.import_module("folder_paths")
        return sorted(folder_paths.get_filename_list("relayvsr"))
    except (ImportError, AttributeError):
        return ["wan.safetensors", "flash.safetensors"]


class AWGMemoryRelayVSR:
    CATEGORY = "AIWorkerGrid/Memory"
    RETURN_TYPES = ("AWG_FRAMES",)
    FUNCTION = "upscale"

    @classmethod
    def IS_CHANGED(cls, **_kwargs: Any) -> float:
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        files = _checkpoints()
        wan_files = [
            name for name in files if Path(name).name == "wan.safetensors"
        ] or ["wan.safetensors"]
        flash_files = [
            name for name in files if Path(name).name == "flash.safetensors"
        ] or ["flash.safetensors"]
        return {
            "required": {
                "images": ("AWG_FRAMES",),
                "wan_checkpoint": (wan_files, {"default": "wan.safetensors"}),
                "flash_checkpoint": (flash_files, {"default": "flash.safetensors"}),
            }
        }

    def upscale(
        self,
        images: FrameStore,
        wan_checkpoint: str = "wan.safetensors",
        flash_checkpoint: str = "flash.safetensors",
    ) -> tuple[FrameStore]:
        if not isinstance(images, FrameStore):
            raise FrameStoreError("RelayVSR requires an AWG frame-store handle.")
        output_shape = (images.shape[0] * 4, images.shape[1] * 4, images.shape[2])
        limit = min(images.batch_frames, 16)
        root = Path(importlib.import_module("folder_paths").base_path)
        source_root = root / "relayvsr"
        checkpoint_dirs = importlib.import_module("folder_paths").get_folder_paths(
            "relayvsr"
        )
        if not checkpoint_dirs:
            raise FrameStoreError("RelayVSR checkpoint folder is not registered.")
        checkpoint_root = Path(checkpoint_dirs[0])
        if (
            wan_checkpoint != "wan.safetensors"
            or flash_checkpoint != "flash.safetensors"
        ):
            raise FrameStoreError("Unsupported RelayVSR checkpoint selection.")
        wan_path = checkpoint_root / wan_checkpoint
        flash_path = checkpoint_root / flash_checkpoint
        for path in (wan_path, flash_path):
            if not path.is_file():
                raise FrameStoreError(
                    f"RelayVSR checkpoint is unavailable: {path.name}."
                )
        if not source_root.is_dir():
            raise FrameStoreError("RelayVSR source checkout is unavailable.")

        # All Relay imports, models, and stream state are scoped to this invocation.
        torch: Any = None
        InputFrame: Any = None
        stream: Any = None
        _model_refs: Any = None
        generator: Any = None
        packet_generator: Any = None
        writer = FrameStoreWriter(
            images.output_root, images.prefix, "relayvsr", images.fps
        )
        succeeded = False
        try:
            _interrupt()
            # Ask ComfyUI's manager to make room without touching caller-held model objects.
            management = importlib.import_module("comfy.model_management")
            management.unload_all_models()
            _interrupt()
            torch, InputFrame, stream, _model_refs = _load_relay(
                source_root, wan_path, flash_path
            )

            _interrupt()
            packet_generator = _input_packets(images, limit, torch, InputFrame)
            generator = stream.frames(packet_generator)
            expected = 0
            with torch.inference_mode():
                for _result in generator:
                    _interrupt()
                    if _result.index != expected or tuple(_result.rgb.shape) != (
                        1,
                        3,
                        output_shape[0],
                        output_shape[1],
                    ):
                        raise FrameStoreError(
                            "RelayVSR output index or 4x shape is invalid."
                        )
                    _raw = _result.rgb[0]
                    if not bool(torch.isfinite(_raw).all().item()):
                        raise FrameStoreError("RelayVSR emitted non-finite RGB values.")
                    if np.dtype(images.dtype).kind == "f":
                        _tensor = _raw.add(1).mul_(0.5).clamp_(0, 1).to(torch.float32)
                    else:
                        _tensor = _raw.add(1).mul_(127.5).clamp_(0, 255).to(torch.uint8)
                    _array = _tensor.permute(1, 2, 0).contiguous().cpu().numpy()
                    if _array.shape != output_shape:
                        raise FrameStoreError(
                            "RelayVSR emitted an invalid output frame shape."
                        )
                    if np.dtype(images.dtype).kind == "f":
                        _array = _array.astype(np.dtype(images.dtype), copy=False)
                    _interrupt()
                    writer.append(_array[None, ...], check=_interrupt)
                    expected += 1
                    del _raw, _tensor, _array, _result
            _interrupt()
            if expected != len(images):
                raise FrameStoreError(
                    "RelayVSR output frame count does not match input."
                )
            result_store = writer.finish()
            succeeded = True
        finally:
            try:
                if generator is not None:
                    close = getattr(generator, "close", None)
                    if close is not None:
                        close()
            finally:
                try:
                    if packet_generator is not None:
                        packet_generator.close()
                finally:
                    writer.close()
                    stream = _model_refs = torch = InputFrame = None
                    _result = _raw = _tensor = _array = None
                    generator = packet_generator = None
        if not succeeded:
            raise FrameStoreError("RelayVSR did not complete successfully.")
        images.discard()
        return (result_store,)
