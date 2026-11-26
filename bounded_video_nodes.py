"""SaoChepCommercial's optional bounded-frame worker nodes.

AIWorkerGrid owns graph integration and process lifecycle; this repository owns
these custom node implementations and their lossless frame-store helpers.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import inspect
import json
import math
import textwrap
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .frame_store import (
    FrameStore,
    FrameStoreError,
    FrameStoreWriter,
    _layout,
)
from .native_anchor_nodes import (
    ANCHOR_COUNT,
    AWGLoadNativeAnchor,
    AWGOwnNativeFrames,
    AWGSliceHeadSwapConditioning,
)
from .relayvsr_node import AWGMemoryRelayVSR

_COMPATIBILITY = {
    "scail.INPUT_TYPES": {
        "parameters": [],
        "sha256": "ec5c87425873f36b52d804ac3357786d1c3e67f1383d9b8cc77016ecf88a8466",
    },
    "scail._plan": {
        "parameters": [
            "frames:POSITIONAL_OR_KEYWORD:required",
            "pose_len:POSITIONAL_OR_KEYWORD:required",
            "length:POSITIONAL_OR_KEYWORD:81",
            "pc:POSITIONAL_OR_KEYWORD:5",
            "anchored:POSITIONAL_OR_KEYWORD:False",
        ],
        "sha256": "69f0441e1314cc300ad55e9d66208bd5dcb928db0805a5892a5bf3fec3737014",
    },
    "scail.go": {
        "parameters": [
            "self:POSITIONAL_OR_KEYWORD:required",
            "model:POSITIONAL_OR_KEYWORD:required",
            "vae:POSITIONAL_OR_KEYWORD:required",
            "positive:POSITIONAL_OR_KEYWORD:required",
            "negative:POSITIONAL_OR_KEYWORD:required",
            "clip_vision_output:POSITIONAL_OR_KEYWORD:required",
            "reference_image:POSITIONAL_OR_KEYWORD:required",
            "pose_video:POSITIONAL_OR_KEYWORD:required",
            "pose_video_mask:POSITIONAL_OR_KEYWORD:required",
            "sampler:POSITIONAL_OR_KEYWORD:required",
            "sigmas:POSITIONAL_OR_KEYWORD:required",
            "frames:POSITIONAL_OR_KEYWORD:required",
            "video_frames:POSITIONAL_OR_KEYWORD:required",
            "seed:POSITIONAL_OR_KEYWORD:required",
            "cfg:POSITIONAL_OR_KEYWORD:required",
            "replacement_mode:POSITIONAL_OR_KEYWORD:required",
            "width:POSITIONAL_OR_KEYWORD:required",
            "height:POSITIONAL_OR_KEYWORD:required",
            "length:POSITIONAL_OR_KEYWORD:81",
            "previous_frame_count:POSITIONAL_OR_KEYWORD:5",
            "reference_image_mask:POSITIONAL_OR_KEYWORD:None",
            "anchor_frames:POSITIONAL_OR_KEYWORD:None",
        ],
        "sha256": "a34e9d92735484917fcefaa733c78056d31c0c5f525de13d76124d9aa1129be5",
    },
    "resize.INPUT_TYPES": {
        "parameters": [],
        "sha256": "c79083c4754a4d3a608b8adc4e3fee2cbbba34aafbf2b5d24ee13edce4a043d8",
    },
    "resize.resize": {
        "parameters": [
            "self:POSITIONAL_OR_KEYWORD:required",
            "image:POSITIONAL_OR_KEYWORD:required",
            "width:POSITIONAL_OR_KEYWORD:required",
            "height:POSITIONAL_OR_KEYWORD:required",
            "keep_proportion:POSITIONAL_OR_KEYWORD:required",
            "upscale_method:POSITIONAL_OR_KEYWORD:required",
            "divisible_by:POSITIONAL_OR_KEYWORD:required",
            "pad_color:POSITIONAL_OR_KEYWORD:required",
            "crop_position:POSITIONAL_OR_KEYWORD:required",
            "unique_id:POSITIONAL_OR_KEYWORD:required",
            "device:POSITIONAL_OR_KEYWORD:'cpu'",
            "mask:POSITIONAL_OR_KEYWORD:None",
            "per_batch:POSITIONAL_OR_KEYWORD:64",
        ],
        "sha256": "dbc4c77e94fb9d17af5f6af8df834cab3251c53707bb301721dd7d3a530d5e7c",
    },
    "vhs.INPUT_TYPES": {
        "parameters": [],
        "sha256": "7be8e04595b52b975d6bfaf3a34563a0295291d76781ffda4dc2bee79f18314e",
    },
    "vhs.combine_video": {
        "parameters": [
            "self:POSITIONAL_OR_KEYWORD:required",
            "frame_rate:POSITIONAL_OR_KEYWORD:required",
            "loop_count:POSITIONAL_OR_KEYWORD:required",
            "images:POSITIONAL_OR_KEYWORD:None",
            "latents:POSITIONAL_OR_KEYWORD:None",
            "filename_prefix:POSITIONAL_OR_KEYWORD:'AnimateDiff'",
            "format:POSITIONAL_OR_KEYWORD:'image/gif'",
            "pingpong:POSITIONAL_OR_KEYWORD:False",
            "save_output:POSITIONAL_OR_KEYWORD:True",
            "prompt:POSITIONAL_OR_KEYWORD:None",
            "extra_pnginfo:POSITIONAL_OR_KEYWORD:None",
            "audio:POSITIONAL_OR_KEYWORD:None",
            "unique_id:POSITIONAL_OR_KEYWORD:None",
            "manual_format_widgets:POSITIONAL_OR_KEYWORD:None",
            "meta_batch:POSITIONAL_OR_KEYWORD:None",
            "vae:POSITIONAL_OR_KEYWORD:None",
            "kwargs:VAR_KEYWORD:required",
        ],
        "sha256": "f0b21fbea9f8528d5660a02eea913ac80fbbc928ab368f6a643a3e0555c15aaa",
    },
    "vhs_module.iterate_format": {
        "parameters": [
            "video_format:POSITIONAL_OR_KEYWORD:required",
            "for_widgets:POSITIONAL_OR_KEYWORD:True",
        ],
        "sha256": "20e6050f71f73c8d2d53004875130b27c15f9e87de6af0814a084c69c237ad04",
    },
    "vhs_module.apply_format_widgets": {
        "parameters": [
            "format_name:POSITIONAL_OR_KEYWORD:required",
            "kwargs:POSITIONAL_OR_KEYWORD:required",
        ],
        "sha256": "ae1cdd8eb81bf16f54bede5c55a805189d2fec62282983063fc0731b609b725c",
    },
    "vhs_module.tensor_to_int": {
        "parameters": [
            "tensor:POSITIONAL_OR_KEYWORD:required",
            "bits:POSITIONAL_OR_KEYWORD:required",
        ],
        "sha256": "abedb637acd9eea0909da005e4267945838675323a71cc79ddf7765a34eab72d",
    },
    "vhs_module.tensor_to_bytes": {
        "parameters": ["tensor:POSITIONAL_OR_KEYWORD:required"],
        "sha256": "f973417356cfb5e834d8d325b8efa8bc45604fce522d60d96849a04f12590666",
    },
    "vhs_module.ffmpeg_process": {
        "parameters": [
            "args:POSITIONAL_OR_KEYWORD:required",
            "video_format:POSITIONAL_OR_KEYWORD:required",
            "video_metadata:POSITIONAL_OR_KEYWORD:required",
            "file_path:POSITIONAL_OR_KEYWORD:required",
            "env:POSITIONAL_OR_KEYWORD:required",
        ],
        "sha256": "3b3d362f04b173a24cad7743bb4ed45b1d2ea02fc0091fe27eeb8f204f9b54b7",
    },
    "condition.execute": {
        "parameters": [
            "positive:POSITIONAL_OR_KEYWORD:required",
            "negative:POSITIONAL_OR_KEYWORD:required",
            "vae:POSITIONAL_OR_KEYWORD:required",
            "width:POSITIONAL_OR_KEYWORD:required",
            "height:POSITIONAL_OR_KEYWORD:required",
            "length:POSITIONAL_OR_KEYWORD:required",
            "batch_size:POSITIONAL_OR_KEYWORD:required",
            "pose_strength:POSITIONAL_OR_KEYWORD:required",
            "pose_start:POSITIONAL_OR_KEYWORD:required",
            "pose_end:POSITIONAL_OR_KEYWORD:required",
            "video_frame_offset:POSITIONAL_OR_KEYWORD:required",
            "previous_frame_count:POSITIONAL_OR_KEYWORD:required",
            "replacement_mode:POSITIONAL_OR_KEYWORD:False",
            "reference_image:POSITIONAL_OR_KEYWORD:None",
            "clip_vision_output:POSITIONAL_OR_KEYWORD:None",
            "pose_video:POSITIONAL_OR_KEYWORD:None",
            "pose_video_mask:POSITIONAL_OR_KEYWORD:None",
            "reference_image_mask:POSITIONAL_OR_KEYWORD:None",
            "previous_frames:POSITIONAL_OR_KEYWORD:None",
        ],
        "sha256": "336209ce1586a0f7a4606fb4faf8d573b9c4a98d4eea1345b1179f6260f00ad5",
    },
    "sampler.execute": {
        "parameters": [
            "model:POSITIONAL_OR_KEYWORD:required",
            "add_noise:POSITIONAL_OR_KEYWORD:required",
            "noise_seed:POSITIONAL_OR_KEYWORD:required",
            "cfg:POSITIONAL_OR_KEYWORD:required",
            "positive:POSITIONAL_OR_KEYWORD:required",
            "negative:POSITIONAL_OR_KEYWORD:required",
            "sampler:POSITIONAL_OR_KEYWORD:required",
            "sigmas:POSITIONAL_OR_KEYWORD:required",
            "latent_image:POSITIONAL_OR_KEYWORD:required",
        ],
        "sha256": "0c3e3249cc4818aad2cea438b866c51a4dc6047b386817abdbe3828be265bd5d",
    },
    "decode.decode": {
        "parameters": [
            "self:POSITIONAL_OR_KEYWORD:required",
            "vae:POSITIONAL_OR_KEYWORD:required",
            "samples:POSITIONAL_OR_KEYWORD:required",
        ],
        "sha256": "234d676adb42bbcf1bff5d0e4c285dc97b6a4bbb0de3b247c75f3c7a26c21db1",
    },
}

_H264_FORMAT = {
    "main_pass": [
        "-n",
        "-c:v",
        "libx264",
        "-pix_fmt",
        ["pix_fmt", ["yuv420p", "yuv420p10le"]],
        "-crf",
        ["crf", "INT", {"default": 19, "min": 0, "max": 100, "step": 1}],
        "-vf",
        "scale=out_color_matrix=bt709",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
    ],
    "fake_trc": "bt709",
    "audio_pass": ["-c:a", "aac", "-movflags", "use_metadata_tags"],
    "save_metadata": ["save_metadata", "BOOLEAN", {"default": True}],
    "trim_to_audio": ["trim_to_audio", "BOOLEAN", {"default": False}],
    "extension": "mp4",
}


# Adaptation references: owned SaoChep helpers use the local versioned contract;
# only external callable ASTs are fingerprint-gated. Temporal formulas retain
# a09671df reference behavior; the clarity shader has one canonical implementation.
# SCAIL helper 2a0d3c0d: SCAIL2AutoVideo._plan/go; Comfy 3c1b7a17 sampler/conditioner/VAE.
# KJ d7641533: ImageResizeKJv2; VHS 4d907bee: VideoCombine and h264-mp4 format.

BOUNDED_VIDEO_API_VERSION = 2

# Static worker schemas keep ordinary SaoChep node inspection independent of
# optional SCAIL/KJ/VHS installation and readiness order. External execution
# helpers remain compatibility-gated; return fresh copies to callers.
_INPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "scail": {
        "required": {
            "model": ("MODEL",),
            "vae": ("VAE",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            "reference_image": ("IMAGE",),
            "pose_video": ("IMAGE",),
            "pose_video_mask": ("IMAGE",),
            "sampler": ("SAMPLER",),
            "sigmas": ("SIGMAS",),
            "frames": ("INT", {"default": 160, "min": 0}),
            "video_frames": ("INT", {"default": 480, "min": 1}),
            "seed": ("INT", {"default": 1, "min": 0, "max": 18446744073709551615}),
            "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            "replacement_mode": ("BOOLEAN", {"default": False}),
            "width": ("INT", {"default": 832, "min": 0, "step": 32}),
            "height": ("INT", {"default": 480, "min": 0, "step": 32}),
            "length": ("INT", {"default": 81, "min": 5, "max": 512, "step": 4}),
            "previous_frame_count": (
                "INT",
                {"default": 5, "min": 1, "max": 32, "step": 4},
            ),
        },
        "optional": {
            "reference_image_mask": ("IMAGE",),
            "anchor_frames": ("IMAGE", {}),
        },
    },
    "resize": {
        "required": {
            "image": ("IMAGE",),
            "width": ("INT", {"default": 512, "min": 0, "max": 16384, "step": 1}),
            "height": ("INT", {"default": 512, "min": 0, "max": 16384, "step": 1}),
            "upscale_method": (
                [
                    "nearest-exact",
                    "bilinear",
                    "area",
                    "bicubic",
                    "lanczos",
                    "nvidia_rtx_vsr",
                ],
            ),
            "keep_proportion": (
                [
                    "stretch",
                    "resize",
                    "pad",
                    "pad_edge",
                    "pad_edge_pixel",
                    "crop",
                    "pillarbox_blur",
                    "total_pixels",
                ],
                {"default": False},
            ),
            "pad_color": ("STRING", {"default": "0, 0, 0"}),
            "crop_position": (
                ["center", "top", "bottom", "left", "right"],
                {"default": "center"},
            ),
            "divisible_by": ("INT", {"default": 2, "min": 0, "max": 512, "step": 1}),
        },
        "optional": {"mask": ("MASK",), "device": (["cpu", "gpu"],)},
        "hidden": {"unique_id": "UNIQUE_ID"},
    },
    "chroma": {
        "required": {
            "images": ("IMAGE", {}),
            "reference_image": ("IMAGE", {}),
            "strength": (
                "FLOAT",
                {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05},
            ),
            "max_ab_offset": (
                "FLOAT",
                {"default": 2.0, "min": 0.0, "max": 5.0, "step": 0.1},
            ),
            "sample_count": ("INT", {"default": 9, "min": 3, "max": 25, "step": 1}),
            "min_valid_samples": (
                "INT",
                {"default": 3, "min": 1, "max": 15, "step": 1},
            ),
            "min_face_width_px": (
                "INT",
                {"default": 64, "min": 32, "max": 256, "step": 8},
            ),
            "min_patch_pixels": (
                "INT",
                {"default": 32, "min": 8, "max": 128, "step": 4},
            ),
            "bypass_offset_threshold": (
                "FLOAT",
                {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.02},
            ),
            "processing_backend": (
                ["auto", "gpu", "cpu_fast", "baseline"],
                {"default": "auto"},
            ),
            "batch_frames": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
            "debug_enabled": ("BOOLEAN", {"default": False}),
            "benchmark_run_id": (
                "INT",
                {"default": 0, "min": 0, "max": 999999, "step": 1},
            ),
            "enabled": ("BOOLEAN", {"default": False}),
        },
        "optional": {
            "branch": (["CHROMA_LIGHT", "WARM_HALF"], {"default": "CHROMA_LIGHT"}),
            "test_mode": (
                ["auto_end_to_end", "TRANSFORM_ONLY_TEST"],
                {"default": "auto_end_to_end"},
            ),
        },
    },
    "clarity": {
        "required": {"images": ("IMAGE",)},
        "optional": {
            "adaptive_mode": ("BOOLEAN", {"default": True}),
            "blend_opacity": (
                "FLOAT",
                {"default": 0.31, "min": 0.0, "max": 1.0, "step": 0.01},
            ),
            "sharpen_strength": (
                "FLOAT",
                {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
            ),
            "clarity_strength": (
                "FLOAT",
                {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05},
            ),
        },
    },
    "vhs": {
        "required": {
            "images": ("AWG_FRAMES",),
            "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0}),
            "loop_count": ("INT", {"default": 0, "min": 0, "max": 0}),
            "filename_prefix": ("STRING",),
            "format": (["video/h264-mp4"],),
            "pingpong": ("BOOLEAN", {"default": False}),
            "save_output": ("BOOLEAN", {"default": True}),
        },
        "optional": {
            "audio": ("AUDIO",),
            "pix_fmt": (["yuv420p", "yuv420p10le"],),
            "crf": ("INT", {"default": 20, "min": 0, "max": 100}),
            "save_metadata": ("BOOLEAN", {"default": False}),
            "trim_to_audio": ("BOOLEAN", {"default": False}),
        },
        "hidden": {
            "prompt": "PROMPT",
            "extra_pnginfo": "EXTRA_PNGINFO",
            "unique_id": "UNIQUE_ID",
        },
    },
}

_BINDINGS: dict[str, Any] = {}


def _torch() -> Any:
    return importlib.import_module("torch")


def _interrupt() -> None:
    importlib.import_module(
        "comfy.model_management"
    ).throw_exception_if_processing_interrupted()


def _binding(name: str) -> Any:
    if not _BINDINGS:
        raise FrameStoreError(
            "Worker memory-node compatibility has not been validated."
        )
    return _BINDINGS[name]


def _canonical_ast(value: Any) -> Any:
    # ast.dump() changes empty-field formatting between Python 3.12 and 3.14.
    # Omit absent/empty AST fields explicitly, never source location attributes.
    if isinstance(value, ast.AST):
        return {
            "node": type(value).__name__,
            **{
                name: _canonical_ast(item)
                for name, item in ast.iter_fields(value)
                if item is not None and item != []
            },
        }
    if isinstance(value, list):
        return [_canonical_ast(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if value is Ellipsis:
        return {"ellipsis": True}
    return value


def _source_fingerprint(source: str) -> str:
    tree = _canonical_ast(ast.parse(textwrap.dedent(source)))
    return hashlib.sha256(
        json.dumps(tree, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _callable_fingerprint(function: Any) -> str:
    return _source_fingerprint(inspect.getsource(function))


def _verify_symbol(name: str, function: Any, expected: dict[str, Any]) -> None:
    try:
        parameters = [
            f"{name}:{parameter.kind.name}:"
            + (
                "required"
                if parameter.default is inspect.Parameter.empty
                else repr(parameter.default)
            )
            for name, parameter in inspect.signature(function).parameters.items()
        ]
        fingerprint = _callable_fingerprint(function)
    except (TypeError, ValueError, OSError, SyntaxError) as exc:
        raise FrameStoreError(
            f"Cannot verify required memory-node helper: {name}."
        ) from exc
    if parameters != expected["parameters"] or fingerprint != expected["sha256"]:
        raise FrameStoreError(
            f"Unsupported memory-node helper behavior/signature: {name}."
        )


def validate_dependencies(registry: Mapping[str, Any]) -> None:
    _BINDINGS.clear()
    required = {
        "scail": "SCAIL2AutoVideo",
        "chroma": "ReferenceChromaLightTest",
        "resize": "ImageResizeKJv2",
        "clarity": "SaoChepAdaptiveClarityEnhancer",
        "vhs": "VHS_VideoCombine",
    }
    try:
        bindings = {key: registry[node] for key, node in required.items()}
        for key in ("scail", "chroma", "vhs"):
            bindings[f"{key}_module"] = inspect.getmodule(bindings[key])
        bindings["condition"] = bindings["scail_module"].WanSCAILToVideo
        bindings["sampler"] = bindings["scail_module"].SamplerCustom
        bindings["decode"] = bindings["scail_module"].VAEDecode
        for name, expected in _COMPATIBILITY.items():
            owner, attribute = name.split(".", 1)
            _verify_symbol(name, getattr(bindings[owner], attribute), expected)
        clarity_chunk = inspect.signature(bindings["clarity"]._process_chunk).parameters
        if (
            clarity_chunk.get("scene_luma") is None
            or clarity_chunk["scene_luma"].default is not None
        ):
            raise FrameStoreError("Clarity helper lacks the compatible luminance seam.")
        raw_format = Path(bindings["vhs_module"].base_formats_dir) / "h264-mp4.json"
        actual_format = json.loads(raw_format.read_text(encoding="utf-8"))
        if actual_format != _H264_FORMAT:
            raise FrameStoreError(
                "Unsupported H.264 format; bounded encoding cannot be guaranteed."
            )
    except (KeyError, AttributeError, OSError, ValueError) as exc:
        raise FrameStoreError(
            "Required bounded-worker helpers/format are unavailable."
        ) from exc
    _BINDINGS.update(bindings)


class _StoreNode:
    CATEGORY = "AIWorkerGrid/Memory"
    RETURN_TYPES = ("AWG_FRAMES",)
    _source = ""

    @classmethod
    def IS_CHANGED(cls, **_kwargs: Any) -> float:
        # Stage files are reclaimed after their last consumer. A cached handle
        # must never be reused, including on reexecution of an identical prompt.
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        schema = copy.deepcopy(_INPUT_SCHEMAS[cls._source])
        schema["required"]["images"] = ("AWG_FRAMES",)
        return schema


def _writer(source: FrameStore, stage: str) -> FrameStoreWriter:
    return FrameStoreWriter(source.output_root, source.prefix, stage, source.fps)


def _numpy(tensor: Any) -> NDArray[Any]:
    return cast(NDArray[Any], tensor.detach().cpu().numpy())


class AWGMemoryGenerate(_StoreNode):
    FUNCTION = "generate"
    BOUNDED_VIDEO_API_VERSION = BOUNDED_VIDEO_API_VERSION
    validate_dependencies = staticmethod(validate_dependencies)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        schema = copy.deepcopy(_INPUT_SCHEMAS["scail"])
        schema["required"]["attempt_prefix"] = ("STRING",)
        return schema

    def generate(self, attempt_prefix: str, **inputs: Any) -> tuple[FrameStore]:
        return self._generate(attempt_prefix, inputs)

    def _generate(
        self,
        attempt_prefix: str,
        inputs: Mapping[str, Any],
        continuation: tuple[int, int] | None = None,
    ) -> tuple[FrameStore]:
        frames = (
            max(1, int(inputs["frames"]))
            if inputs["frames"] > 0
            else max(
                1,
                int(inputs["video_frames"])
                if inputs["video_frames"] and inputs["video_frames"] > 0
                else 81,
            )
        )
        length, pc = (
            max(1, int(inputs.get("length", 81))),
            max(1, int(inputs.get("previous_frame_count", 5))),
        )
        pose = inputs["pose_video"]
        pose_len = pose.shape[0] if pose is not None else frames
        previous = inputs.get("anchor_frames")
        anchored = previous is not None and previous.shape[0] > 0
        if continuation is not None:
            owned, context = continuation
            mask = inputs.get("pose_video_mask")
            if (
                not (0 < owned <= frames <= owned + 3 or (not anchored and (owned, frames) == (1, 5)))
                or pc != ANCHOR_COUNT or length != 85
                or context != frames + (pc if anchored else 0)
                or pose is None or pose_len != context or (context - 1) % 4
                or mask is None or mask.shape != pose.shape
                or (anchored and previous.shape[0] != pc)
            ):
                raise FrameStoreError("Invalid SCAIL anchor conditioning window.")
        if not anchored:
            previous = None
        elif previous is not None and previous.shape[0] > pc:
            previous = previous[-pc:].clone()
        plan = _binding("scail")._plan(frames, pose_len, length, pc, anchored)
        output_root = Path(
            importlib.import_module("folder_paths").get_output_directory()
        )
        written, last = 0, None
        with FrameStoreWriter(output_root, attempt_prefix, "generated") as writer:
            for index, (offset, chunk_length) in enumerate(plan):
                _interrupt()
                call_offset = offset + pc if index == 0 and anchored else offset
                had_previous = previous is not None
                conditioning = _binding("condition").execute(
                    inputs["positive"],
                    inputs["negative"],
                    inputs["vae"],
                    int(inputs["width"]),
                    int(inputs["height"]),
                    chunk_length,
                    1,
                    1.0,
                    0.0,
                    1.0,
                    call_offset,
                    pc,
                    bool(inputs["replacement_mode"]),
                    reference_image=inputs["reference_image"],
                    clip_vision_output=inputs["clip_vision_output"],
                    pose_video=pose,
                    pose_video_mask=inputs["pose_video_mask"],
                    reference_image_mask=inputs.get("reference_image_mask"),
                    previous_frames=previous,
                )
                positive, negative, latent = (
                    conditioning[0],
                    conditioning[1],
                    conditioning[2],
                )
                if hasattr(positive, "result"):
                    positive, negative = positive.result[0], negative.result[0]
                sampled = _binding("sampler").execute(
                    inputs["model"],
                    True,
                    int(inputs["seed"]),
                    float(inputs["cfg"]),
                    positive,
                    negative,
                    inputs["sampler"],
                    inputs["sigmas"],
                    latent,
                )
                if hasattr(sampled, "result"):
                    sampled = sampled.result
                decoded = _binding("decode")().decode(inputs["vae"], sampled[0])[0]
                if continuation is not None:
                    if index + 1 < len(plan) and decoded.shape[0] != chunk_length:
                        raise FrameStoreError("SCAIL intermediate decode did not match its planned length.")
                    if decoded.shape[0] > chunk_length:
                        raise FrameStoreError("SCAIL final decode exceeds its planned length.")
                if had_previous and decoded.shape[0] > 0:
                    decoded = decoded[min(pc, decoded.shape[0]) :]
                # The upstream conditioner uses only this tail. clone() releases
                # the rest of the decoded segment rather than retaining a view.
                previous = decoded[-pc:].clone()
                _interrupt()
                take = min(decoded.shape[0], max(0, frames - written))
                if take:
                    pixels = _numpy(decoded[:take])
                    writer.append(pixels, check=_interrupt)
                    last = pixels[-1:].copy()
                    written += take
                    del pixels
                del decoded, sampled, conditioning, positive, negative, latent
            if last is None:
                raise FrameStoreError(
                    "SCAIL produced no frames for the requested segment plan."
                )
            if continuation is not None and written < continuation[0]:
                raise FrameStoreError("SCAIL did not decode every owned frame.")
            while written < frames:
                _interrupt()
                count = min(
                    _layout(
                        (last.shape[1], last.shape[2], last.shape[3]), last.dtype.str
                    )[1],
                    frames - written,
                )
                writer.append(np.repeat(last, count, axis=0), check=_interrupt)
                written += count
            return (writer.finish(),)


class AWGAnchoredGenerate(AWGMemoryGenerate):
    """Bounded generation that distinguishes owned frames from alignment."""

    NATIVE_ANCHOR_API_VERSION = 1

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        schema = super().INPUT_TYPES()
        schema["required"].update(owned_frames=("INT",), context_frames=("INT",))
        return schema

    def generate(self, attempt_prefix: str, **inputs: Any) -> tuple[FrameStore]:
        owned = inputs.pop("owned_frames")
        context = inputs.pop("context_frames")
        return self._generate(attempt_prefix, inputs, (owned, context))


def _chroma_offsets(
    source: FrameStore, reference: Any, node: Any, settings: Mapping[str, Any]
) -> Any:
    # Same sample selection, validity, scalar bypass, drift and interpolation as
    # ReferenceChromaLightTest.apply_chroma_test (a09671df), without its RGB buffer.
    sample = node._sample_cheek_color
    face_w, patch_pixels = (
        settings.get("min_face_width_px", 64),
        settings.get("min_patch_pixels", 32),
    )
    ref = sample(
        np.clip(_numpy(reference[0]) * 255.0, 0, 255).astype(np.uint8),
        face_w,
        patch_pixels,
    )
    if not ref["valid"]:
        return None
    count = settings.get("sample_count", 9)
    indices = sorted(
        {
            round((0.10 + i * (0.80 / max(1, count - 1))) * (len(source) - 1))
            for i in range(count)
        }
    )
    valid = []
    for index in indices:
        _interrupt()
        result = sample(
            np.clip(source.read(index, 1)[0] * 255.0, 0, 255).astype(np.uint8),
            face_w,
            patch_pixels,
        )
        if result["valid"]:
            valid.append({**result, "frame_idx": index})
    if len(valid) < settings.get("min_valid_samples", 3):
        return None
    strength, maximum = (
        settings.get("strength", 0.35),
        settings.get("max_ab_offset", 2.0),
    )
    medians = [
        float(np.median([sample[axis] for sample in valid])) for axis in ("a", "b")
    ]
    offsets = [
        np.clip(strength * (ref[axis] - median), -maximum, maximum)
        for axis, median in zip(("a", "b"), medians)
    ]
    per_sample = [
        [
            float(np.clip(strength * (ref[axis] - sample[axis]), -maximum, maximum))
            for sample in valid
        ]
        for axis in ("a", "b")
    ]
    drift = any(max(values) - min(values) > 0.4 for values in per_sample)
    if max(abs(float(value)) for value in offsets) <= settings.get(
        "bypass_offset_threshold", 0.20
    ):
        return None  # Preserve the global bypass even when sample offsets drift.
    if drift and len(valid) >= 3 and len(source) > 1:
        return tuple(
            np.interp(
                np.arange(len(source)),
                [sample["frame_idx"] for sample in valid],
                values,
            ).astype(np.float32)
            for values in per_sample
        )
    return tuple(float(value) for value in offsets)


class AWGMemoryChroma(_StoreNode):
    _source = "chroma"
    FUNCTION = "correct"

    def correct(
        self, images: FrameStore, reference_image: Any, **settings: Any
    ) -> tuple[FrameStore]:
        _interrupt()
        if (
            settings.get("branch", "CHROMA_LIGHT") != "CHROMA_LIGHT"
            or settings.get("test_mode", "auto_end_to_end") != "auto_end_to_end"
            or settings.get("debug_enabled", False)
        ):
            raise FrameStoreError(
                "Bounded chroma supports the prepared end-to-end processing path only."
            )
        if not settings.get("enabled", False):
            return (images,)
        node = _binding("chroma")()
        offsets = _chroma_offsets(images, reference_image, node, settings)
        if offsets is None:
            return (images,)
        torch = _torch()
        if (
            settings.get("processing_backend", "gpu") != "gpu"
            or not torch.cuda.is_available()
        ):
            raise FrameStoreError(
                "Prepared bounded chroma requires CUDA; no backend fallback."
            )
        transform = _binding("chroma_module").transform_cielab_batch_torch
        requested = min(images.batch_frames, int(settings.get("batch_frames", 16)))
        limit = max(
            1,
            min(
                images.batch_frames,
                node._select_adaptive_batch_frames(
                    requested, images.shape[0], images.shape[1]
                ),
            ),
        )
        with _writer(images, "chroma") as writer:
            start = 0
            while start < len(images):
                _interrupt()
                count = min(limit, len(images) - start)
                cpu = batch = corrected = a = b = None
                try:
                    cpu = torch.from_numpy(images.read(start, count))
                    batch = cpu.to("cuda", non_blocking=True)
                    a, b = [
                        torch.from_numpy(value[start : start + count])
                        .to(device=batch.device, dtype=torch.float32)
                        .view(-1, 1, 1)
                        if isinstance(value, np.ndarray)
                        else value
                        for value in offsets
                    ]
                    corrected, _ = transform(
                        batch, a, b, device="cuda", sync_cuda=False
                    )
                    writer.append(
                        _numpy(corrected.cpu().to(dtype=cpu.dtype)), check=_interrupt
                    )
                    start += count
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower() or limit <= 2:
                        raise
                    # Same retry policy as upstream: change only group size, not
                    # pixels, offsets, precision, frame ordering or backend.
                    limit = max(2, limit // 2)
                    cpu = batch = corrected = a = b = None
                    torch.cuda.empty_cache()
                finally:
                    cpu = batch = corrected = a = b = None
            result = writer.finish()
        images.discard()
        return (result,)


class AWGMemoryResize(_StoreNode):
    _source = "resize"
    FUNCTION = "resize"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        schema = copy.deepcopy(_INPUT_SCHEMAS["resize"])
        schema["required"]["image"] = ("AWG_FRAMES",)
        schema["optional"].pop("mask", None)
        return schema

    def resize(self, image: FrameStore, **settings: Any) -> tuple[FrameStore]:
        # KJ's input is named image, not images.
        limit = min(
            image.batch_frames,
            _layout(
                (int(settings["height"]), int(settings["width"]), image.shape[2]),
                image.dtype,
            )[1],
        )
        with _writer(image, "delivery") as writer:
            for start in range(0, len(image), limit):
                _interrupt()
                cpu = _torch().from_numpy(
                    image.read(start, min(limit, len(image) - start))
                )
                resized = _binding("resize")().resize(image=cpu, **settings)[0]
                writer.append(_numpy(resized), check=_interrupt)
                del cpu, resized
            result = writer.finish()
        image.discard()
        return (result,)


def _frame_means(source: FrameStore, skip: int = 0) -> Any:
    torch = _torch()
    means = None
    for start in range(0, len(source) - skip, source.batch_frames):
        _interrupt()
        count = min(source.batch_frames, len(source) - skip - start)
        batch = torch.from_numpy(source.read(start + skip, count))
        if means is None:
            means = torch.empty(
                (len(source) - skip, 3), dtype=batch.dtype, device="cpu"
            )
        means[start : start + count] = batch.mean(dim=(1, 2))
    return means


def _head_plan(means: Any) -> Any:
    if len(means) < 30:
        return None
    head = min(60, max(24, int(len(means) * 0.20)))
    window = means[head : min(head + 48, len(means))]
    if not len(window):
        return None
    return head, _torch().median(window, dim=0).values


def _correct_head(batch: Any, start: int, means: Any, plan: Any) -> None:
    if plan is None:
        return
    torch = _torch()
    head, reference = plan
    for index in range(start, min(start + len(batch), head)):
        _interrupt()
        delta = means[index] - reference
        if torch.max(torch.abs(delta)) > 0.008:
            weight = 0.5 * (
                1.0
                + torch.cos(
                    torch.tensor(
                        3.141592653589793 * index / float(head),
                        device=batch.device,
                        dtype=batch.dtype,
                    )
                )
            )
            batch[index - start] = torch.clamp(
                batch[index - start] - (delta * weight).view(1, 1, 3), 0.0, 1.0
            )


def _tail_plan(means: Any) -> Any:
    # Exact temporal orchestration from a09671df, operating only on N x 3 means.
    count = len(means)
    if count < 60:
        return None
    head = min(60, max(24, int(count * 0.20)))
    end = min(count, max(head + 48, int(count * 0.50)))
    window = means[head:end]
    if not len(window) or end >= count - 8:
        return None
    torch = _torch()
    reference = torch.median(window, dim=0).values
    kernel = min(31, max(5, ((count - end) // 4) * 2 + 1))
    padded = (
        torch.nn.functional.pad(
            means.unsqueeze(0).transpose(1, 2),
            (kernel // 2, kernel // 2),
            mode="replicate",
        )
        .transpose(1, 2)
        .squeeze(0)
    )
    smoothed = torch.empty_like(means)
    for index in range(count):
        smoothed[index] = padded[index : index + kernel].mean(dim=0)
    return end, max(12, int((count - end) * 0.35)), reference, smoothed


def _correct_tail(batch: Any, start: int, plan: Any) -> None:
    if plan is None:
        return
    torch = _torch()
    tail, ramp, reference, smoothed = plan
    for index in range(max(start, tail), start + len(batch)):
        _interrupt()
        delta = smoothed[index] - reference
        luminance = 0.299 * delta[0] + 0.587 * delta[1] + 0.114 * delta[2]
        target = delta - luminance + 0.3 * luminance
        progress = min(1.0, float(index - tail) / float(ramp))
        correction = target * (0.5 * (1.0 - math.cos(3.141592653589793 * progress)))
        if torch.max(torch.abs(correction)) > 0.004:
            batch[index - start] = torch.clamp(
                batch[index - start] - correction.view(1, 1, 3), 0.0, 1.0
            )


class AWGMemoryClarity(_StoreNode):
    _source = "clarity"
    FUNCTION = "enhance"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        schema = super().INPUT_TYPES()
        schema["optional"]["preserve_frame_count"] = ("BOOLEAN", {"default": False})
        return schema

    def enhance(
        self, images: FrameStore, preserve_frame_count: bool = False, **settings: Any
    ) -> tuple[FrameStore]:
        if type(preserve_frame_count) is not bool:
            raise TypeError("preserve_frame_count must be a boolean.")
        torch = _torch()
        crop = (
            0
            if preserve_frame_count
            else (1 if len(images) % 4 == 1 and len(images) > 16 else 0)
        )
        means = _frame_means(images, crop)
        plan = _head_plan(means)
        # Preserve head correction before computing tail statistics. Clipping may
        # change the means: subtracting the planned delta from old means is wrong.
        head_store, skip, corrected_means = images, crop, means
        if plan is not None:
            with _writer(images, "head") as writer:
                corrected_means = torch.empty_like(means)
                for start in range(0, len(images) - crop, images.batch_frames):
                    _interrupt()
                    count = min(images.batch_frames, len(images) - crop - start)
                    batch = torch.from_numpy(images.read(start + crop, count))
                    _correct_head(batch, start, means, plan)
                    corrected_means[start : start + count] = batch.mean(dim=(1, 2))
                    writer.append(_numpy(batch), check=_interrupt)
                head_store = writer.finish()
            images.discard()
            skip = 0
        tail = _tail_plan(corrected_means)
        # The original shader measures brightness AFTER both temporal corrections,
        # over groups of 32 frames. Persist that corrected input once; averaging
        # the old frame means would miss clipping introduced by tail correction.
        with _writer(head_store, "temporal") as writer:
            for start in range(0, len(head_store) - skip, head_store.batch_frames):
                _interrupt()
                batch = torch.from_numpy(
                    head_store.read(
                        start + skip,
                        min(head_store.batch_frames, len(head_store) - skip - start),
                    )
                )
                _correct_tail(batch, start, tail)
                writer.append(_numpy(batch), check=_interrupt)
            temporal = writer.finish()
        head_store.discard()
        node = _binding("clarity")()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lut = node.build_tone_lut_tensor(device)
        with _writer(temporal, "clarity") as writer:
            for group_start in range(0, len(temporal), 32):
                group_end = min(group_start + 32, len(temporal))
                # Reduce the same Y pixels, not RGB frame means. Only scalars
                # survive each bounded read. GPU reduction roundoff is a manual
                # validation boundary; no bit-identical GPU claim is made.
                luminance_sum = 0.0
                for start in range(group_start, group_end, temporal.batch_frames):
                    _interrupt()
                    count = min(temporal.batch_frames, group_end - start)
                    pixels = temporal.read(start, count)
                    y = (
                        0.299 * pixels[..., 0]
                        + 0.587 * pixels[..., 1]
                        + 0.114 * pixels[..., 2]
                    )
                    luminance_sum += float(np.sum(y, dtype=np.float64))
                    del pixels, y
                scene_luma = luminance_sum / (
                    (group_end - group_start) * temporal.shape[0] * temporal.shape[1]
                )
                for start in range(group_start, group_end, temporal.batch_frames):
                    _interrupt()
                    batch = torch.from_numpy(
                        temporal.read(
                            start, min(temporal.batch_frames, group_end - start)
                        )
                    )
                    processed = node._process_chunk(
                        batch.to(device),
                        lut,
                        settings.get("adaptive_mode", True),
                        settings.get("blend_opacity", 0.31),
                        settings.get("sharpen_strength", 1.0),
                        settings.get("clarity_strength", 0.4),
                        scene_luma=scene_luma,
                    )
                    writer.append(
                        _numpy(processed.cpu().to(dtype=batch.dtype)), check=_interrupt
                    )
                    del batch, processed
            result = writer.finish()
        temporal.discard()
        return (result,)


class FrameSequence(Sequence[Any]):
    """VHS-compatible lazy frames; one current persisted chunk, never the whole clip."""

    def __init__(self, source: FrameStore):
        self.source = source
        self._batch: NDArray[np.generic] | None = None
        self._start = 0
        self._closed = False

    def __len__(self) -> int:
        return len(self.source)

    def _frame(self, index: int) -> NDArray[np.generic]:
        _interrupt()
        if self._closed:
            raise FrameStoreError("Frame sequence is closed.")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._batch is None or not self._start <= index < self._start + len(
            self._batch
        ):
            self._batch = None
            offset = 0
            for chunk in self.source.chunks:
                if index < offset + chunk.count:
                    self._batch = self.source.read(offset, chunk.count)
                    self._start = offset
                    break
                offset += chunk.count
        _interrupt()
        assert self._batch is not None
        return cast(NDArray[np.generic], self._batch[index - self._start])

    def __getitem__(self, index: int | slice) -> Any:
        if not isinstance(index, int):
            raise TypeError(
                "Bounded encoder does not support materializing frame slices."
            )
        # VHS retains images[0] for metadata. Detach that one frame from the chunk.
        return _torch().from_numpy(self._frame(index).copy())

    def __iter__(self) -> Iterator[Any]:
        for index in range(len(self)):
            yield _torch().from_numpy(self._frame(index))

    def close(self) -> None:
        self._closed = True
        self._batch = None


class AWGEncodingFailure(RuntimeError):
    """Marker for runtime-owned group termination before attempt cleanup."""


class AWGMemoryEncode(_StoreNode):
    _source = "vhs"
    FUNCTION = "encode"
    OUTPUT_NODE = True
    RETURN_TYPES = ("VHS_FILENAMES",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        schema = super().INPUT_TYPES()
        schema["required"]["format"] = (["video/h264-mp4"],)
        for name in ("images", "latents", "meta_batch", "vae"):
            schema["optional"].pop(name, None)
        schema["required"].pop("latents", None)
        return schema

    def encode(self, images: FrameStore, **settings: Any) -> Any:
        sequence = FrameSequence(images)
        try:
            _interrupt()
            if (
                settings.get("format") != "video/h264-mp4"
                or settings.get("loop_count") != 0
                or settings.get("pingpong") is not False
                or settings.get("save_output") is not True
                or settings.get("filename_prefix") != f"{images.prefix}/raw"
                or any(
                    settings.get(name) is not None
                    for name in ("latents", "meta_batch", "vae")
                )
            ):
                raise FrameStoreError("Unsupported bounded encoder settings.")
            # The pinned format fingerprint excludes pre_pass/gifski materialization.
            result = _binding("vhs")().combine_video(images=sequence, **settings)
        except BaseException as exc:
            raise AWGEncodingFailure(
                "AWG_ENCODING_FAILURE: bounded encoding failed or was interrupted."
            ) from exc
        finally:
            sequence.close()
        images.discard()  # Only normal encoder completion authorizes stage reclamation.
        return result


NODE_CLASS_MAPPINGS = {
    "AWGMemoryGenerate": AWGMemoryGenerate,
    "AWGAnchoredGenerate": AWGAnchoredGenerate,
    "AWGOwnNativeFrames": AWGOwnNativeFrames,
    "AWGLoadNativeAnchor": AWGLoadNativeAnchor,
    "AWGSliceHeadSwapConditioning": AWGSliceHeadSwapConditioning,
    "AWGMemoryChroma": AWGMemoryChroma,
    "AWGMemoryResize": AWGMemoryResize,
    "AWGMemoryClarity": AWGMemoryClarity,
    "AWGMemoryEncode": AWGMemoryEncode,
    "AWGMemoryRelayVSR": AWGMemoryRelayVSR,
}
