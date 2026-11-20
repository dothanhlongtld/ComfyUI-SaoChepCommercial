"""Safe bridge from an owned, already-encoded MP4 to VHS_FILENAMES."""
from pathlib import Path
import re


_MAX_VIDEO_LENGTH = 2048
_MAX_PREFIX_LENGTH = 512
_SEGMENT = re.compile(r"^[A-Za-z0-9._ -]+$")


def _relative_parts(value, *, field, limit):
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ValueError(f"Invalid {field}")
    if "\x00" in value or "\\" in value or value.startswith("/") or ":" in value:
        raise ValueError(f"Invalid {field}")
    parts = value.split("/")
    if any(part in ("", ".", "..") or not _SEGMENT.fullmatch(part) for part in parts):
        raise ValueError(f"Invalid {field}")
    return tuple(parts)


def _reject_symlink_components(root, parts):
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Invalid export source")


def resolve_attempt_mp4(video, attempt_prefix, output_root, *, require_output_relative=False):
    """Resolve an MP4 only after proving it belongs to the exact attempt subtree."""
    if not isinstance(attempt_prefix, str) or not attempt_prefix:
        raise ValueError("Invalid export source")
    prefix_parts = _relative_parts(attempt_prefix, field="attempt prefix", limit=_MAX_PREFIX_LENGTH)
    if not isinstance(video, str) or not video or len(video) > _MAX_VIDEO_LENGTH:
        raise ValueError("Invalid export source")
    if "\x00" in video or "\\" in video or ":" in video or ".." in video.split("/"):
        raise ValueError("Invalid export source")

    root = Path(output_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Invalid export source")
    _reject_symlink_components(root, prefix_parts)
    attempt_root = root.joinpath(*prefix_parts).resolve(strict=True)
    if not attempt_root.is_dir():
        raise ValueError("Invalid export source")

    candidate = Path(video)
    if require_output_relative and candidate.is_absolute():
        raise ValueError("Invalid export source")
    if candidate.is_absolute():
        try:
            relative_parts = candidate.relative_to(root).parts
        except ValueError:
            raise ValueError("Invalid export source") from None
    else:
        relative_parts = _relative_parts(video, field="video", limit=_MAX_VIDEO_LENGTH)
    if not relative_parts or relative_parts[:len(prefix_parts)] != prefix_parts:
        raise ValueError("Invalid export source")
    if Path(relative_parts[-1]).suffix.lower() != ".mp4":
        raise ValueError("Invalid export source")
    _reject_symlink_components(root, relative_parts)
    target = root.joinpath(*relative_parts).resolve(strict=True)
    try:
        target.relative_to(root)
        target.relative_to(attempt_root)
    except ValueError:
        raise ValueError("Invalid export source") from None
    if not target.is_file():
        raise ValueError("Invalid export source")
    return target


class SaoChepHeadSwapExportSource:
    """Return an existing attempt-owned MP4 using the VHS_FILENAMES contract."""

    CATEGORY = "SaoChep/HeadSwap"
    FUNCTION = "resolve"
    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("filenames",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("STRING", {"forceInput": True}),
            "attempt_prefix": ("STRING", {"forceInput": True}),
        }}

    @staticmethod
    def IS_CHANGED(**_kwargs):
        return float("nan")

    def resolve(self, video, attempt_prefix):
        try:
            import folder_paths
            root = folder_paths.get_output_directory()
            target = resolve_attempt_mp4(video, attempt_prefix, root, require_output_relative=True)
            if "-audio" not in target.stem.lower():
                audio_sibling = target.with_name(f"{target.stem}-audio{target.suffix}")
                if audio_sibling.exists():
                    target = resolve_attempt_mp4(str(audio_sibling), attempt_prefix, root)
            return ((True, [str(target)]),)
        except Exception:
            # Keep untrusted paths and filesystem details out of UI/log messages.
            raise ValueError("Unable to resolve an MP4 in the requested attempt") from None
