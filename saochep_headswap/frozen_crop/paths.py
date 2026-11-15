"""Resolve provisioned assets without runtime downloads or Colab-specific paths."""
from pathlib import Path


def model_roots():
    # One canonical, package-relative root keeps crop, segmentation and matting
    # model/config lookup coherent regardless of the custom-node clone folder.
    return [Path(__file__).parent / "models"]


def asset_path(name):
    for root in model_roots():
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Required provisioned head-swap asset missing: {name}; expected under {model_roots()[0]}")


def model_directory():
    return asset_path("multi.tflite").parent
