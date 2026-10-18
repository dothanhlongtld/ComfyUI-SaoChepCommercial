"""Source-aware native and delivery dimensions for prepared video workflows."""

from typing import Any


def native_dimensions(width: int, height: int, budget_pixels: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("source image must have non-empty dimensions")
    if budget_pixels <= 0:
        raise ValueError("native pixel budget must be positive")
    scale = (budget_pixels / (width * height)) ** 0.5
    scale = min(scale, 1280 / width, 1280 / height)
    # A 256-pixel minimum canvas accommodates sources beyond the native aspect limits.
    w = max(256, min(1280, round(width * scale / 32) * 32))
    h = max(256, min(1280, round(height * scale / 32) * 32))
    return w, h


def delivery_dimensions(width: int, height: int, short_side: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("source image must have non-empty dimensions")
    if short_side <= 0 or short_side % 2:
        raise ValueError("delivery short side must be a positive even integer")
    if width >= height:
        return max(short_side, round((short_side * width / height) / 2) * 2), short_side
    return short_side, max(short_side, round((short_side * height / width) / 2) * 2)


class SaoChepSourceAspectSizing:
    CATEGORY = "SaoChep/Commercial"
    FUNCTION = "size"
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "native_width",
        "native_height",
        "delivery_width",
        "delivery_height",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "source": ("IMAGE",),
                "native_pixel_budget": ("INT", {"default": 589824, "min": 1}),
                "delivery_short_side": ("INT", {"default": 720, "min": 2, "step": 2}),
            }
        }

    def size(
        self, source: Any, native_pixel_budget: int, delivery_short_side: int
    ) -> tuple[int, int, int, int]:
        shape: tuple[int, ...] = getattr(source, "shape", ())
        if len(shape) != 4 or min(shape) <= 0:
            raise ValueError("source must be a non-empty IMAGE batch [B,H,W,C]")
        height, width = shape[1:3]
        return (
            *native_dimensions(width, height, native_pixel_budget),
            *delivery_dimensions(width, height, delivery_short_side),
        )
