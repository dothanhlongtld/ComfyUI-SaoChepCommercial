"""Namespaced head-swap workflow nodes adapted from the supplied reference bundle."""
from .nodes import NODE_CLASS_MAPPINGS
from .export_source_node import SaoChepHeadSwapExportSource

NODE_CLASS_MAPPINGS["SaoChepHeadSwapExportSource"] = SaoChepHeadSwapExportSource

NODE_DISPLAY_NAME_MAPPINGS = {
    name: name.replace("SaoChep", "SaoChep ").replace("HeadSwap", "HeadSwap • ")
    for name in NODE_CLASS_MAPPINGS
}
