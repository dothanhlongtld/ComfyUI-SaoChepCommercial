"""Namespaced head-swap workflow nodes adapted from the supplied reference bundle."""
from .nodes import NODE_CLASS_MAPPINGS

NODE_DISPLAY_NAME_MAPPINGS = {
    name: name.replace("SaoChep", "SaoChep ").replace("HeadSwap", "HeadSwap • ")
    for name in NODE_CLASS_MAPPINGS
}
