"""CPU-only fixtures; production ComfyUI package is not imported."""

pytest_plugins = ["bounded_test_support", "bounded_tensor_support"]
