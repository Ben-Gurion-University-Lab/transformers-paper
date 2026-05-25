"""Shared Torch helpers."""

import torch


def get_default_device() -> str:
    """Return the best available Torch device name."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
