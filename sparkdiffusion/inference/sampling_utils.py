"""Shared CLI validation and output naming for sequential inference."""

import argparse
from pathlib import Path


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("num_samples must be at least 1")
    return value


def sample_output_path(save_path, sample_idx, num_samples, seed):
    """Preserve a single video's name; suffix multiple videos with index and seed."""
    path = Path(save_path)
    if path.suffix.lower() != ".mp4":
        path = path / "sample.mp4"
    if num_samples > 1:
        path = path.with_name(f"{path.stem}_sample_{sample_idx:02d}_seed_{seed}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)
