"""Small CUDA architecture helpers shared by inference optimizations."""

from __future__ import annotations

import functools

import torch


# FP8 E4M3 tensor cores landed with Ada (SM89) and exist on every later family,
# so FP8 support is a capability question, not a Blackwell question.
_FP8_MIN_CC = (8, 9)

# CUDA currently assigns Blackwell to the SM100, SM110, and SM120 families.
# Keep this explicit instead of accepting every future ``major >= 10`` device.
# These two predicates only select Blackwell-specific *tuning* (accumulator
# split/join, warp-count narrowing); they must not gate FP8 availability.
_BLACKWELL_CC_MAJORS = frozenset({10, 11, 12})


def get_compute_capability(device: torch.device | None = None) -> tuple[int, int] | None:
    """Return ``(major, minor)`` for a CUDA device, or ``None`` off CUDA."""
    if not torch.cuda.is_available() or (device is not None and device.type != "cuda"):
        return None
    return tuple(torch.cuda.get_device_capability(device))


def is_blackwell(device: torch.device | None = None) -> bool:
    """Whether *device* belongs to a currently known Blackwell SM family."""
    capability = get_compute_capability(device)
    return capability is not None and capability[0] in _BLACKWELL_CC_MAJORS


def is_rtx_blackwell(device: torch.device | None = None) -> bool:
    """Whether *device* is an SM120-family RTX/Workstation Blackwell GPU."""
    capability = get_compute_capability(device)
    return capability is not None and capability[0] == 12


def supports_fp8(device: torch.device | None = None) -> bool:
    """Whether *device* has FP8 E4M3 tensor cores (Ada SM89 and later)."""
    capability = get_compute_capability(device)
    return capability is not None and capability >= _FP8_MIN_CC


@functools.lru_cache(maxsize=None)
def _probe_fp8_rowwise_scaled_mm(device_index: int) -> bool:
    device = torch.device("cuda", device_index)
    x = torch.zeros((16, 64), device=device, dtype=torch.float8_e4m3fn)
    w = torch.zeros((64, 64), device=device, dtype=torch.float8_e4m3fn).t()
    scale_a = torch.ones((16, 1), device=device, dtype=torch.float32)
    scale_b = torch.ones((1, 64), device=device, dtype=torch.float32)
    try:
        torch._scaled_mm(x, w, scale_a=scale_a, scale_b=scale_b,
                         out_dtype=torch.bfloat16, use_fast_accum=True)
    except Exception:
        # An unsupported recipe surfaces as RuntimeError, NotImplementedError or
        # a bare assertion depending on the PyTorch build; all mean "unusable".
        return False
    return True


def supports_fp8_rowwise_scaled_mm(device: torch.device | None = None) -> bool:
    """Whether ``torch._scaled_mm`` accepts the row-wise FP8 recipe here.

    Row-wise scaling dispatches to different backends per architecture and
    PyTorch build, and an unsupported combination only reports itself once a
    GEMM actually runs. Probing a tiny GEMM up front turns that into a clear
    failure at model-conversion time.
    """
    if not supports_fp8(device):
        return False
    index = torch.cuda.current_device() if device is None or device.index is None else device.index
    return _probe_fp8_rowwise_scaled_mm(index)
