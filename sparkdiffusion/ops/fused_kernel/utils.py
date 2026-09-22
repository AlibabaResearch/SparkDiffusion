# Copyright (c) 2025 Alibaba Group Holding Limited.
# All rights reserved.
#
# SparkDiffusion-specific Triton kernel and integration code.

"""Shared utilities for SparkDiffusion RoLa fused kernels."""

import torch
import triton
import triton.language as tl


def reshape_blhd_to_bhl_d(t: torch.Tensor) -> torch.Tensor:
    """Reshape [B, L, H, D] → [B*H, L, D] (contiguous). Works on any GPU."""
    B, L, H, D = t.shape
    return t.permute(0, 2, 1, 3).reshape(B * H, L, D).contiguous()


@triton.jit
def compress_kernel(
    X, XM, CENTER,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
    SUBTRACT_CENTER: tl.constexpr,
):
    idx_l = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)

    x_offset = idx_bh * L * D
    xm_offset = idx_bh * ((L + BLOCK_L - 1) // BLOCK_L) * D
    x = tl.load(
        X + x_offset + offs_l[:, None] * D + offs_d[None, :],
        mask=offs_l[:, None] < L,
        other=0.0,
    )
    if SUBTRACT_CENTER:
        center = tl.load(CENTER + idx_bh * D + offs_d)
        # Reproduce the eager smooth-K path's BF16 materialization before the
        # FP32 block reduction, without writing the full centered K tensor.
        # Re-mask the tail after subtraction: masked loads contain zero, which
        # would otherwise turn into -center and corrupt the final partial block.
        x = tl.where(
            offs_l[:, None] < L,
            x - center[None, :],
            0.0,
        ).to(XM.dtype.element_ty)

    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    x_mean = tl.sum(x, axis=0, dtype=tl.float32) / nx
    tl.store(XM + xm_offset + idx_l * D + offs_d, x_mean.to(XM.dtype.element_ty))


def mean_pool(x, BLK, *, center=None):
    """Pool sequence blocks, optionally subtracting a per-head center in-kernel."""
    assert x.is_contiguous()

    B, H, L, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    x_mean = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=x.dtype)

    subtract_center = center is not None
    if subtract_center:
        if center.shape != (B, H, D):
            raise ValueError(
                f"center must have shape {(B, H, D)}, got {tuple(center.shape)}"
            )
        if center.device != x.device or center.dtype != x.dtype:
            raise ValueError("center must have the same device and dtype as x")
        center = center.contiguous()
    else:
        center = x  # dummy pointer; not dereferenced when SUBTRACT_CENTER=False

    grid = (L_BLOCKS, B * H)
    compress_kernel[grid](
        x, x_mean, center, L, D, BLK,
        SUBTRACT_CENTER=subtract_center,
    )
    return x_mean
