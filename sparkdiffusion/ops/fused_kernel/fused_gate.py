# Copyright (c) 2025 Alibaba Group Holding Limited.
# All rights reserved.
#
# SparkDiffusion-specific Triton kernel and integration code.

"""
Gated fusion of low-rank and sparse attention outputs.

Takes o_lr [B, H, L, D] and o_sparse [B, H, L, D], fuses them via:
  - RMSNorm on o_lr → o_lr_normed
  - RMS scale of o_sparse
  - Learned gate (per-head bias, optionally per-token via gate_proj on x)
  - Fusion: (o_sparse / os_rms + gate * o_lr_normed) * os_rms

Output: [B, L, H*D]
"""

import torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════
# Triton kernel: gated fusion
# ═══════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_L': 64}, num_warps=4),
        triton.Config({'BLOCK_L': 128}, num_warps=4),
        triton.Config({'BLOCK_L': 128}, num_warps=8),
        triton.Config({'BLOCK_L': 256}, num_warps=8),
    ],
    key=['L', 'HEAD_DIM', 'NUM_HEADS'],
    cache_results=True,
)
@triton.jit
def _gate_fusion_kernel(
    o_lr_ptr,    # [BH, L, D] bf16
    os_ptr,      # [BH, L, D] bf16
    x_ptr,       # [B, L, dim] bf16 — input x (read directly, no permute)
    gate_proj_w_ptr,  # [D] bf16 (shared across heads)
    o_ptr,       # [B, L, H*D] bf16 — output (BLHD layout, no post-kernel permute)
    gate_bias_ptr,  # [H] input dtype
    L,
    stride_lr_bh, stride_lr_l,
    stride_osbh, stride_osl,
    stride_xb, stride_xl,
    stride_ob, stride_ol,
    BLOCK_L: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    EPS_LR: tl.constexpr,
    EPS_OS: tl.constexpr,
    HAS_GATE_PROJ: tl.constexpr,
):
    pid_l = tl.program_id(0).to(tl.int64)
    pid_bh = tl.program_id(1).to(tl.int64)

    head_idx = pid_bh % NUM_HEADS
    batch_idx = pid_bh // NUM_HEADS

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, HEAD_DIM)
    l_mask = offs_l < L

    o_lr = tl.load(
        o_lr_ptr + pid_bh * stride_lr_bh + offs_l[:, None] * stride_lr_l + offs_d[None, :],
        mask=l_mask[:, None], other=0.0,
    ).to(tl.float32)

    rms_sq = tl.sum(o_lr * o_lr, axis=1) / HEAD_DIM + EPS_LR
    # Eager _lowrank_attn returns a BF16 F.rms_norm result.  Keep that
    # materialization boundary in registers before the FP32 gated fusion.
    o_lr_n = (o_lr * tl.rsqrt(rms_sq)[:, None]).to(
        o_lr_ptr.dtype.element_ty
    ).to(tl.float32)

    os_tile = tl.load(
        os_ptr + pid_bh * stride_osbh + offs_l[:, None] * stride_osl + offs_d[None, :],
        mask=l_mask[:, None], other=0.0,
    ).to(tl.float32)

    # Reproduce eager's BF16 pow -> mean -> add -> sqrt chain.  The later
    # division intentionally promotes this scale to FP32.
    os_sq = (os_tile * os_tile).to(os_ptr.dtype.element_ty)
    os_mean = (tl.sum(os_sq.to(tl.float32), axis=1) / HEAD_DIM).to(
        os_ptr.dtype.element_ty
    )
    os_mean_eps = (os_mean.to(tl.float32) + EPS_OS).to(
        os_ptr.dtype.element_ty
    )
    os_rms = tl.sqrt(os_mean_eps.to(tl.float32)).to(
        os_ptr.dtype.element_ty
    ).to(tl.float32)

    gate_bias = tl.load(gate_bias_ptr + head_idx)
    if HAS_GATE_PROJ:
        x_tile = tl.load(
            x_ptr + batch_idx * stride_xb + offs_l[:, None] * stride_xl
            + head_idx * HEAD_DIM + offs_d[None, :],
            mask=l_mask[:, None], other=0.0,
        )
        gate_proj_w = tl.load(gate_proj_w_ptr + offs_d)
        gate_linear = tl.sum(
            x_tile.to(tl.float32) * gate_proj_w[None, :].to(tl.float32),
            axis=1,
        ).to(x_ptr.dtype.element_ty)
        gate_logit = (
            gate_linear.to(tl.float32) + gate_bias.to(tl.float32)
        ).to(x_ptr.dtype.element_ty)
        gate = (1.0 / (1.0 + tl.exp(-gate_logit.to(tl.float32)))).to(
            x_ptr.dtype.element_ty
        ).to(tl.float32)
    else:
        gate = (1.0 / (1.0 + tl.exp(-gate_bias.to(tl.float32)))).to(
            gate_bias_ptr.dtype.element_ty
        ).to(tl.float32)

    fused = os_tile / os_rms[:, None] + gate[:, None] * o_lr_n
    out = fused * os_rms[:, None]

    # Store directly to [B, L, H*D] — no post-kernel permute needed
    out_ptrs = (o_ptr + batch_idx * stride_ob
                + offs_l[:, None] * stride_ol
                + head_idx * HEAD_DIM + offs_d[None, :])
    tl.store(out_ptrs, out.to(o_ptr.type.element_ty), mask=l_mask[:, None])


# ═══════════════════════════════════════════════════════════════
# PyTorch reference
# ═══════════════════════════════════════════════════════════════

def fused_gate_ref(o_lr, o_sparse, x=None, gate_proj_weight=None,
                   gate_bias=None):
    """
    PyTorch reference for gated fusion.

    Args:
      o_lr: [B, H, L, D] float — low-rank attention output
      o_sparse: [B, H, L, D] float — sparse attention output
      x: [B, L, dim] — raw input (for gate_proj, optional)
      gate_proj_weight: [1, D] — per-token gate projection weight (optional)
      gate_bias: [H] input dtype — per-head gate bias

    Returns: [B, L, H*D]
    """
    B, H, L, D = o_lr.shape

    # This is folded into the fused gate kernel, but eager materializes BF16.
    o_lr_n = torch.nn.functional.rms_norm(o_lr, [D])

    # RMS scale of o_sparse
    os_rms = torch.sqrt(o_sparse.pow(2).mean(-1, keepdim=True) + 1e-5)

    # Gate
    gate_bias_flat = gate_bias.view(H)
    if gate_proj_weight is not None and x is not None:
        x_bhld = x.view(B, L, H, D).transpose(1, 2)
        gate_logit = torch.nn.functional.linear(x_bhld, gate_proj_weight)
        gate = torch.sigmoid(
            gate_logit + gate_bias_flat.view(1, H, 1, 1)
        )
    else:
        gate = torch.sigmoid(gate_bias_flat).view(1, H, 1, 1)

    fused = o_sparse.float() / os_rms.float() + gate * o_lr_n.float()
    out = fused * os_rms.float()

    return out.permute(0, 2, 1, 3).reshape(B, L, H * D).to(o_lr.dtype)


# ═══════════════════════════════════════════════════════════════
# Python wrapper
# ═══════════════════════════════════════════════════════════════

def fused_gate(o_lr, o_sparse, x=None, gate_proj_weight=None,
               gate_bias=None):
    """
    Gated fusion of low-rank and sparse attention outputs.

    Args:
      o_lr: [B, H, L, D] bf16 — low-rank attention output
      o_sparse: [B, H, L, D] bf16 — sparse attention output
      x: [B, L, dim] bf16 — raw input (for gate_proj, optional)
      gate_proj_weight: [1, D] — per-token gate projection weight (optional)
      gate_bias: [H] input dtype — per-head gate bias

    Returns: [B, L, H*D] bf16
    """
    B, H, L, D = o_lr.shape
    BH = B * H
    dim = H * D

    o_lr_2d = o_lr.reshape(BH, L, D)
    os_2d = o_sparse.reshape(BH, L, D)
    out = torch.empty(B, L, dim, dtype=o_lr.dtype, device=o_lr.device)

    gate_bias_flat = gate_bias.view(H).to(o_lr.dtype)

    has_gate_proj = gate_proj_weight is not None and x is not None
    if has_gate_proj:
        gate_proj_w_flat = gate_proj_weight.reshape(-1).to(o_lr.dtype)
    else:
        gate_proj_w_flat = o_lr_2d.flatten()[:D]

    def grid(meta):
        return (triton.cdiv(L, meta["BLOCK_L"]), BH)

    _gate_fusion_kernel[grid](
        o_lr_2d, os_2d, x, gate_proj_w_flat, out, gate_bias_flat,
        L,
        o_lr_2d.stride(0), o_lr_2d.stride(1),
        os_2d.stride(0), os_2d.stride(1),
        x.stride(0) if x is not None else 0, x.stride(1) if x is not None else 0,
        out.stride(0), out.stride(1),
        HEAD_DIM=D, NUM_HEADS=H,
        # The eager path uses F.rms_norm without an explicit epsilon, whose
        # contract is torch.finfo(input.dtype).eps.
        EPS_LR=torch.finfo(o_lr.dtype).eps, EPS_OS=1e-5,
        HAS_GATE_PROJ=has_gate_proj,
    )

    return out
