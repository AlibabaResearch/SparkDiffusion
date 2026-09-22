# Copyright (c) 2025 Alibaba Group Holding Limited.
# All rights reserved.
#
# SparkDiffusion-specific Triton kernel and integration code.

"""
Fused low-rank linear attention.

Produces o_lr [B, H, L, D] — the linear attention approximation via:
  Phase 1 (Triton kernel _proj_silu_rope_kernel):
    - Optional full-dim RMSNorm (over H*D, matching WanRMSNorm)
    - Proj GEMM: q @ proj_w^T → [BH, L, R]
    - SiLU activation
    - Truncated interleaved RoPE on rank dimension
  C computation (cuBLAS):
    C = k_lr^T @ v → [B, H, R, D]
  Output (cuBLAS):
    o_lr = q_lr @ C → [B, H, L, D]
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════
# Phase 1: Fused proj + SiLU + optional RMSNorm + RoPE kernel
# ═══════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_L': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_L': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_L': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_L': 256}, num_warps=8, num_stages=3),
    ],
    key=['L', 'RANK', 'HEAD_DIM'],
    cache_results=True,
)
@triton.jit
def _proj_silu_rope_kernel(
    q_ptr,          # [B, L, H, D] bf16 — input (normed or raw)
    proj_w_ptr,      # [R, D] bf16 — projection weight (shared across heads)
    cos_ptr,         # [L, rope_dim] input dtype — precomputed cos values
    sin_ptr,         # [L, rope_dim] input dtype — precomputed sin values
    norm_w_ptr,      # [H*D] bf16 — RMSNorm weight (used when HAS_NORM)
    rstd_ptr,        # [B, L] fp32 — pre-computed full-dim rstd (used when HAS_NORM)
    out_ptr,         # [BH, L, R] bf16 — output
    L,
    stride_qb, stride_ql,   # q strides for [B, L, H, D]
    stride_obh, stride_ol,  # out strides for [BH, L, R]
    BLOCK_L: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RANK: tl.constexpr,
    ROPE_DIM: tl.constexpr,          # padded to power-of-2 for tl.arange
    ROPE_DIM_ACTUAL: tl.constexpr,   # actual rope_dim (may be < ROPE_DIM)
    NUM_HEADS: tl.constexpr,
    EPS: tl.constexpr,
    HAS_NORM: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    pid_l = tl.program_id(0).to(tl.int64)
    pid_bh = tl.program_id(1).to(tl.int64)

    head_idx = pid_bh % NUM_HEADS
    batch_idx = pid_bh // NUM_HEADS

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_r = tl.arange(0, RANK)
    l_mask = offs_l < L

    # --- Load q tile [BLOCK_L, D] from [B, L, H, D] ---
    q = tl.load(
        q_ptr + batch_idx * stride_qb + offs_l[:, None] * stride_ql
        + head_idx * HEAD_DIM + offs_d[None, :],
        mask=l_mask[:, None], other=0.0,
    )

    # --- Optional full-dim RMSNorm (rstd pre-computed over H*D in wrapper) ---
    if HAS_NORM:
        q_f32 = q.to(tl.float32)
        norm_w = tl.load(
            norm_w_ptr + head_idx * HEAD_DIM + offs_d
        )
        rstd = tl.load(rstd_ptr + batch_idx * L + offs_l, mask=l_mask, other=0.0)
        q_unit = (q_f32 * rstd[:, None]).to(q_ptr.dtype.element_ty)
        q = (q_unit * norm_w[None, :]).to(q_ptr.dtype.element_ty)

    # --- Load proj_w as [D, R] (transposed view of [R, D]) ---
    proj_w = tl.load(
        proj_w_ptr + offs_d[:, None] + offs_r[None, :] * HEAD_DIM
    )

    # --- GEMM: [BLOCK_L, D] @ [D, R] → [BLOCK_L, R] (fp32 accumulate) ---
    linear = tl.dot(q, proj_w).to(out_ptr.dtype.element_ty)

    # PyTorch's BF16 Linear materializes before SiLU, and the BF16 SiLU kernel
    # computes in opmath FP32 before casting its output back to BF16.
    linear_f32 = linear.to(tl.float32)
    o = (linear_f32 * tl.sigmoid(linear_f32)).to(out_ptr.dtype.element_ty)

    # --- Store + RoPE ---
    if HAS_ROPE:
        # Load cos/sin for this token block, padded to RANK//2 with identity (cos=1, sin=0)
        half_offs = tl.arange(0, RANK // 2)
        half_mask = half_offs < ROPE_DIM_ACTUAL

        cos = tl.load(
            cos_ptr + offs_l[:, None] * ROPE_DIM + half_offs[None, :],
            mask=l_mask[:, None] & half_mask[None, :], other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            sin_ptr + offs_l[:, None] * ROPE_DIM + half_offs[None, :],
            mask=l_mask[:, None] & half_mask[None, :], other=0.0,
        ).to(tl.float32)

        # Split o into even/odd pairs IN REGISTERS (no GMEM round-trip).
        o_pairs = tl.reshape(o, (BLOCK_L, RANK // 2, 2))
        o_even, o_odd = tl.split(o_pairs)

        # Eager evaluates two BF16 multiplies and one BF16 add/subtract for each
        # rotated component.  Explicit casts retain those rounding points while
        # keeping the intermediates in registers.
        even_cos = (o_even.to(tl.float32) * cos).to(out_ptr.dtype.element_ty)
        odd_sin = (o_odd.to(tl.float32) * sin).to(out_ptr.dtype.element_ty)
        even_sin = (o_even.to(tl.float32) * sin).to(out_ptr.dtype.element_ty)
        odd_cos = (o_odd.to(tl.float32) * cos).to(out_ptr.dtype.element_ty)
        rot_even = (
            even_cos.to(tl.float32) - odd_sin.to(tl.float32)
        ).to(out_ptr.dtype.element_ty)
        rot_odd = (
            even_sin.to(tl.float32) + odd_cos.to(tl.float32)
        ).to(out_ptr.dtype.element_ty)

        # Join back and store once (no reload!)
        o_out = tl.reshape(
            tl.join(rot_even, rot_odd),
            (BLOCK_L, RANK),
        )
        tl.store(
            out_ptr + pid_bh * stride_obh + offs_l[:, None] * stride_ol + offs_r[None, :],
            o_out,
            mask=l_mask[:, None],
        )
    else:
        tl.store(
            out_ptr + pid_bh * stride_obh + offs_l[:, None] * stride_ol + offs_r[None, :],
            o,
            mask=l_mask[:, None],
        )


# ═══════════════════════════════════════════════════════════════
# PyTorch reference implementations
# ═══════════════════════════════════════════════════════════════

def _apply_rope_rank(x, freqs, rank):
    """
    Efficient interleaved RoPE on rank dimension.
    x: [B, H, L, R], freqs: [L, head_dim//2]
    Returns: [B, H, L, R] with interleaved rotation.
    """
    rope_dim = min(rank // 2, freqs.shape[-1])
    L = x.shape[2]
    freqs_typed = freqs[:L, :rope_dim].to(x.dtype)
    cos = torch.cos(freqs_typed)
    sin = torch.sin(freqs_typed)
    ROT = 2 * rope_dim

    x_rot = x[..., :ROT]
    x_pairs = x_rot.unflatten(-1, (-1, 2))
    x0, x1 = x_pairs.unbind(-1)

    cos_b = cos[None, None]
    sin_b = sin[None, None]

    out0 = x0 * cos_b - x1 * sin_b
    out1 = x0 * sin_b + x1 * cos_b
    out = torch.stack([out0, out1], dim=-1).flatten(-2)

    if ROT < rank:
        out = torch.cat([out, x[..., ROT:]], dim=-1)
    return out


def _proj_silu_rope_ref(q, proj_weight, norm_weight, freqs, rank, eps=1e-6):
    """
    PyTorch reference for _proj_silu_rope_kernel.
    q: [B, L, H, D], proj_weight: [R, D], norm_weight: [H*D] or None
    Returns: [BH, L, R]
    """
    B, L, H, D = q.shape
    BH = B * H

    q_bhld = q.permute(0, 2, 1, 3)  # [B, H, L, D]

    if norm_weight is not None:
        # Full-dim RMSNorm over H*D (matching WanRMSNorm)
        rstd = torch.rsqrt(
            q.float().reshape(B, L, -1).pow(2).mean(-1) + eps
        )  # [B, L]
        nw = norm_weight.view(1, H, 1, D).to(q.dtype)
        q_bhld = (q_bhld.float() * rstd[:, None, :, None]).to(q.dtype)
        q_bhld = q_bhld * nw

    # Proj + SiLU
    q_lr = F.silu(q_bhld @ proj_weight.T)  # [B, H, L, R]

    # RoPE on rank
    if freqs is not None:
        q_lr = _apply_rope_rank(q_lr, freqs, rank)

    return q_lr.reshape(BH, L, rank).contiguous()


def fused_lowrank_ref(q, k, v, freqs=None,
                      norm_q_weight=None, norm_k_weight=None,
                      proj_q_weight=None, proj_k_weight=None,
                      rola_rank=64, eps=1e-6):
    """
    PyTorch reference for fused_lowrank.
    v uses the fused-QKV layout [B*H, L, D].
    Returns: o_lr [B, H, L, D]
    """
    B, L, H, _D = q.shape
    if v.shape != (B * H, L, _D):
        raise ValueError(
            f"v must have shape {(B * H, L, _D)}, got {tuple(v.shape)}"
        )

    q_lr = _proj_silu_rope_ref(q, proj_q_weight, norm_q_weight, freqs, rola_rank, eps)
    k_lr = _proj_silu_rope_ref(k, proj_k_weight, norm_k_weight, freqs, rola_rank, eps)

    # [BH, L, R] → [B, H, L, R]
    q_lr_4d = q_lr.view(B, H, L, rola_rank)
    k_lr_4d = k_lr.view(B, H, L, rola_rank)
    v_4d = v.view(B, H, L, _D)

    # C = k_lr^T @ v → [B, H, R, D]
    C = torch.matmul(k_lr_4d.transpose(-2, -1), v_4d)

    # o_lr = q_lr @ C → [B, H, L, D]
    o_lr = torch.matmul(q_lr_4d, C)

    return o_lr


# ═══════════════════════════════════════════════════════════════
# Python wrapper
# ═══════════════════════════════════════════════════════════════

def prepare_lowrank_cos_sin(freqs, L, rank, dtype):
    """Build the dtype-matched RoPE tables shared by all RoLa blocks."""
    rope_dim = min(rank // 2, freqs.shape[-1])
    rope_dim_pow2 = triton.next_power_of_2(rope_dim) if rope_dim > 0 else 1

    # Match the eager low-rank branch: quantize the raw angles to the input
    # dtype before evaluating cos/sin.
    freqs_typed = freqs[:L, :rope_dim].to(dtype)
    cos = torch.cos(freqs_typed).contiguous()
    sin = torch.sin(freqs_typed).contiguous()
    if rope_dim_pow2 > rope_dim:
        cos = F.pad(cos, (0, rope_dim_pow2 - rope_dim))
        sin = F.pad(sin, (0, rope_dim_pow2 - rope_dim))
    return cos, sin, rope_dim, rope_dim_pow2


def fused_lowrank(q, k, v, freqs=None,
                  norm_q_weight=None, norm_k_weight=None,
                  proj_q_weight=None, proj_k_weight=None,
                  rola_rank=64, eps=1e-6, rope_tables=None):
    """
    Fused low-rank linear attention.

    Phase 1 (Triton kernel): proj + SiLU + optional RMSNorm + RoPE → [BH, L, R]
    C computation (cuBLAS): C = k_lr^T @ v → [B, H, R, D]
    Output (cuBLAS): o_lr = q_lr @ C → [B, H, L, D]

    Args:
      q, k: [B, L, H, D] bf16 — normed (or raw if norm weights provided)
      v: [B*H, L, D] bf16 — layout-transformed values from fused QKV
      freqs: [L, head_dim//2] RoPE frequencies (optional)
      norm_q_weight/norm_k_weight: [H*D] if not None, apply full-dim RMSNorm
      proj_q_weight/proj_k_weight: [R, D] projection weights
      rola_rank: R (low-rank dimension)
      eps: RMSNorm epsilon
      rope_tables: optional output of ``prepare_lowrank_cos_sin`` shared by layers

    Returns: o_lr [B, H, L, D] bf16
    """
    B, L, H, D = q.shape
    RANK = rola_rank
    BH = B * H
    if v.ndim != 3 or v.shape != (BH, L, D):
        raise ValueError(
            f"v must have shape {(BH, L, D)} from fused QKV, got {tuple(v.shape)}"
        )

    has_norm = norm_q_weight is not None
    has_rope = rope_tables is not None or freqs is not None

    # Pre-compute full-dim RMSNorm rstd (over H*D, matching WanRMSNorm)
    if has_norm:
        q_flat = q.float().reshape(B * L, -1)
        rstd_q = torch.rsqrt(q_flat.pow(2).mean(-1) + eps).reshape(B, L)
        k_flat = k.float().reshape(B * L, -1)
        rstd_k = torch.rsqrt(k_flat.pow(2).mean(-1) + eps).reshape(B, L)
    else:
        rstd_q = q.new_empty(1)
        rstd_k = k.new_empty(1)

    if has_rope:
        if rope_tables is None:
            rope_tables = prepare_lowrank_cos_sin(freqs, L, RANK, q.dtype)
        cos, sin, rope_dim, rope_dim_pow2 = rope_tables
    else:
        rope_dim = 0
        rope_dim_pow2 = 1
        cos = q  # dummy (never dereferenced when HAS_ROPE=False)
        sin = q

    # Ensure weights are contiguous and match input dtype (kernel uses tl.dot)
    proj_q_w = proj_q_weight.to(q.dtype).contiguous()
    proj_k_w = proj_k_weight.to(q.dtype).contiguous()
    nq_w = norm_q_weight.contiguous() if has_norm else q
    nk_w = norm_k_weight.contiguous() if has_norm else k

    q_lr = torch.empty(BH, L, RANK, dtype=q.dtype, device=q.device)
    k_lr = torch.empty(BH, L, RANK, dtype=q.dtype, device=q.device)

    # Autotune: BLOCK_L and num_warps chosen by autotuner
    def grid(meta):
        return (triton.cdiv(L, meta["BLOCK_L"]), BH)

    def _launch(inp, proj_w, norm_w, rstd, out_buf):
        _proj_silu_rope_kernel[grid](
            inp, proj_w, cos, sin, norm_w,
            rstd,
            out_buf,
            L,
            inp.stride(0), inp.stride(1),
            out_buf.stride(0), out_buf.stride(1),
            HEAD_DIM=D, RANK=RANK,
            ROPE_DIM=rope_dim_pow2,
            ROPE_DIM_ACTUAL=rope_dim,
            NUM_HEADS=H, EPS=eps,
            HAS_NORM=has_norm, HAS_ROPE=has_rope,
        )

    _launch(q, proj_q_w, nq_w, rstd_q, q_lr)
    _launch(k, proj_k_w, nk_w, rstd_k, k_lr)

    # --- C = k_lr^T @ v → [B, H, R, D] (cuBLAS batched) ---
    k_lr_4d = k_lr.view(B, H, L, RANK)
    v_4d = v.view(B, H, L, D)
    C = torch.matmul(k_lr_4d.transpose(-2, -1), v_4d)  # [B, H, R, D]

    # --- o_lr = q_lr @ C → [B, H, L, D] (cuBLAS batched) ---
    q_lr_4d = q_lr.view(B, H, L, RANK)
    o_lr = torch.matmul(q_lr_4d, C)  # [B, H, L, D]

    return o_lr
