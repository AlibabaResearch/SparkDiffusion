# Copyright (c) 2025 Alibaba Group Holding Limited.
# All rights reserved.
#
# SparkDiffusion-specific Triton kernel and integration code.

"""
Fused QKV Epilogue: RMSNorm + RoPE + layout transform.

Uses 3 separate cuBLAS GEMMs (matching the reference forward path) followed by
a single Triton epilogue kernel that fuses:
  - Full hidden-width RMSNorm (Q/K only)
  - Interleaved RoPE (Q/K only)
  - [B, L, H, D] → [B*H, L, D] layout transform (all)

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │ Step 1: 3 × cuBLAS GEMM (identical to forward path)        │
  │   q_raw = F.linear(x, Wq, bq)   [M, dim]                  │
  │   k_raw = F.linear(x, Wk, bk)   [M, dim]                  │
  │   v_raw = F.linear(x, Wv, bv)   [M, dim]                  │
  └──────────────────────────┬──────────────────────────────────┘
                             │
  ┌──────────────────────────▼──────────────────────────────────┐
  │ Step 2: Epilogue kernel (1 Triton launch)                   │
  │   Q/K: RMSNorm(GEMM output) * w_norm, then RoPE            │
  │   V:   pass-through (already includes bias from GEMM)       │
  │   All: [B, L, H, D] → [B*H, L, D] layout transform        │
  └─────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def prepare_cos_sin(freqs: torch.Tensor, L: int, D: int) -> torch.Tensor:
    """
    Prepare interleaved cos_sin tensor for GPT-J style RoPE.

    Returns:
        cos_sin: [L, D] interleaved [cos0, sin0, cos1, sin1, ...]
    """
    freqs = freqs[:L, : D // 2]
    # Match rope_apply: evaluate trig in the angle tensor's dtype, then promote
    # the tables to FP32 for the FP32 rotary arithmetic.
    cos = torch.cos(freqs).to(torch.float32)
    sin = torch.sin(freqs).to(torch.float32)
    cos_sin = torch.stack([cos, sin], dim=-1).reshape(L, D)
    return cos_sin.contiguous()


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['HEAD_DIM', 'NUM_HEADS'],
    cache_results=True,
)
@triton.jit
def _epilogue_kernel(
    # ── GEMM outputs: each [M, dim] where M = B*L, dim = H*D ──
    q_raw_ptr,
    k_raw_ptr,
    v_raw_ptr,
    # ── cos_sin: [L, D] interleaved [cos0, sin0, cos1, sin1, ...] ──
    cos_sin_ptr,
    # ── per-head norm weights: [dim] each (viewed as [H, D]) ──
    wq_ptr,
    wk_ptr,
    # ── outputs: [B*H, L, D] × 3 ──
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    # ── normed outputs: [B, L, H, D] × 2 ──
    q_normed_ptr,
    k_normed_ptr,
    # ── dims ──
    L,
    H,
    # ── strides for GEMM output [M, dim] ──
    stride_raw_m,   # = H * D = dim
    # ── strides for output [B*H, L, D] ──
    stride_out_bh,   # = L * D
    stride_out_l,    # = D
    # ── constexprs ──
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    DIM: tl.constexpr,           # H * D
    PAD_H: tl.constexpr,
    PAD_D: tl.constexpr,
    EPS: tl.constexpr,
    HAS_NORM: tl.constexpr,
    RETURN_NORMED: tl.constexpr,
):
    """
    Epilogue kernel: RMSNorm + interleaved RoPE + layout transform.
    One program per token.
    """
    pid = tl.program_id(0).to(tl.int64)
    batch_idx = pid // L
    seq_idx = pid % L

    half_d = HEAD_DIM // 2
    half_arange = tl.arange(0, PAD_D // 2)

    # ── Load cos/sin for this token ──
    cos_sin_full = tl.load(cos_sin_ptr + seq_idx * HEAD_DIM + tl.arange(0, PAD_D),
                           mask=tl.arange(0, PAD_D) < HEAD_DIM, other=0.0).to(tl.float32)
    cs_pairs = tl.reshape(cos_sin_full, (PAD_D // 2, 2))
    cos, sin = tl.split(cs_pairs)
    cos_mask = half_arange < half_d
    cos = tl.where(cos_mask, cos, 1.0)
    sin = tl.where(cos_mask, sin, 0.0)
    cos_2d = cos[None, :]
    sin_2d = sin[None, :]

    # ── Common indices ──
    head_arange = tl.arange(0, PAD_H)[:, None]
    h_mask = head_arange < H
    d_arange = tl.arange(0, PAD_D)[None, :]
    d_mask = d_arange < HEAD_DIM
    hd_mask = h_mask & d_mask

    # Input offset: [M, dim] → token pid, head h, dim d
    raw_base = pid * stride_raw_m + head_arange * HEAD_DIM

    # Output: [B*H, L, D]
    out_head_off = (batch_idx * H + head_arange) * stride_out_bh + seq_idx * stride_out_l

    # Normed output: [B, L, H, D]
    normed_off = pid * DIM + head_arange * HEAD_DIM

    # ═══════════════════════════════════════════════════════
    # Process Q (RMSNorm + RoPE)
    # ═══════════════════════════════════════════════════════
    q_input = tl.load(
        q_raw_ptr + raw_base + d_arange,
        mask=hd_mask,
        other=0.0,
    )
    q_full = q_input.to(tl.float32)

    if HAS_NORM:
        q_flat = tl.reshape(q_full, (PAD_H * PAD_D,))
        q_sq_sum = tl.sum(q_flat * q_flat, axis=0)
        rstd_q = tl.rsqrt(q_sq_sum / DIM + EPS)
        # WanRMSNorm computes the unit RMSNorm in FP32, casts it back to the
        # input dtype, and only then multiplies the BF16 weight.  Preserve both
        # eager rounding boundaries before promoting to FP32 for main RoPE.
        q_unit = (q_full * rstd_q).to(q_raw_ptr.dtype.element_ty)
        wq_full = tl.load(
            wq_ptr + head_arange * HEAD_DIM + d_arange,
            mask=hd_mask,
            other=0.0,
        )
        q_normed_typed = (q_unit * wq_full).to(q_raw_ptr.dtype.element_ty)
    else:
        q_normed_typed = q_input

    if RETURN_NORMED:
        tl.store(
            q_normed_ptr + normed_off + d_arange,
            q_normed_typed,
            mask=hd_mask,
        )

    q_normed = q_normed_typed.to(tl.float32)
    q_pairs = tl.reshape(q_normed, (PAD_H, PAD_D // 2, 2))
    q_even, q_odd = tl.split(q_pairs)
    q_out_even = q_even * cos_2d - q_odd * sin_2d
    q_out_odd = q_even * sin_2d + q_odd * cos_2d
    q_out = tl.reshape(tl.join(q_out_even, q_out_odd), (PAD_H, PAD_D))
    tl.store(q_out_ptr + out_head_off + d_arange, q_out, mask=hd_mask)

    # ═══════════════════════════════════════════════════════
    # Process K (RMSNorm + RoPE)
    # ═══════════════════════════════════════════════════════
    k_input = tl.load(
        k_raw_ptr + raw_base + d_arange,
        mask=hd_mask,
        other=0.0,
    )
    k_full = k_input.to(tl.float32)

    if HAS_NORM:
        k_flat = tl.reshape(k_full, (PAD_H * PAD_D,))
        k_sq_sum = tl.sum(k_flat * k_flat, axis=0)
        rstd_k = tl.rsqrt(k_sq_sum / DIM + EPS)
        k_unit = (k_full * rstd_k).to(k_raw_ptr.dtype.element_ty)
        wk_full = tl.load(
            wk_ptr + head_arange * HEAD_DIM + d_arange,
            mask=hd_mask,
            other=0.0,
        )
        k_normed_typed = (k_unit * wk_full).to(k_raw_ptr.dtype.element_ty)
    else:
        k_normed_typed = k_input

    if RETURN_NORMED:
        tl.store(
            k_normed_ptr + normed_off + d_arange,
            k_normed_typed,
            mask=hd_mask,
        )

    k_normed = k_normed_typed.to(tl.float32)
    k_pairs = tl.reshape(k_normed, (PAD_H, PAD_D // 2, 2))
    k_even, k_odd = tl.split(k_pairs)
    k_out_even = k_even * cos_2d - k_odd * sin_2d
    k_out_odd = k_even * sin_2d + k_odd * cos_2d
    k_out = tl.reshape(tl.join(k_out_even, k_out_odd), (PAD_H, PAD_D))
    tl.store(k_out_ptr + out_head_off + d_arange, k_out, mask=hd_mask)

    # ═══════════════════════════════════════════════════════
    # Process V (layout transform only)
    # ═══════════════════════════════════════════════════════
    v_full = tl.load(
        v_raw_ptr + raw_base + d_arange,
        mask=hd_mask,
        other=0.0,
    )
    tl.store(v_out_ptr + out_head_off + d_arange, v_full, mask=hd_mask)


def fused_qkv_gemm_rope(
    x: torch.Tensor,
    q_proj: torch.nn.Module,
    k_proj: torch.nn.Module,
    v_proj: torch.nn.Module,
    freqs: torch.Tensor,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float = 1e-6,
    return_normed: bool = False,
    fp8_weights: dict | None = None,
    cos_sin: torch.Tensor | None = None,
):
    """
    3 × cuBLAS GEMM + fused Triton epilogue (RMSNorm + RoPE + layout transform).

    Uses the same cuBLAS calls as the reference forward path to ensure
    numerical consistency.

    When ``fp8_weights`` is given (``{'q'|'k'|'v': (w_fp8, w_scale_t)}`` from
    :func:`sparkdiffusion.ops.quantization.fp8_linear.prequantize_qkv`), the three projections run in
    FP8 via ``torch._scaled_mm`` and share one row-wise activation quantization
    of ``x`` (a single amax feeds all three GEMMs). The epilogue is unchanged.
    ``cos_sin`` may be precomputed once and shared by all model layers.

    Returns:
        q_out: [B*H, L, D]  RMSNorm + RoPE applied
        k_out: [B*H, L, D]  RMSNorm + RoPE applied
        v_out: [B*H, L, D]  layout-transformed only
        (if return_normed):
        q_normed: [B, L, H, D]  normed, pre-RoPE
        k_normed: [B, L, H, D]  normed, pre-RoPE
    """
    B, L, dim = x.shape
    H = num_heads
    D = head_dim
    M = B * L

    # ── q/k/v projections: FP8 (shared activation quant) or BF16 cuBLAS ──
    x_flat = x.reshape(M, dim)
    if fp8_weights is not None:
        from sparkdiffusion.ops.quantization.fp8_linear import quantize_activation_rowwise

        x_fp8, x_scale = quantize_activation_rowwise(x_flat)

        def _fp8_linear(proj, name):
            w_fp8, w_scale_t = fp8_weights[name]
            return torch._scaled_mm(
                x_fp8, w_fp8.t(),
                scale_a=x_scale, scale_b=w_scale_t,
                bias=proj.bias, out_dtype=x.dtype, use_fast_accum=True,
            )

        q_raw = _fp8_linear(q_proj, "q")  # [M, dim]
        k_raw = _fp8_linear(k_proj, "k")  # [M, dim]
        v_raw = _fp8_linear(v_proj, "v")  # [M, dim]
    else:
        q_raw = F.linear(x_flat, q_proj.weight, q_proj.bias)  # [M, dim]
        k_raw = F.linear(x_flat, k_proj.weight, k_proj.bias)  # [M, dim]
        v_raw = F.linear(x_flat, v_proj.weight, v_proj.bias)  # [M, dim]

    # Standalone callers keep the original behaviour. WanModel supplies a
    # forward-scoped table so every transformer block can reuse it.
    if cos_sin is None:
        cos_sin = prepare_cos_sin(freqs, L, D)

    # ── Epilogue kernel ──
    q_out = torch.empty(B * H, L, D, dtype=x.dtype, device=x.device)
    k_out = torch.empty(B * H, L, D, dtype=x.dtype, device=x.device)
    v_out = torch.empty(B * H, L, D, dtype=x.dtype, device=x.device)

    q_normed_out = torch.empty(B, L, H, D, dtype=x.dtype, device=x.device) if return_normed else None
    k_normed_out = torch.empty(B, L, H, D, dtype=x.dtype, device=x.device) if return_normed else None

    PAD_H = triton.next_power_of_2(H)
    PAD_D = triton.next_power_of_2(D)

    has_norm = norm_q_weight is not None and norm_k_weight is not None
    dummy = q_raw

    _epilogue_kernel[(M,)](
        q_raw, k_raw, v_raw,
        cos_sin,
        norm_q_weight if has_norm else dummy,
        norm_k_weight if has_norm else dummy,
        q_out, k_out, v_out,
        q_normed_out if return_normed else dummy,
        k_normed_out if return_normed else dummy,
        L, H,
        dim,
        L * D, D,
        HEAD_DIM=D,
        NUM_HEADS=H,
        DIM=dim,
        PAD_H=PAD_H,
        PAD_D=PAD_D,
        EPS=eps,
        HAS_NORM=has_norm,
        RETURN_NORMED=return_normed,
    )

    if return_normed:
        return q_out, k_out, v_out, q_normed_out, k_normed_out
    return q_out, k_out, v_out
