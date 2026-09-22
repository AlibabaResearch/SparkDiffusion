# Copyright (c) 2025 Alibaba Group Holding Limited.
# All rights reserved.
#
# SparkDiffusion-specific Triton kernel and integration code.

"""
Block-sparse Flash Attention (BF16, pure Triton).

Reimplements the sparse-attention forward operation as a self-contained
Triton kernel + wrapper:
  1. Block map: mean-pool + smooth-K + topk selection
  2. Block-sparse attention: BF16 QK + BF16 PV, online softmax (exp2), LUT iteration

Source attribution:
  - Triton fused-attention tutorial:
    https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py
  - The optional backward-compatible training interface remains compatible
    with the external sparse-attention package used by the training path.

Optimizations adapted from the Triton fused-attention tutorial:
  - Autotune: search num_warps × num_stages with persistent result caching
  - Blackwell acc split/join: reduce register pressure for 128×D tiles

Inference-only (no backward).
"""

import torch
import triton
import triton.language as tl

from sparkdiffusion.utils.cuda_arch import is_blackwell

from .utils import mean_pool


# ═══════════════════════════════════════════════════════════════
# Autotune configuration (matches Triton FA tutorial pattern)
# ═══════════════════════════════════════════════════════════════

# No early_config_prune hook: Dynamo cannot trace the autotuner's prune
# callback, and every transformer block calls this kernel, so wrapping the
# shared config list twice aborts the compile. The autotuner already drops
# slow configs by measurement, so a prune heuristic buys nothing here.
_ATTN_CONFIGS = [
    triton.Config({}, num_warps=w, num_stages=s)
    for w in [4, 8]
    for s in [2, 3, 4]
]


# ═══════════════════════════════════════════════════════════════
# Kernel: Block-sparse Flash Attention (BF16)
# ═══════════════════════════════════════════════════════════════

@triton.autotune(
    configs=_ATTN_CONFIGS,
    key=['topk', 'D', 'BLOCK_M', 'BLOCK_N'],
    cache_results=True,
)
@triton.jit
def _rola_fwd_kernel(
    Q_ptr,          # [B, H, L, D] bf16
    K_ptr,          # [B, H, L, D] bf16
    V_ptr,          # [B, H, L, D] bf16
    LUT_ptr,        # [B*H, M_BLOCKS, topk] int64
    O_ptr,          # [B, H, L, D] bf16 output
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_BLACKWELL: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    qkv_offset = idx_bh * L * D
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q_ptr + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K_ptr + qkv_offset + offs_n[None, :] * D + offs_d[:, None]
    V_ptrs = V_ptr + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    O_ptrs = O_ptr + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    LUT_base = LUT_ptr + lut_offset

    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L, other=0.0)

    for block_idx in tl.range(topk):
        idx_n = tl.load(LUT_base + block_idx)
        n_mask = offs_n < L - idx_n * BLOCK_N

        k = tl.load(
            K_ptrs + idx_n * BLOCK_N * D,
            mask=n_mask[None, :],
            other=0.0,
        )
        qk = tl.dot(q, k) * qk_scale
        if L - idx_n * BLOCK_N < BLOCK_N:
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

        # Load V early to overlap with softmax compute (matches original SLA kernel)
        v = tl.load(
            V_ptrs + idx_n * BLOCK_N * D,
            mask=n_mask[:, None],
            other=0.0,
        )

        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        qk = qk - new_m[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - new_m)

        # Blackwell acc rescale: split/join to reduce register pressure
        # (from Triton FA tutorial 06-fused-attention.py:87-93)
        if IS_BLACKWELL and BLOCK_M == 128 and D == 128:
            BM: tl.constexpr = o_acc.shape[0]
            BN: tl.constexpr = o_acc.shape[1]
            a0, a1 = o_acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            a0 = a0 * alpha[:, None]
            a1 = a1 * alpha[:, None]
            o_acc = tl.join(a0, a1).permute(0, 2, 1).reshape([BM, BN])
        else:
            o_acc = o_acc * alpha[:, None]

        o_acc += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_acc = o_acc / l_i[:, None]
    tl.store(O_ptrs, o_acc.to(O_ptr.type.element_ty), mask=offs_m[:, None] < L)


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def fused_sparse_attn(q, k, v, topk_ratio=0.1, blkq=64, blkk=64,
                      num_heads=None, force_local_sink=False):
    """
    Block-sparse Flash Attention (BF16, pure Triton).

    Inputs: q, k, v [B*H, L, D] bf16, already norm+RoPE applied.
    force_local_sink: guarantee every query block keeps its own (diagonal)
        block and block 0 (attention sink) in the top-k selection, regardless
        of pooled scores. Enabling it costs no extra kernel (just two writes
        into pooled_score before the existing top-k) and brings a slight gain
        in output completeness/stability, but it is DISABLED by default for
        now to keep block selection identical to the training-time behavior.
    Returns: [B, H, L, D]
    """
    BH, L, D = q.shape
    if num_heads is None or BH % num_heads != 0:
        raise ValueError(f"invalid num_heads={num_heads} for leading dimension {BH}")
    H = num_heads
    B = BH // H
    q_4d = q.view(B, H, L, D).contiguous()
    k_4d = k.view(B, H, L, D).contiguous()
    v_4d = v.view(B, H, L, D).contiguous()

    M_BLOCKS = triton.cdiv(L, blkq)
    N_BLOCKS = triton.cdiv(L, blkk)

    # Match the training-time smooth-K router while avoiding a full-sized
    # centered K allocation: subtract the sequence mean inside mean_pool.
    pooled_q = mean_pool(q_4d, blkq)
    pooled_k = mean_pool(k_4d, blkk, center=k_4d.mean(dim=-2))
    pooled_score = pooled_q @ pooled_k.transpose(-1, -2)
    if force_local_sink:
        # Optional (off by default): force two always-relevant blocks into the
        # top-k by pinning their pooled score to +inf before selection:
        #   - diagonal block (query block m attends to key block m): the local
        #     window around each token; mean-pool scoring can otherwise drop a
        #     query block's own context.
        #   - block 0 (attention sink): a stable anchor that most heads attend
        #     to; keeping it reduces frame-to-frame drift.
        # Enabling gives a slight quality gain at zero kernel cost (only two
        # writes into pooled_score; top-k and the attention kernel are
        # unchanged), but it is kept off for now.
        inf = torch.finfo(pooled_score.dtype).max
        diag = torch.arange(M_BLOCKS, device=pooled_score.device).clamp_max_(N_BLOCKS - 1)
        pooled_score.scatter_(-1, diag.view(1, 1, M_BLOCKS, 1).expand(B, H, M_BLOCKS, 1), inf)
        pooled_score[..., 0] = inf
    real_topk = max(1, min(N_BLOCKS, int(topk_ratio * N_BLOCKS)))
    lut = torch.topk(pooled_score, real_topk, dim=-1, sorted=False).indices

    qk_scale = D ** -0.5 * 1.4426950408889634  # 1/sqrt(D) * log2(e)

    o = torch.empty_like(v_4d)

    _rola_fwd_kernel[(M_BLOCKS, B * H)](
        q_4d, k_4d, v_4d,
        lut.reshape(B * H, M_BLOCKS, real_topk),
        o,
        qk_scale,
        real_topk,
        L, M_BLOCKS,
        D, blkq, blkk,
        IS_BLACKWELL=is_blackwell(q.device),
    )

    return o  # [B, H, L, D]
