"""Fused Triton kernels for SparkDiffusion RoLa attention.

Modules:
  - fused_qkv_rope:          3×cuBLAS GEMM + fused RMSNorm + RoPE epilogue
  - fused_sparse_attn:       Block-sparse softmax attention (BF16)
  - fused_lowrank:           Low-rank linear attention (proj + SiLU + RoPE + matmul)
  - fused_gate:              Gated fusion of sparse + low-rank outputs
  - utils:                   Shared layout and pooling helpers

The backward-compatible training path may still use the optional external
sparse-attention package. These modules provide the repository's inference
kernels and do not require that package.
"""

from .utils import (
    mean_pool,
    reshape_blhd_to_bhl_d,
)
from .fused_qkv_rope import (
    fused_qkv_gemm_rope,
)
from .fused_sparse_attn import fused_sparse_attn
from .fused_lowrank import fused_lowrank
from .fused_gate import fused_gate

__all__ = [
    # utils
    "reshape_blhd_to_bhl_d",
    "mean_pool",
    # fused_qkv_rope
    "fused_qkv_gemm_rope",
    # attention kernels
    "fused_sparse_attn",
    "fused_lowrank",
    "fused_gate",
]
