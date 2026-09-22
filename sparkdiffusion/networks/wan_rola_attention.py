"""
RoLa sparse + low-rank self-attention for Wan diffusion models.

The attention design is adapted from the sparse attention processor used by
finetrainers. The block-map and backward-compatible training kernel are
provided by the optional external sparse-attention package; the inference path
uses the Alibaba-developed Triton kernels in ``sparkdiffusion.ops``.

Source attribution:
  - NVIDIA rCM: https://github.com/NVlabs/rcm
  - Optional backward-compatible training kernel: https://github.com/thu-ml/SLA

Key differences from finetrainers:
  - Uses WanModel's freqs format: [L, head_dim//2] raw theta values (not (cos, sin) pair)
  - Uses WanModel's BLHD tensor layout (not diffusers BHLD)
  - No diffusers AttnProcessor wrapper — integrates directly into WanSelfAttention
  - Stage1 MSE loss accumulated via RoLaTrainingContext (not ProfilerMetrics)
"""

import math
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from imaginaire.utils import log
from sparkdiffusion.networks.sparse_attn_registry import register_sparse_attn


# ---------------------------------------------------------------------------
# Attention Profiler: toggle-able CUDA-event timing for forward passes
# ---------------------------------------------------------------------------

class AttnProfiler:
    """
    Global profiler that records per-layer CUDA-event timings.
    Usage:
        AttnProfiler.enable()          # before inference
        ... run inference ...
        AttnProfiler.report()           # print stats
        AttnProfiler.disable()          # stop recording
    """
    _enabled: bool = False
    _records: dict = {}  # {layer_name: [elapsed_ms, ...]}

    @classmethod
    def enable(cls):
        cls._enabled = True
        cls._records.clear()

    @classmethod
    def disable(cls):
        cls._enabled = False

    @classmethod
    def is_enabled(cls) -> bool:
        return cls._enabled

    @classmethod
    def record(cls, name: str, start_event, end_event):
        if not cls._enabled:
            return
        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event)
        cls._records.setdefault(name, []).append(elapsed)

    @classmethod
    def report(cls):
        if not cls._records:
            print("[AttnProfiler] No records.")
            return
        print("\n" + "=" * 70)
        print("[AttnProfiler] Attention Timing Report")
        print("=" * 70)
        for name, times in sorted(cls._records.items()):
            t = torch.tensor(times)
            print(f"  {name:40s}: mean={t.mean():.3f}ms  std={t.std():.3f}ms  "
                  f"min={t.min():.3f}ms  max={t.max():.3f}ms  n={len(times)}")
        # Aggregate by type
        rola_times = [v for k, v in cls._records.items() if "RoLa" in k]
        dense_times = [v for k, v in cls._records.items() if "Dense" in k]
        if rola_times:
            all_rola = torch.tensor([x for sublist in rola_times for x in sublist])
            print(f"\n  {'[ALL RoLa layers]':40s}: mean={all_rola.mean():.3f}ms  total_calls={len(all_rola)}")
        if dense_times:
            all_dense = torch.tensor([x for sublist in dense_times for x in sublist])
            print(f"  {'[ALL Dense layers]':40s}: mean={all_dense.mean():.3f}ms  total_calls={len(all_dense)}")
        print("=" * 70 + "\n")

    @classmethod
    def clear(cls):
        cls._records.clear()

_SLA_SRC = os.environ.get("SLA_SRC", "")
SLA_AVAILABLE = False
_SLA_IMPORT_ERROR: Exception | None = None
try:
    if not _SLA_SRC or not os.path.isdir(_SLA_SRC):
        raise ImportError("SLA_SRC is not set to an external sparse-attention checkout")
    if _SLA_SRC not in sys.path:
        sys.path.insert(0, _SLA_SRC)
    from sparse_linear_attention.utils import get_block_map
    from sparse_linear_attention.kernel import _attention as _sla_attention
    SLA_AVAILABLE = True
except Exception as exc:
    _SLA_IMPORT_ERROR = exc

# ---------------------------------------------------------------------------
# SageSLA (spas_sage_attn) is a legacy implementation that depends on an external
# package. INT8 keeps the original SageSLA/PyTorch path until a validated fused kernel
# is available, so these imports stay for the legacy _sparse_attn() path.
# ---------------------------------------------------------------------------
SAGESLA_AVAILABLE = False
try:
    _sagesla_path = os.path.join(_SLA_SRC, "SageSLA") if _SLA_SRC else None
    if _sagesla_path and os.path.dirname(_sagesla_path) not in sys.path:
        sys.path.insert(0, os.path.dirname(_sagesla_path))

    from SageSLA.utils import get_block_map as _sage_get_block_map        # noqa: F401
    from SageSLA.utils import get_cuda_arch as _sage_get_cuda_arch        # noqa: F401
    from spas_sage_attn.utils import get_vanilla_qk_quant as _sage_get_qk_quant  # noqa: F401
    from spas_sage_attn.utils import block_map_lut_triton as _sage_lut_triton    # noqa: F401
    import spas_sage_attn._qattn as _sage_qattn                           # noqa: F401
    import spas_sage_attn._fused as _sage_fused                           # noqa: F401

    _SAGE2PP_ENABLED = False
    try:
        from spas_sage_attn._qattn import \
            qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold \
            as _sage_qattn_sm100                                          # noqa: F401
        _SAGE2PP_ENABLED = True
    except ImportError:
        pass

    SAGESLA_AVAILABLE = True
except Exception:
    pass


# ---------------------------------------------------------------------------
# Global training context (thread-unsafe but single-GPU friendly)
# ---------------------------------------------------------------------------

class RoLaTrainingContext:
    """
    Lightweight class-level context passed between the trainer and RoLa attention layers.
    Set flags before a forward pass; read accumulated losses after.

    NOTE: the Stage-1 MSE alignment records a loss via this global side effect.
    It is INCOMPATIBLE with activation-checkpoint recomputation (the block forward
    would run the dense-attn MSE subgraph twice, corrupting SAC's saved-tensor
    bookkeeping → CheckpointError). The finetune model therefore disables
    activation checkpointing whenever Stage-1 MSE is enabled.
    """
    _ctx: dict = {}
    _mse_losses: list = []

    @classmethod
    def set(cls, **kwargs):
        cls._ctx.update(kwargs)

    @classmethod
    def get(cls) -> dict:
        return cls._ctx

    @classmethod
    def clear_mse_losses(cls):
        cls._mse_losses.clear()

    @classmethod
    def add_mse_loss(cls, loss: torch.Tensor):
        cls._mse_losses.append(loss)

    @classmethod
    def pop_mse_loss_mean(cls) -> Optional[torch.Tensor]:
        if not cls._mse_losses:
            return None
        mean = torch.stack(cls._mse_losses).mean()
        cls._mse_losses.clear()
        return mean



# ---------------------------------------------------------------------------
# WanSelfAttentionRoLa
# ---------------------------------------------------------------------------

@register_sparse_attn("rola", sparse_param_names=["proj_q", "proj_k", "gate_proj", "gate_bias"])
class WanSelfAttentionRoLa(nn.Module):
    """
    Drop-in replacement for WanSelfAttention that adds:
      - Block-sparse softmax attention via SLA kernel (sparse branch)
      - Low-rank linear attention with truncated RoPE (low-rank branch)
      - Learnable gate for fusing sparse + low-rank outputs
      - Optional Stage-1 MSE alignment loss (accumulated in RoLaTrainingContext)

    Inherits the same constructor signature as WanSelfAttention so it can be
    swapped in transparently. New RoLa parameters:
      proj_q, proj_k : nn.Linear  head_dim -> rola_rank
      gate_proj       : nn.Linear  head_dim -> 1
      gate_bias       : nn.Parameter [1, num_heads, 1, 1]

    The dense attn_op (MinimalA2AAttnOp) is kept for Stage-1 MSE reference.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
        rola_topk_ratio: float = 0.1,
        rola_rank: int = 64,
        rola_blkq: int = 64,
        rola_blkk: int = 64,
        attn_precision: str = "bf16",
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        self.rola_topk_ratio = rola_topk_ratio
        self.rola_rank = rola_rank
        self.rola_blkq = rola_blkq
        self.rola_blkk = rola_blkk
        # attn_precision: "bf16" | "int8"
        # Controls which sparse attention kernel is used in fused_forward.
        #   bf16  — fused_sparse_attn       (SM80+, default)
        #   int8  — original path (no validated fused INT8 kernel yet)
        if attn_precision not in ("bf16", "int8"):
            raise ValueError(f"attn_precision must be bf16/int8, got {attn_precision!r}")
        self.attn_precision = attn_precision

        # Standard backbone projections (same names as WanSelfAttention)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        if qk_norm:
            from sparkdiffusion.networks.wan2pt1 import WanRMSNorm
            self.norm_q = WanRMSNorm(dim, eps=eps)
            self.norm_k = WanRMSNorm(dim, eps=eps)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

        # Dense attention op kept for Stage-1 MSE reference
        from sparkdiffusion.utils.a2a_cp import MinimalA2AAttnOp
        self.attn_op = MinimalA2AAttnOp()

        # ---- New RoLa parameters ----
        h = self.head_dim
        r = rola_rank
        self.proj_q = nn.Linear(h, r, bias=False)
        self.proj_k = nn.Linear(h, r, bias=False)
        self.gate_proj = nn.Linear(h, 1, bias=False)
        self.gate_bias = nn.Parameter(torch.full((1, num_heads, 1, 1), -1.946))

        # Initialise new params
        nn.init.kaiming_uniform_(self.proj_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.proj_k.weight, a=math.sqrt(5))
        nn.init.zeros_(self.gate_proj.weight)

        # Profiling label (set externally if needed, e.g. "block_3.RoLa")
        self._layer_name: str = "RoLa"
        self._use_fused_inference = False

    def set_context_parallel_group(self, process_group, ranks, stream):
        self.attn_op.set_context_parallel_group(process_group, ranks, stream)

    def init_weights(self):
        # WanModel.init_weights() calls xavier_uniform on all nn.Linear, overwriting
        # the RoLa-specific init from __init__.  Re-apply the correct initialisation:
        #   proj_q/k  : kaiming_uniform
        #   gate_proj : zeros
        #   gate_bias : -1.946  (→ sigmoid ≈ 0.125, sparse dominates initially)
        nn.init.kaiming_uniform_(self.proj_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.proj_k.weight, a=math.sqrt(5))
        nn.init.zeros_(self.gate_proj.weight)
        self.gate_bias.data.fill_(-1.946)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sparse_attn(
        self,
        q_bhld: torch.Tensor,
        k_bhld: torch.Tensor,
        v_bhld: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Block-sparse softmax attention.
        Inference (no_grad): SageSLA INT8/FP8 quantized path (faster).
        Training: SLA BF16 exact path (has backward).
        """
        if not SLA_AVAILABLE:
            raise RuntimeError(
                "RoLa sparse attention requires the external SLA training kernel; "
                "set SLA_SRC to its source checkout before starting Python"
            ) from _SLA_IMPORT_ERROR
        L = q_bhld.shape[2]
        if L < max(self.rola_blkq, self.rola_blkk):
            return None

        orig_dtype = q_bhld.dtype
        q = q_bhld.contiguous().to(torch.bfloat16)
        k = k_bhld.contiguous().to(torch.bfloat16)
        v = v_bhld.contiguous().to(torch.bfloat16)

        # ── Training / SageSLA unavailable: SLA BF16 exact path ─────────────────────
        try:
            sparse_map, lut, real_topk = get_block_map(
                q, k, self.rola_topk_ratio, self.rola_blkq, self.rola_blkk
            )
            out = _sla_attention.apply(q, k, v, sparse_map, lut, real_topk,
                                       self.rola_blkq, self.rola_blkk)
            return out.to(orig_dtype)
        except Exception as exc:
            raise RuntimeError("SLA sparse-attention kernel execution failed") from exc

    def _apply_rope_rank(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply truncated RoPE to low-rank tensor using INTERLEAVED rotation
        (matching finetrainers' _apply_rotary_emb_rank).

        x     : [B, H, L, rank]
        freqs : [L_max, head_dim//2]  raw theta values
        Returns [B, H, L, rank] with interleaved rotation.

        Interleaved pairing: (x[0],x[1]), (x[2],x[3]), ... — same as main-branch RoPE.
        """
        r = self.rola_rank
        L = x.shape[2]
        # Cap rotation to available frequencies
        rope_dim = min(r // 2, freqs.shape[-1])
        freqs_r = freqs[:L, :rope_dim].to(x.dtype)   # [L, rope_dim]
        cos = torch.cos(freqs_r)  # [L, rope_dim]
        sin = torch.sin(freqs_r)  # [L, rope_dim]

        # Interleaved split: x1 = x[..., 0::2], x2 = x[..., 1::2] (adjacent pairs)
        x_rot = x[..., : 2 * rope_dim]  # [B, H, L, 2*rope_dim]
        x1, x2 = x_rot.unflatten(-1, (-1, 2)).unbind(-1)  # each [B, H, L, rope_dim]

        # Apply rotation
        out = torch.empty_like(x_rot)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos

        # Append non-rotated tail dimensions if rank > 2*rope_dim
        if 2 * rope_dim < r:
            out = torch.cat([out, x[..., 2 * rope_dim:]], dim=-1)
        return out

    def _lowrank_attn(
        self,
        q_normed_bhld: torch.Tensor,
        k_normed_bhld: torch.Tensor,
        v_bhld: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Low-rank linear attention branch.
        Inputs : [B, H, L, D]
        Returns: [B, H, L, D]
        """
        # Project head_dim → rank
        q_lr = F.silu(self.proj_q(q_normed_bhld))  # [B, H, L, rank]
        k_lr = F.silu(self.proj_k(k_normed_bhld))  # [B, H, L, rank]

        # Apply truncated RoPE after activation
        q_lr = self._apply_rope_rank(q_lr, freqs)
        k_lr = self._apply_rope_rank(k_lr, freqs)

        # Linear attention: O = Q_lr (K_lr^T V)
        # k_lr^T @ v: [B, H, rank, L] @ [B, H, L, D] = [B, H, rank, D]
        c = k_lr.transpose(-2, -1) @ v_bhld  # [B, H, rank, D]
        out = q_lr @ c                        # [B, H, L, D]

        return F.rms_norm(out, [out.shape[-1]])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def fused_forward(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,  # noqa: ARG002 — kept for API compatibility
        freqs: torch.Tensor,
        rope_cache=None,
    ) -> torch.Tensor:
        """Full-fusion inference path composed from Triton and PyTorch ops.

        The Wan T2V inference entry points wrap the complete model with
        ``torch.compile``.  Keeping this function free of manual CUDA Graph
        capture lets Inductor optimize the surrounding PyTorch operations
        without retaining per-layer static activation buffers.
        """
        from sparkdiffusion.ops.fused_kernel import fused_gate, fused_lowrank, fused_sparse_attn
        from sparkdiffusion.ops.fused_kernel.fused_qkv_rope import fused_qkv_gemm_rope
        from sparkdiffusion.ops.quantization.fp8_linear import fp8_qkv_weights

        def _compute(x, freqs):
            n, d = self.num_heads, self.head_dim
            norm_q_w = self.norm_q.weight if self.qk_norm else None
            norm_k_w = self.norm_k.weight if self.qk_norm else None

            q_rope, k_rope, v_out, q_normed, k_normed = fused_qkv_gemm_rope(
                x, self.q, self.k, self.v, freqs, norm_q_w, norm_k_w, n, d,
                eps=self.eps, return_normed=True,
                fp8_weights=fp8_qkv_weights(self),
                cos_sin=rope_cache[0] if rope_cache is not None else None,
            )

            o_sparse = fused_sparse_attn(
                q_rope, k_rope, v_out,
                topk_ratio=self.rola_topk_ratio,
                blkq=self.rola_blkq, blkk=self.rola_blkk, num_heads=n,
            )

            o_lr = fused_lowrank(
                q_normed, k_normed, v_out, freqs,
                norm_q_weight=None, norm_k_weight=None,
                proj_q_weight=self.proj_q.weight,
                proj_k_weight=self.proj_k.weight,
                rola_rank=self.rola_rank, eps=self.eps,
                rope_tables=rope_cache[1] if rope_cache is not None else None,
            )

            gate_bias_flat = self.gate_bias.view(self.num_heads)
            out = fused_gate(
                o_lr, o_sparse, x,
                gate_proj_weight=self.gate_proj.weight,
                gate_bias=gate_bias_flat,
            )
            return self.o(out.to(x.dtype))

        return _compute(x, freqs)


    def forward(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,  # noqa: ARG002 — kept for API compatibility
        freqs: torch.Tensor,
        rope_cache=None,
    ) -> torch.Tensor:
        """
        Args:
            x        : [B, L, dim]
            seq_lens : [B]  (unused; accepted for API compatibility with WanSelfAttention)
            freqs    : [L, head_dim//2]  raw theta values (WanModel RoPE format)
        Returns:
            [B, L, dim]
        """
        use_fused = (
            self._use_fused_inference
            and not self.training
            and not torch.is_grad_enabled()
            and x.shape[1] >= max(self.rola_blkq, self.rola_blkk)
            and self.attn_precision != "int8"
            and (
                getattr(self, "_fp8_qkv", False)
                or all(isinstance(proj, nn.Linear) for proj in (self.q, self.k, self.v))
            )
        )
        if use_fused:
            return self.fused_forward(x, seq_lens, freqs, rope_cache=rope_cache)

        from sparkdiffusion.networks.wan2pt1 import rope_apply

        _profiling = AttnProfiler.is_enabled()
        if _profiling:
            _start = torch.cuda.Event(enable_timing=True)
            _end = torch.cuda.Event(enable_timing=True)
            _start.record()

        b, s, n, d = x.shape[0], x.shape[1], self.num_heads, self.head_dim

        # ---- QKV projections + QK norm → BLHD ----
        q = self.norm_q(self.q(x)).view(b, s, n, d)  # [B, L, H, D]
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        # ---- Save pre-RoPE for low-rank branch ----
        q_normed_bhld = q.transpose(1, 2)  # [B, H, L, D]
        k_normed_bhld = k.transpose(1, 2)

        # ---- Apply RoPE to main-branch Q/K ----
        q_rope = rope_apply(q, freqs)  # [B, L, H, D]
        k_rope = rope_apply(k, freqs)

        # ---- Transpose to BHLD for RoLa / low-rank kernels ----
        q_bhld = q_rope.transpose(1, 2)   # [B, H, L, D]
        k_bhld = k_rope.transpose(1, 2)
        v_bhld = v.transpose(1, 2)

        # ---- Sparse softmax branch ----
        o_s = self._sparse_attn(q_bhld, k_bhld, v_bhld)  # [B, H, L, D] or None

        # ---- Low-rank linear branch ----
        o_l = self._lowrank_attn(q_normed_bhld, k_normed_bhld, v_bhld, freqs)

        # ---- Gated fusion ----
        # gate input: original hidden states x reshaped to BHLD
        x_bhld = x.view(b, s, n, d).transpose(1, 2)           # [B, H, L, D]
        gate_logits = self.gate_proj(x_bhld) + self.gate_bias  # [B, H, L, 1]
        gate = torch.sigmoid(gate_logits)

        if o_s is not None:
            # RMSnorm-scaled sparse + gated low-rank
            o_s_scale = (o_s.pow(2).mean(dim=-1, keepdim=True) + 1e-5).sqrt()
            fused = o_s.float() / o_s_scale.float() + gate * o_l.float()
            out_bhld = (fused * o_s_scale.float()).to(o_l.dtype)
        else:
            out_bhld = gate * o_l

        # ---- BHLD → BL(HD) ----
        out = out_bhld.transpose(1, 2).reshape(b, s, n * d)  # [B, L, H*D]

        # ---- Stage-1 MSE alignment loss ----
        # NOTE: incompatible with activation-checkpoint recompute; the finetune
        # model disables checkpointing while Stage-1 MSE is active.
        ctx = RoLaTrainingContext.get()
        if ctx.get("enable_stage1_mse"):
            with torch.no_grad():
                # Dense reference via existing attn_op (MinimalA2AAttnOp, takes BLHD)
                # attn_op returns [B, L, H, D]; flatten to [B, L, H*D] to match out.
                out_dense = self.attn_op(q_rope, k_rope, v).reshape(b, s, n * d)
            mse = F.mse_loss(out.float(), out_dense.float())
            RoLaTrainingContext.add_mse_loss(mse)

        # ---- Output projection ----
        result = self.o(out)

        if _profiling:
            _end.record()
            AttnProfiler.record(self._layer_name, _start, _end)

        return result


# ---------------------------------------------------------------------------
# WanSelfAttentionPureSLA — original SLA library (SparseLinearAttention)
# ---------------------------------------------------------------------------
# Differences from WanSelfAttentionRoLa:
#   - The linear branch uses the SLA paper's feature-map linear attention
#     (softmax/elu/relu + proj_l) instead of truncated-RoPE low-rank projection
#   - No gate mechanism; fusion is o_s + proj_l(o_l) (proj_l zero-initialized)
#   - Reuses the SparseLinearAttention module from the SLA library directly
# ---------------------------------------------------------------------------

class WanSelfAttentionPureSLA(nn.Module):
    """
    Drop-in replacement for WanSelfAttention using the original SparseLinearAttention
    from the SLA library (https://github.com/thu-ml/SLA).

    Sparse branch  : SLA Triton block-sparse softmax attention (topk_ratio blocks)
    Linear branch  : feature-map linear attention (softmax/elu/relu) + proj_l correction
    Fusion         : o_s + o_l  (proj_l zero-initialized → starts as pure sparse)

    New learnable params (beyond backbone q/k/v/o):
      proj_l : nn.Linear(head_dim, head_dim, dtype=float32) — zero-initialized
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
        # RoLa-specific
        rola_topk_ratio: float = 0.1,
        rola_blkq: int = 64,
        rola_blkk: int = 64,
        sla_feature_map: str = "softmax",   # "softmax" | "elu" | "relu"
        attn_precision: str = "bf16",       # kept for API compat, not used here
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self._layer_name: str = "PureSLA"

        # Standard backbone projections (same names as WanSelfAttention for weight loading)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        if qk_norm:
            from sparkdiffusion.networks.wan2pt1 import WanRMSNorm
            self.norm_q = WanRMSNorm(dim, eps=eps)
            self.norm_k = WanRMSNorm(dim, eps=eps)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

        # Dense attention op for Stage-1 MSE reference
        from sparkdiffusion.utils.a2a_cp import MinimalA2AAttnOp
        self.attn_op = MinimalA2AAttnOp()

        # SparseLinearAttention module from SLA library
        from sparse_linear_attention import SparseLinearAttention
        self.sla = SparseLinearAttention(
            head_dim=self.head_dim,
            topk=rola_topk_ratio,
            feature_map=sla_feature_map,
            BLKQ=rola_blkq,
            BLKK=rola_blkk,
            use_bf16=True,
            tie_feature_map_qk=True,
        )

    def init_weights(self) -> None:
        pass  # SparseLinearAttention.__init__ already calls init_weights_()

    def set_context_parallel_group(self, process_group, ranks, stream) -> None:
        self.attn_op.set_context_parallel_group(process_group, ranks, stream)

    def forward(
        self,
        x: torch.Tensor,          # [B, L, dim]
        seq_lens: torch.Tensor,   # [B]  (unused, kept for API compat)
        freqs: torch.Tensor,      # [L, head_dim//2]
    ) -> torch.Tensor:
        from sparkdiffusion.networks.wan2pt1 import rope_apply

        b, s, n, d = x.shape[0], x.shape[1], self.num_heads, self.head_dim

        # QKV + QK norm
        q = self.norm_q(self.q(x)).view(b, s, n, d)  # [B, L, H, D]
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        # Apply RoPE
        q_rope = rope_apply(q, freqs)  # [B, L, H, D]
        k_rope = rope_apply(k, freqs)

        # Transpose to [B, H, L, D] for SLA
        q_bhld = q_rope.transpose(1, 2)
        k_bhld = k_rope.transpose(1, 2)
        v_bhld = v.transpose(1, 2)

        # SparseLinearAttention forward: sparse + linear(feature_map) + proj_l
        out_bhld = self.sla(q_bhld, k_bhld, v_bhld)  # [B, H, L, D]

        # Reshape back to [B, L, H*D]
        out = out_bhld.transpose(1, 2).reshape(b, s, n * d)

        # Stage-1 MSE alignment loss
        ctx = RoLaTrainingContext.get()
        if ctx.get("enable_stage1_mse"):
            with torch.no_grad():
                out_dense = self.attn_op(q_rope, k_rope, v).reshape(b, s, n * d)
            mse = F.mse_loss(out.float(), out_dense.float())
            RoLaTrainingContext.add_mse_loss(mse)

        return self.o(out)
