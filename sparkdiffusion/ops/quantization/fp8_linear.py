# Copyright (c) 2025 Alibaba Group Holding Limited.
# All rights reserved.
#
# SparkDiffusion-specific FP8 quantization and Triton kernel integration.

"""Inference-only FP8 GEMM via ``torch._scaled_mm``.

Activations use dynamic row-wise scales on every call. This costs an amax
reduction, but unlike first-batch static calibration it cannot silently clip a
later diffusion step whose activation range is larger.

Two entry points share the same row-wise FP8 recipe:
  * ``FP8Linear`` wraps a standalone ``nn.Linear`` (FFN GEMMs, attention output
    and cross-attention projections).
  * ``prequantize_qkv`` replaces self-attention q/k/v with ``FP8Linear`` so the
    fused QKV+RoPE kernel can share one activation quantization while the eager
    fallback consumes the same FP8 weights.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

from sparkdiffusion.utils.cuda_arch import get_compute_capability, supports_fp8_rowwise_scaled_mm

_FP8_MAX = 448.0
_FP8_DTYPE = torch.float8_e4m3fn

# Widest row the fused GELU+quant kernel keeps resident. Beyond this the tile
# would spill and the two-pass fallback is faster.
_FUSED_GELU_MAX_BLOCK = 16384


def _require_fp8_gemm(device: torch.device) -> None:
    if device.type != "cuda" or not supports_fp8_rowwise_scaled_mm(device):
        capability = get_compute_capability(device)
        suffix = "non-CUDA device" if capability is None else f"SM{capability[0]}{capability[1]}"
        raise RuntimeError(
            "FP8Linear requires a GPU with FP8 E4M3 tensor cores (SM89+) whose "
            f"torch._scaled_mm build supports row-wise scaling; got {suffix}"
        )


def quantize_weight_rowwise(weight: torch.Tensor):
    """Row-wise (per-output-channel) FP8 weight quantization.

    Returns ``(w_fp8 [out, in], w_scale_t [1, out])`` so the scale is already in
    the layout ``torch._scaled_mm`` wants for ``scale_b``.
    """
    w = weight.detach()
    w_scale = w.abs().amax(dim=1, keepdim=True).float() / _FP8_MAX + 1e-12  # [out, 1]
    w_fp8 = (w / w_scale.to(w.dtype)).to(_FP8_DTYPE)
    return w_fp8, w_scale.t().contiguous()  # [out,in], [1,out]


def quantize_activation_rowwise(x2: torch.Tensor):
    """Row-wise (per-token) FP8 activation quantization for a 2-D ``[M, K]`` tensor."""
    x_scale = x2.abs().amax(dim=1, keepdim=True).float() / _FP8_MAX + 1e-12  # [M, 1]
    x_fp8 = (x2 / x_scale.to(x2.dtype)).to(_FP8_DTYPE)
    return x_fp8, x_scale


@triton.jit
def _gelu_quant_rowwise_kernel(
    X,
    Xq,
    Xs,
    K,
    stride_xm,
    BLOCK_K: tl.constexpr,
):
    """Fuse tanh-approximation GELU with row-wise FP8 quantization."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K

    x = tl.load(
        X + row * stride_xm + offs,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    # tanh(z) = 2*sigmoid(2z) - 1, so the factor of two is folded into
    # sqrt(2 / pi). This is the tanh approximation used by nn.GELU.
    gelu = x * tl.sigmoid(
        1.5957691216057308 * (x + 0.044715 * x * x * x)
    )

    # Masked lanes hold GELU(0) == 0 and cannot raise the non-negative amax.
    scale = tl.max(tl.abs(gelu), axis=0) / 448.0 + 1e-12
    tl.store(Xs + row, scale)
    tl.store(
        Xq + row * K + offs,
        (gelu / scale).to(tl.float8e4nv),
        mask=mask,
    )


def gelu_quantize_activation_rowwise(x2: torch.Tensor):
    """Fuse tanh-GELU with row-wise FP8 quantization for a 2-D tensor.

    Returns ``(x_fp8, x_scale)``, or ``None`` when the row is too wide to keep
    resident so the caller can fall back to the unfused path.
    """
    M, K = x2.shape
    block_k = triton.next_power_of_2(K)
    if block_k > _FUSED_GELU_MAX_BLOCK:
        return None
    x_fp8 = torch.empty((M, K), device=x2.device, dtype=_FP8_DTYPE)
    x_scale = torch.empty((M, 1), device=x2.device, dtype=torch.float32)
    _gelu_quant_rowwise_kernel[(M,)](
        x2,
        x_fp8,
        x_scale,
        K,
        x2.stride(0),
        BLOCK_K=block_k,
        num_warps=8,
    )
    return x_fp8, x_scale


def _prequant_weight(module: nn.Module, linear: nn.Linear):
    w_fp8, w_scale_t = quantize_weight_rowwise(linear.weight)
    module.register_buffer("weight_fp8", w_fp8)
    module.register_buffer("w_scale_t", w_scale_t)  # [1, out]
    if linear.bias is not None:
        module.register_buffer("bias", linear.bias.detach().clone())
    else:
        module.bias = None
    module.out_features = linear.out_features


class FP8Linear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        _prequant_weight(self, linear)
        self.in_features = linear.in_features
        self.train(linear.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("FP8Linear is inference-only; call eval() under no_grad()")
        x2 = x.reshape(-1, x.shape[-1])
        x_fp8, x_scale = quantize_activation_rowwise(x2)
        y = torch._scaled_mm(
            x_fp8,
            self.weight_fp8.t(),
            scale_a=x_scale,
            scale_b=self.w_scale_t,
            bias=self.bias,
            out_dtype=x.dtype,
            use_fast_accum=True,
        )
        return y.reshape(*x.shape[:-1], self.out_features)


class FP8LinearFusedGELU(FP8Linear):
    """FP8 linear that absorbs the preceding tanh-approximation GELU.

    The installer must replace the original ``nn.GELU`` with ``nn.Identity``
    so the activation is applied exactly once.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("FP8Linear is inference-only; call eval() under no_grad()")
        x2 = x.reshape(-1, x.shape[-1])
        fused = gelu_quantize_activation_rowwise(x2)
        if fused is None:
            x_fp8, x_scale = quantize_activation_rowwise(
                nn.functional.gelu(x2, approximate="tanh")
            )
        else:
            x_fp8, x_scale = fused
        y = torch._scaled_mm(
            x_fp8,
            self.weight_fp8.t(),
            scale_a=x_scale,
            scale_b=self.w_scale_t,
            bias=self.bias,
            out_dtype=x.dtype,
            use_fast_accum=True,
        )
        return y.reshape(*x.shape[:-1], self.out_features)


@torch.no_grad()
def prequantize_qkv(attn: nn.Module) -> bool:
    """Replace the q/k/v projections of a self-attention with ``FP8Linear``.

    The fused QKV+RoPE kernel reads the prequantized buffers directly and shares
    one activation quantization across all three GEMMs. If fused kernels are
    disabled, ``FP8Linear.forward`` remains the inference fallback. Replacing
    the modules also releases the original BF16 weights. Returns whether the
    conversion was applied.
    """
    if not all(isinstance(getattr(attn, n, None), nn.Linear) for n in ("q", "k", "v")):
        return False
    for name in ("q", "k", "v"):
        linear = getattr(attn, name)
        setattr(attn, name, FP8Linear(linear))
    attn._fp8_qkv = True
    return True


def fp8_qkv_weights(attn: nn.Module):
    """Return {name: (w_fp8, w_scale_t)} for q/k/v, or None if not prequantized."""
    if not getattr(attn, "_fp8_qkv", False):
        return None
    return {
        name: (getattr(attn, name).weight_fp8, getattr(attn, name).w_scale_t)
        for name in ("q", "k", "v")
    }


@torch.no_grad()
def convert_dit_linears_to_fp8(
    net: nn.Module,
    *,
    include_attention: bool = True,
    quant_device: torch.device | str | None = None,
) -> int:
    """FP8-convert the largest GEMMs in every transformer block.

    Always converts the two FFN GEMMs. With ``include_attention`` (default) it
    also converts the attention output projection and the cross-attention
    projections to :class:`FP8Linear`, and replaces the self-attention q/k/v
    projections with :class:`FP8Linear` for both fused and eager inference.

    A CPU-resident model is staged one transformer block at a time through
    ``quant_device``. Each converted block is moved back to its source device
    before the next block is staged, so the full BF16 model never has to reside
    on the GPU at once. Returns the number of GEMMs converted.
    """
    if net.training:
        raise RuntimeError("FP8 conversion is inference-only; call net.eval() first")
    first_param = next(net.parameters())
    if first_param.device.type not in ("cpu", "cuda"):
        raise RuntimeError(
            "FP8 conversion requires a CPU- or CUDA-resident model; "
            f"got {first_param.device}"
        )
    if quant_device is None:
        quant_device = (
            first_param.device
            if first_param.device.type == "cuda"
            else torch.device("cuda", torch.cuda.current_device())
        )
    else:
        quant_device = torch.device(quant_device)
    if quant_device.type == "cuda" and quant_device.index is None:
        quant_device = torch.device("cuda", torch.cuda.current_device())
    _require_fp8_gemm(quant_device)

    n = 0
    for block_index, block in enumerate(net.blocks):
        block_param = next(block.parameters())
        source_device = block_param.device
        if source_device.type not in ("cpu", "cuda"):
            raise RuntimeError(
                f"block {block_index} must reside on CPU or CUDA; got {source_device}"
            )
        needs_staging = source_device != quant_device
        if needs_staging:
            block.to(quant_device)

        try:
            ffn = block.ffn
            if not isinstance(ffn[0], nn.Linear) or not isinstance(ffn[2], nn.Linear):
                raise TypeError(
                    f"block {block_index} FFN has already been replaced or is unsupported"
                )
            ffn[0] = FP8Linear(ffn[0])
            n += 1
            activation = ffn[1]
            if isinstance(activation, nn.GELU) and activation.approximate == "tanh":
                ffn[2] = FP8LinearFusedGELU(ffn[2])
                ffn[1] = nn.Identity()
            else:
                ffn[2] = FP8Linear(ffn[2])
            n += 1

            if include_attention:
                self_attn = block.self_attn
                if isinstance(self_attn.o, nn.Linear):
                    self_attn.o = FP8Linear(self_attn.o)
                    n += 1
                if prequantize_qkv(self_attn):
                    n += 3

                cross_attn = getattr(block, "cross_attn", None)
                if cross_attn is not None:
                    for name in ("q", "k", "v", "o"):
                        proj = getattr(cross_attn, name, None)
                        if isinstance(proj, nn.Linear):
                            setattr(cross_attn, name, FP8Linear(proj))
                            n += 1
        finally:
            if needs_staging:
                block.to(source_device)
    return n
