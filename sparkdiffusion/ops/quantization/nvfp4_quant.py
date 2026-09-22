"""
Standalone W8A8 Quantized Linear Layer.

Supports: fp8 / int8 / fp8_block / nvfp4  ×  w8a8 / w8a16
GPU support:
  - nvfp4  : SM100+ (B200/GB200) only — cuDNN FP4 GEMM
  - fp8    : SM89+ (RTX 4090, H100, B200)
  - int8   : SM80+ (A100 and above)
  - w8a16  : any GPU, weight-only compression, dequant before GEMM

Usage:
    from sparkdiffusion.ops.quantization.nvfp4_quant import replace_linear_with_quantized

    # B200
    replace_linear_with_quantized(net, quant_type="nvfp4")
    # H100 / 4090
    replace_linear_with_quantized(net, quant_type="fp8")
    # A100 or any GPU (memory only)
    replace_linear_with_quantized(net, quant_type="fp8", quant_mode="w8a16")
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional

# ── Optional backends — all fail-soft ────────────────────────────────────────
try:
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
        cutlass_fp8_supported,
    )
    VLLM_AVAILABLE = True
except Exception:
    ops = None

    def cutlass_fp8_supported():
        return False

    VLLM_AVAILABLE = False

try:
    from vllm._custom_ops import fusedQuantizeNv as vllm_fusedQuantizeNv
    from vllm.model_executor.layers.quantization.qutlass_utils import (
        to_blocked as vllm_to_blocked,
    )
    VLLM_HADAMARD_AVAILABLE = True
except Exception:
    vllm_fusedQuantizeNv = None
    vllm_to_blocked = None
    VLLM_HADAMARD_AVAILABLE = False

try:
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
    FLASHINFER_AVAILABLE = True
except Exception:
    SfLayout = None
    mm_fp4 = None
    nvfp4_quantize = None
    FLASHINFER_AVAILABLE = False

try:
    from qutlass import fusedQuantizeNv as qutlass_fusedQuantizeNv
    from qutlass import matmul_nvf4_bf16_tn
    from qutlass.utils import to_blocked as qutlass_to_blocked
    QUTLASS_AVAILABLE = True
except Exception:
    qutlass_fusedQuantizeNv = None
    matmul_nvf4_bf16_tn = None
    qutlass_to_blocked = None
    QUTLASS_AVAILABLE = False

try:
    from compressed_tensors.transform.utils.hadamard import (
        deterministic_hadamard_matrix,
    )
    HADAMARD_AVAILABLE = True
except Exception:
    deterministic_hadamard_matrix = None
    HADAMARD_AVAILABLE = False

# fp8/int8 helpers use the local implementations below.
#
# fp8_block requires quantize_subchannel / fp8_gemm / quantize_block_trans.
# No --quant_type choice offers it, so nothing reachable from an entrypoint uses
# these; they stay None. nvfp4/fp8/int8 go through vllm ops and are unaffected.
fp8_gemm = None
quantize_block_trans = None
quantize_subchannel = None


def fused_dequant_weight(weight: torch.Tensor, scale: torch.Tensor,
                         out_dtype: torch.dtype) -> torch.Tensor:
    """per-channel dequant: weight [N,K] * scale [N,1] → out_dtype."""
    return (weight.float() * scale.float()).to(out_dtype)

def apply_fp8_linear(input: torch.Tensor, weight: torch.Tensor,
                     weight_scale: torch.Tensor, bias=None,
                     cutlass_fp8_supported: bool = False,
                     use_per_token_if_dynamic: bool = True) -> torch.Tensor:
    """FP8 linear: try torch._scaled_mm (SM89+), fallback to dequant + bf16 GEMM."""
    out_dtype = input.dtype
    if hasattr(torch, "_scaled_mm"):
        try:
            x2d = input.view(-1, input.shape[-1])
            x_scale = x2d.float().abs().amax() / 448.0 + 1e-12
            x_fp8 = (x2d / x_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            out = torch._scaled_mm(
                x_fp8, weight,  # weight already transposed [in, out]
                scale_a=x_scale.to(torch.float32),
                scale_b=weight_scale.squeeze().to(torch.float32) if weight_scale.numel() == 1
                        else weight_scale.to(torch.float32),
                out_dtype=out_dtype,
            )
            if bias is not None:
                out = out + bias
            return out.view(*input.shape[:-1], weight.shape[-1])
        except Exception:
            pass
    # Dequant fallback (any GPU)
    w_fp = fused_dequant_weight(weight.t(), weight_scale, out_dtype)
    return nn.functional.linear(input, w_fp, bias)

def apply_int8_linear(input: torch.Tensor, weight: torch.Tensor,
                      weight_scale: torch.Tensor, bias=None) -> torch.Tensor:
    """INT8 linear: try vllm cutlass, fallback to dequant + bf16 GEMM."""
    if VLLM_AVAILABLE:
        try:
            x2d = input.view(-1, input.shape[-1])
            qx, xs, _ = ops.scaled_int8_quant(x2d)
            out = ops.cutlass_scaled_mm(
                qx, weight, scale_a=xs, scale_b=weight_scale,
                out_dtype=input.dtype, bias=bias,
            )
            return out.view(*input.shape[:-1], weight.shape[-1])
        except Exception:
            pass
    w_fp = fused_dequant_weight(weight.t(), weight_scale, input.dtype)
    return nn.functional.linear(input, w_fp, bias)

def block_dequant(weight: torch.Tensor, scale: torch.Tensor,
                  block_size: list, out_dtype: torch.dtype,
                  transpose_scale: bool = False) -> torch.Tensor:
    """Block-wise dequant (fp8_block fallback)."""
    bn, bk = block_size
    n, k = weight.shape
    if transpose_scale:
        scale = scale.t().contiguous()
    x = weight.float()
    s = scale.float().repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)
    return (x * s[:n, :k]).to(out_dtype)


# ── Minimal runtime state ───────────────────────────────────────────────────
class _RuntimeState:
    """Thread-local-ish state for per-step quantization control."""
    skip_quant: bool = False
    step_counter: int = 0

_runtime_state = _RuntimeState()

def get_runtime_state() -> _RuntimeState:
    return _runtime_state


# ── Core quantized linear layer ───────────────────────────────────────────────

class W8A8Linear(nn.Module):
    """
    Inference quantized linear layer.  Drop-in replacement for nn.Linear.

    Args:
        layer         : original nn.Linear to quantize
        quant_type    : 'fp8' | 'int8' | 'fp8_block' | 'nvfp4'
        quant_mode    : 'w8a8' (default) | 'w8a16' (weight-only, no GEMM speedup)
        out_dtype     : output tensor dtype (bfloat16 recommended)
        global_scale  : global activation scale for nvfp4 (default 1.0)
        quant_step_range : only quantize when step in this range (None = always)
        use_hadamard  : apply Hadamard transform before nvfp4 quantization
        hadamard_group_size : group size for Hadamard (16/32/64/128)
        quant_gemm_backend  : 'vllm' | 'flashinfer' | 'qutlass'
        packed_weight : pack int8/fp8 quantized weights into bf16 tensors (FSDP2 compat)
        svd_rank      : SVDQuant low-rank correction rank (0 = disabled)
    """

    def __init__(
        self,
        layer: nn.Linear,
        quant_type: str = "fp8",
        quant_mode: str = "w8a8",
        out_dtype: torch.dtype = torch.bfloat16,
        global_scale: float = 1.0,
        quant_step_range: Optional[range] = None,
        use_hadamard: bool = False,
        hadamard_group_size: int = 128,
        quant_gemm_backend: str = "vllm",
        packed_weight: bool = False,
        svd_rank: int = 0,
    ) -> None:
        super().__init__()

        if quant_type not in ("fp8", "int8", "fp8_block", "nvfp4"):
            raise ValueError(f"quant_type must be fp8/int8/fp8_block/nvfp4, got {quant_type!r}")
        if quant_mode not in ("w8a8", "w8a16"):
            raise ValueError(f"quant_mode must be w8a8/w8a16, got {quant_mode!r}")
        if hadamard_group_size not in (16, 32, 64, 128):
            raise ValueError(f"hadamard_group_size must be 16/32/64/128, got {hadamard_group_size}")

        self.layer = layer
        self.quant_type = quant_type
        self.quant_mode = quant_mode
        self.out_dtype = out_dtype
        self.quant_step_range = quant_step_range
        self.use_hadamard = use_hadamard
        self.hadamard_group_size = hadamard_group_size
        self.quant_gemm_backend = quant_gemm_backend
        self.packed_weight = packed_weight
        self.is_quantized = False
        self.has_svd = False
        self.cutlass_fp8_supported = cutlass_fp8_supported() if VLLM_AVAILABLE else False
        self.transpose_scale = True if quant_type == "fp8_block" else None
        self.weight_channel_wise = True
        self.input_token_wise = True

        if quant_type == "nvfp4":
            self.FLOAT4_E2M1_MAX = 6.0
            self.FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
            self.global_scale_activation = torch.tensor(
                [global_scale], dtype=torch.float32,
                device=layer.weight.device if layer.weight.device.type != "meta" else "cpu",
            )
            if use_hadamard and HADAMARD_AVAILABLE:
                self.hadamard = deterministic_hadamard_matrix(
                    hadamard_group_size, dtype=torch.bfloat16,
                    device=layer.weight.device,
                ) * (hadamard_group_size ** -0.5)

        # Register weight_scale placeholder
        self.register_buffer("weight_scale", torch.empty(1))

        # SVDQuant buffers
        if svd_rank > 0 and quant_type == "nvfp4":
            out_f, in_f = layer.weight.shape
            self.svd_L = nn.Parameter(
                torch.empty(out_f, svd_rank, dtype=out_dtype), requires_grad=False)
            self.svd_R = nn.Parameter(
                torch.empty(svd_rank, in_f, dtype=out_dtype), requires_grad=False)

        self.process_weights_after_loading()

    def process_weights_after_loading(self) -> None:
        """Quantize weights. Call once after all weights are loaded."""
        weight = self.layer.weight
        if weight.device.type == "meta":
            return  # skip on meta device
        qweight, scale = self._quantize(weight.data)

        if self.packed_weight and self.quant_type in ("fp8", "int8"):
            self.out_features, self.in_features = qweight.shape
            qweight = qweight.view(torch.bfloat16)
        elif self.quant_type == "nvfp4":
            self.out_features, self.in_features = weight.shape

        self.layer.weight = nn.Parameter(qweight, requires_grad=False)
        del self.weight_scale
        self.register_buffer("weight_scale", scale)
        self.is_quantized = True

    @torch._dynamo.disable()
    def forward(self, x: torch.Tensor, input_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert self.is_quantized, "Call process_weights_after_loading() first"
        rs = get_runtime_state()
        skip = rs.skip_quant or (
            self.quant_step_range is not None and rs.step_counter not in self.quant_step_range
        )
        if self.quant_mode == "w8a16" or skip:
            w = self.layer.weight
            if self.packed_weight and self.quant_type in ("fp8", "int8"):
                tgt = torch.float8_e4m3fn if self.quant_type == "fp8" else torch.int8
                w = w.view(tgt)
            dw = self._dequant_weight(w.to(self.out_dtype), self.weight_scale).to(self.out_dtype)
            return nn.functional.linear(x, dw, self.layer.bias)

        if input_scale is None:
            x = x.to(self.out_dtype)
        x2d = x.view(-1, x.shape[-1]) if x.dim() > 2 else x
        out_shape = list(x.shape)
        return self._apply_quantized_linear(x2d, out_shape, input_scale=input_scale)

    def _apply_quantized_linear(
        self, x2d: torch.Tensor, out_shape: list,
        input_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        w = self.layer.weight

        if self.quant_type == "fp8":
            fp8w = w.view(torch.float8_e4m3fn).t() if self.packed_weight else w.to(torch.float8_e4m3fn).t()
            out_shape[-1] = fp8w.shape[-1]
            if input_scale is not None:
                return ops.cutlass_scaled_mm(
                    x2d, fp8w, out_dtype=self.out_dtype,
                    scale_a=input_scale, scale_b=self.weight_scale, bias=self.layer.bias,
                ).reshape(out_shape)
            return apply_fp8_linear(
                input=x2d, weight=fp8w, weight_scale=self.weight_scale,
                bias=self.layer.bias, cutlass_fp8_supported=self.cutlass_fp8_supported,
                use_per_token_if_dynamic=self.input_token_wise,
            ).reshape(out_shape)

        elif self.quant_type == "int8":
            int8w = w.view(torch.int8).t() if self.packed_weight else w.to(torch.int8).t()
            out_shape[-1] = int8w.shape[-1]
            if input_scale is not None:
                return ops.cutlass_scaled_mm(
                    x2d, int8w, out_dtype=self.out_dtype,
                    scale_a=input_scale, scale_b=self.weight_scale, bias=self.layer.bias,
                ).reshape(out_shape)
            return apply_int8_linear(
                input=x2d, weight=int8w, weight_scale=self.weight_scale, bias=self.layer.bias,
            ).reshape(out_shape)

        elif self.quant_type == "fp8_block":
            fp8w = w.to(torch.float8_e4m3fn)
            out_shape[-1] = fp8w.shape[0]
            output = torch.empty(out_shape, device=fp8w.device, dtype=self.out_dtype)
            if input_scale is not None:
                qx, sx = x2d, input_scale
            else:
                qx, sx = quantize_subchannel(x2d, transpose_scale=self.transpose_scale)
            fp8_gemm(
                fp8_a=qx, a_scale=sx, fp8_b=fp8w, b_scale=self.weight_scale,
                output=output, transpose_scale=self.transpose_scale,
                workspace=torch.zeros(200000, dtype=torch.int32, device=fp8w.device),
            )
            return output

        elif self.quant_type == "nvfp4":
            w_u8 = w.view(torch.uint8)
            # Quantize activation
            if input_scale is not None:
                input_fp4, input_scales = x2d, input_scale
            elif self.use_hadamard and VLLM_HADAMARD_AVAILABLE:
                fn = vllm_fusedQuantizeNv or qutlass_fusedQuantizeNv
                input_fp4, hf = fn(x2d, self.hadamard, self.global_scale_activation)
                to_b = vllm_to_blocked or qutlass_to_blocked
                input_scales = to_b(hf).view(-1, self.in_features // 16)
            elif self.quant_gemm_backend == "vllm":
                input_fp4, input_scales = ops.scaled_fp4_quant(
                    x2d, input_global_scale=self.global_scale_activation)
            elif self.quant_gemm_backend == "flashinfer":
                input_fp4, input_scales = nvfp4_quantize(
                    x2d, self.global_scale_activation,
                    sfLayout=SfLayout.layout_128x4, do_shuffle=False,
                )
            elif self.quant_gemm_backend == "qutlass":
                raise RuntimeError("qutlass has no standalone quant kernel without Hadamard")
            else:
                raise RuntimeError(f"Unsupported quant_gemm_backend: {self.quant_gemm_backend}")

            alpha = self.global_weight_scale

            if self.quant_gemm_backend == "vllm":
                output = ops.cutlass_scaled_fp4_mm(
                    input_fp4, w_u8, input_scales,
                    self.weight_scale.view(torch.float8_e4m3fn), alpha, self.out_dtype,
                )
            elif self.quant_gemm_backend == "flashinfer":
                out_shape[-1] = w_u8.shape[0]
                output = torch.empty(out_shape, device=x2d.device, dtype=self.out_dtype)
                mm_fp4(
                    input_fp4, w_u8.t(),
                    input_scales.view(torch.uint8),
                    self.weight_scale.view(torch.uint8).t(),
                    alpha, self.out_dtype, output,
                    block_size=16, use_8x4_sf_layout=False,
                    backend="cudnn", use_nvfp4=True, skip_check=True,
                )
            elif self.quant_gemm_backend == "qutlass":
                output = matmul_nvf4_bf16_tn(
                    input_fp4, w_u8, input_scales,
                    self.weight_scale.view(torch.float8_e4m3fn), alpha,
                )
            else:
                raise RuntimeError(f"Unsupported quant_gemm_backend: {self.quant_gemm_backend}")

            # SVDQuant low-rank correction
            if self.has_svd:
                output = output + (x2d @ self.svd_R.t()) @ self.svd_L.t()

            return output

        raise RuntimeError(f"Unsupported quant_type: {self.quant_type}")

    def _quantize(self, weight: torch.Tensor):
        dev = weight.device
        if dev.type == "cpu":
            weight = weight.cuda()
        if self.quant_type == "fp8":
            qw, sw = ops.scaled_fp8_quant(
                weight, use_per_token_if_dynamic=self.weight_channel_wise)
        elif self.quant_type == "int8":
            qw, sw, _ = ops.scaled_int8_quant(weight)
        elif self.quant_type == "fp8_block":
            qw, sw, _, _ = quantize_block_trans(weight, transpose_scale=self.transpose_scale)
        elif self.quant_type == "nvfp4":
            qw, sw = ops.scaled_fp4_quant(
                weight, input_global_scale=self.global_weight_scale)
        else:
            raise RuntimeError(f"Unsupported quant_type: {self.quant_type}")
        if dev.type == "cpu":
            qw = qw.to(dev)
        return qw, sw

    def _dequant_weight(self, weight, scale):
        if self.quant_type == "fp8_block":
            return block_dequant(weight, scale, [128, 128],
                                 self.out_dtype, self.transpose_scale).to(self.out_dtype)
        return fused_dequant_weight(weight, scale, self.out_dtype)

    @property
    def global_weight_scale(self):
        """Lazy-init buffer for nvfp4 global weight scale."""
        if not hasattr(self, "_global_weight_scale"):
            self.register_buffer(
                "_global_weight_scale",
                torch.tensor([1.0], dtype=torch.float32, device=self.layer.weight.device),
            )
        return self._global_weight_scale


# ── Replacement helper ────────────────────────────────────────────────────────

def _set_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def replace_linear_with_quantized(
    model: nn.Module,
    quant_type: str = "fp8",
    quant_mode: str = "w8a8",
    target_modules: Optional[List[str]] = None,
    path_contains: Optional[List[str]] = None,
    min_dim: int = 16,
    out_dtype: torch.dtype = torch.bfloat16,
    quant_gemm_backend: str = "vllm",
    **kwargs,
) -> nn.Module:
    """
    Replace nn.Linear layers in *model* with W8A8Linear (quantized).

    Filtering (OR logic — a layer is quantized if ANY condition matches):
        target_modules : list of last-path-component names to match.
                         e.g. ["q", "k", "v", "o"] for native WanModel.
        path_contains  : list of substrings to match against the full dotted path.
                         e.g. ["ffn"] matches "blocks.0.ffn.0" and "blocks.0.ffn.2".
        If both are None → quantize ALL nn.Linear layers.

    Args:
        model            : model to modify in-place
        quant_type       : 'fp8' | 'int8' | 'fp8_block' | 'nvfp4'
        quant_mode       : 'w8a8' (quantize both W and A) | 'w8a16' (W-only, any GPU)
        target_modules   : filter by last name component
        path_contains    : filter by full path substring
        min_dim          : skip layers where any dim < min_dim (default 16)
        out_dtype        : output dtype (bfloat16 recommended)
        quant_gemm_backend : 'vllm' | 'flashinfer' | 'qutlass'
        **kwargs         : forwarded to W8A8Linear (e.g. svd_rank, use_hadamard)

    Returns:
        model (modified in-place)
    """
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        last = name.split(".")[-1]
        if target_modules is not None or path_contains is not None:
            matched = False
            if target_modules is not None and last in target_modules:
                matched = True
            if path_contains is not None and any(s in name for s in path_contains):
                matched = True
            if not matched:
                continue
        if module.in_features % min_dim != 0 or module.out_features % min_dim != 0:
            print(f"[quant] skip {name}: dims not divisible by {min_dim} "
                  f"(in={module.in_features}, out={module.out_features})")
            continue
        replacements.append((name, module))

    for name, module in replacements:
        quant_layer = W8A8Linear(
            layer=module,
            quant_type=quant_type,
            quant_mode=quant_mode,
            out_dtype=out_dtype,
            quant_gemm_backend=quant_gemm_backend,
            **kwargs,
        )
        _set_module(model, name, quant_layer)
        print(f"[quant] {name}: {module.in_features}→{module.out_features} → {quant_type}/{quant_mode}")

    print(f"[quant] replaced {len(replacements)} Linear layers with W8A8Linear ({quant_type}/{quant_mode})")
    return model
