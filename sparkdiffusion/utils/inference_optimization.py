"""Inference-only model preparation shared by Wan entry points."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch._inductor import config as inductor_config


COMPILE_MODE = "default"


def apply_inference_quantization(
    model: nn.Module,
    *,
    quant_type: str,
    quant_mode: str = "w8a8",
    quant_backend: str = "vllm",
    quant_device: torch.device | None = None,
) -> int:
    """Apply enabled inference quantization and return the converted GEMM count.

    The optimized FP8 route converts the two large FFN GEMMs and all self- and
    cross-attention Q/K/V/O projection GEMMs per block. The QK-softmax-PV
    attention operator itself remains BF16.
    """
    if quant_type == "fp8":
        if quant_mode != "w8a8":
            raise ValueError("The compiler-friendly FP8 path supports only --quant_mode w8a8")
        from sparkdiffusion.ops.quantization.fp8_linear import convert_dit_linears_to_fp8

        return convert_dit_linears_to_fp8(
            model,
            quant_device=quant_device,
        )

    # Preserve the pre-existing INT8/NVFP4 legacy options, but keep them
    # FFN-only. Only the BF16 and compiler-friendly FP8 routes are actively
    # maintained; FP8 runs on any GPU with FP8 tensor cores (SM89+).
    from sparkdiffusion.ops.quantization.nvfp4_quant import (
        W8A8Linear,
        replace_linear_with_quantized,
    )

    before = sum(1 for module in model.modules() if isinstance(module, W8A8Linear))
    replace_linear_with_quantized(
        model,
        quant_type=quant_type,
        quant_mode=quant_mode,
        quant_gemm_backend=quant_backend,
        target_modules=[],
        path_contains=["ffn"],
    )
    after = sum(1 for module in model.modules() if isinstance(module, W8A8Linear))
    return after - before


def optimize_model_for_inference(
    model: nn.Module,
    *,
    quant_type: str = "",
    quant_mode: str = "w8a8",
    quant_backend: str = "vllm",
    device: torch.device | str | None = None,
) -> tuple[nn.Module, int]:
    """Quantize (optionally), move to CUDA, then compile an inference model.

    For FP8, a CPU-resident model is quantized one layer at a time using
    ``device`` as a staging GPU. Each compressed weight is returned to CPU
    immediately, so the full BF16 model never has to fit on the GPU before
    quantization. Legacy quantizers retain their existing CUDA-resident flow.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Inference optimization requires CUDA")
    target_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda", torch.cuda.current_device())
    )
    if target_device.type != "cuda":
        raise ValueError(f"inference device must be CUDA, got {target_device}")
    if target_device.index is None:
        target_device = torch.device("cuda", torch.cuda.current_device())

    first_parameter = next(model.parameters())
    if first_parameter.device.type not in ("cpu", "cuda"):
        raise RuntimeError(
            "Inference optimization requires a CPU- or CUDA-resident model, "
            f"got {first_parameter.device}"
        )
    if model.training:
        raise RuntimeError("Inference optimization requires model.eval()")
    converted_gemms = 0
    with torch.cuda.device(target_device):
        if quant_type and quant_type != "fp8":
            model.to(target_device)

        if quant_type:
            converted_gemms = apply_inference_quantization(
                model,
                quant_type=quant_type,
                quant_mode=quant_mode,
                quant_backend=quant_backend,
                quant_device=target_device,
            )
            # Release FP8 per-layer staging allocations before moving the
            # compressed model and its remaining BF16 weights to the target.
            if quant_type == "fp8" and first_parameter.device.type == "cpu":
                torch.cuda.empty_cache()

        if not quant_type or quant_type == "fp8":
            model.to(target_device)

        if quant_type == "fp8":
            # Autotuning benchmarks ~100 generated Triton matmuls per FP8 GEMM
            # shape and then selects ATen's _scaled_mm regardless: cuBLASLt /
            # CUTLASS wins the core GEMM, and the hand-written Triton epilogues
            # leave a template nothing to fuse. Skip a search with no candidate
            # that can win. Only the small residual BF16 GEMMs pay for this.
            inductor_config.max_autotune_gemm_backends = "ATEN"
    model.eval()  # also covers newly attached legacy quantized submodules
    compiled = torch.compile(model, mode=COMPILE_MODE, dynamic=False)
    return compiled, converted_gemms
