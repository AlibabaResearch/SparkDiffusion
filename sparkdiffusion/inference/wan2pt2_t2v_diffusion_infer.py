#!/usr/bin/env python
"""
Wan 2.2 T2V MoE diffusion-style CFG inference.

Loads two DiT experts (high-noise + low-noise) and switches between them
at boundary_ratio based on timestep, matching the official Wan 2.2 pipeline.
"""
from tqdm import tqdm
import argparse
from sparkdiffusion.inference.sampling_utils import positive_int, sample_output_path
import os
import re
import time
import unicodedata

import torch

from imaginaire.utils.io import save_image_or_video
from imaginaire.lazy_config import LazyCall as L, LazyDict, instantiate
from imaginaire.utils import log

from sparkdiffusion.datasets.utils import VIDEO_RES_SIZE_INFO
from sparkdiffusion.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from sparkdiffusion.utils.model_utils import init_weights_on_device, load_checkpoint_auto
from sparkdiffusion.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from sparkdiffusion.networks.wan2pt2 import WanModel
from sparkdiffusion.samplers.euler import FlowEulerSampler
from sparkdiffusion.samplers.unipc import FlowUniPCMultistepSampler
from sparkdiffusion.utils.inference_optimization import COMPILE_MODE, optimize_model_for_inference
from sparkdiffusion.utils.selective_activation_checkpoint import CheckpointMode, SACConfig

_DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
_DEFAULT_PROMPT = "A cat playing in the garden under the sun."

tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}


def _sanitize_filename(text: str, max_len: int = 50) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\s/\\:*?\"<>|]+", "_", text)
    text = re.sub(r"[^\w\-]", "", text, flags=re.UNICODE)
    return text[:max_len].rstrip("_")


WAN2PT2_A14B_T2V: LazyDict = L(WanModel)(
    dim=5120, eps=1e-06, ffn_dim=13824, freq_dim=256, in_dim=16,
    model_type="t2v", num_heads=40, num_layers=40, out_dim=16, text_len=512,
)

WAN2PT2_A14B_T2V_ROLA: LazyDict = L(WanModel)(
    dim=5120, eps=1e-06, ffn_dim=13824, freq_dim=256, in_dim=16,
    model_type="t2v", num_heads=40, num_layers=40, out_dim=16, text_len=512,
    use_rola_attn=True, rola_topk_ratio=0.1, rola_rank=64, rola_blkq=64, rola_blkk=64,
)

WAN2PT2_A14B_T2V_PURE_SLA: LazyDict = L(WanModel)(
    dim=5120, eps=1e-06, ffn_dim=13824, freq_dim=256, in_dim=16,
    model_type="t2v", num_heads=40, num_layers=40, out_dim=16, text_len=512,
    use_pure_sla_attn=True, pure_sla_topk_ratio=0.1,
    pure_sla_blkq=64, pure_sla_blkk=64, pure_sla_feature_map="softmax",
)

dit_configs = {
    "A14B": WAN2PT2_A14B_T2V,
    "A14B_rola": WAN2PT2_A14B_T2V_ROLA,
    "A14B_pure_sla": WAN2PT2_A14B_T2V_PURE_SLA,
}


def _build_inference_config(base_config: LazyDict, attn_precision: str, use_fused_kernels: bool,
                            rola_topk_ratio: float = None) -> LazyDict:
    """Return an inference-only config without mutating the shared config."""
    import copy

    cfg = copy.deepcopy(base_config)
    cfg.rola_attn_precision = attn_precision
    cfg.use_fused_inference = use_fused_kernels
    cfg.sac_config = SACConfig(mode=CheckpointMode.NONE)
    if rola_topk_ratio is not None:
        # top-k is a ratio, so sparsity = 1 - ratio (0.1 -> 90%, 0.05 -> 95%)
        if cfg.get("use_pure_sla_attn", False):
            cfg.pure_sla_topk_ratio = rola_topk_ratio
        else:
            cfg.rola_topk_ratio = rola_topk_ratio
    return cfg


def _load_expert(ckpt_path: str, name: str, model_config: LazyDict, args) -> torch.nn.Module:
    """Load one MoE expert, optionally FP8-quantize and torch.compile it.

    The optimized expert is left on CPU (device-only move that preserves FP8
    buffers); the sampling loop moves the active expert onto the GPU so two
    14B experts never need to be resident at once.
    """
    with init_weights_on_device():
        net = instantiate(model_config).eval()
    load_checkpoint_auto(ckpt_path, net)
    net.to(device="cpu", dtype=tensor_kwargs["dtype"])
    torch.cuda.empty_cache()

    net, quantized_gemms = optimize_model_for_inference(
        net,
        quant_type=args.quant_type,
        quant_mode=args.quant_mode,
        quant_backend=args.quant_backend,
        device=tensor_kwargs["device"],
    )
    if args.quant_type:
        scope = "FFN and attention projection GEMMs" if args.quant_type == "fp8" else "FFN GEMMs"
        log.info(f"[{name}] Quantization applied to {quantized_gemms} {scope}: {args.quant_type}/{args.quant_mode}")
    log.info(f"[{name}] torch.compile enabled ({COMPILE_MODE})")
    net.cpu()  # device-only move; preserves FP8 buffers
    torch.cuda.empty_cache()
    log.success(f"Loaded {name} expert from {ckpt_path}")
    return net


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan 2.2 T2V MoE Diffusion Inference")
    parser.add_argument("--dit_path", type=str, required=True,
                        help="Path to high-noise expert checkpoint")
    parser.add_argument("--dit_path_2", type=str, default="",
                        help="Path to low-noise expert checkpoint (if empty, use single-model mode)")
    parser.add_argument("--model_size", choices=["A14B", "A14B_rola", "A14B_pure_sla"], default="A14B_rola",
                        help="Model variant")
    parser.add_argument("--quant_type", type=str, default="", choices=["", "fp8", "int8", "nvfp4"])
    parser.add_argument("--quant_mode", type=str, default="w8a8", choices=["w8a8", "w8a16"])
    parser.add_argument("--quant_backend", type=str, default="vllm", choices=["vllm", "flashinfer", "qutlass"])
    parser.add_argument("--attn_precision", type=str, default="bf16", choices=["bf16", "int8"])
    parser.add_argument("--disable_fused_kernels", action="store_true",
                        help="Use the original attention implementation instead of inference-only Triton kernels.")
    parser.add_argument("--rola_topk_ratio", type=float, default=None,
                        help="Override the RoLa top-k ratio; sparsity = 1 - ratio (0.1 -> 90%%, 0.05 -> 95%%). "
                             "Default keeps the value declared in the model config.")
    parser.add_argument("--boundary_ratio", type=float, default=0.875,
                        help="RF-domain boundary between high/low noise experts")
    parser.add_argument("--num_samples", type=positive_int, default=1, help="Sequential samples per prompt (batch size 1); seeds start at --seed")
    parser.add_argument("--num_steps", type=int, default=40,
                        help="Official Wan2.2 A14B default sampling steps")
    parser.add_argument("--sigma_max", type=float, default=0.999,
                        help="Official UniPC raw sigma_max before timestep shift")
    parser.add_argument("--sampler", choices=["Euler", "UniPC"], default="UniPC")
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                        help="CFG scale for low-noise expert (official: 3.0)")
    parser.add_argument("--guidance_scale_high", type=float, default=4.0,
                        help="CFG scale for high-noise expert (official: 4.0)")
    parser.add_argument("--timestep_shift", type=float, default=12.0,
                        help="Official Wan2.2 T2V A14B timestep shift")
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--text_encoder_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--prompt", type=str, default=_DEFAULT_PROMPT,
                        help="Text prompt for this inference case")
    parser.add_argument("--negative_prompt", type=str, default=_DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--resolution", default="480p", type=str)
    parser.add_argument("--aspect_ratio", default="16:9", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="outputs/distill/generated_video.mp4")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # This entrypoint intentionally evaluates one prompt per process.
    prompts = [args.prompt]

    # Output directory
    if args.save_path.lower().endswith(".mp4"):
        save_dir = os.path.dirname(args.save_path) or "."
    else:
        save_dir = args.save_path
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load MoE experts
    # ------------------------------------------------------------------
    has_moe = bool(args.dit_path_2)
    _cfg = _build_inference_config(
        dit_configs[args.model_size],
        args.attn_precision,
        use_fused_kernels=not args.disable_fused_kernels,
        rola_topk_ratio=args.rola_topk_ratio,
    )
    if args.model_size.endswith("_rola"):
        log.info(f"RoLa attn_precision: {args.attn_precision}")
        # LazyCall configs expose the model's signature defaults, so read the
        # field belonging to the variant that is actually enabled.
        _field = "pure_sla_topk_ratio" if _cfg.get("use_pure_sla_attn", False) else "rola_topk_ratio"
        _topk = _cfg.get(_field)
        log.info(f"RoLa {_field}: {_topk} (sparsity {100 * (1 - _topk):.0f}%)")
    log.info(f"Inference Triton kernels: {'disabled' if args.disable_fused_kernels else 'enabled'}")
    net_high = _load_expert(args.dit_path, "high-noise", _cfg, args)
    net_low = _load_expert(args.dit_path_2, "low-noise", _cfg, args) if has_moe else None

    # VAE
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]

    # Encode prompts
    log.info("Computing the prompt embedding...")
    all_text_embs = (
        get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=prompts,
                           tokenizer_path=args.tokenizer_path)
        .to(dtype=torch.bfloat16).cuda()
    )
    neg_text_emb = (
        get_umt5_embedding(checkpoint_path=args.text_encoder_path,
                           prompts=args.negative_prompt,
                           tokenizer_path=args.tokenizer_path)
        .to(dtype=torch.bfloat16).cuda()
    )
    clear_umt5_memory()

    # Sampler
    samplers = {"Euler": FlowEulerSampler, "UniPC": FlowUniPCMultistepSampler}
    sampler = samplers[args.sampler](
        num_train_timesteps=1000, sigma_max=args.sigma_max, sigma_min=0.0,
    )
    sampler.set_timesteps(
        num_inference_steps=args.num_steps, device=tensor_kwargs["device"],
        shift=args.timestep_shift,
    )

    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(args.num_frames),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]

    # Only the currently-active expert stays on the GPU; two 14B experts do not
    # fit at once. The loop moves the low-noise expert in (and high out) at the
    # boundary switch.
    net_high.cuda()

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    for sample_idx in range(args.num_samples):
        p_idx, prompt = 0, prompts[0]
        sample_seed = args.seed + sample_idx
        phase = "warmup (may include compilation/autotuning)" if sample_idx == 0 else "after warmup"
        sample_label = f"[sample {sample_idx + 1}/{args.num_samples}, seed={sample_seed}, {phase}]"
        log.info(sample_label)
        output_path = sample_output_path(args.save_path, sample_idx, args.num_samples, sample_seed)
        log.info(f"[{p_idx + 1}/{len(prompts)}] {prompt[:80]}")

        sampler.set_timesteps(
            num_inference_steps=args.num_steps, device=tensor_kwargs["device"],
            shift=args.timestep_shift,
        )

        text_emb = all_text_embs[p_idx:p_idx + 1]
        condition = {"crossattn_emb": text_emb.to(**tensor_kwargs)}
        uncondition = {"crossattn_emb": neg_text_emb.to(**tensor_kwargs)}

        generator = torch.Generator(device=tensor_kwargs["device"])
        generator.manual_seed(sample_seed)

        x = torch.randn(1, *state_shape, dtype=torch.float32,
                        device=tensor_kwargs["device"], generator=generator)
        ones = torch.ones(1, device=tensor_kwargs["device"]).float()

        torch.cuda.synchronize()
        denoise_start = time.perf_counter()
        n_high, n_low = 0, 0

        # Wan 2.2 official: boundary = 0.875 * num_train_timesteps (1000) = 875
        # Official scheduler timesteps = sigmas * 1000 → t ∈ [0, 1000]
        # Our distilled sampler also: timesteps = sigmas * num_train_timesteps → same domain
        boundary_scaled = args.boundary_ratio * 1000  # = 875
        log.info(f"MoE boundary: t >= {boundary_scaled:.0f} → high-noise (ratio={args.boundary_ratio})")

        last_expert = None
        for i, t in enumerate(tqdm(sampler.timesteps, desc="Sampling", leave=False)):
            t_scalar = t.item() if isinstance(t, torch.Tensor) else t
            use_high = (t_scalar >= boundary_scaled) if has_moe else True

            # Log expert switch
            if use_high != last_expert:
                expert_name = "HIGH" if use_high else "LOW "
                log.info(f"  step {i:3d}: t={t_scalar:.4f} → {expert_name}-noise expert")
                last_expert = use_high
                # Keep only the active expert on the GPU (two 14B don't fit).
                if has_moe and net_low is not None:
                    if use_high:
                        net_low.cpu()
                        net_high.cuda()
                    else:
                        net_high.cpu()
                        net_low.cuda()
                    torch.cuda.empty_cache()

            net = net_high if use_high else net_low
            if use_high:
                n_high += 1
            else:
                n_low += 1

            timesteps = (t * ones).unsqueeze(1)  # [B, 1]
            with torch.no_grad():
                v_cond = net(
                    x_B_C_T_H_W=x.to(**tensor_kwargs),
                    timesteps_B_T=timesteps.to(**tensor_kwargs),
                    **condition,
                ).float()
                v_uncond = net(
                    x_B_C_T_H_W=x.to(**tensor_kwargs),
                    timesteps_B_T=timesteps.to(**tensor_kwargs),
                    **uncondition,
                ).float()

            cfg = args.guidance_scale_high if use_high else args.guidance_scale
            v_pred = v_uncond + cfg * (v_cond - v_uncond)
            x = sampler.step(v_pred, t, x)

        torch.cuda.synchronize()
        denoise_end = time.perf_counter()
        log.info(f"{sample_label} denoising time: {denoise_end - denoise_start:.2f}s "
                 f"(high-noise: {n_high} steps, low-noise: {n_low} steps)")

        samples = x.float()

        # Decode
        video = tokenizer.decode(samples.to("cuda"))
        save_image_or_video(video[0], output_path, fps=16)
        log.success(f"Saved: {output_path}")
        # Do not retain the previous decoded video during the next sample.
        del video, samples, x

    log.success(f"Done! Generated {args.num_samples} samples for one prompt -> {save_dir}")
