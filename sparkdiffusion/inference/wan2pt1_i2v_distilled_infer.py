#!/usr/bin/env python
"""
Wan 2.1 I2V RoLa few-step (distilled) inference.

Single model (no MoE, no CFG), with online CLIP image feature extraction.
Combines:
  - image / y / CLIP conditioning from wan2pt1_i2v_diffusion_infer.py
  - the few-step rf-Euler sampler from wan2pt1_t2v_distilled_infer.py

The sampler is aligned with the training-time
`T2VDistillModel.generate_samples_from_batch`, so it must be used to
evaluate the distilled student, NOT the 40-step CFG diffusion script.

Usage mirrors wan2pt1_t2v_distilled_infer.py; the extra required arguments are
--image_path and --clip_encoder_path.
"""
import argparse
import os
import time

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from einops import rearrange, repeat
from tqdm import tqdm

from imaginaire.utils.io import save_image_or_video
from imaginaire.lazy_config import LazyCall as L, LazyDict, instantiate
from imaginaire.utils import log

from sparkdiffusion.datasets.utils import VIDEO_RES_SIZE_INFO
from sparkdiffusion.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from sparkdiffusion.utils.inference_optimization import COMPILE_MODE, optimize_model_for_inference
from sparkdiffusion.utils.model_utils import init_weights_on_device, load_checkpoint_auto
from sparkdiffusion.utils.selective_activation_checkpoint import CheckpointMode, SACConfig
from sparkdiffusion.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from sparkdiffusion.networks.wan2pt1 import WanModel
from sparkdiffusion.utils.clip_image_encoder import WanCLIPImageEncoder

torch._dynamo.config.suppress_errors = True

# in_dim=36 (16 noisy latent + 20 i2v condition), model_type=i2v (CLIP cross-attn
# branch). Only 14B exists for I2V: there is no 1.3B and no pure-SLA i2v config.
WAN2PT1_14B_I2V: LazyDict = L(WanModel)(
    dim=5120, eps=1e-06, ffn_dim=13824, freq_dim=256,
    in_dim=36, model_type="i2v", num_heads=40, num_layers=40, out_dim=16, text_len=512,
)

WAN2PT1_14B_I2V_ROLA: LazyDict = L(WanModel)(
    dim=5120, eps=1e-06, ffn_dim=13824, freq_dim=256,
    in_dim=36, model_type="i2v", num_heads=40, num_layers=40, out_dim=16, text_len=512,
    use_rola_attn=True, rola_topk_ratio=0.1, rola_rank=64, rola_blkq=64, rola_blkk=64,
)

# Placeholder. model_type only selects cross_attn_type/img_emb while the attention
# implementation only replaces self-attention, so i2v + pure-SLA is a valid combo.
# Two things are missing before it can run: the external sparse_linear_attention
# package (imported by WanSelfAttentionPureSLA), and a pure-SLA-distilled i2v
# checkpoint — the RoLa students carry proj_q/proj_k/gate_proj/gate_bias, which
# PureSLA has no slot for, so loading one here silently drops that branch.
WAN2PT1_14B_I2V_PURE_SLA: LazyDict = L(WanModel)(
    dim=5120, eps=1e-06, ffn_dim=13824, freq_dim=256,
    in_dim=36, model_type="i2v", num_heads=40, num_layers=40, out_dim=16, text_len=512,
    use_pure_sla_attn=True, pure_sla_topk_ratio=0.1,
    pure_sla_blkq=64, pure_sla_blkk=64, pure_sla_feature_map="softmax",
)

dit_configs = {
    "14B": WAN2PT1_14B_I2V,
    "14B_rola": WAN2PT1_14B_I2V_ROLA,
    "14B_pure_sla": WAN2PT1_14B_I2V_PURE_SLA,
}


def _build_inference_config(
    base_config: LazyDict,
    attn_precision: str,
    use_fused_kernels: bool,
    rola_topk_ratio: float = None,
) -> LazyDict:
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


tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}

_DEFAULT_PROMPT = "A cat playing in the garden under the sun."

# few-step distilled schedule (rf domain), identical to training backward_timesteps
# and to wan2pt1_t2v_distilled_infer.py. Legacy TrigFlow knots pre-converted via rf = sin/(cos+sin).
MID_T_BY_NSTEPS = {
    1: [],
    2: [0.933781],
    3: [0.933781, 0.852895],
    4: [0.933781, 0.852895, 0.608979],
    8: [0.933781, 0.852895, 0.782708, 0.690833, 0.583052, 0.457197, 0.297157],
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-step distilled inference script for Wan2.1 I2V (single model, CLIP)")
    parser.add_argument("--model_size", choices=["14B", "14B_rola", "14B_pure_sla"], default="14B_rola",
                        help="Model variant: dense / rola / pure_sla. I2V has no 1.3B. "
                             "pure_sla is a placeholder: it needs the external sparse_linear_attention "
                             "package and there is no pure-SLA-distilled i2v checkpoint.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4, 8], default=4,
                        help="1~4 or 8 for timestep-distilled inference")
    parser.add_argument("--sigma_max", type=float, default=1600.0, help="Initial sigma for the distilled sampler (matches training config)")
    parser.add_argument("--dit_path", type=str, required=True,
                        help="Path to the distilled I2V DiT checkpoint (.pth or DCP dir)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input reference image")
    parser.add_argument("--clip_encoder_path", type=str, required=True, help="Path to the CLIP image_encoder directory")
    parser.add_argument("--vae_path", type=str, default="", help="Path to the Wan2.1 VAE.")
    parser.add_argument(
        "--text_encoder_path", type=str, default="",
        help="Path to the umT5 text encoder."
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default="",
        help="Local path to umt5 tokenizer directory"
    )
    parser.add_argument("--num_frames", type=int, default=81, help="Pixel frames. state_t=21 -> 81 frames.")
    parser.add_argument("--prompt", type=str, default=_DEFAULT_PROMPT,
                        help="Text prompt for this inference case")
    parser.add_argument("--resolution", default="480p", type=str, help="Resolution of the generated output")
    parser.add_argument("--aspect_ratio", default="16:9", type=str,
                        help="Aspect ratio of the generated output (width:height)")
    parser.add_argument("--fixed_resolution", action="store_true",
                        help="Use fixed resolution/aspect_ratio instead of official I2V input-aspect sizing.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_path", type=str, default="outputs/distill/generated_video.mp4",
        help="Output .mp4 path for this inference case."
    )
    parser.add_argument(
        "--t_scaling_factor", type=float, default=1000.0,
        help="rectified_flow_t_scaling_factor used to feed timesteps to the net.",
    )
    parser.add_argument(
        "--quant_type", type=str, default="", choices=["", "fp8", "int8", "nvfp4"],
        help="Linear layer quantization: "
             "empty=BF16 and fp8=compiler-friendly FFN+attention-Q/K/V/O W8A8 FP8 "
             "(any GPU with FP8 tensor cores, SM89+); int8/nvfp4 are pre-existing "
             "legacy backends."
    )
    parser.add_argument(
        "--quant_mode", type=str, default="w8a8", choices=["w8a8", "w8a16"],
        help="w8a8: weights+activations (required by the optimized FP8 path). "
             "w8a16 is retained for legacy INT8/NVFP4 backends."
    )
    parser.add_argument(
        "--quant_backend", type=str, default="vllm", choices=["vllm", "flashinfer", "qutlass"],
        help="GEMM backend for legacy INT8/NVFP4 inference; FP8 uses torch._scaled_mm."
    )
    parser.add_argument(
        "--attn_precision", type=str, default="bf16", choices=["bf16", "int8"],
        help="Sparse attention kernel precision for RoLa models: "
             "bf16=default fused path (SM80+), int8=legacy unfused SLA path. "
             "Only effective with --model_size *_rola."
    )
    parser.add_argument(
        "--disable_fused_kernels",
        action="store_true",
        help="Use the original attention implementation instead of inference-only Triton kernels.",
    )
    parser.add_argument("--rola_topk_ratio", type=float, default=None,
                        help="Override the RoLa top-k ratio; sparsity = 1 - ratio (0.1 -> 90%%, 0.05 -> 95%%). "
                             "Default keeps the value declared in the model config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # This entrypoint intentionally evaluates one prompt per process.
    prompts = [args.prompt]
    # --- Determine output directory ---
    save_dir = os.path.dirname(args.save_path) or "output"
    os.makedirs(save_dir, exist_ok=True)

    # --- Load model (once) ---
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
    log.info(
        "Inference Triton kernels: "
        f"{'disabled' if args.disable_fused_kernels else 'enabled'}"
    )

    with init_weights_on_device():
        net = instantiate(_cfg).eval()

    load_checkpoint_auto(args.dit_path, net)
    log.success(f"Successfully loaded DiT from {args.dit_path}")

    # Keep the checkpoint on CPU until optional per-layer quantization has
    # released the large BF16 weights. Moving to CUDA and immediately back to
    # CPU defeats offload and can OOM before quantization starts.
    net.to(device="cpu", dtype=tensor_kwargs["dtype"])
    torch.cuda.empty_cache()

    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)

    # --- Image preprocessing ---
    log.info(f"Loading image: {args.image_path}")
    input_image = Image.open(args.image_path).convert("RGB")
    if args.fixed_resolution:
        w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
        log.info(f"Fixed resolution set to: {w}x{h}")
    else:
        base_w, base_h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
        max_area = base_w * base_h
        orig_w, orig_h = input_image.size
        image_aspect_ratio = orig_h / orig_w
        stride = tokenizer.spatial_compression_factor * 2
        h = int(np.sqrt(max_area * image_aspect_ratio) // stride * stride)
        w = int(np.sqrt(max_area / image_aspect_ratio) // stride * stride)
        log.info(f"Official I2V input-aspect sizing: input={orig_w}x{orig_h}, output={w}x{h}")
    image_transforms = T.Compose([
        T.ToImage(),
        T.Resize(size=(h, w), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image_tensor = image_transforms(input_image).unsqueeze(0).to(**tensor_kwargs)  # [1,3,h,w] in [-1,1]
    num_pixel_frames = args.num_frames

    # --- Build y = cat([mask(4ch), VAE_encode([first_frame, zeros(F-1)])(16ch)]) ---
    with torch.no_grad():
        frames_to_encode = torch.cat(
            [image_tensor.unsqueeze(2),
             torch.zeros(1, 3, num_pixel_frames - 1, h, w, device=image_tensor.device)],
            dim=2,
        )
        encoded_latents = tokenizer.encode(frames_to_encode)
    lat_t = tokenizer.get_latent_num_frames(num_pixel_frames)
    lat_h = h // tokenizer.spatial_compression_factor
    lat_w = w // tokenizer.spatial_compression_factor

    msk = torch.zeros(1, 4, lat_t, lat_h, lat_w, device=tensor_kwargs["device"], dtype=tensor_kwargs["dtype"])
    msk[:, :, 0, :, :] = 1.0
    y = torch.cat([msk, encoded_latents.to(**tensor_kwargs)], dim=1)  # [1, 20, T, H, W]
    y = y.repeat(args.num_samples, 1, 1, 1, 1)

    # --- CLIP first-frame feature for cross-attention ---
    log.info(f"Loading CLIP encoder from {args.clip_encoder_path}")
    clip_encoder = WanCLIPImageEncoder(args.clip_encoder_path, dtype=torch.float16, device="cuda")
    frame_cond = clip_encoder.encode_first_frame(image_tensor).to(dtype=torch.bfloat16)  # [1, 257, 1280]
    frame_cond = frame_cond.repeat(args.num_samples, 1, 1)
    # The DiT needs every spare byte at 720p, so drop CLIP once its feature is out.
    del clip_encoder
    torch.cuda.empty_cache()

    # Encode the single prompt once.
    log.info("Computing the prompt embedding...")
    all_text_embs = get_umt5_embedding(
        checkpoint_path=args.text_encoder_path, prompts=prompts, tokenizer_path=args.tokenizer_path
    ).to(dtype=torch.bfloat16).cuda()
    clear_umt5_memory()

    # --- few-step distilled schedule (rf domain) ---
    mid_t = MID_T_BY_NSTEPS[args.num_steps]
    log.info(f"num_steps={args.num_steps}  mid_t(rf)={mid_t}")
    rf_start = args.sigma_max / (args.sigma_max + 1.0)
    t_steps = torch.tensor([rf_start, *mid_t, 0.0], dtype=torch.float64, device="cuda")
    total_steps = t_steps.shape[0] - 1

    state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]

    net, quantized_gemms = optimize_model_for_inference(
        net,
        quant_type=args.quant_type,
        quant_mode=args.quant_mode,
        quant_backend=args.quant_backend,
        device=tensor_kwargs["device"],
    )
    if args.quant_type:
        quantized_scope = (
            "FFN and attention projection GEMMs"
            if args.quant_type == "fp8"
            else "FFN GEMMs"
        )
        log.info(
            f"Quantization applied to {quantized_gemms} {quantized_scope}: "
            f"{args.quant_type}/{args.quant_mode}"
        )
    log.info(f"torch.compile enabled ({COMPILE_MODE})")

    # Generate the requested inference case.
    for p_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        log.info(f"[{p_idx + 1}/{len(prompts)}] {prompt[:80]}")

        text_emb = all_text_embs[p_idx : p_idx + 1]  # [1, L, D]
        condition = {
            "crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples),
            "y_B_C_T_H_W": y,
            "frame_cond_crossattn_emb_B_L_D": frame_cond,
        }

        generator = torch.Generator(device=tensor_kwargs["device"])
        generator.manual_seed(args.seed)

        init_noise = torch.randn(
            args.num_samples,
            *state_shape,
            dtype=torch.float32,
            device=tensor_kwargs["device"],
            generator=generator,
        )

        x = init_noise.to(torch.float64)
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)

        torch.cuda.synchronize()
        denoise_start = time.perf_counter()
        x = x * t_steps[0]
        for t_cur, t_next in tqdm(list(zip(t_steps[:-1], t_steps[1:])), desc="Sampling", total=total_steps):
            with torch.no_grad():
                v_pred = net(
                    x_B_C_T_H_W=x.to(**tensor_kwargs),
                    timesteps_B_T=(t_cur.float() * ones * args.t_scaling_factor).to(**tensor_kwargs),
                    **condition,
                ).to(torch.float64)
                x = x + (t_next - t_cur) * v_pred

        torch.cuda.synchronize()
        denoise_end = time.perf_counter()
        log.info(f"[prompt{p_idx}] denoising time: {denoise_end - denoise_start:.2f}s")

        samples = x.float()
        video = tokenizer.decode(samples)
        video = (1.0 + video.float().cpu().clamp(-1, 1)) / 2.0  # [B, C, T, H, W]

        to_show = video.unsqueeze(0)  # [1, B, C, T, H, W]
        save_image_or_video(
            rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
            args.save_path,
            fps=16,
        )
        log.info(f"Saved: {args.save_path}")

    log.success(f"Done! Generated one prompt -> {args.save_path}")
