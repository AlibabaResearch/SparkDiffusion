# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Few-step distilled inference script for Wan 2.2 T2V with dual-student (high-noise + low-noise).

Follows the official Wan 2.2 dual-model pattern:
  - timesteps are in RF domain
  - boundary_ratio = 0.875 separates high-noise and low-noise models
  - high-noise student handles rf_t >= boundary
  - low-noise student handles rf_t < boundary
"""

import argparse
from sparkdiffusion.inference.sampling_utils import positive_int, sample_output_path
import os
import re
import unicodedata
import time

import torch
from einops import rearrange
from tqdm import tqdm

from imaginaire.utils.io import save_image_or_video
from imaginaire.lazy_config import LazyCall as L, LazyDict, instantiate
from imaginaire.utils import log

from sparkdiffusion.datasets.utils import VIDEO_RES_SIZE_INFO
from sparkdiffusion.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from sparkdiffusion.utils.model_utils import init_weights_on_device, load_checkpoint_auto
from sparkdiffusion.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from sparkdiffusion.networks.wan2pt2 import WanModel
from sparkdiffusion.utils.inference_optimization import COMPILE_MODE, optimize_model_for_inference
from sparkdiffusion.utils.selective_activation_checkpoint import CheckpointMode, SACConfig

torch._dynamo.config.suppress_errors = True

tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}

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

_DEFAULT_PROMPT = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."


def _sanitize_filename(text: str, max_len: int = 50) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\s/\\:*?\"<>|]+", "_", text)
    text = re.sub(r"[^\w\-]", "", text, flags=re.UNICODE)
    return text[:max_len].rstrip("_")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-step distilled inference for Wan2.2 T2V dual-student")
    parser.add_argument("--model_size", choices=["A14B", "A14B_rola", "A14B_pure_sla"], default="A14B_rola",
                        help="Model variant")
    parser.add_argument("--high_noise_model_path", type=str, required=True,
                        help="Path to the high-noise student checkpoint")
    parser.add_argument("--low_noise_model_path", type=str, required=True,
                        help="Path to the low-noise student checkpoint")
    parser.add_argument("--boundary", type=float, default=0.875,
                        help="RF-domain boundary for switching high/low noise students (official Wan2.2 t2v boundary, trig≈1.4289)")
    parser.add_argument("--num_samples", type=positive_int, default=1, help="Sequential samples per prompt (batch size 1); seeds start at --seed")
    parser.add_argument("--num_steps_high", type=int, default=2,
                        help="Sampling steps for high-noise student (covers [pi/2, boundary])")
    parser.add_argument("--num_steps_low", type=int, default=2,
                        help="Sampling steps for low-noise student (covers [boundary, 0])")
    parser.add_argument("--sigma_max", type=float, default=1600, help="Initial sigma for the distilled sampler")
    parser.add_argument("--vae_path", type=str, default="")
    parser.add_argument("--text_encoder_path", type=str,
                        default="")
    parser.add_argument("--tokenizer_path", type=str,
                        default="",
                        help="Local path to the umt5 tokenizer directory")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--prompt", type=str, default=_DEFAULT_PROMPT,
                        help="Text prompt for this inference case")
    parser.add_argument("--resolution", default="480p", type=str)
    parser.add_argument("--aspect_ratio", default="16:9", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="outputs/distill/generated_video.mp4")
    parser.add_argument("--t_scaling_factor", type=float, default=1000.0)
    parser.add_argument("--quant_type", type=str, default="", choices=["", "fp8", "int8", "nvfp4"])
    parser.add_argument("--quant_mode", type=str, default="w8a8", choices=["w8a8", "w8a16"])
    parser.add_argument("--quant_backend", type=str, default="vllm", choices=["vllm", "flashinfer", "qutlass"])
    parser.add_argument("--attn_precision", type=str, default="bf16", choices=["bf16", "int8"])
    parser.add_argument("--disable_fused_kernels", action="store_true",
                        help="Use the original attention implementation instead of inference-only Triton kernels.")
    parser.add_argument("--rola_topk_ratio", type=float, default=None,
                        help="Override the RoLa top-k ratio; sparsity = 1 - ratio (0.1 -> 90%%, 0.05 -> 95%%). "
                             "Default keeps the value declared in the model config.")
    return parser.parse_args()


def _load_model(model_path, model_config, args):
    with init_weights_on_device():
        net = instantiate(model_config).eval()
    load_checkpoint_auto(model_path, net)
    # Keep on CPU until per-layer FP8 quantization stages each block on the GPU,
    # so the full BF16 model never has to fit on the card at once.
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
        log.info(f"Quantization applied to {quantized_gemms} {scope}: {args.quant_type}/{args.quant_mode}")
    log.info(f"torch.compile enabled ({COMPILE_MODE})")
    return net


if __name__ == "__main__":
    args = parse_arguments()
    # This entrypoint intentionally evaluates one prompt per process.
    prompts = [args.prompt]
    save_dir = os.path.dirname(args.save_path) or "output"
    os.makedirs(save_dir, exist_ok=True)

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

    log.info("Loading high-noise student...")
    high_noise_model = _load_model(args.high_noise_model_path, _cfg, args)
    high_noise_model.cpu()  # device-only move; preserves FP8 buffers
    torch.cuda.empty_cache()
    log.success(f"Loaded high-noise student from {args.high_noise_model_path}")

    log.info("Loading low-noise student...")
    low_noise_model = _load_model(args.low_noise_model_path, _cfg, args)
    low_noise_model.cpu()  # device-only move; preserves FP8 buffers
    torch.cuda.empty_cache()
    log.success(f"Loaded low-noise student from {args.low_noise_model_path}")

    torch.cuda.empty_cache()

    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]

    log.info("Computing the prompt embedding...")
    all_text_embs = get_umt5_embedding(
        checkpoint_path=args.text_encoder_path, prompts=prompts, tokenizer_path=args.tokenizer_path
    ).to(dtype=torch.bfloat16).cuda()
    clear_umt5_memory()

    # Build timestep schedule directly in the rf domain:
    #   High-noise: [rf_start, ..mid_t_high.., boundary]
    #   Low-noise:  [boundary, ..mid_t_low.., 0]
    rf_start = args.sigma_max / (args.sigma_max + 1.0)

    # rf-domain mid knots (legacy TrigFlow knots 1.5/1.0/1.2/0.8/0.4 pre-converted
    # via rf = sin(t)/(cos(t)+sin(t))).
    MID_T_HIGH = {1: [], 2: [0.933781]}
    MID_T_LOW = {1: [], 2: [0.608979], 3: [0.720057, 0.507301], 4: [0.720057, 0.507301, 0.297157]}

    mid_t_high = MID_T_HIGH.get(args.num_steps_high, [0.933781])
    mid_t_low = MID_T_LOW.get(args.num_steps_low, [0.720057, 0.507301, 0.297157])

    t_steps_high_rf = torch.tensor(
        [rf_start] + mid_t_high + [args.boundary],
        dtype=torch.float64, device="cuda",
    )
    t_steps_low_rf = torch.tensor(
        [args.boundary] + mid_t_low + [0.0],
        dtype=torch.float64, device="cuda",
    )

    log.info(f"High-noise steps (RF): {t_steps_high_rf.tolist()}")
    log.info(f"Low-noise steps (RF): {t_steps_low_rf.tolist()}")
    log.info(f"Boundary (RF): {args.boundary}")

    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(args.num_frames),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]

    # Generate the requested inference case.
    for sample_idx in range(args.num_samples):
        p_idx, prompt = 0, prompts[0]
        sample_seed = args.seed + sample_idx
        phase = "warmup (may include compilation/autotuning)" if sample_idx == 0 else "after warmup"
        sample_label = f"[sample {sample_idx + 1}/{args.num_samples}, seed={sample_seed}, {phase}]"
        log.info(sample_label)
        output_path = sample_output_path(args.save_path, sample_idx, args.num_samples, sample_seed)
        log.info(f"[{p_idx + 1}/{len(prompts)}] {prompt[:80]}")

        text_emb = all_text_embs[p_idx : p_idx + 1]
        condition = {"crossattn_emb": text_emb.to(**tensor_kwargs)}

        generator = torch.Generator(device=tensor_kwargs["device"])
        generator.manual_seed(sample_seed)

        init_noise = torch.randn(
            1, *state_shape,
            dtype=torch.float32, device=tensor_kwargs["device"], generator=generator,
        )

        x = init_noise.to(torch.float64) * t_steps_high_rf[0]
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)

        torch.cuda.synchronize()
        denoise_start = time.perf_counter()

        # Phase 1: High-noise student
        high_noise_model.cuda()
        for i, (t_cur, t_next) in enumerate(zip(t_steps_high_rf[:-1], t_steps_high_rf[1:])):
            with torch.no_grad():
                v_pred = high_noise_model(
                    x_B_C_T_H_W=x.to(**tensor_kwargs),
                    timesteps_B_T=(t_cur.float() * ones * args.t_scaling_factor).to(**tensor_kwargs),
                    **condition,
                ).to(torch.float64)
                x = x + (t_next - t_cur) * v_pred
        high_noise_model.cpu()

        # Phase 2: Low-noise student (x is now at boundary)
        low_noise_model.cuda()
        for i, (t_cur, t_next) in enumerate(zip(t_steps_low_rf[:-1], t_steps_low_rf[1:])):
            with torch.no_grad():
                v_pred = low_noise_model(
                    x_B_C_T_H_W=x.to(**tensor_kwargs),
                    timesteps_B_T=(t_cur.float() * ones * args.t_scaling_factor).to(**tensor_kwargs),
                    **condition,
                ).to(torch.float64)
                x = x + (t_next - t_cur) * v_pred
        low_noise_model.cpu()

        torch.cuda.synchronize()
        denoise_end = time.perf_counter()
        log.info(f"{sample_label} denoising time: {denoise_end - denoise_start:.2f}s "
                 f"({args.num_steps_high}+{args.num_steps_low} steps)")

        samples = x.float()
        video = tokenizer.decode(samples)
        video = (1.0 + video.float().cpu().clamp(-1, 1)) / 2.0

        to_show = video.unsqueeze(0)
        save_image_or_video(
            rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
            output_path, fps=16,
        )
        log.info(f"Saved: {output_path}")
        # Do not retain the previous decoded video during the next sample.
        del video, samples, x, to_show
