from tqdm import tqdm
import argparse
import os
import torch
from einops import repeat
from PIL import Image
import torchvision.transforms.v2 as T
import numpy as np

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

_DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
_DEFAULT_PROMPT = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."

tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}

WAN2PT2_A14B_I2V: LazyDict = L(WanModel)(
    dim=5120, eps=1e-06, ffn_dim=13824, freq_dim=256,
    in_dim=36, model_type="i2v", num_heads=40, num_layers=40, out_dim=16, text_len=512,
)

WAN2PT2_A14B_I2V_ROLA: LazyDict = L(WanModel)(
    dim=5120, eps=1e-06, ffn_dim=13824, freq_dim=256,
    in_dim=36, model_type="i2v", num_heads=40, num_layers=40, out_dim=16, text_len=512,
    use_rola_attn=True, rola_topk_ratio=0.1, rola_rank=64, rola_blkq=64, rola_blkk=64,
)

dit_configs = {"A14B": WAN2PT2_A14B_I2V, "A14B_rola": WAN2PT2_A14B_I2V_ROLA}


def load_dit_model(model_path, model_config):
    """Instantiates, loads state dict, and moves a DiT model to the correct device."""
    with init_weights_on_device():
        model = instantiate(model_config).eval()
    load_checkpoint_auto(model_path, model)
    log.success(f"Successfully loaded DiT from {model_path}")
    return model


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diffusion inference script for Wan2.2 I2V with High/Low Noise models")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image for I2V generation.")
    parser.add_argument(
        "--high_noise_model_path", type=str, default="pretrain_weights/Wan2.2-I2V-A14B-high.pth", help="Path to the high-noise model."
    )
    parser.add_argument("--low_noise_model_path", type=str, default="", help="Path to the low-noise model. Empty = single-expert high-noise mode.")
    parser.add_argument("--boundary", type=float, default=0.9, help="Timestep boundary for switching from high to low noise model.")

    parser.add_argument("--model_size", choices=["A14B", "A14B_rola"], default="A14B_rola", help="A14B = dense, A14B_rola = RoLa sparse attention")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, default=40, help="Official Wan2.2 I2V default sampling steps")
    parser.add_argument("--sigma_max", type=float, default=0.999, help="Official UniPC raw sigma_max before timestep shift.")
    parser.add_argument("--sampler", choices=["Euler", "UniPC"], default="UniPC", help="Sampler")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Official Wan2.2 I2V CFG scale for both experts")
    parser.add_argument("--timestep_shift", type=float, default=5.0, help="Timestep shift as in Wan")
    parser.add_argument("--vae_path", type=str, default="", help="Path to the Wan2.1 VAE.")
    parser.add_argument(
        "--text_encoder_path", type=str, default="", help="Path to the umT5 text encoder."
    )
    parser.add_argument(
        "--tokenizer_path", type=str,
        default="",
        help="Path to the umT5 tokenizer directory."
    )
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--prompt", type=str, default=_DEFAULT_PROMPT, help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, default=_DEFAULT_NEGATIVE_PROMPT, help="Negative text prompt for video generation")
    parser.add_argument("--resolution", default="720p", type=str, help="Resolution of the generated output")
    parser.add_argument("--aspect_ratio", default="16:9", type=str, help="Aspect ratio of the generated output (width:height)")
    parser.add_argument(
        "--adaptive_resolution",
        action="store_true",
        help="If set, adapts the output resolution to the input image's aspect ratio, "
        "using the area defined by --resolution and --aspect_ratio as a target.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_path", type=str, default="outputs/distill/generated_video.mp4", help="Path to save the generated video (include file extension)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    model_config = dit_configs[args.model_size]
    high_noise_model = load_dit_model(args.high_noise_model_path, model_config).to(**tensor_kwargs).cpu()
    low_noise_model = (
        load_dit_model(args.low_noise_model_path, model_config).to(**tensor_kwargs).cpu()
        if args.low_noise_model_path
        else None
    )
    torch.cuda.empty_cache()

    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)

    log.info(f"Loading and preprocessing image from: {args.image_path}")
    input_image = Image.open(args.image_path).convert("RGB")
    if args.adaptive_resolution:
        log.info("Adaptive resolution mode enabled.")
        base_w, base_h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
        max_resolution_area = base_w * base_h
        log.info(f"Target area is based on {args.resolution} {args.aspect_ratio} (~{max_resolution_area} pixels).")

        orig_w, orig_h = input_image.size
        image_aspect_ratio = orig_h / orig_w

        stride = tokenizer.spatial_compression_factor * 2
        h = int(np.sqrt(max_resolution_area * image_aspect_ratio) // stride * stride)
        w = int(np.sqrt(max_resolution_area / image_aspect_ratio) // stride * stride)

        log.info(f"Input image aspect ratio: {image_aspect_ratio:.4f}. Adaptive resolution set to: {w}x{h}")
    else:
        log.info("Fixed resolution mode.")
        w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
        log.info(f"Resolution set to: {w}x{h}")
    F = args.num_frames
    lat_h = h // tokenizer.spatial_compression_factor
    lat_w = w // tokenizer.spatial_compression_factor
    lat_t = tokenizer.get_latent_num_frames(F)

    log.info(f"Preprocessing image to {w}x{h}...")
    image_transforms = T.Compose(
        [
            T.ToImage(),
            T.Resize(size=(h, w), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image_tensor = image_transforms(input_image).unsqueeze(0).to(device=tensor_kwargs["device"], dtype=torch.float32)

    with torch.no_grad():
        frames_to_encode = torch.cat(
            [image_tensor.unsqueeze(2), torch.zeros(1, 3, F - 1, h, w, device=image_tensor.device)], dim=2
        )  # -> B, C, T, H, W
        encoded_latents = tokenizer.encode(frames_to_encode)  # -> B, C_lat, T_lat, H_lat, W_lat

    msk = torch.zeros(1, 4, lat_t, lat_h, lat_w, device=tensor_kwargs["device"], dtype=tensor_kwargs["dtype"])
    msk[:, :, 0, :, :] = 1.0

    y = torch.cat([msk, encoded_latents.to(**tensor_kwargs)], dim=1)
    y = y.repeat(args.num_samples, 1, 1, 1, 1)

    log.info(f"Computing embedding for prompt: {args.prompt}")
    text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.prompt, tokenizer_path=args.tokenizer_path).to(dtype=torch.bfloat16).cuda()
    neg_text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.negative_prompt, tokenizer_path=args.tokenizer_path).to(dtype=torch.bfloat16).cuda()
    clear_umt5_memory()

    log.info(f"Generating with prompt: {args.prompt}")
    condition = {"crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples), "y_B_C_T_H_W": y}
    uncondition = {"crossattn_emb": repeat(neg_text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples), "y_B_C_T_H_W": y}

    to_show = []

    state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]

    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(args.seed)

    init_noise = torch.randn(
        args.num_samples,
        *state_shape,
        dtype=torch.float32,
        device=tensor_kwargs["device"],
        generator=generator,
    )

    x = init_noise

    samplers = {"Euler": FlowEulerSampler, "UniPC": FlowUniPCMultistepSampler}
    sampler = samplers[args.sampler](num_train_timesteps=1000, sigma_max=args.sigma_max, sigma_min=0.0)
    sampler.set_timesteps(num_inference_steps=args.num_steps, device=tensor_kwargs["device"], shift=args.timestep_shift)

    # log.info(sampler.timesteps)

    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    high_noise_model.cuda()
    net = high_noise_model
    switched = False
    for _, t in enumerate(tqdm(sampler.timesteps)):
        if low_noise_model is not None and t.item() < args.boundary * 1000 and not switched:
            high_noise_model.cpu()
            low_noise_model.cuda()
            net = low_noise_model
            switched = True
            log.info("Switched to low noise model.")
        timesteps = t * ones

        with torch.no_grad():
            v_cond = net(x_B_C_T_H_W=x.to(**tensor_kwargs), timesteps_B_T=timesteps.to(**tensor_kwargs), **condition).float()
            v_uncond = net(x_B_C_T_H_W=x.to(**tensor_kwargs), timesteps_B_T=timesteps.to(**tensor_kwargs), **uncondition).float()

        v_pred = v_uncond + args.guidance_scale * (v_cond - v_uncond)

        x = sampler.step(v_pred, t, x)

    samples = x.float()
    video = tokenizer.decode(samples)  # [B, C, T, H, W] in [-1, 1]

    # Save each sample as individual video
    save_dir = args.save_path if not args.save_path.endswith('.mp4') else os.path.dirname(args.save_path)
    os.makedirs(save_dir, exist_ok=True)
    for s_idx, vid in enumerate(video):
        out_path = os.path.join(save_dir, f'sample_{s_idx:02d}.mp4')
        save_image_or_video(vid, out_path, fps=16)
        log.success(f'Saved: {out_path}')
