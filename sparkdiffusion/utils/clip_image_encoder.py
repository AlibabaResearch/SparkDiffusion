"""
CLIP vision encoder wrapper for Wan 2.1 I2V.

Supports two loading paths:
  - native .pth file (open_clip ViT-H-14)      — default input format
  - HF CLIPVisionModel directory (image_encoder/) — backward compatible

Key points (matching the original Wan / FastVideo):
  - Take the penultimate layer output (31st layer), not the last layer, and skip
    post_layernorm. Implemented via output_hidden_states=True taking hidden_states[-2].
  - Input first-frame pixels are in [-1,1] (VAE decode domain) → map to [0,1] →
    CLIP mean/std normalization + resize to 224.
  - Online extraction: computed on the fly for each batch's first frame during
    training, not precomputed in the dataset.
"""

import os
import re as _re
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from imaginaire.utils import log

# CLIP ViT-H/14 normalization constants (matching image_processor/preprocessor_config.json)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_CLIP_SIZE = 224


def _remap_vision_key(k: str) -> str:
    """Remap Wan-native open_clip visual key to open_clip_torch ViT-H-14 key."""
    m = _re.match(r'^transformer\.(\d+)\.(.+)', k)
    if m:
        idx, rest = m.group(1), m.group(2)
        rest = rest.replace('norm1.', 'ln_1.')
        rest = rest.replace('norm2.', 'ln_2.')
        rest = rest.replace('attn.to_qkv.', 'attn.in_proj_')
        rest = rest.replace('attn.proj.', 'attn.out_proj.')
        rest = rest.replace('mlp.0.', 'mlp.c_fc.')
        rest = rest.replace('mlp.2.', 'mlp.c_proj.')
        return f'transformer.resblocks.{idx}.{rest}'
    _map = {
        'cls_embedding': 'class_embedding', 'pos_embedding': 'positional_embedding',
        'patch_embedding.weight': 'conv1.weight',
        'pre_norm.weight': 'ln_pre.weight', 'pre_norm.bias': 'ln_pre.bias',
        'post_norm.weight': 'ln_post.weight', 'post_norm.bias': 'ln_post.bias',
        'head': 'proj',
    }
    return _map.get(k, k)


class WanCLIPImageEncoder(nn.Module):
    """Load the Wan2.1-I2V CLIP vision encoder and extract [B,257,1280] features from the first frame."""

    def __init__(self, image_encoder_path: str, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
        super().__init__()
        self.dtype = dtype
        self.device = device

        if os.path.splitext(image_encoder_path)[1] in ('.pth', '.pt'):
            self._init_open_clip(image_encoder_path, dtype, device)
        else:
            self._init_hf(image_encoder_path, dtype, device)

        # Register normalization constants as buffers so they move with .to(device)
        self.register_buffer("_mean", torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(_CLIP_STD).view(1, 3, 1, 1))

    def _init_open_clip(self, pth_path: str, dtype: torch.dtype, device: str):
        import os
        import torch as _torch
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "Loading a native CLIP .pth checkpoint requires open_clip_torch. "
                "Install it with `pip install open_clip_torch` (see requirements.txt)."
            ) from exc

        log.info(f"Loading CLIP vision encoder from open_clip .pth: {pth_path}")
        model = open_clip.factory.create_model('ViT-H-14', pretrained=None, force_custom_text=True)
        self.model = model.visual

        sd = _torch.load(pth_path, map_location='cpu', weights_only=True)
        vis_sd = {}
        for k, v in sd.items():
            if k.startswith('visual.'):
                new_k = _remap_vision_key(k[len('visual.'):])
                if new_k in ('class_embedding', 'positional_embedding'):
                    v = v.squeeze()
                vis_sd[new_k] = v
        self.model.load_state_dict(vis_sd, strict=True)
        self.model = self.model.to(device=device, dtype=dtype).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._backend = 'open_clip'

    def _init_hf(self, image_encoder_path: str, dtype: torch.dtype, device: str):
        from transformers import CLIPVisionModel

        log.info(f"Loading CLIP vision encoder from HF dir: {image_encoder_path}")
        self.model = CLIPVisionModel.from_pretrained(image_encoder_path).to(device=device, dtype=dtype).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._backend = 'hf'

    @torch.no_grad()
    def encode_first_frame(self, first_frame_B_C_H_W: torch.Tensor) -> torch.Tensor:
        """First-frame pixels [B,3,H,W] (range [-1,1]) → CLIP features [B,257,1280].

        Take the penultimate hidden state, matching the original Wan clip.visual behavior.
        """
        x = first_frame_B_C_H_W.float()
        x = (x + 1.0) / 2.0
        x = x.clamp(0.0, 1.0)
        x = F.interpolate(x, size=(_CLIP_SIZE, _CLIP_SIZE), mode="bicubic", align_corners=False, antialias=True)
        x = (x - self._mean.to(x.device)) / self._std.to(x.device)
        x = x.to(self.dtype)

        if self._backend == 'open_clip':
            # Manual forward to capture penultimate layer (31st, index 30)
            model = self.model
            x = model.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            x = torch.cat([
                model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x
            ], dim=1)
            x = x + model.positional_embedding.to(x.dtype)
            x = model.ln_pre(x)
            for i, blk in enumerate(model.transformer.resblocks):
                x = blk(x)
                if i == 30:
                    feat = x.clone()  # penultimate layer [B, 257, 1280]
            return feat
        else:
            outputs = self.model(pixel_values=x, output_hidden_states=True)
            return outputs.hidden_states[-2]  # [B, 257, 1280]
