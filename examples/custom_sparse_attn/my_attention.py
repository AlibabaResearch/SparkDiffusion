"""
Custom sparse attention example — three-step integration into finetune/distillation/inference,
with no changes to any core file.

How to run / integrate:
  1. Make sure this file is imported (import it before training, or add it to your entry point):
       import examples.custom_sparse_attn.my_attention
  2. Set in the experiment:
       model.config.net.attn_variant="my_identity"
       model.config.net.attn_variant_kwargs={}          # your hyperparameters
  3. Finetune Stage-1 will automatically train only the parameters declared in SPARSE_PARAM_NAMES.

This example implements an "identity/dense" attention (for teaching, with no sparse speedup)
to demonstrate the interface contract. Replace the body of forward with your own sparse kernel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparkdiffusion.networks.sparse_attn_registry import register_sparse_attn


@register_sparse_attn(
    "my_identity",
    # Declares the names of the new trainable parameters this variant adds (substring
    # match). When finetune Stage-1 freezes the backbone, these are automatically
    # whitelisted so they stay trainable. Leave as [] if there are no new parameters.
    sparse_param_names=["my_scale"],
)
class MyIdentityAttention(nn.Module):
    """Example of the interface contract for a custom sparse attention.

    The constructor signature must be compatible with WanSelfAttention's first 4 positional
    args (dim, num_heads, qk_norm, eps); other hyperparameters are passed as keywords via
    attn_variant_kwargs.
    """

    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        # Standard QKVO projections (same names as WanSelfAttention, so the backbone can load from dense weights)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        # The "new" trainable parameter this variant adds (corresponds to "my_scale" in SPARSE_PARAM_NAMES)
        self.my_scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def init_weights(self):
        """WanModel.init_weights() resets every nn.Linear to xavier; if this variant's new
        parameters need specific initial values, re-apply them here (called after backbone init)."""
        nn.init.ones_(self.my_scale)

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : [B, L, dim]   video tokens
            seq_lens : sequence lengths (unused here, kept for signature compatibility)
            freqs    : RoPE frequencies (unused here; a real implementation should apply RoPE to q/k)
        Returns:
            [B, L, dim]
        """
        B, L, _ = x.shape
        n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x).view(B, L, n, d))
        k = self.norm_k(self.k(x).view(B, L, n, d))
        v = self.v(x).view(B, L, n, d)
        # [B, n, L, d]
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        # —— replace this with your own sparse attention kernel ——
        out = F.scaled_dot_product_attention(q, k, v)
        out = out * self.my_scale  # demonstrates the new parameter participating in the computation
        out = out.transpose(1, 2).reshape(B, L, self.dim)
        return self.o(out)
