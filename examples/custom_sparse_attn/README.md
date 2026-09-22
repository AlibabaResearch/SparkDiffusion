# Custom Sparse Attention (Plugin Integration)

Plug your own sparse attention into Wan's full finetune, distillation, and inference
pipelines without touching any core network / finetune / distillation code.

## Three-Step Integration

### 1. Write an attention class and register it

```python
from sparkdiffusion.networks.sparse_attn_registry import register_sparse_attn

@register_sparse_attn("my_attn", sparse_param_names=["my_router", "my_gate"])
class MyAttention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6, **kwargs): ...
    def init_weights(self): ...                 # called after WanModel.init_weights
    def forward(self, x, seq_lens, freqs): ...  # x:[B,L,dim] -> [B,L,dim]
```

- **Constructor signature**: the first 4 positional args are fixed as `(dim, num_heads, qk_norm, eps)`; any other hyperparameters go through `**kwargs`.
- **forward contract**: `forward(x[B,L,dim], seq_lens, freqs) -> [B,L,dim]`.
- **`sparse_param_names`**: declares the names of the new trainable parameters this variant
  adds (matched as substrings). When finetune Stage-1 freezes the backbone, these
  parameters are automatically whitelisted — **otherwise your new parameters would stay
  frozen and never train**.
- See [`my_attention.py`](my_attention.py) for a complete, runnable example.

### 2. Make sure the module is imported

Registration runs via the decorator at import time. Import the module once before your
training/inference entry point:

```python
import examples.custom_sparse_attn.my_attention   # triggers @register_sparse_attn
```

### 3. Select it in the experiment / on the command line

```python
model=dict(config=dict(net=dict(
    attn_variant="my_attn",
    attn_variant_kwargs=dict(...),   # hyperparameters passed to the MyAttention constructor
)))
```

Or override on the command line:
```bash
model.config.net.attn_variant=my_attn
```

Finetune / distillation / inference will use your attention automatically, **with zero
changes to the core code**.

## How It Works (No Impact on Existing Algorithms)

- The network layers `wan2pt1.py` / `wan2pt2.py` only query the registry when
  `attn_variant` is explicitly set; when `attn_variant=None` (the default), they **follow
  the existing `use_sla_attn` (and similar) bool-switch logic unchanged**.
- Verified: a network built with `attn_variant="sla"` has **state_dict keys identical**
  (1255/1255) to the legacy `use_sla_attn=True` path — the plugin route is an equivalent
  parallel path with zero side effects.
- The built-in SLA is already registered via `@register_sparse_attn("sla", ...)` and can
  serve as a reference.

## Notes on Weight Loading

If your custom attention's backbone projections (q/k/v/o) share names with
`WanSelfAttention`, they can load from the official dense weights; new parameters (e.g.
`my_router`) stay randomly initialized during loading (`load_checkpoint_auto` uses
`strict=False`), which matches the finetune/distillation expectation.
