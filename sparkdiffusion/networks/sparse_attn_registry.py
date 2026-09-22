"""
Sparse-attention plugin registry.

Purpose: let users plug in a custom sparse-attention implementation without
touching the core network / finetuning / distillation code.

Design principles (important):
  - This module is an ADDITIVE parallel path; it does not change the behavior of
    any existing boolean switch (use_rola_attn, etc.).
  - Network layers only query the registry when `attn_variant` is explicitly set;
    when it is None the original logic is used unchanged.
  - The registry stores only "class + metadata" and imports no network module,
    avoiding circular dependencies.

Three steps to plug in a custom sparse attention:
  1. Write a class whose constructor is compatible with WanSelfAttention
     (dim, num_heads, qk_norm, eps, **kwargs) and whose
     forward(x[B, L, dim], seq_lens, freqs) -> [B, L, dim]; implement init_weights().
  2. Decorate it with @register_sparse_attn("my_name", sparse_param_names=[...]).
     sparse_param_names declares which new params finetuning Stage-1 should train
     (matched by substring).
  3. Set model.config.net.attn_variant="my_name" in the experiment (plus any
     needed attn_kwargs). Finetuning / distillation / inference pick it up
     automatically, with no core-file changes.
"""

from typing import Dict, List, Optional, Type

# name -> attention class
SPARSE_ATTN_REGISTRY: Dict[str, Type] = {}
# name -> trainable sparse param names added by the variant (substring match),
# used by the finetuning Stage-1 freeze logic
SPARSE_ATTN_PARAM_NAMES: Dict[str, List[str]] = {}


def register_sparse_attn(name: str, sparse_param_names: Optional[List[str]] = None):
    """Register a sparse-attention class into the global registry.

    Args:
        name: the string key referenced via attn_variant in an experiment.
        sparse_param_names: trainable param names added by this variant (substring
            match). Finetuning Stage-1 adds these to the freeze whitelist so the
            custom params are not frozen. Optional.
    """
    def _decorator(cls):
        if name in SPARSE_ATTN_REGISTRY and SPARSE_ATTN_REGISTRY[name] is not cls:
            raise ValueError(f"sparse attention variant '{name}' is already registered as {SPARSE_ATTN_REGISTRY[name]}")
        SPARSE_ATTN_REGISTRY[name] = cls
        SPARSE_ATTN_PARAM_NAMES[name] = list(sparse_param_names or [])
        return cls
    return _decorator


def get_sparse_attn_class(name: str) -> Type:
    """Return the registered sparse-attention class by name; raise a clear error if unregistered."""
    if name not in SPARSE_ATTN_REGISTRY:
        raise KeyError(
            f"unknown attn_variant='{name}'. Registered: {sorted(SPARSE_ATTN_REGISTRY)}. "
            f"Register your sparse-attention class with @register_sparse_attn('{name}') "
            f"and make sure its module is imported (see the sparkdiffusion/networks/sparse_attn_registry.py docstring)."
        )
    return SPARSE_ATTN_REGISTRY[name]


def get_sparse_param_names(name: str) -> List[str]:
    """Return the trainable sparse param names declared by a variant (empty list if none)."""
    return list(SPARSE_ATTN_PARAM_NAMES.get(name, []))