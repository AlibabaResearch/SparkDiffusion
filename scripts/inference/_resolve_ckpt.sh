#!/bin/bash
# ============================================================================
# Shared checkpoint resolver for scripts/inference/eval_student_*.sh
#
# Default (native) Wan release format — one flat directory holding every module:
#
#   <root>/
#   ├── diffusion_pytorch_model.safetensors           DiT: single file (T2V-1.3B)
#   │   or  diffusion_pytorch_model-0000X-of-0000Y.safetensors + index.json  (14B / I2V)
#   ├── Wan2.1_VAE.pth                                VAE
#   ├── models_t5_umt5-xxl-enc-bf16.pth               T5 text encoder
#   ├── google/umt5-xxl                               tokenizer
#   ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth   CLIP vision (I2V)
#   └── xlm-roberta-large/                            CLIP tokenizer (I2V)
#
# Explicit env overrides always win, so a checkpoint living outside the root
# (loose .pth students, DCP training output) is still reachable:
#
#   DIT_PATH=/path/to/other.pth  bash ...eval_student_2pt1_distilled.sh <root> ...
#
# Sets: DIT_PATH VAE_PATH TEXT_ENCODER TOKENIZER CLIP_ENCODER
# ============================================================================

_rc_die() { echo "$@" >&2; exit 2; }

# A DCP checkpoint is a directory of .distcp shards, either directly or under model/.
_rc_is_dcp_dir() {
    local d="$1"
    [ -d "$d" ] || return 1
    compgen -G "$d/*.distcp" >/dev/null 2>&1 && return 0
    compgen -G "$d/model/*.distcp" >/dev/null 2>&1 && return 0
    return 1
}

# resolve_ckpt_root <root> <log_tag>
resolve_ckpt_root() {
    local root="${1%/}"
    local tag="$2"

    [ -n "$root" ] || _rc_die "ckpt_root is empty"

    # ---- DiT ----
    DIT_PATH="${DIT_PATH:-}"
    if [ -z "$DIT_PATH" ]; then
        if [ -d "$root" ]; then
            # Native release root: resolve DiT as the safetensors file(s) in the root.
            if [ -f "$root/diffusion_pytorch_model.safetensors" ]; then
                DIT_PATH="$root/diffusion_pytorch_model.safetensors"
            elif compgen -G "$root/diffusion_pytorch_model-*.safetensors" >/dev/null 2>&1; then
                DIT_PATH="$root"
            else
                _rc_die "No DiT found in $root (expected diffusion_pytorch_model.safetensors or sharded safetensors with index.json). Override with DIT_PATH=."
            fi
        else
            # root is a file — treat it as the DiT directly (e.g. a .pth)
            DIT_PATH="$root"
        fi
    fi

    # ---- VAE ----
    VAE_PATH="${VAE_PATH:-$root/Wan2.1_VAE.pth}"
    # ---- T5 ----
    TEXT_ENCODER="${TEXT_ENCODER:-$root/models_t5_umt5-xxl-enc-bf16.pth}"
    # ---- tokenizer ----
    TOKENIZER="${TOKENIZER:-$root/google/umt5-xxl}"
    # ---- CLIP (I2V) ----
    CLIP_ENCODER="${CLIP_ENCODER:-$root/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth}"

    # A DCP training checkpoint needs a one-off conversion; a directory of
    # safetensors loads directly and must not be mistaken for one.
    if _rc_is_dcp_dir "$DIT_PATH"; then
        local name pt
        name="$(basename "$DIT_PATH")"
        [ "$name" = "model" ] && name="$(basename "$(dirname "$DIT_PATH")")"
        pt="assets/_ckpts/${name}_ema.pt"
        mkdir -p "$(dirname "$pt")"
        if [ ! -f "$pt" ]; then
            echo "[$tag] dcp_to_pth: converting $DIT_PATH -> $pt"
            python scripts/dcp_to_pth.py --dcp_checkpoint_dir "$DIT_PATH" --save_path "$pt"
        else
            echo "[$tag] dcp_to_pth: using cached $pt"
        fi
        DIT_PATH="$pt"
    fi

    if [ -d "$DIT_PATH" ]; then
        compgen -G "$DIT_PATH/*.safetensors" >/dev/null \
            || _rc_die "no .safetensors in DiT directory: $DIT_PATH (override with DIT_PATH=)"
    else
        [ -f "$DIT_PATH" ] || _rc_die "DiT checkpoint not found: $DIT_PATH"
    fi
    [ -f "$VAE_PATH" ] || _rc_die "VAE not found: $VAE_PATH (override with VAE_PATH=)"
    [ -f "$TEXT_ENCODER" ] || _rc_die "text encoder not found: $TEXT_ENCODER (override with TEXT_ENCODER=)"
    [ -d "$TOKENIZER" ] || _rc_die "tokenizer dir not found: $TOKENIZER (override with TOKENIZER=)"

    echo "[$tag] ckpt_root=$root"
    echo "[$tag]   dit      =$DIT_PATH"
    echo "[$tag]   vae      =$VAE_PATH"
    echo "[$tag]   t5       =$TEXT_ENCODER"
    echo "[$tag]   tokenizer=$TOKENIZER"
}