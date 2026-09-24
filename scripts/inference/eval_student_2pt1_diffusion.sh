#!/bin/bash
# ============================================================================
# Wan2.1 T2V / I2V single-case diffusion evaluation (UniPC + CFG).
#   entry point: sparkdiffusion/inference/wan2pt1_t2v_diffusion_infer.py   (no image_path)
#                sparkdiffusion/inference/wan2pt1_i2v_diffusion_infer.py   (image_path given)
#
#   bash scripts/inference/eval_student_2pt1_diffusion.sh <ckpt_root> [output_dir] [num_steps] [bf16|fp8] [model_size] [topk] [image_path] [prompt]
#
# WARNING: feeding a *distilled* checkpoint into this multi-step CFG path is
# a regime mismatch — the video will look like noise. That is expected, not a
# bug: distilled students only sample correctly on the distilled path. Use this script
# with a non-distilled (teacher/base) checkpoint.
#
# ---------------------------------------------------------------------------
# Positional args (defaults in brackets)
# ---------------------------------------------------------------------------
#   1 ckpt_root          weight ROOT dir (required). dit/vae/t5/tokenizer are
#                        derived from it; see 'Path resolution' below.
#   2 output_dir        [$OUT_ROOT/<ckpt name>]  directory the .mp4 lands in
#   3 num_steps         [50]
#   4 precision         [fp8]      bf16 | fp8
#   5 model_size        [14B_rola]  1.3B* | 14B*  (dense or _rola / _pure_sla)
#   6 topk              [config]   RoLa top-k *ratio*; sparsity = 1 - ratio
#                                  (0.1 -> 90%, 0.05 -> 95%, 0.03 -> 97%).
#                                  Only meaningful for *_rola / *_pure_sla.
#   7 image_path        [none]     Passing an image switches the run to I2V.
#                                  I2V needs an I2V checkpoint (in_dim=36) and
#                                  only has 14B / 14B_rola / 14B_pure_sla configs
#                                  (14B_pure_sla is an unrunnable placeholder).
#   8 prompt            [built-in] Text prompt for this inference case.
#
# ---------------------------------------------------------------------------
# EXAMPLES
# ---------------------------------------------------------------------------
# Output paths below are examples; replace checkpoint paths with local paths.
#
# 1) 14B_rola, 8 steps, FP8 (480p 5s)
#    CUDA_VISIBLE_DEVICES=2 bash scripts/inference/eval_student_2pt1_diffusion.sh \
#      pretrain_weights/Wan2.1-T2V-14B-Diffusers outputs/inference/wan21_diff 8 fp8 14B_rola "" \
#      "A cat walking through a sunlit garden."
#
# 2) Dense 14B, BF16, 50 steps (reference quality)
#    CUDA_VISIBLE_DEVICES=2 bash scripts/inference/eval_student_2pt1_diffusion.sh \
#      pretrain_weights/Wan2.1-T2V-14B-Diffusers outputs/inference/wan21_diff_ref 50 bf16 14B "" \
#      "A cat walking through a sunlit garden."
#
# 3) 95% sparsity (top-k 0.05 instead of the default 0.1)
#    CUDA_VISIBLE_DEVICES=2 bash scripts/inference/eval_student_2pt1_diffusion.sh \
#      pretrain_weights/Wan2.1-T2V-14B-Diffusers outputs/inference/wan21_diff 8 fp8 14B_rola 0.05 "" \
#      "A cat walking through a sunlit garden."
#
# 4) One prompt at 720p
#    CUDA_VISIBLE_DEVICES=2 RESOLUTION=720p \
#      bash scripts/inference/eval_student_2pt1_diffusion.sh \
#      pretrain_weights/Wan2.1-T2V-14B-Diffusers outputs/inference/wan21_diff_720p 8 fp8 14B_rola "" \
#      "A cat walking through a sunlit garden."
#
# 5) I2V 480p — 7th arg is the reference image
#    CUDA_VISIBLE_DEVICES=2 bash scripts/inference/eval_student_2pt1_diffusion.sh \
#      pretrain_weights/Wan2.1-I2V-14B-480P-Diffusers outputs/inference/wan21_i2v_diff 40 fp8 14B_rola \
#      "" examples/i2v_input_1.jpg "A person walks through a forest."
#
# ---------------------------------------------------------------------------
# Path resolution (explicit env wins over the ckpt_root default)
# ---------------------------------------------------------------------------
#   DIT_PATH       <ckpt_root>/diffusion_pytorch_model*.safetensors.
#                  May instead point at a loose .pt/.pth, or at a DCP dir
#                  (converted once, then cached).
#   VAE_PATH       <ckpt_root>/Wan2.1_VAE.pth
#   TEXT_ENCODER   <ckpt_root>/models_t5_umt5-xxl-enc-bf16.pth
#   TOKENIZER      <ckpt_root>/google/umt5-xxl
#   CLIP_ENCODER   <ckpt_root>/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
#                  or an HF image_encoder/ directory (I2V only)
#
# ---------------------------------------------------------------------------
# Env overrides (defaults in brackets)
# ---------------------------------------------------------------------------
#   NUM_FRAMES  [81]    81 frames @16fps = 5.06 s. Use 77 for the older reports.
#   RESOLUTION  [480p]  480p | 720p
#   ASPECT_RATIO[16:9]
#   SEED        [1]
#   NUM_SAMPLES [1] Sequential samples, batch size 1; seeds SEED, SEED+1, ...
#   PROMPT       [built-in]  single prompt passed to the Python entrypoint
#   OUT_ROOT    [outputs/inference]
#                       only the default parent of output_dir (arg 2)
#   CLIP_ENCODER        I2V only. Native CLIP .pth/.pt or HF image_encoder directory.
#   FIXED_RESOLUTION [0]  I2V only. 1 = force RESOLUTION/ASPECT_RATIO instead of
#                       the official input-aspect sizing (which preserves the
#                       target pixel area but follows the image's aspect).
#
# Notes
#   - "fp8" means --quant_type fp8, which W8A8-quantizes the FFN *and* the
#     attention Q/K/V/O projections; there is no separate --quant_attn and no
#     FP8 RoLa-attention kernel. QUANT_ATTN=1 is accepted but is a no-op.
#   - CFG doubles the per-step cost vs the distilled path.
#   - 720p needs an otherwise idle GPU.
#   - For Wan2.2 (dual high/low-noise experts) use scripts/inference/eval_student_2pt2_diffusion.sh.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
source scripts/inference/_resolve_ckpt.sh

# ---- Defaults (env-overridable) ----
NUM_SAMPLES="${NUM_SAMPLES:-1}"
NUM_FRAMES="${NUM_FRAMES:-81}"
RESOLUTION="${RESOLUTION:-480p}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"
SEED="${SEED:-1}"
OUT_ROOT="${OUT_ROOT:-outputs/inference}"
ATTN_PRECISION="bf16"
QUANT_MODE="w8a8"

USAGE="Usage: bash scripts/inference/eval_student_2pt1_diffusion.sh <ckpt_root> [output_dir] [num_steps] [bf16|fp8] [model_size] [topk] [image_path] [prompt]"
CKPT_ROOT="${1:?$USAGE}"
CKPT_NAME="$(basename "${CKPT_ROOT%/}")"
OUTPUT_DIR="${2:-$OUT_ROOT/$CKPT_NAME}"
NUM_STEPS="${3:-50}"
PRECISION="${4:-fp8}"
MODEL_SIZE="${5:-14B_rola}"
ROLA_TOPK="${6:-}"
IMAGE_PATH="${7:-}"
PROMPT="${8:-${PROMPT:-A cat playing in the garden under the sun.}}"

# ---- An image switches the run to I2V ----
if [ -n "$IMAGE_PATH" ]; then
    TASK="i2v"
    ENTRY="sparkdiffusion/inference/wan2pt1_i2v_diffusion_infer.py"
    [ -f "$IMAGE_PATH" ] || { echo "image_path not found: $IMAGE_PATH" >&2; exit 2; }
else
    TASK="t2v"
    ENTRY="sparkdiffusion/inference/wan2pt1_t2v_diffusion_infer.py"
fi

# ---- model_size must be a Wan2.1 config for this task ----
case "$MODEL_SIZE" in
    1.3B|14B|1.3B_rola|14B_rola|1.3B_pure_sla|14B_pure_sla) ;;
    A14B*)
        echo "model_size '$MODEL_SIZE' is a Wan2.2 config; use scripts/inference/eval_student_2pt2_diffusion.sh instead." >&2
        exit 2 ;;
    *)
        echo "Unknown model_size: $MODEL_SIZE" >&2
        echo "Wan2.1 options: 1.3B 14B 1.3B_rola 14B_rola 1.3B_pure_sla 14B_pure_sla" >&2
        exit 2 ;;
esac
if [ "$TASK" = "i2v" ]; then
    case "$MODEL_SIZE" in
        14B|14B_rola) ;;
        14B_pure_sla)
            echo "[$(basename "$0" .sh)] warning: 14B_pure_sla is a placeholder config -- it needs the" >&2
            echo "  external sparse_linear_attention package and there is no pure-SLA-distilled i2v ckpt." >&2 ;;
        *)
            echo "model_size '$MODEL_SIZE' has no I2V config; I2V only has 14B, 14B_rola and 14B_pure_sla." >&2
            exit 2 ;;
    esac
fi

# ---- topk is a ratio in (0,1]; sparsity = 1 - ratio ----
SPARSITY=""
if [ -n "$ROLA_TOPK" ]; then
    SPARSITY=$(python3 -c 'import sys; v=float(sys.argv[1]); assert 0 < v <= 1; print(f"{100*(1-v):.0f}%")' "$ROLA_TOPK" 2>/dev/null) || {
        echo "Invalid topk: '$ROLA_TOPK' (expected a ratio in (0,1]: 0.1 -> 90%, 0.05 -> 95%, 0.03 -> 97%)" >&2
        exit 2
    }
fi

QUANT_TYPE=""
case "$PRECISION" in
    bf16) ;;
    fp8|--fp8) QUANT_TYPE="fp8" ;;
    *) echo "Unsupported precision: $PRECISION (expected bf16 or fp8)" >&2; exit 2 ;;
esac

if [ "${QUANT_ATTN:-0}" = "1" ] && [ -n "$QUANT_TYPE" ]; then
    echo "[eval_student_2pt1_diffusion] note: QUANT_ATTN=1 is a no-op here; fp8 already covers FFN + attention Q/K/V/O."
fi

resolve_ckpt_root "$CKPT_ROOT" "eval_student_2pt1_diffusion"

if [ "$TASK" = "i2v" ]; then
    if [ -d "$CLIP_ENCODER" ]; then
        :
    elif [ -f "$CLIP_ENCODER" ] && [[ "$CLIP_ENCODER" == *.pth || "$CLIP_ENCODER" == *.pt ]]; then
        :
    else
        echo "CLIP encoder not found or unsupported: $CLIP_ENCODER" >&2
        echo "I2V needs a native CLIP .pth/.pt file or an HF image_encoder directory; override with CLIP_ENCODER=." >&2
        exit 2
    fi
fi

mkdir -p "$OUTPUT_DIR"
SUFFIX=""
[ -n "$ROLA_TOPK" ] && SUFFIX="_topk${ROLA_TOPK}"
TASK_TAG=""
[ "$TASK" = "i2v" ] && TASK_TAG="_i2v"
SAVE_PATH="$OUTPUT_DIR/${MODEL_SIZE}_${PRECISION}${TASK_TAG}_diffusion${SUFFIX}.mp4"

echo "======================================================================"
echo "[eval_student_2pt1_diffusion] Wan2.1 | mode=diffusion | task=$TASK | model=$MODEL_SIZE"
echo "[eval_student_2pt1_diffusion] entry=$ENTRY"
echo "[eval_student_2pt1_diffusion] steps=$NUM_STEPS | frames=$NUM_FRAMES | resolution=$RESOLUTION $ASPECT_RATIO | seed=$SEED | samples=$NUM_SAMPLES (first sample: warmup)"
echo "[eval_student_2pt1_diffusion] precision=$PRECISION | quant=${QUANT_TYPE:-disabled} | attn=$ATTN_PRECISION"
echo "[eval_student_2pt1_diffusion] ckpt=$DIT_PATH"
[ "$TASK" = "i2v" ] && echo "[eval_student_2pt1_diffusion] image=$IMAGE_PATH"
[ "$TASK" = "i2v" ] && echo "[eval_student_2pt1_diffusion] clip=$CLIP_ENCODER (fixed_resolution=${FIXED_RESOLUTION:-0})"
[ -n "$ROLA_TOPK" ] && echo "[eval_student_2pt1_diffusion] rola_topk_ratio=$ROLA_TOPK (sparsity $SPARSITY)"
echo "[eval_student_2pt1_diffusion] prompt=$PROMPT"
echo "[eval_student_2pt1_diffusion] output=$SAVE_PATH"
echo "======================================================================"

ARGS=(
    --model_size   "$MODEL_SIZE"
    --num_samples  "$NUM_SAMPLES"
    --num_frames   "$NUM_FRAMES"
    --resolution   "$RESOLUTION"
    --aspect_ratio "$ASPECT_RATIO"
    --seed         "$SEED"
    --num_steps    "$NUM_STEPS"
    --dit_path     "$DIT_PATH"
    --prompt       "$PROMPT"
    --save_path    "$SAVE_PATH"
    --vae_path     "$VAE_PATH"
    --text_encoder_path "$TEXT_ENCODER"
    --tokenizer_path "$TOKENIZER"
    --attn_precision "$ATTN_PRECISION"
)
[ -n "$QUANT_TYPE" ] && ARGS+=(--quant_type "$QUANT_TYPE" --quant_mode "$QUANT_MODE")
[ -n "$ROLA_TOPK" ] && ARGS+=(--rola_topk_ratio "$ROLA_TOPK")
if [ "$TASK" = "i2v" ]; then
    ARGS+=(--image_path "$IMAGE_PATH" --clip_encoder_path "$CLIP_ENCODER")
    [ "${FIXED_RESOLUTION:-0}" = "1" ] && ARGS+=(--fixed_resolution)
fi

echo "[eval_student_2pt1_diffusion] exec: python $ENTRY"
python "$ENTRY" "${ARGS[@]}"

echo "Done -> $OUTPUT_DIR"
