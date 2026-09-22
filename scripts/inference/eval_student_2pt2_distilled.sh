#!/bin/bash
# ============================================================================
# Wan2.2 T2V single-case distilled evaluation (high-noise + low-noise experts).
#   entry point: sparkdiffusion/inference/wan2pt2_t2v_distilled_infer.py
#
#   bash scripts/inference/eval_student_2pt2_distilled.sh <ckpt_root> [output_dir] [num_steps] [bf16|fp8] [model_size] [topk] [prompt]
#
# num_steps is split across the two experts (4 -> 2 high + 2 low).
# Without CKPT_LOW the same checkpoint feeds both experts, which is what the
# 2.1-weights-on-2.2-architecture measurements use.
#
# ---------------------------------------------------------------------------
# Positional args (defaults in brackets)
# ---------------------------------------------------------------------------
#   1 ckpt_root          weight ROOT dir (required). dit/vae/t5/tokenizer are
#                        derived from it; see 'Path resolution' below.
#   2 output_dir        [$OUT_ROOT/<ckpt name>]  directory the .mp4 lands in
#   3 num_steps         [4]        split high/low
#   4 precision         [fp8]      bf16 | fp8
#   5 model_size        [A14B_rola] A14B | A14B_rola | A14B_pure_sla
#   6 topk              [config]   RoLa top-k *ratio*; sparsity = 1 - ratio
#                                  (0.1 -> 90%, 0.05 -> 95%, 0.03 -> 97%).
#                                  Only meaningful for *_rola / *_pure_sla.
#   7 prompt            [built-in] Text prompt for this inference case.
#
# ---------------------------------------------------------------------------
# EXAMPLES
# ---------------------------------------------------------------------------
# Output paths below are examples; replace checkpoint paths with local paths.
#
# 1) A14B_rola, 4 steps (=2 high + 2 low), FP8 — 480p 5s
#    CUDA_VISIBLE_DEVICES=2 bash scripts/inference/eval_student_2pt2_distilled.sh \
#      pretrain_weights/Wan2.2-T2V-A14B-Diffusers outputs/inference/wan22_480p 4 fp8 A14B_rola "" \
#      "A cat playing in the garden under the sun."
#
# 2) 720p 5s
#    CUDA_VISIBLE_DEVICES=2 RESOLUTION=720p \
#      bash scripts/inference/eval_student_2pt2_distilled.sh \
#      pretrain_weights/Wan2.2-T2V-A14B-Diffusers outputs/inference/wan22_720p 4 fp8 A14B_rola "" \
#      "A cat playing in the garden under the sun."
#
# 3) 95% sparsity (top-k 0.05 instead of the default 0.1)
#    CUDA_VISIBLE_DEVICES=2 RESOLUTION=720p \
#      bash scripts/inference/eval_student_2pt2_distilled.sh \
#      pretrain_weights/Wan2.2-T2V-A14B-Diffusers outputs/inference/wan22_720p 4 fp8 A14B_rola 0.05 \
#      "A cat playing in the garden under the sun."
#
# 4) One prompt at 97% sparsity
#    CUDA_VISIBLE_DEVICES=2 \
#      bash scripts/inference/eval_student_2pt2_distilled.sh \
#      pretrain_weights/Wan2.2-T2V-A14B-Diffusers outputs/inference/wan22_480p 4 fp8 A14B_rola 0.03 \
#      "A cat playing in the garden under the sun."
#
# 5) Real distilled weights with different high/low experts
#    CUDA_VISIBLE_DEVICES=2 CKPT_LOW=pretrain_weights/Wan2.2-T2V-A14B-Diffusers/transformer_2 \
#      bash scripts/inference/eval_student_2pt2_distilled.sh \
#      pretrain_weights/Wan2.2-T2V-A14B-Diffusers outputs/inference/wan22_real 4 fp8 A14B_rola "" \
#      "A cat playing in the garden under the sun."
#
# ---------------------------------------------------------------------------
# Path resolution (explicit env wins over the ckpt_root default)
# ---------------------------------------------------------------------------
#   DIT_PATH       <ckpt_root>/transformer   (sharded safetensors + index, plus
#                  finetrainers_extra.safetensors for the RoLa params).
#                  May instead point at a loose .pt/.pth, or at a DCP dir
#                  (converted once, then cached).
#   VAE_PATH       <ckpt_root>/vae
#   TEXT_ENCODER   <ckpt_root>/text_encoder
#   TOKENIZER      <ckpt_root>/tokenizer
#
# ---------------------------------------------------------------------------
# Env overrides (defaults in brackets)
# ---------------------------------------------------------------------------
#   NUM_FRAMES  [81]    81 frames @16fps = 5.06 s. Use 77 for the older reports.
#   RESOLUTION  [480p]  480p | 720p
#   ASPECT_RATIO[16:9]
#   SEED        [1]
#   NUM_SAMPLES [1]
#   PROMPT       [built-in]  single prompt passed to the Python entrypoint
#   OUT_ROOT    [outputs/inference]
#                       only the default parent of output_dir (arg 2)
#   CKPT_LOW    [=<ckpt>]  low-noise expert, if it differs from the high-noise one
#
# Notes
#   - "fp8" means --quant_type fp8, which W8A8-quantizes the FFN *and* the
#     attention Q/K/V/O projections; there is no separate --quant_attn and no
#     FP8 RoLa-attention kernel. QUANT_ATTN=1 is accepted but is a no-op.
#   - The expert swap is mandatory on 32 GB: both 14B experts resident OOMs.
#   - 720p needs an otherwise idle GPU.
#   - For Wan2.1 (single model) use scripts/inference/eval_student_2pt1_distilled.sh.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
source scripts/inference/_resolve_ckpt.sh

ENTRY="sparkdiffusion/inference/wan2pt2_t2v_distilled_infer.py"

# ---- Defaults (env-overridable) ----
NUM_SAMPLES="${NUM_SAMPLES:-1}"
NUM_FRAMES="${NUM_FRAMES:-81}"
RESOLUTION="${RESOLUTION:-480p}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"
SEED="${SEED:-1}"
OUT_ROOT="${OUT_ROOT:-outputs/inference}"
ATTN_PRECISION="bf16"
QUANT_MODE="w8a8"

USAGE="Usage: bash scripts/inference/eval_student_2pt2_distilled.sh <ckpt_root> [output_dir] [num_steps] [bf16|fp8] [model_size] [topk] [prompt]"
CKPT_ROOT="${1:?$USAGE}"
CKPT_NAME="$(basename "${CKPT_ROOT%/}")"
OUTPUT_DIR="${2:-$OUT_ROOT/$CKPT_NAME}"
NUM_STEPS="${3:-4}"
PRECISION="${4:-fp8}"
MODEL_SIZE="${5:-A14B_rola}"
ROLA_TOPK="${6:-}"
PROMPT="${7:-${PROMPT:-A cat playing in the garden under the sun.}}"

# ---- model_size must be a Wan2.2 config ----
case "$MODEL_SIZE" in
    A14B|A14B_rola|A14B_pure_sla) ;;
    1.3B*|14B*)
        echo "model_size '$MODEL_SIZE' is a Wan2.1 config; use scripts/inference/eval_student_2pt1_distilled.sh instead." >&2
        exit 2 ;;
    *)
        echo "Unknown model_size: $MODEL_SIZE" >&2
        echo "Wan2.2 options: A14B A14B_rola A14B_pure_sla" >&2
        exit 2 ;;
esac

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
    echo "[eval_student_2pt2_distilled] note: QUANT_ATTN=1 is a no-op here; fp8 already covers FFN + attention Q/K/V/O."
fi

resolve_ckpt_root "$CKPT_ROOT" "eval_student_2pt2_distilled"
DIT_PATH_LOW="${CKPT_LOW:-$DIT_PATH}"

# ---- Steps split across the two experts ----
STEPS_HIGH=$((NUM_STEPS / 2))
STEPS_LOW=$((NUM_STEPS - STEPS_HIGH))

mkdir -p "$OUTPUT_DIR"
SUFFIX=""
[ -n "$ROLA_TOPK" ] && SUFFIX="_topk${ROLA_TOPK}"
SAVE_PATH="$OUTPUT_DIR/${MODEL_SIZE}_${PRECISION}_distilled${SUFFIX}.mp4"

echo "======================================================================"
echo "[eval_student_2pt2_distilled] Wan2.2 | mode=distilled | model=$MODEL_SIZE"
echo "[eval_student_2pt2_distilled] entry=$ENTRY"
echo "[eval_student_2pt2_distilled] steps=$NUM_STEPS (high=$STEPS_HIGH low=$STEPS_LOW) | frames=$NUM_FRAMES | resolution=$RESOLUTION $ASPECT_RATIO | seed=$SEED"
echo "[eval_student_2pt2_distilled] precision=$PRECISION | quant=${QUANT_TYPE:-disabled} | attn=$ATTN_PRECISION"
echo "[eval_student_2pt2_distilled] high=$DIT_PATH"
echo "[eval_student_2pt2_distilled] low =$DIT_PATH_LOW"
[ -n "$ROLA_TOPK" ] && echo "[eval_student_2pt2_distilled] rola_topk_ratio=$ROLA_TOPK (sparsity $SPARSITY)"
echo "[eval_student_2pt2_distilled] prompt=$PROMPT"
echo "[eval_student_2pt2_distilled] output=$SAVE_PATH"
echo "======================================================================"

ARGS=(
    --model_size   "$MODEL_SIZE"
    --num_samples  "$NUM_SAMPLES"
    --num_frames   "$NUM_FRAMES"
    --resolution   "$RESOLUTION"
    --aspect_ratio "$ASPECT_RATIO"
    --seed         "$SEED"
    --high_noise_model_path "$DIT_PATH"
    --low_noise_model_path  "$DIT_PATH_LOW"
    --num_steps_high "$STEPS_HIGH"
    --num_steps_low  "$STEPS_LOW"
    --prompt       "$PROMPT"
    --save_path    "$SAVE_PATH"
    --vae_path     "$VAE_PATH"
    --text_encoder_path "$TEXT_ENCODER"
    --tokenizer_path "$TOKENIZER"
    --attn_precision "$ATTN_PRECISION"
)
[ -n "$QUANT_TYPE" ] && ARGS+=(--quant_type "$QUANT_TYPE" --quant_mode "$QUANT_MODE")
[ -n "$ROLA_TOPK" ] && ARGS+=(--rola_topk_ratio "$ROLA_TOPK")

echo "[eval_student_2pt2_distilled] exec: python $ENTRY"
python "$ENTRY" "${ARGS[@]}"

echo "Done -> $OUTPUT_DIR"
