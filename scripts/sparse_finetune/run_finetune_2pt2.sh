#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"
source "${SCRIPT_DIR}/../distill/_common.sh"
require_dir "SLA_SRC" "${SLA_SRC:-}"
export SLA_SRC
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-${ROLA_OUTPUT_ROOT}}"
cd "${PROJECT_ROOT}"

TASK="${TASK:-t2v}"
RESOLUTION="${RESOLUTION:-480p}"
EXPERT="${EXPERT:-both}"

case "${TASK,,}" in
  t2v|i2v) TASK="${TASK,,}" ;;
  *) echo "TASK must be t2v or i2v"; exit 1 ;;
esac
case "${RESOLUTION}" in
  480p|720p) ;;
  *) echo "RESOLUTION must be 480p or 720p"; exit 1 ;;
esac
case "${EXPERT,,}" in
  high|low|joint|both) EXPERT="${EXPERT,,}" ;;
  *) echo "EXPERT must be high, low, joint, or both"; exit 1 ;;
esac

if [[ "${EXPERT}" == "both" ]]; then
  TASK="${TASK}" RESOLUTION="${RESOLUTION}" EXPERT=high \
    bash "${SCRIPT_DIR}/run_finetune_2pt2.sh"
  TASK="${TASK}" RESOLUTION="${RESOLUTION}" EXPERT=low \
    bash "${SCRIPT_DIR}/run_finetune_2pt2.sh"
  exit 0
fi

MODEL_ROOT="${MODEL_ROOT:-${PRETRAIN_ROOT}/Wan2.2-${TASK^^}-A14B}"
VAE_PATH="${VAE_PATH:-${MODEL_ROOT}/Wan2.1_VAE.pth}"
T5_PATH="${T5_PATH:-${MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_ROOT}/google/umt5-xxl}"
DATASET="${DATASET:-${ROLA_DATA_ROOT}/wan2pt2_${TASK}_a14b_${RESOLUTION}/shard*.tar}"
JOB_NAME="${JOB_NAME:-wan2pt2_A14B_res${RESOLUTION}_${TASK}_finetune_${EXPERT}}"

FLOW_SHIFT_DEFAULT="12.0"
RF_BOUNDARY_DEFAULT="0.3684210526"
if [[ "${TASK}" == "i2v" ]]; then
  FLOW_SHIFT_DEFAULT="5.0"
  RF_BOUNDARY_DEFAULT="0.6428571429"
fi

if [[ "${RESOLUTION}" == "720p" && -z "${EXPERIMENT:-}" ]]; then
  echo "Wan 2.2 finetune has no standard 720p experiment in this snapshot."
  echo "Set EXPERIMENT explicitly if a compatible 720p config is available."
  exit 1
fi

if [[ -z "${EXPERIMENT:-}" ]]; then
  if [[ "${EXPERT}" == "joint" ]]; then
    EXPERIMENT="wan2pt2_A14B_res480p_${TASK}_finetune_joint"
  else
    EXPERIMENT="wan2pt2_A14B_res480p_${TASK}_finetune_${EXPERT}_noise"
  fi
fi

ARGS=(
  -m scripts.train
  --config=sparkdiffusion/configs/registry_distill.py
  --
  "experiment=${EXPERIMENT}"
  "job.name=${JOB_NAME}"
  "model.config.tokenizer.vae_pth=${VAE_PATH}"
  "model.config.text_encoder_path=${T5_PATH}"
  "model.config.tokenizer_path=${TOKENIZER_PATH}"
  "model.config.resolution=${RESOLUTION}"
  "model.config.flow_shift=${FLOW_SHIFT:-${FLOW_SHIFT_DEFAULT}}"
  "model.config.fsdp_shard_size=${FSDP_SHARD_SIZE:-2}"
  "model.config.state_t=${STATE_T:-21}"
  "model.config.stage1_steps=${STAGE1_STEPS:-0}"
  "model.config.stage1_mse_weight=${STAGE1_MSE_WEIGHT:-0}"
  "model_parallel.context_parallel_size=${CP_SIZE:-1}"
  "optimizer.lr=${LR:-1e-5}"
  "trainer.max_iter=${MAX_ITER:-20}"
  "trainer.logging_iter=${LOG_ITER:-2}"
  "trainer.grad_accum_iter=${GRAD_ACCUM:-1}"
  "checkpoint.save_iter=${SAVE_ITER:-20}"
  "dataloader_train.batch_size=${BATCH_SIZE:-1}"
  "dataloader_train.tar_path_pattern=${DATASET}"
)

case "${EXPERT}" in
  high)
    PRETRAINED_CKPT="${PRETRAINED_CKPT:-${MODEL_ROOT}/high_noise_model}"
    ARGS+=("model.config.pretrained_ckpt=${PRETRAINED_CKPT}")
    ARGS+=("model.config.rf_t_min=${RF_T_MIN:-${RF_BOUNDARY_DEFAULT}}")
    ARGS+=("model.config.rf_t_max=${RF_T_MAX:-1.0}")
    ARGS+=("model.config.time_sampling=${TIME_SAMPLING:-uniform_sigma}")
    ;;
  low)
    PRETRAINED_CKPT="${PRETRAINED_CKPT:-${MODEL_ROOT}/low_noise_model}"
    ARGS+=("model.config.pretrained_ckpt=${PRETRAINED_CKPT}")
    ARGS+=("model.config.rf_t_min=${RF_T_MIN:-0.0}")
    ARGS+=("model.config.rf_t_max=${RF_T_MAX:-${RF_BOUNDARY_DEFAULT}}")
    ARGS+=("model.config.time_sampling=${TIME_SAMPLING:-lognormal_raw_rf}")
    ;;
  joint)
    PRETRAINED_CKPT_HIGH="${PRETRAINED_CKPT_HIGH:-${MODEL_ROOT}/high_noise_model}"
    PRETRAINED_CKPT_LOW="${PRETRAINED_CKPT_LOW:-${MODEL_ROOT}/low_noise_model}"
    ARGS+=("model.config.pretrained_ckpt_high=${PRETRAINED_CKPT_HIGH}")
    ARGS+=("model.config.pretrained_ckpt_low=${PRETRAINED_CKPT_LOW}")
    ARGS+=("model.config.rf_split_t=${RF_SPLIT_T:-0.875}")
    ARGS+=("model.config.time_sampling=${TIME_SAMPLING:-uniform_sigma}")
    ;;
esac

"${TORCHRUN_BIN}" --nproc_per_node="${NUM_GPUS:-2}" --master_port="${MASTER_PORT:-29520}" "${ARGS[@]}"
