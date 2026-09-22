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
MODEL_SIZE="${MODEL_SIZE:-14b}"
RESOLUTION="${RESOLUTION:-480p}"

case "${MODEL_SIZE,,}" in
  1.3b|1pt3b|1p3b) MODEL_SIZE="1pt3b"; MODEL_LABEL="1pt3B" ;;
  14b) MODEL_SIZE="14b"; MODEL_LABEL="14B" ;;
  *) echo "MODEL_SIZE must be 1pt3b or 14b"; exit 1 ;;
esac

case "${TASK,,}" in
  t2v|i2v) TASK="${TASK,,}" ;;
  *) echo "TASK must be t2v or i2v"; exit 1 ;;
esac

case "${RESOLUTION}" in
  480p|720p) ;;
  *) echo "RESOLUTION must be 480p or 720p"; exit 1 ;;
esac

if [[ "${MODEL_SIZE}" == "1pt3b" && "${TASK}" != "t2v" ]]; then
  echo "Wan 2.1 1.3B supports T2V only in this entrypoint."
  exit 1
fi
if [[ "${MODEL_SIZE}" == "1pt3b" && "${RESOLUTION}" != "480p" ]]; then
  echo "Wan 2.1 1.3B finetune is configured for 480p."
  exit 1
fi
if [[ "${RESOLUTION}" == "720p" && -z "${EXPERIMENT:-}" ]]; then
  echo "Wan 2.1 finetune has no standard 720p experiment in this snapshot."
  echo "Set EXPERIMENT explicitly if a compatible 720p config is available."
  exit 1
fi

if [[ "${TASK}" == "t2v" ]]; then
  if [[ "${MODEL_SIZE}" == "1pt3b" ]]; then
    MODEL_ROOT="${MODEL_ROOT:-${PRETRAIN_ROOT}/Wan2.1-T2V-1.3B}"
  else
    MODEL_ROOT="${MODEL_ROOT:-${PRETRAIN_ROOT}/Wan2.1-T2V-14B}"
  fi
  # Default: load the native Wan repository directory (contains
  # diffusion_pytorch_model.safetensors, optionally sharded with a .index.json).
  PRETRAINED_CKPT="${PRETRAINED_CKPT:-${MODEL_ROOT}}"
  TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_ROOT}/google/umt5-xxl}"
  EXPERIMENT="${EXPERIMENT:-wan2pt1_${MODEL_LABEL}_res${RESOLUTION}_t2v_finetune}"
  if [[ "${MODEL_SIZE}" == "1pt3b" ]]; then
    FLOW_SHIFT_DEFAULT="8.0"
  else
    FLOW_SHIFT_DEFAULT="5.0"
  fi
else
  if [[ "${MODEL_SIZE}" != "14b" ]]; then
    echo "Wan 2.1 I2V finetune requires MODEL_SIZE=14b."
    exit 1
  fi
  I2V_RESOLUTION="${RESOLUTION/480p/480P}"
  I2V_RESOLUTION="${I2V_RESOLUTION/720p/720P}"
  MODEL_ROOT="${MODEL_ROOT:-${PRETRAIN_ROOT}/Wan2.1-I2V-14B-${I2V_RESOLUTION}}"
  PRETRAINED_CKPT="${PRETRAINED_CKPT:-${MODEL_ROOT}}"
  CLIP_ENCODER="${CLIP_ENCODER:-${MODEL_ROOT}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth}"
  TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_ROOT}/google/umt5-xxl}"
  EXPERIMENT="${EXPERIMENT:-wan2pt1_14B_res${RESOLUTION}_i2v_finetune}"
  FLOW_SHIFT_DEFAULT="3.0"
fi

JOB_NAME="${JOB_NAME:-wan2pt1_${MODEL_LABEL}_res${RESOLUTION}_${TASK}_finetune}"
VAE_PATH="${VAE_PATH:-${MODEL_ROOT}/Wan2.1_VAE.pth}"
T5_PATH="${T5_PATH:-${MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth}"
DATASET="${DATASET:-${ROLA_DATA_ROOT}/wan2pt1_${TASK}_${MODEL_SIZE}_${RESOLUTION}/shard*.tar}"

ARGS=(
  -m scripts.train
  --config=sparkdiffusion/configs/registry_distill.py
  --
  "experiment=${EXPERIMENT}"
  "job.name=${JOB_NAME}"
  "model.config.pretrained_ckpt=${PRETRAINED_CKPT}"
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

if [[ "${TASK}" == "i2v" ]]; then
  ARGS+=("model.config.i2v_clip_encoder_path=${CLIP_ENCODER}")
fi

"${TORCHRUN_BIN}" --nproc_per_node="${NUM_GPUS:-2}" --master_port="${MASTER_PORT:-29512}" "${ARGS[@]}"
