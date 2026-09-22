#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/distill/_common.sh

export PYTHONPATH=.
export WANDB_MODE=${WANDB_MODE:-offline}
export NCCL_DEBUG=${NCCL_DEBUG:-ERROR}

WAN_REPO=${WAN_REPO:-pretrain_weights/Wan2.1-T2V-1.3B}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/distill}
export IMAGINAIRE_OUTPUT_ROOT=${OUTPUT_ROOT}
# teacher: native 1.3B dense DiT (single-file safetensors)
TEACHER_CKPT=${TEACHER_CKPT:-${WAN_REPO}}
# student: 1.3B RoLa sparse-finetuning checkpoint (Stage 1 output of the full framework)
STUDENT_CKPT=${STUDENT_CKPT:-pretrain_weights/finetuned_sparse.pth}
NEG_EMBED=${NEG_EMBED:-pretrain_weights/umT5_wan_negative_emb.pt}
VAE_PATH=${VAE_PATH:-${WAN_REPO}/Wan2.1_VAE.pth}
T5_PATH=${T5_PATH:-${WAN_REPO}/models_t5_umt5-xxl-enc-bf16.pth}
TOKENIZER_PATH=${TOKENIZER_PATH:-${WAN_REPO}/google/umt5-xxl}
DATASET_ROOT=${DATASET_ROOT:-datasets/distill/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K}

require_dir "SLA_SRC" "${SLA_SRC:-}"
export SLA_SRC
require_path "TEACHER_CKPT" "${TEACHER_CKPT}"
require_path "STUDENT_CKPT" "${STUDENT_CKPT}"
require_path "VAE_PATH" "${VAE_PATH}"
require_path "T5_PATH" "${T5_PATH}"
require_dir "TOKENIZER_PATH" "${TOKENIZER_PATH}"
require_file "NEG_EMBED" "${NEG_EMBED}"
require_dir "DATASET_ROOT" "${DATASET_ROOT}"

torchrun --nproc_per_node=${NPROC_PER_NODE:-8} \
    -m scripts.train --config=sparkdiffusion/configs/registry_distill.py -- \
    experiment=wan2pt1_1pt3B_res480p_t2v_rola_distill \
    model.config.student_ckpt="${STUDENT_CKPT}" \
    model.config.teacher_ckpt="${TEACHER_CKPT}" \
    model.config.tokenizer.vae_pth=${VAE_PATH} \
    model.config.text_encoder_path=${T5_PATH} \
    model.config.tokenizer_path=${TOKENIZER_PATH} \
    model.config.neg_embed_path=${NEG_EMBED} \
    "dataloader_train.tar_path_pattern=${DATASET_ROOT}/shard*.tar"
