#!/bin/bash

# Derive the project root without exporting a helper variable that can
# overwrite a caller's script-local path when this file is sourced.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export NCCL_DEBUG="${NCCL_DEBUG:-ERROR}"

export PRETRAIN_ROOT="${PRETRAIN_ROOT:-${PROJECT_ROOT}/pretrain_weights}"
export DISTILL_DATA_ROOT="${DISTILL_DATA_ROOT:-${PROJECT_ROOT}/datasets/distill}"
export ROLA_DATA_ROOT="${ROLA_DATA_ROOT:-${PROJECT_ROOT}/datasets/rola}"
export DISTILL_OUTPUT_ROOT="${DISTILL_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/distill}"
export ROLA_OUTPUT_ROOT="${ROLA_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/rola}"
