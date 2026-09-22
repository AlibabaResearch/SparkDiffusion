---
name: sparkdiffusion-setup
description: Prepare a SparkDiffusion checkout for training or inference. Use when a user asks to install dependencies, configure model/data/output paths, validate a new machine, or troubleshoot missing weights and datasets.
---

# SparkDiffusion Setup

Use this skill before launching training or inference on a fresh checkout or a new server.

## Rules

- Work from the repository root.
- Prefer repository-relative paths and environment variables. Do not hard-code machine-specific paths.
- Do not download or copy model weights into git-tracked source directories.
- Do not launch a GPU job until the preflight checks pass.
- Preserve the user's existing environment; only install packages after showing the proposed command.

## Workflow

1. Identify the repository root and inspect `requirements.txt`, `scripts/env.sh`, and `README.md`.
2. Verify Python, PyTorch, CUDA, and GPU visibility:

   ```bash
   python --version
   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
   ```

3. Install the repository requirements in the active environment, using a CUDA-matched PyTorch build:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure repository roots:

   ```bash
   source scripts/env.sh
   ```

   Override only the roots that live outside the checkout:

   ```bash
   PRETRAIN_ROOT=/path/to/pretrain_weights \
   DISTILL_DATA_ROOT=/path/to/distill_data \
   ROLA_DATA_ROOT=/path/to/rola_data \
   DISTILL_OUTPUT_ROOT=/path/to/distill_outputs \
   ROLA_OUTPUT_ROOT=/path/to/rola_outputs \
   source scripts/env.sh
   ```

5. For sparse finetuning or RoLa distillation, configure the external SLA checkout before starting Python:

   ```bash
   export SLA_SRC=/absolute/path/to/SLA
   test -d "${SLA_SRC}/sparse_linear_attention"
   ```

   Standard dense and fused RoLa inference do not require `SLA_SRC`. PureSLA, unfused RoLa, and legacy INT8 sparse inference do.

6. Check the expected layout:

   ```text
   pretrain_weights/
   datasets/distill/
   datasets/rola/
   outputs/distill/
   outputs/rola/
   ```

7. Validate source and launchers without starting a job:

   ```bash
   python -m compileall -q sparkdiffusion imaginaire scripts
   bash -n scripts/env.sh scripts/distill/*.sh scripts/sparse_finetune/*.sh scripts/inference/*.sh
   ```

## Path Contract

- Wan model and tokenizer assets belong under `PRETRAIN_ROOT`.
- Distillation shards belong under `DISTILL_DATA_ROOT`.
- Sparse-finetuning shards belong under `ROLA_DATA_ROOT`.
- Training outputs belong under `DISTILL_OUTPUT_ROOT` or `ROLA_OUTPUT_ROOT`.
- Generated videos belong under `outputs/inference/` unless `OUT_ROOT` or an explicit output argument is supplied.

When a required path is missing, report the exact expected path and stop. Do not silently substitute a local absolute path or a different model variant.

