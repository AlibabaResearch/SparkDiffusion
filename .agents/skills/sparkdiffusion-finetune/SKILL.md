---
name: sparkdiffusion-finetune
description: Configure and launch SparkDiffusion sparse finetuning for Wan 2.1 or Wan 2.2. Use when a user asks to train sparse attention parameters, select Wan experts, resume finetuning, or validate a finetuning configuration.
---

# SparkDiffusion Sparse Finetuning

This skill handles the sparse-finetuning stage only. Distillation is a separate workflow.

## Before Launching

1. Work from the repository root and source the shared environment:

   ```bash
   source scripts/env.sh
   ```

2. Export the required external SLA checkout before starting Python or `torchrun`:

   ```bash
   export SLA_SRC=/absolute/path/to/SLA
   test -d "${SLA_SRC}/sparse_linear_attention"
   ```

   All standard sparse-finetuning experiments use RoLa and require the SLA backward kernel. Do not use a machine-specific default.
3. Confirm the requested model, task, resolution, pretrained checkpoint, dataset shard pattern, GPU count, and output root.
4. Confirm that the checkpoint and dataset exist. Use repository-relative defaults or explicit environment variables; never insert paths from another machine.
5. Run launcher validation:

   ```bash
   bash -n scripts/sparse_finetune/*.sh
   ```

6. For a new setup, start with a short smoke run using `MAX_ITER`, `SAVE_ITER`, and a small batch size before a full run.

## Wan 2.1

Use one model and one training process:

```bash
SLA_SRC=/absolute/path/to/SLA \
TASK=t2v \
MODEL_SIZE=14b \
RESOLUTION=480p \
NUM_GPUS=2 \
MAX_ITER=20 \
bash scripts/sparse_finetune/run_finetune_2pt1.sh
```

Important overrides:

- `TASK=t2v|i2v`
- `MODEL_SIZE=1pt3b|14b`
- `RESOLUTION=480p|720p`
- `MODEL_ROOT` or `PRETRAINED_CKPT`
- `DATASET`
- `NUM_GPUS`, `CP_SIZE`, `FSDP_SHARD_SIZE`
- `MAX_ITER`, `BATCH_SIZE`, `LR`, `SAVE_ITER`
- `EXPERIMENT` when using a custom registered configuration

Wan 2.1 1.3B is T2V-only in the standard launcher. Wan 2.1 I2V requires the 14B model and its image encoder.

## Wan 2.2

The standard launcher supports high-noise, low-noise, and joint expert paths:

```bash
export SLA_SRC=/absolute/path/to/SLA
TASK=t2v RESOLUTION=480p EXPERT=high \
  bash scripts/sparse_finetune/run_finetune_2pt2.sh

TASK=t2v RESOLUTION=480p EXPERT=low \
  bash scripts/sparse_finetune/run_finetune_2pt2.sh

TASK=t2v RESOLUTION=480p EXPERT=joint \
  bash scripts/sparse_finetune/run_finetune_2pt2.sh
```

`EXPERT=both` launches high and low training sequentially. It does not mean joint training.

For the two-expert setup, keep the high and low pretrained paths explicit:

```bash
SLA_SRC=/absolute/path/to/SLA \
PRETRAINED_CKPT_HIGH=pretrain_weights/Wan2.2-T2V-A14B-Diffusers/transformer \
PRETRAINED_CKPT_LOW=pretrain_weights/Wan2.2-T2V-A14B-Diffusers/transformer_2 \
TASK=t2v RESOLUTION=480p EXPERT=joint \
bash scripts/sparse_finetune/run_finetune_2pt2.sh
```

## Outputs and Resume

Training writes distributed checkpoints under the configured job output root, normally:

```text
outputs/rola/<job>/checkpoints/iter_XXXXXXXXX/
  model/
  optim/
  scheduler/
  trainer/
```

Treat DCP as the resumable training checkpoint. Do not delete `optim`, `scheduler`, or `trainer` when resuming. Use the repository's configured `checkpoint.load_path` and `load_training_state` rather than manually copying shards.

Before passing a finetuning result to distillation or inference, inspect whether the consumer expects a model directory, a PTH file, or a DCP model directory. If the requested workflow requires a final portable PTH export, use the checkpoint-conversion workflow instead of assuming the DCP directory is a standalone weight file.

