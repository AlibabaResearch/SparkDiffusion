---
name: sparkdiffusion-distill
description: Configure and launch SparkDiffusion few-step distillation for Wan 2.1 or Wan 2.2. Use when a user asks to distill a sparse student against a dense teacher, choose T2V/I2V or high/low experts, resume a distillation run, or validate its inputs.
---

# SparkDiffusion Distillation

Use this skill after the teacher, student, and distillation dataset are available. Do not use it for sparse finetuning.

## Preflight

1. Source the repository environment:

   ```bash
   source scripts/env.sh
   ```

2. Export the required external SLA checkout before starting Python or `torchrun`:

   ```bash
   export SLA_SRC=/absolute/path/to/SLA
   test -d "${SLA_SRC}/sparse_linear_attention"
   ```

   The teacher may be dense, but the trainable RoLa student requires the SLA backward kernel. Do not use a machine-specific default.
3. Select a launcher under `scripts/distill/` instead of reconstructing the `torchrun` command by hand.
4. Check all model, tokenizer, negative-embedding, and dataset paths before launching.
5. Keep W&B offline unless online logging is explicitly requested:

   ```bash
   WANDB_MODE=offline
   ```

6. Start with a short smoke run by overriding `MAX_ITER`, `SAVE_ITER`, and `NPROC_PER_NODE` where supported.

## Wan 2.1 T2V

```bash
SLA_SRC=/absolute/path/to/SLA \
WAN_REPO=pretrain_weights/Wan2.1-T2V-14B \
TEACHER_CKPT=pretrain_weights/Wan2.1-T2V-14B \
STUDENT_CKPT=outputs/rola/<finetune_job>/checkpoints/<student_model> \
DATASET_ROOT=datasets/distill/<dataset_name> \
NPROC_PER_NODE=8 \
bash scripts/distill/wan2.1_14b_t2v_480p.sh
```

Use `wan2.1_1.3b_t2v_480p.sh` for the 1.3B T2V variant.

## Wan 2.1 I2V

Use the appropriate 480p or 720p launcher and provide the image encoder:

```bash
SLA_SRC=/absolute/path/to/SLA \
WAN_REPO=pretrain_weights/Wan2.1-I2V-14B-480P \
CLIP_ENCODER=pretrain_weights/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
TEACHER_CKPT=pretrain_weights/Wan2.1-I2V-14B-480P \
STUDENT_CKPT=outputs/rola/<finetune_job>/checkpoints/<student_model> \
DATASET_ROOT=datasets/distill/<dataset_name> \
bash scripts/distill/wan2.1_14b_i2v_480p.sh
```

I2V must use an I2V-compatible checkpoint and dataset. Do not pass a T2V student to an I2V configuration.

## Wan 2.2 Joint Experts

Provide native high and low teachers plus the corresponding sparse-finetuning outputs:

```bash
export SLA_SRC=/absolute/path/to/SLA
WAN_REPO=pretrain_weights/Wan2.2-T2V-A14B
WAN_REPO=${WAN_REPO} \
TEACHER_CKPT=${WAN_REPO}/high_noise_model \
TEACHER_CKPT_LOW=${WAN_REPO}/low_noise_model \
STUDENT_CKPT=outputs/rola/<high_job>/checkpoints/<student_model> \
STUDENT_CKPT_LOW=outputs/rola/<low_job>/checkpoints/<student_model> \
DATASET_ROOT=datasets/distill/<dataset_name> \
bash scripts/distill/wan2.2_a14b_t2v_480p_joint.sh
```

The student checkpoints are Stage-1 outputs and are not stored inside `WAN_REPO`. The 720p launcher follows the same contract. Do not omit the low expert when the experiment is configured as joint.

## Checkpoints

- DCP directories preserve model, optimizer, scheduler, and trainer state for resuming.
- Model-only PTH files are portable inputs for downstream inference and stage transitions when produced by the repository's conversion/export path.
- Do not use a distilled student with the multi-step diffusion inference launcher; that is a sampling-regime mismatch.
- Record the exact teacher, student, dataset, experiment name, and output directory with every run.

