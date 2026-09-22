---
name: sparkdiffusion-inference
description: Run validated SparkDiffusion inference for Wan 2.1 or Wan 2.2 T2V/I2V models. Use when a user asks to generate videos, compare dense and sparse checkpoints, choose distilled versus diffusion sampling, or debug checkpoint/path errors.
---

# SparkDiffusion Inference

Use the public shell wrappers under `scripts/inference/`. They resolve model assets, validate paths, choose the correct Python entrypoint, and write outputs under a user-selected directory.

## Choose the Correct Path

- Use `*_distilled.sh` for a few-step distilled student.
- Use `*_diffusion.sh` for the original multi-step teacher/base model.
- Use the `2pt1` wrappers for Wan 2.1.
- Use the `2pt2` wrappers for Wan 2.2.
- Supplying an image to the Wan 2.1 wrapper switches it to I2V.
- Wan 2.2 uses separate high- and low-noise experts; set `CKPT_LOW` when they are stored separately.

Do not feed a distilled checkpoint to a diffusion wrapper. The result can be noise even when the checkpoint and code are valid.

Standard dense and fused RoLa inference use repository-provided kernels and do not require `SLA_SRC`. Set it only for PureSLA, explicitly unfused RoLa, or legacy INT8 sparse-attention paths.

## Wan 2.1 T2V

Distilled student:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/inference/eval_student_2pt1_distilled.sh \
  pretrain_weights/Wan2.1-T2V-14B-Diffusers \
  outputs/inference/wan21_t2v \
  4 fp8 14B_sla "" "" \
  "A cat playing in the garden under the sun."
```

Dense or sparse diffusion baseline:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/inference/eval_student_2pt1_diffusion.sh \
  pretrain_weights/Wan2.1-T2V-14B-Diffusers \
  outputs/inference/wan21_diffusion \
  50 bf16 14B "" "" \
  "A cat walking through a sunlit garden."
```

The `topk` argument is a keep ratio: `0.1`, `0.05`, and `0.03` mean 90%, 95%, and 97% sparsity respectively.

## Wan 2.1 I2V

Pass the reference image as the seventh positional argument:

```bash
bash scripts/inference/eval_student_2pt1_distilled.sh \
  pretrain_weights/Wan2.1-I2V-14B-480P-Diffusers \
  outputs/inference/wan21_i2v \
  4 fp8 14B_sla "" examples/i2v_input_1.jpg \
  "A person walks through a forest."
```

The checkpoint root must contain the matching VAE, text encoder, tokenizer, and image encoder, unless overridden with `VAE_PATH`, `TEXT_ENCODER`, `TOKENIZER`, or `CLIP_ENCODER`.

## Wan 2.2 T2V

Distilled path:

```bash
CKPT_LOW=pretrain_weights/Wan2.2-T2V-A14B-Diffusers/transformer_2 \
bash scripts/inference/eval_student_2pt2_distilled.sh \
  pretrain_weights/Wan2.2-T2V-A14B-Diffusers \
  outputs/inference/wan22_t2v \
  4 fp8 A14B_sla "" \
  "A cat playing in the garden under the sun."
```

The total step count is split between the high- and low-noise experts. On memory-constrained GPUs, follow the wrapper's expert paging behavior rather than loading both large experts permanently.

## Preflight and Debugging

Before launching:

```bash
source scripts/env.sh
bash -n scripts/inference/*.sh
test -e pretrain_weights/<model-root>
```

If the resolver cannot find an asset, either fix the repository layout or set the documented override variable. Do not change source code for a local path problem.

Use `NUM_FRAMES`, `RESOLUTION`, `ASPECT_RATIO`, `SEED`, `NUM_SAMPLES`, `OUT_ROOT`, and `FIXED_RESOLUTION` as environment overrides. Keep generated videos outside git-tracked source files.

