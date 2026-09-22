<p align="center">
  <img src="assets/logo.png" alt="SparkDiffusion" width="520">
</p>

<h3 align="center">A DiT video-generation acceleration framework for 265× faster inference</h3>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
</p>

<p align="center">
  <a href="https://sparkdiffusion.github.io/"><img alt="Blog" src="https://img.shields.io/badge/📖_Blog-SparkDiffusion-orange"></a>
  <a href="#"><img alt="Hugging Face" src="https://img.shields.io/badge/🤗_HuggingFace-Weights-yellow"></a>
  <a href="LICENSE.txt"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue"></a>
</p>

---

## Overview

**SparkDiffusion** is a video-generation acceleration framework for Diffusion
Transformer (DiT) models. It combines **sparse low-rank attention (RoLa)**,
**few-step distillation (CrossDistill)**, and **custom high-performance operators** to deliver **200×+ end-to-end inference speedups** over the dense multi-step baseline, while preserving generation quality.

The framework targets the Wan 2.1 and Wan 2.2 video diffusion models and provides
an end-to-end pipeline: sparse-attention finetuning, few-step distillation, and
optimized single-case T2V/I2V inference. It also ships optional weight-activation
quantization and self-developed inference operators that require no external
sparse-attention checkout.

### Demo

<video src="assets/demo.mp4" controls width="100%"></video>

[Open the demo video](assets/demo.mp4)

- 📖 **Blog**: [sparkdiffusion.github.io](https://sparkdiffusion.github.io/)
- 🤗 **Model weights**: _coming soon_
- 📄 **Papers**:
  - SparkDiffusion: [arXiv:2609.23153](https://arxiv.org/abs/2609.23153)
  - RoLa (sparse low-rank attention): [arXiv:2609.06712](https://arxiv.org/abs/2609.06712)
  - CrossDistill (few-step distillation): [arXiv:2609.14725](https://arxiv.org/pdf/2609.14725v1)

## Highlights

- **200×+ inference acceleration** through joint sparse attention, low-rank
  factorization, and few-step distillation.
- **RoLa sparse low-rank attention** — an efficient attention design usable for
  both training (finetuning/distillation) and inference.
- **CrossDistill few-step distillation** — trajectory-level hybrid distillation
  that balances generation quality and diversity.
- **Custom high-performance operators** under `sparkdiffusion/ops/`,
  self-developed and dependency-free at inference time.
- **Wan 2.1 & Wan 2.2 support** for both T2V and I2V, with dense/sparse/distilled
  checkpoints comparable through the same inference wrappers.
- **Pluggable sparse-attention registry** so custom attention variants integrate
  without touching the core training/distillation/inference code.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `sparkdiffusion/` | Model, sampler, dataset, checkpoint, inference, and operator code |
| `imaginaire/` | Training framework and configuration utilities |
| `scripts/sparse_finetune/` | Wan 2.1/2.2 sparse finetuning launchers |
| `scripts/distill/` | Distillation launchers for supported configurations |
| `scripts/inference/` | Single-case shell wrappers for inference |
| `datasets/distill/` | Local distillation dataset mount point |
| `datasets/rola/` | Local sparse-finetuning dataset mount point |
| `pretrain_weights/` | Local pretrained model mount point |
| `outputs/distill/` | Distillation outputs |
| `outputs/rola/` | Sparse-finetuning outputs |

Weights, datasets, checkpoints, and generated videos are intentionally not
included in the repository.

## Requirements

- Linux with a CUDA-capable GPU
- Python 3.10 or newer
- A CUDA-compatible PyTorch installation
- Triton, `flash-attn`, and the packages listed in `requirements.txt`

Install the Python dependencies after installing the CUDA-matched PyTorch:

```bash
pip install -r requirements.txt
source scripts/env.sh
```

`scripts/env.sh` adds the repository to `PYTHONPATH`, enables offline defaults
for Hugging Face and W&B, and defines repository-relative data/output roots.
Override any root when local storage is elsewhere:

```bash
PRETRAIN_ROOT=/path/to/pretrain_weights \
DISTILL_DATA_ROOT=/path/to/distill_data \
ROLA_DATA_ROOT=/path/to/rola_data \
DISTILL_OUTPUT_ROOT=/path/to/distill_outputs \
ROLA_OUTPUT_ROOT=/path/to/rola_outputs \
source scripts/env.sh
```

## Data and Weights

Use the following layout convention:

```text
pretrain_weights/
  Wan2.1-T2V-14B/
  Wan2.1-I2V-14B-480P/
  Wan2.2-T2V-A14B/
datasets/
  distill/
  rola/
outputs/
  distill/
  rola/
```

Wan 2.1 native repositories keep the DiT safetensors, `Wan2.1_VAE.pth`,
`models_t5_umt5-xxl-enc-bf16.pth`, and `google/umt5-xxl` directly under the
model root; I2V repositories additionally contain the native CLIP `.pth`.
Wan 2.2 native repositories keep shared assets at the root and the two DiT
experts under `high_noise_model/` and `low_noise_model/`.

The exact dataset shard names are experiment-specific. Set `DATASET` for sparse
finetuning or `DATASET_ROOT` for distillation when using a different layout.

## Sparse Finetuning

Wan 2.1:

```bash
SLA_SRC=path/to/SLA \
  bash scripts/sparse_finetune/run_finetune_2pt1.sh
```

Wan 2.2 high-noise and low-noise experts with a native model repository:

```bash
export SLA_SRC=path/to/SLA

EXPERT=high bash scripts/sparse_finetune/run_finetune_2pt2.sh
EXPERT=low bash scripts/sparse_finetune/run_finetune_2pt2.sh
```

Use `EXPERT=joint` for the joint two-expert training path. Use
`EXPERT=both` to launch high-noise and low-noise training sequentially.
RoLa training requires `SLA_SRC` to point to the external SLA checkout before
launch; every training launcher validates it before starting `torchrun`.
Important overrides include `PRETRAINED_CKPT`, `DATASET`, `NUM_GPUS`,
`MAX_ITER`, `BATCH_SIZE`, `LR`, and `EXPERIMENT`.

### Pretrained checkpoint (`PRETRAINED_CKPT`)

The finetuning loader auto-detects the checkpoint format and adapts the state
dict. Point `PRETRAINED_CKPT` at the path required by your format:

| Format | Required path |
| --- | --- |
| **Native Wan 2.1 (default)** | The model repository directory, e.g. `pretrain_weights/Wan2.1-T2V-1.3B`, containing `diffusion_pytorch_model.safetensors` (optionally sharded with a `*.index.json`). You may also pass the `.safetensors` file directly. |
| **Native Wan 2.2** | The required expert directory, e.g. `pretrain_weights/Wan2.2-T2V-A14B/high_noise_model` or `low_noise_model`, each containing native sharded safetensors. |
| `.pth` / `.pt` | A Wan-official or SparkDiffusion training checkpoint file. |
| DCP | A distributed-checkpoint directory containing `*.distcp` shards. |

By default the launchers load the **native Wan repository directory**. If the
path or format is wrong, loading fails fast: when a checkpoint matches **zero**
backbone parameters the loader raises an error (instead of silently training
from random weights), and a partial match logs a warning.

> Note: RoLa sparse parameters (`proj_q`, `proj_k`, `gate_proj`, `gate_bias`)
> are newly added and are expected to be missing from a stock checkpoint; they
> start at random init and are trained during finetuning. Only missing
> *backbone* weights indicate a wrong path/format.

## Distillation

The supported distillation launchers are grouped under `scripts/distill/`:

```bash
export SLA_SRC=path/to/SLA
bash scripts/distill/wan2.1_14b_t2v_480p.sh
bash scripts/distill/wan2.1_14b_i2v_480p.sh

STUDENT_CKPT=path/to/high_noise_student.pth \
STUDENT_CKPT_LOW=path/to/low_noise_student.pth \
DATASET_ROOT=path/to/distillation_dataset \
  bash scripts/distill/wan2.2_a14b_t2v_480p_joint.sh
```

Each launcher uses repository-relative defaults. Override `WAN_REPO`,
`STUDENT_CKPT`, `TEACHER_CKPT`, `DATASET_ROOT`, `NEG_EMBED`, and
`OUTPUT_ROOT` for a different local layout. The Wan 2.2 joint launcher loads
both noise experts in one process (also `TEACHER_CKPT_LOW` / `STUDENT_CKPT_LOW`).

The VAE, text encoder, tokenizer, and DiT paths can be overridden independently:

- `VAE_PATH`, `T5_PATH`, `TOKENIZER_PATH` (and `CLIP_ENCODER` for Wan 2.1 I2V).
- Wan 2.1 uses the native repository layout: `${WAN_REPO}/Wan2.1_VAE.pth`,
  `${WAN_REPO}/models_t5_umt5-xxl-enc-bf16.pth`, `${WAN_REPO}/google/umt5-xxl`,
  and native DiT safetensors at the repository root.
- For a native Wan 2.2 repository, set `WAN_REPO=pretrain_weights/Wan2.2-T2V-A14B`,
  `TEACHER_CKPT=${WAN_REPO}/high_noise_model`,
  `TEACHER_CKPT_LOW=${WAN_REPO}/low_noise_model`, and use the shared root assets
  `${WAN_REPO}/Wan2.1_VAE.pth`, `${WAN_REPO}/models_t5_umt5-xxl-enc-bf16.pth`,
  and `${WAN_REPO}/google/umt5-xxl` through `VAE_PATH`, `T5_PATH`, and
  `TOKENIZER_PATH`.

RoLa distillation requires `SLA_SRC` even when the teacher is dense because the
student sparse-attention path needs the external backward kernel. The launchers
validate it and every other required path before starting, then abort with a
clear message if one is missing.

## Inference

Inference wrappers run one prompt per process. Use `--prompt` in the Python
entrypoint or pass the prompt as the final positional argument to a shell
wrapper. `PROMPT_FILE` is not used by the public inference path. The first
positional argument is the native asset root for the VAE, text encoder,
tokenizer, and optional CLIP encoder; set `DIT_PATH` to the distilled student
checkpoint, and set `CKPT_LOW` for the Wan 2.2 low-noise student.

Wan 2.1 distilled T2V:

```bash
DIT_PATH=path/to/distill_model.pt \
  bash scripts/inference/eval_student_2pt1_distilled.sh \
  pretrain_weights/Wan2.1-T2V-14B \
  outputs/inference/wan21_t2v \
  4 fp8 14B_rola 0.1 "" \
  "A playful raccoon is seen playing an electronic guitar, strumming the strings with its front paws. The raccoon has distinctive black facial markings and a bushy tail. It sits comfortably on a small stool, its body slightly tilted as it focuses intently on the instrument. The setting is a cozy, dimly lit room with vintage posters on the walls, adding a retro vibe. The raccoon's expressive eyes convey a sense of joy and concentration. Medium close-up shot, focusing on the raccoon's face and hands interacting with the guitar."
```

Wan 2.1 distilled I2V:

```bash
DIT_PATH=path/to/distill_model.pt \
  bash scripts/inference/eval_student_2pt1_distilled.sh \
  pretrain_weights/Wan2.1-I2V-14B-480P \
  outputs/inference/wan21_i2v \
  4 fp8 14B_rola 0.05 examples/i2v_input_1.jpg \
  "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

Wan 2.2 distilled T2V:

```bash
DIT_PATH=path/to/distill_high_noise_model.pt \
CKPT_LOW=path/to/distill_low_noise_model.pt \
  bash scripts/inference/eval_student_2pt2_distilled.sh \
  pretrain_weights/Wan2.2-T2V-A14B \
  outputs/inference/wan22_t2v \
  4 fp8 A14B_rola 0.1 \
  "A playful raccoon is seen playing an electronic guitar, strumming the strings with its front paws. The raccoon has distinctive black facial markings and a bushy tail. It sits comfortably on a small stool, its body slightly tilted as it focuses intently on the instrument. The setting is a cozy, dimly lit room with vintage posters on the walls, adding a retro vibe. The raccoon's expressive eyes convey a sense of joy and concentration. Medium close-up shot, focusing on the raccoon's face and hands interacting with the guitar."
```

The corresponding `*_diffusion.sh` wrappers run the original multi-step CFG
sampler for teacher/reference comparisons. The final positional argument is
always the text prompt. Use `CKPT_LOW` when Wan 2.2 high- and low-noise
checkpoints are stored separately.

Common environment variables are `NUM_FRAMES`, `RESOLUTION`, `ASPECT_RATIO`,
`SEED`, `NUM_SAMPLES`, `OUT_ROOT`, and `FIXED_RESOLUTION` for I2V. The
`topk` argument is a keep ratio: `0.1`, `0.05`, and `0.03` correspond to 90%,
95%, and 97% sparsity.

## Operators

The fused operators in `sparkdiffusion/ops/fused_kernel/` are
SparkDiffusion-specific, self-developed operators and carry Alibaba copyright
headers. RoLa sparse finetuning and distillation require the external
backward-compatible training kernel selected by the explicitly configured
`SLA_SRC` environment variable. Standard dense and fused RoLa inference use the
repository's inference operators and do not require `SLA_SRC`.

## Checkpoints

Training checkpoints may use the repository's distributed checkpoint format.
Inference loading supports the checkpoint layouts handled by
`sparkdiffusion.utils.model_utils.load_checkpoint_auto` and the inference
wrappers: native Wan repository directories, native safetensors, supported
`.pth` / `.pt` checkpoints, and DCP directories. Checkpoint conversion is not
required for standard launcher usage.

## License Agreement

This repository is released under the Apache License 2.0. Model weights,
datasets, upstream dependencies, and generated content may have separate
licenses and usage restrictions.

## Acknowledgments

We learned the design and reused or adapted code from the following projects:

- [NVIDIA rCM](https://github.com/NVlabs/rcm) — the distillation implementation
  and usage workflow are based on this project.
- [thu-ml SLA (Sparse-Linear Attention)](https://github.com/thu-ml/SLA) — RoLa
  training requires this backward-compatible sparse-attention kernel through
  `SLA_SRC`; the `WanSelfAttentionPureSLA` variant also reuses this library.
- [Hugging Face finetrainers](https://github.com/huggingface/finetrainers) — the
  RoLa sparse low-rank attention design is adapted from its sparse attention
  processor.
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)

We thank the authors and contributors of these projects for making their work
available to the community. Relevant source files retain local attribution
comments where an adaptation is implementation-specific. Please review the
upstream licenses before redistributing derived artifacts.

## Citation

If you use this code or find our work valuable, please cite:

```bibtex
@misc{liu2026sparkdiffusionmitigatinghighsparsitytrap,
  title={SparkDiffusion: Mitigating the High-Sparsity Trap --- A Unified Framework for up to $265\times$ Single-GPU Acceleration of Visual Generation},
  author={Yuxi Liu and Haoyu Li and Zekun Zhang and Tengxu Sun and Yixiang Cai and Jiayong Li and Yifei Xia and Tianle Liu and Baole Ai and Ang Wang and Jiamang Wang and Lin Qu and Kai Zhang and Kun Yuan and Bin Cui},
  year={2026},
  eprint={2609.23153},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.23153},
}

@misc{zhang2026rolarotarypositionedlowranklinear,
  title={RoLA: Rotary-Positioned Low-Rank Linear Attention for Efficient Diffusion Transformers},
  author={Zekun Zhang and Yixiang Cai and Yuxi Liu and Tengxu Sun and Tianle Liu and Zhoutong Wu and Haoyu Li and Baole Ai and Ang Wang and Jiamang Wang and Lin Qu and Kun Yuan},
  year={2026},
  eprint={2609.06712},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.06712},
}

@misc{liu2026crossdistillbalancingqualitydiversity,
  title={CrossDistill: Balancing Quality and Diversity via Trajectory-Level Hybrid Few-Step Distillation},
  author={Yuxi Liu and Haoyu Li and Yixiang Cai and Tengxu Sun and Zekun Zhang and Baole Ai and Ang Wang and Jiamang Wang and Lin Qu and Kun Yuan and Kai Zhang},
  year={2026},
  eprint={2609.14725},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.14725},
}

@misc{liu2026ropeslr3dropedrivensparselowrank,
  title={RoPeSLR: 3D RoPE-driven Sparse-LowRank Attention for Efficient Diffusion Transformers},
  author={Yuxi Liu and Zekun Zhang and Yixiang Cai and Renjia Deng and Yutong He and Kun Yuan},
  year={2026},
  eprint={2605.20659},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.20659},
}
```
