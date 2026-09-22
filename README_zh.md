<p align="center">
  <img src="assets/logo.png" alt="SparkDiffusion" width="520">
</p>

<h3 align="center">面向 DiT 视频生成的推理加速框架,实现 265× 推理加速</h3>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
</p>

<p align="center">
  <a href="https://sparkdiffusion.github.io/"><img alt="Blog" src="https://img.shields.io/badge/📖_Blog-SparkDiffusion-orange"></a>
  <a href="#"><img alt="Hugging Face" src="https://img.shields.io/badge/🤗_HuggingFace-Weights-yellow"></a>
  <a href="LICENSE.txt"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue"></a>
</p>

---

## 项目简介

**SparkDiffusion** 是面向 Diffusion Transformer(DiT)模型的视频生成推理加速
框架。它将**稀疏低秩注意力(RoLa)**、**少步蒸馏(CrossDistill)**与**自研高性能
算子**相结合,在保持生成质量的前提下,相较稠密多步基线实现**端到端 200× 以上的
推理加速**。

框架面向 Wan 2.1 与 Wan 2.2 视频扩散模型,提供端到端流程:稀疏注意力微调、
少步蒸馏,以及优化过的单样例 T2V/I2V 推理。同时提供可选的权重-激活量化,以及
无需任何外部稀疏注意力代码库的自研推理算子。

### 演示视频

<div align="center">
  <video src="https://github.com/user-attachments/assets/c7f78418-50ac-4924-9251-ef070c406ecc" width="70%" controls></video>
</div>

### 相关资源

- 📖 **博客**: [sparkdiffusion.github.io](https://sparkdiffusion.github.io/)
- 🤗 **模型权重**: _即将上线_
- 📄 **论文**:
  - SparkDiffusion: [arXiv:2609.23153](https://arxiv.org/abs/2609.23153)
  - RoLa（稀疏低秩注意力）: [arXiv:2609.06712](https://arxiv.org/abs/2609.06712)
  - CrossDistill（少步蒸馏）: [arXiv:2609.14725](https://arxiv.org/pdf/2609.14725v1)

## 核心特性

- **200× 以上推理加速**:联合稀疏注意力、低秩分解与少步蒸馏共同实现。
- **RoLa 稀疏低秩注意力**:高效的注意力设计,训练(微调/蒸馏)与推理均可使用。
- **CrossDistill 少步蒸馏**:通过轨迹级混合蒸馏平衡生成质量与多样性。
- **自研高性能算子**:位于 `sparkdiffusion/ops/`,推理时自包含、无外部依赖。
- **Wan 2.1 & Wan 2.2 支持**:覆盖 T2V 与 I2V,稠密/稀疏/蒸馏 checkpoint 可通过同一
  套推理封装脚本对比。
- **可插拔稀疏注意力注册表**:自定义注意力变体无需改动核心训练/蒸馏/推理代码即可接入。

## 仓库结构

| 路径 | 用途 |
| --- | --- |
| `sparkdiffusion/` | 模型、采样器、数据集、checkpoint、推理与算子代码 |
| `imaginaire/` | 训练框架与配置工具 |
| `scripts/sparse_finetune/` | Wan 2.1/2.2 稀疏微调启动脚本 |
| `scripts/distill/` | 已支持配置的蒸馏启动脚本 |
| `scripts/inference/` | 单样例推理的 shell 封装脚本 |
| `datasets/distill/` | 本地蒸馏数据集挂载点 |
| `datasets/rola/` | 本地稀疏微调数据集挂载点 |
| `pretrain_weights/` | 本地预训练模型挂载点 |
| `outputs/distill/` | 蒸馏输出 |
| `outputs/rola/` | 稀疏微调输出 |

权重、数据集、checkpoint 和生成的视频均有意不纳入仓库。

## 环境要求

- 带 CUDA 兼容 GPU 的 Linux
- Python 3.10 或更高版本
- 与 CUDA 匹配的 PyTorch 安装
- Triton、`flash-attn`,以及 `requirements.txt` 中列出的依赖包

在安装好与 CUDA 匹配的 PyTorch 之后,再安装 Python 依赖:

```bash
pip install -r requirements.txt
source scripts/env.sh
```

`scripts/env.sh` 会把仓库加入 `PYTHONPATH`,为 Hugging Face 和 W&B 启用离线默认
设置,并定义基于仓库相对路径的数据/输出根目录。当本地存储位于别处时,可覆盖
任意根目录:

```bash
PRETRAIN_ROOT=/path/to/pretrain_weights \
DISTILL_DATA_ROOT=/path/to/distill_data \
ROLA_DATA_ROOT=/path/to/rola_data \
DISTILL_OUTPUT_ROOT=/path/to/distill_outputs \
ROLA_OUTPUT_ROOT=/path/to/rola_outputs \
source scripts/env.sh
```

## 数据与权重

请使用以下目录布局约定:

```text
pretrain_weights/
  Wan2.1-T2V-14B/
  Wan2.1-I2V-14B-480P/
  Wan2.2-T2V-A14B/
    high_noise_model/
    low_noise_model/
datasets/
  distill/
  rola/
outputs/
  distill/
  rola/
```

具体的数据集分片名称因实验而异。使用不同布局时,稀疏微调请设置 `DATASET`,
蒸馏请设置 `DATASET_ROOT`。

## 稀疏微调

Wan 2.1:

```bash
SLA_SRC=path/to/SLA \
  bash scripts/sparse_finetune/run_finetune_2pt1.sh
```

使用原生模型仓库分别微调 Wan 2.2 高噪声与低噪声专家:

```bash
export SLA_SRC=path/to/SLA

EXPERT=high bash scripts/sparse_finetune/run_finetune_2pt2.sh
EXPERT=low bash scripts/sparse_finetune/run_finetune_2pt2.sh
```

使用 `EXPERT=joint` 走双专家联合训练路径。使用 `EXPERT=both` 依次启动高噪声和
低噪声训练。RoLa 训练启动前必须将 `SLA_SRC` 指向外部 SLA 源码目录；所有训练
脚本都会在启动 `torchrun` 前校验该目录。常用的覆盖项包括 `PRETRAINED_CKPT`、
`DATASET`、`NUM_GPUS`、`MAX_ITER`、`BATCH_SIZE`、`LR` 和 `EXPERIMENT`。

### 预训练权重(`PRETRAINED_CKPT`)

微调加载器会自动识别 checkpoint 格式并适配 state dict。请按你的格式把
`PRETRAINED_CKPT` 指向对应路径:

| 格式 | 要求的路径 |
| --- | --- |
| **原生 Wan 2.1(默认)** | 模型仓库目录,如 `pretrain_weights/Wan2.1-T2V-1.3B`,其中包含 `diffusion_pytorch_model.safetensors`(可能带 `*.index.json` 分片)。也可直接传 `.safetensors` 文件。 |
| **原生 Wan 2.2** | 对应的专家目录,如 `pretrain_weights/Wan2.2-T2V-A14B/high_noise_model` 或 `low_noise_model`,其中包含原生分片 safetensors。 |
| `.pth` / `.pt` | Wan 官方或 SparkDiffusion 训练输出的 checkpoint 文件。 |
| DCP | 含 `*.distcp` 分片的分布式 checkpoint 目录。 |

启动脚本默认加载**原生 Wan 仓库目录**。若路径或格式不对,会快速失败:当
checkpoint 匹配到的骨干参数为**零**时,加载器会直接报错(而不是静默地从随机
权重开始训练);部分匹配则打印警告。

> 注意:RoLa 稀疏参数(`proj_q`、`proj_k`、`gate_proj`、`gate_bias`)是新增的,
> 在原始 checkpoint 中本就不存在;它们从随机初始化开始,在微调中训练。只有
> *骨干*权重缺失才说明路径/格式不对。

## 蒸馏

已支持的蒸馏启动脚本归置在 `scripts/distill/` 下:

```bash
export SLA_SRC=path/to/SLA
bash scripts/distill/wan2.1_14b_t2v_480p.sh
bash scripts/distill/wan2.1_14b_i2v_480p.sh

STUDENT_CKPT=path/to/high_noise_student.pth \
STUDENT_CKPT_LOW=path/to/low_noise_student.pth \
DATASET_ROOT=path/to/distillation_dataset \
  bash scripts/distill/wan2.2_a14b_t2v_480p_joint.sh
```

每个启动脚本都使用基于仓库相对路径的默认值。若本地布局不同,可覆盖
`WAN_REPO`、`STUDENT_CKPT`、`TEACHER_CKPT`、`DATASET_ROOT`、`NEG_EMBED` 和
`OUTPUT_ROOT`。Wan 2.2 联合启动脚本会在同一进程中加载两个噪声专家
(还可用 `TEACHER_CKPT_LOW` / `STUDENT_CKPT_LOW`)。

VAE、文本编码器和 tokenizer 默认按原生 `WAN_REPO` 布局拼接,但都可以独立覆盖——
变量名与微调启动脚本一致:

- `VAE_PATH`、`T5_PATH`、`TOKENIZER_PATH`(Wan 2.1 I2V 另有 `CLIP_ENCODER`)。
- Wan 2.1 和 Wan 2.2 使用 `${WAN_REPO}/Wan2.1_VAE.pth`、
  `${WAN_REPO}/models_t5_umt5-xxl-enc-bf16.pth` 和 `${WAN_REPO}/google/umt5-xxl`。
- Wan 2.1 的原生 DiT 位于仓库根目录;Wan 2.2 的高、低噪声专家分别位于
  `${WAN_REPO}/high_noise_model` 和 `${WAN_REPO}/low_noise_model`。
- Wan 2.2 的 student checkpoint 来自独立的稀疏微调输出,不属于 `WAN_REPO`。

即使 teacher 是 dense 模型，RoLa 蒸馏也必须设置 `SLA_SRC`，因为 student 的稀疏
注意力训练依赖外部 backward kernel。启动脚本会在开始前校验该变量及所有其他
必需路径，缺失时打印清晰信息并退出。

## 单样例推理

推理封装脚本每个进程运行一条 prompt。在 Python 入口使用 `--prompt`,或将 prompt
作为最后一个位置参数传给 shell 封装脚本。公开推理路径不使用 `PROMPT_FILE`。第一个
位置参数是提供 VAE、文本编码器、tokenizer 和可选 CLIP 编码器的原生资产根目录；
使用 `DIT_PATH` 指向蒸馏后的 student checkpoint，Wan 2.2 的低噪声 student 则由
`CKPT_LOW` 指定。

Wan 2.1 蒸馏 T2V:

```bash
DIT_PATH=path/to/distill_model.pt \
  bash scripts/inference/eval_student_2pt1_distilled.sh \
  pretrain_weights/Wan2.1-T2V-14B \
  outputs/inference/wan21_t2v \
  4 fp8 14B_rola 0.1 "" \
  "A playful raccoon is seen playing an electronic guitar, strumming the strings with its front paws. The raccoon has distinctive black facial markings and a bushy tail. It sits comfortably on a small stool, its body slightly tilted as it focuses intently on the instrument. The setting is a cozy, dimly lit room with vintage posters on the walls, adding a retro vibe. The raccoon's expressive eyes convey a sense of joy and concentration. Medium close-up shot, focusing on the raccoon's face and hands interacting with the guitar."
```

Wan 2.1 蒸馏 I2V:

```bash
DIT_PATH=path/to/distill_model.pt \
  bash scripts/inference/eval_student_2pt1_distilled.sh \
  pretrain_weights/Wan2.1-I2V-14B-480P \
  outputs/inference/wan21_i2v \
  4 fp8 14B_rola 0.05 examples/i2v_input_1.jpg \
  "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

Wan 2.2 蒸馏 T2V:

```bash
DIT_PATH=path/to/distill_high_noise_model.pt \
CKPT_LOW=path/to/distill_low_noise_model.pt \
  bash scripts/inference/eval_student_2pt2_distilled.sh \
  pretrain_weights/Wan2.2-T2V-A14B \
  outputs/inference/wan22_t2v \
  4 fp8 A14B_rola 0.1 \
  "A playful raccoon is seen playing an electronic guitar, strumming the strings with its front paws. The raccoon has distinctive black facial markings and a bushy tail. It sits comfortably on a small stool, its body slightly tilted as it focuses intently on the instrument. The setting is a cozy, dimly lit room with vintage posters on the walls, adding a retro vibe. The raccoon's expressive eyes convey a sense of joy and concentration. Medium close-up shot, focusing on the raccoon's face and hands interacting with the guitar."
```

对应的 `*_diffusion.sh` 封装脚本运行原始的多步 CFG 采样器,用于 teacher/参考
对比。最后一个位置参数始终是文本 prompt。当 Wan 2.2 高噪声与低噪声 checkpoint
分开存放时,使用 `CKPT_LOW`。

常用环境变量有 `NUM_FRAMES`、`RESOLUTION`、`ASPECT_RATIO`、`SEED`、
`NUM_SAMPLES`、`OUT_ROOT`,以及用于 I2V 的 `FIXED_RESOLUTION`。`topk` 参数是
保留比例:`0.1`、`0.05` 和 `0.03` 分别对应 90%、95% 和 97% 的稀疏度。

## 算子与致谢

`sparkdiffusion/ops/fused_kernel/` 下的融合算子是 SparkDiffusion 专属的自研算子,
带有阿里巴巴版权头。RoLa 稀疏微调和蒸馏必须通过显式配置的 `SLA_SRC` 使用外部
向后兼容训练 kernel。标准 dense 与 fused RoLa 推理使用仓库内推理算子，不需要
`SLA_SRC`。

本仓库包含或改编自以下项目的思路与组件:

- [NVIDIA rCM](https://github.com/NVlabs/rcm)
- [thu-ml SLA (Sparse-Linear Attention)](https://github.com/thu-ml/SLA)
  —— RoLa 训练通过 `SLA_SRC` 使用该向后兼容稀疏注意力 kernel；
  `WanSelfAttentionPureSLA` 变体也复用了该库。
- [Hugging Face finetrainers](https://github.com/huggingface/finetrainers) —— RoLa
  稀疏 + 低秩注意力设计改编自 finetrainers 项目中使用的稀疏注意力处理器。
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [PyTorch](https://github.com/pytorch/pytorch)

相关源文件在改编属实现特定之处带有本地致谢注释。在重新分发衍生产物之前,请查阅
`LICENSE.txt` 中的仓库许可证以及上游项目的许可证。

## Checkpoint

训练 checkpoint 可能使用仓库的分布式 checkpoint 格式。推理加载支持
`sparkdiffusion.utils.model_utils.load_checkpoint_auto` 和推理封装脚本所处理的
checkpoint 布局,包括原生 Wan 仓库目录、原生 safetensors 以及受支持的 PTH/DCP
输入。Checkpoint 转换细节有意保留在相应脚本中,标准启动脚本的使用无需关心。

## 许可证

本仓库以 Apache License 2.0 发布。模型权重、数据集、上游依赖以及生成内容可能
有各自独立的许可证和使用限制。

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
