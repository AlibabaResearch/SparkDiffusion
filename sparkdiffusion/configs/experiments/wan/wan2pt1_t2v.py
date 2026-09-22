# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hydra.core.config_store import ConfigStore

from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict
from sparkdiffusion.utils.timestep_utils import LogNormal, UniformShift

# ── Wan 2.2 A14B MoE boundary (single rf-domain source of truth) ─────────────
# Distillation is fully rf-native: rf time s∈[0,1], x_s=(1-s)x0+s·ε. Both the
# rf_t loss-sampling split and the backward_simulation boundary use the official
# boundary directly in rf (TurboDiffusion-style, no shift inversion).
# Official values differ per task (Wan2.2 wan/configs/): t2v boundary=0.875
# (sample_shift 12.0), i2v boundary=0.900 (sample_shift 5.0).
# Both tasks split the 4-step rf trajectory as high = first 2 steps, low = last 2:
#   HIGH: [1.0, 0.933781, boundary]   backward_timesteps=[0.933781]
#   LOW:  [boundary, 0.608979, 0]     backward_timesteps=[0.608979]
# (0.933781/0.608979 = legacy TrigFlow knots 1.5/1.0 via rf = sin(t)/(cos(t)+sin(t)).)
_WAN22_T2V_BOUNDARY_RF = 0.875  # official t2v boundary, rf split
_WAN22_I2V_BOUNDARY_RF = 0.9    # official i2v boundary, rf split


def build_debug_run(job):
    """Create a short debug variant for any registered training job."""
    return dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        trainer=dict(
            max_iter=25,
            logging_iter=2,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=6,
                    num_samples=5,
                ),
                every_n_sample_ema=dict(
                    every_n=6,
                    num_samples=5,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=10,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        model=dict(
            config=dict(
                critic_warmup=8,
            )
        ),
    )


cs = ConfigStore.instance()

# ══════════════════════════════════════════════════════════════════════════════
# RoLa step-distillation: load finetrainers checkpoint as teacher,
# do PCM Phase0 + DMD Phase1+ segmented distillation.
# sCM is disabled because RoLa routing is non-differentiable.
#
# Usage:
#   torchrun ... experiment=wan2pt1_1pt3B_res480p_t2v_rola_distill \
#       model.config.teacher_ckpt=/path/to/finetrainers/checkpoint \
#       checkpoint.load_path=/path/to/finetrainers/checkpoint \
#       dataloader_train.tar_path_pattern=/path/to/dataset/shard*.tar
# ══════════════════════════════════════════════════════════════════════════════
WAN2PT1_1PT3B_ROLA_DISTILL: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_distill"},
            {"override /net": "wan2pt1_1pt3B_t2v_rola"},
            {"override /net_teacher": "wan2pt1_1pt3B_t2v"},
            {"override /net_fake_score": "wan2pt1_1pt3B_t2v"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {"override /optimizer_fake_score": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                    "viz_online_sampling_distill",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="Wan_RoLa",
            name="wan2pt1_1pt3B_res480p_t2v_rola_distill",
        ),
        optimizer=dict(
            lr=2e-6,
            weight_decay=0.01,
            betas=(0.0, 0.999),
        ),
        model=dict(
            config=dict(
                # sCM is disabled because RoLa routing is non-differentiable.
                loss_scale=0.0,
                loss_scale_dmd=0,
                loss_scale_tdm=1.0,
                fsdp_shard_size=8,
                resolution="480p",
                p_G=L(LogNormal)(p_mean=0.7, p_std=1.6),
                p_D=L(UniformShift)(shift=5.0),
                max_simulation_steps_fake=4,
                state_t=21,
                sigma_max=1600,
                grad_clip=False,
                rectified_flow_t_scaling_factor=1000.0,
                student_update_freq=5,
                critic_warmup=0,
                # Segmented distillation: PCM Phase 0 + DMD Phase 1+
                loss_scale_div=10.0,
                div_mode="pcm_random",
                optimizer_fake_score=dict(
                    lr=4e-7,
                    weight_decay=0.01,
                    betas=(0.0, 0.999),
                ),
                tokenizer=dict(vae_pth="pretrain_weights/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
                text_encoder_path="pretrain_weights/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                tokenizer_path="pretrain_weights/Wan2.1-T2V-1.3B/google/umt5-xxl",
                # Set via CLI: model.config.teacher_ckpt=/path/to/finetrainers/ckpt
                teacher_ckpt="",
                neg_embed_path="",
                teacher_guidance=5.0,
                precision="bfloat16",
                net=dict(sac_config=dict(mode="block_wise")),
            )
        ),
        checkpoint=dict(
            save_iter=500,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=10_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(every_n=1000, num_samples=1, run_at_start=True),
                every_n_sample_ema=dict(every_n=1000, num_samples=1, run_at_start=True),
            ),
            grad_accum_iter=2,
        ),
        model_parallel=dict(context_parallel_size=1),
        dataloader_train=dict(
            tar_path_pattern="datasets/distill/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard*.tar",
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)

WAN2PT1_14B_ROLA_DISTILL: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_distill"},
            {"override /net": "wan2pt1_14B_t2v_rola"},
            {"override /net_teacher": "wan2pt1_14B_t2v"},
            {"override /net_fake_score": "wan2pt1_14B_t2v"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {"override /optimizer_fake_score": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                    "viz_online_sampling_distill",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="Wan_RoLa",
            name="wan2pt1_14B_res480p_t2v_rola_distill",
        ),
        optimizer=dict(
            lr=2e-6,
            weight_decay=0.01,
            betas=(0.0, 0.999),
        ),
        model=dict(
            config=dict(
                # sCM is disabled because RoLa routing is non-differentiable.
                loss_scale=0.0,
                loss_scale_dmd=0,
                loss_scale_tdm=1.0,
                fsdp_shard_size=8,
                resolution="480p",
                p_G=L(LogNormal)(p_mean=0.7, p_std=1.6),
                p_D=L(UniformShift)(shift=5.0),
                max_simulation_steps_fake=4,
                state_t=21,
                sigma_max=1600,
                grad_clip=False,
                rectified_flow_t_scaling_factor=1000.0,
                student_update_freq=10,
                critic_warmup=0,
                # Segmented distillation: PCM Phase 0 + DMD Phase 1+
                loss_scale_div=10.0,
                div_mode="pcm_random",
                optimizer_fake_score=dict(
                    lr=4e-7,
                    weight_decay=0.01,
                    betas=(0.0, 0.999),
                ),
                tokenizer=dict(vae_pth="pretrain_weights/Wan2.1-T2V-14B/Wan2.1_VAE.pth"),
                text_encoder_path="pretrain_weights/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
                tokenizer_path="pretrain_weights/Wan2.1-T2V-14B/google/umt5-xxl",
                # Set via CLI: model.config.teacher_ckpt=/path/to/finetrainers/ckpt
                teacher_ckpt="",
                neg_embed_path="",
                teacher_guidance=5.0,
                precision="bfloat16",
                net=dict(sac_config=dict(mode="block_wise")),
            )
        ),
        checkpoint=dict(
            save_iter=500,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=10_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(every_n=1000, num_samples=1, run_at_start=True),
                every_n_sample_ema=dict(every_n=1000, num_samples=1, run_at_start=True),
            ),
            grad_accum_iter=2,
        ),
        model_parallel=dict(context_parallel_size=1),
        dataloader_train=dict(
            tar_path_pattern="datasets/distill/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard*.tar",
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)

rola_job_list = [WAN2PT1_1PT3B_ROLA_DISTILL, WAN2PT1_14B_ROLA_DISTILL]
for job in rola_job_list:
    cs.store(group="experiment", package="_global_", name=job["job"]["name"], node=job)
    cs.store(group="experiment", package="_global_", name=job["job"]["name"] + "_debug", node=build_debug_run(job))


# ══════════════════════════════════════════════════════════════════════════════
# Sparse Finetuning: flow-matching velocity loss with two-stage training.
# Stage1: only sparse RoLa parameters (proj_q/k, gate_proj, gate_bias, etc.)
# Stage2: full model unfreeze
#
# Output checkpoint is directly loadable by T2VDistillModel as student_ckpt.
#
# Usage:
#   torchrun ... experiment=wan2pt1_1pt3B_res480p_t2v_finetune \
#       model.config.pretrained_ckpt=/path/to/Wan2.1-T2V-1.3B-diffusers \
#       dataloader_train.tar_path_pattern=/path/to/dataset/shard*.tar
# ══════════════════════════════════════════════════════════════════════════════
WAN2PT1_1PT3B_RES480P_T2V_FINETUNE: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_finetune"},
            {"override /net": "wan2pt1_1pt3B_t2v_rola"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="Finetune_Wan",
            name="wan2pt1_1pt3B_res480p_t2v_finetune",
        ),
        optimizer=dict(
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        ),
        model=dict(
            config=dict(
                fsdp_shard_size=8,
                resolution="480p",
                state_t=21,
                rectified_flow_t_scaling_factor=1000.0,
                precision="bfloat16",
                loss_scale=1.0,
                # Two-stage training
                stage1_steps=0,
                stage1_mse_weight=1.0,
                # Pretrained checkpoint (set via CLI)
                pretrained_ckpt="",
                tokenizer=dict(vae_pth="pretrain_weights/Wan2.1-T2V-1.3B/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
                text_encoder_path="pretrain_weights/Wan2.1-T2V-1.3B/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                tokenizer_path="pretrain_weights/Wan2.1-T2V-1.3B/Wan2.1-T2V-1.3B/google/umt5-xxl",
                neg_embed_path="",
                net=dict(sac_config=dict(mode="block_wise")),
            )
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=20_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(every_n=2000, num_samples=2, run_at_start=True),
            ),
            grad_accum_iter=2,
        ),
        model_parallel=dict(context_parallel_size=1),
        dataloader_train=dict(
            tar_path_pattern="datasets/distill/wan2pt1_t2v_1pt3b_480p/shard*.tar",
            batch_size=2,
        ),
    ),
    flags={"allow_objects": True},
)

WAN2PT1_14B_RES480P_T2V_FINETUNE: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_finetune"},
            {"override /net": "wan2pt1_14B_t2v_rola"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="Finetune_Wan",
            name="wan2pt1_14B_res480p_t2v_finetune",
        ),
        optimizer=dict(
            lr=5e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        ),
        model=dict(
            config=dict(
                fsdp_shard_size=8,
                resolution="480p",
                state_t=21,
                flow_shift=5.0,
                rectified_flow_t_scaling_factor=1000.0,
                precision="bfloat16",
                loss_scale=1.0,
                stage1_steps=0,
                stage1_mse_weight=1.0,
                pretrained_ckpt="",
                tokenizer=dict(vae_pth="pretrain_weights/Wan2.1-T2V-14B/Wan2.1-T2V-14B/Wan2.1_VAE.pth"),
                text_encoder_path="pretrain_weights/Wan2.1-T2V-14B/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
                tokenizer_path="pretrain_weights/Wan2.1-T2V-14B/Wan2.1-T2V-14B/google/umt5-xxl",
                neg_embed_path="",
                net=dict(sac_config=dict(mode="block_wise")),
            )
        ),
        checkpoint=dict(
            save_iter=2000,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=20_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(every_n=2000, num_samples=1, run_at_start=True),
            ),
            grad_accum_iter=4,
        ),
        model_parallel=dict(context_parallel_size=4),
        dataloader_train=dict(
            tar_path_pattern="datasets/distill/wan2pt1_t2v_14b_480p/shard*.tar",
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)

finetune_job_list = [WAN2PT1_1PT3B_RES480P_T2V_FINETUNE, WAN2PT1_14B_RES480P_T2V_FINETUNE]
for job in finetune_job_list:
    cs.store(group="experiment", package="_global_", name=job["job"]["name"], node=job)
    cs.store(group="experiment", package="_global_", name=job["job"]["name"] + "_debug", node=build_debug_run(job))


# ══════════════════════════════════════════════════════════════════════════════
# Wan 2.1 14B I2V (image-to-video, with CLIP) SLA finetune + distillation — single model (not MoE)
# Unlike 2.2, 2.1 I2V adds a CLIP branch (first frame -> CLIP -> img_emb -> cross-attn).
# is_i2v=True builds the first-frame condition y; i2v_clip_encoder_path enables online CLIP feature extraction.
# net=wan2pt1_14B_i2v_rola(in_dim=36, model_type=i2v, image_dim=1280).
# ══════════════════════════════════════════════════════════════════════════════
_WAN21_I2V_CLIP = "pretrain_weights/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
_WAN21_I2V_CLIP_720P = "pretrain_weights/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"

WAN2PT1_14B_RES480P_I2V_FINETUNE: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_finetune"},
            {"override /net": "wan2pt1_14B_i2v_rola"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="Finetune_Wan_I2V",
            name="wan2pt1_14B_res480p_i2v_finetune",
        ),
        optimizer=dict(
            lr=5e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        ),
        model=dict(
            config=dict(
                is_i2v=True,
                flow_shift=3.0,
                i2v_clip_encoder_path=_WAN21_I2V_CLIP,
                fsdp_shard_size=8,
                resolution="480p",
                state_t=21,
                rectified_flow_t_scaling_factor=1000.0,
                precision="bfloat16",
                loss_scale=1.0,
                stage1_steps=0,
                stage1_mse_weight=1.0,
                pretrained_ckpt="pretrain_weights/Wan2.1-I2V-14B-480P",
                tokenizer=dict(vae_pth="pretrain_weights/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth"),
                text_encoder_path="pretrain_weights/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
                tokenizer_path="pretrain_weights/Wan2.1-I2V-14B-480P/google/umt5-xxl",
                neg_embed_path="",
                net=dict(sac_config=dict(mode="block_wise")),
            )
        ),
        checkpoint=dict(
            save_iter=2000,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=20_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(every_n=2000, num_samples=1, run_at_start=True),
            ),
            grad_accum_iter=4,
        ),
        model_parallel=dict(context_parallel_size=4),
        dataloader_train=dict(
            tar_path_pattern="datasets/distill/wan2pt1_i2v_14b_480p/shard*.tar",
            batch_size=1,
            include_first_frame_rgb=True,
        ),
    ),
    flags={"allow_objects": True},
)

WAN2PT1_14B_RES480P_I2V_ROLA_DISTILL: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_distill"},
            {"override /net": "wan2pt1_14B_i2v_rola"},
            {"override /net_teacher": "wan2pt1_14B_i2v"},
            {"override /net_fake_score": "wan2pt1_14B_i2v"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {"override /optimizer_fake_score": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                    "viz_online_sampling_distill",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="Wan_I2V_ROLA",
            name="wan2pt1_14B_res480p_i2v_rola_distill",
        ),
        optimizer=dict(
            lr=2e-6,
            weight_decay=0.01,
            betas=(0.0, 0.999),
        ),
        model=dict(
            config=dict(
                is_i2v=True,
                i2v_clip_encoder_path=_WAN21_I2V_CLIP,
                loss_scale=0.0,
                loss_scale_dmd=0,
                loss_scale_tdm=1.0,
                fsdp_shard_size=8,
                resolution="480p",
                p_G=L(LogNormal)(p_mean=0.7, p_std=1.6),
                p_D=L(UniformShift)(shift=3.0),
                teacher_timestep_shift=3.0,
                max_simulation_steps_fake=4,
                state_t=21,
                sigma_max=1600,
                grad_clip=False,
                rectified_flow_t_scaling_factor=1000.0,
                student_update_freq=10,
                critic_warmup=0,
                loss_scale_div=10.0,
                div_mode="pcm_random",
                optimizer_fake_score=dict(
                    lr=4e-7,
                    weight_decay=0.01,
                    betas=(0.0, 0.999),
                ),
                tokenizer=dict(vae_pth="pretrain_weights/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth"),
                text_encoder_path="pretrain_weights/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
                tokenizer_path="pretrain_weights/Wan2.1-I2V-14B-480P/google/umt5-xxl",
                teacher_ckpt="pretrain_weights/Wan2.1-I2V-14B-480P",
                neg_embed_path="",
                teacher_guidance=5.0,
                precision="bfloat16",
                net=dict(sac_config=dict(mode="block_wise")),
            )
        ),
        checkpoint=dict(
            save_iter=500,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=10_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(every_n=1000, num_samples=1, run_at_start=True),
                every_n_sample_ema=dict(every_n=1000, num_samples=1, run_at_start=True),
            ),
            grad_accum_iter=2,
        ),
        model_parallel=dict(context_parallel_size=1),
        dataloader_train=dict(
            tar_path_pattern="datasets/distill/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard*.tar",
            batch_size=1,
            include_first_frame_rgb=True,
        ),
    ),
    flags={"allow_objects": True},
)

wan2pt1_i2v_job_list = [WAN2PT1_14B_RES480P_I2V_FINETUNE, WAN2PT1_14B_RES480P_I2V_ROLA_DISTILL]
for job in wan2pt1_i2v_job_list:
    cs.store(group="experiment", package="_global_", name=job["job"]["name"], node=job)
    cs.store(group="experiment", package="_global_", name=job["job"]["name"] + "_debug", node=build_debug_run(job))


# ══════════════════════════════════════════════════════════════════════════════
# Wan 2.1 14B 720p distillation — derived from the matching 480p config, overriding resolution and training cadence.
# Training tensor shapes come from the dataset's pre-encoded latents; resolution only determines the frame size for
# sampling visualization (sparkdiffusion/callbacks/every_n_draw_distill.py: resolution2hw).
# ══════════════════════════════════════════════════════════════════════════════
WAN2PT1_14B_RES720P_T2V_ROLA_DISTILL: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{WAN2PT1_14B_ROLA_DISTILL['job']['name']}",
            "_self_",
        ],
        job=dict(
            group="Wan_RoLa",
            name="wan2pt1_14B_res720p_t2v_rola_distill",
        ),
        model=dict(config=dict(resolution="720p")),
        dataloader_train=dict(
            tar_path_pattern="datasets/distill/Wan2.1_14B_720p_16:9_Euler-step100_shift-5.0_cfg-5.0_seed-0_250K/shard*.tar",
        ),
        checkpoint=dict(save_iter=500),
        trainer=dict(grad_accum_iter=1),
    ),
    flags={"allow_objects": True},
)

WAN2PT1_14B_RES720P_I2V_ROLA_DISTILL: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{WAN2PT1_14B_RES480P_I2V_ROLA_DISTILL['job']['name']}",
            "_self_",
        ],
        job=dict(
            group="Wan_I2V_ROLA",
            name="wan2pt1_14B_res720p_i2v_rola_distill",
        ),
        model=dict(
            config=dict(
                resolution="720p",
                i2v_clip_encoder_path=_WAN21_I2V_CLIP_720P,
                p_D=L(UniformShift)(shift=5.0),
                teacher_timestep_shift=5.0,
                tokenizer=dict(vae_pth="pretrain_weights/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth"),
                text_encoder_path="pretrain_weights/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
                tokenizer_path="pretrain_weights/Wan2.1-I2V-14B-720P/google/umt5-xxl",
                teacher_ckpt="pretrain_weights/Wan2.1-I2V-14B-720P",
            )
        ),
        dataloader_train=dict(
            tar_path_pattern="datasets/distill/Wan2.1_14B_720p_16:9_Euler-step100_shift-5.0_cfg-5.0_seed-0_250K/shard*.tar",
        ),
        checkpoint=dict(save_iter=500),
        trainer=dict(grad_accum_iter=1),
    ),
    flags={"allow_objects": True},
)

wan2pt1_720p_distill_job_list = [WAN2PT1_14B_RES720P_T2V_ROLA_DISTILL, WAN2PT1_14B_RES720P_I2V_ROLA_DISTILL]
for job in wan2pt1_720p_distill_job_list:
    cs.store(group="experiment", package="_global_", name=job["job"]["name"], node=job)
    cs.store(group="experiment", package="_global_", name=job["job"]["name"] + "_debug", node=build_debug_run(job))


# ══════════════════════════════════════════════════════════════════════════════
# Wan 2.2 T2V Sparse Finetuning
#
# 2.2 uses dual-model (high-noise + low-noise). For finetuning we train
# each model separately. Specify which one via pretrained_ckpt path:
#   .../transformer   = high-noise model
#   .../transformer_2 = low-noise model
#
# Usage:
#   High-noise: torchrun ... experiment=wan2pt2_A14B_res480p_t2v_finetune_high_noise
#   Low-noise:  torchrun ... experiment=wan2pt2_A14B_res480p_t2v_finetune_low_noise
# ══════════════════════════════════════════════════════════════════════════════
# Wan 2.2 dual-model finetune: high-noise model (t > boundary in RF domain)
WAN2PT2_A14B_RES480P_T2V_FINETUNE_HIGH: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_finetune"},
            {"override /net": "wan2pt2_A14B_t2v_rola"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="Finetune_Wan2pt2",
            name="wan2pt2_A14B_res480p_t2v_finetune_high_noise",
        ),
        optimizer=dict(
            lr=5e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        ),
        model=dict(
            config=dict(
                fsdp_shard_size=8,
                resolution="480p",
                state_t=21,
                rectified_flow_t_scaling_factor=1000.0,
                precision="bfloat16",
                loss_scale=1.0,
                stage1_steps=0,
                stage1_mse_weight=1.0,
                rf_t_min=0.3684210526,
                rf_t_max=1.0,
                flow_shift=12.0,
                pretrained_ckpt="pretrain_weights/Wan2.2-T2V-A14B-Diffusers/transformer",
                tokenizer=dict(vae_pth="pretrain_weights/Wan2.2-T2V-A14B-Diffusers/vae"),
                text_encoder_path="pretrain_weights/Wan2.2-T2V-A14B-Diffusers/text_encoder",
                tokenizer_path="pretrain_weights/Wan2.2-T2V-A14B-Diffusers/tokenizer",
                neg_embed_path="",
                net=dict(sac_config=dict(mode="block_wise")),
            )
        ),
        checkpoint=dict(
            save_iter=2000,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=20_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(every_n=2000, num_samples=1, run_at_start=True),
            ),
            grad_accum_iter=4,
        ),
        model_parallel=dict(context_parallel_size=4),
        dataloader_train=dict(
            tar_path_pattern="datasets/distill/wan2pt2_t2v_a14b_480p/shard*.tar",
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)

# Wan 2.2 dual-model finetune: low-noise model (t <= boundary in RF domain)
WAN2PT2_A14B_RES480P_T2V_FINETUNE_LOW: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_finetune"},
            {"override /net": "wan2pt2_A14B_t2v_rola"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="Finetune_Wan2pt2",
            name="wan2pt2_A14B_res480p_t2v_finetune_low_noise",
        ),
        optimizer=dict(
            lr=5e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        ),
        model=dict(
            config=dict(
                fsdp_shard_size=8,
                resolution="480p",
                state_t=21,
                rectified_flow_t_scaling_factor=1000.0,
                precision="bfloat16",
                loss_scale=1.0,
                stage1_steps=0,
                stage1_mse_weight=1.0,
                rf_t_min=0.0,
                rf_t_max=0.3684210526,
                flow_shift=12.0,
                pretrained_ckpt="pretrain_weights/Wan2.2-T2V-A14B-Diffusers/transformer_2",
                tokenizer=dict(vae_pth="pretrain_weights/Wan2.2-T2V-A14B-Diffusers/vae"),
                text_encoder_path="pretrain_weights/Wan2.2-T2V-A14B-Diffusers/text_encoder",
                tokenizer_path="pretrain_weights/Wan2.2-T2V-A14B-Diffusers/tokenizer",
                neg_embed_path="",
                net=dict(sac_config=dict(mode="block_wise")),
            )
        ),
        checkpoint=dict(
            save_iter=2000,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=20_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(every_n=2000, num_samples=1, run_at_start=True),
            ),
            grad_accum_iter=4,
        ),
        model_parallel=dict(context_parallel_size=4),
        dataloader_train=dict(
            tar_path_pattern="datasets/distill/wan2pt2_t2v_a14b_480p/shard*.tar",
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)

wan2pt2_finetune_job_list = [WAN2PT2_A14B_RES480P_T2V_FINETUNE_HIGH, WAN2PT2_A14B_RES480P_T2V_FINETUNE_LOW]
for job in wan2pt2_finetune_job_list:
    cs.store(group="experiment", package="_global_", name=job["job"]["name"], node=job)
    cs.store(group="experiment", package="_global_", name=job["job"]["name"] + "_debug", node=build_debug_run(job))


WAN2PT2_A14B_RES480P_T2V_FINETUNE_JOINT: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_finetune"},
            {"override /net": "wan2pt2_A14B_t2v_rola"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="Finetune_Wan2pt2",
            name="wan2pt2_A14B_res480p_t2v_finetune_joint",
        ),
        optimizer=dict(
            lr=5e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        ),
        model=dict(
            config=dict(
                joint_dual_expert=True,
                rf_split_t=_WAN22_T2V_BOUNDARY_RF,
                fsdp_shard_size=8,
                resolution="480p",
                state_t=21,
                rectified_flow_t_scaling_factor=1000.0,
                precision="bfloat16",
                loss_scale=1.0,
                stage1_steps=0,
                stage1_mse_weight=1.0,
                rf_t_min=0.0,
                rf_t_max=1.0,
                flow_shift=12.0,
                pretrained_ckpt_high="pretrain_weights/Wan2.2-T2V-A14B-Diffusers/transformer",
                pretrained_ckpt_low="pretrain_weights/Wan2.2-T2V-A14B-Diffusers/transformer_2",
                tokenizer=dict(vae_pth="pretrain_weights/Wan2.2-T2V-A14B-Diffusers/vae"),
                text_encoder_path="pretrain_weights/Wan2.2-T2V-A14B-Diffusers/text_encoder",
                tokenizer_path="pretrain_weights/Wan2.2-T2V-A14B-Diffusers/tokenizer",
                neg_embed_path="",
                net=dict(sac_config=dict(mode="block_wise")),
            )
        ),
        checkpoint=dict(
            save_iter=2000,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=20_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(every_n=2000, num_samples=1, run_at_start=True),
            ),
            grad_accum_iter=4,
        ),
        model_parallel=dict(context_parallel_size=4),
        dataloader_train=dict(
            tar_path_pattern="datasets/distill/wan2pt2_t2v_a14b_480p/shard*.tar",
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)
cs.store(
    group="experiment",
    package="_global_",
    name=WAN2PT2_A14B_RES480P_T2V_FINETUNE_JOINT["job"]["name"],
    node=WAN2PT2_A14B_RES480P_T2V_FINETUNE_JOINT,
)
cs.store(
    group="experiment",
    package="_global_",
    name=WAN2PT2_A14B_RES480P_T2V_FINETUNE_JOINT["job"]["name"] + "_debug",
    node=build_debug_run(WAN2PT2_A14B_RES480P_T2V_FINETUNE_JOINT),
)


# ══════════════════════════════════════════════════════════════════════════════
# Wan 2.2 A14B I2V (image-to-video) SLA finetune — dual expert
# Same structure as T2V finetune; the only differences: net=i2v_rola(in_dim=36), is_i2v=True (builds first-frame
# condition y), and pretrained_ckpt points to the official Wan2.2-I2V-A14B (transformer=high / transformer_2=low).
# ══════════════════════════════════════════════════════════════════════════════
WAN2PT2_A14B_RES480P_I2V_FINETUNE_HIGH: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_finetune"},
            {"override /net": "wan2pt2_A14B_i2v_rola"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="Finetune_Wan2pt2_I2V",
            name="wan2pt2_A14B_res480p_i2v_finetune_high_noise",
        ),
        optimizer=dict(
            lr=5e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        ),
        model=dict(
            config=dict(
                is_i2v=True,
                fsdp_shard_size=8,
                resolution="480p",
                state_t=21,
                rectified_flow_t_scaling_factor=1000.0,
                precision="bfloat16",
                loss_scale=1.0,
                stage1_steps=0,
                stage1_mse_weight=1.0,
                rf_t_min=0.6428571429,
                flow_shift=5.0,
                rf_t_max=1.0,
                pretrained_ckpt="pretrain_weights/Wan2.2-I2V-A14B-Diffusers/transformer",
                tokenizer=dict(vae_pth="pretrain_weights/Wan2.2-I2V-A14B-Diffusers/vae"),
                text_encoder_path="pretrain_weights/Wan2.2-I2V-A14B-Diffusers/text_encoder",
                tokenizer_path="pretrain_weights/Wan2.2-I2V-A14B-Diffusers/tokenizer",
                neg_embed_path="",
                net=dict(sac_config=dict(mode="block_wise")),
            )
        ),
        checkpoint=dict(
            save_iter=2000,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=20_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(every_n=2000, num_samples=1, run_at_start=True),
            ),
            grad_accum_iter=4,
        ),
        model_parallel=dict(context_parallel_size=4),
        dataloader_train=dict(
            tar_path_pattern="datasets/distill/wan2pt2_i2v_a14b_480p/shard*.tar",
            batch_size=1,
            include_first_frame_rgb=True,
        ),
    ),
    flags={"allow_objects": True},
)

WAN2PT2_A14B_RES480P_I2V_FINETUNE_LOW: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_finetune"},
            {"override /net": "wan2pt2_A14B_i2v_rola"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="Finetune_Wan2pt2_I2V",
            name="wan2pt2_A14B_res480p_i2v_finetune_low_noise",
        ),
        optimizer=dict(
            lr=5e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        ),
        model=dict(
            config=dict(
                is_i2v=True,
                fsdp_shard_size=8,
                resolution="480p",
                state_t=21,
                rectified_flow_t_scaling_factor=1000.0,
                precision="bfloat16",
                loss_scale=1.0,
                stage1_steps=0,
                stage1_mse_weight=1.0,
                rf_t_min=0.0,
                rf_t_max=0.6428571429,
                flow_shift=5.0,
                pretrained_ckpt="pretrain_weights/Wan2.2-I2V-A14B-Diffusers/transformer_2",
                tokenizer=dict(vae_pth="pretrain_weights/Wan2.2-I2V-A14B-Diffusers/vae"),
                text_encoder_path="pretrain_weights/Wan2.2-I2V-A14B-Diffusers/text_encoder",
                tokenizer_path="pretrain_weights/Wan2.2-I2V-A14B-Diffusers/tokenizer",
                neg_embed_path="",
                net=dict(sac_config=dict(mode="block_wise")),
            )
        ),
        checkpoint=dict(
            save_iter=2000,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=20_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(every_n=2000, num_samples=1, run_at_start=True),
            ),
            grad_accum_iter=4,
        ),
        model_parallel=dict(context_parallel_size=4),
        dataloader_train=dict(
            tar_path_pattern="datasets/distill/wan2pt2_i2v_a14b_480p/shard*.tar",
            batch_size=1,
            include_first_frame_rgb=True,
        ),
    ),
    flags={"allow_objects": True},
)

wan2pt2_i2v_finetune_job_list = [WAN2PT2_A14B_RES480P_I2V_FINETUNE_HIGH, WAN2PT2_A14B_RES480P_I2V_FINETUNE_LOW]
for job in wan2pt2_i2v_finetune_job_list:
    cs.store(group="experiment", package="_global_", name=job["job"]["name"], node=job)
    cs.store(group="experiment", package="_global_", name=job["job"]["name"] + "_debug", node=build_debug_run(job))


WAN2PT2_A14B_RES480P_I2V_FINETUNE_JOINT: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_finetune"},
            {"override /net": "wan2pt2_A14B_i2v_rola"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="Finetune_Wan2pt2_I2V",
            name="wan2pt2_A14B_res480p_i2v_finetune_joint",
        ),
        optimizer=dict(
            lr=5e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        ),
        model=dict(
            config=dict(
                is_i2v=True,
                joint_dual_expert=True,
                rf_split_t=_WAN22_I2V_BOUNDARY_RF,
                fsdp_shard_size=8,
                resolution="480p",
                state_t=21,
                rectified_flow_t_scaling_factor=1000.0,
                precision="bfloat16",
                loss_scale=1.0,
                stage1_steps=0,
                stage1_mse_weight=1.0,
                rf_t_min=0.0,
                rf_t_max=1.0,
                flow_shift=5.0,
                pretrained_ckpt_high="pretrain_weights/Wan2.2-I2V-A14B-Diffusers/transformer",
                pretrained_ckpt_low="pretrain_weights/Wan2.2-I2V-A14B-Diffusers/transformer_2",
                tokenizer=dict(vae_pth="pretrain_weights/Wan2.2-I2V-A14B-Diffusers/vae"),
                text_encoder_path="pretrain_weights/Wan2.2-I2V-A14B-Diffusers/text_encoder",
                tokenizer_path="pretrain_weights/Wan2.2-I2V-A14B-Diffusers/tokenizer",
                neg_embed_path="",
                net=dict(sac_config=dict(mode="block_wise")),
            )
        ),
        checkpoint=dict(
            save_iter=2000,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=20_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(every_n=2000, num_samples=1, run_at_start=True),
            ),
            grad_accum_iter=4,
        ),
        model_parallel=dict(context_parallel_size=4),
        dataloader_train=dict(
            tar_path_pattern="datasets/distill/wan2pt2_i2v_a14b_480p/shard*.tar",
            batch_size=1,
            include_first_frame_rgb=True,
        ),
    ),
    flags={"allow_objects": True},
)
cs.store(
    group="experiment",
    package="_global_",
    name=WAN2PT2_A14B_RES480P_I2V_FINETUNE_JOINT["job"]["name"],
    node=WAN2PT2_A14B_RES480P_I2V_FINETUNE_JOINT,
)
cs.store(
    group="experiment",
    package="_global_",
    name=WAN2PT2_A14B_RES480P_I2V_FINETUNE_JOINT["job"]["name"] + "_debug",
    node=build_debug_run(WAN2PT2_A14B_RES480P_I2V_FINETUNE_JOINT),
)


# ══════════════════════════════════════════════════════════════════════════════
# Wan 2.2 A14B T2V RoLa joint distillation — dual expert
# ══════════════════════════════════════════════════════════════════════════════
WAN2PT2_A14B_RES480P_T2V_ROLA_DISTILL_JOINT: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_distill"},
            {"override /net": "wan2pt2_A14B_t2v_rola"},
            {"override /net_teacher": "wan2pt2_A14B_t2v"},
            {"override /net_fake_score": "wan2pt2_A14B_t2v"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {"override /optimizer_fake_score": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                    "viz_online_sampling_distill",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="Wan2pt2_RoLa",
            name="wan2pt2_A14B_res480p_t2v_rola_distill_joint",
        ),
        optimizer=dict(
            lr=2e-6,
            weight_decay=0.01,
            betas=(0.0, 0.999),
        ),
        model=dict(
            config=dict(
                joint_wan22_t2v=True,
                joint_boundary_rf=_WAN22_T2V_BOUNDARY_RF,
                joint_boundary_step_index=2,
                joint_backward_timesteps=[0.933781, _WAN22_T2V_BOUNDARY_RF, 0.608979],
                joint_backward_start_t=-1.0,
                joint_backward_end_t=0.0,
                joint_low_loss_scale_div=0.0,
                loss_scale=0.0,
                loss_scale_dmd=0,
                loss_scale_tdm=1.0,
                fsdp_shard_size=8,
                resolution="480p",
                p_G=L(LogNormal)(p_mean=0.7, p_std=1.6),
                p_D=L(UniformShift)(shift=12.0),
                teacher_timestep_shift=12.0,
                max_simulation_steps_fake=4,
                state_t=21,
                sigma_max=1600,
                grad_clip=False,
                rectified_flow_t_scaling_factor=1000.0,
                student_update_freq=10,
                critic_warmup=0,
                loss_scale_div=10.0,
                div_mode="pcm_random",
                rf_t_min=0.0,
                rf_t_max=1.0,
                backward_simulation_start_t=-1.0,
                backward_simulation_end_t=0.0,
                backward_timesteps=[0.933781, _WAN22_T2V_BOUNDARY_RF, 0.608979],
                viz_full_timesteps=[0.933781, _WAN22_T2V_BOUNDARY_RF, 0.608979],
                optimizer_fake_score=dict(
                    lr=4e-7,
                    weight_decay=0.01,
                    betas=(0.0, 0.999),
                ),
                tokenizer=dict(vae_pth="pretrain_weights/Wan2.2-T2V-A14B/Wan2.1_VAE.pth"),
                text_encoder_path="pretrain_weights/Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth",
                tokenizer_path="pretrain_weights/Wan2.2-T2V-A14B/google/umt5-xxl",
                teacher_ckpt="pretrain_weights/Wan2.2-T2V-A14B/high_noise_model",
                teacher_ckpt_low="pretrain_weights/Wan2.2-T2V-A14B/low_noise_model",
                neg_embed_path="",
                teacher_guidance=5.0,
                precision="bfloat16",
                net=dict(sac_config=dict(mode="block_wise")),
            )
        ),
        checkpoint=dict(
            save_iter=500,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=10_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(every_n=1000, num_samples=1, run_at_start=True),
                every_n_sample_ema=dict(every_n=1000, num_samples=1, run_at_start=True),
            ),
            grad_accum_iter=2,
        ),
        model_parallel=dict(context_parallel_size=1),
        dataloader_train=dict(
            tar_path_pattern="datasets/distill/Wan2.2_A14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard*.tar",
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)

WAN2PT2_A14B_RES720P_T2V_ROLA_DISTILL_JOINT: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{WAN2PT2_A14B_RES480P_T2V_ROLA_DISTILL_JOINT['job']['name']}",
            "_self_",
        ],
        job=dict(
            group="Wan2pt2_RoLa",
            name="wan2pt2_A14B_res720p_t2v_rola_distill_joint",
        ),
        model=dict(config=dict(resolution="720p")),
        trainer=dict(grad_accum_iter=1),
    ),
    flags={"allow_objects": True},
)

wan2pt2_rola_distill_job_list = [
    WAN2PT2_A14B_RES480P_T2V_ROLA_DISTILL_JOINT,
    WAN2PT2_A14B_RES720P_T2V_ROLA_DISTILL_JOINT,
]
for job in wan2pt2_rola_distill_job_list:
    cs.store(group="experiment", package="_global_", name=job["job"]["name"], node=job)
    cs.store(group="experiment", package="_global_", name=job["job"]["name"] + "_debug", node=build_debug_run(job))
