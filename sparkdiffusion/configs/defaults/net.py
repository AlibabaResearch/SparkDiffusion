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

from sparkdiffusion.networks.wan2pt1 import WanModel
from sparkdiffusion.networks.wan2pt2 import WanModel as WanModel2pt2

wan2pt1_1pt3B_net_args = dict(
    dim=1536,
    eps=1e-06,
    ffn_dim=8960,
    freq_dim=256,
    in_dim=16,
    num_heads=12,
    num_layers=30,
    out_dim=16,
    text_len=512,
)

wan2pt1_14B_net_args = dict(
    dim=5120,
    eps=1e-06,
    ffn_dim=13824,
    freq_dim=256,
    in_dim=16,
    num_heads=40,
    num_layers=40,
    out_dim=16,
    text_len=512,
)

WAN2PT1_1PT3B_T2V: LazyDict = L(WanModel)(**wan2pt1_1pt3B_net_args, model_type="t2v")

WAN2PT1_14B_T2V: LazyDict = L(WanModel)(**wan2pt1_14B_net_args, model_type="t2v")

WAN2PT1_1PT3B_T2V_ROLA: LazyDict = L(WanModel)(
    **wan2pt1_1pt3B_net_args, model_type="t2v",
    use_rola_attn=True, rola_topk_ratio=0.1, rola_rank=64, rola_blkq=64, rola_blkk=64,
)

WAN2PT1_14B_T2V_ROLA: LazyDict = L(WanModel)(
    **wan2pt1_14B_net_args, model_type="t2v",
    use_rola_attn=True, rola_topk_ratio=0.1, rola_rank=64, rola_blkq=64, rola_blkk=64,
)

# ── Wan 2.1 14B I2V (image-to-video, with CLIP): in_dim=36 (16 noise + 4 mask + 16 first-frame latent) ──
# When model_type="i2v", WanModel automatically builds img_emb (MLPProj 1280→dim) and requires CLIP features
# frame_cond_crossattn_emb_B_L_D [B,257,1280] + first-frame condition y_B_C_T_H_W.
# Official Wan2.1-I2V-14B-480P-Diffusers: in_channels=36, image_dim=1280.
wan2pt1_14B_i2v_net_args = {**wan2pt1_14B_net_args, "in_dim": 36}

WAN2PT1_14B_I2V: LazyDict = L(WanModel)(**wan2pt1_14B_i2v_net_args, model_type="i2v")

WAN2PT1_14B_I2V_ROLA: LazyDict = L(WanModel)(
    **wan2pt1_14B_i2v_net_args, model_type="i2v",
    use_rola_attn=True, rola_topk_ratio=0.1, rola_rank=64, rola_blkq=64, rola_blkk=64,
)

# Pure-SLA: original SLA library (SparseLinearAttention), no low-rank/gate branch
WAN2PT1_1PT3B_T2V_PURE_SLA: LazyDict = L(WanModel)(
    **wan2pt1_1pt3B_net_args, model_type="t2v",
    use_pure_sla_attn=True, pure_sla_topk_ratio=0.1,
    pure_sla_blkq=64, pure_sla_blkk=64, pure_sla_feature_map="softmax",
)

WAN2PT1_14B_T2V_PURE_SLA: LazyDict = L(WanModel)(
    **wan2pt1_14B_net_args, model_type="t2v",
    use_pure_sla_attn=True, pure_sla_topk_ratio=0.1,
    pure_sla_blkq=64, pure_sla_blkk=64, pure_sla_feature_map="softmax",
)

# Placeholder. model_type only selects cross_attn_type/img_emb while the attention
# implementation only replaces self-attention, so i2v + pure-SLA is a valid combo.
# Two things are missing before it can run: the external sparse_linear_attention
# package (imported by WanSelfAttentionPureSLA), and a pure-SLA-distilled i2v
# checkpoint — the RoLa students carry proj_q/proj_k/gate_proj/gate_bias, which
# PureSLA has no slot for, so loading one here silently drops that branch.
WAN2PT1_14B_I2V_PURE_SLA: LazyDict = L(WanModel)(
    **wan2pt1_14B_i2v_net_args, model_type="i2v",
    use_pure_sla_attn=True, pure_sla_topk_ratio=0.1,
    pure_sla_blkq=64, pure_sla_blkk=64, pure_sla_feature_map="softmax",
)

# ══════════════════════════════════════════════════════════════════════════════
# Wan 2.2 network configs
# ══════════════════════════════════════════════════════════════════════════════
wan2pt2_A14B_net_args = dict(
    dim=5120,
    eps=1e-06,
    ffn_dim=13824,
    freq_dim=256,
    in_dim=16,
    num_heads=40,
    num_layers=40,
    out_dim=16,
    text_len=512,
)

WAN2PT2_A14B_T2V: LazyDict = L(WanModel2pt2)(**wan2pt2_A14B_net_args, model_type="t2v")

WAN2PT2_A14B_T2V_ROLA: LazyDict = L(WanModel2pt2)(
    **wan2pt2_A14B_net_args, model_type="t2v",
    use_rola_attn=True, rola_topk_ratio=0.1, rola_rank=64, rola_blkq=64, rola_blkk=64,
)

# ── I2V (image-to-video): pure channel concatenation, no CLIP (image_dim=null) ──
# First-frame latent(16) + mask(4) concatenated with noise(16) along the channel dim → in_dim=36.
# Official Wan2.2-I2V-A14B-Diffusers: in_channels=36, image_dim=null, out_channels=16.
wan2pt2_A14B_i2v_net_args = {**wan2pt2_A14B_net_args, "in_dim": 36}

WAN2PT2_A14B_I2V: LazyDict = L(WanModel2pt2)(**wan2pt2_A14B_i2v_net_args, model_type="i2v")

WAN2PT2_A14B_I2V_ROLA: LazyDict = L(WanModel2pt2)(
    **wan2pt2_A14B_i2v_net_args, model_type="i2v",
    use_rola_attn=True, rola_topk_ratio=0.1, rola_rank=64, rola_blkq=64, rola_blkk=64,
)


def register_net():
    cs = ConfigStore.instance()
    cs.store(group="net", package="model.config.net", name="wan2pt1_1pt3B_t2v", node=WAN2PT1_1PT3B_T2V)
    cs.store(group="net", package="model.config.net", name="wan2pt1_14B_t2v", node=WAN2PT1_14B_T2V)
    cs.store(group="net", package="model.config.net", name="wan2pt1_1pt3B_t2v_rola", node=WAN2PT1_1PT3B_T2V_ROLA)
    cs.store(group="net", package="model.config.net", name="wan2pt1_14B_t2v_rola", node=WAN2PT1_14B_T2V_ROLA)
    cs.store(group="net", package="model.config.net", name="wan2pt1_14B_i2v", node=WAN2PT1_14B_I2V)
    cs.store(group="net", package="model.config.net", name="wan2pt1_14B_i2v_rola", node=WAN2PT1_14B_I2V_ROLA)
    cs.store(group="net", package="model.config.net", name="wan2pt1_1pt3B_t2v_pure_sla", node=WAN2PT1_1PT3B_T2V_PURE_SLA)
    cs.store(group="net", package="model.config.net", name="wan2pt1_14B_t2v_pure_sla", node=WAN2PT1_14B_T2V_PURE_SLA)
    cs.store(group="net", package="model.config.net", name="wan2pt1_14B_i2v_pure_sla", node=WAN2PT1_14B_I2V_PURE_SLA)
    # RoLa variants.
    # Wan 2.2
    cs.store(group="net", package="model.config.net", name="wan2pt2_A14B_t2v", node=WAN2PT2_A14B_T2V)
    cs.store(group="net", package="model.config.net", name="wan2pt2_A14B_t2v_rola", node=WAN2PT2_A14B_T2V_ROLA)
    cs.store(group="net", package="model.config.net", name="wan2pt2_A14B_i2v", node=WAN2PT2_A14B_I2V)
    cs.store(group="net", package="model.config.net", name="wan2pt2_A14B_i2v_rola", node=WAN2PT2_A14B_I2V_ROLA)


def register_net_fake_score():
    cs = ConfigStore.instance()
    cs.store(group="net_fake_score", package="model.config.net_fake_score", name="wan2pt1_1pt3B_t2v", node=WAN2PT1_1PT3B_T2V)
    cs.store(group="net_fake_score", package="model.config.net_fake_score", name="wan2pt1_14B_t2v", node=WAN2PT1_14B_T2V)
    cs.store(group="net_fake_score", package="model.config.net_fake_score", name="wan2pt1_1pt3B_t2v_rola", node=WAN2PT1_1PT3B_T2V_ROLA)
    cs.store(group="net_fake_score", package="model.config.net_fake_score", name="wan2pt1_14B_t2v_rola", node=WAN2PT1_14B_T2V_ROLA)
    cs.store(group="net_fake_score", package="model.config.net_fake_score", name="wan2pt1_14B_i2v", node=WAN2PT1_14B_I2V)
    cs.store(group="net_fake_score", package="model.config.net_fake_score", name="wan2pt1_14B_i2v_rola", node=WAN2PT1_14B_I2V_ROLA)
    cs.store(group="net_fake_score", package="model.config.net_fake_score", name="wan2pt1_1pt3B_t2v_pure_sla", node=WAN2PT1_1PT3B_T2V_PURE_SLA)
    cs.store(group="net_fake_score", package="model.config.net_fake_score", name="wan2pt1_14B_t2v_pure_sla", node=WAN2PT1_14B_T2V_PURE_SLA)
    # RoLa variants.
    # Wan 2.2
    cs.store(group="net_fake_score", package="model.config.net_fake_score", name="wan2pt2_A14B_t2v", node=WAN2PT2_A14B_T2V)
    cs.store(group="net_fake_score", package="model.config.net_fake_score", name="wan2pt2_A14B_t2v_rola", node=WAN2PT2_A14B_T2V_ROLA)
    cs.store(group="net_fake_score", package="model.config.net_fake_score", name="wan2pt2_A14B_i2v", node=WAN2PT2_A14B_I2V)
    cs.store(group="net_fake_score", package="model.config.net_fake_score", name="wan2pt2_A14B_i2v_rola", node=WAN2PT2_A14B_I2V_ROLA)


def register_net_teacher():
    cs = ConfigStore.instance()
    cs.store(group="net_teacher", package="model.config.net_teacher", name="wan2pt1_1pt3B_t2v", node=WAN2PT1_1PT3B_T2V)
    cs.store(group="net_teacher", package="model.config.net_teacher", name="wan2pt1_14B_t2v", node=WAN2PT1_14B_T2V)
    cs.store(group="net_teacher", package="model.config.net_teacher", name="wan2pt1_1pt3B_t2v_rola", node=WAN2PT1_1PT3B_T2V_ROLA)
    cs.store(group="net_teacher", package="model.config.net_teacher", name="wan2pt1_14B_t2v_rola", node=WAN2PT1_14B_T2V_ROLA)
    cs.store(group="net_teacher", package="model.config.net_teacher", name="wan2pt1_14B_i2v", node=WAN2PT1_14B_I2V)
    cs.store(group="net_teacher", package="model.config.net_teacher", name="wan2pt1_14B_i2v_rola", node=WAN2PT1_14B_I2V_ROLA)
    cs.store(group="net_teacher", package="model.config.net_teacher", name="wan2pt1_1pt3B_t2v_pure_sla", node=WAN2PT1_1PT3B_T2V_PURE_SLA)
    cs.store(group="net_teacher", package="model.config.net_teacher", name="wan2pt1_14B_t2v_pure_sla", node=WAN2PT1_14B_T2V_PURE_SLA)
    # RoLa variants.
    # Wan 2.2
    cs.store(group="net_teacher", package="model.config.net_teacher", name="wan2pt2_A14B_t2v", node=WAN2PT2_A14B_T2V)
    cs.store(group="net_teacher", package="model.config.net_teacher", name="wan2pt2_A14B_t2v_rola", node=WAN2PT2_A14B_T2V_ROLA)
    cs.store(group="net_teacher", package="model.config.net_teacher", name="wan2pt2_A14B_i2v", node=WAN2PT2_A14B_I2V)
    cs.store(group="net_teacher", package="model.config.net_teacher", name="wan2pt2_A14B_i2v_rola", node=WAN2PT2_A14B_I2V_ROLA)
