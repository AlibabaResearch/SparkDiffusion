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

from __future__ import annotations

import collections
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Literal

import attrs
import numpy as np
import torch
import torch._dynamo
import torch.distributed.checkpoint as dcp
from einops import rearrange, repeat
try:
    from megatron.core import parallel_state
except (ImportError, TypeError, AttributeError):
    parallel_state = None
from torch import Tensor
from torch.distributed._composable.fsdp import FSDPModule, fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor.api import DTensor
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.nn.modules.module import _IncompatibleKeys
from imaginaire.config import ObjectStoreConfig
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict
from imaginaire.lazy_config import instantiate as lazy_instantiate
from imaginaire.model import ImaginaireModel
from imaginaire.utils import log, misc
from imaginaire.utils.easy_io import easy_io
from imaginaire.utils.ema import FastEmaModelUpdater
from sparkdiffusion.conditioner import DataType, TextCondition
from sparkdiffusion.utils.optim_instantiate_dtensor import get_base_scheduler
from sparkdiffusion.utils.checkpointer import non_strict_load_model
from sparkdiffusion.utils.model_utils import _load_safetensors_dir, _detect_ckpt_format
from sparkdiffusion.utils.context_parallel import broadcast
from sparkdiffusion.utils.dtensor_helper import DTensorFastEmaModelUpdater, broadcast_dtensor_model_states
from sparkdiffusion.utils.fsdp_helper import hsdp_device_mesh
from sparkdiffusion.utils.misc import count_params
from sparkdiffusion.utils.torch_future import clip_grad_norm_
from sparkdiffusion.utils.timestep_utils import LogNormal, UniformShift, shift_rf_time
from sparkdiffusion.configs.defaults.ema import EMAConfig
from sparkdiffusion.samplers.euler import FlowEulerSampler
from sparkdiffusion.samplers.unipc import FlowUniPCMultistepSampler

IS_PREPROCESSED_KEY = "is_preprocessed"
IS_PROCESSED_KEY = "is_processed"


@dataclass
class DenoisePrediction:
    x0: torch.Tensor  # clean data prediction
    F: torch.Tensor = None  # rf velocity v = eps - x0


@attrs.define(slots=False)
class T2VDistillConfig:

    tokenizer: LazyDict = None
    conditioner: LazyDict = None
    net: LazyDict = None
    net_teacher: LazyDict = None
    net_fake_score: LazyDict = None
    net_low: LazyDict = None
    net_teacher_low: LazyDict = None
    net_fake_score_low: LazyDict = None
    optimizer_fake_score: LazyDict = None
    teacher_ckpt: str = ""  # default source used to initialize teacher/student/fake_score
    student_ckpt: str = ""  # when set, overrides the load source of the student net (otherwise uses teacher_ckpt)
    fake_score_ckpt: str = ""  # when set, overrides the load source of the fake_score (critic) net (otherwise uses teacher_ckpt)
    teacher_ckpt_low: str = ""
    student_ckpt_low: str = ""
    fake_score_ckpt_low: str = ""
    joint_wan22_t2v: bool = False
    joint_boundary_rf: float = 0.875
    joint_boundary_step_index: int = 2
    joint_backward_timesteps: list = [0.933781, 0.875, 0.608979]
    joint_backward_start_t: float = -1.0
    joint_backward_end_t: float = 0.0
    joint_low_loss_scale_div: float = 0.0
    joint_step_cycle: list = []
    joint_low_d_time_full_range: bool = False
    critic_warmup: int = 0
    teacher_guidance: float = 5.0
    grad_clip: bool = False
    sigma_max: float = 1600

    ema: EMAConfig = EMAConfig()
    checkpoint: ObjectStoreConfig = ObjectStoreConfig()
    p_G: LazyDict = L(LogNormal)(p_mean=-0.8, p_std=1.6)
    p_D: LazyDict = L(LogNormal)(p_mean=0.0, p_std=1.6)
    student_update_freq: int = 5
    fsdp_shard_size: int = 1
    sigma_data: float = 1.0
    precision: str = "bfloat16"
    input_data_key: str = "videos"
    input_latent_key: str = "latents"
    input_caption_key: str = "prompts"
    loss_scale: float = 100.0
    loss_scale_dmd: float = 1.0
    # TDM (Trajectory Distribution Matching) loss weight on the student.
    # TDM differs from DMD in three ways (mutually exclusive with loss_scale_dmd):
    #   1) backward_simulation last step does NOT reach 0; it stops at t_{i-1},
    #      which is one extra timestep sampled from the SAME schedule as the
    #      intermediate timesteps (i.e. "the next timestep in the list").
    #   2) D_time is clamped to be > t_{i-1}.
    #   3) fake/teacher denoise outputs are mapped to t_{i-1} via the rf
    #      PF-ODE analytical step (NOT all the way to x0).
    # Set 0 to disable. Cannot coexist with loss_scale_dmd > 0.
    loss_scale_tdm: float = 0.0
    # Forces student's one-step RF velocity at noise (rf s = 1) to match the
    # average RF velocity of a teacher K-step rollout from s=1 to t_tilde:
    #   eps   = x at rf s = 1                              (pure noise, identical to backward_simulation's start)
    #   z_tilde = teacher_rollout(eps, K)                  (K rf-Euler steps to t_tilde, no_grad)
    #   v_tilde = (x_rf(T) - x_rf(t_tilde)) / (u(T) - u(t_tilde))   (RF avg velocity, ode_match target)
    #   v_stu   = student_net(eps * c_in, c_noise)         (RF velocity at noise, with grad)
    #   loss_div = mean((v_stu - v_tilde)^2)
    loss_scale_div: float = 0.0
    div_K: int = 8            # teacher rollout step count 
    div_sigma_max: float = 5000.0
    # Diversity Supervision algorithm selection. All three variants share the
    # same loss_scale_div weight and are mutually exclusive.
    #   "ode_match": RF avg-velocity matching at rf s = 1 with a
    #                 teacher K-step rollout to t_tilde (uses div_K, div_sigma_max).
    #   "pcm_fixed":  PCM-style boundary alignment on phase 0 [1.0, bwd[0]] (rf),
    #                 t_start fixed at rf s=1 (pure-noise input).
    #   "pcm_random": PCM-style boundary alignment on phase 0, t_start sampled
    #                 uniformly within phase 0's valid u-range (same strategy
    #                 as _student_pcm_step). Uses dcm_total_steps,
    #                 dcm_skipping_interval_steps, teacher_timestep_shift.
    div_mode: Literal["ode_match", "pcm_fixed", "pcm_random"] = "ode_match"
    max_simulation_steps_fake: int = 4
    neg_embed_path: str = ""

    state_ch: int = 16
    state_t: int = 21  # Number of latent frames
    resolution: str = "480p"
    rectified_flow_t_scaling_factor: float = 1000.0

    # I2V (image-to-video): when True, build the condition y_B_C_T_H_W from the first frame (first-frame latent + mask)
    # and attach it to the condition (shared by teacher/student/fake_score). Requires net=wan2pt2_A14B_i2v_rola.
    is_i2v: bool = False
    # Wan2.1 I2V only: directory path of the CLIP image_encoder. When non-empty, extract first-frame CLIP features online
    # and attach them to the condition as frame_cond_crossattn_emb_B_L_D (2.2 I2V has no CLIP, leave empty).
    i2v_clip_encoder_path: str = ""

    text_encoder_class: str = "umT5"
    text_encoder_path: str = ""
    tokenizer_path: str = ""  # local directory of the umt5 tokenizer (used by the sampling-visualization callback, avoids downloading from HF)

    backward_timesteps: list = [0.933781, 0.852895, 0.608979]  # rf time (legacy trig 1.5/1.4/1.0)
    dmd_fix_timesteps: bool = True

    # Timestep range restriction (for dual-model distillation)
    rf_t_min: float = 0.0
    rf_t_max: float = 1.0
    backward_simulation_start_t: float = -1.0  # rf time; -1 = auto (rf 1.0 = pure noise)
    backward_simulation_end_t: float = 0.0     # rf time; 0 = x0 (clean data)

    # ── Wan2.2 low-noise expert: online rollout of the high segment ──────────
    # The low expert starts at rf = backward_simulation_start_t (the MoE boundary),
    # where a plain randn is NOT the right distribution: at inference the low expert
    # receives the HIGH expert's output. Set init_rollout_ckpt to the already-trained
    # high-noise student to generate that starting point online (frozen, no_grad),
    # so training input matches inference and the low expert learns to correct the
    # high expert's residual error.
    #   init_rollout_ckpt:      path to the trained high-noise student ("" = disabled -> randn)
    #   init_rollout_timesteps: rf mid knots of the HIGH segment, e.g. [0.933781]
    #                           rollout path = [1.0, *init_rollout_timesteps, start_t]
    init_rollout_ckpt: str = ""
    init_rollout_timesteps: list = []
    # Full-trajectory rf mid knots for training-time visualization sampling.
    # For Wan2.2 experts set the WHOLE 4-step trajectory's mid knots (e.g. t2v
    # [0.933781, 0.875, 0.608979]); the viz sampler runs the high half with
    # net_init_rollout (LOW) or self.net (HIGH) and the low half with self.net.
    # Empty -> fall back to the per-expert config schedule (Wan2.1 single model).
    viz_full_timesteps: list = []

    # Discrete-time CM grid used by the pcm div modes.
    dcm_total_steps: int = 48
    dcm_skipping_interval_steps: int = 1
    # Official teacher sampling shift (scheduler sample_shift). MUST match the
    # teacher being distilled, since it defines the teacher's timestep density:
    #   Wan2.2 t2v A14B = 12.0, Wan2.2 i2v A14B = 5.0
    #   Wan2.1 t2v      =  5.0, Wan2.1 i2v 480p = 3.0 (720p 5.0)
    # Single source for all shift consumers: pcm-div phase schedule,
    # ode_match teacher rollout grid, and the teacher visualization sampler.
    teacher_timestep_shift: float = 5.0


class T2VDistillModel(ImaginaireModel):

    def __init__(self, config: T2VDistillConfig):
        super().__init__()

        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        log.warning(f"DiffusionModel: precision {self.precision}")

        # 1. setup up diffusion processing and scaling~(pre-condition), sampler
        self.p_G = lazy_instantiate(config.p_G)
        self.p_D = lazy_instantiate(config.p_D)
        if config.neg_embed_path:
            self.neg_embed = easy_io.load(config.neg_embed_path)
        else:
            self.neg_embed = None

        # 2. tokenizer
        with misc.timer("DiffusionModel: set_up_tokenizer"):
            self.tokenizer = lazy_instantiate(config.tokenizer)
            assert self.tokenizer.latent_ch == config.state_ch, f"latent_ch {self.tokenizer.latent_ch} != state_shape {config.state_ch}"

        # 3. create fsdp mesh if needed
        if config.fsdp_shard_size > 1:
            log.info(f"FSDP size: {config.fsdp_shard_size}")
            self.fsdp_device_mesh = hsdp_device_mesh(sharding_group_size=config.fsdp_shard_size)
        else:
            self.fsdp_device_mesh = None

        # 4. diffusion neural networks part
        self.set_up_model()

        # 5. training states
        if parallel_state is not None and parallel_state.is_initialized():
            self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_size = 1

        if self.config.loss_scale == 0:
            assert self.config.critic_warmup == 0

        # Wan2.1 I2V: load the CLIP image encoder on demand (extracts first-frame features online)
        self._clip_encoder = None
        if self.config.is_i2v and self.config.i2v_clip_encoder_path:
            from sparkdiffusion.utils.clip_image_encoder import WanCLIPImageEncoder
            self._clip_encoder = WanCLIPImageEncoder(
                self.config.i2v_clip_encoder_path, dtype=torch.float16, device="cuda"
            )

        torch._dynamo.config.suppress_errors = True

    def build_net(self, net_dict: LazyDict):
        init_device = "meta"
        with misc.timer("Creating PyTorch model"):
            with torch.device(init_device):
                net = lazy_instantiate(net_dict)

            if self.fsdp_device_mesh:
                net.fully_shard(mesh=self.fsdp_device_mesh, mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.float32))
                net = fully_shard(
                    net, mesh=self.fsdp_device_mesh, mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.float32), reshard_after_forward=True
                )

            with misc.timer("meta to cuda and broadcast model states"):
                net.to_empty(device="cuda")
                net.init_weights()

            if self.fsdp_device_mesh:
                broadcast_dtensor_model_states(net, self.fsdp_device_mesh)
                for name, param in net.named_parameters():
                    assert isinstance(param, DTensor), f"param should be DTensor, {name} got {type(param)}"
        return net

    def _load_safetensors_to_net(self, net, path: str):
        """Load a native Wan safetensors checkpoint into WanModel."""
        raw_sd = _load_safetensors_dir(path)

        converted: dict = {}
        for k, v in raw_sd.items():
            if k == "patch_embedding.weight" and v.ndim == 5:
                v = v.reshape(v.shape[0], -1)
            converted[k] = v

        if self.fsdp_device_mesh is not None:
            result = set_model_state_dict(
                net, converted,
                options=StateDictOptions(strict=False, full_state_dict=True),
            )
            log.info(f"Loaded safetensors (FSDP) from {path} → {len(converted)} tensors")
        else:
            clean_to_actual: dict = {}
            for actual in net.state_dict():
                clean = actual.replace("._checkpoint_wrapped_module.", ".")
                clean_to_actual[clean] = actual
            final_sd: dict = {clean_to_actual.get(ck, ck): v for ck, v in converted.items()}
            result = net.load_state_dict(final_sd, strict=False)
            new_params = [k for k in result.missing_keys
                          if any(p in k for p in ("proj_q", "proj_k", "gate_proj", "gate_bias"))]
            if new_params:
                log.info(f"RoLa new params kept at random init ({len(new_params)})")
            log.info(f"Loaded safetensors from {path} → {len(final_sd) - len(result.unexpected_keys)} tensors matched")

    def load_ckpt_to_net(self, net, ckpt_path):
        storage_reader = FileSystemReader(ckpt_path)
        _state_dict = get_model_state_dict(net)

        metadata = storage_reader.read_metadata()
        checkpoint_keys = metadata.state_dict_metadata.keys()

        model_keys = set(_state_dict.keys())

        prefix = "net_ema" if any(k.startswith("net_ema.") for k in checkpoint_keys) else "net"

        # Add the prefix to the model keys for comparison
        prefixed_model_keys = {f"{prefix}.{k}" for k in model_keys}

        missing_keys = prefixed_model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - prefixed_model_keys

        if missing_keys:
            log.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            log.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        if not missing_keys and not unexpected_keys:
            log.info("All keys matched successfully.")

        _new_state_dict = collections.OrderedDict()
        for k in _state_dict.keys():
            if "_extra_state" in k:
                log.warning(k)
            _new_state_dict[f"{prefix}.{k}"] = _state_dict[k]
        dcp.load(_new_state_dict, storage_reader=storage_reader, planner=DefaultLoadPlanner(allow_partial_load=True))
        for k in _state_dict.keys():
            _state_dict[k] = _new_state_dict[f"{prefix}.{k}"]

        log.info(set_model_state_dict(net, _state_dict, options=StateDictOptions(strict=False)))
        del _state_dict, _new_state_dict

    @misc.timer("DiffusionModel: set_up_model")
    def set_up_model(self):
        config = self.config
        with misc.timer("Creating PyTorch model and ema if enabled"):
            self.conditioner = lazy_instantiate(config.conditioner)
            assert sum(p.numel() for p in self.conditioner.parameters() if p.requires_grad) == 0, "conditioner should not have learnable parameters"
            if config.joint_wan22_t2v:
                assert not config.ema.enabled, "joint_wan22_t2v does not support EMA yet"
            self.net, self.net_teacher = self.build_net(config.net), self.build_net(config.net_teacher)
            self.net_fake_score = self.build_net(config.net_fake_score) if config.net_fake_score else None
            self.net_low = self.build_net(config.net_low or config.net) if config.joint_wan22_t2v else None
            self.net_teacher_low = self.build_net(config.net_teacher_low or config.net_teacher) if config.joint_wan22_t2v else None
            self.net_fake_score_low = (
                self.build_net(config.net_fake_score_low or config.net_fake_score)
                if config.joint_wan22_t2v and config.net_fake_score
                else None
            )
            self.net_init_rollout = self.build_net(config.net) if (config.init_rollout_ckpt and not config.joint_wan22_t2v) else None
            if config.net_fake_score:
                assert config.loss_scale_dmd > 0 or config.loss_scale_tdm > 0, "loss_scale_dmd or loss_scale_tdm must be greater than 0"

            # ── Helper: resolve a checkpoint path and detect its format ──
            import os as _os

            def _load_ckpt_to_nets(ckpt_path, nets):
                """Load a single checkpoint into one or more nets."""
                _resolved = ckpt_path
                fmt = _detect_ckpt_format(_resolved)
                if fmt == "safetensors":
                    for n in nets:
                        self._load_safetensors_to_net(n, _resolved)
                elif fmt == "pth":
                    from sparkdiffusion.utils.model_utils import load_state_dict
                    _sd = load_state_dict(_resolved)
                    _sd_clean = {(k[4:] if k.startswith("net.") else k): v for k, v in _sd.items()}
                    if self.fsdp_device_mesh is not None:
                        from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
                        _opts = StateDictOptions(strict=False, full_state_dict=True)
                        for n in nets:
                            set_model_state_dict(n, _sd_clean, options=_opts)
                    else:
                        for n in nets:
                            n.load_state_dict(_sd_clean, strict=False)
                    log.info(f"Loaded .pth checkpoint from {_resolved}")
                else:  # dcp
                    # DCP only supports loading into one net at a time
                    self.load_ckpt_to_net(nets[0], ckpt_path)
                    for n in nets[1:]:
                        n.load_state_dict(nets[0].state_dict(), strict=False)

            # ── Load checkpoints: each net defaults to teacher_ckpt, and can be overridden per role ──
            # student  ← student_ckpt or teacher_ckpt
            # fake_score(critic) ← fake_score_ckpt or teacher_ckpt (defaults to initializing from the teacher
            #   rather than following the student —— using the teacher's true score estimate as the critic's starting point is more reasonable, and fake_score
            #   is a full-attn dense structure that cannot load the student's SLA weights)
            net_ckpts = [
                ("teacher", self.net_teacher, config.teacher_ckpt),
                ("student", self.net, config.student_ckpt or config.teacher_ckpt),
            ]
            if self.net_fake_score is not None:
                net_ckpts.append(("fake_score", self.net_fake_score, config.fake_score_ckpt or config.teacher_ckpt))
            if config.joint_wan22_t2v:
                low_teacher_ckpt = config.teacher_ckpt_low or config.teacher_ckpt
                net_ckpts.extend([
                    ("teacher_low", self.net_teacher_low, low_teacher_ckpt),
                    ("student_low", self.net_low, config.student_ckpt_low or low_teacher_ckpt),
                ])
                if self.net_fake_score_low is not None:
                    net_ckpts.append(("fake_score_low", self.net_fake_score_low, config.fake_score_ckpt_low or low_teacher_ckpt))
            if self.net_init_rollout is not None:
                net_ckpts.append(("init_rollout", self.net_init_rollout, config.init_rollout_ckpt))

            # Group by path: nets sharing the same ckpt are read from disk only once
            ckpt_groups = defaultdict(list)
            for role, net, ckpt_path in net_ckpts:
                if ckpt_path:
                    ckpt_groups[ckpt_path].append((role, net))
            for ckpt_path, group in ckpt_groups.items():
                _load_ckpt_to_nets(ckpt_path, [net for _, net in group])
                log.info(f"Loaded checkpoint from {ckpt_path} into {', '.join(role for role, _ in group)}")

            self.net_teacher.requires_grad_(False)
            if self.net_teacher_low is not None:
                self.net_teacher_low.requires_grad_(False)
            if self.net_init_rollout is not None:
                self.net_init_rollout.requires_grad_(False)
            self._param_count = count_params(self.net, verbose=False)
            if self.net_low is not None:
                self._param_count += count_params(self.net_low, verbose=False)

            # Enable/disable CP once; all CP comm/split/gather happens inside net.forward now.
            cp_group = self.get_context_parallel_group()
            cp_nets = [self.net, self.net_teacher, self.net_fake_score, self.net_low, self.net_teacher_low, self.net_fake_score_low, self.net_init_rollout]
            cp_nets = [net for net in cp_nets if net is not None]
            if cp_group is not None and cp_group.size() > 1:
                for net in cp_nets:
                    net.enable_context_parallel(cp_group)
            else:
                for net in cp_nets:
                    net.disable_context_parallel()

            if config.ema.enabled:
                self.net_ema = self.build_net(config.net)
                self.net_ema.requires_grad_(False)

                if self.fsdp_device_mesh:
                    self.net_ema_worker = DTensorFastEmaModelUpdater()
                else:
                    self.net_ema_worker = FastEmaModelUpdater()

                s = config.ema.rate
                self.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()

                self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)
        torch.cuda.empty_cache()

    def init_optimizer_scheduler(self, optimizer_config: LazyDict, scheduler_config: LazyDict):
        """Creates the optimizer and scheduler for the model."""
        net_optimizer = lazy_instantiate(optimizer_config, model=self.net)
        self.optimizer_dict = {"net": net_optimizer}
        net_scheduler = get_base_scheduler(net_optimizer, self, scheduler_config)
        self.scheduler_dict = {"net": net_scheduler}

        if self.net_low is not None:
            net_low_optimizer = lazy_instantiate(optimizer_config, model=self.net_low)
            net_low_scheduler = get_base_scheduler(net_low_optimizer, self, scheduler_config)
            self.optimizer_dict["net_low"] = net_low_optimizer
            self.scheduler_dict["net_low"] = net_low_scheduler

        if self.net_fake_score:
            fake_score_optimizer = lazy_instantiate(self.config.optimizer_fake_score, model=self.net_fake_score)
            fake_score_scheduler = get_base_scheduler(fake_score_optimizer, self, scheduler_config)
            self.optimizer_dict["fake_score"] = fake_score_optimizer
            self.scheduler_dict["fake_score"] = fake_score_scheduler

        if self.net_fake_score_low:
            fake_score_low_optimizer = lazy_instantiate(self.config.optimizer_fake_score, model=self.net_fake_score_low)
            fake_score_low_scheduler = get_base_scheduler(fake_score_low_optimizer, self, scheduler_config)
            self.optimizer_dict["fake_score_low"] = fake_score_low_optimizer
            self.scheduler_dict["fake_score_low"] = fake_score_low_scheduler

    def is_student_phase(self, iteration: int):
        # Critic (fake_score net) only needs to be trained when at least one of the
        # distribution-matching losses (DMD or TDM) is active. When both are 0 the
        # fake_score net is unused, so we stay in student phase forever.
        no_distrib_loss = self.config.loss_scale_dmd == 0 and self.config.loss_scale_tdm == 0
        return (
            (self.net_fake_score is None or no_distrib_loss)
            or iteration < self.config.critic_warmup
            or (iteration - self.config.critic_warmup) % self.config.student_update_freq == 0
        )

    def get_effective_iteration(self, iteration: int):
        return (
            iteration
            if self.net_fake_score is None or iteration < self.config.critic_warmup
            else self.config.critic_warmup + (iteration - self.config.critic_warmup) // self.config.student_update_freq
        )

    def get_effective_iteration_fake(self, iteration: int):
        return iteration - self.get_effective_iteration(iteration) - 1

    def _joint_num_simulation_steps(self, iteration: int, phase: Literal["student", "critic"]) -> int:
        if phase == "student":
            effective_iteration = self.get_effective_iteration(iteration)
        elif phase == "critic":
            effective_iteration = self.get_effective_iteration_fake(iteration)
        else:
            raise ValueError(f"Unknown training phase: {phase}")
        if self.config.joint_step_cycle:
            cycle = [int(x) for x in self.config.joint_step_cycle]
            return cycle[effective_iteration % len(cycle)]
        return effective_iteration % self.config.max_simulation_steps_fake + 1

    def _joint_region_for_num_steps(self, num_steps: int) -> Literal["high", "low"]:
        return "high" if num_steps <= self.config.joint_boundary_step_index else "low"

    def _joint_region_for_iteration(self, iteration: int, phase: Literal["student", "critic"]) -> Literal["high", "low"]:
        return self._joint_region_for_num_steps(self._joint_num_simulation_steps(iteration, phase))

    def _joint_optimizer_key(self, iteration: int) -> str:
        phase = "student" if self.is_student_phase(iteration) else "critic"
        region = self._joint_region_for_iteration(iteration, phase)
        if phase == "student":
            return "net" if region == "high" else "net_low"
        return "fake_score" if region == "high" else "fake_score_low"

    def _joint_step_schedule(self) -> list[float]:
        start_t = self.config.joint_backward_start_t
        if start_t < 0:
            start_t = 1.0
        return [start_t, *list(self.config.joint_backward_timesteps), self.config.joint_backward_end_t]

    def _student_net(self, region: Literal["high", "low"]):
        return self.net if region == "high" or not self.config.joint_wan22_t2v else self.net_low

    def _teacher_net(self, region: Literal["high", "low"]):
        return self.net_teacher if region == "high" or not self.config.joint_wan22_t2v else self.net_teacher_low

    def _fake_score_net(self, region: Literal["high", "low"]):
        return self.net_fake_score if region == "high" or not self.config.joint_wan22_t2v else self.net_fake_score_low

    def _joint_region_for_step_index(self, step_index: int) -> Literal["high", "low"]:
        return "high" if step_index < self.config.joint_boundary_step_index else "low"

    # ------------------------ training hooks ------------------------
    def on_before_zero_grad(self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int) -> None:
        """
        update the net_ema
        """

        if self.config.ema.enabled and self.is_student_phase(iteration):
            # calculate beta for EMA update
            ema_beta = self.ema_beta(self.get_effective_iteration(iteration))
            self.net_ema_worker.update_average(self.net, self.net_ema, beta=ema_beta)

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        if self.config.ema.enabled:
            self.net_ema.to(dtype=torch.float32)
        if hasattr(self.tokenizer, "reset_dtype"):
            self.tokenizer.reset_dtype()
        self.net = self.net.to(memory_format=memory_format, **self.tensor_kwargs)
        if self.net_teacher:
            self.net_teacher = self.net_teacher.to(memory_format=memory_format, **self.tensor_kwargs)
        if self.net_fake_score:
            self.net_fake_score = self.net_fake_score.to(memory_format=memory_format, **self.tensor_kwargs)
        if self.net_low is not None:
            self.net_low = self.net_low.to(memory_format=memory_format, **self.tensor_kwargs)
        if self.net_teacher_low is not None:
            self.net_teacher_low = self.net_teacher_low.to(memory_format=memory_format, **self.tensor_kwargs)
        if self.net_fake_score_low is not None:
            self.net_fake_score_low = self.net_fake_score_low.to(memory_format=memory_format, **self.tensor_kwargs)
        if self.net_init_rollout is not None:
            self.net_init_rollout = self.net_init_rollout.to(memory_format=memory_format, **self.tensor_kwargs)

    # ------------------------ training ------------------------

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        """
        Get the optimizers for the current iteration
        Args:
            iteration (int): The current training iteration

        """
        if self.config.joint_wan22_t2v:
            return [self.optimizer_dict[self._joint_optimizer_key(iteration)]]
        if self.is_student_phase(iteration):
            return [self.optimizer_dict["net"]]
        else:
            return [self.optimizer_dict["fake_score"]]

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler.LRScheduler]:
        """
        Get the lr schedulers for the current iteration
        Args:
            iteration (int): The current training iteration

        """
        if self.config.joint_wan22_t2v:
            return [self.scheduler_dict[self._joint_optimizer_key(iteration)]]
        if self.is_student_phase(iteration):
            return [self.scheduler_dict["net"]]
        else:
            return [self.scheduler_dict["fake_score"]]

    def _sample_rf_time(self, sampler, time_shape: Any) -> torch.Tensor:
        assert isinstance(time_shape, (int, tuple, list, torch.Size)), f"Unsupported time shape type: {type(time_shape)}"
        rf_min = self.config.rf_t_min
        rf_max = self.config.rf_t_max
        if rf_min == 0.0 and rf_max == 1.0:
            sampled = sampler(shape=time_shape, device="cuda", dtype=torch.float64)
            domain = getattr(sampler, "output_domain", "rf")
            assert domain == "rf", f"Expected RF-domain timestep sampler, got {domain}"
            return sampled.clamp(min=0.0, max=1.0)
        else:
            n = time_shape if isinstance(time_shape, int) else int(torch.Size(time_shape).numel())
            if isinstance(sampler, UniformShift):
                # UniformShift maps u~U[0,1) monotonically via shift_rf_time, so
                # truncating rf to [rf_min, rf_max] is exact: sample u in the
                # inverse-mapped interval, then re-apply shift_rf_time.
                shift = sampler.shift
                def _inv(rf):
                    return rf / (shift - (shift - 1.0) * rf) if shift > 0 else rf
                u_lo, u_hi = _inv(rf_min), _inv(rf_max)
                u = torch.rand(n, device="cuda", dtype=torch.float64) * (u_hi - u_lo) + u_lo
                rf_t = shift_rf_time(u, shift)
            else:
                # LogNormal: truncate via the sigma-space normal CDF.
                import torch.distributions as D
                normal = D.Normal(sampler.p_mean, sampler.p_std)
                eps_val = torch.finfo(torch.float64).eps
                log_min = torch.tensor(rf_min / (1.0 - rf_min + eps_val)).log() if rf_min > 0 else torch.tensor(-40.0)
                log_max = torch.tensor(rf_max / (1.0 - rf_max + eps_val)).log() if rf_max < 1.0 else torch.tensor(40.0)
                cdf_lo = normal.cdf(log_min.double())
                cdf_hi = normal.cdf(log_max.double())
                u = torch.rand(n, device="cuda", dtype=torch.float64)
                u = cdf_lo + u * (cdf_hi - cdf_lo)
                u = u.clamp(min=eps_val, max=1.0 - eps_val)
                log_sigma = normal.icdf(u)
                sigma = log_sigma.exp()
                rf_t = sigma / (sigma + 1.0)
            if not isinstance(time_shape, int):
                rf_t = rf_t.view(time_shape)
            return rf_t

    def draw_training_time_G(self, time_shape: Any) -> torch.Tensor:
        return self._sample_rf_time(self.p_G, time_shape)

    def draw_training_time_D(self, time_shape: Any) -> torch.Tensor:
        return self._sample_rf_time(self.p_D, time_shape)

    def draw_training_time_D_region(
        self,
        time_shape: Any,
        region: Literal["high", "low"],
        lower_bound: torch.Tensor | float | None = None,
        strict_lower: bool = False,
    ) -> torch.Tensor:
        if not self.config.joint_wan22_t2v:
            D_time = self.draw_training_time_D(time_shape)
            if lower_bound is not None:
                lb = lower_bound if isinstance(lower_bound, torch.Tensor) else torch.as_tensor(lower_bound, device=D_time.device, dtype=D_time.dtype)
                if not isinstance(lb, torch.Tensor):
                    lb = torch.as_tensor(lb, device=D_time.device, dtype=D_time.dtype)
                while lb.ndim < D_time.ndim:
                    lb = lb.unsqueeze(-1)
                D_time = torch.maximum(D_time, lb + (1e-4 if strict_lower else 0.0))
            return D_time

        D_time = self.draw_training_time_D(time_shape)
        boundary = float(self.config.joint_boundary_rf)
        if region == "high":
            lo = torch.full_like(D_time, boundary)
            hi = torch.ones_like(D_time)
        elif region == "low":
            lo = torch.zeros_like(D_time)
            hi = torch.ones_like(D_time) if self.config.joint_low_d_time_full_range else torch.full_like(D_time, boundary)
        else:
            raise ValueError(f"Unknown joint region: {region}")
        if lower_bound is not None:
            lb = lower_bound if isinstance(lower_bound, torch.Tensor) else torch.as_tensor(lower_bound, device=D_time.device, dtype=D_time.dtype)
            lb = lb.to(device=D_time.device, dtype=D_time.dtype)
            while lb.ndim < D_time.ndim:
                lb = lb.unsqueeze(-1)
            lo = torch.maximum(lo, lb)
        margin = 1e-4 if strict_lower else 0.0
        min_allowed = lo + margin
        width = (hi - min_allowed).clamp(min=1e-6)
        invalid = (D_time < min_allowed) | (D_time > hi)
        fallback = min_allowed + torch.rand_like(D_time) * width
        return torch.where(invalid, fallback, D_time).clamp(min=0.0, max=1.0)

    def denoise(
        self,
        xt_B_C_T_H_W: torch.Tensor,
        time: torch.Tensor,
        condition: TextCondition,
        net_type: Literal["teacher", "fake_score", "student"] = "teacher",
        region: Literal["high", "low"] = "high",
    ) -> DenoisePrediction:
        """
        Network forward to denoise the input noised data given noise level, and condition.

        rf-native (rectified flow): time is rf time s in [0,1], xt is the linear-rf
        latent x_s = (1-s)*x0 + s*eps. The net predicts the rf velocity v = eps - x0,
        fed with timesteps = s * rectified_flow_t_scaling_factor. We reconstruct
        x0 = x_s - s*v; DenoisePrediction.F is the rf velocity v.

        This function supports different net types:
        - teacher: the teacher diffusion model
        - fake_score: the fake score net on student generator's outputs
        - student: the student net (few-step generator)

        Returns:
            DenoisePrediction: clean data prediction (x0) and rf velocity (F=v).
        """
        if time.ndim == 1:
            time_B_T = repeat(time, "b -> b 1")
        elif time.ndim == 2:
            time_B_T = time
        else:
            raise ValueError(f"time shape {time.shape} is not supported")
        s_B_1_T_1_1 = rearrange(time_B_T, "b t -> b 1 t 1 1")

        if net_type == "student":
            net = self._student_net(region)
        elif net_type == "teacher":
            net = self._teacher_net(region)
        elif net_type == "fake_score":
            net = self._fake_score_net(region)
        else:
            raise ValueError(f"Unknown net_type: {net_type}")

        v_B_C_T_H_W = net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=(time_B_T * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
            **condition.to_dict(),
        ).float()

        # rf reconstruction of x0:  x_s = (1-s)*x0 + s*eps, v = eps - x0  =>  x0 = x_s - s*v
        x0_pred_B_C_T_H_W = xt_B_C_T_H_W - s_B_1_T_1_1 * v_B_C_T_H_W
        return DenoisePrediction(x0=x0_pred_B_C_T_H_W, F=v_B_C_T_H_W)

    def _init_simulation_x(self, x_B_C_T_H_W_size, condition, start_t: float):
        """Starting latent for backward_simulation at rf time `start_t`.

        Default: pure noise (correct only when start_t == 1.0).
        With `init_rollout_ckpt` set (Wan2.2 low-noise expert): roll the frozen
        high-noise student from rf 1.0 down to `start_t` with no_grad rf-Euler, so
        the low expert trains on the same input distribution it sees at inference.
        """
        x_B_C_T_H_W = self.sync(torch.randn(x_B_C_T_H_W_size, device="cuda"))
        if self.net_init_rollout is None or start_t >= 1.0:
            return x_B_C_T_H_W
        s_traj = [1.0, *list(self.config.init_rollout_timesteps), start_t]
        with torch.no_grad():
            for s_cur, s_next in zip(s_traj[:-1], s_traj[1:]):
                s_cur_B_1 = s_cur * torch.ones(x_B_C_T_H_W_size[0], 1, device="cuda")
                v_B_C_T_H_W = self.net_init_rollout(
                    x_B_C_T_H_W=x_B_C_T_H_W.to(**self.tensor_kwargs),
                    timesteps_B_T=(s_cur_B_1 * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
                    **condition.to_dict(),
                ).float()
                x_B_C_T_H_W = x_B_C_T_H_W + (s_next - s_cur) * v_B_C_T_H_W
        return x_B_C_T_H_W.detach()

    def backward_simulation(self, condition, x_B_C_T_H_W_size, n_steps, with_grad: bool = False):
        # rf-native: all times are rf s in [0,1]; state x is the linear-rf latent
        # x_s = (1-s)*x0 + s*eps. start_t < 0 => auto rf start 1.0 (pure noise).
        if self.config.joint_wan22_t2v:
            schedule = self._joint_step_schedule()
            assert n_steps <= len(schedule) - 1, f"n_steps={n_steps} exceeds joint schedule length {len(schedule) - 1}"
            t_values = schedule[:n_steps] + [schedule[-1]]
            t_traj = [t * torch.ones(x_B_C_T_H_W_size[0], 1, device="cuda") for t in t_values]
            x_B_C_T_H_W = self.sync(torch.randn(x_B_C_T_H_W_size, device="cuda"))
            x_traj = [x_B_C_T_H_W]
            for step, (s_cur_B_1, s_next_B_1) in enumerate(zip(t_traj[:-1], t_traj[1:])):
                region = self._joint_region_for_step_index(step)
                net = self._student_net(region)
                context_fn = torch.enable_grad if with_grad and step == n_steps - 1 else torch.no_grad
                with context_fn():
                    s_cur_B_1_T_1_1 = rearrange(s_cur_B_1, "b 1 -> b 1 1 1 1")
                    s_next_B_1_T_1_1 = rearrange(s_next_B_1, "b 1 -> b 1 1 1 1")
                    v_B_C_T_H_W = net(
                        x_B_C_T_H_W=x_B_C_T_H_W.to(**self.tensor_kwargs),
                        timesteps_B_T=(s_cur_B_1 * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
                        **condition.to_dict(),
                    ).float()
                    x_B_C_T_H_W = x_B_C_T_H_W + (s_next_B_1_T_1_1 - s_cur_B_1_T_1_1) * v_B_C_T_H_W
                x_traj.append(x_B_C_T_H_W.detach())
            return x_B_C_T_H_W, (t_traj, x_traj)

        start_t = self.config.backward_simulation_start_t
        if start_t < 0:
            start_t = 1.0
        end_t = self.config.backward_simulation_end_t

        t_i_B_1 = start_t * torch.ones(x_B_C_T_H_W_size[0], 1, device="cuda")
        x_B_C_T_H_W = self._init_simulation_x(x_B_C_T_H_W_size, condition, start_t)
        t_traj, x_traj = [t_i_B_1], [x_B_C_T_H_W]
        for i in range(n_steps - 1):
            if not self.config.dmd_fix_timesteps:
                t_i_B_1 = torch.minimum(self.draw_training_time_D((x_B_C_T_H_W_size[0], 1)), t_i_B_1)
                t_i_B_1 = self.sync(t_i_B_1)
                t_traj.append(t_i_B_1)
            else:
                # Fall back to end_t when the schedule is exhausted (same guard as
                # backward_simulation_tdm); Wan2.2 per-expert schedules have 1 entry.
                if i < len(self.config.backward_timesteps):
                    backward_t = self.config.backward_timesteps[i]
                else:
                    backward_t = end_t
                t_i_B_1 = backward_t * torch.ones(x_B_C_T_H_W_size[0], 1, device="cuda")
                t_traj.append(t_i_B_1)
        t_traj.append(end_t * torch.ones(x_B_C_T_H_W_size[0], 1, device="cuda"))
        for step, (s_cur_B_1, s_next_B_1) in enumerate(zip(t_traj[:-1], t_traj[1:])):
            context_fn = torch.enable_grad if with_grad and step == n_steps - 1 else torch.no_grad
            with context_fn():
                # rf-Euler: dx/ds = v, v = net(x_s, s*scale) = eps - x0.
                s_cur_B_1_T_1_1 = rearrange(s_cur_B_1, "b 1 -> b 1 1 1 1")
                s_next_B_1_T_1_1 = rearrange(s_next_B_1, "b 1 -> b 1 1 1 1")
                v_B_C_T_H_W = self.net(
                    x_B_C_T_H_W=x_B_C_T_H_W.to(**self.tensor_kwargs),
                    timesteps_B_T=(s_cur_B_1 * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
                    **condition.to_dict(),
                ).float()
                x_B_C_T_H_W = x_B_C_T_H_W + (s_next_B_1_T_1_1 - s_cur_B_1_T_1_1) * v_B_C_T_H_W
            x_traj.append(x_B_C_T_H_W.detach())
        return x_B_C_T_H_W, (t_traj, x_traj)

    def backward_simulation_tdm(self, condition, x_B_C_T_H_W_size, n_steps, with_grad: bool = False):
        """TDM variant of backward_simulation.

        Identical to backward_simulation except: instead of collapsing the last
        step to x0 at t=0, we draw ONE EXTRA timestep t_{i-1} from the same
        schedule (i.e. "the next timestep in the list") and run the same
        ode/sde branch from t_{i} (= t_traj[-2]) to t_{i-1} (= t_traj[-1]).

        The returned t_traj has length n_steps + 1:
            [1.0, s_1, ..., s_{i}, s_{i-1}]   (rf times)
        where t_{i-1} is taken from self.config.backward_timesteps when
        available (index n_steps - 1). When the schedule is exhausted we fall
        back to t_{i-1} = 0 (i.e. that batch element degenerates to DMD).
        """
        if self.config.joint_wan22_t2v:
            schedule = self._joint_step_schedule()
            assert n_steps <= len(schedule) - 1, f"n_steps={n_steps} exceeds joint schedule length {len(schedule) - 1}"
            t_values = schedule[: n_steps + 1]
            t_traj = [t * torch.ones(x_B_C_T_H_W_size[0], 1, device="cuda") for t in t_values]
            x_B_C_T_H_W = self.sync(torch.randn(x_B_C_T_H_W_size, device="cuda"))
            x_traj = [x_B_C_T_H_W]
            for step, (s_cur_B_1, s_next_B_1) in enumerate(zip(t_traj[:-1], t_traj[1:])):
                region = self._joint_region_for_step_index(step)
                net = self._student_net(region)
                context_fn = torch.enable_grad if with_grad and step == n_steps - 1 else torch.no_grad
                with context_fn():
                    s_cur_B_1_T_1_1 = rearrange(s_cur_B_1, "b 1 -> b 1 1 1 1")
                    s_next_B_1_T_1_1 = rearrange(s_next_B_1, "b 1 -> b 1 1 1 1")
                    v_B_C_T_H_W = net(
                        x_B_C_T_H_W=x_B_C_T_H_W.to(**self.tensor_kwargs),
                        timesteps_B_T=(s_cur_B_1 * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
                        **condition.to_dict(),
                    ).float()
                    x_B_C_T_H_W = x_B_C_T_H_W + (s_next_B_1_T_1_1 - s_cur_B_1_T_1_1) * v_B_C_T_H_W
                x_traj.append(x_B_C_T_H_W.detach())
            return x_B_C_T_H_W, (t_traj, x_traj)

        start_t = self.config.backward_simulation_start_t
        if start_t < 0:
            start_t = 1.0
        end_t = self.config.backward_simulation_end_t
        t_i_B_1 = start_t * torch.ones(x_B_C_T_H_W_size[0], 1, device="cuda")
        x_B_C_T_H_W = self._init_simulation_x(x_B_C_T_H_W_size, condition, start_t)
        t_traj, x_traj = [t_i_B_1], [x_B_C_T_H_W]
        # n_steps timesteps in total (one extra compared to DMD's n_steps - 1).
        # The last appended timestep IS t_{i-1}, drawn from the same schedule.
        for i in range(n_steps):
            assert self.config.dmd_fix_timesteps is True
            if i < len(self.config.backward_timesteps):
                backward_t = self.config.backward_timesteps[i]
            else:
                backward_t = end_t
            t_i_B_1 = backward_t * torch.ones(x_B_C_T_H_W_size[0], 1, device="cuda")
            t_traj.append(t_i_B_1)
        for step, (s_cur_B_1, s_next_B_1) in enumerate(zip(t_traj[:-1], t_traj[1:])):
            context_fn = torch.enable_grad if with_grad and step == n_steps - 1 else torch.no_grad
            with context_fn():
                # rf-Euler; unlike DMD's backward_simulation the LAST step also performs
                # the Euler step (no x0 collapse) so we land at s_{i-1}.
                s_cur_B_1_T_1_1 = rearrange(s_cur_B_1, "b 1 -> b 1 1 1 1")
                s_next_B_1_T_1_1 = rearrange(s_next_B_1, "b 1 -> b 1 1 1 1")
                v_B_C_T_H_W = self.net(
                    x_B_C_T_H_W=x_B_C_T_H_W.to(**self.tensor_kwargs),
                    timesteps_B_T=(s_cur_B_1 * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
                    **condition.to_dict(),
                ).float()
                x_B_C_T_H_W = x_B_C_T_H_W + (s_next_B_1_T_1_1 - s_cur_B_1_T_1_1) * v_B_C_T_H_W
            x_traj.append(x_B_C_T_H_W.detach())
        return x_B_C_T_H_W, (t_traj, x_traj)

    def _student_one_step(self, x_t_B_C_T_H_W, t_cur_B_1, t_next_B_1, condition):
        """Single rf-Euler student jump from rf time s_cur to s_next.

        x is the linear-rf latent x_s=(1-s)x0+s*eps; v=net(x, s*scale)=eps-x0.

        Returns:
            x_t_next: x at s_next.
            x0_pred:  student x0 prediction = x - s_cur*v.
        """
        s_cur_5d = rearrange(t_cur_B_1, "b 1 -> b 1 1 1 1")
        s_next_5d = rearrange(t_next_B_1, "b 1 -> b 1 1 1 1")
        v_rf = self.net(
            x_B_C_T_H_W=x_t_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=(t_cur_B_1 * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
            **condition.to_dict(),
        ).float()
        x_t_next = x_t_B_C_T_H_W + (s_next_5d - s_cur_5d) * v_rf
        x0_pred = x_t_B_C_T_H_W - s_cur_5d * v_rf
        return x_t_next, x0_pred

    @staticmethod
    def _build_div_teacher_schedule(K: int, sigma_max: float, shift: float, device, dtype=torch.float32):
        """Build rf-domain timesteps for the teacher rollout (ode_match diversity).

        Matches generate_samples_from_batch_teacher's schedule (shift=5, N=30) and
        returns rf times directly (s in [0,1]); no TrigFlow conversion.
        """
        K = int(K)
        assert K >= 1, f"div_K must be >= 1, got {K}"
        N_INFERENCE = 30  # match generate_samples_from_batch_teacher's default num_steps
        assert K + 1 <= N_INFERENCE, (f"div_K + 1 = {K + 1} must be <= {N_INFERENCE} (teacher inference step count)")
        sigma_max_rf = sigma_max / (sigma_max + 1.0)
        unshifted = sigma_max_rf / (shift - (shift - 1.0) * sigma_max_rf)
        raw_full = torch.linspace(unshifted, 0.0, N_INFERENCE + 1, device=device, dtype=dtype)[:-1]
        sigmas_full = shift * raw_full / (1.0 + (shift - 1.0) * raw_full)
        rf_t = sigmas_full[: K + 1]  # rf times (s), already in [0,1]
        return rf_t

    def _rollout_teacher_to_t_tilde(self, condition, x_B_C_T_H_W_size):
        """rf-Euler teacher K-step rollout from pure noise (s=1) to s=t_tilde (no grad).

        Returns:
            eps_B_C_T_H_W: x at s=1 (pure noise).
            z_tilde_B_C_T_H_W: x at s=t_tilde (last rf grid entry).
            t_tilde: scalar rf time of the rollout endpoint.
        """
        s_grid = self._build_div_teacher_schedule(
            self.config.div_K, self.config.div_sigma_max, self.config.teacher_timestep_shift, device="cuda", dtype=torch.float32
        )
        K = s_grid.shape[0] - 1

        eps_B_C_T_H_W = torch.randn(x_B_C_T_H_W_size, device="cuda")
        eps_B_C_T_H_W = self.sync(eps_B_C_T_H_W)

        x_B_C_T_H_W = eps_B_C_T_H_W
        with torch.no_grad():
            for k in range(K):
                s_cur_B_1 = s_grid[k].view(1, 1).expand(x_B_C_T_H_W_size[0], 1).contiguous()
                s_next_B_1 = s_grid[k + 1].view(1, 1).expand(x_B_C_T_H_W_size[0], 1).contiguous()
                s_next_5d = rearrange(s_next_B_1, "b 1 -> b 1 1 1 1")
                s_cur_5d = rearrange(s_cur_B_1, "b 1 -> b 1 1 1 1")
                # Teacher rf velocity at s_cur; rf-Euler step.
                v_B_C_T_H_W = self.denoise(x_B_C_T_H_W, s_cur_B_1, condition, net_type="teacher").F.float()
                x_B_C_T_H_W = x_B_C_T_H_W + (s_next_5d - s_cur_5d) * v_B_C_T_H_W
        t_tilde = float(s_grid[-1].item())
        return eps_B_C_T_H_W, x_B_C_T_H_W, t_tilde

    def _compute_diversity_loss(self, condition, eps_B_C_T_H_W, z_tilde_B_C_T_H_W, t_tilde):
        """rf diversity loss (ode_match): match student's rf velocity at s=1 to the
        teacher's rf average velocity over [t_tilde, 1].

            v_tilde = (eps - z_tilde) / (1 - t_tilde)       (no grad; rf avg velocity)
            v_stu   = net(eps, 1.0 * scale)                 (with grad; rf velocity at s=1)
            loss_div = sum_dims((v_stu - v_tilde)^2)
        """
        d_rf_t = 1.0 - t_tilde  # rf_t(s=1)=1 minus rf_t(t_tilde)=t_tilde
        assert d_rf_t > 1e-6, f"1 - t_tilde too small: {d_rf_t}"
        with torch.no_grad():
            v_tilde = (eps_B_C_T_H_W - z_tilde_B_C_T_H_W) / d_rf_t
        B = eps_B_C_T_H_W.shape[0]
        s_cur_B_1 = torch.ones((B, 1), device=eps_B_C_T_H_W.device, dtype=torch.float32)  # s=1
        v_stu = self.net(
            x_B_C_T_H_W=eps_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=(s_cur_B_1 * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
            **condition.to_dict(),
        ).float()
        diff = v_stu - v_tilde
        loss_div = (diff ** 2).sum(dim=(1, 2, 3, 4))
        bad = ~torch.isfinite(loss_div)
        if bad.any():
            loss_div = torch.where(bad, torch.zeros_like(loss_div), loss_div)
        return loss_div

    def _student_dmd_step(self, ctx, iteration):
        """rf-native DMD student update (sibling of TDM).

        Uses backward_simulation (collapses to x0 at s=0) -> G_x0, then a
        score-distillation surrogate compared in x0 space. Diversity dispatch
        (ode_match / pcm) is shared with TDM.
        """
        log.debug(f"Student update {iteration} (DMD)")
        x0_B_C_T_H_W, condition, uncondition = ctx
        if self.config.joint_wan22_t2v:
            num_simulation_steps_fake = self._joint_num_simulation_steps(iteration, "student")
            region = self._joint_region_for_num_steps(num_simulation_steps_fake)
        else:
            region = "high"
            if self.config.loss_scale_div > 0:
                # Diversity supplies the s=1 supervision, so the first (s=1) simulation
                # segment must stay no_grad => force n_steps >= 2.
                assert self.config.max_simulation_steps_fake >= 2, (
                    "loss_scale_div > 0 requires max_simulation_steps_fake >= 2 "
                    "so that the first (s=1) segment of backward_simulation stays no_grad.")
                num_simulation_steps_fake = (self.get_effective_iteration(iteration) % (self.config.max_simulation_steps_fake - 1) + 2)
            else:
                num_simulation_steps_fake = self.get_effective_iteration(iteration) % self.config.max_simulation_steps_fake + 1
        G_x0_B_C_T_H_W, (t_traj, x_traj) = self.backward_simulation(condition, x0_B_C_T_H_W.size(), num_simulation_steps_fake, with_grad=True)
        D_time_B_1 = self.draw_training_time_D_region((x0_B_C_T_H_W.shape[0], 1), region)
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), device="cuda")
        D_time_B_1, epsilon_B_C_T_H_W = self.sync(D_time_B_1, epsilon_B_C_T_H_W)
        s_D_5d = rearrange(D_time_B_1, "b t -> b 1 t 1 1")
        # rf forward noise: x_{s_D} = (1-s_D)*G_x0 + s_D*eps
        D_xt_B_C_T_H_W = (1.0 - s_D_5d) * G_x0_B_C_T_H_W + s_D_5d * epsilon_B_C_T_H_W

        with torch.no_grad():
            x0_fake_B_C_T_H_W = self.denoise(D_xt_B_C_T_H_W, D_time_B_1, condition, net_type="fake_score", region=region).x0
        with torch.no_grad():
            x0_teacher_B_C_T_H_W = self.denoise(D_xt_B_C_T_H_W, D_time_B_1, condition, net_type="teacher", region=region).x0
            if self.config.teacher_guidance > 1.0:
                x0_teacher_uncond_B_C_T_H_W = self.denoise(D_xt_B_C_T_H_W, D_time_B_1, uncondition, net_type="teacher", region=region).x0
                x0_teacher_B_C_T_H_W = x0_teacher_uncond_B_C_T_H_W + self.config.teacher_guidance * (
                    x0_teacher_B_C_T_H_W - x0_teacher_uncond_B_C_T_H_W
                )
        with torch.no_grad():
            weight_factor = (
                torch.abs(G_x0_B_C_T_H_W.double() - x0_teacher_B_C_T_H_W.double()).mean(dim=[1, 2, 3, 4], keepdim=True).clip(min=0.00001)
            )
        grad_B_C_T_H_W = (x0_fake_B_C_T_H_W.double() - x0_teacher_B_C_T_H_W.double()) / weight_factor
        loss_dmd = (G_x0_B_C_T_H_W.double() - (G_x0_B_C_T_H_W.double() - grad_B_C_T_H_W).detach()) ** 2
        loss_dmd[torch.isnan(loss_dmd).flatten(start_dim=1).any(dim=1)] = 0
        loss_dmd = loss_dmd.sum(dim=(1, 2, 3, 4))
        kendall_loss = self.config.loss_scale_dmd * loss_dmd

        output_batch = {
            "G_x0": G_x0_B_C_T_H_W.detach().cpu(),
            "D_xt": D_xt_B_C_T_H_W.detach().cpu(),
            "D_time": D_time_B_1.detach().cpu(),
            "x0_fake": x0_fake_B_C_T_H_W.detach().cpu(),
            "x0_teacher": x0_teacher_B_C_T_H_W.detach().cpu(),
            "loss_dmd": float((self.config.loss_scale_dmd * loss_dmd).detach().sum()),
        }

        # Diversity supervision (shared with TDM): ode_match or pcm.
        div_scale = self.config.loss_scale_div if (not self.config.joint_wan22_t2v or region == "high") else self.config.joint_low_loss_scale_div
        if div_scale > 0 and (not self.config.joint_wan22_t2v or num_simulation_steps_fake >= 2):
            div_mode = self.config.div_mode
            if div_mode == "ode_match":
                eps_div_B_C_T_H_W, z_tilde_B_C_T_H_W, t_tilde = self._rollout_teacher_to_t_tilde(condition, x0_B_C_T_H_W.size())
                loss_div = self._compute_diversity_loss(condition, eps_div_B_C_T_H_W, z_tilde_B_C_T_H_W, t_tilde)
                output_batch["z_tilde"] = z_tilde_B_C_T_H_W.detach().cpu()
            elif div_mode in ("pcm_fixed", "pcm_random"):
                loss_div, pcm_div_aux = self._compute_pcm_div_loss(x0_B_C_T_H_W, condition, uncondition, div_mode)
                output_batch.update(pcm_div_aux)
            else:
                raise ValueError(f"Unknown div_mode: {div_mode}")
            kendall_loss = kendall_loss + div_scale * loss_div
            output_batch["loss_div"] = float((div_scale * loss_div).detach().sum())

        return output_batch, kendall_loss

    def _compute_pcm_div_loss(self, x0_B_C_T_H_W, condition, uncondition, mode):
        """PCM-style Diversity Supervision on the FIRST backward interval.

        Aligns the student boundary prediction at t = backward_timesteps[0]
        between two starting points within phase 0 (rf [1.0, backward_timesteps[0]]):
          - student G(x_t0, t0) -> x_boundary_pred       (with grad)
          - teacher K-step Euler from t0 to tK,
            student G(x_tK, tK) -> x_boundary_target     (no grad)
          - L = || x_boundary_pred - sg(x_boundary_target) ||^2

        One of the three variants under loss_scale_div (selected by div_mode).
        Reuses dcm_total_steps / dcm_skipping_interval_steps / teacher_timestep_shift.

        mode:
          "pcm_fixed":  w_0 = 0  (=> rf t0 = 1, pure-noise input).
          "pcm_random": w_0 ~ U(0, valid_width) within phase 0 (same as _student_pcm_step).

        Returns (loss [B], aux dict).
        """
        assert mode in ("pcm_fixed", "pcm_random"), f"Unknown PCM-div mode: {mode}"
        B = x0_B_C_T_H_W.shape[0]
        K = self.config.dcm_skipping_interval_steps
        dw = 1.0 / self.config.dcm_total_steps
        shift = self.config.teacher_timestep_shift

        epsilon_B_C_T_H_W = self.sync(torch.randn(x0_B_C_T_H_W.size(), device="cuda"))

        # Phase 0 right boundary (rf domain) — the alignment target.
        rf_b = float(self.config.joint_backward_timesteps[0] if self.config.joint_wan22_t2v else self.config.backward_timesteps[0])
        s_b = rf_b / (shift - (shift - 1.0) * rf_b) if shift > 0 else rf_b
        w_right = 1.0 - s_b  # phase 0 width = w_right - 0
        valid_width = w_right - K * dw

        # 1. Sample w_0 within phase 0 according to mode.
        if mode == "pcm_fixed":
            # w_0 = 0 -> s_0 = 1 -> rf_t0 = 1 -> xt = epsilon (pure noise).
            w_B_1 = torch.zeros(B, 1, device="cuda", dtype=torch.float32)
        else:
            assert valid_width > 0, (
                f"PCM-div phase 0 valid width <= 0: w_right={w_right:.4f}, K*dw={K*dw:.4f}. "
                f"Reduce dcm_skipping_interval_steps or widen backward_timesteps[0]."
            )
            w_B_1 = self.sync(torch.rand(B, 1, device="cuda") * valid_width)

        # 2. Build t_list[0..K] in rf domain (same convention as _student_pcm_step).
        t_list = []
        for k in range(K + 1):
            s_k_B_1 = 1.0 - (w_B_1 + k * dw)
            rf_t_k_B_1 = shift_rf_time(s_k_B_1, shift)
            t_list.append(rf_t_k_B_1)
        t0_B_1, tK_B_1 = t_list[0], t_list[-1]

        # 3. Phase boundary (rf, broadcast to B).
        t_bnd_B_1 = torch.full((B, 1), rf_b, device="cuda", dtype=torch.float32)

        # 4. Noisy input at t0 (linear rf). pcm_fixed -> rf_t0=1 -> xt = epsilon.
        t0_5d = rearrange(t0_B_1, "b 1 -> b 1 1 1 1")
        xt_B_C_T_H_W = (1.0 - t0_5d) * x0_B_C_T_H_W + t0_5d * epsilon_B_C_T_H_W

        # 5. Teacher K-step Euler + target student (no_grad).
        # IMPORTANT: do no_grad forward BEFORE the grad-enabled forward below.
        # This avoids FSDP deadlock: FSDP's all-gather inside no_grad would conflict
        # with the retained activations from a preceding grad-enabled forward on the
        # same module. By running no_grad first, FSDP state is clean.
        with torch.no_grad():
            xk_B_C_T_H_W = xt_B_C_T_H_W
            for k in range(K):
                tk_B_1 = t_list[k]
                tk1_B_1 = t_list[k + 1]
                dt_B_1 = tk_B_1 - tk1_B_1
                F_teacher_B_C_T_H_W = self.denoise(xk_B_C_T_H_W, tk_B_1, condition, net_type="teacher").F
                if self.config.teacher_guidance > 1.0:
                    F_uncond_B_C_T_H_W = self.denoise(xk_B_C_T_H_W, tk_B_1, uncondition, net_type="teacher").F
                    F_teacher_B_C_T_H_W = F_uncond_B_C_T_H_W + self.config.teacher_guidance * (F_teacher_B_C_T_H_W - F_uncond_B_C_T_H_W)
                xk_B_C_T_H_W = xk_B_C_T_H_W - rearrange(dt_B_1, "b 1 -> b 1 1 1 1") * F_teacher_B_C_T_H_W

            # Student one-step jump from tK to t_bnd (no_grad target).
            x_boundary_target, _ = self._student_one_step(
                xk_B_C_T_H_W, tK_B_1, t_bnd_B_1, condition,
            )
        x_boundary_target = x_boundary_target.detach()

        # 6. Student prediction one-step jump from t0 to t_bnd (with grad).
        # Runs AFTER the no_grad block so FSDP state is clean.
        x_boundary_pred, _ = self._student_one_step(
            xt_B_C_T_H_W, t0_B_1, t_bnd_B_1, condition,
        )

        loss_pcm_div = (x_boundary_pred - x_boundary_target) ** 2
        loss_pcm_div[torch.isnan(loss_pcm_div).flatten(start_dim=1).any(dim=1)] = 0
        loss_pcm_div = loss_pcm_div.sum(dim=(1, 2, 3, 4))

        aux = {
            "pcm_div_t0": t0_B_1.detach().cpu(),
            "pcm_div_x_target": x_boundary_target.detach().cpu(),
        }
        return loss_pcm_div, aux

    def _make_training_ctx(self, x0_B_C_T_H_W, condition, uncondition, iteration):
        x0_B_C_T_H_W, condition, uncondition = self.sync(x0_B_C_T_H_W, condition, uncondition)
        return (x0_B_C_T_H_W, condition, uncondition)

    # ===================== Gradient Conflict Diagnostic =====================

    def _student_tdm_step(self, ctx, iteration):
        """TDM (Trajectory Distribution Matching) student update.

        Differences from _student_dmd_step:
          1. backward_simulation_tdm: last step does NOT reach 0; it stops at
             t_{i-1}, the next sampled timestep in t_traj (one extra step
             beyond DMD). Returns G_x_{t_{i-1}}, NOT G_x0.
          2. D_time is clamped to be > t_{i-1}.
          3. fake/teacher denoise outputs are further mapped to t_{i-1} via the
             rf PF-ODE analytical step (NOT all the way to x0).
        The DMD-style score-distillation surrogate is reused, with the comparison
        domain shifted from x0 to x_{t_{i-1}}.
        """
        log.debug(f"Student update {iteration} (TDM)")
        x0_B_C_T_H_W, condition, uncondition = ctx
        if self.config.joint_wan22_t2v:
            num_simulation_steps_fake = self._joint_num_simulation_steps(iteration, "student")
            region = self._joint_region_for_num_steps(num_simulation_steps_fake)
        else:
            region = "high"
            if self.config.loss_scale_div > 0:
                # diversity onstraint
                assert self.config.max_simulation_steps_fake >= 2, (
                    "loss_scale_div > 0 requires max_simulation_steps_fake >= 2 "
                    "so that the first (rf s = 1) segment of backward_simulation_tdm "
                    "stays no_grad."
                )
                num_simulation_steps_fake = (
                    self.get_effective_iteration(iteration) % (self.config.max_simulation_steps_fake - 1) + 2
                )
            else:
                num_simulation_steps_fake = self.get_effective_iteration(iteration) % self.config.max_simulation_steps_fake + 1
        G_x_t_im1_B_C_T_H_W, (t_traj, x_traj) = self.backward_simulation_tdm(condition, x0_B_C_T_H_W.size(), num_simulation_steps_fake, with_grad=True)
        t_im1_B_1 = t_traj[-1]  # [B, 1], = the next sampled timestep (t_{i-1})
        s1_5d = rearrange(t_im1_B_1, "b t -> b 1 t 1 1")  # rf time s_{i-1}

        D_time_B_1 = self.draw_training_time_D_region(
            (x0_B_C_T_H_W.shape[0], 1),
            region,
            lower_bound=t_im1_B_1,
            strict_lower=True,
        )
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), device="cuda")
        D_time_B_1, epsilon_B_C_T_H_W = self.sync(D_time_B_1, epsilon_B_C_T_H_W)
        s_D_5d = rearrange(D_time_B_1, "b t -> b 1 t 1 1")

        # rf forward transition q(x_{s_D} | x_{s_{i-1}}), linear-rf equivalent of the
        # VP SDE transition. G is x_{s_{i-1}} (NOT x0):
        #   D_xt = ((1-s_D)/(1-s1))*G + sqrt((s_D-s1)*(s_D+s1-2*s_D*s1))/(1-s1) * eps
        # Boundary checks: s1->0 => (1-s_D)*x0 + s_D*eps (matches DMD);
        #                  s_D->s1 => noise->0, D_xt->G (consistent).
        one_minus_s1 = (1.0 - s1_5d).clamp(min=1e-6)
        sigma_cond = torch.sqrt(
            ((s_D_5d - s1_5d) * (s_D_5d + s1_5d - 2.0 * s_D_5d * s1_5d)).clamp(min=0.0)
        ) / one_minus_s1
        D_xt_B_C_T_H_W = ((1.0 - s_D_5d) / one_minus_s1) * G_x_t_im1_B_C_T_H_W + sigma_cond * epsilon_B_C_T_H_W

        # Helper: rf PF-ODE analytic step from (x_{s_D}, s_D) back to s_{i-1}.
        def to_t_im1(x_t, x0_pred):
            eps_pred = (x_t - (1.0 - s_D_5d) * x0_pred) / s_D_5d.clamp(min=1e-6)
            return (1.0 - s1_5d) * x0_pred + s1_5d * eps_pred

        with torch.no_grad():
            x0_fake_B_C_T_H_W = self.denoise(D_xt_B_C_T_H_W, D_time_B_1, condition, net_type="fake_score", region=region).x0
            x_t_im1_fake_B_C_T_H_W = to_t_im1(D_xt_B_C_T_H_W, x0_fake_B_C_T_H_W)
        with torch.no_grad():
            x0_teacher_B_C_T_H_W = self.denoise(D_xt_B_C_T_H_W, D_time_B_1, condition, net_type="teacher", region=region).x0
            if self.config.teacher_guidance > 1.0:
                x0_teacher_uncond_B_C_T_H_W = self.denoise(D_xt_B_C_T_H_W, D_time_B_1, uncondition, net_type="teacher", region=region).x0
                x0_teacher_B_C_T_H_W = x0_teacher_uncond_B_C_T_H_W + self.config.teacher_guidance * (x0_teacher_B_C_T_H_W - x0_teacher_uncond_B_C_T_H_W)
            x_t_im1_teacher_B_C_T_H_W = to_t_im1(D_xt_B_C_T_H_W, x0_teacher_B_C_T_H_W)
        with torch.no_grad():
            weight_factor = (
                torch.abs(G_x_t_im1_B_C_T_H_W.double() - x_t_im1_teacher_B_C_T_H_W.double())
                .mean(dim=[1, 2, 3, 4], keepdim=True)
                .clip(min=0.00001)
            )
        grad_B_C_T_H_W = (x_t_im1_fake_B_C_T_H_W.double() - x_t_im1_teacher_B_C_T_H_W.double()) / weight_factor
        loss_tdm = (G_x_t_im1_B_C_T_H_W.double() - (G_x_t_im1_B_C_T_H_W.double() - grad_B_C_T_H_W).detach()) ** 2
        loss_tdm[torch.isnan(loss_tdm).flatten(start_dim=1).any(dim=1)] = 0
        loss_tdm = loss_tdm.sum(dim=(1, 2, 3, 4))
        kendall_loss = self.config.loss_scale_tdm * loss_tdm

        output_batch = {
            "G_x_t_im1": G_x_t_im1_B_C_T_H_W.detach().cpu(),
            "D_xt": D_xt_B_C_T_H_W.detach().cpu(),
            "D_time": D_time_B_1.detach().cpu(),
            "t_im1": t_im1_B_1.detach().cpu(),
            "x_t_im1_fake": x_t_im1_fake_B_C_T_H_W.detach().cpu(),
            "x_t_im1_teacher": x_t_im1_teacher_B_C_T_H_W.detach().cpu(),
            # Scaled TDM term, same convention as loss_dmd.
            "loss_tdm": float((self.config.loss_scale_tdm * loss_tdm).detach().sum()),
        }

        # ===================== Diversity Supervision =====================
        # Shared with DMD: ode_match (rf avg-velocity match) or PCM boundary alignment.
        div_scale = self.config.loss_scale_div if (not self.config.joint_wan22_t2v or region == "high") else self.config.joint_low_loss_scale_div
        if div_scale > 0 and (not self.config.joint_wan22_t2v or num_simulation_steps_fake >= 2):
            div_mode = self.config.div_mode
            if div_mode == "ode_match":
                eps_div_B_C_T_H_W, z_tilde_B_C_T_H_W, t_tilde = self._rollout_teacher_to_t_tilde(condition, x0_B_C_T_H_W.size())
                loss_div = self._compute_diversity_loss(condition, eps_div_B_C_T_H_W, z_tilde_B_C_T_H_W, t_tilde)
                output_batch["z_tilde"] = z_tilde_B_C_T_H_W.detach().cpu()
            elif div_mode in ("pcm_fixed", "pcm_random"):
                loss_div, pcm_div_aux = self._compute_pcm_div_loss(x0_B_C_T_H_W, condition, uncondition, div_mode)
                output_batch.update(pcm_div_aux)
            else:
                raise ValueError(f"Unknown div_mode: {div_mode}")
            kendall_loss = kendall_loss + div_scale * loss_div
            output_batch["loss_div"] = float((div_scale * loss_div).detach().sum())
        # ========================================================================

        return output_batch, kendall_loss

    def training_step_critic(self, ctx, iteration) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        log.debug(f"Critic update {iteration}")
        x0_B_C_T_H_W, condition, uncondition = ctx
        if self.config.joint_wan22_t2v:
            num_simulation_steps_fake = self._joint_num_simulation_steps(iteration, "critic")
            region = self._joint_region_for_num_steps(num_simulation_steps_fake)
        else:
            num_simulation_steps_fake = self.get_effective_iteration_fake(iteration) % self.config.max_simulation_steps_fake + 1
            region = "high"
        G_x0_B_C_T_H_W, _ = self.backward_simulation(condition, x0_B_C_T_H_W.size(), num_simulation_steps_fake, with_grad=False)

        D_time_B_1 = self.draw_training_time_D_region((x0_B_C_T_H_W.shape[0], 1), region)
        D_epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), device="cuda")
        D_time_B_1, D_epsilon_B_C_T_H_W = self.sync(D_time_B_1, D_epsilon_B_C_T_H_W)
        s_D_B_1_1_1_1 = rearrange(D_time_B_1, "b t -> b 1 t 1 1")
        # rf forward noise: x_{s_D} = (1-s_D)*x0 + s_D*eps
        D_xt_B_C_T_H_W = (1.0 - s_D_B_1_1_1_1) * G_x0_B_C_T_H_W + s_D_B_1_1_1_1 * D_epsilon_B_C_T_H_W
        x0_fake_B_C_T_H_W = self.denoise(D_xt_B_C_T_H_W, D_time_B_1, condition, net_type="fake_score", region=region).x0
        # Original 1/sin^2(D) weight in rf: sin^2(D) = s_D^2 / (s_D^2 + (1-s_D)^2).
        weight_B_1_1_1_1 = (s_D_B_1_1_1_1**2 + (1.0 - s_D_B_1_1_1_1) ** 2) / s_D_B_1_1_1_1.clamp(min=1e-6) ** 2
        kendall_loss = ((G_x0_B_C_T_H_W - x0_fake_B_C_T_H_W) ** 2 * weight_B_1_1_1_1).sum(dim=(1, 2, 3, 4))
        output_batch = {
            "G_x0": G_x0_B_C_T_H_W.detach().cpu(),
            "D_xt": D_xt_B_C_T_H_W.detach().cpu(),
            "D_time": D_time_B_1.detach().cpu(),
            "x0_fake": x0_fake_B_C_T_H_W.detach().cpu(),
        }
        return output_batch, kendall_loss

    def training_step_closures(self, data_batch, iteration: int):
        _, x0_B_C_T_H_W, condition, uncondition = self.get_data_and_condition(data_batch)

        ctx = self._make_training_ctx(x0_B_C_T_H_W, condition, uncondition, iteration)

        if self.is_student_phase(iteration):
            emit_dmd = self.net_fake_score is not None and self.config.loss_scale_dmd > 0
            emit_tdm = self.net_fake_score is not None and self.config.loss_scale_tdm > 0
            assert not (emit_dmd and emit_tdm), (
                "loss_scale_dmd and loss_scale_tdm are mutually exclusive; set exactly one > 0."
            )
            assert emit_dmd or emit_tdm, (
                "rf-native distillation needs loss_scale_dmd > 0 or loss_scale_tdm > 0."
            )
            region = self._joint_region_for_iteration(iteration, "student") if self.config.joint_wan22_t2v else ""
            prefix = f"{region}_" if region else ""
            if emit_dmd:
                yield f"{prefix}dmd", lambda: self._student_dmd_step(ctx, iteration), True
            else:
                yield f"{prefix}tdm", lambda: self._student_tdm_step(ctx, iteration), True
        else:
            region = self._joint_region_for_iteration(iteration, "critic") if self.config.joint_wan22_t2v else ""
            prefix = f"{region}_" if region else ""
            yield f"{prefix}critic", lambda: self.training_step_critic(ctx, iteration), True

    @torch.no_grad()
    def forward(self, xt, t, condition: TextCondition):
        pass

    # ------------------------ Sampling ------------------------

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        num_steps: int = 4,
        init_noise: torch.Tensor = None,
        mid_t: List[float] | None = None,
    ) -> torch.Tensor:
        input_key = self.config.input_data_key

        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        _, _, condition, uncondition = self.get_data_and_condition(data_batch)

        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)

        if init_noise is None:
            init_noise = torch.randn(
                n_sample,
                *state_shape,
                dtype=torch.float32,
                device=self.tensor_kwargs["device"],
                generator=generator,
            )
        init_noise, condition = self.sync(init_noise, condition)

        if mid_t is None:
            mid_t = self.config.backward_timesteps[: num_steps - 1]

        start_t = self.config.backward_simulation_start_t
        if start_t < 0:
            start_t = self.config.sigma_max / (self.config.sigma_max + 1.0)
        end_t = self.config.backward_simulation_end_t

        t_steps = torch.tensor(
            [start_t] + list(mid_t) + [end_t],
            dtype=torch.float64,
            device=init_noise.device,
        )

        # rf-native sampling: x is the linear-rf latent, scale init noise to the rf start.
        x = init_noise.to(torch.float64) * t_steps[0]
        ones = torch.ones(x.size(0), device=x.device, dtype=x.dtype)
        for i, (s_cur, s_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            timesteps_B_1 = (s_cur.float() * ones.float() * self.config.rectified_flow_t_scaling_factor).unsqueeze(-1)  # [B, 1]
            v_rf = self.net(
                x_B_C_T_H_W=x.to(**self.tensor_kwargs),
                timesteps_B_T=timesteps_B_1.to(**self.tensor_kwargs),
                **condition.to_dict(),
            ).to(torch.float64)
            x = x + (s_next - s_cur) * v_rf
            x = self.sync(x)
        samples = x.float()
        return torch.nan_to_num(samples)

    @torch.no_grad()
    def generate_samples_viz(
        self,
        data_batch: Dict,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        init_noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """Training-time visualization sampler (rf-native, full trajectory).

        Wan2.1 single-student: viz_full_timesteps empty -> defer to
        generate_samples_from_batch (per-expert schedule).

        Wan2.2 dual-expert: viz_full_timesteps holds the WHOLE 4-step
        trajectory's mid knots (e.g. t2v [0.933781, 0.875, 0.608979]). The
        trajectory is walked once from s=1.0 down to 0.0; each step uses
        net_init_rollout (frozen high student) while s_cur is above the boundary
        and self.net below it. For the HIGH expert net_init_rollout is None, so
        self.net drives every step.
        """
        viz_knots = list(self.config.joint_backward_timesteps) if self.config.joint_wan22_t2v else list(self.config.viz_full_timesteps)
        if len(viz_knots) == 0:
            # Wan2.1 single-student: use the per-expert schedule.
            return self.generate_samples_from_batch(
                data_batch,
                seed=seed,
                state_shape=state_shape,
                n_sample=n_sample,
                init_noise=init_noise,
            )

        input_key = self.config.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        _, _, condition, uncondition = self.get_data_and_condition(data_batch)

        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)
        if init_noise is None:
            init_noise = torch.randn(
                n_sample,
                *state_shape,
                dtype=torch.float32,
                device=self.tensor_kwargs["device"],
                generator=generator,
            )
        init_noise, condition = self.sync(init_noise, condition)

        # Full viz trajectory always starts at the rf noise point (s ~ 1.0),
        # independent of the LOW expert's backward_simulation_start_t (= boundary).
        start_t = self._joint_step_schedule()[0] if self.config.joint_wan22_t2v else self.config.sigma_max / (self.config.sigma_max + 1.0)
        boundary = self.config.joint_boundary_rf if self.config.joint_wan22_t2v else self.config.backward_simulation_start_t  # LOW: high/low split at boundary

        t_steps = torch.tensor(
            [start_t] + viz_knots + [0.0],
            dtype=torch.float64,
            device=init_noise.device,
        )

        x = init_noise.to(torch.float64) * t_steps[0]
        ones = torch.ones(x.size(0), device=x.device, dtype=x.dtype)
        if not self.config.joint_wan22_t2v and self.net_init_rollout is None and boundary > 0.0:
            log.warning(
                f"viz: low expert (boundary={boundary}) has no init_rollout_ckpt; the high-noise "
                "segment will be run by the low student itself, which it was never trained on. "
                "Samples will look broken and do not reflect inference."
            )
        for step, (s_cur, s_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            if self.config.joint_wan22_t2v:
                net = self._student_net(self._joint_region_for_step_index(step))
            else:
                use_rollout = self.net_init_rollout is not None and s_cur.item() > boundary + 1e-6
                net = self.net_init_rollout if use_rollout else self.net
            timesteps_B_1 = (s_cur.float() * ones.float() * self.config.rectified_flow_t_scaling_factor).unsqueeze(-1)
            v_rf = net(
                x_B_C_T_H_W=x.to(**self.tensor_kwargs),
                timesteps_B_T=timesteps_B_1.to(**self.tensor_kwargs),
                **condition.to_dict(),
            ).to(torch.float64)
            x = x + (s_next - s_cur) * v_rf
            x = self.sync(x)
        samples = x.float()
        return torch.nan_to_num(samples)

    @torch.no_grad()
    def generate_samples_from_batch_teacher(
        self,
        data_batch: Dict,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        init_noise: torch.Tensor = None,
        num_steps: int = 50,
        sampler="UniPC",
        timestep_shift=None,  # None -> config.teacher_timestep_shift (official per-task value)
    ) -> torch.Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.
            guidance (float): guidance weights
            seed (int): random seed
            state_shape (tuple): shape of the state, default to data batch if not provided
            n_sample (int): number of samples to generate
            num_steps (int): number of steps for the diffusion process
        """
        _, _, condition, uncondition = self.get_data_and_condition(data_batch)

        if timestep_shift is None:
            timestep_shift = self.config.teacher_timestep_shift

        input_key = self.config.input_data_key

        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)

        if init_noise is None:
            init_noise = torch.randn(
                n_sample,
                *state_shape,
                dtype=torch.float32,
                device=self.tensor_kwargs["device"],
                generator=generator,
            )
        init_noise, condition, uncondition = self.sync(init_noise, condition, uncondition)

        x = init_noise.to(torch.float64)

        sigma_max = self.config.sigma_max / (self.config.sigma_max + 1)
        unshifted_sigma_max = sigma_max / (timestep_shift - (timestep_shift - 1) * sigma_max)

        samplers = {"Euler": FlowEulerSampler, "UniPC": FlowUniPCMultistepSampler}
        sampler = samplers[sampler](num_train_timesteps=1000, sigma_max=unshifted_sigma_max, sigma_min=0.0)
        sampler.set_timesteps(num_inference_steps=num_steps, device=self.tensor_kwargs["device"], shift=timestep_shift)

        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        for _, t in enumerate(sampler.timesteps):
            timesteps = t * ones

            with torch.no_grad():
                if self.config.joint_wan22_t2v:
                    t_value = float(t.item())
                    t_rf = t_value / self.config.rectified_flow_t_scaling_factor if t_value > 1.0 else t_value
                    teacher_net = self._teacher_net("high" if t_rf > self.config.joint_boundary_rf else "low")
                else:
                    teacher_net = self.net_teacher
                v_cond = teacher_net(
                    x_B_C_T_H_W=x.to(**self.tensor_kwargs), timesteps_B_T=timesteps.to(**self.tensor_kwargs), **condition.to_dict()
                ).float()
                v_uncond = teacher_net(
                    x_B_C_T_H_W=x.to(**self.tensor_kwargs), timesteps_B_T=timesteps.to(**self.tensor_kwargs), **uncondition.to_dict()
                ).float()

            v_pred = v_uncond + self.config.teacher_guidance * (v_cond - v_uncond)

            x = sampler.step(v_pred, t, x)

        samples = x.float()

        return torch.nan_to_num(samples)

    @torch.no_grad()
    def validation_step(self, data: dict[str, torch.Tensor], iteration: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Current code does nothing.
        """
        pass

    # ------------------------ Distributed Parallel ------------------------

    @staticmethod
    def get_context_parallel_group():
        if parallel_state is not None and parallel_state.is_initialized():
            return parallel_state.get_context_parallel_group()
        return None

    def sync(self, *args):
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1:
            out = tuple(broadcast(arg, cp_group) if isinstance(arg, torch.Tensor) else arg.broadcast(cp_group) for arg in args)
        else:
            out = args
        return out[0] if len(out) == 1 else out

    # ------------------ Data Preprocessing ------------------

    def _normalize_video_inplace(self, data_batch: dict[str, Tensor]) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Warning:
            A warning is issued if the data has not been previously normalized.

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.

        Side Effects:
            Modifies the 'input_data_key' tensor within the 'data_batch' dictionary in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        input_key = self.config.input_data_key
        # only handle video batch
        # Check if the data has already been normalized and avoid re-normalizing
        if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
            assert torch.is_floating_point(data_batch[input_key]), "Video data is not in float format."
            assert torch.all(
                (data_batch[input_key] >= -1.0001) & (data_batch[input_key] <= 1.0001)
            ), f"Video data is not in the range [-1, 1]. get data range [{data_batch[input_key].min()}, {data_batch[input_key].max()}]"
        else:
            assert data_batch[input_key].dtype == torch.uint8, "Video data is not in uint8 format."
            data_batch[input_key] = data_batch[input_key].to(**self.tensor_kwargs) / 127.5 - 1.0
            data_batch[IS_PREPROCESSED_KEY] = True

        from torchvision.transforms.v2 import UniformTemporalSubsample

        expected_length = self.tokenizer.get_pixel_num_frames(self.config.state_t)
        original_length = data_batch[input_key].shape[2]
        if original_length != expected_length:
            video = rearrange(data_batch[input_key], "b c t h w -> b t c h w")
            video = UniformTemporalSubsample(expected_length)(video)
            data_batch[input_key] = rearrange(video, "b t c h w -> b c t h w")

    def _normalize_latent_inplace(self, data_batch: dict[str, Tensor]) -> None:
        latents = data_batch[self.config.input_latent_key]
        assert latents.shape[2] >= self.config.state_t
        data_batch[self.config.input_latent_key] = latents[:, :, : self.config.state_t, :, :]

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor]) -> Tuple[Tensor, TextCondition]:
        if IS_PROCESSED_KEY not in data_batch or not data_batch[IS_PROCESSED_KEY]:
            if self.config.input_latent_key in data_batch:
                self._normalize_latent_inplace(data_batch)
                data_batch[self.config.input_data_key] = self.decode(data_batch[self.config.input_latent_key]).contiguous().float().clamp(-1, 1)
                data_batch[IS_PREPROCESSED_KEY] = True

            self._normalize_video_inplace(data_batch)
            data_batch[self.config.input_latent_key] = self.encode(data_batch[self.config.input_data_key]).contiguous().float()
            data_batch[IS_PROCESSED_KEY] = True

        raw_state = data_batch[self.config.input_data_key]
        latent_state = data_batch[self.config.input_latent_key]
        # Condition
        if self.neg_embed is not None:
            data_batch["neg_t5_text_embeddings"] = repeat(
                self.neg_embed.to(**self.tensor_kwargs), "l d -> b l d", b=data_batch["t5_text_embeddings"].shape[0]
            )
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)
        condition = condition.edit_data_type(DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.VIDEO)

        # I2V: build the condition y_B_C_T_H_W from the first frame and attach it to condition/uncondition.
        # Once the condition object carries y, all teacher/student/fake_score denoise calls
        # and the condition.to_dict() in the sampling loop pass it through automatically, with no need to change each call site.
        if self.config.is_i2v:
            y, first_frame = self._build_i2v_y(latent_state)  # first_frame: [B,3,H,W] range [-1,1]
            # Wan2.1 I2V: extract first-frame CLIP features online
            frame_cond = None
            if self._clip_encoder is not None:
                frame_cond = self._clip_encoder.encode_first_frame(first_frame).to(dtype=self.tensor_kwargs["dtype"])
            condition = self._attach_i2v(condition, y, frame_cond)
            uncondition = self._attach_i2v(uncondition, y, frame_cond)
        return raw_state, latent_state, condition, uncondition

    # ------------------ Checkpointing ------------------

    def model_dict(self) -> Dict[str, Any]:
        model_dict = {"net": self.net}
        if self.net_low is not None:
            model_dict["net_low"] = self.net_low
        if self.net_fake_score:
            model_dict["fake_score"] = self.net_fake_score
        if self.net_fake_score_low:
            model_dict["fake_score_low"] = self.net_fake_score_low
        return model_dict

    def state_dict(self) -> Dict[str, Any]:
        net_state_dict = self.net.state_dict(prefix="net.")
        if self.config.ema.enabled:
            ema_state_dict = self.net_ema.state_dict(prefix="net_ema.")
            net_state_dict.update(ema_state_dict)
        if self.net_low is not None:
            low_state_dict = self.net_low.state_dict(prefix="net_low.")
            net_state_dict.update(low_state_dict)
        if self.net_fake_score:
            fake_score_state_dict = self.net_fake_score.state_dict(prefix="net_fake_score.")
            net_state_dict.update(fake_score_state_dict)
        if self.net_fake_score_low:
            fake_score_low_state_dict = self.net_fake_score_low.state_dict(prefix="net_fake_score_low.")
            net_state_dict.update(fake_score_low_state_dict)
        return net_state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """
        Loads a state dictionary into the model and optionally its EMA counterpart.
        Different from torch strict=False mode, the method will not raise error for unmatched state shape while raise warning.

        Parameters:e
            state_dict (Mapping[str, Any]): A dictionary containing separate state dictionaries for the model and
                                            potentially for an EMA version of the model under the keys 'model' and 'ema', respectively.
            strict (bool, optional): If True, the method will enforce that the keys in the state dict match exactly
                                    those in the model and EMA model (if applicable). Defaults to True.
            assign (bool, optional): If True and in strict mode, will assign the state dictionary directly rather than
                                    matching keys one-by-one. This is typically used when loading parts of state dicts
                                    or using customized loading procedures. Defaults to False.
        """
        _reg_state_dict = collections.OrderedDict()
        _ema_state_dict = collections.OrderedDict()
        _low_state_dict = collections.OrderedDict()
        _fake_score_state_dict = collections.OrderedDict()
        _fake_score_low_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("net_low."):
                _low_state_dict[k.replace("net_low.", "")] = v
            elif k.startswith("net_fake_score_low."):
                _fake_score_low_state_dict[k.replace("net_fake_score_low.", "")] = v
            elif k.startswith("net."):
                _reg_state_dict[k.replace("net.", "")] = v
            elif k.startswith("net_ema."):
                _ema_state_dict[k.replace("net_ema.", "")] = v
            elif k.startswith("net_fake_score."):
                _fake_score_state_dict[k.replace("net_fake_score.", "")] = v

        state_dict = _reg_state_dict

        if strict:
            reg_results: _IncompatibleKeys = self.net.load_state_dict(_reg_state_dict, strict=strict, assign=assign)

            if self.config.ema.enabled:
                ema_results: _IncompatibleKeys = self.net_ema.load_state_dict(_ema_state_dict, strict=strict, assign=assign)
            if self.net_low is not None:
                low_results: _IncompatibleKeys = self.net_low.load_state_dict(_low_state_dict, strict=strict, assign=assign)
            if self.net_fake_score:
                fake_score_results: _IncompatibleKeys = self.net_fake_score.load_state_dict(_fake_score_state_dict, strict=strict, assign=assign)
            if self.net_fake_score_low:
                fake_score_low_results: _IncompatibleKeys = self.net_fake_score_low.load_state_dict(_fake_score_low_state_dict, strict=strict, assign=assign)

            return _IncompatibleKeys(
                missing_keys=reg_results.missing_keys
                + (ema_results.missing_keys if self.config.ema.enabled else [])
                + (low_results.missing_keys if self.net_low is not None else [])
                + (fake_score_results.missing_keys if self.net_fake_score else [])
                + (fake_score_low_results.missing_keys if self.net_fake_score_low else []),
                unexpected_keys=reg_results.unexpected_keys
                + (ema_results.unexpected_keys if self.config.ema.enabled else [])
                + (low_results.unexpected_keys if self.net_low is not None else [])
                + (fake_score_results.unexpected_keys if self.net_fake_score else [])
                + (fake_score_low_results.unexpected_keys if self.net_fake_score_low else []),
            )
        else:
            log.critical("load model in non-strict mode")
            log.critical(non_strict_load_model(self.net, _reg_state_dict), rank0_only=False)
            if self.config.ema.enabled:
                log.critical("load ema model in non-strict mode")
                log.critical(non_strict_load_model(self.net_ema, _ema_state_dict), rank0_only=False)
            if self.net_low is not None:
                log.critical("load low model in non-strict mode")
                log.critical(non_strict_load_model(self.net_low, _low_state_dict), rank0_only=False)
            if self.net_fake_score:
                log.critical("load fake score model in non-strict mode")
                log.critical(non_strict_load_model(self.net_fake_score, _fake_score_state_dict), rank0_only=False)
            if self.net_fake_score_low:
                log.critical("load low fake score model in non-strict mode")
                log.critical(non_strict_load_model(self.net_fake_score_low, _fake_score_low_state_dict), rank0_only=False)

    # ------------------ public methods ------------------
    def ema_beta(self, iteration: int) -> float:
        """
        Calculate the beta value for EMA update.
        weights = weights * beta + (1 - beta) * new_weights

        Args:
            iteration (int): Current iteration number.

        Returns:
            float: The calculated beta value.
        """
        iteration = iteration + self.config.ema.iteration_shift
        if iteration < 1:
            return 0.0
        return (1 - 1 / (iteration + 1)) ** (self.ema_exp_coefficient + 1)

    def model_param_stats(self) -> Dict[str, int]:
        return {"total_learnable_param_num": self._param_count}

    def is_image_batch(self, data_batch: dict[str, Tensor]) -> bool:
        return False

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.encode(state) * self.config.sigma_data

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.decode(latent / self.config.sigma_data)

    @torch.no_grad()
    def _build_i2v_y(self, latent_state: torch.Tensor):
        """Build the I2V first-frame condition y_B_C_T_H_W = cat([mask(4ch), first-frame latent(16ch)], dim=1).

        Scheme A (strictly aligned with inference/finetuning): decode the whole latent back to pixels and take the first frame, then
        re-encode through the VAE as [first frame, zeros×(F-1)], avoiding a train/infer distribution mismatch.
        Kept consistent with the implementation in t2v_model_finetune.py.
        Returns: (y [B,20,T,H,W], first_frame [B,3,H,W] range [-1,1] used for CLIP)
        """
        B, C_lat, T_lat, H_lat, W_lat = latent_state.shape
        del C_lat
        pixels = self.decode(latent_state)                       # [B, 3, F, H, W]
        F_pix, H_pix, W_pix = pixels.shape[2], pixels.shape[3], pixels.shape[4]
        first_frame = pixels[:, :, 0:1]                          # [B, 3, 1, H, W]
        zeros = torch.zeros(B, 3, F_pix - 1, H_pix, W_pix, device=pixels.device, dtype=pixels.dtype)
        frames_to_encode = torch.cat([first_frame, zeros], dim=2)
        encoded_latents = self.encode(frames_to_encode)          # [B, 16, T_lat, H_lat, W_lat]
        msk = torch.zeros(B, 4, T_lat, H_lat, W_lat, device=latent_state.device, dtype=encoded_latents.dtype)
        msk[:, :, 0, :, :] = 1.0
        y = torch.cat([msk, encoded_latents.to(msk.dtype)], dim=1)  # [B, 20, T_lat, H_lat, W_lat]
        # y will be concatenated inside the net with x (already cast to bf16), so the dtype must match
        return y.to(**self.tensor_kwargs), first_frame[:, :, 0]   # first_frame: [B, 3, H, W]

    @staticmethod
    def _attach_i2v(condition, y: torch.Tensor, frame_cond: torch.Tensor = None):
        """Attach the I2V condition (y + optional CLIP features) onto the frozen dataclass condition, returning a new instance."""
        kwargs = condition.to_dict(skip_underscore=False)
        kwargs["y_B_C_T_H_W"] = y
        if frame_cond is not None:
            kwargs["frame_cond_crossattn_emb_B_L_D"] = frame_cond
        return type(condition)(**kwargs)

    def get_num_video_latent_frames(self) -> int:
        return self.config.state_t

    @property
    def text_encoder_class(self) -> str:
        return self.config.text_encoder_class

    @contextmanager
    def ema_scope(self, context=None, is_cpu=False):
        if self.config.ema.enabled:
            # https://github.com/pytorch/pytorch/issues/144289
            for module in self.net.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()
            self.net_ema_worker.cache(self.net.parameters(), is_cpu=is_cpu)
            self.net_ema_worker.copy_to(src_model=self.net_ema, tgt_model=self.net)
            if context is not None:
                log.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.config.ema.enabled:
                for module in self.net.modules():
                    if isinstance(module, FSDPModule):
                        module.reshard()
                self.net_ema_worker.restore(self.net.parameters())
                if context is not None:
                    log.info(f"{context}: Restored training weights")

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
        iteration: int = 0,
    ):
        if not self.config.grad_clip:
            max_norm = 1e12
        if self.config.joint_wan22_t2v:
            key = self._joint_optimizer_key(iteration)
            params = {
                "net": self.net.parameters(),
                "net_low": self.net_low.parameters(),
                "fake_score": self.net_fake_score.parameters(),
                "fake_score_low": self.net_fake_score_low.parameters(),
            }[key]
            return clip_grad_norm_(
                params,
                max_norm=max_norm,
                norm_type=norm_type,
                error_if_nonfinite=error_if_nonfinite,
                foreach=foreach,
            ).cpu()
        if self.is_student_phase(iteration):
            return clip_grad_norm_(
                self.net.parameters(),
                max_norm=max_norm,
                norm_type=norm_type,
                error_if_nonfinite=error_if_nonfinite,
                foreach=foreach,
            ).cpu()
        if self.net_fake_score:
            clip_grad_norm_(
                self.net_fake_score.parameters(),
                max_norm=max_norm,
                norm_type=norm_type,
                error_if_nonfinite=error_if_nonfinite,
                foreach=foreach,
            )
        return None
