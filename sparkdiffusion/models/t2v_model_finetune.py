# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import collections
import math
from contextlib import contextmanager
from typing import Any, Dict, List, Mapping, Optional, Tuple

import attrs
import torch
import torch._dynamo
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
import torch.distributed.checkpoint as dcp

from imaginaire.config import ObjectStoreConfig
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict
from imaginaire.lazy_config import instantiate as lazy_instantiate
from imaginaire.model import ImaginaireModel
from imaginaire.utils import log, misc
from imaginaire.utils.easy_io import easy_io

from sparkdiffusion.conditioner import DataType, TextCondition
from sparkdiffusion.utils.fused_adam_dtensor import FusedAdam
from sparkdiffusion.utils.optim_instantiate_dtensor import get_base_scheduler
from sparkdiffusion.utils.model_utils import _load_safetensors_dir, _detect_ckpt_format, load_state_dict as _load_pth
from sparkdiffusion.utils.context_parallel import broadcast
from sparkdiffusion.utils.dtensor_helper import broadcast_dtensor_model_states
from sparkdiffusion.utils.fsdp_helper import hsdp_device_mesh
from sparkdiffusion.utils.denoiser_scaling import RectifiedFlow_TrigFlowWrapper
from sparkdiffusion.utils.timestep_utils import LogNormal, rf_to_trig_time, shift_rf_time, trig_to_rf_time
from sparkdiffusion.networks.wan_rola_attention import RoLaTrainingContext
from sparkdiffusion.configs.defaults.ema import EMAConfig

IS_PREPROCESSED_KEY = "is_preprocessed"
IS_PROCESSED_KEY = "is_processed"

# Patterns that define "sparse" parameters for stage1 training
DEFAULT_SPARSE_PATTERNS = [
    "proj_q", "proj_k", "gate_proj", "gate_bias",
    "pe_scale", "W_proj", "W_delta", "gate_anchor",
]


@attrs.define(slots=False)
class T2VFinetuneConfig:
    tokenizer: LazyDict = None
    conditioner: LazyDict = None
    net: LazyDict = None
    net_high: LazyDict = None
    net_low: LazyDict = None

    pretrained_ckpt: str = ""
    pretrained_ckpt_high: str = ""
    pretrained_ckpt_low: str = ""
    joint_dual_expert: bool = False
    rf_split_t: float = 0.875

    # Placeholders for registry compat: the global registry_distill defaults inject
    # these distill-only groups (optimizer_fake_score / ema / net_teacher /
    # net_fake_score). Finetune ignores them, but the fields must exist or Hydra
    # composition raises ConfigKeyError.
    optimizer_fake_score: LazyDict = None
    ema: EMAConfig = EMAConfig()
    net_teacher: LazyDict = None
    net_fake_score: LazyDict = None
    fsdp_shard_size: int = 1
    sigma_data: float = 1.0
    precision: str = "bfloat16"
    input_data_key: str = "videos"
    input_latent_key: str = "latents"
    input_caption_key: str = "prompts"

    state_ch: int = 16
    state_t: int = 21
    resolution: str = "480p"
    rectified_flow_t_scaling_factor: float = 1000.0

    # I2V (image-to-video): when True, build the condition y_B_C_T_H_W from the first frame (first-frame latent + mask)
    # and attach it to the condition. Requires net=wan2pt2_A14B_i2v_rola(in_dim=36, model_type=i2v).
    is_i2v: bool = False
    # Wan 2.1 I2V uses a CLIP image encoder (2.2 I2V does not need it); requires is_i2v=True
    clip_encoder_path: str = ""
    # Wan2.1 I2V only: directory path of the CLIP image_encoder. When non-empty, extract first-frame CLIP features online
    # and attach them to the condition as frame_cond_crossattn_emb_B_L_D (2.2 I2V has no CLIP, leave empty).
    i2v_clip_encoder_path: str = ""

    text_encoder_class: str = "umT5"
    text_encoder_path: str = ""
    tokenizer_path: str = ""  # local directory of the umt5 tokenizer (used by the sampling-visualization callback, avoids downloading from HF)

    # Timestep sampling
    p_mean: float = -0.8
    p_std: float = 1.6
    rf_t_min: float = 0.0
    rf_t_max: float = 1.0
    # lognormal_raw_rf: legacy truncated LogNormal in raw RF domain, then flow_shift.
    # uniform_raw_rf: uniform in raw RF domain, then flow_shift.
    # uniform_sigma: uniform in final shifted RF/sigma domain.
    time_sampling: str = "lognormal_raw_rf"
    flow_shift: float = 0.0    # 0=no shift. Set 12.0 for Wan2.2 T2V, 5.0 for Wan2.2 I2V, 3.0 for Wan2.1 I2V.

    # Two-stage training
    stage1_steps: int = 0
    stage1_mse_weight: float = 1.0
    sparse_patterns: List[str] = attrs.Factory(lambda: list(DEFAULT_SPARSE_PATTERNS))
    sparse_lr_scale: float = 1.0

    # Grouped learning rates for RoLa parameters (< 0 = use backbone LR)
    lr_proj: float = -1.0            # proj_q, proj_k
    lr_gate_proj: float = -1.0       # gate_proj linear
    lr_gate_bias: float = -1.0       # gate_bias parameter

    # Grouped weight decay for RoLa parameters (< 0 = use backbone wd)
    wd_proj: float = -1.0
    wd_gate_proj: float = -1.0
    wd_gate_bias: float = -1.0

    # Loss
    loss_scale: float = 1.0
    grad_clip: bool = False

    neg_embed_path: str = ""

    checkpoint: ObjectStoreConfig = ObjectStoreConfig()


class T2VFinetuneModel(ImaginaireModel):

    def __init__(self, config: T2VFinetuneConfig):
        super().__init__()
        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}

        # Timestep sampler (LogNormal in RF domain)
        self.p_G = LogNormal(p_mean=config.p_mean, p_std=config.p_std)

        # EDM-style scaling wrapper
        self.scaling = RectifiedFlow_TrigFlowWrapper(config.sigma_data, config.rectified_flow_t_scaling_factor)

        # Negative embedding for CFG-free training (optional)
        if config.neg_embed_path:
            self.neg_embed = easy_io.load(config.neg_embed_path)
        else:
            self.neg_embed = None

        # Tokenizer (VAE)
        with misc.timer("FinetuneModel: set_up_tokenizer"):
            self.tokenizer = lazy_instantiate(config.tokenizer)
            assert self.tokenizer.latent_ch == config.state_ch

        # FSDP mesh
        if config.fsdp_shard_size > 1:
            self.fsdp_device_mesh = hsdp_device_mesh(sharding_group_size=config.fsdp_shard_size)
        else:
            self.fsdp_device_mesh = None

        # Build network and load pretrained weights
        self.net_high = None
        self.net_low = None
        self.is_dual_expert = self._is_dual_expert_enabled()
        self.set_up_model()

        # Data parallel info
        if parallel_state is not None and parallel_state.is_initialized():
            self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_size = 1

        # Stage tracking
        self._current_stage = 1 if config.stage1_steps > 0 else 2

        # Wan2.1 I2V: load the CLIP image encoder on demand (extracts first-frame features online)
        self._clip_encoder = None
        if config.is_i2v and config.i2v_clip_encoder_path:
            from sparkdiffusion.utils.clip_image_encoder import WanCLIPImageEncoder
            self._clip_encoder = WanCLIPImageEncoder(
                config.i2v_clip_encoder_path, dtype=torch.float16, device="cuda"
            )

        torch._dynamo.config.suppress_errors = True

    # ======================== Model Setup ========================

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

    def _is_dual_expert_enabled(self) -> bool:
        if self.config.joint_dual_expert:
            return True
        return self.config.net_high is not None and self.config.net_low is not None

    def _net_spec_items(self):
        if self.is_dual_expert:
            return [
                ("high", self.config.net_high or self.config.net, self.config.pretrained_ckpt_high or self.config.pretrained_ckpt),
                ("low", self.config.net_low or self.config.net, self.config.pretrained_ckpt_low or self.config.pretrained_ckpt),
            ]
        return [("net", self.config.net, self.config.pretrained_ckpt)]

    def _net_items(self):
        if self.is_dual_expert:
            return [("high", self.net_high), ("low", self.net_low)]
        return [("net", self.net)]

    @staticmethod
    def _maybe_disable_checkpointing_for_stage1_mse_on_net(net_dict: LazyDict):
        try:
            net_sac = net_dict.get("sac_config", None)
            if net_sac is not None and net_sac.get("mode", None) != "none":
                log.warning(
                    "Stage-1 RoLa MSE is active → forcing net.sac_config.mode='none' "
                    "(activation checkpointing is incompatible with the MSE side-effect)."
                )
                net_sac["mode"] = "none"
        except Exception as e:
            log.warning(f"Could not auto-disable checkpointing for Stage-1 MSE: {e}")

    @misc.timer("FinetuneModel: set_up_model")
    def set_up_model(self):
        config = self.config

        self.conditioner = lazy_instantiate(config.conditioner)
        assert sum(p.numel() for p in self.conditioner.parameters() if p.requires_grad) == 0

        # Stage-1 RoLa MSE alignment records a per-layer loss as a global side
        # effect inside the block forward. This is incompatible with activation
        # checkpointing (the block forward is replayed during backward, and the
        # duplicated dense-attn MSE subgraph corrupts SAC's saved-tensor
        # bookkeeping → CheckpointError). Force-disable checkpointing when Stage-1
        # MSE is active so the run just works.
        stage1_mse_active = config.stage1_steps > 0 and config.stage1_mse_weight > 0
        if stage1_mse_active:
            for _, net_dict, _ in self._net_spec_items():
                self._maybe_disable_checkpointing_for_stage1_mse_on_net(net_dict)

        built_nets = {}
        for name, net_dict, pretrained_ckpt in self._net_spec_items():
            if net_dict is None:
                raise ValueError(f"Missing network config for finetune branch '{name}'")
            built_nets[name] = self.build_net(net_dict)
            if pretrained_ckpt:
                self._load_pretrained_into_net(built_nets[name], pretrained_ckpt)

        if self.is_dual_expert:
            self.net_high = built_nets["high"]
            self.net_low = built_nets["low"]
            # Keep self.net as a compatibility alias for callbacks/hooks that
            # still expect a primary network to exist.
            self.net = self.net_high
        else:
            self.net = built_nets["net"]

        # Apply stage1 freeze if needed
        if config.stage1_steps > 0:
            self._freeze_except_sparse()

        self._param_count = sum(
            p.numel() for _, net in self._net_items() for p in net.parameters() if p.requires_grad
        )
        log.info(f"FinetuneModel: trainable params = {self._param_count:,}")

    def _load_pretrained_into_net(self, net, ckpt_path: str):
        fmt = _detect_ckpt_format(ckpt_path)
        if fmt == "safetensors":
            self._load_safetensors_to_net(net, ckpt_path)
        elif fmt == "pth":
            self._load_pth_to_net(net, ckpt_path)
        elif fmt == "dcp":
            self._load_dcp_to_net(net, ckpt_path)
        else:
            raise ValueError(f"Unknown checkpoint format '{fmt}' at {ckpt_path}")

    def _assert_backbone_matched(self, net, incoming_keys, path: str, fmt: str):
        """Fail loudly when a pretrained checkpoint matches (almost) no backbone weights.

        RoLa sparse params are expected to be missing (they are newly added and
        stay at random init), but the dense backbone must load. If essentially no
        backbone tensor matched, the path or format is wrong and training would
        silently start from random weights, so we raise instead.
        """
        backbone = [
            n.replace("._checkpoint_wrapped_module.", ".")
            for n, _ in net.named_parameters()
            if not any(p in n for p in self.config.sparse_patterns)
        ]
        if not backbone:
            return
        incoming = {k.replace("._checkpoint_wrapped_module.", ".") for k in incoming_keys}
        matched = sum(1 for n in backbone if n in incoming)
        ratio = matched / len(backbone)
        if matched == 0:
            raise RuntimeError(
                f"Checkpoint at '{path}' (detected format: {fmt}) matched 0 / {len(backbone)} "
                f"backbone parameters. The path or checkpoint format is almost certainly wrong, "
                f"and finetuning would start from random weights.\n"
                f"Expected paths by format:\n"
                f"  - Native Wan / single-file: a directory containing "
                f"'diffusion_pytorch_model.safetensors' (+ optional '.index.json' shards), "
                f"or the .safetensors file itself.\n"
                f"  - Diffusers: the model's 'transformer/' subdirectory.\n"
                f"  - .pth / .pt: a Wan-official or SparkDiffusion training checkpoint file.\n"
                f"  - DCP: a distributed-checkpoint directory containing '.distcp' shards."
            )
        if ratio < 0.5:
            log.warning(
                f"Only {matched}/{len(backbone)} ({ratio:.0%}) backbone parameters matched "
                f"the checkpoint at '{path}' (format: {fmt}). Verify the path and format."
            )

    def _load_safetensors_to_net(self, net, path: str):
        raw_sd = _load_safetensors_dir(path)
        converted: dict = {}
        for k, v in raw_sd.items():
            if k == "patch_embedding.weight" and v.ndim == 5:
                v = v.reshape(v.shape[0], -1)
            converted[k] = v

        self._assert_backbone_matched(net, converted.keys(), path, "safetensors")

        if self.fsdp_device_mesh is not None:
            set_model_state_dict(
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
                          if any(p in k for p in self.config.sparse_patterns)]
            if new_params:
                log.info(f"RoLa new params kept at random init ({len(new_params)})")
            log.info(f"Loaded safetensors from {path} → {len(final_sd) - len(result.unexpected_keys)} matched")

    def _load_dcp_to_net(self, net, ckpt_path: str):
        storage_reader = FileSystemReader(ckpt_path)
        _state_dict = get_model_state_dict(net)

        metadata = storage_reader.read_metadata()
        checkpoint_keys = metadata.state_dict_metadata.keys()

        prefix = "net_ema" if any(k.startswith("net_ema.") for k in checkpoint_keys) else "net"

        _new_state_dict = collections.OrderedDict()
        for k in _state_dict.keys():
            _new_state_dict[f"{prefix}.{k}"] = _state_dict[k]
        dcp.load(_new_state_dict, storage_reader=storage_reader, planner=DefaultLoadPlanner(allow_partial_load=True))
        for k in _state_dict.keys():
            _state_dict[k] = _new_state_dict[f"{prefix}.{k}"]

        log.info(set_model_state_dict(net, _state_dict, options=StateDictOptions(strict=False)))
        del _state_dict, _new_state_dict

    def _load_pth_to_net(self, net, path: str):
        """Load a .pth checkpoint (Wan official format or our training output)."""
        sd = _load_pth(path)
        sd_clean = {}
        for k, v in sd.items():
            if k.startswith("net_ema."):
                sd_clean[k[len("net_ema."):]] = v
            elif k.startswith("net."):
                sd_clean[k[len("net."):]] = v
            else:
                sd_clean[k] = v

        self._assert_backbone_matched(net, sd_clean.keys(), path, "pth")

        if self.fsdp_device_mesh is not None:
            set_model_state_dict(
                net, sd_clean,
                options=StateDictOptions(strict=False, full_state_dict=True),
            )
        else:
            result = net.load_state_dict(sd_clean, strict=False, assign=True)
            new_params = [k for k in result.missing_keys
                          if any(p in k for p in self.config.sparse_patterns)]
            if new_params:
                log.info(f"RoLa new params kept at random init ({len(new_params)})")
        log.info(f"Loaded .pth checkpoint from {path}")

    # ======================== Freeze Logic ========================

    def _resolve_sparse_patterns(self, net_dict: LazyDict | None) -> list[str]:
        """Stage1: freeze all params except those matching sparse_patterns."""
        patterns = list(self.config.sparse_patterns)
        # Pluggable: if the net uses a registry custom sparse attention (attn_variant), automatically merge in the
        # trainable parameter names declared by that variant, so user-defined parameters are not frozen. When there is no attn_variant (default), patterns stay unchanged.
        _av = None
        try:
            _av = net_dict.get("attn_variant", None) if net_dict is not None else None
        except Exception:
            _av = getattr(net_dict, "attn_variant", None)
        if _av:
            from sparkdiffusion.networks.sparse_attn_registry import get_sparse_param_names
            extra = [p for p in get_sparse_param_names(_av) if p not in patterns]
            if extra:
                patterns = patterns + extra
                log.info(f"Stage1 freeze: attn_variant='{_av}' appended sparse-param allowlist {extra}")
        return patterns

    def _freeze_except_sparse_net(self, net, net_dict: LazyDict | None, branch_name: str):
        patterns = self._resolve_sparse_patterns(net_dict)
        for name, param in net.named_parameters():
            if any(p in name for p in patterns):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        total = sum(p.numel() for p in net.parameters())
        log.info(
            f"Stage1 freeze [{branch_name}]: {trainable:,} / {total:,} params trainable "
            f"({100 * trainable / total:.2f}%)"
        )

    def _freeze_except_sparse(self):
        for branch_name, net in self._net_items():
            if self.is_dual_expert:
                net_dict = getattr(self.config, f"net_{branch_name}", None) or self.config.net
            else:
                net_dict = self.config.net
            self._freeze_except_sparse_net(net, net_dict, branch_name)

    @staticmethod
    def _unfreeze_all_net(net):
        for param in net.parameters():
            param.requires_grad_(True)

    def _unfreeze_all(self):
        """Stage2: unfreeze all parameters."""
        for _, net in self._net_items():
            self._unfreeze_all_net(net)
        self._current_stage = 2
        self._param_count = sum(
            p.numel() for _, net in self._net_items() for p in net.parameters() if p.requires_grad
        )
        log.info(f"Stage2: unfroze all params, trainable = {self._param_count:,}")

    def _maybe_transition_stage(self, iteration: int):
        """Check if we need to transition from stage1 to stage2."""
        if self._current_stage == 1 and iteration >= self.config.stage1_steps:
            self._unfreeze_all()

    # ======================== Training Hooks ========================

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        if hasattr(self.tokenizer, "reset_dtype"):
            self.tokenizer.reset_dtype()
        if self.is_dual_expert:
            self.net_high = self.net_high.to(memory_format=memory_format, **self.tensor_kwargs)
            self.net_low = self.net_low.to(memory_format=memory_format, **self.tensor_kwargs)
            self.net = self.net_high
        else:
            self.net = self.net.to(memory_format=memory_format, **self.tensor_kwargs)

    def on_before_zero_grad(self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int) -> None:
        pass

    # ======================== Optimizer / Scheduler ========================

    def _build_param_groups(self, net, lr_base: float, wd_base: float):
        lr_proj = self.config.lr_proj if self.config.lr_proj >= 0 else lr_base
        lr_gate_proj = self.config.lr_gate_proj if self.config.lr_gate_proj >= 0 else lr_base
        lr_gate_bias = self.config.lr_gate_bias if self.config.lr_gate_bias >= 0 else lr_base

        wd_proj = self.config.wd_proj if self.config.wd_proj >= 0 else wd_base
        wd_gate_proj = self.config.wd_gate_proj if self.config.wd_gate_proj >= 0 else wd_base
        wd_gate_bias = self.config.wd_gate_bias if self.config.wd_gate_bias >= 0 else wd_base

        backbone_params = []
        proj_params = []
        gate_proj_params = []
        gate_bias_params = []

        for name, param in net.named_parameters():
            if not param.requires_grad:
                continue
            if "proj_q" in name or "proj_k" in name:
                proj_params.append(param)
            elif "gate_proj" in name:
                gate_proj_params.append(param)
            elif "gate_bias" in name:
                gate_bias_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": lr_base, "weight_decay": wd_base})
        if proj_params:
            param_groups.append({"params": proj_params, "lr": lr_proj, "weight_decay": wd_proj})
        if gate_proj_params:
            param_groups.append({"params": gate_proj_params, "lr": lr_gate_proj, "weight_decay": wd_gate_proj})
        if gate_bias_params:
            param_groups.append({"params": gate_bias_params, "lr": lr_gate_bias, "weight_decay": wd_gate_bias})

        return {
            "backbone": (len(backbone_params), lr_base, wd_base),
            "proj": (len(proj_params), lr_proj, wd_proj),
            "gate_proj": (len(gate_proj_params), lr_gate_proj, wd_gate_proj),
            "gate_bias": (len(gate_bias_params), lr_gate_bias, wd_gate_bias),
        }, param_groups

    def init_optimizer_scheduler(self, optimizer_config: LazyDict, scheduler_config: LazyDict):
        lr_base = optimizer_config.get("lr", 5e-5)
        wd_base = optimizer_config.get("weight_decay", 0.01)

        optim_type = optimizer_config.get("optim_type", "fusedadam")
        from omegaconf import OmegaConf
        # ListConfig values (e.g. betas) pickle their parent lazy config, which
        # holds module objects and crashes checkpoint saving (dcp.save on
        # param_groups.*.betas → TypeError: cannot pickle 'module' object).
        # Convert config containers to plain Python types before passing to optimizer.
        opt_kwargs = {
            k: (OmegaConf.to_object(v) if OmegaConf.is_config(v) else v)
            for k, v in optimizer_config.items()
            if k not in ("lr", "weight_decay", "optim_type", "_target_", "model")
        }

        if optim_type == "adamw":
            opt_cls = torch.optim.AdamW
        elif optim_type == "fusedadam":
            opt_cls = FusedAdam
        else:
            raise ValueError(f"Unknown optimizer type: {optim_type}")

        self.optimizer_dict = {}
        self.scheduler_dict = {}
        for branch_name, net in self._net_items():
            stats, param_groups = self._build_param_groups(net, lr_base, wd_base)
            log.info(
                f"Grouped optimizer [{branch_name}]: backbone lr={stats['backbone'][1]:.2e} "
                f"wd={stats['backbone'][2]:.2e} ({stats['backbone'][0]} params)"
            )
            if stats["proj"][0]:
                log.info(f"  proj       lr={stats['proj'][1]:.2e} wd={stats['proj'][2]:.2e} ({stats['proj'][0]} params)")
            if stats["gate_proj"][0]:
                log.info(
                    f"  gate_proj  lr={stats['gate_proj'][1]:.2e} wd={stats['gate_proj'][2]:.2e} "
                    f"({stats['gate_proj'][0]} params)"
                )
            if stats["gate_bias"][0]:
                log.info(
                    f"  gate_bias  lr={stats['gate_bias'][1]:.2e} wd={stats['gate_bias'][2]:.2e} "
                    f"({stats['gate_bias'][0]} params)"
                )

            optimizer_key = "net" if branch_name == "net" else f"net_{branch_name}"
            optimizer = opt_cls(param_groups, **opt_kwargs)
            scheduler = get_base_scheduler(optimizer, self, scheduler_config)
            self.optimizer_dict[optimizer_key] = optimizer
            self.scheduler_dict[optimizer_key] = scheduler

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        self._maybe_transition_stage(iteration)
        return [self.optimizer_dict[k] for k in sorted(self.optimizer_dict.keys())]

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler.LRScheduler]:
        return [self.scheduler_dict[k] for k in sorted(self.scheduler_dict.keys())]

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
        iteration: int = 0,
    ):
        """Clip gradients of self.net and return the total grad norm (on CPU).

        Called by the GradClip callback. When config.grad_clip is False we use a
        huge max_norm so clipping is effectively a no-op, but we still compute and
        return the real grad norm for logging.
        """
        from sparkdiffusion.utils.torch_future import clip_grad_norm_ as _clip
        if not self.config.grad_clip:
            max_norm = 1e12
        parameters = [
            param for _, net in self._net_items() for param in net.parameters()
            if param.requires_grad
        ]
        return _clip(
            parameters,
            max_norm=max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        ).cpu()

    # ======================== Timestep Sampling ========================

    def _sample_time(self, batch_size: int) -> torch.Tensor:
        """Sample TrigFlow time for training, truncated to [rf_t_min, rf_t_max]."""
        rf_min = self.config.rf_t_min
        rf_max = self.config.rf_t_max
        sampling = self.config.time_sampling
        if not (0.0 <= rf_min <= rf_max <= 1.0):
            raise ValueError(f"Invalid RF timestep range [{rf_min}, {rf_max}]")

        if sampling == "lognormal_raw_rf":
            if rf_min == 0.0 and rf_max == 1.0:
                rf_t = self.p_G(shape=batch_size, device="cuda", dtype=torch.float64)
                rf_t = rf_t.clamp(min=0.0, max=1.0)
            else:
                # Rejection-free truncated sampling via inverse CDF on uniform in [CDF(min), CDF(max)]
                import torch.distributions as D
                normal = D.Normal(self.p_G.p_mean, self.p_G.p_std)
                eps_val = torch.finfo(torch.float64).eps
                log_min = torch.tensor(rf_min / (1.0 - rf_min + eps_val)).log() if rf_min > 0 else torch.tensor(-40.0)
                log_max = torch.tensor(rf_max / (1.0 - rf_max + eps_val)).log() if rf_max < 1.0 else torch.tensor(40.0)
                cdf_lo = normal.cdf(log_min.double())
                cdf_hi = normal.cdf(log_max.double())
                u = torch.rand(batch_size, device="cuda", dtype=torch.float64)
                u = cdf_lo + u * (cdf_hi - cdf_lo)
                u = u.clamp(min=eps_val, max=1.0 - eps_val)
                log_sigma = normal.icdf(u)
                sigma = log_sigma.exp()
                rf_t = sigma / (sigma + 1.0)
            rf_t = shift_rf_time(rf_t, self.config.flow_shift)
        elif sampling == "uniform_raw_rf":
            u = torch.rand(batch_size, device="cuda", dtype=torch.float64)
            rf_t = rf_min + u * (rf_max - rf_min)
            rf_t = shift_rf_time(rf_t, self.config.flow_shift)
        elif sampling == "uniform_sigma":
            lo = torch.tensor(rf_min, device="cuda", dtype=torch.float64)
            hi = torch.tensor(rf_max, device="cuda", dtype=torch.float64)
            sigma_min = shift_rf_time(lo, self.config.flow_shift)
            sigma_max = shift_rf_time(hi, self.config.flow_shift)
            u = torch.rand(batch_size, device="cuda", dtype=torch.float64)
            rf_t = sigma_min + u * (sigma_max - sigma_min)
        else:
            raise ValueError(
                "Unknown time_sampling "
                f"{sampling!r}; expected lognormal_raw_rf, uniform_raw_rf, or uniform_sigma"
            )
        return rf_to_trig_time(rf_t)

    # ======================== Forward / Denoise ========================

    def denoise(
        self,
        xt_B_C_T_H_W: torch.Tensor,
        time: torch.Tensor,
        condition: TextCondition,
    ) -> torch.Tensor:
        """
        Forward pass through the network with EDM scaling.
        Returns x0 prediction.
        """
        if not self.is_dual_expert:
            return self.denoise_with_net(self.net, xt_B_C_T_H_W, time, condition)

        time_route = time[:, 0] if time.ndim == 2 else time
        rf_t = trig_to_rf_time(time_route)
        mask_high = rf_t > self.config.rf_split_t
        mask_low = ~mask_high
        x0_pred = torch.zeros_like(xt_B_C_T_H_W)

        if mask_high.any():
            x0_pred[mask_high] = self.denoise_with_net(
                self.net_high,
                xt_B_C_T_H_W[mask_high],
                time[mask_high],
                self._slice_condition(condition, mask_high),
            )
        if mask_low.any():
            x0_pred[mask_low] = self.denoise_with_net(
                self.net_low,
                xt_B_C_T_H_W[mask_low],
                time[mask_low],
                self._slice_condition(condition, mask_low),
            )
        return x0_pred

    def denoise_with_net(
        self,
        net,
        xt_B_C_T_H_W: torch.Tensor,
        time: torch.Tensor,
        condition: TextCondition,
    ) -> torch.Tensor:
        if time.ndim == 1:
            time_B_T = repeat(time, "b -> b 1")
        elif time.ndim == 2:
            time_B_T = time
        else:
            raise ValueError(f"time shape {time.shape} is not supported")
        time_B_1_T_1_1 = rearrange(time_B_T, "b t -> b 1 t 1 1")

        c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = self.scaling(trigflow_t=time_B_1_T_1_1)

        net_output_B_C_T_H_W = net(
            x_B_C_T_H_W=(xt_B_C_T_H_W * c_in_B_1_T_1_1).to(**self.tensor_kwargs),
            timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(**self.tensor_kwargs),
            **condition.to_dict(),
        ).float()

        x0_pred_B_C_T_H_W = c_skip_B_1_T_1_1 * xt_B_C_T_H_W + c_out_B_1_T_1_1 * net_output_B_C_T_H_W
        return x0_pred_B_C_T_H_W

    @staticmethod
    def _slice_condition(condition: TextCondition, index) -> TextCondition:
        kwargs = condition.to_dict(skip_underscore=False)
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                kwargs[key] = value[index]
        return type(condition)(**kwargs)

    def _compute_flow_matching_loss(
        self,
        net,
        x0_B_C_T_H_W: torch.Tensor,
        xt: torch.Tensor,
        t_trig: torch.Tensor,
        condition: TextCondition,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_B_1_T_1_1 = rearrange(repeat(t_trig, "b -> b 1"), "b t -> b 1 t 1 1")
        cos_t = torch.cos(t_B_1_T_1_1)
        sin_t = torch.sin(t_B_1_T_1_1)
        eps = (xt - cos_t * x0_B_C_T_H_W) / sin_t

        enable_rola_mse = (self._current_stage == 1 and self.config.stage1_mse_weight > 0)
        if enable_rola_mse:
            T_lat = x0_B_C_T_H_W.shape[2]
            H_lat = x0_B_C_T_H_W.shape[3]
            W_lat = x0_B_C_T_H_W.shape[4]
            RoLaTrainingContext.set(
                enable_stage1_mse=True,
                token_grid_shape=(T_lat, H_lat, W_lat),
            )
            RoLaTrainingContext.clear_mse_losses()

        x0_pred = self.denoise_with_net(net, xt, t_trig, condition)
        v_pred = (cos_t * xt - x0_pred) / sin_t
        v_target = cos_t * eps - sin_t * x0_B_C_T_H_W
        loss_per_sample = ((v_pred - v_target) ** 2).mean(dim=[1, 2, 3, 4])
        loss = loss_per_sample * self.config.loss_scale

        if enable_rola_mse:
            rola_loss = RoLaTrainingContext.pop_mse_loss_mean()
            if rola_loss is not None:
                loss = loss + rola_loss * self.config.stage1_mse_weight
            RoLaTrainingContext.set(enable_stage1_mse=False)

        return loss_per_sample, loss

    # ======================== Training Step ========================

    def training_step_closures(self, data_batch, iteration: int):
        # Data from .tar is already VAE-encoded; skip decode+re-encode
        # BUT still need state_t slicing before setting IS_PROCESSED_KEY,
        # otherwise _normalize_latent_inplace is skipped.
        self._normalize_latent_inplace(data_batch)
        data_batch[IS_PROCESSED_KEY] = True
        data_batch[IS_PREPROCESSED_KEY] = True
        data_batch[self.config.input_data_key] = data_batch[self.config.input_latent_key]
        _, x0_B_C_T_H_W, condition, uncondition = self.get_data_and_condition(data_batch)

        def flow_matching_closure():
            B = x0_B_C_T_H_W.shape[0]
            t_trig = self._sample_time(B)  # [B] in TrigFlow domain

            # Noising: x_t = cos(t)*x0 + sin(t)*eps
            t_B_1_T_1_1 = rearrange(repeat(t_trig, "b -> b 1"), "b t -> b 1 t 1 1")
            eps = torch.randn_like(x0_B_C_T_H_W)
            cos_t = torch.cos(t_B_1_T_1_1)
            sin_t = torch.sin(t_B_1_T_1_1)
            xt = cos_t * x0_B_C_T_H_W + sin_t * eps

            if not self.is_dual_expert:
                loss_per_sample, loss = self._compute_flow_matching_loss(self.net, x0_B_C_T_H_W, xt, t_trig, condition)
                output_batch = {"loss_velocity": loss_per_sample.detach().mean()}
                return output_batch, loss

            rf_t = trig_to_rf_time(t_trig)
            mask_high = rf_t > self.config.rf_split_t
            mask_low = ~mask_high

            loss_chunks = []
            velocity_chunks = []
            output_batch = {}

            if mask_high.any():
                high_condition = self._slice_condition(condition, mask_high)
                high_velocity, high_loss = self._compute_flow_matching_loss(
                    self.net_high,
                    x0_B_C_T_H_W[mask_high],
                    xt[mask_high],
                    t_trig[mask_high],
                    high_condition,
                )
                loss_chunks.append(high_loss)
                velocity_chunks.append(high_velocity)
                output_batch["loss_velocity_high"] = high_velocity.detach().mean()

            if mask_low.any():
                low_condition = self._slice_condition(condition, mask_low)
                low_velocity, low_loss = self._compute_flow_matching_loss(
                    self.net_low,
                    x0_B_C_T_H_W[mask_low],
                    xt[mask_low],
                    t_trig[mask_low],
                    low_condition,
                )
                loss_chunks.append(low_loss)
                velocity_chunks.append(low_velocity)
                output_batch["loss_velocity_low"] = low_velocity.detach().mean()

            loss = torch.cat(loss_chunks, dim=0)
            loss_velocity = torch.cat(velocity_chunks, dim=0)
            output_batch["loss_velocity"] = loss_velocity.detach().mean()
            output_batch["samples_high"] = mask_high.sum().item()
            output_batch["samples_low"] = mask_low.sum().item()
            return output_batch, loss

        yield "flow_matching", flow_matching_closure, True

    # ======================== Data Processing ========================

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor]) -> Tuple:
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
        # Both share the same y (the first frame is a clean condition, independent of the positive/negative text).
        if self.config.is_i2v:
            y, first_frame = self._build_i2v_y(
                latent_state,
                first_frame_rgb=data_batch.get("first_frame_rgb"),
            )  # first_frame: [B,3,H,W] range [-1,1]
            # Wan2.1 I2V: extract first-frame CLIP features online
            frame_cond = None
            if self._clip_encoder is not None:
                frame_cond = self._clip_encoder.encode_first_frame(first_frame).to(dtype=self.tensor_kwargs["dtype"])
            condition = self._attach_i2v(condition, y, frame_cond)
            uncondition = self._attach_i2v(uncondition, y, frame_cond)
        return raw_state, latent_state, condition, uncondition

    @staticmethod
    def _attach_i2v(condition, y: torch.Tensor, frame_cond: torch.Tensor = None):
        """Attach the I2V condition (y + optional CLIP features) onto the frozen dataclass condition, returning a new instance."""
        kwargs = condition.to_dict(skip_underscore=False)
        kwargs["y_B_C_T_H_W"] = y
        if frame_cond is not None:
            kwargs["frame_cond_crossattn_emb_B_L_D"] = frame_cond
        return type(condition)(**kwargs)

    def _normalize_video_inplace(self, data_batch: dict[str, Tensor]) -> None:
        input_key = self.config.input_data_key
        if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
            return
        if data_batch[input_key].dtype == torch.uint8:
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
        data_batch[self.config.input_latent_key] = latents[:, :, :self.config.state_t, :, :]

    # ======================== Encode / Decode ========================

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.encode(state) * self.config.sigma_data

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.decode(latent / self.config.sigma_data)

    @torch.no_grad()
    def _build_i2v_y(self, latent_state: torch.Tensor, first_frame_rgb: torch.Tensor | None = None):
        """Build the I2V first-frame condition y_B_C_T_H_W = cat([mask(4ch), first-frame latent(16ch)], dim=1).

        Prefer the real RGB first frame stored in the dataset; when an old tar lacks that field, fall back to
        the reconstructed first frame decoded from the latent. Then, following the inference procedure [first frame, zeros×(F-1)],
        re-encode through the VAE; see sparkdiffusion/inference/wan2pt2_i2v_diffusion_infer.py.

        Args:
            latent_state: [B, C_lat, T_lat, H_lat, W_lat] the encoded video latent (includes sigma_data scaling)
            first_frame_rgb: optional [B,3,H,W], range [-1,1], the real video first frame
        Returns:
            (y, first_frame_B_C_H_W): y=[B,20,T_lat,H_lat,W_lat]; first_frame=[B,3,H,W] used for CLIP
        """
        B, C_lat, T_lat, H_lat, W_lat = latent_state.shape
        del C_lat  # only B/T_lat/H_lat/W_lat are needed
        if first_frame_rgb is not None:
            first_frame = first_frame_rgb.to(device=latent_state.device, dtype=torch.float32).clamp(-1, 1).unsqueeze(2)
            F_pix = self.tokenizer.get_pixel_num_frames(T_lat)
            H_pix, W_pix = first_frame.shape[-2], first_frame.shape[-1]
        else:
            # Old-dataset fallback: decode back to pixels and take the VAE-reconstructed first frame.
            pixels = self.decode(latent_state)                       # [B, 3, F, H, W]
            F_pix, H_pix, W_pix = pixels.shape[2], pixels.shape[3], pixels.shape[4]
            first_frame = pixels[:, :, 0:1]                          # [B, 3, 1, H, W]
        # First frame + (F-1) all-zero frames, re-encode (aligned with inference)
        zeros = torch.zeros(B, 3, F_pix - 1, H_pix, W_pix, device=latent_state.device, dtype=first_frame.dtype)
        frames_to_encode = torch.cat([first_frame, zeros], dim=2)  # [B, 3, F, H, W]
        encoded_latents = self.encode(frames_to_encode)          # [B, 16, T_lat, H_lat, W_lat]
        if encoded_latents.shape[2:] != (T_lat, H_lat, W_lat):
            raise ValueError(
                "first_frame_rgb resolution/frame count does not match latent.pt: "
                f"encoded y latent shape={tuple(encoded_latents.shape[2:])}, "
                f"target latent shape={(T_lat, H_lat, W_lat)}"
            )
        # 3. mask: only the first latent frame = 1
        msk = torch.zeros(B, 4, T_lat, H_lat, W_lat, device=latent_state.device, dtype=encoded_latents.dtype)
        msk[:, :, 0, :, :] = 1.0
        y = torch.cat([msk, encoded_latents.to(msk.dtype)], dim=1)  # [B, 20, T_lat, H_lat, W_lat]
        # y will be concatenated inside the net with x (already cast to bf16), so the dtype must match
        return y.to(**self.tensor_kwargs), first_frame[:, :, 0]   # first_frame: [B, 3, H, W]

    # ======================== Checkpointing ========================

    def model_dict(self) -> Dict[str, Any]:
        if self.is_dual_expert:
            return {"net_high": self.net_high, "net_low": self.net_low}
        return {"net": self.net}

    def state_dict(self) -> Dict[str, Any]:
        if self.is_dual_expert:
            state_dict = self.net_high.state_dict(prefix="net_high.")
            state_dict.update(self.net_low.state_dict(prefix="net_low."))
            return state_dict
        return self.net.state_dict(prefix="net.")

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        _reg_state_dict = collections.OrderedDict()
        _high_state_dict = collections.OrderedDict()
        _low_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("net."):
                _reg_state_dict[k.replace("net.", "")] = v
            elif k.startswith("net_high."):
                _high_state_dict[k.replace("net_high.", "")] = v
            elif k.startswith("net_low."):
                _low_state_dict[k.replace("net_low.", "")] = v

        if self.is_dual_expert:
            if not _high_state_dict and _reg_state_dict:
                log.warning("Dual-expert finetune loading legacy 'net.' weights into high branch")
                _high_state_dict = collections.OrderedDict(_reg_state_dict)
            if not _low_state_dict and _reg_state_dict:
                log.warning("Dual-expert finetune loading legacy 'net.' weights into low branch")
                _low_state_dict = collections.OrderedDict(_reg_state_dict)

            if strict:
                high_results = self.net_high.load_state_dict(_high_state_dict, strict=strict, assign=assign)
                low_results = self.net_low.load_state_dict(_low_state_dict, strict=strict, assign=assign)
                return _IncompatibleKeys(
                    missing_keys=high_results.missing_keys + low_results.missing_keys,
                    unexpected_keys=high_results.unexpected_keys + low_results.unexpected_keys,
                )

            def _filter_for_model(net, cur_state_dict):
                model_sd = net.state_dict()
                filtered = {}
                skipped = []
                for key, value in cur_state_dict.items():
                    if key in model_sd and model_sd[key].shape == value.shape:
                        filtered[key] = value
                    else:
                        skipped.append(key)
                if skipped:
                    log.warning(f"Skipped {len(skipped)} keys with shape mismatch")
                return filtered

            high_filtered = _filter_for_model(self.net_high, _high_state_dict)
            low_filtered = _filter_for_model(self.net_low, _low_state_dict)
            high_results = self.net_high.load_state_dict(high_filtered, strict=False, assign=assign)
            low_results = self.net_low.load_state_dict(low_filtered, strict=False, assign=assign)
            return _IncompatibleKeys(
                missing_keys=high_results.missing_keys + low_results.missing_keys,
                unexpected_keys=high_results.unexpected_keys + low_results.unexpected_keys,
            )

        if strict:
            return self.net.load_state_dict(_reg_state_dict, strict=strict, assign=assign)
        else:
            model_sd = self.net.state_dict()
            filtered = {}
            skipped = []
            for k, v in _reg_state_dict.items():
                if k in model_sd and model_sd[k].shape == v.shape:
                    filtered[k] = v
                else:
                    skipped.append(k)
            if skipped:
                log.warning(f"Skipped {len(skipped)} keys with shape mismatch")
            return self.net.load_state_dict(filtered, strict=False, assign=assign)

    # ======================== Distributed ========================

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

    # ======================== Misc ========================

    def model_param_stats(self) -> Dict[str, int]:
        return {"total_learnable_param_num": self._param_count}

    def is_image_batch(self, data_batch: dict[str, Tensor]) -> bool:
        return False

    def get_num_video_latent_frames(self) -> int:
        return self.config.state_t

    @property
    def text_encoder_class(self) -> str:
        return self.config.text_encoder_class

    @torch.no_grad()
    def forward(self, xt, t, condition: TextCondition):
        pass

    @torch.no_grad()
    def validation_step(self, data: dict[str, torch.Tensor], iteration: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        pass
