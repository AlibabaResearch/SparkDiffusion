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

from typing import Any, List

import attrs

from imaginaire import config
from imaginaire.utils.config_helper import import_all_modules_from_package
from sparkdiffusion.configs.defaults.trainer import register_trainer
from sparkdiffusion.configs.defaults.checkpoint import register_checkpoint
from sparkdiffusion.configs.defaults.ema import register_ema
from sparkdiffusion.configs.defaults.optimizer import register_optimizer, register_optimizer_fake_score
from sparkdiffusion.configs.defaults.scheduler import register_scheduler
from sparkdiffusion.configs.defaults.conditioner import register_conditioner
from sparkdiffusion.configs.defaults.callbacks import register_callbacks
from sparkdiffusion.configs.defaults.ckpt_type import register_ckpt_type
from sparkdiffusion.configs.defaults.dataloader import register_dataloader
from sparkdiffusion.configs.defaults.tokenizer import register_tokenizer
from sparkdiffusion.configs.defaults.model import register_model
from sparkdiffusion.configs.defaults.net import register_net, register_net_fake_score, register_net_teacher


@attrs.define(slots=False)
class Config(config.Config):
    # default config groups that will be used unless overwritten
    # see config groups in registry.py
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"trainer": "standard"},
            {"data_train": "dummy"},
            {"data_val": "dummy"},
            {"optimizer": "fusedadamw"},
            {"scheduler": "lambdalinear"},
            {"callbacks": "basic"},
            {"checkpoint": "local"},
            {"ckpt_type": "dcp"},
            {"model": "fsdp_t2v_distill"},
            {"net": None},
            {"net_teacher": None},
            {"net_fake_score": None},
            {"optimizer_fake_score": "fusedadamw"},
            {"conditioner": "text_nodrop"},
            {"ema": "power"},
            {"tokenizer": "wan2pt1_tokenizer"},
            # the list is with order, we need global experiment to be the last one
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    # Specifying values through instances of attrs
    c.job.project = "sparkdiffusion"  # this decides the wandb project name
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 100
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None

    # Call this function to register config groups for advanced overriding. the order follows the default config groups
    register_trainer()
    register_dataloader()
    register_optimizer()
    register_optimizer_fake_score()
    register_scheduler()
    register_callbacks()
    register_checkpoint()
    register_ckpt_type()
    register_model()
    register_net()
    register_net_teacher()
    register_net_fake_score()
    register_conditioner()
    register_ema()
    register_tokenizer()

    # experiment config are defined in the experiments folder
    import_all_modules_from_package("sparkdiffusion.configs.experiments.wan", reload=True)
    return c
