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

import hashlib
import json
import os
import re as _re
from contextlib import contextmanager

import torch
from safetensors.torch import load as safetensors_torch_load

from imaginaire.utils.easy_io import easy_io
from imaginaire.utils import log


@contextmanager
def init_weights_on_device(device=torch.device("meta"), include_buffers: bool = False):  # noqa: B008
    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def load_state_dict_from_folder(file_path, torch_dtype=None):
    state_dict = {}
    for file_name in os.listdir(file_path):
        if "." in file_name and file_name.split(".")[-1] in ["safetensors", "bin", "ckpt", "pth", "pt"]:
            state_dict.update(load_state_dict(os.path.join(file_path, file_name), torch_dtype=torch_dtype))
    return state_dict


def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    backend_args = None
    state_dict = {}
    byte_stream = easy_io.load(file_path, backend_args=backend_args, file_format="byte")
    state_dict = safetensors_torch_load(byte_stream)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    if os.path.isfile(file_path):
        # mmap avoids reading the whole checkpoint into RAM up front; tensor
        # pages are pulled lazily during load_state_dict, which cuts several
        # seconds off multi-GB .pt/.pth loads. Prefer weights_only=True (no
        # arbitrary pickle execution); fall back only for legacy checkpoints
        # that store non-tensor objects.
        try:
            state_dict = torch.load(file_path, map_location="cpu", weights_only=True, mmap=True)
        except Exception:
            log.warning(f"weights_only load failed for {file_path}; falling back to full unpickling (trusted file assumed)")
            # Legacy pickle-based checkpoints are not guaranteed to use the
            # mmap-compatible zip format. Disable mmap on this compatibility
            # path so the fallback actually handles those files.
            state_dict = torch.load(file_path, map_location="cpu", weights_only=False, mmap=False)
    else:
        backend_args = None
        state_dict = easy_io.load(
            file_path, backend_args=backend_args, file_format="pt", map_location="cpu", weights_only=False
        )
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict


# RoLa parameters do not exist in a stock Wan checkpoint, so they are the one group
# that may legitimately be missing after a load (see load_checkpoint_auto).
_ROLA_PARAM_MARKERS = ("proj_q", "proj_k", "gate_proj", "gate_bias")


def _is_rola_param(name: str) -> bool:
    return any(marker in name for marker in _ROLA_PARAM_MARKERS)


def _load_safetensors_dir(path: str) -> dict:
    """Load all .safetensors files from a directory (or a single file) into one state_dict."""
    from safetensors import safe_open

    raw_sd: dict = {}
    if os.path.isdir(path):
        index_path = os.path.join(path, "diffusion_pytorch_model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                shard_files = set(json.load(f)["weight_map"].values())
            for shard in sorted(shard_files):
                with safe_open(os.path.join(path, shard), framework="pt", device="cpu") as f:
                    for k in f.keys():
                        raw_sd[k] = f.get_tensor(k)
        else:
            for fname in sorted(os.listdir(path)):
                if fname.endswith(".safetensors"):
                    with safe_open(os.path.join(path, fname), framework="pt", device="cpu") as f:
                        for k in f.keys():
                            raw_sd[k] = f.get_tensor(k)
    else:
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                raw_sd[k] = f.get_tensor(k)
    return raw_sd


def _detect_ckpt_format(path: str) -> str:
    """Detect checkpoint format: 'safetensors', 'pth', or 'dcp'."""
    if path.endswith(".safetensors") or (
        os.path.isdir(path) and not any(f.endswith(".distcp") for f in os.listdir(path))
    ):
        return "safetensors"
    if path.endswith(".pth") or path.endswith(".pt"):
        return "pth"
    return "dcp"


def load_checkpoint_auto(path: str, net: torch.nn.Module, strict: bool = False):
    """Auto-detect checkpoint format and load into net.

    Supports:
      - Native safetensors dir or single file (keys match WanModel params directly)
      - .pth / .pt file (with optional 'net.' or 'net_ema.' prefix stripping)

    Returns:
      torch.nn.modules.module._IncompatibleKeys (missing_keys, unexpected_keys)
    """
    fmt = _detect_ckpt_format(path)
    if fmt == "safetensors":
        raw_sd = _load_safetensors_dir(path)
        converted: dict = {}
        for k, v in raw_sd.items():
            rk = k
            if rk == "patch_embedding.weight" and v.ndim == 5:
                v = v.reshape(v.shape[0], -1)
            converted[rk] = v
        result = net.load_state_dict(converted, strict=strict, assign=True)
        rola_new = [k for k in result.missing_keys if _is_rola_param(k)]
        if rola_new:
            log.info(f"RoLa new params kept at random init ({len(rola_new)})")
        n_matched = len(converted) - len(result.unexpected_keys)
        log.success(f"Loaded safetensors checkpoint from {path} → {n_matched}/{len(converted)} tensors matched")
    elif fmt == "pth":
        sd = load_state_dict(path)
        sd_clean = {}
        for k, v in sd.items():
            if k.startswith("net_ema."):
                sd_clean[k[len("net_ema."):]] = v
            elif k.startswith("net."):
                sd_clean[k[len("net."):]] = v
            else:
                sd_clean[k] = v
        result = net.load_state_dict(sd_clean, strict=strict, assign=True)
        log.success(f"Loaded .pth checkpoint from {path}")
    else:
        raise ValueError(f"Unsupported checkpoint format for load_checkpoint_auto: {fmt} (path={path}). "
                         "DCP loading requires distributed setup; use load_ckpt_to_net instead.")

    # The DiT loads with strict=False, so unlike the VAE and text encoder it has no
    # backstop: a gap in the key mapping would only log a warning and leave real
    # weights at random init. That is exactly how the i2v image branch stayed broken
    # and silent (208 tensors; see wiki/0830_merge_plan_ckpt_adapt.md section 3.1.2).
    # Missing RoLa params are the one legitimate case, since initialising an RoLa model
    # from a dense checkpoint leaves them at random init on purpose.
    if result.missing_keys:
        non_rola_missing = [k for k in result.missing_keys if not _is_rola_param(k)]
        if non_rola_missing:
            raise RuntimeError(
                f"{len(non_rola_missing)} non-RoLa parameter(s) missing after loading {path}; "
                "they would silently stay at random init. This usually means the "
                "checkpoint-to-WanModel key mapping is incomplete for this model variant. "
                f"First 10: {non_rola_missing[:10]}. If you are deliberately loading a "
                "partially trained checkpoint, whitelist those names here."
            )
        log.info(f"Missing keys are all RoLa params ({len(result.missing_keys)}), kept at random init")
    if result.unexpected_keys:
        log.warning(f"Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:10]}{'...' if len(result.unexpected_keys) > 10 else ''}")

    # Materialize any params still on meta device (e.g. RoLa params missing from a dense checkpoint)
    meta_count = 0
    for name, param in list(net.named_parameters()):
        if param.device == torch.device("meta"):
            parts = name.split(".")
            mod = net
            for p in parts[:-1]:
                mod = getattr(mod, p)
            attr = parts[-1]
            new_param = torch.nn.Parameter(
                torch.empty(param.shape, dtype=torch.bfloat16, device="cpu"),
                requires_grad=param.requires_grad,
            )
            # Use proper init based on param name
            if "gate_bias" in name:
                new_param.data.fill_(-1.946)
            elif "gate_proj" in name:
                torch.nn.init.zeros_(new_param)
            else:
                torch.nn.init.kaiming_uniform_(new_param, a=2.236)  # a=sqrt(5)
            setattr(mod, attr, new_param)
            meta_count += 1
    if meta_count > 0:
        log.info(f"Materialized {meta_count} meta params with proper init")

    return result


def search_for_embeddings(state_dict):
    embeddings = []
    for k in state_dict:
        if isinstance(state_dict[k], torch.Tensor):
            embeddings.append(state_dict[k])
        elif isinstance(state_dict[k], dict):
            embeddings += search_for_embeddings(state_dict[k])
    return embeddings


def search_parameter(param, state_dict):
    for name, param_ in state_dict.items():
        if param.numel() == param_.numel():
            if param.shape == param_.shape:
                if torch.dist(param, param_) < 1e-3:
                    return name
            else:
                if torch.dist(param.flatten(), param_.flatten()) < 1e-3:
                    return name
    return None


def build_rename_dict(source_state_dict, target_state_dict, split_qkv=False):
    matched_keys = set()
    with torch.no_grad():
        for name in source_state_dict:
            rename = search_parameter(source_state_dict[name], target_state_dict)
            if rename is not None:
                print(f'"{name}": "{rename}",')
                matched_keys.add(rename)
            elif split_qkv and len(source_state_dict[name].shape) >= 1 and source_state_dict[name].shape[0] % 3 == 0:
                length = source_state_dict[name].shape[0] // 3
                rename = []
                for i in range(3):
                    rename.append(
                        search_parameter(source_state_dict[name][i * length : i * length + length], target_state_dict)
                    )
                if None not in rename:
                    print(f'"{name}": {rename},')
                    for rename_ in rename:
                        matched_keys.add(rename_)
    for name in target_state_dict:
        if name not in matched_keys:
            print("Cannot find", name, target_state_dict[name].shape)


def search_for_files(folder, extensions):
    files = []
    if os.path.isdir(folder):
        for file in sorted(os.listdir(folder)):
            files += search_for_files(os.path.join(folder, file), extensions)
    elif os.path.isfile(folder):
        for extension in extensions:
            if folder.endswith(extension):
                files.append(folder)
                break
    return files


def convert_state_dict_keys_to_single_str(state_dict, with_shape=True):
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor):
                if with_shape:
                    shape = "_".join(map(str, list(value.shape)))
                    keys.append(key + ":" + shape)
                keys.append(key)
            elif isinstance(value, dict):
                keys.append(key + "|" + convert_state_dict_keys_to_single_str(value, with_shape=with_shape))
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str


def split_state_dict_with_prefix(state_dict):
    keys = sorted([key for key in state_dict if isinstance(key, str)])
    prefix_dict = {}
    for key in keys:
        prefix = key if "." not in key else key.split(".")[0]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = []
        prefix_dict[prefix].append(key)
    state_dicts = []
    for prefix, keys in prefix_dict.items():  # noqa: B007
        sub_state_dict = {key: state_dict[key] for key in keys}
        state_dicts.append(sub_state_dict)
    return state_dicts


def hash_state_dict_keys(state_dict, with_shape=True):
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    return hashlib.md5(keys_str).hexdigest()
