from __future__ import annotations

import contextlib
import random
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import torch


AmpMode = Literal["auto", "bf16", "fp16", "off"]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str | None) -> torch.device:
    if device_str is not None:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_amp_dtype(amp_mode: AmpMode, device: torch.device) -> torch.dtype | None:
    if amp_mode == "off" or device.type != "cuda":
        return None
    if amp_mode == "bf16":
        return torch.bfloat16
    if amp_mode == "fp16":
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def unwrap_data_parallel(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def strip_module_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return {}
    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return dict(state_dict)


def extract_model_state_dict(
    checkpoint: Mapping[str, Any], checkpoint_path: Path
) -> dict[str, Any]:
    state_dict_obj = checkpoint.get("model_state")
    if state_dict_obj is None:
        tensor_values = all(torch.is_tensor(value) for value in checkpoint.values())
        if tensor_values:
            state_dict_obj = dict(checkpoint)
        else:
            state_dict_obj = checkpoint.get("state_dict")
    if not isinstance(state_dict_obj, Mapping):
        raise KeyError(f"Could not find model weights in checkpoint: {checkpoint_path}")
    return strip_module_prefix(state_dict_obj)
