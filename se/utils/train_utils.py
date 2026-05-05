import json
import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from se.configs import PROJECT_ROOT, TrainConfig, resolve_image_mode
from se.utils.runtime_utils import (
    extract_model_state_dict,
    set_random_seed,
    unwrap_data_parallel,
)


def resolve_output_dir(
    *,
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
    prefix: str = "",
) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    if resume is not None:
        resume_parent = Path(resume).expanduser().resolve().parent
        if resume_parent.name == "checkpoints":
            return resume_parent.parent
        return resume_parent

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{prefix}{timestamp}" if prefix else timestamp
    return Path(PROJECT_ROOT) / "logs" / run_dir_name


def _serialize_config(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError(f"Unsupported config type: {type(config)!r}.")


def save_json(data: Any, path: str | Path) -> dict[str, Any]:
    payload = _serialize_config(data)
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return payload


def setup_experiment(cfg: TrainConfig) -> str:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_random_seed(cfg.seed)

    save_dir = resolve_output_dir()
    save_dir.mkdir(parents=True, exist_ok=True)
    return str(save_dir)


def save_config(
    cfg: Any,
    save_dir: str | Path,
    filename: str = "config.json",
) -> dict[str, Any]:
    config_path = Path(save_dir) / filename
    return save_json(cfg, config_path)


def save_checkpoint(
    model: torch.nn.Module,
    save_dir_or_path: str | Path,
    name: str | None = None,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    step: int | None = None,
    epoch: int | None = None,
    best_psnr: float | None = None,
    extra_state: Mapping[str, Any] | None = None,
) -> Path:
    checkpoint_path = Path(save_dir_or_path)
    if name is not None:
        checkpoint_path = checkpoint_path / f"weights_{name}.pt"

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "model_state": unwrap_data_parallel(model).state_dict()
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state"] = scaler.state_dict()
    if step is not None:
        checkpoint["step"] = int(step)
    if epoch is not None:
        checkpoint["epoch"] = int(epoch)
    if best_psnr is not None:
        checkpoint["best_psnr"] = float(best_psnr)
    if extra_state is not None:
        checkpoint.update(dict(extra_state))

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_training_state(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    resolved_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(resolved_path, map_location=device or "cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Expected checkpoint at {resolved_path} to be a mapping.")

    unwrap_data_parallel(model).load_state_dict(
        extract_model_state_dict(checkpoint, resolved_path)
    )
    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    scheduler_state = checkpoint.get("scheduler_state")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    scaler_state = checkpoint.get("scaler_state")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    return dict(checkpoint)


def save_model_weights(model: torch.nn.Module, save_dir: str, name: str) -> str:
    checkpoint_path = save_checkpoint(model, save_dir, name)
    return str(checkpoint_path)


def format_noise_level_tag(noise_level: float) -> str:
    if math.isclose(noise_level, round(noise_level), rel_tol=0.0, abs_tol=1e-9):
        return str(int(round(noise_level)))
    return f"{noise_level:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def run_name(cfg: TrainConfig) -> str:
    noise_type = getattr(cfg, "noise_type", "gaussian").lower()
    suffix_map = {"laplace": "_l", "uniform": "_u", "rayleigh": "_r"}
    noise_type_brief = suffix_map.get(noise_type, "")

    # model mode/ wrapper mode
    if cfg.soft_ne_loss:
        model_mode = "softne"
    elif cfg.model_cfg.wrapper_mode == "norm-equiv":
        model_mode = "wne"
    elif cfg.model_cfg.wrapper_mode == "norm-equiv-input":
        model_mode = "wnei"
    elif cfg.model_cfg.wrapper_mode == "scale-equiv":
        model_mode = "wse"
    elif cfg.model_cfg.model_mode == "norm-equiv":
        model_mode = "ne"
    elif cfg.model_cfg.model_mode == "scale-equiv":
        model_mode = "se"
    else:
        model_mode = "b"

    # model name
    model_name = cfg.model.lower()

    # pred mode
    if cfg.model_cfg.pred_mode == "residual":
        pred_mode = "res"
    else:
        pred_mode = "dir"

    # loss
    loss_type = cfg.loss_type.lower()

    # dataset type
    if cfg.train_dataset_type.lower() == "m":
        dataset_type = "m"
    else:
        dataset_type = "h"

    # patch size
    patch_size = cfg.s_patch_size

    # noise level
    min_noise = format_noise_level_tag(cfg.min_noise)
    max_noise = format_noise_level_tag(cfg.max_noise)
    if min_noise == max_noise:
        noise_level = min_noise
    else:
        noise_level = f"{min_noise}-{max_noise}"

    base_name = f"{model_mode}_{model_name}_{pred_mode}_{loss_type}_{noise_level}{noise_type_brief}_{dataset_type}_{patch_size}"
    if resolve_image_mode(cfg) == "rgb":
        base_name = f"{base_name}_rgb"
    if getattr(cfg, "train_objective", "supervised").lower() != "supervised":
        base_name = f"{base_name}_{cfg.train_objective.lower()}"
    return f"{base_name}"
