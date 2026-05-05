from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from model_logs import models_log
from sidd.validation_utils import evaluate_model
from se.configs import PROJECT_ROOT
from se.data import build_sidd_validation_loader
from se.utils.eval_utils import load_model_for_eval, select_logs_by_model_keys
from se.utils.runtime_utils import (
    AmpMode,
    resolve_amp_dtype,
    resolve_device,
)


@dataclass
class EvalSIDDConfig:
    model_keys: list[str] | None = None
    log_dir: str | None = None
    checkpoint: str | None = None
    checkpoint_override: str | None = None
    step: int | None = None
    val_noisy_mat: str = f"{PROJECT_ROOT}/data/SIDD/ValidationNoisyBlocksSrgb.mat"
    val_gt_mat: str = f"{PROJECT_ROOT}/data/SIDD/ValidationGtBlocksSrgb.mat"
    batch_size: int = 16
    num_workers: int = 0
    device: str | None = None
    amp: AmpMode = "auto"


def resolve_eval_paths(cfg: EvalSIDDConfig) -> list[tuple[str, Path]]:
    if cfg.model_keys:
        if cfg.log_dir is not None or cfg.checkpoint is not None:
            raise ValueError(
                "Set either EvalSIDDConfig.model_keys or log_dir/checkpoint, not both."
            )
        log_dirs, names = select_logs_by_model_keys(cfg.model_keys, models_log)
        return [
            (name, Path(log_dir).expanduser().resolve())
            for name, log_dir in zip(names, log_dirs)
        ]

    if cfg.checkpoint is None and cfg.log_dir is None:
        raise ValueError("Set EvalSIDDConfig.model_keys, log_dir, or checkpoint.")

    target = Path(cfg.checkpoint or cfg.log_dir).expanduser().resolve()  # type: ignore[arg-type]
    return [(str(target), target)]


def main(config: EvalSIDDConfig | None = None) -> None:
    cfg = config or EvalSIDDConfig(
        model_keys=["sidd_b_swinir", "sidd_wne_swinir"],
        # checkpoint_override="weights_best.pt",  # omit to use weights_last.pt
    )
    device = resolve_device(cfg.device)
    amp_dtype = resolve_amp_dtype(cfg.amp, device)
    checkpoint_override = (
        Path(cfg.checkpoint_override).expanduser()
        if cfg.checkpoint_override is not None
        else None
    )

    dataloader = build_sidd_validation_loader(
        cfg.val_noisy_mat,
        cfg.val_gt_mat,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    for name, path_or_log_dir in resolve_eval_paths(cfg):
        loaded = load_model_for_eval(
            path_or_log_dir,
            device,
            step=cfg.step,
            checkpoint_override=checkpoint_override,
        )
        model = loaded.model

        if device.type == "cuda" and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        metrics = evaluate_model(model, dataloader, device, amp_dtype)
        print(
            f"{name} | SIDD validation over {int(metrics['num_images'])} blocks | "
            f"PSNR: {metrics['psnr']:.4f} dB | SSIM: {metrics['ssim']:.6f}"
        )


if __name__ == "__main__":
    main()
