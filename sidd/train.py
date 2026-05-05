from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import wandb
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from sidd.validation_utils import evaluate_model
from se.configs import (
    ImageMode,
    ModelConfig,
    ModelMode,
    PredMode,
    PROJECT_ROOT,
    TrainConfig,
    WandbConfig,
    WrapperMode,
)
from se.data import build_sidd_loaders
from se.models import build_model
from se.utils.runtime_utils import (
    AmpMode,
    autocast_context,
    set_random_seed,
    resolve_amp_dtype,
    resolve_device,
    unwrap_data_parallel,
)
from se.utils.train_utils import (
    load_training_state,
    resolve_output_dir,
    save_checkpoint,
    save_config,
)


@dataclass
class SIDDTrainConfig:
    train_dir: str = f"{PROJECT_ROOT}/data/SIDD/patches_png"
    val_noisy_mat: str = f"{PROJECT_ROOT}/data/SIDD/ValidationNoisyBlocksSrgb.mat"
    val_gt_mat: str = f"{PROJECT_ROOT}/data/SIDD/ValidationGtBlocksSrgb.mat"
    output_dir: str | None = None
    model: str = "swinir"
    image_mode: ImageMode = "rgb"
    model_mode: ModelMode = "ordinary"
    wrapper_mode: WrapperMode = "idem"
    pred_mode: PredMode = "direct"
    crop_size: int = 64
    batch_size: int = 32
    val_batch_size: int = 16
    num_workers: int = 8
    val_num_workers: int = 0
    num_steps: int = 400_000
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.0
    log_every: int = 100
    val_every: int = 20_000
    save_every: int = 5_000
    seed: int = 0
    amp: AmpMode = "auto"
    device: str | None = None
    resume: str | None = None
    clip_grad_norm: float | None = None
    wandb_cfg: WandbConfig = field(default_factory=WandbConfig)


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_runtime_model_config(cfg: SIDDTrainConfig) -> TrainConfig:
    return TrainConfig(
        model=cfg.model,
        image_mode=cfg.image_mode,
        model_cfg=ModelConfig(
            model_mode=cfg.model_mode,
            wrapper_mode=cfg.wrapper_mode,
            pred_mode=cfg.pred_mode,
        ),
    )


def compute_fast_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.detach().float().clamp(0.0, 1.0), target.detach().float())
    mse_value = max(float(mse.item()), 1e-12)
    return -10.0 * math.log10(mse_value)


def build_wandb_run_name(cfg: SIDDTrainConfig) -> str:
    timestamp = datetime.now().strftime("%m%d_%H%M")
    return f"sidd_{cfg.model}_{cfg.wrapper_mode}_{cfg.pred_mode}_{timestamp}"


@torch.inference_mode()
def make_validation_preview(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> torch.Tensor | None:
    was_training = model.training
    model.eval()
    iterator = iter(dataloader)
    try:
        noisy, gt = next(iterator)
    except StopIteration:
        if was_training:
            model.train()
        return None

    noisy = noisy[:1].to(device, non_blocking=True)
    gt = gt[:1].to(device, non_blocking=True)
    with autocast_context(device, amp_dtype):
        pred = model(noisy)

    preview = torchvision.utils.make_grid(
        torch.cat([gt, noisy, pred], dim=0).clamp(0.0, 1.0),
        nrow=3,
        normalize=False,
    )
    if was_training:
        model.train()
    return preview.detach().cpu()


def main(config: SIDDTrainConfig | None = None) -> None:
    cfg = config or SIDDTrainConfig()
    device = resolve_device(cfg.device)
    amp_dtype = resolve_amp_dtype(cfg.amp, device)
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and amp_dtype == torch.float16
    )
    set_random_seed(cfg.seed)
    torch.backends.cudnn.benchmark = device.type == "cuda"

    output_dir = resolve_output_dir(
        output_dir=cfg.output_dir,
        resume=cfg.resume,
        prefix="sidd_",
    )
    cfg.output_dir = str(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.wandb_cfg.name == WandbConfig().name:
        cfg.wandb_cfg.name = build_wandb_run_name(cfg)
    runtime_cfg = build_runtime_model_config(cfg)
    save_config(runtime_cfg, output_dir)
    config_dict = save_config(cfg, output_dir, filename="sidd_train_config.json")

    wandb_run = None
    if cfg.wandb_cfg.mode != "disabled":
        wandb_run = wandb.init(**cfg.wandb_cfg.__dict__)
        if wandb_run is not None:
            wandb_run.config.update(config_dict, allow_val_change=True)

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    train_loader, val_loader = build_sidd_loaders(
        cfg.train_dir,
        cfg.val_noisy_mat,
        cfg.val_gt_mat,
        crop_size=cfg.crop_size,
        batch_size=cfg.batch_size,
        val_batch_size=cfg.val_batch_size,
        num_workers=cfg.num_workers,
        val_num_workers=cfg.val_num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model = build_model(runtime_cfg).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        unwrap_data_parallel(model).parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.num_steps,
        eta_min=cfg.min_lr,
    )

    start_step = 0
    best_psnr = float("-inf")
    if cfg.resume is not None:
        checkpoint_state = load_training_state(
            Path(cfg.resume),
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if scaler.is_enabled() else None,
            device=device,
        )
        start_step = int(checkpoint_state.get("step", 0))
        best_psnr = float(checkpoint_state.get("best_psnr", float("-inf")))
        print(
            f"Resumed from {cfg.resume} at step {start_step} with best PSNR {best_psnr:.4f} dB",
            flush=True,
        )

    train_iter = iter(train_loader)
    log_loss_sum = 0.0
    log_psnr_sum = 0.0
    log_batches = 0
    log_images = 0
    log_start_time = time.perf_counter()

    for step in range(start_step + 1, cfg.num_steps + 1):
        try:
            noisy, gt = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            noisy, gt = next(train_iter)

        noisy = noisy.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_dtype):
            pred = model(noisy)
            loss = F.l1_loss(pred, gt)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if cfg.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(unwrap_data_parallel(model).parameters(), cfg.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.clip_grad_norm is not None:
                clip_grad_norm_(unwrap_data_parallel(model).parameters(), cfg.clip_grad_norm)
            optimizer.step()
        scheduler.step()

        batch_psnr = compute_fast_psnr(pred, gt)
        log_loss_sum += float(loss.item())
        log_psnr_sum += batch_psnr
        log_batches += 1
        log_images += int(noisy.shape[0])

        if step % cfg.log_every == 0 or step == 1:
            elapsed = max(time.perf_counter() - log_start_time, 1e-6)
            avg_loss = log_loss_sum / max(log_batches, 1)
            avg_psnr = log_psnr_sum / max(log_batches, 1)
            images_per_second = log_images / elapsed
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"step {step:7d}/{cfg.num_steps} | "
                f"lr {current_lr:.3e} | "
                f"train_l1 {avg_loss:.6f} | "
                f"train_psnr {avg_psnr:.3f} dB | "
                f"{images_per_second:.1f} img/s",
                flush=True,
            )
            if wandb_run is not None:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/psnr": avg_psnr,
                        "train/lr": current_lr,
                        "train/img_per_s": images_per_second,
                    },
                    step=step,
                )
            log_loss_sum = 0.0
            log_psnr_sum = 0.0
            log_batches = 0
            log_images = 0
            log_start_time = time.perf_counter()

        should_validate = step % cfg.val_every == 0 or step == cfg.num_steps
        should_save = step % cfg.save_every == 0 or step == cfg.num_steps

        if should_validate:
            metrics = evaluate_model(model, val_loader, device, amp_dtype)
            print(
                f"validation @ step {step:7d} | "
                f"PSNR {metrics['psnr']:.4f} dB | "
                f"SSIM {metrics['ssim']:.6f}",
                flush=True,
            )
            if wandb_run is not None:
                log_payload: dict[str, Any] = {
                    "valid/psnr": metrics["psnr"],
                    "valid/ssim": metrics["ssim"],
                }
                preview = make_validation_preview(model, val_loader, device, amp_dtype)
                if preview is not None:
                    log_payload["valid/preview"] = wandb.Image(preview)
                wandb.log(log_payload, step=step)
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                save_checkpoint(
                    model,
                    output_dir,
                    "best",
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler if scaler.is_enabled() else None,
                    step=step,
                    best_psnr=best_psnr,
                )
                print(
                    f"new best checkpoint saved to {output_dir / 'weights_best.pt'}",
                    flush=True,
                )

        if should_save:
            save_checkpoint(
                model,
                output_dir,
                "last",
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if scaler.is_enabled() else None,
                step=step,
                best_psnr=best_psnr,
            )
            save_checkpoint(
                model,
                checkpoints_dir / f"weights_step_{step:07d}.pt",
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if scaler.is_enabled() else None,
                step=step,
                best_psnr=best_psnr,
            )

    print(
        f"training complete | best validation PSNR {best_psnr:.4f} dB | "
        f"artifacts in {output_dir}",
        flush=True,
    )
    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
