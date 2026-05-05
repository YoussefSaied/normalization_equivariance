import itertools
import math
import os
import sys
import time

from tqdm import tqdm
import wandb
import torch
from torch import Tensor
import torchvision
import torch.nn.functional as F

from se.configs import TrainConfig, resolve_image_mode
from se.data import build_loaders
from se.models import build_model
from se.utils.noise_model import get_noise, get_noise_pair
from se.utils.metrics import psnr, ssim
from se.utils.train_utils import (
    save_config,
    save_model_weights,
    setup_experiment,
    run_name,
)
from se.utils.psnr_plot import (
    default_psnr_sigma_values,
    normalize_sigma_values,
    plot_psnr_io,
    resolve_training_sigma_eval_grid,
)
from experiments_cfg import *


def l2_sum(output: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(output, target, reduction="sum") / (target.size(0) * 2)


def sample_soft_ne_affine(
    cfg: TrainConfig, reference: Tensor
) -> tuple[Tensor, Tensor] | None:
    if not cfg.soft_ne_loss:
        return None

    shape = (reference.shape[0],) + (1,) * (reference.ndim - 1)
    alpha = torch.empty(shape, device=reference.device, dtype=reference.dtype)
    alpha.uniform_(cfg.soft_ne_alpha_min, cfg.soft_ne_alpha_max)
    mu = torch.empty(shape, device=reference.device, dtype=reference.dtype)
    mu.uniform_(cfg.soft_ne_mu_min, cfg.soft_ne_mu_max)
    return alpha, mu


def apply_soft_ne_transform(
    cfg: TrainConfig, *tensors: Tensor
) -> tuple[tuple[Tensor, ...], tuple[Tensor, Tensor] | None]:
    if not tensors:
        return tuple(), None
    affine = sample_soft_ne_affine(cfg, tensors[0])
    if affine is None:
        return tensors, None

    alpha, mu = affine
    return tuple(alpha * tensor + mu for tensor in tensors), affine


def invert_soft_ne_transform(output: Tensor, affine: tuple[Tensor, Tensor]) -> Tensor:
    alpha, mu = affine
    safe_alpha = alpha.clamp_min(torch.finfo(alpha.dtype).eps)
    return (output - mu) / safe_alpha


def build_training_batch(
    cfg: TrainConfig,
    clean_inputs: Tensor,
) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor] | None]:
    noise_min = cfg.min_noise / 255.0
    noise_max = cfg.max_noise / 255.0
    objective = cfg.train_objective.lower()

    if objective == "supervised":
        noisy_inputs = clean_inputs + get_noise(
            clean_inputs,
            min_noise=noise_min,
            max_noise=noise_max,
            noise_type=cfg.noise_type,
        )
        (train_inputs, train_targets), soft_ne_affine = apply_soft_ne_transform(
            cfg,
            noisy_inputs,
            clean_inputs,
        )
        return train_inputs, train_targets, soft_ne_affine

    if objective == "n2n":
        noise_first, noise_second = get_noise_pair(
            clean_inputs,
            min_noise=noise_min,
            max_noise=noise_max,
            noise_type=cfg.noise_type,
        )
        noisy_first = clean_inputs + noise_first
        noisy_second = clean_inputs + noise_second
        (train_inputs, train_targets), soft_ne_affine = apply_soft_ne_transform(
            cfg,
            noisy_first,
            noisy_second,
        )
        return train_inputs, train_targets, soft_ne_affine

    raise ValueError(f"Unknown train_objective '{cfg.train_objective}'.")


def resolve_validation_sigma_values(cfg: TrainConfig) -> list[float]:
    base_sigma_values = (
        normalize_sigma_values(cfg.psnr_eval_sigma_values)
        if cfg.psnr_eval_sigma_values is not None
        else default_psnr_sigma_values()
    )
    validation_sigmas = resolve_training_sigma_eval_grid(
        training_sigma=(cfg.min_noise, cfg.max_noise),
        sigma_values=base_sigma_values,
    )
    if validation_sigmas:
        return validation_sigmas
    fallback_sigma = max(max(cfg.min_noise, cfg.max_noise) / 255.0, 1e-6)
    return [fallback_sigma]


def main(cfg: TrainConfig):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    save_dir = setup_experiment(cfg)
    cfg.wandb_cfg.name = run_name(cfg) + f"_{time.strftime('%m%d_%H%M')}"
    config_dict = save_config(cfg, save_dir)

    # Build data loaders, a model and an optimizer
    model = build_model(cfg).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    scheduler = None
    scheduler_step_mode = None
    if cfg.lr_halving_epochs:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.lr_halving_epochs, gamma=0.5
        )
        scheduler_step_mode = "epoch"
    elif cfg.lr_halving_steps:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_halving_steps, gamma=0.5
        )
        scheduler_step_mode = "step"

    print(
        f"Built a model consisting of {sum(p.numel() for p in model.parameters())/1e6:,}M parameters",
        flush=True,
    )
    loss_fn_dict = {
        "l2_sum": l2_sum,
        "l1": F.l1_loss,
        "l2": F.mse_loss,
    }
    loss_fn = loss_fn_dict[cfg.loss_type.lower()]

    wandb_run = None
    wandb_cfg = cfg.wandb_cfg
    if not wandb_cfg.mode == "disabled":
        wandb_run = wandb.init(
            **wandb_cfg.__dict__,
        )
        wandb_run.config.update(config_dict, allow_val_change=True)

    global_step = -1
    start_epoch = 0

    train_loader, valid_loader = build_loaders(cfg)

    stop_training = False
    best_valid_psnr = float("-inf")
    if cfg.num_steps is None:
        epoch_iter = range(start_epoch, cfg.num_epochs)
    else:
        epoch_iter = itertools.count(start_epoch)

    for epoch in tqdm(epoch_iter):
        # Training loop
        for batch_id, batch in enumerate(train_loader):
            model.train()

            global_step += 1
            clean_inputs = batch.to(device)
            train_inputs, train_targets, soft_ne_affine = build_training_batch(
                cfg, clean_inputs
            )

            outputs = model(train_inputs)
            loss: Tensor = loss_fn(outputs, train_targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None and scheduler_step_mode == "step":
                scheduler.step()
            if cfg.num_steps is not None and global_step + 1 >= cfg.num_steps:
                stop_training = True
                break

            should_log = cfg.log_interval <= 0 or global_step % cfg.log_interval == 0
            if wandb_run is not None and should_log:
                stats = {
                    "epoch": epoch,
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                metric_outputs = outputs
                if soft_ne_affine is not None:
                    metric_outputs = invert_soft_ne_transform(
                        metric_outputs, soft_ne_affine
                    )
                train_psnr = psnr(metric_outputs, clean_inputs)
                train_ssim = ssim(metric_outputs, clean_inputs)
                stats["train/psnr"] = train_psnr
                stats["train/ssim"] = train_ssim
                wandb.log(
                    stats,
                    step=global_step,
                )

        # Validation loop
        if epoch % cfg.valid_interval == 0:
            model.eval()
            valid_psnr_total = 0.0
            valid_ssim_total = 0.0
            valid_count = 0
            validation_sigma_values = resolve_validation_sigma_values(cfg)
            validation_sigma_values_8bit = [
                round(255.0 * s, 3) for s in validation_sigma_values
            ]
            display_sigma_idx = len(validation_sigma_values) // 2
            print(
                f"\nStarting validation at epoch {epoch=}, {global_step=}, "
                f"sigmas_8bit={validation_sigma_values_8bit}\n",
                flush=True,
            )

            for sample_id, sample in enumerate(valid_loader):
                with torch.no_grad():
                    sample = sample.to(device)
                    preview_image = None
                    for sigma_idx, sigma_std in enumerate(validation_sigma_values):
                        noise = get_noise(
                            sample,
                            min_noise=sigma_std,
                            max_noise=sigma_std,
                            noise_type=cfg.noise_type,
                        )

                        noisy_inputs = noise + sample
                        output = model(noisy_inputs)
                        valid_psnr = psnr(output, sample)
                        valid_ssim = ssim(output, sample)
                        valid_psnr_total += valid_psnr
                        valid_ssim_total += valid_ssim
                        valid_count += 1

                        if (
                            wandb_run is not None
                            and sample_id < 10
                            and sigma_idx == display_sigma_idx
                        ):
                            preview_image = torchvision.utils.make_grid(
                                torch.cat([sample, noisy_inputs, output], dim=0).clamp(
                                    0, 1
                                ),
                                nrow=3,
                                normalize=False,
                            )

                    if (
                        wandb_run is not None
                        and sample_id < 10
                        and preview_image is not None
                    ):
                        wandb.log(
                            {
                                f"valid_samples/{sample_id}": wandb.Image(
                                    preview_image.detach().cpu()
                                ),
                            },
                            step=global_step,
                        )

            avg_valid_psnr = valid_psnr_total / valid_count if valid_count > 0 else None
            avg_valid_ssim = valid_ssim_total / valid_count if valid_count > 0 else None

            if wandb_run is not None and valid_count > 0:
                wandb.log(
                    {
                        "valid/psnr": avg_valid_psnr,
                        "valid/ssim": avg_valid_ssim,
                        "valid/epoch": epoch,
                    },
                    step=global_step,
                )
                sys.stdout.flush()

            plot_path = os.path.join(save_dir, f"epoch_{epoch}_psnr_plot.png")
            training_sigma = (cfg.min_noise, cfg.max_noise)
            plot_data_dirs = (
                cfg.valid_path if cfg.valid_path is not None else cfg.test_path
            )
            plot_max_images = (
                cfg.valid_max_images if cfg.valid_path is not None else None
            )
            _, _, _, psnr_auc = plot_psnr_io(
                models=[model],
                data_dirs=plot_data_dirs,
                device=str(device),
                training_sigma=training_sigma,
                sigma_values=cfg.psnr_eval_sigma_values,
                save_path=plot_path,
                n_averages=10,
                dataset_mode=resolve_image_mode(cfg),
                max_images=plot_max_images,
                noise_type=cfg.noise_type,
                x_axis="sigma",
            )
            if wandb_run is not None:
                plot_stats: dict[str, object] = {
                    "valid/psnr_plot": wandb.Image(plot_path)
                }
                if isinstance(psnr_auc, float) and math.isfinite(psnr_auc):
                    plot_stats["valid/psnr_auc"] = psnr_auc
                wandb.log(plot_stats, step=global_step)

            if avg_valid_psnr is not None and avg_valid_psnr > best_valid_psnr:
                best_valid_psnr = avg_valid_psnr
                save_model_weights(model, save_dir, "best")
            save_model_weights(model, save_dir, "last")

        if scheduler is not None and scheduler_step_mode == "epoch":
            scheduler.step()
        if stop_training:
            break

    save_model_weights(model, save_dir, "last")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main(cfg_50_dncnn_wne)
