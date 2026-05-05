from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm

from model_logs import models_log
from se.configs import PROJECT_ROOT, resolve_image_mode
from se.utils.eval_utils import (
    EvalMetricSpec,
    cache_or_compute_metric,
    default_color_overrides,
    load_models_from_dirs,
    prepare_sigma_values,
    pretty_label_map,
    select_logs_by_model_keys,
    write_train_metric_csv,
)
from se.utils.ssim_plot import (
    compute_ssim_io,
    plot_ssim_curves,
    resolve_training_sigma_ssim,
)
from se.utils.train_utils import format_noise_level_tag


SSIM_METRIC_SPEC = EvalMetricSpec(
    name="ssim",
    input_column="input_ssim",
    train_value_column="ssim",
)


@dataclass
class EvalSsimConfig:
    """
    Provides SSIM evaluation settings.
    Use either `model_keys` for an explicit mixed-model comparison or
    `architecture` + `variants` + `noise_level` for one family.
    """

    model_keys: list[str] | None = None
    architecture: str = "fdncnn"
    noise_level: float = 50
    variants: list[str] = field(default_factory=lambda: ["b", "ne", "se", "wne"])
    test_path: list[str] | None = None
    device: str | None = None
    n_averages: int = 10
    sigma_values: list[float] | None = field(
        default_factory=lambda: np.geomspace(5, 100, num=20).tolist()
    )
    save_name: str | None = None
    show_legend: bool = True
    use_cache: bool = True
    include_train_sigma_in_grid: bool = True
    max_images: int | None = None
    checkpoint: Path | None = None
    epoch: int | None = None


def select_architecture_family_logs(
    architecture: str,
    noise_level: float,
    variants: Sequence[str],
) -> tuple[list[Path], list[str]]:
    noise_tag = format_noise_level_tag(noise_level)
    keys = [f"{variant}_{architecture}_{noise_tag}" for variant in variants]
    missing_keys = [key for key in keys if key not in models_log]
    if missing_keys:
        available = sorted(
            key
            for key in models_log
            if key.endswith(f"_{noise_tag}") and architecture in key
        )
        raise ValueError(
            "Missing model_logs entries for "
            f"{missing_keys}. Available matching keys: {available}"
        )
    return [Path(models_log[key]) for key in keys], keys


def main(eval_cfg: EvalSsimConfig | None = None):
    if eval_cfg is None:
        eval_cfg = EvalSsimConfig(
            model_keys=[
                "wne_swinir_10",
                "softne_swinir_10",
            ],
            save_name="swinir_sigma10_wne_vs_softne_ssim.pdf",
            test_path=[f"{PROJECT_ROOT}/data/Set12"],
            show_legend=True,
            n_averages=10,
            include_train_sigma_in_grid=True,
        )

    if eval_cfg.model_keys:
        log_dirs, canonical_names = select_logs_by_model_keys(
            eval_cfg.model_keys,
            models_log,
        )
    else:
        log_dirs, canonical_names = select_architecture_family_logs(
            architecture=eval_cfg.architecture,
            noise_level=eval_cfg.noise_level,
            variants=eval_cfg.variants,
        )

    pretty_labels = pretty_label_map(canonical_names)
    model_colors = default_color_overrides(canonical_names)
    save_name = (
        eval_cfg.save_name
        if eval_cfg.save_name is not None
        else f"{eval_cfg.architecture}_sigma{format_noise_level_tag(eval_cfg.noise_level)}_ssim.pdf"
    )
    eval_cfg.save_name = save_name

    resolved_dirs = [Path(path).expanduser().resolve() for path in log_dirs]
    for path in resolved_dirs:
        if not path.is_dir():
            raise FileNotFoundError(f"Log directory {path} does not exist.")

    checkpoint_override = (
        Path(eval_cfg.checkpoint).expanduser().resolve()
        if eval_cfg.checkpoint is not None
        else None
    )
    if checkpoint_override is not None and len(resolved_dirs) > 1:
        raise ValueError(
            "checkpoint overrides are only supported when evaluating a single log dir."
        )

    device_str = (
        eval_cfg.device
        if eval_cfg.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = torch.device(device_str)

    models, models_names, checkpoints, training_sigmas, cfgs = load_models_from_dirs(
        resolved_dirs=resolved_dirs,
        provided_names=canonical_names,
        use_auto_names=False,
        device=device,
        checkpoint_override=checkpoint_override,
        epoch=eval_cfg.epoch,
    )
    if not models:
        raise RuntimeError("No models were built for evaluation.")

    reference_cfg = cfgs[0]
    dataset_mode = resolve_image_mode(reference_cfg)
    noise_type = reference_cfg.noise_type
    if any(resolve_image_mode(cfg) != dataset_mode for cfg in cfgs[1:]):
        raise ValueError(
            "All models must use the same image mode to share evaluation data."
        )
    if any(cfg.noise_type != noise_type for cfg in cfgs[1:]):
        raise ValueError(
            "All models must use the same noise_type for a shared SSIM evaluation."
        )

    if eval_cfg.test_path is not None:
        test_path = [Path(path).expanduser() for path in eval_cfg.test_path]
    else:
        test_path = [Path(path) for path in reference_cfg.test_path]
        for cfg in cfgs[1:]:
            if cfg.test_path != reference_cfg.test_path:
                print(
                    "Warning: differing test_path values detected; "
                    "using the paths from the first config."
                )
                break

    for data_dir in test_path:
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Test data directory {data_dir} does not exist.")

    sigma_values, training_sigma, training_sigma_single = prepare_sigma_values(
        eval_cfg,
        training_sigmas,
        noise_type=noise_type,
    )
    save_root = PROJECT_ROOT / Path("eval_logs")
    save_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Running SSIM sweep for {len(models)} models on {device} "
        f"using data at {', '.join(str(path) for path in test_path)}."
    )
    for name, checkpoint_path in zip(models_names, checkpoints):
        print(f" - {pretty_labels.get(name, name)}: {checkpoint_path.name}")

    for data_dir in tqdm(test_path):
        dataset_name = data_dir.resolve().name
        dataset_save_root = save_root / dataset_name
        dataset_save_root.mkdir(parents=True, exist_ok=True)

        save_path = dataset_save_root / save_name
        (
            x_vals,
            per_model_curves,
            label_order,
            sigma_values_list,
            _ssim_df,
            ssim_csv_path,
            input_ssim_vals,
        ) = cache_or_compute_metric(
            data_dir=data_dir,
            dataset_save_root=dataset_save_root,
            save_name=save_name,
            sigma_values=sigma_values,
            training_sigma=training_sigma,
            models_names=models_names,
            models=models,
            noise_type=noise_type,
            use_cache=eval_cfg.use_cache,
            metric_spec=SSIM_METRIC_SPEC,
            compute_curves=compute_ssim_io,
            compute_kwargs={
                "n_averages": eval_cfg.n_averages,
                "device": str(device),
                "dataset_mode": dataset_mode,
                "max_images": eval_cfg.max_images,
                "noise_type": noise_type,
            },
        )

        plot_labels = [pretty_labels.get(name, name) for name in label_order]
        plot_colors = {
            pretty_labels.get(name, name): color for name, color in model_colors.items()
        }
        training_sigma_ssim = resolve_training_sigma_ssim(
            training_sigma=training_sigma,
            sigma_values=sigma_values_list,
            input_ssim_vals=list(input_ssim_vals),
        )
        plot_ssim_curves(
            x_vals=x_vals,
            per_model_curves=[per_model_curves[label] for label in label_order],
            label_order=plot_labels,
            training_sigma_ssim=training_sigma_ssim,
            save_path=save_path,
            model_colors=plot_colors,
            show_legend=eval_cfg.show_legend,
        )

        write_train_metric_csv(
            dataset_save_root=dataset_save_root,
            save_name=save_name,
            training_sigma_single=training_sigma_single,
            models_names=models_names,
            training_sigmas=training_sigmas,
            sigma_values_list=sigma_values_list,
            per_model_curves=per_model_curves,
            metric_spec=SSIM_METRIC_SPEC,
        )

        print(
            f"Saved SSIM plot to {save_path} and CSV to {ssim_csv_path} "
            f"for dataset {dataset_name}."
        )


if __name__ == "__main__":
    main()
