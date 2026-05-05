from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

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
from se.utils.psnr_plot import (
    compute_psnr_io,
    default_psnr_sigma_values,
    plot_psnr_curves,
    resolve_training_sigma_psnr,
)
from se.utils.train_utils import format_noise_level_tag


PSNR_METRIC_SPEC = EvalMetricSpec(
    name="psnr",
    input_column="input_psnr_db",
    train_value_column="psnr_db",
    raw_sigma_noise_types=frozenset({"jpeg"}),
)


@dataclass
class EvalConfig:
    """
    Provides evaluation settings. For custom sweeps, instantiate EvalConfig
    from a small Python entry point or modify the default block in main().
    """

    model_keys: Optional[list[str]] = None
    log_dirs: Optional[list[Path]] = None
    checkpoint: Optional[Path] = None
    epoch: Optional[int] = None
    test_path: list[str] | None = field(
        default_factory=lambda: [
            f"{PROJECT_ROOT}/data/Set12",
            # f"{PROJECT_ROOT}/data/Set68",
        ]
    )
    device: Optional[str] = None
    n_averages: int = 1
    sigma_values: Optional[list[float]] = None
    save_name: str = "ne_wne_50.png"
    show_legend: bool = True
    legend_loc: str = "lower right"
    legend_bbox_to_anchor: Optional[tuple[float, float]] = (1.01, -0.02)
    psnr_log_mean_mse: bool = False
    models_names: Optional[list[str] | dict] = None
    pretty_labels: Optional[dict[str, str]] = None
    model_colors: Optional[dict] = None
    eval_noise_type: Optional[str] = None
    use_cache: bool = True
    noise_level: Optional[float] = None  # when set, auto-pick logs matching this sigma
    key_substr: Optional[str] = None  # e.g., "dncnn" to filter model keys
    include_train_sigma_in_grid: bool = False


def select_logs_by_noise(
    noise_level: float, name_contains: Optional[str] = None
) -> tuple[list[Path], list[str]]:
    """
    Pick log dirs from models_log whose key ends with the given noise level and
    optionally contains a substring (case-insensitive). Returns (dirs, short names).
    """
    target_noise = format_noise_level_tag(noise_level)
    substring = name_contains.lower() if name_contains else None

    selected: list[tuple[str, Path]] = []
    for key, path in models_log.items():
        if substring and substring not in key.lower():
            continue
        parts = key.rsplit("_", 1)
        if len(parts) < 2 or parts[1] != target_noise:
            continue
        selected.append((key, path))

    selected.sort(key=lambda item: item[0])
    log_dirs = [Path(path) for _, path in selected]
    names = [key for key, _ in selected]
    return log_dirs, names


def main(eval_cfg: EvalConfig | None = None):
    if eval_cfg is None:
        desired_keys = [
            "b_swinir_10_n2n",
            "wne_swinir_10_n2n",
        ]
        # Keep EvalConfig defaults (in the dataclass declaration),
        # unless this specific plot explicitly needs an override.
        eval_cfg = EvalConfig(
            model_keys=desired_keys,
            save_name="swinir_sigma10_n2n_baseline_wne.pdf",
            sigma_values=default_psnr_sigma_values(),
            include_train_sigma_in_grid=True,
        )

    if eval_cfg.model_keys:
        configured_dirs, configured_names = select_logs_by_model_keys(
            eval_cfg.model_keys,
            models_log,
        )
    else:
        configured_dirs, configured_names = list(eval_cfg.log_dirs or []), []

    provided_names_raw = eval_cfg.models_names
    canonical_names = list(configured_names)
    pretty_labels = dict(eval_cfg.pretty_labels or {})
    model_colors = dict(default_color_overrides(configured_names))
    model_colors.update(dict(eval_cfg.model_colors or {}))

    if isinstance(provided_names_raw, Mapping):
        provided_names = [str(key) for key in provided_names_raw.keys()]
        model_colors.update({str(key): value for key, value in provided_names_raw.items()})
        use_auto_names = False
    else:
        provided_names = list(provided_names_raw or configured_names)
        use_auto_names = provided_names_raw is None and not configured_names

    if not configured_dirs and eval_cfg.noise_level is not None:
        configured_dirs, auto_names = select_logs_by_noise(
            eval_cfg.noise_level,
            eval_cfg.key_substr,
        )
        if not configured_dirs:
            raise ValueError(
                f"No log dirs found for noise={eval_cfg.noise_level} "
                f"with key containing '{eval_cfg.key_substr}'."
            )
        provided_names = auto_names
        canonical_names = list(auto_names)
        use_auto_names = False

    if not configured_dirs:
        raise ValueError("Provide at least one log directory via log_dirs.")

    resolved_dirs = [Path(raw_dir).expanduser().resolve() for raw_dir in configured_dirs]
    for candidate in resolved_dirs:
        if not candidate.is_dir():
            raise FileNotFoundError(f"Log directory {candidate} does not exist.")

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
        provided_names=provided_names,
        use_auto_names=use_auto_names,
        device=device,
        checkpoint_override=checkpoint_override,
        epoch=eval_cfg.epoch,
    )
    if not models:
        raise RuntimeError("No models were built for evaluation.")

    if not pretty_labels:
        pretty_labels = pretty_label_map(models_names)

    reference_cfg = cfgs[0]
    dataset_mode = resolve_image_mode(reference_cfg)
    train_noise_type = reference_cfg.noise_type
    if any(resolve_image_mode(cfg) != dataset_mode for cfg in cfgs[1:]):
        raise ValueError(
            "All models must use the same image mode to share evaluation data."
        )
    if any(cfg.noise_type != train_noise_type for cfg in cfgs[1:]):
        raise ValueError(
            "All models must use the same noise_type for a shared PSNR evaluation."
        )
    noise_type = eval_cfg.eval_noise_type or train_noise_type

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
        default_sigma_values=reference_cfg.psnr_eval_sigma_values,
    )
    save_root = PROJECT_ROOT / Path("eval_logs")
    save_root.mkdir(parents=True, exist_ok=True)

    if len(models) == 1:
        print(
            f"Running PSNR sweep for checkpoint {checkpoints[0].name} "
            f"on {device} using data at {', '.join(str(path) for path in test_path)}."
        )
    else:
        print(
            f"Running PSNR sweep for {len(models)} models on {device} "
            f"using data at {', '.join(str(path) for path in test_path)}."
        )
        for name, checkpoint_path in zip(models_names, checkpoints):
            print(f" - {name}: {checkpoint_path.name}")

    for data_dir in tqdm(test_path):
        dataset_name = data_dir.resolve().name
        dataset_save_root = save_root / dataset_name
        dataset_save_root.mkdir(parents=True, exist_ok=True)

        save_path = dataset_save_root / eval_cfg.save_name
        (
            x_vals,
            per_model_curves,
            label_order,
            sigma_values_list,
            _psnr_df,
            psnr_csv_path,
            input_psnr_vals,
        ) = cache_or_compute_metric(
            data_dir=data_dir,
            dataset_save_root=dataset_save_root,
            save_name=eval_cfg.save_name,
            sigma_values=sigma_values,
            training_sigma=training_sigma,
            models_names=models_names,
            models=models,
            noise_type=noise_type,
            use_cache=eval_cfg.use_cache,
            metric_spec=PSNR_METRIC_SPEC,
            compute_curves=compute_psnr_io,
            compute_kwargs={
                "n_averages": eval_cfg.n_averages,
                "device": str(device),
                "dataset_mode": dataset_mode,
                "log_mean_mse": eval_cfg.psnr_log_mean_mse,
                "noise_type": noise_type,
            },
        )

        plot_labels = [pretty_labels.get(name, name) for name in label_order]
        plot_colors = {
            pretty_labels.get(name, name): color for name, color in model_colors.items()
        }
        x_axis_label = "quality factor" if noise_type == "jpeg" else None
        identity_curve = input_psnr_vals if noise_type == "jpeg" else None
        identity_line = identity_curve is None
        x_ticks_major = list(range(5, 86, 10)) if noise_type == "jpeg" else None
        x_ticks_minor = list(range(5, 86, 5)) if noise_type == "jpeg" else None
        x_limits = (5, 85) if noise_type == "jpeg" else None
        y_limits = (24, 39) if noise_type == "jpeg" else None
        training_sigma_psnr = resolve_training_sigma_psnr(
            training_sigma=training_sigma,
            sigma_values=sigma_values_list,
            input_psnr_vals=list(input_psnr_vals if noise_type == "jpeg" else x_vals),
        )

        plot_psnr_curves(
            x_vals=x_vals,
            per_model_curves=[per_model_curves[label] for label in label_order],
            label_order=plot_labels,
            training_sigma=training_sigma,
            training_sigma_psnr=training_sigma_psnr,
            save_path=save_path,
            model_colors=plot_colors,
            show_legend=eval_cfg.show_legend,
            x_label=x_axis_label,
            show_identity=identity_line,
            identity_curve=identity_curve,
            x_ticks=x_ticks_major,
            x_ticks_minor=x_ticks_minor,
            x_limits=x_limits,
            y_limits=y_limits,
            legend_loc=eval_cfg.legend_loc,
            legend_bbox_to_anchor=eval_cfg.legend_bbox_to_anchor,
        )

        write_train_metric_csv(
            dataset_save_root=dataset_save_root,
            save_name=eval_cfg.save_name,
            training_sigma_single=training_sigma_single,
            models_names=models_names,
            training_sigmas=training_sigmas,
            sigma_values_list=sigma_values_list,
            per_model_curves=per_model_curves,
            metric_spec=PSNR_METRIC_SPEC,
        )

        print(
            f"Saved PSNR plot to {save_path} and CSV to {psnr_csv_path} "
            f"for dataset {dataset_name}."
        )


if __name__ == "__main__":
    main()
