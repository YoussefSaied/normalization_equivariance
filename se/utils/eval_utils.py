from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

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
from se.models import build_model
from se.utils.psnr_plot import normalize_sigma_values
from se.utils.runtime_utils import extract_model_state_dict
from se.utils.train_utils import run_name


@dataclass
class LoadedEvalModel:
    model: torch.nn.Module
    cfg: TrainConfig
    log_dir: Path
    checkpoint_path: Path


@dataclass(frozen=True)
class EvalMetricSpec:
    name: str
    input_column: str
    train_value_column: str
    raw_sigma_noise_types: frozenset[str] = frozenset()


EvalMetricComputer = Callable[
    ...,
    tuple[np.ndarray, list[np.ndarray], list[str], np.ndarray],
]


def _filter_known_fields(data: dict, cls: type) -> dict:
    allowed = {field.name for field in fields(cls)}
    return {key: value for key, value in data.items() if key in allowed}


def _load_json_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected config at {path} to be a mapping.")
    return dict(data)


def load_train_config(config_path: Path) -> TrainConfig:
    data = _load_json_mapping(config_path)
    if "model_cfg" in data:
        model_cfg_raw = data.get("model_cfg", {})
        wandb_cfg_raw = data.get("wandb_cfg", {})
        if not isinstance(model_cfg_raw, Mapping):
            raise TypeError(f"Expected model_cfg in {config_path} to be a mapping.")
        if not isinstance(wandb_cfg_raw, Mapping):
            raise TypeError(f"Expected wandb_cfg in {config_path} to be a mapping.")
        model_cfg = ModelConfig(
            **_filter_known_fields(dict(model_cfg_raw), ModelConfig)
        )
        wandb_cfg = WandbConfig(
            **_filter_known_fields(dict(wandb_cfg_raw), WandbConfig)
        )
        train_data = _filter_known_fields(data, TrainConfig)
        train_data.pop("model_cfg", None)
        train_data.pop("wandb_cfg", None)
        return TrainConfig(model_cfg=model_cfg, wandb_cfg=wandb_cfg, **train_data)

    return TrainConfig(
        model=cast(str, data.get("model", TrainConfig.model)),
        image_mode=cast(ImageMode, data.get("image_mode", "rgb")),
        model_cfg=ModelConfig(
            model_mode=cast(ModelMode, data.get("model_mode", "ordinary")),
            wrapper_mode=cast(WrapperMode, data.get("wrapper_mode", "idem")),
            pred_mode=cast(PredMode, data.get("pred_mode", "direct")),
        ),
    )


def resolve_checkpoint_path(
    log_dir: Path,
    epoch: int | None,
    explicit: Path | None,
    *,
    step: int | None = None,
) -> Path:
    if epoch is not None and step is not None:
        raise ValueError("Specify at most one of epoch or step.")

    if explicit is not None:
        explicit_candidates = [explicit.expanduser()]
        if not explicit.is_absolute():
            explicit_candidates.insert(0, log_dir / explicit)
        for candidate in explicit_candidates:
            ckpt = candidate.resolve()
            if ckpt.is_file():
                return ckpt
        raise FileNotFoundError(f"Checkpoint {explicit} does not exist.")

    if step is not None:
        step_candidates = [
            log_dir / f"weights_step_{step:07d}.pt",
            log_dir / "checkpoints" / f"weights_step_{step:07d}.pt",
            log_dir / "checkpoints" / f"step_{step:07d}.pt",
        ]
        for candidate in step_candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            f"No step checkpoint for step={step} found in {log_dir}."
        )

    last_ckpt = log_dir / "weights_last.pt"
    if epoch is None and last_ckpt.is_file():
        return last_ckpt
    legacy_last_ckpt = log_dir / "last.pt"
    if epoch is None and legacy_last_ckpt.is_file():
        return legacy_last_ckpt

    if epoch is None:
        step_checkpoints = sorted(log_dir.glob("weights_step_*.pt"))
        if step_checkpoints:
            return step_checkpoints[-1]
        step_checkpoints = sorted((log_dir / "checkpoints").glob("weights_step_*.pt"))
        if step_checkpoints:
            return step_checkpoints[-1]
        legacy_step_checkpoints = sorted((log_dir / "checkpoints").glob("step_*.pt"))
        if legacy_step_checkpoints:
            return legacy_step_checkpoints[-1]
        epoch_checkpoints = sorted(log_dir.glob("weights_epoch_*.pt"))
        if epoch_checkpoints:
            return epoch_checkpoints[-1]
        best_ckpt = log_dir / "weights_best.pt"
        if best_ckpt.is_file():
            return best_ckpt
        legacy_best_ckpt = log_dir / "best.pt"
        if legacy_best_ckpt.is_file():
            return legacy_best_ckpt
        raise FileNotFoundError(f"No checkpoints found in {log_dir}.")

    matching = log_dir / f"weights_epoch_{epoch:04d}.pt"
    if not matching.is_file():
        raise FileNotFoundError(f"Checkpoint {matching} does not exist.")
    return matching


def resolve_log_dir_from_checkpoint(
    checkpoint_path: Path, checkpoints_dir_name: str = "checkpoints"
) -> Path:
    if checkpoint_path.parent.name == checkpoints_dir_name:
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def load_model_for_eval(
    path_or_log_dir: Path,
    device: torch.device,
    *,
    epoch: int | None = None,
    step: int | None = None,
    checkpoint_override: Path | None = None,
) -> LoadedEvalModel:
    resolved_path = path_or_log_dir.expanduser().resolve()
    if resolved_path.is_file():
        checkpoint_path = resolved_path
        log_dir = resolve_log_dir_from_checkpoint(checkpoint_path)
    elif resolved_path.is_dir():
        log_dir = resolved_path
        checkpoint_path = resolve_checkpoint_path(
            log_dir,
            epoch,
            checkpoint_override,
            step=step,
        )
    else:
        raise FileNotFoundError(f"Path {resolved_path} does not exist.")

    config_path = log_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {log_dir}.")

    cfg = load_train_config(config_path)
    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, Mapping):
        state_dict = extract_model_state_dict(checkpoint, checkpoint_path)
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    return LoadedEvalModel(
        model=model,
        cfg=cfg,
        log_dir=log_dir,
        checkpoint_path=checkpoint_path,
    )


def select_logs_by_model_keys(
    model_keys: Iterable[str],
    registry: Mapping[str, Path | str],
) -> tuple[list[Path], list[str]]:
    requested_keys = list(model_keys)
    missing_keys = [key for key in requested_keys if key not in registry]
    if missing_keys:
        available = sorted(registry)
        raise ValueError(
            f"Missing models_log entries for {missing_keys}. Available keys: {available}"
        )
    return [Path(registry[key]) for key in requested_keys], requested_keys


def parse_model_key(key: str) -> tuple[str | None, str | None, str | None]:
    tokens = key.split("_")
    variant_tokens = {"b", "ne", "se", "wne", "wnei", "softne"}
    architecture_tokens = {"dncnn", "fdncnn", "swinir", "restormer"}
    objective_tokens = {"n2n"}
    variant = next((token for token in tokens if token in variant_tokens), None)
    architecture = next(
        (token for token in tokens if token in architecture_tokens), None
    )
    objective = next((token for token in tokens if token in objective_tokens), None)
    return variant, architecture, objective


def pretty_label_map(keys: Iterable[str]) -> dict[str, str]:
    label_by_variant = {
        "b": "Baseline",
        "ne": "NE-arch",
        "se": "SE-arch",
        "wne": r"$\mathbf{WNE}$",
        "wnei": "WNEI",
        "softne": "Soft-NE",
    }
    pretty: dict[str, str] = {}
    for key in keys:
        variant, _architecture, objective = parse_model_key(key)
        label = label_by_variant.get(variant, key) if variant else key
        if objective == "n2n":
            label = f"{label}-N2N"
        pretty[key] = label
    return pretty


def default_color_overrides(keys: Iterable[str]) -> dict[str, str | int]:
    color_palette = {
        "b": 5,
        "se": 6,
        "ne": 7,
        "wne": 8,
        "wnei": 9,
        "softne": 10,
    }
    return {
        key: color_palette[variant]
        for key in keys
        if (variant := parse_model_key(key)[0]) in color_palette
    }


def load_models_from_dirs(
    resolved_dirs: list[Path],
    provided_names: list[str],
    use_auto_names: bool,
    device: torch.device,
    checkpoint_override: Path | None,
    epoch: int | None,
) -> tuple[
    list[torch.nn.Module],
    list[str],
    list[Path],
    list[tuple[float, float]],
    list[TrainConfig],
]:
    models: list[torch.nn.Module] = []
    models_names: list[str] = []
    checkpoints: list[Path] = []
    training_sigmas: list[tuple[float, float]] = []
    cfgs: list[TrainConfig] = []

    for idx, log_dir in enumerate(resolved_dirs):
        config_path = log_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"No config.json found in {log_dir}.")
        loaded = load_model_for_eval(
            log_dir,
            device,
            epoch=epoch,
            checkpoint_override=checkpoint_override,
        )
        cfg = loaded.cfg
        cfgs.append(cfg)
        models.append(loaded.model)
        checkpoints.append(loaded.checkpoint_path)
        training_sigmas.append((cfg.min_noise, cfg.max_noise))

        if use_auto_names:
            name = run_name(cfg)
        elif idx < len(provided_names):
            name = provided_names[idx]
        else:
            name = run_name(cfg)
        models_names.append(name)

    return models, models_names, checkpoints, training_sigmas, cfgs


def prepare_sigma_values(
    eval_cfg: Any,
    training_sigmas: list[tuple[float, float]],
    noise_type: str | None = None,
    default_sigma_values: Sequence[float] | None = None,
) -> tuple[list[float], tuple[float, float] | None, float | None]:
    resolved_noise = (noise_type or "").lower()
    if resolved_noise == "jpeg":
        sigma_values = sorted(
            list(eval_cfg.sigma_values or default_sigma_values or range(5, 86, 5))
        )
        return sigma_values, None, None

    explicit_sigma_grid = eval_cfg.sigma_values is not None
    if explicit_sigma_grid:
        assert eval_cfg.sigma_values is not None
        sigma_values = normalize_sigma_values(eval_cfg.sigma_values)
    elif default_sigma_values is not None:
        sigma_values = normalize_sigma_values(list(default_sigma_values))
    else:
        sigma_values = [s / 255.0 for s in range(5, 100, 10)]

    training_sigma_single = None
    training_sigma = training_sigmas[0]
    for sigma in training_sigmas[1:]:
        if sigma != training_sigma:
            print(
                "Warning: models have different training noise ranges; "
                "skipping shaded training-sigma region."
            )
            training_sigma = None
            break

    if training_sigma is not None and math.isclose(*training_sigma):
        training_sigma_single = training_sigma[0]
        norm_train_sigma = training_sigma_single / 255.0
        has_train_sigma = any(
            math.isclose(norm_train_sigma, s, rel_tol=1e-6, abs_tol=1e-8)
            for s in sigma_values
        )
        if eval_cfg.include_train_sigma_in_grid and not has_train_sigma:
            if len(sigma_values) < 2:
                sigma_values = list(sigma_values) + [norm_train_sigma]
            else:
                tol = 1e-12
                is_non_decreasing = all(
                    sigma_values[idx] <= sigma_values[idx + 1] + tol
                    for idx in range(len(sigma_values) - 1)
                )
                is_non_increasing = all(
                    sigma_values[idx] >= sigma_values[idx + 1] - tol
                    for idx in range(len(sigma_values) - 1)
                )
                if is_non_decreasing:
                    insert_idx = next(
                        (
                            idx
                            for idx, value in enumerate(sigma_values)
                            if value > norm_train_sigma + tol
                        ),
                        len(sigma_values),
                    )
                    sigma_values = (
                        list(sigma_values[:insert_idx])
                        + [norm_train_sigma]
                        + list(sigma_values[insert_idx:])
                    )
                elif is_non_increasing:
                    insert_idx = next(
                        (
                            idx
                            for idx, value in enumerate(sigma_values)
                            if value < norm_train_sigma - tol
                        ),
                        len(sigma_values),
                    )
                    sigma_values = (
                        list(sigma_values[:insert_idx])
                        + [norm_train_sigma]
                        + list(sigma_values[insert_idx:])
                    )
                else:
                    sigma_values = list(sigma_values) + [norm_train_sigma]

    if not explicit_sigma_grid:
        sigma_values = sorted(sigma_values)
    return sigma_values, training_sigma, training_sigma_single


def _sigma_to_display_value(
    sigma_value: float,
    noise_type: str,
    raw_sigma_noise_types: frozenset[str],
) -> float:
    if noise_type.lower() in raw_sigma_noise_types:
        return sigma_value
    return sigma_value * 255.0


def _in_training_range(
    training_sigma: tuple[float, float] | None,
    sigma_8bit: float,
) -> bool:
    if training_sigma is None:
        return False
    lo, hi = training_sigma
    lo, hi = (min(lo, hi), max(lo, hi))
    return (lo - 1e-6) <= sigma_8bit <= (hi + 1e-6)


def _row_for_sigma(df: pd.DataFrame, target: float) -> pd.Series | None:
    matches = df.loc[
        df["sigma"].apply(
            lambda value: math.isclose(target, value, rel_tol=1e-6, abs_tol=1e-8)
        )
    ]
    if matches.empty:
        return None
    return matches.iloc[-1]


def _metric_x_values(
    metric_df: pd.DataFrame,
    metric_spec: EvalMetricSpec,
    noise_type: str,
) -> np.ndarray:
    if noise_type.lower() in metric_spec.raw_sigma_noise_types:
        return metric_df["sigma"].to_numpy()
    return metric_df[metric_spec.input_column].to_numpy()


def cache_or_compute_metric(
    *,
    data_dir: Path,
    dataset_save_root: Path,
    save_name: str,
    sigma_values: list[float],
    training_sigma: tuple[float, float] | None,
    models_names: list[str],
    models: list[torch.nn.Module],
    noise_type: str,
    use_cache: bool,
    metric_spec: EvalMetricSpec,
    compute_curves: EvalMetricComputer,
    compute_kwargs: Mapping[str, Any],
) -> tuple[
    np.ndarray,
    dict[str, list[float]],
    list[str],
    list[float],
    pd.DataFrame,
    Path,
    np.ndarray,
]:
    metric_csv_path = (
        dataset_save_root / f"{Path(save_name).stem}_{metric_spec.name}.csv"
    )
    base_cols = ["sigma_8bit", "sigma", metric_spec.input_column, "is_train_sigma"]

    existing_df = pd.read_csv(metric_csv_path) if metric_csv_path.is_file() else None
    cached_df: pd.DataFrame | None = None
    merged_df: pd.DataFrame | None = None
    metric_df: pd.DataFrame | None = None
    label_order: list[str] = []
    per_model_curves: dict[str, list[float]] = {}
    sigma_values_list = list(sigma_values)

    if use_cache and existing_df is not None:
        cached_df = existing_df.copy()
        model_cols = [column for column in cached_df.columns if column not in base_cols]
        expected_cols = models_names if models_names else model_cols
        missing_model_cols = [name for name in expected_cols if name not in model_cols]
        if missing_model_cols:
            print(
                f"Cache {metric_csv_path} missing model columns {missing_model_cols}; "
                "recomputing requested sigmas."
            )
            cached_df = None
            existing_df = None
        else:
            cached_df = cached_df[base_cols + list(expected_cols)]
            label_order = list(expected_cols)
            missing_sigmas: list[float] = []
            for sigma_value in sigma_values_list:
                row = _row_for_sigma(cached_df, sigma_value)
                if row is None:
                    missing_sigmas.append(sigma_value)
                    continue
                if any(pd.isna(row.get(name, np.nan)) or row.get(name) == "" for name in expected_cols):
                    missing_sigmas.append(sigma_value)

            if missing_sigmas:
                print(
                    f"Cache {metric_csv_path} missing sigmas {missing_sigmas}; "
                    "computing only those."
                )
                _, y_new, labels_new, input_metric_new = compute_curves(
                    models=models,
                    models_names=expected_cols,
                    data_dirs=[str(data_dir)],
                    sigma_values=missing_sigmas,
                    **compute_kwargs,
                )
                if len(missing_sigmas) != len(input_metric_new):
                    raise RuntimeError(
                        "Mismatch between missing sigma count and computed metric length."
                    )
                new_rows = []
                for idx, sigma_value in enumerate(missing_sigmas):
                    sigma_8bit = _sigma_to_display_value(
                        sigma_value,
                        noise_type,
                        metric_spec.raw_sigma_noise_types,
                    )
                    row = {
                        "sigma_8bit": round(sigma_8bit, 6),
                        "sigma": round(float(sigma_value), 8),
                        metric_spec.input_column: round(
                            float(input_metric_new[idx]), 6
                        ),
                        "is_train_sigma": _in_training_range(training_sigma, sigma_8bit),
                    }
                    for name, curve in zip(labels_new, y_new):
                        row[name] = round(float(curve[idx]), 6)
                    new_rows.append(row)
                cached_df = pd.concat(
                    [cached_df, pd.DataFrame(new_rows)],
                    ignore_index=True,
                )
                cached_df = cached_df.drop_duplicates(
                    subset=["sigma"],
                    keep="last",
                ).sort_values("sigma")

            ordered_rows = []
            for sigma_value in sigma_values_list:
                row = _row_for_sigma(cached_df, sigma_value)
                if row is None:
                    raise RuntimeError(f"Sigma {sigma_value} missing after cache refresh.")
                ordered_rows.append(row)
            metric_df = pd.DataFrame(ordered_rows)
            merged_df = cached_df

    if metric_df is None:
        _, y_arrays, label_order, input_metric_vals = compute_curves(
            models=models,
            models_names=models_names,
            data_dirs=[str(data_dir)],
            sigma_values=sigma_values_list,
            **compute_kwargs,
        )
        if len(sigma_values_list) != len(input_metric_vals):
            raise RuntimeError(
                "Mismatch between sigma_values and returned metric curve lengths."
            )
        per_model_curves = {
            label: [float(value) for value in curve]
            for label, curve in zip(label_order, y_arrays)
        }
        rows = []
        for idx, sigma_value in enumerate(sigma_values_list):
            sigma_8bit = _sigma_to_display_value(
                sigma_value,
                noise_type,
                metric_spec.raw_sigma_noise_types,
            )
            row = {
                "sigma_8bit": round(sigma_8bit, 6),
                "sigma": round(float(sigma_value), 8),
                metric_spec.input_column: round(float(input_metric_vals[idx]), 6),
                "is_train_sigma": _in_training_range(training_sigma, sigma_8bit),
            }
            for name in label_order:
                row[name] = round(float(per_model_curves[name][idx]), 6)
            rows.append(row)
        metric_df = pd.DataFrame(rows)
    else:
        input_metric_vals = metric_df[metric_spec.input_column].to_numpy()

    if merged_df is None:
        merged_df = cached_df if cached_df is not None else existing_df
    if merged_df is None:
        merged_df = metric_df
    else:
        merged_df = pd.concat([merged_df, metric_df], ignore_index=True)
        merged_df = merged_df.sort_values("sigma").drop_duplicates(
            subset=["sigma"],
            keep="last",
        )

    if not per_model_curves:
        if not label_order:
            label_order = [
                column for column in metric_df.columns if column not in base_cols
            ]
        per_model_curves = {
            label: metric_df[label].to_list()
            for label in label_order
            if label in metric_df
        }
        label_order = [label for label in label_order if label in per_model_curves]
        input_metric_vals = metric_df[metric_spec.input_column].to_numpy()

    ordered_cols = base_cols + [column for column in label_order if column not in base_cols]
    merged_df[ordered_cols].to_csv(metric_csv_path, index=False)

    return (
        _metric_x_values(metric_df, metric_spec, noise_type),
        per_model_curves,
        label_order,
        metric_df["sigma"].to_list(),
        metric_df,
        metric_csv_path,
        metric_df[metric_spec.input_column].to_numpy(),
    )


def write_train_metric_csv(
    *,
    dataset_save_root: Path,
    save_name: str,
    training_sigma_single: float | None,
    models_names: list[str],
    training_sigmas: list[tuple[float, float]],
    sigma_values_list: list[float],
    per_model_curves: dict[str, list[float]],
    metric_spec: EvalMetricSpec,
) -> None:
    if training_sigma_single is None:
        return

    train_csv_path = (
        dataset_save_root / f"{Path(save_name).stem}_train_{metric_spec.name}.csv"
    )
    train_rows = []
    for name, sigma_range in zip(models_names, training_sigmas):
        lo, hi = sigma_range
        if not math.isclose(lo, hi):
            train_rows.append(
                {
                    "model": name,
                    "train_sigma_8bit": f"{lo}-{hi}",
                    "train_sigma": "",
                    metric_spec.train_value_column: "",
                }
            )
            continue

        target_sigma = lo / 255.0
        try:
            idx = next(
                idx
                for idx, sigma_value in enumerate(sigma_values_list)
                if math.isclose(sigma_value, target_sigma, rel_tol=1e-6, abs_tol=1e-8)
            )
            metric_value = per_model_curves[name][idx]
        except StopIteration:
            metric_value = ""

        train_rows.append(
            {
                "model": name,
                "train_sigma_8bit": lo,
                "train_sigma": round(target_sigma, 8),
                metric_spec.train_value_column: (
                    round(float(metric_value), 6) if metric_value != "" else ""
                ),
            }
        )

    pd.DataFrame(train_rows).to_csv(train_csv_path, index=False)
