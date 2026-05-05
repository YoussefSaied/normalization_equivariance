from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter
from torch import nn
from tqdm import tqdm

from se.configs import NoiseType, PROJECT_ROOT
from se.utils.metrics import ssim
from se.utils.noise_model import get_noise
from se.utils.psnr_plot import to_tensor01


def _resolve_color(
    label: str, color_map: Mapping[str, str | int], color_cycle: list[str]
) -> str | None:
    if label in color_map:
        value = color_map[label]
        if isinstance(value, int) and color_cycle:
            return color_cycle[value % len(color_cycle)]
        if isinstance(value, str):
            return value
    return None


def _merge_color_overrides(
    models_names: Mapping[str, str | int] | None,
    model_colors: Mapping[str, str | int] | None,
) -> dict[str, str | int]:
    merged: dict[str, str | int] = {}
    if models_names is not None:
        merged.update({str(k): v for k, v in models_names.items()})
    if model_colors is not None:
        merged.update(model_colors)
    return merged


def _prepare_models(
    models: nn.Module | Sequence[nn.Module],
    provided_names: Sequence[str],
) -> list[tuple[str, nn.Module]]:
    if isinstance(models, nn.Module):
        models_to_eval = [models]
    else:
        models_to_eval = list(models)
    if len(models_to_eval) == 0:
        raise ValueError("No models were supplied.")

    named_models: list[tuple[str, nn.Module]] = []
    for idx, mdl in enumerate(models_to_eval):
        if mdl is None:
            raise ValueError(f"models[{idx}] is None.")
        label = (
            provided_names[idx] if idx < len(provided_names) else mdl.__class__.__name__
        )
        named_models.append((label, mdl))
    return named_models


def resolve_training_sigma_ssim(
    training_sigma: tuple[float, float] | None,
    sigma_values: Sequence[float],
    input_ssim_vals: Sequence[float],
) -> float | tuple[float, float] | None:
    if training_sigma is None or len(sigma_values) != len(input_ssim_vals):
        return None

    lo8, hi8 = training_sigma
    lo_sigma = lo8 / 255.0
    hi_sigma = hi8 / 255.0

    if math.isclose(lo_sigma, hi_sigma, rel_tol=1e-6, abs_tol=1e-8):
        target = lo_sigma
        for sigma_value, input_ssim in zip(sigma_values, input_ssim_vals):
            if math.isclose(sigma_value, target, rel_tol=1e-6, abs_tol=1e-8):
                return float(input_ssim)
        return None

    low_target, high_target = sorted((lo_sigma, hi_sigma))
    low_ssim = None
    high_ssim = None
    for sigma_value, input_ssim in zip(sigma_values, input_ssim_vals):
        if low_ssim is None and math.isclose(
            sigma_value, low_target, rel_tol=1e-6, abs_tol=1e-8
        ):
            low_ssim = float(input_ssim)
        if high_ssim is None and math.isclose(
            sigma_value, high_target, rel_tol=1e-6, abs_tol=1e-8
        ):
            high_ssim = float(input_ssim)
        if low_ssim is not None and high_ssim is not None:
            break
    if low_ssim is not None and high_ssim is not None:
        return (low_ssim, high_ssim)
    return None


def compute_ssim_io(
    models: nn.Module | Sequence[nn.Module],
    data_dirs: list[str] | None = None,
    sigma_values: Sequence[float] | None = None,
    n_averages: int = 10,
    device: str = "cuda",
    dataset_mode: str = "m",
    max_inference_batch: int | None = None,
    max_images: int | None = None,
    models_names: Sequence[str] | Mapping[str, str | int] | None = None,
    noise_type: NoiseType = "gaussian",
) -> tuple[np.ndarray, list[np.ndarray], list[str], np.ndarray]:
    if n_averages <= 0:
        raise ValueError("n_averages must be a positive integer.")
    provided_names: list[str] = []
    if isinstance(models_names, Mapping):
        provided_names = [str(k) for k in models_names.keys()]
    elif models_names is not None:
        provided_names = [str(name) for name in models_names]

    if data_dirs is None:
        raise ValueError("data_dirs must be provided.")

    named_models = _prepare_models(models, provided_names)

    files: list[Path] = []
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        files_in_dir = sorted(
            [
                path
                for path in data_path.iterdir()
                if path.suffix.lower()
                in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            ]
        )
        if max_images is not None:
            files_in_dir = files_in_dir[:max_images]
        files.extend(files_in_dir)

    if len(files) == 0:
        raise FileNotFoundError(f"no images in {data_dirs}")

    imgs = [to_tensor01(cv2.imread(path.as_posix()), mode=dataset_mode) for path in files]

    if sigma_values is None:
        sigma_values = [s / 255.0 for s in range(5, 100, 10)]
    sigma_values = [float(value) for value in sigma_values]
    if len(sigma_values) == 0:
        raise ValueError("sigma_values must contain at least one value.")

    device_t = torch.device(device)
    for _, mdl in named_models:
        mdl.to(device_t).eval()

    sigma_tensor = torch.tensor(sigma_values, device=device_t, dtype=imgs[0].dtype)
    max_batch = (
        n_averages if max_inference_batch is None else max(1, int(max_inference_batch))
    )
    oom_auto_reduced = False

    sum_input_metric = np.zeros(len(sigma_values), dtype=np.float64)
    sum_output_metric = np.zeros((len(named_models), len(sigma_values)), dtype=np.float64)

    for img in tqdm(imgs):
        clean = img.to(device_t, non_blocking=True)
        per_img_input = np.zeros(len(sigma_values), dtype=np.float64)
        per_img_output = np.zeros((len(named_models), len(sigma_values)), dtype=np.float64)

        for sigma_idx, sigma_value in enumerate(sigma_tensor):
            sigma_std = float(sigma_value)
            sigma_input_sum = 0.0
            sigma_output_sum = np.zeros(len(named_models), dtype=np.float64)
            repeats_done = 0

            while repeats_done < n_averages:
                current_batch = min(max_batch, n_averages - repeats_done)
                while True:
                    batch_clean = None
                    noise = None
                    noisy_batch = None
                    try:
                        batch_clean = clean.expand(current_batch, -1, -1, -1)
                        noise = get_noise(
                            batch_clean,
                            min_noise=sigma_std,
                            max_noise=sigma_std,
                            noise_type=noise_type,
                        )
                        noisy_batch = batch_clean + noise

                        sigma_input_sum += float(ssim(batch_clean, noisy_batch)) * float(
                            current_batch
                        )

                        with torch.inference_mode():
                            for model_idx, (_, mdl) in enumerate(named_models):
                                denoised_batch = mdl(noisy_batch).clamp(0.0, 1.0)
                                sigma_output_sum[model_idx] += float(
                                    ssim(batch_clean, denoised_batch)
                                ) * float(current_batch)
                        break
                    except torch.OutOfMemoryError:
                        if device_t.type != "cuda" or current_batch <= 1:
                            raise
                        if not oom_auto_reduced:
                            print(
                                "compute_ssim_io: CUDA OOM during evaluation; "
                                "reducing inference batch size automatically."
                            )
                            oom_auto_reduced = True
                        del batch_clean, noise, noisy_batch
                        torch.cuda.empty_cache()
                        current_batch = max(1, current_batch // 2)

                repeats_done += current_batch

            per_img_input[sigma_idx] = sigma_input_sum / float(n_averages)
            per_img_output[:, sigma_idx] = sigma_output_sum / float(n_averages)

        sum_input_metric += per_img_input
        sum_output_metric += per_img_output

    divisor = float(len(imgs))
    x_vals = sum_input_metric / divisor
    per_model_curves = [
        sum_output_metric[idx] / divisor for idx in range(len(named_models))
    ]
    label_order = [label for label, _ in named_models]
    input_ssim_vals = np.array(x_vals, dtype=np.float64)
    return input_ssim_vals, per_model_curves, label_order, input_ssim_vals


def _default_limits(values: Sequence[float] | np.ndarray) -> tuple[float, float]:
    values_array = np.asarray(values, dtype=np.float64)
    finite_values = values_array[np.isfinite(values_array)]
    if finite_values.size == 0:
        return (0.0, 1.0)

    lo = float(np.nanmin(finite_values))
    hi = float(np.nanmax(finite_values))
    if math.isclose(lo, hi, rel_tol=1e-6, abs_tol=1e-8):
        pad = 0.05
    else:
        pad = max(0.02, 0.06 * (hi - lo))
    return (max(0.0, lo - pad), min(1.0, hi + pad))


def _default_x_ticks(
    x_vals: Sequence[float] | np.ndarray,
) -> tuple[list[float], list[float] | None]:
    unique_vals = np.unique(np.round(np.asarray(x_vals, dtype=np.float64), 6))
    if unique_vals.size == 0:
        return [0.0, 0.5, 1.0], None
    if unique_vals.size <= 12:
        tick_values = [float(value) for value in unique_vals]
        return tick_values, tick_values

    major_ticks = unique_vals[::2]
    if not math.isclose(float(major_ticks[-1]), float(unique_vals[-1]), abs_tol=1e-9):
        major_ticks = np.append(major_ticks, unique_vals[-1])
    return [float(value) for value in major_ticks], [float(value) for value in unique_vals]


def _format_tick_labels(values: Sequence[float]) -> list[str]:
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.size <= 1:
        precision = 2
    else:
        diffs = np.diff(np.sort(values_array))
        positive_diffs = diffs[diffs > 1e-9]
        min_gap = float(np.min(positive_diffs)) if positive_diffs.size > 0 else 0.1
        precision = 3 if min_gap < 0.05 else 2
    labels = []
    for value in values_array:
        label = f"{value:.{precision}f}".rstrip("0").rstrip(".")
        if "." not in label:
            label = f"{label}.0"
        labels.append(label)
    return labels


def _unit_axis_ticks(limits: tuple[float, float]) -> tuple[list[float], list[float]]:
    lo, hi = limits
    major_start = max(0.0, math.floor((lo + 1e-9) * 10.0) / 10.0)
    major_stop = min(1.0, math.ceil((hi - 1e-9) * 10.0) / 10.0)
    major_ticks = np.arange(major_start, major_stop + 1e-9, 0.1).round(10).tolist()
    minor_ticks = np.arange(major_start, major_stop + 1e-9, 0.05).round(10).tolist()
    return major_ticks, minor_ticks


def plot_ssim_curves(
    x_vals: Sequence[float] | np.ndarray,
    per_model_curves: Sequence[Sequence[float] | np.ndarray],
    label_order: Sequence[str],
    training_sigma_ssim: float | tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    style_path: str | Path | None = None,
    model_colors: Mapping[str, str | int] | None = None,
    show_legend: bool = True,
    x_label: str | None = None,
    show_identity: bool = True,
    identity_curve: Sequence[float] | np.ndarray | None = None,
    x_ticks: Sequence[float] | None = None,
    x_ticks_minor: Sequence[float] | None = None,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
):
    color_overrides = _merge_color_overrides(None, model_colors)

    style_file = (
        Path(style_path)
        if style_path is not None
        else Path(PROJECT_ROOT) / "icml_like.mplstyle"
    )
    style_ctx: Any = (
        plt.style.context(str(style_file)) if style_file.is_file() else nullcontext()
    )
    with style_ctx:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

        for label, curve in zip(label_order, per_model_curves):
            color = _resolve_color(label, color_overrides, color_cycle)
            ax.plot(x_vals, curve, marker="o", linewidth=1.8, label=label, color=color)
            ax.scatter(x_vals, curve, color=color, s=18, zorder=8, label="_nolegend_")

        x_vals_array = np.asarray(x_vals, dtype=np.float64)
        all_y_vals = np.concatenate(
            [np.asarray(curve, dtype=np.float64) for curve in per_model_curves]
        )

        resolved_x_limits = x_limits if x_limits is not None else _default_limits(x_vals_array)
        combined_y = np.concatenate([all_y_vals, x_vals_array])
        resolved_y_limits = y_limits if y_limits is not None else _default_limits(combined_y)

        x_min, x_max = float(resolved_x_limits[0]), float(resolved_x_limits[1])
        if identity_curve is not None:
            ax.plot(
                x_vals,
                identity_curve,
                linestyle="--",
                color="k",
                linewidth=1.2,
                label="identity",
            )
        elif show_identity:
            line_x = np.linspace(x_min, x_max, 256)
            ax.plot(
                line_x,
                line_x,
                linestyle="--",
                color="k",
                linewidth=1.2,
                label="identity",
            )

        if training_sigma_ssim is not None:
            if isinstance(training_sigma_ssim, tuple):
                left, right = sorted(training_sigma_ssim)
                if math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-8):
                    ax.axvline(left, color="#69a8d8", linewidth=3.0, alpha=0.8)
                else:
                    ax.axvspan(left, right, alpha=0.18, color="#69a8d8")
            else:
                ax.axvline(training_sigma_ssim, color="#69a8d8", linewidth=3.0, alpha=0.8)

        ax.set_xlabel(x_label or "input SSIM")
        ax.set_ylabel("output SSIM")

        resolved_x_ticks = list(x_ticks) if x_ticks is not None else None
        resolved_x_minor_ticks = list(x_ticks_minor) if x_ticks_minor is not None else None
        if resolved_x_ticks is None:
            resolved_x_ticks, resolved_x_minor_ticks = _default_x_ticks(x_vals_array)

        if resolved_x_ticks:
            ax.set_xticks(resolved_x_ticks)
            ax.set_xticklabels(_format_tick_labels(resolved_x_ticks), rotation=90)
        if resolved_x_minor_ticks is not None:
            ax.set_xticks(resolved_x_minor_ticks, minor=True)

        y_major_ticks, y_minor_ticks = _unit_axis_ticks(resolved_y_limits)
        ax.set_yticks(y_major_ticks)
        ax.set_yticks(y_minor_ticks, minor=True)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        ax.grid(
            True,
            which="major",
            axis="both",
            linestyle="-",
            linewidth=0.45,
            alpha=0.35,
            color="0.65",
        )
        ax.grid(True, which="minor", linewidth=0.25, alpha=0.2, color="0.82")

        if show_legend:
            legend = ax.legend(
                loc="lower right",
                bbox_to_anchor=(1.01, -0.02),
                frameon=True,
                facecolor="1.0",
                edgecolor="0.4",
                framealpha=1.0,
                borderpad=0.35,
                handlelength=1.6,
            )
            for line in legend.get_lines():
                line.set_linewidth(2.2)
                if line.get_label() != "identity":
                    line.set_marker("o")
                    line.set_markersize(6)

        ax.set_xlim(resolved_x_limits)
        ax.set_ylim(resolved_y_limits)
        ax.tick_params(axis="x", labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("black")
        fig.tight_layout()

        if save_path is not None:
            target_path = Path(save_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target_path, bbox_inches="tight")

    return fig


def _summarize_curves(
    x_vals: Sequence[float] | np.ndarray,
    per_model_curves: Sequence[Sequence[float] | np.ndarray],
    label_order: Sequence[str],
):
    x_vals_np = np.asarray(x_vals, dtype=np.float64)
    y_arrays = [np.asarray(curve, dtype=np.float64) for curve in per_model_curves]
    order = np.argsort(x_vals_np)
    auc_values = [
        float(abs(np.trapezoid(y_arr[order], x_vals_np[order]))) for y_arr in y_arrays
    ]

    if len(label_order) == 1:
        y_return = y_arrays[0]
        auc_return = auc_values[0]
    else:
        y_return = {label: y_arr for label, y_arr in zip(label_order, y_arrays)}
        auc_return = {label: auc for label, auc in zip(label_order, auc_values)}
    return x_vals_np, y_return, auc_return


def plot_ssim_io(
    models: nn.Module | Sequence[nn.Module],
    data_dirs: list[str] | None = None,
    sigma_values: Sequence[float] | None = None,
    n_averages: int = 10,
    training_sigma: tuple[int, int] | tuple[float, float] | None = None,
    device: str = "cuda",
    save_path: str | Path | None = None,
    dataset_mode: str = "m",
    max_inference_batch: int | None = None,
    max_images: int | None = None,
    models_names: Sequence[str] | Mapping[str, str | int] | None = None,
    model_colors: dict[str, str | int] | None = None,
    style_path: str | Path | None = None,
    show_legend: bool = True,
    noise_type: NoiseType = "gaussian",
):
    if sigma_values is None:
        sigma_values = [s / 255.0 for s in range(5, 100, 10)]
    sigma_values = [float(value) for value in sigma_values]

    if training_sigma is not None:
        lo8, hi8 = training_sigma
        targets = {float(lo8) / 255.0, float(hi8) / 255.0}
        for target in targets:
            if not any(
                math.isclose(target, sigma_value, rel_tol=1e-6, abs_tol=1e-8)
                for sigma_value in sigma_values
            ):
                sigma_values.append(target)
        sigma_values = sorted(sigma_values)

    x_vals, per_model_curves, label_order, input_ssim_vals = compute_ssim_io(
        models=models,
        data_dirs=data_dirs,
        sigma_values=sigma_values,
        n_averages=n_averages,
        device=device,
        dataset_mode=dataset_mode,
        max_inference_batch=max_inference_batch,
        max_images=max_images,
        models_names=models_names,
        noise_type=noise_type,
    )

    color_overrides = _merge_color_overrides(
        models_names if isinstance(models_names, Mapping) else None, model_colors
    )
    training_sigma_ssim = resolve_training_sigma_ssim(
        training_sigma=training_sigma,
        sigma_values=sigma_values,
        input_ssim_vals=list(input_ssim_vals),
    )

    fig = plot_ssim_curves(
        x_vals=x_vals,
        per_model_curves=per_model_curves,
        label_order=label_order,
        training_sigma_ssim=training_sigma_ssim,
        save_path=save_path,
        style_path=style_path,
        model_colors=color_overrides,
        show_legend=show_legend,
    )

    x_ret, y_return, auc_return = _summarize_curves(
        x_vals=x_vals,
        per_model_curves=per_model_curves,
        label_order=label_order,
    )
    return x_ret, y_return, fig, auc_return
