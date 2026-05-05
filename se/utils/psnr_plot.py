import math
import os
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
from torch import nn
import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm

from se.configs import PROJECT_ROOT, NoiseType
from se.utils.noise_model import get_noise


# ----------------------- helpers -----------------------
def to_tensor01(img: np.ndarray, mode: str = "m") -> torch.Tensor:
    # img is a numpy HWC uint8 array loaded by cv2.imread (BGR)
    if img.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape {img.shape}.")

    if img.ndim == 3:
        mode = mode.lower()
        if mode == "rgb":
            y = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == "m":
            y = img[..., 0]  # channel 0 (blue)
        elif mode in {"gray", "h"}:
            y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(
                f"Unsupported mode '{mode}'. Use 'm', 'h', 'gray', or 'rgb'."
            )
    else:
        if mode.lower() == "rgb":
            y = np.stack([img, img, img], axis=-1)
        else:
            y = img  # already HxW

    y = torch.from_numpy(y.astype("float32")) / 255.0
    y = y.clamp(0.0, 1.0)

    if y.ndim == 2:
        return y.unsqueeze(0).unsqueeze(0)
    return y.permute(2, 0, 1).unsqueeze(0)


def mse_per_image(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x,y: [B,1,H,W] or [B,C,H,W]
    return torch.mean((x - y) ** 2, dim=(1, 2, 3))  # per-image MSE, shape [B]


def psnr_from_mse(mse: torch.Tensor, data_range=1.0) -> torch.Tensor:
    data_range_tensor = torch.as_tensor(data_range, dtype=mse.dtype, device=mse.device)
    return 20 * torch.log10(data_range_tensor) - 10 * torch.log10(mse)


DEFAULT_PSNR_SIGMA_VALUES = tuple(s / 255.0 for s in range(5, 100, 10))


def default_psnr_sigma_values() -> list[float]:
    return [float(s) for s in DEFAULT_PSNR_SIGMA_VALUES]


def normalize_sigma_values(values: Sequence[float]) -> list[float]:
    normalized: list[float] = []
    for value in values:
        sigma = float(value)
        normalized.append(sigma / 255.0 if sigma > 1.0 else sigma)
    return normalized


def _sorted_unique_positive_sigmas(values: Sequence[float]) -> list[float]:
    unique_sigmas: list[float] = []
    finite_sigmas = sorted(
        float(value)
        for value in values
        if math.isfinite(float(value)) and float(value) > 0.0
    )
    for sigma in finite_sigmas:
        if unique_sigmas and math.isclose(
            sigma, unique_sigmas[-1], rel_tol=1e-6, abs_tol=1e-8
        ):
            continue
        unique_sigmas.append(sigma)
    return unique_sigmas


def augment_sigma_values_with_training_sigma(
    sigma_values: Sequence[float],
    training_sigma: tuple[float, float] | None,
) -> list[float]:
    augmented = [float(s) for s in sigma_values]
    if training_sigma is None:
        return augmented

    eval_sigmas = _sorted_unique_positive_sigmas(augmented)
    if not eval_sigmas:
        return augmented

    lo8, hi8 = training_sigma
    lo_sigma, hi_sigma = sorted((float(lo8) / 255.0, float(hi8) / 255.0))
    domain_lo, domain_hi = eval_sigmas[0], eval_sigmas[-1]
    for target in (lo_sigma, hi_sigma):
        if target <= 0.0:
            continue
        if target < domain_lo or target > domain_hi:
            continue
        if any(
            math.isclose(target, sigma, rel_tol=1e-6, abs_tol=1e-8)
            for sigma in augmented
        ):
            continue
        augmented.append(target)
    return sorted(float(s) for s in augmented)


def sigma_to_input_psnr(s):
    # For x in [0,1] with AWGN std s
    return 20.0 * math.log10(1.0 / s)


def resolve_training_sigma_eval_grid(
    training_sigma: tuple[float, float] | None,
    sigma_values: Sequence[float],
) -> list[float]:
    eval_sigmas = _sorted_unique_positive_sigmas(sigma_values)
    if not eval_sigmas:
        return []
    if training_sigma is None:
        return eval_sigmas

    lo8, hi8 = training_sigma
    lo_sigma, hi_sigma = sorted((float(lo8) / 255.0, float(hi8) / 255.0))
    domain_lo, domain_hi = eval_sigmas[0], eval_sigmas[-1]
    clipped_lo = min(max(lo_sigma, domain_lo), domain_hi)
    clipped_hi = min(max(hi_sigma, domain_lo), domain_hi)
    if clipped_lo > clipped_hi:
        clipped_lo, clipped_hi = clipped_hi, clipped_lo

    targets = [
        sigma
        for sigma in eval_sigmas
        if clipped_lo - 1e-8 <= sigma <= clipped_hi + 1e-8
    ]
    targets.extend([clipped_lo, clipped_hi])
    return _sorted_unique_positive_sigmas(targets)


def _interpolate_metric_at_sigma(
    target_sigma: float,
    sigma_values: Sequence[float],
    metric_values: Sequence[float],
) -> float | None:
    if len(sigma_values) != len(metric_values):
        return None

    pairs = sorted(
        (
            float(sigma),
            float(metric),
        )
        for sigma, metric in zip(sigma_values, metric_values)
        if math.isfinite(float(sigma))
        and float(sigma) > 0.0
        and math.isfinite(float(metric))
    )
    if not pairs:
        return None

    clipped_target = min(max(float(target_sigma), pairs[0][0]), pairs[-1][0])
    for sigma, metric in pairs:
        if math.isclose(clipped_target, sigma, rel_tol=1e-6, abs_tol=1e-8):
            return metric

    for idx in range(1, len(pairs)):
        left_sigma, left_metric = pairs[idx - 1]
        right_sigma, right_metric = pairs[idx]
        if clipped_target > right_sigma + 1e-8:
            continue
        if math.isclose(left_sigma, right_sigma, rel_tol=1e-6, abs_tol=1e-8):
            return right_metric
        weight = (clipped_target - left_sigma) / (right_sigma - left_sigma)
        return float(left_metric + weight * (right_metric - left_metric))

    return float(pairs[-1][1])


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


def resolve_training_sigma_psnr(
    training_sigma: tuple[float, float] | None,
    sigma_values: Sequence[float],
    input_psnr_vals: Sequence[float],
) -> float | tuple[float, float] | None:
    """
    Pick the measured input PSNR(s) corresponding to the training sigma values.
    Returns a single value for a fixed sigma, a (low, high) tuple for a range,
    or None if no match is found.
    """
    if training_sigma is None:
        return None

    target_sigmas = resolve_training_sigma_eval_grid(training_sigma, sigma_values)
    if not target_sigmas:
        return None

    boundary_sigmas = [target_sigmas[0]]
    if not math.isclose(
        target_sigmas[0], target_sigmas[-1], rel_tol=1e-6, abs_tol=1e-8
    ):
        boundary_sigmas.append(target_sigmas[-1])

    resolved_psnrs: list[float] = []
    for target_sigma in boundary_sigmas:
        target_psnr = _interpolate_metric_at_sigma(
            target_sigma=target_sigma,
            sigma_values=sigma_values,
            metric_values=input_psnr_vals,
        )
        if target_psnr is None:
            return None
        resolved_psnrs.append(float(target_psnr))

    if len(resolved_psnrs) == 1 or math.isclose(
        resolved_psnrs[0], resolved_psnrs[-1], rel_tol=1e-6, abs_tol=1e-8
    ):
        return resolved_psnrs[0]
    return (resolved_psnrs[0], resolved_psnrs[-1])


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


def _accumulate_outputs(
    noisy_batch: torch.Tensor,
    clean_batch: torch.Tensor,
    named_models: Sequence[tuple[str, nn.Module]],
    log_mean_mse: bool,
) -> list[torch.Tensor]:
    per_model_outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for _, mdl in named_models:
            denoised_batch = mdl(noisy_batch).clamp(0.0, 1.0)
            output_mse = mse_per_image(clean_batch, denoised_batch)
            if log_mean_mse:
                per_model_outputs.append(output_mse.double())
            else:
                per_model_outputs.append(psnr_from_mse(output_mse).double())
    return per_model_outputs


def _accumulate_sigma_sums(
    noisy_batch: torch.Tensor,
    clean_batch: torch.Tensor,
    sigma_indices: torch.Tensor,
    num_sigmas: int,
    named_models: Sequence[tuple[str, nn.Module]],
    log_mean_mse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate per-sigma sums for inputs and each model output."""
    device = noisy_batch.device
    sigma_indices = sigma_indices.to(device)

    input_mse = mse_per_image(clean_batch, noisy_batch)
    if log_mean_mse:
        input_metric = input_mse.double()
    else:
        input_metric = psnr_from_mse(input_mse).double()

    input_acc = torch.zeros(num_sigmas, dtype=torch.float64, device=device)
    input_acc.index_add_(0, sigma_indices, input_metric)

    outputs_acc = torch.zeros(
        (len(named_models), num_sigmas), dtype=torch.float64, device=device
    )
    per_model_outputs = _accumulate_outputs(
        noisy_batch, clean_batch, named_models, log_mean_mse
    )
    for model_idx, output_metric in enumerate(per_model_outputs):
        outputs_acc[model_idx].index_add_(0, sigma_indices, output_metric)

    return input_acc, outputs_acc


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



def compute_psnr_io(
    models: nn.Module | Sequence[nn.Module],
    data_dirs: list[str] | None = None,
    sigma_values: (
        Sequence[float] | None
    ) = None,  # list[float] in [0,1]; default = [5,15,...,95]/255
    n_averages: int = 10,  # Monte-Carlo repeats per σ
    device: str = "cuda",
    dataset_mode: str = "m",
    log_mean_mse: bool = False,
    max_inference_batch: int | None = None,
    max_images: int | None = None,
    models_names: Sequence[str] | Mapping[str, str | int] | None = None,
    noise_type: NoiseType = "gaussian",
) -> tuple[np.ndarray, list[np.ndarray], list[str], np.ndarray]:
    """
    Compute input/output PSNR curves for one or more denoisers.

    Assumptions:
      - models: sequence (or single) denoiser mapping [B,C,H,W] in [0,1] -> [B,C,H,W]
      - data_dir: folder of images (BSD68-style). Images are converted according to dataset_mode.
      - max_inference_batch: optional cap on how many noisy samples are denoised at once.
    noise_type controls the sampling distribution (gaussian/laplace/uniform/rayleigh).
    """

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

    # -------------------- data --------------------
    files: list[Path] = []
    for data_dir in data_dirs:
        p = Path(data_dir)
        files_ = sorted(
            [
                f
                for f in p.iterdir()
                if f.suffix.lower()
                in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            ]
        )
        if max_images is not None:
            files_ = files_[:max_images]
        files.extend(files_)

    if len(files) == 0:
        raise FileNotFoundError(f"no images in {data_dirs}")

    imgs = [to_tensor01(cv2.imread(f.as_posix()), mode=dataset_mode) for f in files]

    # -------------------- sweep --------------------
    resolved_noise = str(noise_type).lower()

    def _compress_jpeg_tensor(img: torch.Tensor, quality: int) -> torch.Tensor:
        """Compress/decompress a [1,1,H,W] tensor in [0,1] with JPEG at given quality."""
        img_np = img.squeeze().cpu().numpy()
        img_uint8 = np.clip(np.round(img_np * 255.0), 0, 255).astype(np.uint8)
        success, encoded = cv2.imencode(
            ".jpg", img_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
        if not success:
            raise RuntimeError(f"JPEG encoding failed at quality={quality}.")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            raise RuntimeError(f"JPEG decoding failed at quality={quality}.")
        decoded_f = decoded.astype("float32") / 255.0
        return torch.from_numpy(decoded_f).unsqueeze(0).unsqueeze(0)

    if resolved_noise == "jpeg":
        qualities = (
            [int(round(q)) for q in sigma_values]
            if sigma_values is not None
            else list(range(5, 86, 5))
        )
        if len(qualities) == 0:
            raise ValueError("At least one JPEG quality factor is required.")

        device_t = torch.device(device)
        for _, mdl in named_models:
            mdl.to(device_t).eval()

        sum_input_metric = torch.zeros(len(qualities), dtype=torch.float64, device=device_t)
        sum_output_metric = torch.zeros(
            (len(named_models), len(qualities)), dtype=torch.float64, device=device_t
        )

        for img in tqdm(imgs):
            clean_cpu = img  # keep CPU for JPEG encode/decode
            clean = clean_cpu.to(device_t, non_blocking=True)

            per_img_input = torch.zeros(len(qualities), dtype=torch.float64, device=device_t)
            per_img_output = torch.zeros(
                (len(named_models), len(qualities)), dtype=torch.float64, device=device_t
            )

            for q_idx, q in enumerate(qualities):
                noisy_cpu = _compress_jpeg_tensor(clean_cpu, q)
                noisy = noisy_cpu.to(device_t, non_blocking=True)

                input_mse = mse_per_image(clean, noisy)
                if log_mean_mse:
                    per_img_input[q_idx] = input_mse.double().mean()
                else:
                    per_img_input[q_idx] = psnr_from_mse(input_mse).double().mean()

                outputs = _accumulate_outputs(
                    noisy_batch=noisy,
                    clean_batch=clean,
                    named_models=named_models,
                    log_mean_mse=log_mean_mse,
                )
                for model_idx, output_metric in enumerate(outputs):
                    per_img_output[model_idx, q_idx] = output_metric.double().mean()

            sum_input_metric += per_img_input
            sum_output_metric += per_img_output

        divisor = float(len(imgs))
        if log_mean_mse:
            _input_psnr = (
                psnr_from_mse(sum_input_metric / divisor)
                .to("cpu")
                .numpy()
                .astype(np.float64)
            )
            per_model_curves = [
                psnr_from_mse((sum_output_metric[idx] / divisor))
                .to("cpu")
                .numpy()
                .astype(np.float64)
                for idx in range(len(named_models))
            ]
        else:
            _input_psnr = (
                sum_input_metric / divisor
            ).to("cpu").numpy().astype(np.float64)
            per_model_curves = [
                (sum_output_metric[idx] / divisor).to("cpu").numpy().astype(np.float64)
                for idx in range(len(named_models))
            ]

        # For JPEG plots, use quality factors on the x-axis; keep input PSNR for identity.
        x_vals = np.array(qualities, dtype=np.float64)
        input_psnr_vals = _input_psnr
        label_order = [label for label, _ in named_models]
        return x_vals, per_model_curves, label_order, input_psnr_vals

    if sigma_values is None:
        sigma_values = [s / 255.0 for s in range(5, 100, 10)]  # 5,15,...,95
    sigma_values = list(float(s) for s in sigma_values)
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

    sum_input_metric = torch.zeros(len(sigma_values), dtype=torch.float64, device=device_t)
    sum_output_metric = torch.zeros(
        (len(named_models), len(sigma_values)), dtype=torch.float64, device=device_t
    )

    repeats_float = float(n_averages)
    for img in tqdm(imgs):
        clean = img.to(device_t, non_blocking=True)
        per_img_input = torch.zeros(len(sigma_tensor), dtype=torch.float64, device=device_t)
        per_img_output = torch.zeros(
            (len(named_models), len(sigma_tensor)), dtype=torch.float64, device=device_t
        )
        for sigma_idx, sigma_value in enumerate(sigma_tensor):
            sigma_std = float(sigma_value)
            sigma_input_sum = torch.zeros((), dtype=torch.float64, device=device_t)
            sigma_output_sum = torch.zeros(
                len(named_models), dtype=torch.float64, device=device_t
            )
            repeats_done = 0

            while repeats_done < n_averages:
                current_batch = min(max_batch, n_averages - repeats_done)
                while True:
                    batch_clean = None
                    noise = None
                    noisy_batch = None
                    sigma_indices = None
                    try:
                        batch_clean = clean.expand(current_batch, -1, -1, -1)
                        noise = get_noise(
                            batch_clean,
                            min_noise=sigma_std,
                            max_noise=sigma_std,
                            noise_type=noise_type,  # type: ignore
                        )
                        noisy_batch = batch_clean + noise

                        sigma_indices = torch.full(
                            (current_batch,),
                            sigma_idx,
                            device=device_t,
                            dtype=torch.long,
                        )
                        input_acc, output_acc = _accumulate_sigma_sums(
                            noisy_batch,
                            batch_clean,
                            sigma_indices=sigma_indices,
                            num_sigmas=len(sigma_tensor),
                            named_models=named_models,
                            log_mean_mse=log_mean_mse,
                        )
                        break
                    except torch.OutOfMemoryError:
                        if device_t.type != "cuda" or current_batch <= 1:
                            raise
                        if not oom_auto_reduced:
                            print(
                                "compute_psnr_io: CUDA OOM during evaluation; "
                                "reducing inference batch size automatically."
                            )
                            oom_auto_reduced = True
                        del batch_clean, noise, noisy_batch, sigma_indices
                        torch.cuda.empty_cache()
                        current_batch = max(1, current_batch // 2)

                sigma_input_sum += input_acc[sigma_idx]
                sigma_output_sum += output_acc[:, sigma_idx]

                repeats_done += current_batch

            per_img_input[sigma_idx] = sigma_input_sum / repeats_float
            per_img_output[:, sigma_idx] = sigma_output_sum / repeats_float

        sum_input_metric += per_img_input
        sum_output_metric += per_img_output

    divisor = float(len(imgs))
    if log_mean_mse:
        mean_input_mse = sum_input_metric / divisor
        x_vals = psnr_from_mse(mean_input_mse).to("cpu").numpy().astype(np.float64)
        per_model_curves = [
            psnr_from_mse((sum_output_metric[idx] / divisor))
            .to("cpu")
            .numpy()
            .astype(np.float64)
            for idx in range(len(named_models))
        ]
    else:
        x_vals = (sum_input_metric / divisor).to("cpu").numpy().astype(np.float64)
        per_model_curves = [
            (sum_output_metric[idx] / divisor).to("cpu").numpy().astype(np.float64)
            for idx in range(len(named_models))
        ]

    label_order = [label for label, _ in named_models]
    input_psnr_vals = np.array(x_vals, dtype=np.float64)
    return x_vals, per_model_curves, label_order, input_psnr_vals


def plot_psnr_curves(
    x_vals: Sequence[float] | np.ndarray,
    per_model_curves: Sequence[Sequence[float] | np.ndarray],
    label_order: Sequence[str],
    training_sigma=None,  # tuple[int,int] in 8-bit units, e.g. (0, 55)
    training_sigma_psnr: float | tuple[float, float] | None = None,
    training_x_marker: float | tuple[float, float] | None = None,
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
    x_scale: str = "linear",
    legend_loc: str = "lower right",
    legend_bbox_to_anchor: tuple[float, float] | None = (1.01, -0.02),
):
    """Plot pre-computed PSNR curves."""
    color_overrides = _merge_color_overrides(None, model_colors)

    style_file = (
        Path(style_path)
        if style_path is not None
        else PROJECT_ROOT / Path("icml_like.mplstyle")
    )
    style_ctx: Any = (
        plt.style.context(str(style_file)) if style_file.is_file() else nullcontext()
    )
    with style_ctx:
        fig, ax = plt.subplots(figsize=(5.0, 3.8))

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

        for label, curve in zip(label_order, per_model_curves):
            color = _resolve_color(label, color_overrides, color_cycle)
            ax.plot(x_vals, curve, marker="o", linewidth=1.8, label=label, color=color)
            ax.scatter(x_vals, curve, color=color, s=18, zorder=8, label="_nolegend_")

        x_vals_array = np.array(x_vals, dtype=np.float64)
        finite_x_vals = x_vals_array[np.isfinite(x_vals_array)]
        if x_limits is None:
            if finite_x_vals.size == 0:
                x_min, x_max = 0.0, 1.0
            else:
                x_min = float(np.nanmin(finite_x_vals))
                x_max = float(np.nanmax(finite_x_vals))
        else:
            x_min, x_max = float(x_limits[0]), float(x_limits[1])

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
                line_x, line_x, linestyle="--", color="k", linewidth=1.2, label="identity"
            )

        marker_x = training_x_marker
        if marker_x is None:
            marker_x = training_sigma_psnr
        if marker_x is not None:
            if isinstance(marker_x, tuple):
                left, right = sorted(marker_x)
                if math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-8):
                    ax.axvline(left, color="#69a8d8", linewidth=3.0, alpha=0.8)
                else:
                    ax.axvspan(left, right, alpha=0.18, color="#69a8d8")
            else:
                ax.axvline(marker_x, color="#69a8d8", linewidth=3.0, alpha=0.8)
        elif training_sigma is not None and x_scale == "linear":
            # Fallback to analytic mapping when no measured PSNR is available.
            lo8, hi8 = float(training_sigma[0]), float(training_sigma[1])
            lo8, hi8 = (min(lo8, hi8), max(lo8, hi8))
            lo_sigma = max(lo8 / 255.0, 1e-6)
            hi_sigma = max(hi8 / 255.0, 1e-6)
            if math.isclose(lo8, hi8):
                sigma_psnr = sigma_to_input_psnr(lo_sigma)
                ax.axvline(
                    sigma_psnr,
                    color="#69a8d8",
                    linewidth=3.0,
                    alpha=0.8,
                )
            else:
                left = sigma_to_input_psnr(hi_sigma)
                right = sigma_to_input_psnr(lo_sigma)
                if left > right:
                    left, right = right, left
                ax.axvspan(left, right, alpha=0.18, color="#69a8d8")

        ax.set_xlabel(x_label or "input PSNR (dB)")
        ax.set_ylabel("output PSNR (dB)")

        if x_scale == "log":
            ax.set_xscale("log")
            if x_ticks is not None:
                ax.set_xticks(x_ticks)
            elif finite_x_vals.size > 0:
                ax.set_xticks(np.unique(finite_x_vals))
            if x_ticks_minor is not None:
                ax.set_xticks(x_ticks_minor, minor=True)
        else:
            if x_ticks is not None:
                ax.set_xticks(x_ticks)
            elif show_identity:
                ax.set_xticks(np.arange(8, 37, 4))
            else:
                x_lo = math.floor(x_min / 10.0) * 10
                x_hi = math.ceil(x_max / 10.0) * 10
                ax.set_xticks(np.arange(x_lo, x_hi + 1, 10))

            if x_ticks_minor is not None:
                ax.set_xticks(x_ticks_minor, minor=True)
            elif show_identity:
                ax.set_xticks(np.arange(8, 37, 1), minor=True)
            else:
                x_lo = math.floor(x_min / 10.0) * 10
                x_hi = math.ceil(x_max / 10.0) * 10
                ax.set_xticks(np.arange(x_lo, x_hi + 1, 5), minor=True)

        ax.set_yticks(np.arange(8, 37, 4))
        ax.set_yticks(np.arange(8, 37, 1), minor=True)
        ax.xaxis.set_major_formatter(
            ScalarFormatter(useOffset=False, useMathText=False)
        )
        ax.yaxis.set_major_formatter(
            ScalarFormatter(useOffset=False, useMathText=False)
        )

        ax.grid(
            True,
            which="major",
            axis="both",
            # linestyle="--",
            linestyle="-",
            linewidth=0.45,
            alpha=0.35,
            color="0.65",
        )
        ax.grid(True, which="minor", linewidth=0.25, alpha=0.2, color="0.82")
        legend = None
        if show_legend:
            legend_kwargs: dict[str, Any] = {"loc": legend_loc}
            if legend_bbox_to_anchor is not None:
                legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
            legend = ax.legend(
                frameon=True,
                facecolor="1.0",
                edgecolor="0.4",
                framealpha=1.0,
                borderpad=0.35,
                handlelength=1.6,
                **legend_kwargs,
            )
            for line in legend.get_lines():
                line.set_linewidth(2.2)
                if line.get_label() != "identity":
                    line.set_marker("o")
                    line.set_markersize(6)
        if x_limits is not None:
            ax.set_xlim(x_limits)
        else:
            if x_scale == "log":
                if finite_x_vals.size > 0:
                    ax.set_xlim(float(np.nanmin(finite_x_vals)), float(np.nanmax(finite_x_vals)))
            else:
                ax.set_xlim(6, 38)
        if y_limits is not None:
            ax.set_ylim(y_limits)
        else:
            ax.set_ylim(6, 38)
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
    sigma_values: Sequence[float] | None = None,
    auc_sigma_values: Sequence[float] | None = None,
):
    x_vals_np = np.array(x_vals, dtype=np.float64)
    y_arrays = [np.array(curve, dtype=np.float64) for curve in per_model_curves]
    sigma_vals_np = (
        np.array(sigma_values, dtype=np.float64) if sigma_values is not None else None
    )
    auc_sigma_set = (
        np.array(auc_sigma_values, dtype=np.float64)
        if auc_sigma_values is not None
        else None
    )
    auc_values: list[float] = []
    for y_arr in y_arrays:
        finite_mask = np.isfinite(x_vals_np) & np.isfinite(y_arr)
        if sigma_vals_np is not None and auc_sigma_set is not None:
            sigma_mask = np.array(
                [
                    any(
                        math.isclose(
                            float(sigma),
                            float(target_sigma),
                            rel_tol=1e-6,
                            abs_tol=1e-8,
                        )
                        for target_sigma in auc_sigma_set
                    )
                    for sigma in sigma_vals_np
                ],
                dtype=bool,
            )
            finite_mask &= sigma_mask
        if np.count_nonzero(finite_mask) < 2:
            auc_values.append(float("nan"))
            continue
        x_finite = x_vals_np[finite_mask]
        y_finite = y_arr[finite_mask]
        order = np.argsort(x_finite)
        auc_values.append(float(abs(np.trapezoid(y_finite[order], x_finite[order]))))

    if len(label_order) == 1:
        y_return = y_arrays[0]
        auc_return = auc_values[0]
    else:
        y_return = {label: y_arr for label, y_arr in zip(label_order, y_arrays)}
        auc_return = {label: auc for label, auc in zip(label_order, auc_values)}
    return x_vals_np, y_return, auc_return


def plot_psnr_io(
    models,
    data_dirs: list[str] | None = None,
    sigma_values=None,  # list[float] in [0,1]; default = [5,15,...,95]/255
    n_averages=10,  # Monte-Carlo repeats per σ
    training_sigma=None,  # tuple[int,int] in 8-bit units, e.g. (0, 55)
    device="cuda",
    save_path=None,
    dataset_mode: str = "m",
    log_mean_mse: bool = False,
    max_inference_batch: int | None = None,
    max_images: int | None = None,
    models_names: Sequence[str] | Mapping[str, str | int] | None = None,
    model_colors: dict[str, str | int] | None = None,
    style_path: str | Path | None = None,
    show_legend: bool = True,
    noise_type: NoiseType = "gaussian",
    x_axis: str = "input_psnr",
):
    """
    Convenience wrapper: compute PSNR curves and plot them.
    """
    resolved_noise = str(noise_type).lower()
    if sigma_values is None:
        if resolved_noise == "jpeg":
            sigma_values = list(range(5, 86, 5))
        else:
            sigma_values = default_psnr_sigma_values()
    sigma_values = [float(s) for s in sigma_values]
    if resolved_noise != "jpeg":
        sigma_values = normalize_sigma_values(sigma_values)
        sigma_values = augment_sigma_values_with_training_sigma(
            sigma_values, training_sigma
        )
    if x_axis not in {"input_psnr", "sigma"}:
        raise ValueError(f"Unsupported x_axis '{x_axis}'.")
    if resolved_noise == "jpeg" and x_axis != "input_psnr":
        raise ValueError("x_axis='sigma' is only supported for additive-noise plots.")

    x_vals, per_model_curves, label_order, input_psnr_vals = compute_psnr_io(
        models=models,
        data_dirs=data_dirs,
        sigma_values=sigma_values,
        n_averages=n_averages,
        device=device,
        dataset_mode=dataset_mode,
        log_mean_mse=log_mean_mse,
        max_inference_batch=max_inference_batch,
        max_images=max_images,
        models_names=models_names,
        noise_type=noise_type,
    )

    color_overrides = _merge_color_overrides(
        models_names if isinstance(models_names, Mapping) else None, model_colors
    )

    training_sigma_psnr = resolve_training_sigma_psnr(
        training_sigma=training_sigma,
        sigma_values=sigma_values,
        input_psnr_vals=list(input_psnr_vals),
    )
    training_x_marker = None
    plot_x_vals = x_vals
    x_label = None
    show_identity = True
    identity_curve = None
    x_ticks = None
    x_scale = "linear"
    x_limits = None
    if x_axis == "sigma":
        plot_x_vals = np.array(sigma_values, dtype=np.float64)
        x_label = r"test $\sigma$"
        show_identity = False
        identity_curve = input_psnr_vals
        x_ticks = list(plot_x_vals)
        x_scale = "log"
        x_limits = (float(np.min(plot_x_vals)), float(np.max(plot_x_vals)))
        if training_sigma is not None:
            lo8, hi8 = training_sigma
            lo_sigma, hi_sigma = sorted((float(lo8) / 255.0, float(hi8) / 255.0))
            if math.isclose(lo_sigma, hi_sigma, rel_tol=1e-6, abs_tol=1e-8):
                training_x_marker = lo_sigma
            else:
                training_x_marker = (lo_sigma, hi_sigma)
    auc_sigma_values = None
    if resolved_noise != "jpeg" and training_sigma is not None:
        lo8, hi8 = training_sigma
        if not math.isclose(float(lo8), float(hi8), rel_tol=1e-6, abs_tol=1e-8):
            auc_sigma_values = resolve_training_sigma_eval_grid(
                training_sigma, sigma_values
            )

    fig = plot_psnr_curves(
        x_vals=plot_x_vals,
        per_model_curves=per_model_curves,
        label_order=label_order,
        training_sigma=training_sigma,
        training_sigma_psnr=training_sigma_psnr,
        training_x_marker=training_x_marker,
        save_path=save_path,
        style_path=style_path,
        model_colors=color_overrides,
        show_legend=show_legend,
        x_label=x_label,
        show_identity=show_identity,
        identity_curve=identity_curve,
        x_ticks=x_ticks,
        x_limits=x_limits,
        x_scale=x_scale,
    )

    x_ret, y_return, auc_return = _summarize_curves(
        x_vals=plot_x_vals,
        per_model_curves=per_model_curves,
        label_order=label_order,
        sigma_values=sigma_values,
        auc_sigma_values=auc_sigma_values,
    )

    return x_ret, y_return, fig, auc_return
