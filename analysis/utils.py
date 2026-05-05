"""Shared helpers for the SNR hypothesis scripts: dataset context, loader prep, normalization utilities, caching helpers, and small statistical routines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from se.configs import PROJECT_ROOT, DatasetConfig, NoiseType, TrainConfig, WrapperMode
from se.data import build_loaders
from se.utils.noise_model import get_noise

EPS = 1e-6
VALID_WRAPPER_MODES: set[str] = {
    "idem",
    "scale-equiv",
    "norm-equiv",
    "norm-equiv-input",
}


@dataclass
class AxisContext:
    label: str
    symbol: str
    bin_range: tuple[float, float]
    default_xlim: tuple[float, float]
    prefix: str


@dataclass
class SigmaResults:
    sigma: float
    delta: np.ndarray
    residual: np.ndarray
    snr: np.ndarray  # local per-sample SNR


def format_sigma_label(sigma: float) -> str:
    return f"{sigma:g}".replace(".", "p")


def overlay_subdir(save_dir: Path, sigmas: list[float]) -> Path:
    if not sigmas:
        return save_dir / "noise_overlay"
    unique_sorted = sorted(set(sigmas))
    label = "_".join(format_sigma_label(s) for s in unique_sorted)
    return save_dir / f"noise_overlay_{label}"


def sigma_cache_path(root: Path, sigma: float) -> Path:
    return root / f"sigma_{format_sigma_label(sigma)}.npz"


def save_sigma_results(result: SigmaResults, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        sigma=result.sigma,
        delta=result.delta,
        residual=result.residual,
        snr=result.snr,
    )


def load_sigma_results(cache_path: Path) -> SigmaResults:
    with np.load(cache_path, allow_pickle=False) as data:
        sigma_value = float(np.atleast_1d(data["sigma"]).astype(np.float64)[0])
        return SigmaResults(
            sigma=sigma_value,
            delta=data["delta"],
            residual=data["residual"],
            snr=data["snr"],
        )


def build_train_loader(cfg: TrainConfig | DatasetConfig) -> DataLoader:
    train_loader, _ = build_loaders(cfg)
    return train_loader


def prepare_batch(
    batch: torch.Tensor,
    min_noise: float,
    max_noise: float,
    device: torch.device,
    noise_type: NoiseType = "gaussian",
) -> tuple[torch.Tensor, torch.Tensor]:
    clean = batch.to(device)
    min_noise = min_noise / 255.0
    max_noise = max_noise / 255.0
    noise = get_noise(
        clean,
        min_noise=min_noise,
        max_noise=max_noise,
        noise_type=noise_type,
    )
    noisy = clean + noise
    return clean, noisy


def resolve_noise_list(cfg: TrainConfig, provided: list[float] | None) -> list[float]:
    if provided:
        return list(provided)
    if cfg.min_noise == cfg.max_noise:
        return [cfg.min_noise]
    return [cfg.min_noise, cfg.max_noise]


def resolve_analysis_wrapper(
    train_cfg: TrainConfig, override: WrapperMode | None
) -> WrapperMode:
    candidate: str
    if override is not None:
        candidate = override
    else:
        candidate = getattr(train_cfg.model_cfg, "wrapper_mode", "idem")
    normalized = candidate.lower()
    if normalized not in VALID_WRAPPER_MODES:
        raise ValueError(
            f"Unsupported wrapper mode '{candidate}'. Expected one of {sorted(VALID_WRAPPER_MODES)}."
        )
    return cast(WrapperMode, normalized)


def normalize_triplet(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    predicted: torch.Tensor,
    wrapper_mode: WrapperMode,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dims = tuple(range(1, clean.ndim))
    if wrapper_mode in {"norm-equiv", "norm-equiv-input"}:
        shift = noisy.mean(dim=dims, keepdim=True)
        scale = noisy.std(dim=dims, keepdim=True).clamp_min(EPS)
    elif wrapper_mode == "scale-equiv":
        shift = torch.zeros_like(noisy.mean(dim=dims, keepdim=True))
        scale = noisy.pow(2).mean(dim=dims, keepdim=True).pow(0.5).clamp_min(EPS)
    elif wrapper_mode == "idem":
        shift = torch.zeros_like(noisy.mean(dim=dims, keepdim=True))
        scale = torch.ones_like(shift)
    else:  # pragma: no cover - safeguarded by validation
        raise ValueError(f"Unexpected wrapper mode '{wrapper_mode}'.")

    tilde_y = (noisy - shift) / scale
    tilde_x = (clean - shift) / scale
    tilde_pred = (predicted - shift) / scale
    return tilde_x, tilde_y, tilde_pred


def compute_delta(
    clean: torch.Tensor, noisy: torch.Tensor, wrapper_mode: WrapperMode
) -> torch.Tensor:
    tilde_x, tilde_y, _ = normalize_triplet(clean, noisy, noisy, wrapper_mode)
    return torch.linalg.norm((tilde_y - tilde_x).reshape(tilde_x.shape[0], -1), dim=1)


def aggregate_bins(
    x_values: np.ndarray,
    values: np.ndarray,
    bins: int,
    x_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start, end = x_range
    if not np.isfinite(start):
        start = float(np.nanmin(x_values))
    if not np.isfinite(end):
        end = float(np.nanmax(x_values))
    if end <= start:
        end = start + 1e-6
    bin_edges = np.linspace(start, end, bins + 1)
    indices = np.digitize(x_values, bin_edges) - 1
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    means = np.full_like(centers, np.nan, dtype=np.float64)
    stds = np.full_like(centers, np.nan, dtype=np.float64)
    counts = np.zeros_like(centers, dtype=np.int64)

    for idx in range(bins):
        mask = (indices == idx) & (indices >= 0) & (indices < bins)
        if not np.any(mask):
            continue
        vals = values[mask]
        means[idx] = vals.mean()
        stds[idx] = vals.std()
        counts[idx] = mask.sum()

    valid = ~np.isnan(means)
    return centers[valid], means[valid], stds[valid], counts[valid]


def _percentile_limits(
    values: np.ndarray, percentiles: tuple[float, float] | None
) -> tuple[float, float] | None:
    if percentiles is None:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    low_p, high_p = percentiles
    low = np.percentile(finite, low_p)
    high = np.percentile(finite, high_p)
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if low == high:
        delta = 0.1 * (abs(low) + 1.0)
        return (float(low - delta), float(high + delta))
    pad = 0.1 * (high - low)
    return (float(low - pad), float(high + pad))


def _validate_percentile_tuple(
    percentiles: tuple[float, float] | list[float],
) -> tuple[float, float]:
    if len(percentiles) != 2:
        raise ValueError(
            f"Expected exactly two percentile values, got {len(percentiles)}: {percentiles}"
        )
    low, high = float(percentiles[0]), float(percentiles[1])
    if not (0.0 <= low <= 100.0 and 0.0 <= high <= 100.0):
        raise ValueError(
            f"Percentiles must be in the range [0, 100], got {percentiles}."
        )
    if low > high:
        raise ValueError(
            f"Percentiles must be ordered as (low, high). Got {percentiles}."
        )
    return (low, high)


def _finite_percentile_range(
    values: np.ndarray, percentiles: tuple[float, float]
) -> tuple[float, float] | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    low, high = np.percentile(finite, percentiles)
    low = float(low)
    high = float(high)
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if low == high:
        high = low + 1e-6
    return (low, high)


def _combine_finite_arrays(arrays: list[np.ndarray]) -> np.ndarray | None:
    finite_parts: list[np.ndarray] = []
    for arr in arrays:
        if arr.size == 0:
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            finite_parts.append(finite)
    if not finite_parts:
        return None
    return np.concatenate(finite_parts, axis=0)


def _format_csv_float(value: float, precision: int = 3) -> str:
    if not np.isfinite(value):
        return ""
    return f"{float(value):.{precision}f}"


def _delta_bin_range(
    results: list[SigmaResults], wrapper_mode: WrapperMode
) -> tuple[float, float]:
    combined = _combine_finite_arrays([res.delta for res in results])
    if combined is None:
        return (0.0, 100.0)
    if wrapper_mode != "scale-equiv":
        return (0.0, 100.0)
    low, high = np.percentile(combined, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high):
        return (0.0, 100.0)
    if high <= low:
        high = low + 1e-6
    span = max(high - low, 1e-6)
    pad = 0.05 * span
    start = float(max(0.0, low - pad))
    end = float(high + pad)
    return (start, end)


def resolve_axis_context(
    results: list[SigmaResults], wrapper_mode: WrapperMode
) -> AxisContext:
    bin_range = _delta_bin_range(results, wrapper_mode)
    return AxisContext(
        label=r"$\Delta = \|\tilde{y}-\tilde{x}\|_2$",
        symbol=r"$\Delta$",
        bin_range=bin_range,
        default_xlim=bin_range,
        prefix="delta",
    )
