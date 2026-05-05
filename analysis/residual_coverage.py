"""Tabulate how delta samples from different noise sigmas fall within each other's percentile bands, caching deltas and emitting a coverage CSV for model-free comparisons."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from se.configs import DatasetConfig, PROJECT_ROOT, WrapperMode
from analysis.utils import (
    build_train_loader,
    compute_delta,
    overlay_subdir,
    prepare_batch,
    sigma_cache_path,
    _finite_percentile_range,
    _format_csv_float,
    _validate_percentile_tuple,
)


@dataclass
class ResidualCoverageConfig:
    """Compute delta samples per sigma and write a coverage table (model-free)."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    samples_per_epoch: int = 1_000_000
    batch_size: int = 2**16
    noise_sigmas: Sequence[float] | None = None
    delta_percentiles: tuple[float, float] = (2.5, 97.5)
    save_dir: Path = PROJECT_ROOT / Path("artifacts/snr_hypothesis")
    device: str | None = None
    max_batches: int | None = None
    use_cache: bool = True
    wrapper_mode: WrapperMode = "norm-equiv"


DEFAULT_CONFIG = ResidualCoverageConfig(
    noise_sigmas=[10.0, 20.0, 30.0, 40.0, 50.0],
    delta_percentiles=(2.5, 97.5),
    save_dir=PROJECT_ROOT / Path("artifacts/snr_hypothesis"),
    device="cpu",
    max_batches=None,
    use_cache=True,
    wrapper_mode="norm-equiv",
)


def collect_delta_for_sigma(
    cfg: DatasetConfig,
    device: torch.device,
    sigma: float,
    max_batches: int | None,
    loader: torch.utils.data.DataLoader | None,
    wrapper_mode: WrapperMode,
) -> np.ndarray:
    if loader is None:
        loader = build_train_loader(cfg)
    total_batches = len(loader)
    limit = (
        min(total_batches, max_batches) if max_batches is not None else total_batches
    )

    delta_samples: list[np.ndarray] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= limit:
                break
            clean, noisy = prepare_batch(
                batch,
                min_noise=sigma,
                max_noise=sigma,
                device=device,
                noise_type=cfg.noise_type,
            )
            delta = compute_delta(clean, noisy, wrapper_mode=wrapper_mode)
            delta_samples.append(delta.cpu().numpy())

    if not delta_samples:
        return np.array([], dtype=np.float64)
    return np.concatenate(delta_samples, axis=0)


def _write_coverage_table(
    deltas: list[np.ndarray],
    sigma_values: list[float],
    delta_percentiles: tuple[float, float],
    coverage_table_path: Path,
) -> None:
    num_sigmas = len(sigma_values)
    coverage_matrix = np.full((num_sigmas, num_sigmas), np.nan, dtype=np.float64)
    quantile_intervals: list[tuple[float, float]] = []
    train_sample_counts: list[int] = []
    has_valid_train = False

    for delta_vals in deltas:
        finite_delta = delta_vals[np.isfinite(delta_vals)]
        train_sample_counts.append(int(finite_delta.size))
        percentile_bounds = _finite_percentile_range(finite_delta, delta_percentiles)
        if percentile_bounds is None:
            quantile_intervals.append((float("nan"), float("nan")))
            continue
        q_low, q_high = percentile_bounds
        quantile_intervals.append((q_low, q_high))
        if np.isfinite(q_low) and np.isfinite(q_high):
            has_valid_train = True

    if not has_valid_train:
        print("Skipping coverage table (no finite delta samples).")
        return

    percentile_headers = [f"q{p/100:g}" for p in delta_percentiles]
    for train_idx, (q_low, q_high) in enumerate(quantile_intervals):
        if not np.isfinite(q_low) or not np.isfinite(q_high):
            continue
        for test_idx, delta_vals in enumerate(deltas):
            finite_delta = delta_vals[np.isfinite(delta_vals)]
            if finite_delta.size == 0:
                continue
            in_range = (finite_delta >= q_low) & (finite_delta <= q_high)
            coverage_matrix[train_idx, test_idx] = float(in_range.mean())

    with coverage_table_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        header = (
            ["sigma_train\\sigma_test"]
            + [f"{sigma:g}" for sigma in sigma_values]
            + percentile_headers
            + ["train_samples"]
        )
        writer.writerow(header)
        for idx, sigma_train in enumerate(sigma_values):
            row = [f"{sigma_train:g}"]
            for test_idx in range(num_sigmas):
                row.append(
                    _format_csv_float(coverage_matrix[idx, test_idx], precision=4)
                )
            q_low, q_high = quantile_intervals[idx]
            row.extend(
                [
                    _format_csv_float(q_low, precision=4),
                    _format_csv_float(q_high, precision=4),
                    str(train_sample_counts[idx]),
                ]
            )
            writer.writerow(row)
    print(f"Saved coverage table to {coverage_table_path}")


def main(script_cfg: ResidualCoverageConfig = DEFAULT_CONFIG) -> None:
    start_time = time.time()
    print("Residual coverage config:")
    print(asdict(script_cfg))

    delta_percentiles = _validate_percentile_tuple(script_cfg.delta_percentiles)
    if not script_cfg.noise_sigmas:
        raise ValueError("noise_sigmas must be provided.")
    sigma_values = list(script_cfg.noise_sigmas)

    ds = script_cfg.dataset
    ds.batch_size = script_cfg.batch_size  # type: ignore[attr-defined]
    ds.s_samples_per_epoch = script_cfg.samples_per_epoch  # type: ignore[attr-defined]

    device_str = (
        script_cfg.device
        if script_cfg.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = torch.device(device_str)

    # Folder layout mirrors delta_histogram, with wrapper_mode baked into root.
    base_root = script_cfg.save_dir / f"residual_coverage_{script_cfg.wrapper_mode}"
    cache_root = base_root / "delta_cache"
    overlay_root = overlay_subdir(base_root, sigma_values)
    cache_root.mkdir(parents=True, exist_ok=True)
    overlay_root.mkdir(parents=True, exist_ok=True)

    loader: torch.utils.data.DataLoader | None = None
    delta_arrays: list[np.ndarray] = []
    for sigma in sigma_values:
        cache_path = sigma_cache_path(cache_root, sigma)
        delta_vals: np.ndarray | None = None
        if script_cfg.use_cache and cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    delta_vals = data["delta"]
                print(f"Loaded cached deltas for σ={sigma:g} from {cache_path.name}")
            except Exception as exc:
                print(
                    f"Failed to load cached deltas for σ={sigma:g}: {exc}. Recomputing."
                )
                delta_vals = None
        if delta_vals is None:
            if loader is None:
                loader = build_train_loader(ds)
            delta_vals = collect_delta_for_sigma(
                cfg=ds,
                device=device,
                sigma=sigma,
                max_batches=script_cfg.max_batches,
                loader=loader,
                wrapper_mode=script_cfg.wrapper_mode,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, delta=delta_vals)
            print(f"Saved deltas for σ={sigma:g} to {cache_path.name}")
        delta_arrays.append(delta_vals)
        print(f"Collected {delta_vals.size} delta samples for σ={sigma:g}")

    coverage_table_path = overlay_root / "delta_coverage.csv"
    _write_coverage_table(
        delta_arrays, sigma_values, delta_percentiles, coverage_table_path
    )

    elapsed = time.time() - start_time
    print(f"Total elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
