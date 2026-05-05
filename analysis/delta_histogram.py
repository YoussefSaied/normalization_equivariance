"""Estimate the distribution of normalized clean-noisy gaps (Delta) across noise sigmas, caching delta samples and rendering overlay histograms with KDE curves."""

from __future__ import annotations

import time
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from se.configs import DatasetConfig, PROJECT_ROOT
from analysis.utils import (
    AxisContext,
    build_train_loader,
    overlay_subdir,
    prepare_batch,
)

EPS = 1e-6


@dataclass
class DeltaHistogramConfig:
    """Generate delta histograms; compute and cache deltas if missing."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    samples_per_epoch: int = 1_000_000
    batch_size: int = 2**16
    noise_sigmas: Sequence[float] | None = None
    bins: int = 40
    save_dir: Path = PROJECT_ROOT / Path("artifacts/snr_hypothesis")
    device: str | None = None
    max_batches: int | None = None
    use_cache: bool = True


DEFAULT_CONFIG = DeltaHistogramConfig(
    noise_sigmas=[10.0, 20.0, 30.0, 40.0, 50.0],
    bins=40,
    save_dir=PROJECT_ROOT / Path("artifacts/snr_hypothesis"),
    device="cpu",
    max_batches=None,
    use_cache=True,
)


def _delta_from_batch(
    clean: torch.Tensor,
    noisy: torch.Tensor,
) -> torch.Tensor:
    dims = tuple(range(1, clean.ndim))
    mean_y = noisy.mean(dim=dims, keepdim=True)
    std_y = noisy.std(dim=dims, keepdim=True).clamp_min(EPS)
    tilde_y = (noisy - mean_y) / std_y
    tilde_x = (clean - mean_y) / std_y
    return torch.linalg.norm((tilde_y - tilde_x).reshape(tilde_x.shape[0], -1), dim=1)


def delta_cache_path(root: Path, sigma: float) -> Path:
    label = f"{sigma:g}".replace(".", "p")
    return root / f"sigma_{label}_delta.npz"


def save_delta_cache(delta: np.ndarray, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, delta=delta)


def load_delta_cache(cache_path: Path) -> np.ndarray:
    with np.load(cache_path, allow_pickle=False) as data:
        return data["delta"]


def collect_delta_for_sigma(
    cfg: DatasetConfig,
    device: torch.device,
    sigma: float,
    max_batches: int | None,
    loader: torch.utils.data.DataLoader | None,
) -> np.ndarray:
    if loader is None:
        loader = build_train_loader(cfg)
    total_batches = len(loader)
    limit = (
        min(total_batches, max_batches) if max_batches is not None else total_batches
    )

    delta_samples: list[np.ndarray] = []
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(loader)):
            if batch_idx >= limit:
                break
            clean, noisy = prepare_batch(
                batch,
                min_noise=sigma,
                max_noise=sigma,
                device=device,
                noise_type=cfg.noise_type,
            )
            delta = _delta_from_batch(clean, noisy)
            delta_samples.append(delta.cpu().numpy())

    if not delta_samples:
        return np.array([], dtype=np.float64)
    return np.concatenate(delta_samples, axis=0)


def _determine_bin_range(deltas: list[np.ndarray]) -> tuple[float, float]:
    finite_parts = [arr[np.isfinite(arr)] for arr in deltas if arr.size > 0]
    if not finite_parts:
        return (0.0, 100.0)
    combined = np.concatenate(finite_parts, axis=0)
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


def plot_axis_histogram_overlay(
    deltas: list[np.ndarray],
    sigmas: list[float],
    axis_ctx: AxisContext,
    bins: int,
    save_path: Path,
) -> None:

    style_path = PROJECT_ROOT / Path("icml_like.mplstyle")

    with plt.style.context(str(style_path)):
        # fig, ax = plt.subplots(figsize=(6.9, 3.2), constrained_layout=True)
        h = 4
        aspect_ratio = 4 / 3
        w = h * aspect_ratio
        fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)

        x0, x1 = axis_ctx.bin_range
        x_grid = np.linspace(x0, x1, 500)

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for idx, delta_vals in enumerate(deltas):
            finite_vals = delta_vals[np.isfinite(delta_vals)]
            if finite_vals.size == 0:
                continue

            color = colors[idx % len(colors)]
            label = rf"$\sigma={sigmas[idx]:g}$"

            ax.hist(
                finite_vals,
                bins=bins,
                range=(x0, x1),
                density=True,
                alpha=0.18,
                color=color,
                edgecolor="none",
            )

            kde = gaussian_kde(finite_vals)
            kde_vals = kde(x_grid)
            # enforce zero at the left boundary to avoid bleed into negative region
            kde_vals[x_grid <= axis_ctx.bin_range[0]] = 0.0
            ax.plot(x_grid, kde_vals, color=color, linewidth=2.2, label=label)

        ax.set_xlim(x0, x1)
        ax.set_xlabel(axis_ctx.label)
        ax.set_ylabel("Density")

        ref = 70  # Reference sqrt(d) for 70x70 patches; update if patch size changes.
        # ax.axvline(ref, color="0.4", linestyle="--", linewidth=1.0, label=r"$\sqrt{d}$")
        ax.axvline(
            ref,
            color="0.20",
            linestyle="--",
            dashes=(6, 3),
            linewidth=1.6,
            zorder=10,
            label=r"$\sqrt{d}=70$",
        )

        ax.grid(
            True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.5
        )
        ax.legend(
            loc="upper left",
            ncol=3,
            frameon=True,
            borderaxespad=0.3,
            handlelength=2.2,
        )
        # thick black border and start x at 0
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("black")
        ax.set_xlim(left=0.0, right=x1)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)


def main(script_cfg: DeltaHistogramConfig = DEFAULT_CONFIG) -> None:
    start_time = time.time()
    print("Delta histogram config:")
    print(asdict(script_cfg))

    if not script_cfg.noise_sigmas:
        raise ValueError("noise_sigmas must be provided.")
    sigma_values = list(script_cfg.noise_sigmas)
    # enrich dataset config with noise/batch size for this run
    ds = script_cfg.dataset
    ds.batch_size = script_cfg.batch_size  # type: ignore[attr-defined]

    device_str = (
        script_cfg.device
        if script_cfg.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = torch.device(device_str)

    cache_root = script_cfg.save_dir / "delta_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    hist_root = overlay_subdir(script_cfg.save_dir, sigma_values)
    hist_root.mkdir(parents=True, exist_ok=True)

    ds.s_samples_per_epoch = script_cfg.samples_per_epoch  # type: ignore[attr-defined]
    loader: torch.utils.data.DataLoader | None = None
    delta_arrays: list[np.ndarray] = []
    for sigma in sigma_values:
        cache_path = delta_cache_path(cache_root, sigma)
        if script_cfg.use_cache and cache_path.is_file():
            try:
                delta_vals = load_delta_cache(cache_path)
                print(f"Loaded cached deltas for σ={sigma:g} from {cache_path.name}")
                delta_arrays.append(delta_vals)
                continue
            except Exception as exc:
                print(
                    f"Failed to load cached deltas for σ={sigma:g}: {exc}. Recomputing."
                )
        if loader is None:
            loader = build_train_loader(ds)
        delta_vals = collect_delta_for_sigma(
            cfg=ds,
            device=device,
            sigma=sigma,
            max_batches=script_cfg.max_batches,
            loader=loader,
        )
        # always refresh cache when recomputing
        save_delta_cache(delta_vals, cache_path)
        print(f"Saved deltas for σ={sigma:g} to {cache_path.name}")
        delta_arrays.append(delta_vals)
        print(f"Collected {delta_vals.size} delta samples for σ={sigma:g}")

    bin_range = _determine_bin_range(delta_arrays)
    axis_ctx = AxisContext(
        label=r"$\Delta = \|\tilde{y}-\tilde{x}\|_2$",
        symbol=r"$\Delta$",
        bin_range=bin_range,
        default_xlim=bin_range,
        prefix="delta",
    )

    hist_path_pdf = hist_root / f"{axis_ctx.prefix}_histogram.pdf"
    plot_axis_histogram_overlay(
        delta_arrays,
        sigma_values,
        axis_ctx,
        bins=script_cfg.bins,
        save_path=hist_path_pdf,
    )
    print(f"Saved {axis_ctx.prefix} histogram to {hist_path_pdf}")
    elapsed = time.time() - start_time
    print(f"Total elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
