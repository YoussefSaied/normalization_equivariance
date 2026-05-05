"""Collect delta, residual energy, and SNR statistics for a trained denoiser across noise levels, caching results and plotting overlays for the SNR hypothesis study."""

from __future__ import annotations

import time
import os
from dataclasses import dataclass, asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
from tqdm import tqdm

from se.configs import NoiseType, PROJECT_ROOT, TrainConfig, WrapperMode
from se.models import build_model
from se.utils.eval_utils import load_train_config, resolve_checkpoint_path
from se.utils.train_utils import run_name
from model_logs import models_log
from analysis.utils import (
    AxisContext,
    SigmaResults,
    build_train_loader,
    normalize_triplet,
    overlay_subdir,
    prepare_batch,
    resolve_analysis_wrapper,
    resolve_axis_context,
    resolve_noise_list,
    save_sigma_results,
    load_sigma_results,
    sigma_cache_path,
    aggregate_bins,
    _finite_percentile_range,
    _percentile_limits,
    _validate_percentile_tuple,
    EPS,
)


@dataclass
class ResidualCalcConfig:
    log_dir: Path | None = None
    model_key: str | None = None
    checkpoint: Path | None = None
    epoch: int | None = None
    device: str | None = None
    max_batches: int | None = None
    batch_size: int | None = None
    bins: int = 40
    save_dir: Path = PROJECT_ROOT / Path("artifacts/snr_hypothesis")
    noise_sigmas: list[float] | None = None
    train_sigma: float | None = None
    delta_percentiles: tuple[float, float] = (2.5, 97.5)
    use_cache: bool = True
    plot: bool = True
    residual_xlim: tuple[float, float] | None = None
    residual_ylim: tuple[float, float] | None = (-40.0, 0.0)
    show_legend: bool = True
    wrapper_mode: WrapperMode | None = None


DEFAULT_CONFIG = ResidualCalcConfig(
    log_dir=None,
    model_key="wne_swinir_10",
    checkpoint=None,
    epoch=None,
    device=None,
    max_batches=None,
    batch_size=128,
    bins=40,
    save_dir=PROJECT_ROOT / Path("artifacts/snr_hypothesis"),
    noise_sigmas=[10.0, 20.0, 30.0, 40.0, 50.0],
    train_sigma=10,
    delta_percentiles=(2.5, 97.5),
    use_cache=True,
    plot=True,
    residual_xlim=(10.0, 70.0),
    residual_ylim=(-40.0, 0.0),
    show_legend=False,
    wrapper_mode="norm-equiv",
)


def first_valid(*values) -> None | tuple[float, float]:
    """Return the first non-None value, or None if all are None."""
    for v in values:
        if v is not None:
            return v
    return None


@dataclass
class AxisLimits:
    """Encapsulates x/y axis limit configuration for plotting."""

    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    bin_range: tuple[float, float] | None = None

    def effective_xlim(
        self, fallback: tuple[float, float] | None = None
    ) -> tuple[float, float] | None:
        return first_valid(self.xlim, self.bin_range, fallback)

    def effective_bin_range(
        self, fallback: tuple[float, float] | None = None
    ) -> tuple[float, float] | None:
        return first_valid(self.bin_range, self.xlim, fallback)


@dataclass
class VarianceRatioResult:
    """Variance-ratio summary for residual dependence on σ after conditioning on Δ."""

    vr_sigma: float
    mean_var_across_bins: float
    var_across_bins: float
    bins_used: int
    bins_total: int
    bin_range: tuple[float, float]
    sigmas: list[float]
    per_sigma_bin_counts: dict[float, int]
    per_sigma_sample_counts: dict[float, int]
    per_sigma_mean_residual: dict[float, float]


class LazyLoader:
    """Lazily builds the data loader only when first accessed."""

    def __init__(self, train_cfg: TrainConfig):
        self._train_cfg = train_cfg
        self._loader: torch.utils.data.DataLoader | None = None

    def get(self) -> torch.utils.data.DataLoader:
        if self._loader is None:
            self._loader = build_train_loader(self._train_cfg)
        return self._loader


def compute_metrics(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    predicted: torch.Tensor,
    wrapper_mode: WrapperMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    tilde_x, tilde_y, tilde_pred = normalize_triplet(
        clean, noisy, predicted, wrapper_mode
    )
    delta = torch.linalg.norm((tilde_y - tilde_x).reshape(tilde_x.shape[0], -1), dim=1)
    tilde_residual = tilde_pred - tilde_x
    residual_norm_sq = (
        tilde_residual.reshape(tilde_residual.shape[0], -1).pow(2).sum(dim=1)
    ).clamp_min(EPS)
    residual_energy_db = -10.0 * torch.log10(residual_norm_sq)
    return delta, residual_energy_db


def _bin_means_for_sigma(
    result: SigmaResults, bin_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-bin mean residuals (aligned to provided bin_edges) and counts for a single σ."""
    bins = bin_edges.size - 1
    indices = np.digitize(result.delta, bin_edges) - 1
    means = np.full(bins, np.nan, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.int64)

    finite_mask = np.isfinite(result.delta) & np.isfinite(result.residual)
    indices = indices[finite_mask]
    residual_values = result.residual[finite_mask]

    for idx in range(bins):
        mask = (indices == idx) & (indices >= 0) & (indices < bins)
        if not np.any(mask):
            continue
        vals = residual_values[mask]
        means[idx] = vals.mean()
        counts[idx] = mask.sum()
    return means, counts


def _intersection_percentile_range(
    results: list[SigmaResults], delta_percentiles: tuple[float, float]
) -> tuple[float, float] | None:
    """Compute intersection of per-σ percentile windows [Lk, Uk]."""
    lows: list[float] = []
    highs: list[float] = []
    for res in results:
        rng = _finite_percentile_range(res.delta, delta_percentiles)
        if rng is None:
            continue
        low, high = rng
        lows.append(low)
        highs.append(high)
    if not lows or not highs:
        return None
    low = max(lows)
    high = min(highs)
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if low >= high:
        return None
    return (float(low), float(high))


def compute_sigma_variance_ratio(
    results: list[SigmaResults],
    bins: int,
    bin_range: tuple[float, float],
    delta_percentiles: tuple[float, float],
    window_mode: str = "per-sigma",
    intersection_range: tuple[float, float] | None = None,
) -> VarianceRatioResult | None:
    """
    Compute VR_σ = (1/B) * sum_b Var_k(m_{b,k}) / Var_b( m̄_b ),
    where m_{b,k} is the mean residual in Δ-bin b for σ_k.
    window_mode:
        - "per-sigma": restrict each σ to its own Δ percentiles.
        - "intersection": restrict all σ to the intersection of their percentile windows.
    """
    if not results or bins <= 0:
        return None

    start, end = bin_range
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return None

    window_mode = window_mode.lower()
    if window_mode not in {"per-sigma", "intersection"}:
        raise ValueError(f"Unsupported window_mode '{window_mode}'.")

    if window_mode == "intersection" and intersection_range is None:
        return None

    bin_edges = np.linspace(start, end, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    sigma_list = [float(res.sigma) for res in results]
    bin_means_matrix = np.full((bins, len(results)), np.nan, dtype=np.float64)
    bin_counts_matrix = np.zeros((bins, len(results)), dtype=np.int64)

    for col, res in enumerate(results):
        bin_means, bin_counts = _bin_means_for_sigma(res, bin_edges)

        # Restrict to Δ percentiles (per-σ or intersection).
        if window_mode == "per-sigma":
            percentile_range = _finite_percentile_range(res.delta, delta_percentiles)
        else:
            percentile_range = intersection_range

        if percentile_range is not None:
            low, high = percentile_range
            outside = (bin_centers < low) | (bin_centers > high)
            bin_means[outside] = np.nan
            bin_counts[outside] = 0

        bin_means_matrix[:, col] = bin_means
        bin_counts_matrix[:, col] = bin_counts

    per_bin_sigma_counts = (bin_counts_matrix > 0).sum(axis=1)
    per_bin_var = np.nanvar(bin_means_matrix, axis=1, ddof=0)

    # Require at least two σ contributions in a bin to assess σ-variance.
    per_bin_var[per_bin_sigma_counts < 2] = np.nan
    valid_bins_mask = np.isfinite(per_bin_var)
    bins_used = int(valid_bins_mask.sum())
    if bins_used == 0:
        return None

    mean_var_across_bins = float(np.nanmean(per_bin_var[valid_bins_mask]))

    per_bin_means = np.nanmean(bin_means_matrix, axis=1)
    per_bin_mean_mask = np.isfinite(per_bin_means)
    if per_bin_mean_mask.sum() < 2:
        var_across_bins = float("nan")
    else:
        var_across_bins = float(np.nanvar(per_bin_means[per_bin_mean_mask], ddof=0))

    if np.isfinite(var_across_bins) and var_across_bins > 0.0:
        vr_sigma = float(mean_var_across_bins / var_across_bins)
    else:
        vr_sigma = float("nan")

    per_sigma_bin_counts = {
        float(res.sigma): int(np.count_nonzero(~np.isnan(bin_means_matrix[:, idx])))
        for idx, res in enumerate(results)
    }
    per_sigma_sample_counts = {
        float(res.sigma): int(res.delta.size) for res in results
    }
    per_sigma_mean_residual = {
        float(res.sigma): float(np.nanmean(res.residual))
        for res in results
    }

    return VarianceRatioResult(
        vr_sigma=vr_sigma,
        mean_var_across_bins=mean_var_across_bins,
        var_across_bins=var_across_bins,
        bins_used=bins_used,
        bins_total=bins,
        bin_range=(float(bin_range[0]), float(bin_range[1])),
        sigmas=sigma_list,
        per_sigma_bin_counts=per_sigma_bin_counts,
        per_sigma_sample_counts=per_sigma_sample_counts,
        per_sigma_mean_residual=per_sigma_mean_residual,
    )


def collect_metrics_for_sigma(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    sigma: float,
    max_batches: int | None,
    wrapper_mode: WrapperMode = "norm-equiv",
    noise_type: NoiseType = "gaussian",
) -> SigmaResults:
    total_batches = len(loader)
    limit = (
        min(total_batches, max_batches) if max_batches is not None else total_batches
    )

    delta_samples: list[np.ndarray] = []
    residual_samples: list[np.ndarray] = []
    snr_samples: list[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, batch in tqdm(
            enumerate(loader), total=limit, desc=f"σ={sigma:g}"
        ):
            if batch_idx >= limit:
                break
            clean, noisy = prepare_batch(
                batch,
                min_noise=sigma,
                max_noise=sigma,
                device=device,
                noise_type=noise_type,
            )
            predicted = model(noisy)
            delta, residual_energy_db = compute_metrics(
                clean, noisy, predicted, wrapper_mode
            )

            dims = tuple(range(1, clean.ndim))
            sigma_x_sq = clean.var(dim=dims, unbiased=False)
            noise_var = (sigma / 255.0) ** 2
            snr = (sigma_x_sq / noise_var).reshape(-1)

            delta_samples.append(delta.cpu().numpy())
            residual_samples.append(residual_energy_db.cpu().numpy())
            snr_samples.append(snr.cpu().numpy())

    return SigmaResults(
        sigma=sigma,
        delta=np.concatenate(delta_samples, axis=0),
        residual=np.concatenate(residual_samples, axis=0),
        snr=np.concatenate(snr_samples, axis=0),
    )


def load_or_compute_sigma_results(
    sigma: float,
    cache_path: Path,
    use_cache: bool,
    model: torch.nn.Module,
    loader: LazyLoader,
    device: torch.device,
    max_batches: int | None,
    wrapper_mode: WrapperMode,
    noise_type: NoiseType,
) -> SigmaResults:
    """Load cached results if available, otherwise compute and save."""
    if use_cache and cache_path.is_file():
        try:
            res = load_sigma_results(cache_path)
            print(f"Loaded cached results for σ={sigma:g} from {cache_path.name}")
            return res
        except Exception as exc:
            print(f"Failed to load cached results for σ={sigma:g}: {exc}. Recomputing.")

    res = collect_metrics_for_sigma(
        model=model,
        loader=loader.get(),
        device=device,
        sigma=sigma,
        max_batches=max_batches,
        wrapper_mode=wrapper_mode,
        noise_type=noise_type,
    )
    save_sigma_results(res, cache_path)
    print(f"Saved results for σ={sigma:g} to {cache_path.name}")
    return res


def plot_metric_overlay(
    results: list[SigmaResults],
    axis_ctx,
    ylabel: str,
    save_path: Path,
    bins: int,
    limits: AxisLimits,
    per_series_range: list[tuple[float, float] | None] | None = None,
    ylim_percentiles: tuple[float, float] | None = (0.5, 99.5),
    show_legend: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    style_path = PROJECT_ROOT / Path("icml_like.mplstyle")
    with plt.style.context(str(style_path)):
        h = 4
        w = h * (4 / 3)
        fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        default_range = limits.effective_bin_range(axis_ctx.bin_range)

        for idx, result in enumerate(results):
            series_range = None
            if per_series_range is not None and idx < len(per_series_range):
                series_range = per_series_range[idx]
            current_range = first_valid(series_range, default_range)
            assert current_range is not None, "No valid range for binning."

            x_values = result.delta
            y_values = result.residual
            if series_range is not None:
                mask = (x_values >= series_range[0]) & (x_values <= series_range[1])
                x_values = x_values[mask]
                y_values = y_values[mask]

            centers, means, stds, _ = aggregate_bins(
                x_values, y_values, bins, current_range
            )
            if centers.size == 0:
                continue

            color = colors[idx % len(colors)]
            ax.plot(
                centers, means, color=color, linewidth=2.2, label=f"σ={result.sigma:g}"
            )
            ax.fill_between(
                centers, means - stds, means + stds, color=color, alpha=0.14
            )

        ax.set_xlabel(axis_ctx.label)
        ax.set_ylabel(ylabel)
        ax.grid(
            True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.5
        )

        if show_legend:
            ax.legend(
                loc="upper left",
                ncol=3,
                frameon=True,
                borderaxespad=0.3,
                handlelength=2.2,
            )

        effective_xlim = limits.effective_xlim(axis_ctx.bin_range)
        if effective_xlim is not None:
            ax.set_xlim(*effective_xlim)

        if limits.ylim is not None:
            ax.set_ylim(*limits.ylim)
        elif ylim_percentiles is not None:
            combined = np.concatenate([res.residual for res in results], axis=0)
            lims = _percentile_limits(combined, ylim_percentiles)
            if lims is not None:
                ax.set_ylim(*lims)

        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("black")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)


def resolve_sigma_values(
    train_cfg: TrainConfig, noise_sigmas: list[float] | None, train_sigma: float | None
) -> list[float]:
    """Resolve sigma values from configuration (no implicit train inclusion)."""
    return resolve_noise_list(train_cfg, noise_sigmas)


def _print_variance_ratio(stats: VarianceRatioResult, label: str) -> None:
    """Pretty-print VR_σ and supporting per-σ coverage stats."""
    vr_value = (
        f"{stats.vr_sigma:.4f}" if np.isfinite(stats.vr_sigma) else "nan"
    )
    var_b_str = (
        f"{stats.var_across_bins:.4f}"
        if np.isfinite(stats.var_across_bins)
        else "nan"
    )
    print(
        f"Residual σ-variance ratio (VRσ) [{label}]: "
        f"{vr_value} | bins used {stats.bins_used}/{stats.bins_total} "
        f"over Δ∈[{stats.bin_range[0]:.3g}, {stats.bin_range[1]:.3g}]"
    )
    print(
        f"  mean Var_k(m_b,k) across bins: {stats.mean_var_across_bins:.4f}; "
        f"Var_b(m̄_b): {var_b_str}"
    )
    print("  per-σ coverage (samples | bins with data | mean residual dB):")
    for sigma in sorted(stats.sigmas):
        sample_count = stats.per_sigma_sample_counts.get(sigma, 0)
        bin_count = stats.per_sigma_bin_counts.get(sigma, 0)
        mean_residual = stats.per_sigma_mean_residual.get(sigma, float("nan"))
        mean_str = (
            f"{mean_residual:.3f}"
            if np.isfinite(mean_residual)
            else "nan"
        )
        print(
            f"    σ={sigma:g}: {sample_count} samples | {bin_count} bins | {mean_str} dB"
        )


def main(script_cfg: ResidualCalcConfig = DEFAULT_CONFIG) -> None:
    start_time = time.time()
    print("Residual calc config:")
    print(asdict(script_cfg))

    delta_percentiles = _validate_percentile_tuple(script_cfg.delta_percentiles)

    if script_cfg.model_key and script_cfg.log_dir:
        raise ValueError("Provide either model_key or log_dir, not both.")

    log_dir = _resolve_log_dir(script_cfg)
    train_cfg = load_train_config(log_dir / "config.json")
    if script_cfg.batch_size is not None:
        train_cfg.batch_size = script_cfg.batch_size

    checkpoint = resolve_checkpoint_path(
        log_dir, script_cfg.epoch, script_cfg.checkpoint
    )
    device = torch.device(
        script_cfg.device
        if script_cfg.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = build_model(train_cfg).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    analysis_wrapper = resolve_analysis_wrapper(train_cfg, script_cfg.wrapper_mode)
    train_wrapper = getattr(train_cfg.model_cfg, "wrapper_mode", "idem").lower()
    if script_cfg.wrapper_mode is not None and analysis_wrapper != train_wrapper:
        print(
            f"Overriding train wrapper '{train_wrapper}' with '{analysis_wrapper}' for analysis."
        )
    else:
        print(f"Using wrapper mode '{analysis_wrapper}' for normalization.")

    sigma_values = resolve_sigma_values(
        train_cfg, script_cfg.noise_sigmas, script_cfg.train_sigma
    )
    train_sigma_value = (
        float(script_cfg.train_sigma) if script_cfg.train_sigma is not None else None
    )

    model_folder = script_cfg.model_key if script_cfg.model_key else run_name(train_cfg)
    model_save_dir = script_cfg.save_dir / model_folder
    cache_root = model_save_dir / "sigma_results"
    overlay_root = overlay_subdir(model_save_dir, sigma_values)
    cache_root.mkdir(parents=True, exist_ok=True)
    overlay_root.mkdir(parents=True, exist_ok=True)

    loader = LazyLoader(train_cfg)
    results: list[SigmaResults] = []
    results_by_sigma: dict[float, SigmaResults] = {}
    noise_type: NoiseType = train_cfg.noise_type

    for sigma in sigma_values:
        res = load_or_compute_sigma_results(
            sigma=sigma,
            cache_path=sigma_cache_path(cache_root, sigma),
            use_cache=script_cfg.use_cache,
            model=model,
            loader=loader,
            device=device,
            max_batches=script_cfg.max_batches,
            wrapper_mode=analysis_wrapper,
            noise_type=noise_type,
        )
        results.append(res)
        results_by_sigma[float(sigma)] = res

    if (
        train_sigma_value is not None
        and train_sigma_value not in results_by_sigma
    ):
        train_res = load_or_compute_sigma_results(
            sigma=train_sigma_value,
            cache_path=sigma_cache_path(cache_root, train_sigma_value),
            use_cache=script_cfg.use_cache,
            model=model,
            loader=loader,
            device=device,
            max_batches=script_cfg.max_batches,
            wrapper_mode=analysis_wrapper,
            noise_type=noise_type,
        )
        results_by_sigma[train_sigma_value] = train_res

    axis_ctx = resolve_axis_context(results, analysis_wrapper)
    vr_stats = compute_sigma_variance_ratio(
        results=results,
        bins=script_cfg.bins,
        bin_range=axis_ctx.bin_range,
        delta_percentiles=delta_percentiles,
    )
    if vr_stats is not None:
        _print_variance_ratio(vr_stats, label="per-σ percentile windows")
    else:
        print("VRσ (per-σ percentile windows) could not be computed (insufficient coverage across Δ bins).")

    intersection_window = _intersection_percentile_range(results, delta_percentiles)
    vr_stats_intersection = compute_sigma_variance_ratio(
        results=results,
        bins=script_cfg.bins,
        bin_range=axis_ctx.bin_range,
        delta_percentiles=delta_percentiles,
        window_mode="intersection",
        intersection_range=intersection_window,
    )
    if vr_stats_intersection is not None:
        _print_variance_ratio(vr_stats_intersection, label="intersection percentile window")
    else:
        print("VRσ (intersection percentile window) could not be computed (insufficient coverage across Δ bins or empty intersection).")

    if script_cfg.plot:
        plotted_results = list(results)
        _generate_plots(
            results=results,
            plotted_results=plotted_results,
            results_by_sigma=results_by_sigma,
            train_sigma_value=train_sigma_value,
            delta_percentiles=delta_percentiles,
            analysis_wrapper=analysis_wrapper,
            overlay_root=overlay_root,
            script_cfg=script_cfg,
            axis_ctx=axis_ctx,
        )

    elapsed = time.time() - start_time
    print(f"Total elapsed time: {elapsed:.2f} seconds")


def _resolve_log_dir(script_cfg: ResidualCalcConfig) -> Path:
    """Resolve and validate the log directory."""
    log_dir: Path | None = script_cfg.log_dir
    if script_cfg.model_key:
        if script_cfg.model_key not in models_log:
            raise KeyError(
                f"model_key '{script_cfg.model_key}' not found in models_log."
            )
        log_dir = PROJECT_ROOT / models_log[script_cfg.model_key]
    if log_dir is None:
        raise ValueError("log_dir must be provided if model_key is not set.")

    log_dir = log_dir.expanduser().resolve()
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Log directory {log_dir} does not exist.")
    if not (log_dir / "config.json").is_file():
        raise FileNotFoundError(f"No config.json found in {log_dir}.")
    return log_dir


def _generate_plots(
    results: list[SigmaResults],
    plotted_results: list[SigmaResults],
    results_by_sigma: dict[float, SigmaResults],
    train_sigma_value: float | None,
    delta_percentiles: tuple[float, float],
    analysis_wrapper: WrapperMode,
    overlay_root: Path,
    script_cfg: ResidualCalcConfig,
    axis_ctx: AxisContext | None = None,
) -> None:
    """Generate all residual overlay plots."""
    if not plotted_results:
        print("No results to plot. Skipping plots.")
        return

    if axis_ctx is None:
        axis_ctx = resolve_axis_context(plotted_results, analysis_wrapper)
    residual_label = r"$Q(\tilde y,\tilde x)$ (dB)"

    # Precompute percentile ranges
    train_result = (
        results_by_sigma.get(train_sigma_value) if train_sigma_value else None
    )
    train_percentile = (
        _finite_percentile_range(train_result.delta, delta_percentiles)
        if train_result is not None
        else None
    )
    per_series_percentiles: list[tuple[float, float] | None] = [
        _finite_percentile_range(res.delta, delta_percentiles) for res in plotted_results
    ]

    percentile_label = f"{delta_percentiles[0]:g}_{delta_percentiles[1]:g}"

    # 1) Normal plot (no suffix): use default binning, no xlim manipulation.
    base_limits = AxisLimits(
        xlim=None,
        ylim=script_cfg.residual_ylim,
        bin_range=axis_ctx.default_xlim,
    )
    plot_metric_overlay(
        plotted_results,
        axis_ctx,
        ylabel=residual_label,
        save_path=overlay_root / f"{axis_ctx.prefix}_vs_norm_residual.pdf",
        bins=script_cfg.bins,
        limits=base_limits,
        per_series_range=None,
        show_legend=script_cfg.show_legend,
    )

    # 2) Normal plot with explicit xlim (_xlim): only if residual_xlim is provided.
    if script_cfg.residual_xlim is not None:
        xlim_limits = AxisLimits(
            xlim=script_cfg.residual_xlim,
            ylim=script_cfg.residual_ylim,
            bin_range=axis_ctx.default_xlim,
        )
        plot_metric_overlay(
            plotted_results,
            axis_ctx,
            ylabel=residual_label,
            save_path=overlay_root / f"{axis_ctx.prefix}_vs_norm_residual_xlim.pdf",
            bins=script_cfg.bins,
            limits=xlim_limits,
            per_series_range=None,
            show_legend=script_cfg.show_legend,
        )

    # 3) Non-clipped percentile: requires train sigma percentile bounds.
    if train_percentile is not None:
        percentile_limits = AxisLimits(
            xlim=train_percentile,
            ylim=script_cfg.residual_ylim,
            bin_range=axis_ctx.default_xlim,
        )
        plot_metric_overlay(
            plotted_results,
            axis_ctx,
            ylabel=residual_label,
            save_path=overlay_root
            / f"{axis_ctx.prefix}_vs_norm_residual_percentiles_{percentile_label}.pdf",
            bins=script_cfg.bins,
            limits=percentile_limits,
            per_series_range=None,
            show_legend=script_cfg.show_legend,
        )

        # 4) Clipped percentile: same as above but mask each curve to its own percentile window.
        plot_metric_overlay(
            plotted_results,
            axis_ctx,
            ylabel=residual_label,
            save_path=overlay_root
            / f"{axis_ctx.prefix}_vs_norm_residual_percentiles_{percentile_label}_clipped.pdf",
            bins=script_cfg.bins,
            limits=percentile_limits,
            per_series_range=per_series_percentiles,
            show_legend=script_cfg.show_legend,
        )

    # 5) Clipped with explicit xlim (_clipped_xlim): per-series percentile mask, no train required.
    if script_cfg.residual_xlim is not None:
        clipped_xlim_limits = AxisLimits(
            xlim=script_cfg.residual_xlim,
            ylim=script_cfg.residual_ylim,
            bin_range=axis_ctx.default_xlim,
        )
        plot_metric_overlay(
            plotted_results,
            axis_ctx,
            ylabel=residual_label,
            save_path=overlay_root
            / f"{axis_ctx.prefix}_vs_norm_residual_percentiles_{percentile_label}_clipped_xlim.pdf",
            bins=script_cfg.bins,
            limits=clipped_xlim_limits,
            per_series_range=per_series_percentiles,
            show_legend=script_cfg.show_legend,
        )

    # 6) Double clipped with explicit xlim: per-series percentile mask + train percentile mask.
    if train_percentile is not None and script_cfg.residual_xlim is not None:
        def _intersect(rng: tuple[float, float] | None, base: tuple[float, float]) -> tuple[float, float] | None:
            if rng is None:
                return base
            low = max(rng[0], base[0])
            high = min(rng[1], base[1])
            if low >= high:
                return None
            return (low, high)

        combined_ranges = [
            _intersect(rng, train_percentile) for rng in per_series_percentiles
        ]
        double_clipped_limits = AxisLimits(
            xlim=script_cfg.residual_xlim,
            ylim=script_cfg.residual_ylim,
            bin_range=axis_ctx.default_xlim,
        )
        plot_metric_overlay(
            plotted_results,
            axis_ctx,
            ylabel=residual_label,
            save_path=overlay_root
            / f"{axis_ctx.prefix}_vs_norm_residual_percentiles_{percentile_label}_double_clipped_xlim.pdf",
            bins=script_cfg.bins,
            limits=double_clipped_limits,
            per_series_range=combined_ranges,
            show_legend=script_cfg.show_legend,
        )


if __name__ == "__main__":
    main()
