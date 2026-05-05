from contextlib import nullcontext
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Iterable, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_logs import models_log
from se.configs import PROJECT_ROOT, TrainConfig, resolve_image_mode
from se.data import ImageFolderDataset
from se.utils.eval_utils import (
    default_color_overrides,
    load_model_for_eval,
    pretty_label_map,
)
from se.utils.noise_model import get_noise
from se.utils.train_utils import run_name


@dataclass
class NEViolationConfig:
    model_keys: list[str] | None = None
    log_dirs: list[Path] | None = None
    checkpoint: Path | None = None
    epoch: int | None = None
    data_dirs: list[str] = field(default_factory=lambda: [f"{PROJECT_ROOT}/data/Set12"])
    device: str | None = None
    sigma_values: list[float] = field(default_factory=lambda: list(range(0, 100, 5)))
    noise_type: str | None = None
    n_averages: int = 1
    num_affine_samples: int = 8
    a_min: float = 0.5
    a_max: float = 1.5
    b_min: float = -0.25
    b_max: float = 0.25
    tau: float = 1e-6
    max_images: int | None = None
    seed: int = 0
    show_legend: bool = True
    use_cache: bool = True
    save_csv: Path | None = None
    save_plot: Path | None = None


@dataclass
class LoadedModel:
    name: str
    model: torch.nn.Module
    cfg: TrainConfig
    checkpoint_path: Path


def normalize_sigma_values(values: Iterable[float]) -> list[float]:
    return [float(v) / 255.0 if float(v) > 1.0 else float(v) for v in values]


def resolve_input_dirs(raw_dirs: list[str]) -> list[str]:
    resolved = []
    for raw_dir in raw_dirs:
        candidate = Path(raw_dir).expanduser()
        if not candidate.is_absolute():
            candidate = Path(PROJECT_ROOT) / candidate
        candidate = candidate.resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(f"Dataset directory {candidate} does not exist.")
        resolved.append(str(candidate))
    return resolved


def resolve_model_selection(
    model_keys: list[str] | None, log_dirs: list[Path] | None
) -> tuple[list[Path], list[str]]:
    if model_keys and log_dirs:
        raise ValueError("Provide either model_keys or log_dirs, not both.")
    if model_keys:
        missing = [key for key in model_keys if key not in models_log]
        if missing:
            raise KeyError(
                f"Unknown model_keys {missing}. Available keys: {sorted(models_log)}"
            )
        return [Path(models_log[key]) for key in model_keys], list(model_keys)
    if log_dirs:
        return [Path(path) for path in log_dirs], []
    raise ValueError("Provide at least one model key or log dir.")


def load_models(
    log_dirs: list[Path],
    provided_names: list[str],
    device: torch.device,
    checkpoint_override: Path | None,
    epoch: int | None,
) -> list[LoadedModel]:
    if checkpoint_override is not None and len(log_dirs) > 1:
        raise ValueError(
            "checkpoint overrides are only supported for a single log dir."
        )

    loaded: list[LoadedModel] = []
    use_provided_names = bool(provided_names)
    for index, raw_dir in enumerate(log_dirs):
        log_dir = raw_dir.expanduser()
        if not log_dir.is_absolute():
            log_dir = (Path(PROJECT_ROOT) / log_dir).resolve()
        else:
            log_dir = log_dir.resolve()
        if not log_dir.is_dir():
            raise FileNotFoundError(f"Log directory {log_dir} does not exist.")

        loaded_eval_model = load_model_for_eval(
            log_dir,
            device,
            epoch=epoch,
            checkpoint_override=checkpoint_override,
        )

        name = (
            provided_names[index]
            if use_provided_names
            else run_name(loaded_eval_model.cfg)
        )
        loaded.append(
            LoadedModel(
                name=name,
                model=loaded_eval_model.model,
                cfg=loaded_eval_model.cfg,
                checkpoint_path=loaded_eval_model.checkpoint_path,
            )
        )
    return loaded


def sample_affine(
    batch_size: int,
    ndim: int,
    *,
    a_min: float,
    a_max: float,
    b_min: float,
    b_max: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (batch_size,) + (1,) * (ndim - 1)
    a = torch.empty(shape, device=device, dtype=dtype).uniform_(a_min, a_max)
    b = torch.empty(shape, device=device, dtype=dtype).uniform_(b_min, b_max)
    return a, b


def _repeat_batch(batch: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats <= 1:
        return batch
    return (
        batch.unsqueeze(1)
        .expand(-1, repeats, *batch.shape[1:])
        .reshape(-1, *batch.shape[1:])
    )


def _forward_in_chunks(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    max_chunk_size: int | None = None,
) -> tuple[torch.Tensor, int]:
    total_batch = inputs.shape[0]
    chunk_size = (
        total_batch
        if max_chunk_size is None
        else max(1, min(int(max_chunk_size), total_batch))
    )

    while True:
        outputs: list[torch.Tensor] = []
        try:
            for input_chunk in inputs.split(chunk_size):
                outputs.append(model(input_chunk))
            return torch.cat(outputs, dim=0), chunk_size
        except torch.OutOfMemoryError:
            del outputs
            if inputs.device.type != "cuda" or chunk_size <= 1:
                raise
            torch.cuda.empty_cache()
            chunk_size = max(1, chunk_size // 2)


def normalized_ne_defect(
    model: torch.nn.Module,
    noisy: torch.Tensor,
    *,
    num_affine_samples: int,
    a_min: float,
    a_max: float,
    b_min: float,
    b_max: float,
    tau: float,
    max_affine_batch: int | None = None,
) -> tuple[torch.Tensor, int, int]:
    batch_size = noisy.shape[0]
    ndim = noisy.ndim

    with torch.inference_mode():
        base_output = model(noisy)
        affine_shape = (batch_size, num_affine_samples) + (1,) * (ndim - 1)
        a = torch.empty(affine_shape, device=noisy.device, dtype=noisy.dtype).uniform_(
            a_min, a_max
        )
        b = torch.empty(affine_shape, device=noisy.device, dtype=noisy.dtype).uniform_(
            b_min, b_max
        )

        transformed_inputs = (a * noisy.unsqueeze(1) + b).flatten(0, 1)
        reference = (a * base_output.unsqueeze(1) + b).flatten(0, 1)
        transformed_output, resolved_affine_batch = _forward_in_chunks(
            model,
            transformed_inputs,
            max_chunk_size=max_affine_batch,
        )

        numer = torch.linalg.vector_norm(
            (transformed_output - reference).flatten(1), ord=2, dim=1
        )
        denom = torch.linalg.vector_norm(reference.flatten(1), ord=2, dim=1) + tau
        defect_values = numer / denom

    return defect_values.double().sum(), defect_values.numel(), resolved_affine_batch


def evaluate_model(
    loaded_model: LoadedModel,
    loader: DataLoader,
    sigma_values: list[float],
    *,
    noise_type_override: str | None,
    n_averages: int,
    num_affine_samples: int,
    a_min: float,
    a_max: float,
    b_min: float,
    b_max: float,
    tau: float,
    device: torch.device,
) -> list[float]:
    noise_type = noise_type_override or loaded_model.cfg.noise_type
    per_sigma_values: list[float] = []
    max_repeat_batch = n_averages
    max_affine_batch: int | None = None
    oom_auto_reduced = False

    sigma_progress = tqdm(
        sigma_values,
        desc=f"{loaded_model.name} sigmas",
        leave=False,
        position=1,
    )
    for sigma in sigma_progress:
        defect_total = torch.zeros((), dtype=torch.float64, device=device)
        defect_count = 0
        progress = tqdm(
            loader,
            desc=f"{loaded_model.name} sigma={sigma * 255.0:.1f}",
            leave=False,
            position=2,
        )
        for clean in progress:
            clean = clean.to(device, non_blocking=True)
            repeats_done = 0
            while repeats_done < n_averages:
                current_batch = min(max_repeat_batch, n_averages - repeats_done)
                batch_clean = None
                noisy = None
                try:
                    batch_clean = _repeat_batch(clean, current_batch)
                    noisy = batch_clean + get_noise(
                        batch_clean,
                        min_noise=sigma,
                        max_noise=sigma,
                        noise_type=noise_type,  # type: ignore[arg-type]
                    )
                    defect_sum, count, resolved_affine_batch = normalized_ne_defect(
                        loaded_model.model,
                        noisy,
                        num_affine_samples=num_affine_samples,
                        a_min=a_min,
                        a_max=a_max,
                        b_min=b_min,
                        b_max=b_max,
                        tau=tau,
                        max_affine_batch=max_affine_batch,
                    )
                    defect_total += defect_sum
                    defect_count += count
                    repeats_done += current_batch
                    max_repeat_batch = current_batch
                    max_affine_batch = resolved_affine_batch
                except torch.OutOfMemoryError:
                    del batch_clean, noisy
                    if device.type != "cuda" or current_batch <= 1:
                        raise
                    if not oom_auto_reduced:
                        print(
                            "eval_ne_violation: CUDA OOM during evaluation; "
                            "reducing repeat/affine batch sizes automatically."
                        )
                        oom_auto_reduced = True
                    torch.cuda.empty_cache()
                    max_repeat_batch = max(1, current_batch // 2)
                    if max_affine_batch is not None:
                        max_affine_batch = max(1, max_affine_batch // 2)
        if defect_count == 0:
            raise RuntimeError("NE-violation evaluation saw zero samples.")
        per_sigma_values.append((defect_total / defect_count).item())

    sigma_progress.close()
    return per_sigma_values


def default_save_path(data_dirs: list[str], model_names: list[str]) -> Path:
    dataset_name = Path(data_dirs[0]).name if len(data_dirs) == 1 else "mixed"
    stem = model_names[0] if len(model_names) == 1 else "comparison"
    return Path(PROJECT_ROOT) / "eval_logs" / dataset_name / f"{stem}_ne_violation.csv"


def _row_for_sigma(df: pd.DataFrame, target: float) -> pd.Series | None:
    sigma_series = pd.to_numeric(df["sigma"], errors="coerce")
    matches = df.loc[np.isclose(sigma_series, target, rtol=1e-6, atol=1e-8)]
    if matches.empty:
        return None
    return matches.iloc[-1]


def resolve_common_training_sigma(
    loaded_models: list[LoadedModel],
) -> tuple[float, float] | None:
    if not loaded_models:
        return None

    training_sigma = (
        float(loaded_models[0].cfg.min_noise),
        float(loaded_models[0].cfg.max_noise),
    )
    for loaded in loaded_models[1:]:
        candidate = (float(loaded.cfg.min_noise), float(loaded.cfg.max_noise))
        if candidate != training_sigma:
            print(
                "Warning: models have different training noise ranges; "
                "skipping shaded training-sigma region."
            )
            return None
    return training_sigma


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


def _default_ne_violation_limits(values: np.ndarray) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return (0.0, 1.0)

    upper = float(np.nanmax(finite_values))
    if upper <= 0.0:
        return (0.0, 0.05)

    pad = max(0.08 * upper, 2e-3)
    return (0.0, upper + pad)


def plot_ne_violation_curves(
    sigma_values_8bit: list[float],
    curves: Mapping[str, list[float]],
    *,
    training_sigma: tuple[float, float] | None,
    save_path: Path,
    pretty_labels: Mapping[str, str],
    model_colors: Mapping[str, str | int],
    show_legend: bool,
) -> None:
    style_file = Path(PROJECT_ROOT) / "icml_like.mplstyle"
    style_ctx = (
        plt.style.context(str(style_file)) if style_file.is_file() else nullcontext()
    )

    with style_ctx:
        fig, ax = plt.subplots(figsize=(5.0, 3.8))
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        color_overrides = {
            pretty_labels.get(name, name): color for name, color in model_colors.items()
        }

        all_y_values: list[np.ndarray] = []
        for name, curve_values in curves.items():
            label = pretty_labels.get(name, name)
            curve = np.asarray(curve_values, dtype=np.float64)
            all_y_values.append(curve)
            color = _resolve_color(label, color_overrides, color_cycle)
            ax.plot(
                sigma_values_8bit,
                curve,
                marker="o",
                linewidth=1.8,
                label=label,
                color=color,
            )
            ax.scatter(
                sigma_values_8bit,
                curve,
                color=color,
                s=18,
                zorder=8,
                label="_nolegend_",
            )

        if all_y_values:
            y_limits = _default_ne_violation_limits(np.concatenate(all_y_values))
        else:
            y_limits = (0.0, 1.0)

        if training_sigma is not None:
            lo8, hi8 = sorted((float(training_sigma[0]), float(training_sigma[1])))
            clipped_lo = max(0.0, lo8)
            clipped_hi = min(95.0, hi8)
            if clipped_lo <= clipped_hi:
                if abs(clipped_hi - clipped_lo) < 1e-6:
                    ax.axvline(
                        clipped_lo,
                        color="#69a8d8",
                        linewidth=3.0,
                        alpha=0.8,
                        zorder=1,
                    )
                else:
                    ax.axvspan(
                        clipped_lo,
                        clipped_hi,
                        alpha=0.18,
                        color="#69a8d8",
                        zorder=0,
                    )

        ax.set_xlabel(r"noise level $\sigma$ (8-bit)")
        ax.set_ylabel(r"$\epsilon_{\mathrm{NE}}$")
        ax.set_xlim(0, 95)
        ax.set_ylim(y_limits)
        ax.set_xticks(list(range(0, 100, 10)))
        ax.set_xticks(list(range(0, 100, 5)), minor=True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_formatter(
            ScalarFormatter(useOffset=False, useMathText=False)
        )

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
                line.set_marker("o")
                line.set_markersize(6)

        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("black")

        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)


def validate_config(eval_cfg: NEViolationConfig) -> None:
    if eval_cfg.a_min <= 0 or eval_cfg.a_max <= 0:
        raise ValueError("Affine scale bounds must satisfy a > 0.")
    if eval_cfg.a_min >= eval_cfg.a_max:
        raise ValueError("Expected a_min < a_max.")
    if eval_cfg.b_min >= eval_cfg.b_max:
        raise ValueError("Expected b_min < b_max.")
    if eval_cfg.n_averages < 1:
        raise ValueError("n_averages must be at least 1.")
    if eval_cfg.num_affine_samples < 1:
        raise ValueError("num_affine_samples must be at least 1.")


def main(eval_cfg: NEViolationConfig | None = None) -> None:
    if eval_cfg is None:
        eval_cfg = NEViolationConfig(
            model_keys=["b_swinir_0-55", "wne_swinir_0-55"],
            save_csv=Path(PROJECT_ROOT)
            / "eval_logs"
            / "Set12"
            / "swinir_multinoise_baseline_vs_wne_ne_violation.csv",
        )

    validate_config(eval_cfg)
    torch.manual_seed(eval_cfg.seed)
    device = torch.device(
        eval_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    resolved_data_dirs = resolve_input_dirs(eval_cfg.data_dirs)
    sigma_values = normalize_sigma_values(eval_cfg.sigma_values)
    selected_dirs, provided_names = resolve_model_selection(
        eval_cfg.model_keys, eval_cfg.log_dirs
    )
    checkpoint_override = (
        Path(eval_cfg.checkpoint).expanduser().resolve()
        if eval_cfg.checkpoint is not None
        else None
    )

    loaded_models = load_models(
        selected_dirs,
        provided_names,
        device,
        checkpoint_override,
        eval_cfg.epoch,
    )
    training_sigma = resolve_common_training_sigma(loaded_models)
    model_names = [loaded.name for loaded in loaded_models]
    pretty_labels = pretty_label_map(model_names)
    model_colors = default_color_overrides(model_names)
    save_path = (
        Path(eval_cfg.save_csv).expanduser().resolve()
        if eval_cfg.save_csv is not None
        else default_save_path(resolved_data_dirs, model_names)
    )
    plot_path = (
        Path(eval_cfg.save_plot).expanduser().resolve()
        if eval_cfg.save_plot is not None
        else save_path.with_suffix(".pdf")
    )
    existing_df = (
        pd.read_csv(save_path) if eval_cfg.use_cache and save_path.is_file() else None
    )
    pending_sigmas_by_model: dict[str, set[float]] = {
        name: set() for name in model_names
    }
    if existing_df is None:
        for name in model_names:
            pending_sigmas_by_model[name].update(float(sigma) for sigma in sigma_values)
    else:
        missing_model_cols = [
            name for name in model_names if name not in existing_df.columns
        ]
        if missing_model_cols:
            print(
                f"Cache {save_path} missing model columns {missing_model_cols}; "
                "recomputing requested sigmas."
            )
            for name in missing_model_cols:
                pending_sigmas_by_model[name].update(
                    float(sigma) for sigma in sigma_values
                )

        for sigma in sigma_values:
            cached_sigma_row = _row_for_sigma(existing_df, sigma)
            if cached_sigma_row is None:
                for name in model_names:
                    pending_sigmas_by_model[name].add(float(sigma))
                continue

            for name in model_names:
                if name in missing_model_cols:
                    continue
                value = cached_sigma_row.get(name, np.nan)
                if pd.isna(value) or value == "":
                    pending_sigmas_by_model[name].add(float(sigma))

        missing_summary = {
            name: sorted(values)
            for name, values in pending_sigmas_by_model.items()
            if values
        }
        if missing_summary:
            print(
                f"Cache {save_path} missing NE-violation values for {missing_summary}; "
                "computing only those entries."
            )
        else:
            print(f"Using cached NE-violation curves from {save_path}.")

    image_modes = {resolve_image_mode(loaded.cfg) for loaded in loaded_models}
    if len(image_modes) != 1:
        raise ValueError(
            f"All models must share the same image mode. Found {sorted(image_modes)}."
        )
    image_mode = image_modes.pop()

    models_to_compute = [
        loaded for loaded in loaded_models if pending_sigmas_by_model[loaded.name]
    ]
    computed_curves: dict[str, dict[float, float]] = {}
    num_evaluations: int | None = None

    if models_to_compute:
        dataset = ImageFolderDataset(
            in_folders=resolved_data_dirs,
            image_mode=image_mode,
            max_images=eval_cfg.max_images,
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=device.type == "cuda",
            shuffle=False,
        )

        model_progress = tqdm(models_to_compute, desc="models", position=0)
        for loaded in model_progress:
            model_progress.set_postfix_str(loaded.name)
            requested_sigmas = sorted(pending_sigmas_by_model[loaded.name])
            values = evaluate_model(
                loaded,
                loader,
                requested_sigmas,
                noise_type_override=eval_cfg.noise_type,
                n_averages=eval_cfg.n_averages,
                num_affine_samples=eval_cfg.num_affine_samples,
                a_min=eval_cfg.a_min,
                a_max=eval_cfg.a_max,
                b_min=eval_cfg.b_min,
                b_max=eval_cfg.b_max,
                tau=eval_cfg.tau,
                device=device,
            )
            computed_curves[loaded.name] = {
                round(float(sigma), 8): float(value)
                for sigma, value in zip(requested_sigmas, values)
            }
        model_progress.close()

        num_evaluations = (
            len(dataset) * eval_cfg.n_averages * eval_cfg.num_affine_samples
        )

    rows: list[dict[str, float | int | str]] = []
    curves: dict[str, list[float]] = {name: [] for name in model_names}
    for sigma in sigma_values:
        cached_sigma_row = (
            _row_for_sigma(existing_df, sigma) if existing_df is not None else None
        )
        num_evaluations_value: int | str
        if num_evaluations is not None:
            num_evaluations_value = num_evaluations
        elif cached_sigma_row is not None:
            cached_num_evaluations = cached_sigma_row.get("num_evaluations", np.nan)
            if not pd.isna(cached_num_evaluations):
                num_evaluations_value = int(cached_num_evaluations)
            else:
                num_evaluations_value = ""
        else:
            num_evaluations_value = ""

        output_row: dict[str, float | int | str] = {
            "sigma_8bit": round(float(sigma * 255.0), 6),
            "sigma": round(float(sigma), 8),
            "num_evaluations": num_evaluations_value,
        }
        for loaded in loaded_models:
            sigma_key = round(float(sigma), 8)
            cached_value = (
                cached_sigma_row.get(loaded.name, np.nan)
                if cached_sigma_row is not None
                else np.nan
            )
            if (
                loaded.name in computed_curves
                and sigma_key in computed_curves[loaded.name]
            ):
                value = computed_curves[loaded.name][sigma_key]
            elif (
                cached_sigma_row is not None
                and not pd.isna(cached_value)
                and cached_value != ""
            ):
                value = float(cached_value)
            else:
                raise RuntimeError(
                    f"Missing NE-violation value for model={loaded.name} sigma={sigma_key}."
                )
            curves[loaded.name].append(value)
            output_row[loaded.name] = round(value, 8)
        rows.append(output_row)

    df = pd.DataFrame(rows)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    plot_ne_violation_curves(
        sigma_values_8bit=[float(sigma * 255.0) for sigma in sigma_values],
        curves=curves,
        training_sigma=training_sigma,
        save_path=plot_path,
        pretty_labels=pretty_labels,
        model_colors=model_colors,
        show_legend=eval_cfg.show_legend,
    )

    print(f"Saved NE violation CSV to {save_path} and plot to {plot_path}")
    for loaded in loaded_models:
        print(
            f"{loaded.name}: mean epsilon_NE={df[loaded.name].mean():.8f} "
            f"(checkpoint={loaded.checkpoint_path})"
        )


if __name__ == "__main__":
    main()
