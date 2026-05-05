"""Run Kadkhodaie-Simoncelli denoiser-residual samplers on Set12.

This script implements the denoiser-residual sampler used for Appendix N.
The unconstrained denoising/sampling case and the constrained linear
inverse-problem case follow:

    Kadkhodaie and Simoncelli, "Stochastic solutions for linear inverse
    problems using the prior implicit in a denoiser", NeurIPS 2021.

Constrained problems share the same update rule and differ only through the
measurement projector P = M M^T.
"""

from __future__ import annotations

from collections import Counter
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
import torch.nn.functional as F
from tqdm import tqdm

from model_logs import models_log
from se.configs import PROJECT_ROOT
from se.data import list_image_files
from se.utils.eval_utils import load_model_for_eval
from se.utils.image_utils import load_grayscale_tensor, save_tensor_gif, save_tensor_image
from se.utils.metrics import psnr as tensor_psnr
from se.utils.metrics import ssim as tensor_ssim


DEFAULT_MODEL_KEY = "b_swinir_10_n2n"
DEFAULT_DATA_DIR = Path(PROJECT_ROOT) / "data" / "Set12"
DEFAULT_OUTPUT_ROOT = Path(PROJECT_ROOT) / "artifacts" / "kadkhodaie_denoiser_sampler"
# DEFAULT_INVERSE_INITIAL_SIGMA_8BIT = 255
DEFAULT_INVERSE_INITIAL_SIGMA_8BIT = 10
SUPPORTED_PROBLEMS = (
    "denoising",
    "inpaint_center",
    "random_missing",
    "block_superres",
    "fourier_lowpass",
)


@dataclass
class SamplerConfig:
    model_key: str = DEFAULT_MODEL_KEY
    data_dir: Path = DEFAULT_DATA_DIR
    image_path: Path | None = None
    max_images: int | None = None
    output_dir: Path | None = None
    problem: str = "random_missing"
    sigma_8bit: float = 10.0
    initial_sigma_8bit: float | None = None
    num_steps: int = 1000
    h0: float = 0.01
    h_max: float | None = None
    beta: float | None = None
    sigma_stop: float = 0.0
    sigma_stop_8bit: float | None = 1.0
    initial_fill: float = 0.5
    inpaint_fraction: float = 0.25
    random_keep_fraction: float = 0.10
    superres_factor: int = 4
    lowpass_fraction: float = 0.05
    save_images: bool = True
    save_gifs: bool = False
    gif_every: int = 25
    gif_duration_ms: int = 80
    show_step_progress: bool = False
    seed: int = 0
    device: str = "auto"


DEFAULT_CONFIG = SamplerConfig()


@dataclass
class ProblemState:
    observed_projection: torch.Tensor | None
    observed_image: torch.Tensor
    initial: torch.Tensor


@dataclass
class StepRecord:
    step: int
    h: float
    sigma_eff: float
    gamma: float
    psnr_db: float
    ssim: float
    data_rmse: float | None
    raw_data_rmse: float | None


NUMERIC_AGGREGATE_FIELDS = (
    "observed_psnr_db",
    "observed_ssim",
    "initial_psnr_db",
    "initial_ssim",
    "one_pass_psnr_db",
    "one_pass_ssim",
    "final_psnr_db",
    "final_ssim",
    "best_psnr_db",
    "best_ssim",
    "final_minus_best_psnr_db",
    "final_minus_one_pass_psnr_db",
    "best_minus_one_pass_psnr_db",
    "initial_data_rmse",
    "one_pass_data_rmse",
    "final_data_rmse",
    "best_data_rmse",
    "initial_raw_data_rmse",
    "final_raw_data_rmse",
    "best_raw_data_rmse",
    "final_step",
    "best_step",
    "final_sigma_eff",
)


class DenoisingProblem:
    name = "denoising"
    observed_label = "noisy"
    is_constrained = False

    def make_state(
        self,
        clean: torch.Tensor,
        cfg: SamplerConfig,
        generator: torch.Generator,
        device: torch.device,
    ) -> ProblemState:
        sigma = resolved_initial_sigma_8bit(cfg) / 255.0
        noisy = clean + randn_like_cpu_seeded(clean, generator, device) * sigma
        return ProblemState(
            observed_projection=None,
            observed_image=noisy,
            initial=noisy,
        )

    def direction(
        self,
        y: torch.Tensor,
        denoised: torch.Tensor,
        observed_projection: torch.Tensor | None,
    ) -> torch.Tensor:
        return denoised - y

    def one_pass(
        self,
        denoised: torch.Tensor,
        observed_projection: torch.Tensor | None,
    ) -> torch.Tensor:
        return denoised

    def final_output(
        self,
        x: torch.Tensor,
        observed_projection: torch.Tensor | None,
    ) -> torch.Tensor:
        return x

    def data_rmse(
        self,
        x: torch.Tensor,
        observed_projection: torch.Tensor | None,
    ) -> float | None:
        return None

    def payload(self) -> dict[str, Any]:
        return {"name": self.name, "is_constrained": self.is_constrained}


class ProjectionProblem:
    observed_label = "observed"
    is_constrained = True

    def __init__(self, name: str, parameters: dict[str, Any] | None = None) -> None:
        self.name = name
        self.parameters = parameters or {}

    def project(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def null_project(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.project(x)

    def make_state(
        self,
        clean: torch.Tensor,
        cfg: SamplerConfig,
        generator: torch.Generator,
        device: torch.device,
    ) -> ProblemState:
        observed_projection = self.project(clean)
        fill = torch.full_like(clean, cfg.initial_fill)
        initial_mean = observed_projection + self.null_project(fill)
        sigma = resolved_initial_sigma_8bit(cfg) / 255.0
        initial = initial_mean + randn_like_cpu_seeded(clean, generator, device) * sigma
        return ProblemState(
            observed_projection=observed_projection,
            observed_image=observed_projection,
            initial=initial,
        )

    def direction(
        self,
        y: torch.Tensor,
        denoised: torch.Tensor,
        observed_projection: torch.Tensor | None,
    ) -> torch.Tensor:
        if observed_projection is None:
            raise ValueError(f"{self.name} requires an observed projection.")
        residual = denoised - y
        prior_direction = self.null_project(residual)
        data_direction = observed_projection - self.project(y)
        return prior_direction + data_direction

    def enforce_measurement(
        self,
        x: torch.Tensor,
        observed_projection: torch.Tensor | None,
    ) -> torch.Tensor:
        if observed_projection is None:
            raise ValueError(f"{self.name} requires an observed projection.")
        return self.null_project(x) + observed_projection

    def one_pass(
        self,
        denoised: torch.Tensor,
        observed_projection: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.enforce_measurement(denoised, observed_projection)

    def final_output(
        self,
        x: torch.Tensor,
        observed_projection: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.enforce_measurement(x, observed_projection)

    def data_rmse(
        self,
        x: torch.Tensor,
        observed_projection: torch.Tensor | None,
    ) -> float | None:
        if observed_projection is None:
            raise ValueError(f"{self.name} requires an observed projection.")
        error = self.project(x) - observed_projection
        return float(torch.linalg.vector_norm(error).item()) / math.sqrt(error.numel())

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "is_constrained": self.is_constrained,
            "parameters": self.parameters,
        }


class MaskProjectionProblem(ProjectionProblem):
    def __init__(self, name: str, mask: torch.Tensor, parameters: dict[str, Any]) -> None:
        super().__init__(name=name, parameters=parameters)
        self.mask = mask

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.mask * x


class BlockSuperResolutionProblem(ProjectionProblem):
    def __init__(self, factor: int) -> None:
        super().__init__(
            name="block_superres",
            parameters={"factor": factor},
        )
        self.factor = factor

    def project(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        if height % self.factor != 0 or width % self.factor != 0:
            raise ValueError(
                f"Image shape {(height, width)} must be divisible by "
                f"superres_factor={self.factor}."
            )
        low = F.avg_pool2d(x, kernel_size=self.factor, stride=self.factor)
        return F.interpolate(low, scale_factor=self.factor, mode="nearest")


class FourierLowpassProblem(ProjectionProblem):
    def __init__(self, mask: torch.Tensor, retained_fraction: float) -> None:
        super().__init__(
            name="fourier_lowpass",
            parameters={"retained_fraction": retained_fraction},
        )
        self.mask = mask

    def project(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
        return torch.fft.ifft2(spectrum * self.mask, dim=(-2, -1), norm="ortho").real


def validate_config(cfg: SamplerConfig) -> None:
    if cfg.model_key not in models_log:
        raise KeyError(
            f"Unknown model key '{cfg.model_key}'. Available keys: {sorted(models_log)}"
        )
    if cfg.problem not in SUPPORTED_PROBLEMS:
        raise KeyError(
            f"Unknown problem '{cfg.problem}'. Available problems: {SUPPORTED_PROBLEMS}"
        )
    if cfg.image_path is None and not cfg.data_dir.expanduser().is_dir():
        raise FileNotFoundError(f"Data directory {cfg.data_dir} does not exist.")
    if cfg.image_path is not None and not cfg.image_path.expanduser().is_file():
        raise FileNotFoundError(f"Image path {cfg.image_path} does not exist.")
    if cfg.max_images is not None and cfg.max_images < 1:
        raise ValueError("max_images must be at least 1 when provided.")
    if cfg.sigma_8bit < 0.0:
        raise ValueError("sigma_8bit must be nonnegative.")
    if cfg.initial_sigma_8bit is not None and cfg.initial_sigma_8bit < 0.0:
        raise ValueError("initial_sigma_8bit must be nonnegative when provided.")
    if cfg.num_steps < 1:
        raise ValueError("num_steps must be at least 1.")
    if not 0.0 <= cfg.h0 <= 1.0:
        raise ValueError("h0 must lie in [0, 1].")
    if cfg.h_max is not None and not 0.0 < cfg.h_max <= 1.0:
        raise ValueError("h_max must lie in (0, 1] when provided.")
    if cfg.beta is not None and not 0.0 <= cfg.beta <= 1.0:
        raise ValueError("beta must lie in [0, 1].")
    if cfg.sigma_stop < 0.0:
        raise ValueError("sigma_stop must be nonnegative.")
    if cfg.sigma_stop_8bit is not None and cfg.sigma_stop_8bit < 0.0:
        raise ValueError("sigma_stop_8bit must be nonnegative when provided.")
    if not 0.0 <= cfg.initial_fill <= 1.0:
        raise ValueError("initial_fill must lie in [0, 1].")
    if not 0.0 < cfg.inpaint_fraction < 1.0:
        raise ValueError("inpaint_fraction must lie in (0, 1).")
    if not 0.0 < cfg.random_keep_fraction <= 1.0:
        raise ValueError("random_keep_fraction must lie in (0, 1].")
    if cfg.superres_factor < 2:
        raise ValueError("superres_factor must be at least 2.")
    if not 0.0 < cfg.lowpass_fraction <= 1.0:
        raise ValueError("lowpass_fraction must lie in (0, 1].")
    if cfg.gif_every < 1:
        raise ValueError("gif_every must be at least 1.")
    if cfg.gif_duration_ms < 1:
        raise ValueError("gif_duration_ms must be at least 1.")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def path_payload(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.expanduser().resolve())
    except FileNotFoundError:
        return str(path.expanduser())


def safe_path_token(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    return "".join(char if char in allowed else "_" for char in value)


def resolve_model_log_dir(model_key: str) -> Path:
    return (Path(PROJECT_ROOT) / models_log[model_key]).expanduser().resolve()


def resolve_image_paths(cfg: SamplerConfig) -> list[Path]:
    if cfg.image_path is not None:
        return [cfg.image_path.expanduser().resolve()]

    data_dir = cfg.data_dir.expanduser().resolve()
    return [
        Path(path).resolve()
        for path in list_image_files([str(data_dir)], max_images=cfg.max_images)
    ]


def resolved_initial_sigma_8bit(cfg: SamplerConfig) -> float:
    if cfg.initial_sigma_8bit is not None:
        return cfg.initial_sigma_8bit
    if cfg.problem == "denoising":
        return cfg.sigma_8bit
    return DEFAULT_INVERSE_INITIAL_SIGMA_8BIT


def resolved_beta(cfg: SamplerConfig) -> float:
    if cfg.beta is not None:
        return cfg.beta
    if cfg.problem == "denoising":
        return 1.0
    return 0.01


def problem_tag(cfg: SamplerConfig) -> str:
    if cfg.problem == "inpaint_center":
        return f"{cfg.problem}_{cfg.inpaint_fraction:g}"
    if cfg.problem == "random_missing":
        return f"{cfg.problem}_keep{cfg.random_keep_fraction:g}"
    if cfg.problem == "block_superres":
        return f"{cfg.problem}_x{cfg.superres_factor}"
    if cfg.problem == "fourier_lowpass":
        return f"{cfg.problem}_keep{cfg.lowpass_fraction:g}"
    return cfg.problem


def resolve_output_dir(cfg: SamplerConfig) -> Path:
    if cfg.output_dir is not None:
        return cfg.output_dir.expanduser().resolve()

    model_log_dir = resolve_model_log_dir(cfg.model_key)
    if cfg.image_path is None:
        data_tag = safe_path_token(cfg.data_dir.name)
    else:
        data_tag = safe_path_token(cfg.image_path.stem)
    initial_sigma = resolved_initial_sigma_8bit(cfg)
    if cfg.problem == "denoising" and math.isclose(initial_sigma, cfg.sigma_8bit):
        sigma_tag = f"sigma{cfg.sigma_8bit:g}".replace(".", "p")
    else:
        sigma_tag = f"init{initial_sigma:g}".replace(".", "p")
    return (
        DEFAULT_OUTPUT_ROOT
        / safe_path_token(cfg.model_key)
        / safe_path_token(model_log_dir.name)
        / f"{data_tag}_{safe_path_token(problem_tag(cfg))}_{sigma_tag}_seed{cfg.seed}"
    ).resolve()


def config_payload(cfg: SamplerConfig, output_dir: Path) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["data_dir"] = path_payload(cfg.data_dir)
    payload["image_path"] = path_payload(cfg.image_path)
    payload["output_dir"] = path_payload(output_dir)
    payload["initial_sigma_8bit"] = resolved_initial_sigma_8bit(cfg)
    payload["beta"] = resolved_beta(cfg)
    return payload


def h_schedule(step: int, h0: float) -> float:
    if h0 == 0.0:
        return 0.0
    return float((h0 * step) / (1.0 + h0 * (step - 1)))


def effective_h(step: int, cfg: SamplerConfig) -> float:
    h = h_schedule(step, cfg.h0)
    if cfg.h_max is not None:
        h = min(h, cfg.h_max)
    return h


def gamma_from_schedule(h: float, beta: float, sigma_eff: float) -> float:
    coefficient = (1.0 - beta * h) ** 2 - (1.0 - h) ** 2
    gamma_sq = max(0.0, coefficient * sigma_eff * sigma_eff)
    return float(math.sqrt(gamma_sq))


def resolved_sigma_stop(cfg: SamplerConfig) -> float:
    thresholds = [cfg.sigma_stop]
    if cfg.sigma_stop_8bit is not None:
        thresholds.append(cfg.sigma_stop_8bit / 255.0)
    return max(thresholds)


def stop_reason_explanation(
    stop_reason: str,
    cfg: SamplerConfig,
    sigma_stop_threshold: float,
) -> str:
    if stop_reason == "completed":
        return f"Reached num_steps={cfg.num_steps} without an early-stop condition."
    if stop_reason == "sigma_stop":
        return (
            "Stopped because the sampler direction effective sigma "
            f"||d_t||/sqrt(N) fell below {sigma_stop_threshold:.6f} "
            f"({sigma_stop_threshold * 255.0:.3f} in 8-bit units). For constrained "
            "problems d_t is (I - P) f(y_t) + P x - P y_t."
        )
    return f"Stopped for unrecognized reason: {stop_reason}."


def randn_like_cpu_seeded(
    reference: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return torch.randn(
        tuple(reference.shape),
        generator=generator,
        dtype=reference.dtype,
    ).to(device)


def make_center_mask(clean: torch.Tensor, fraction: float) -> torch.Tensor:
    mask = torch.ones_like(clean)
    height, width = clean.shape[-2:]
    hole_h = max(1, int(round(height * fraction)))
    hole_w = max(1, int(round(width * fraction)))
    top = (height - hole_h) // 2
    left = (width - hole_w) // 2
    mask[..., top : top + hole_h, left : left + hole_w] = 0.0
    return mask


def make_random_mask(
    clean: torch.Tensor,
    keep_fraction: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    mask = torch.rand(tuple(clean.shape), generator=generator, dtype=clean.dtype)
    return (mask < keep_fraction).to(device=device, dtype=clean.dtype)


def make_fourier_lowpass_mask(
    clean: torch.Tensor,
    retained_fraction: float,
    device: torch.device,
) -> torch.Tensor:
    height, width = clean.shape[-2:]
    fy = torch.fft.fftfreq(height, device=device).view(height, 1)
    fx = torch.fft.fftfreq(width, device=device).view(1, width)
    radius_sq = fy.square() + fx.square()
    flat = radius_sq.flatten()
    k = max(1, min(flat.numel(), int(round(retained_fraction * flat.numel()))))
    threshold = torch.kthvalue(flat, k).values
    mask = (radius_sq <= threshold).to(dtype=clean.dtype)
    return mask.view(1, 1, height, width)


def build_problem(
    cfg: SamplerConfig,
    clean: torch.Tensor,
    image_index: int,
    device: torch.device,
) -> DenoisingProblem | ProjectionProblem:
    if cfg.problem == "denoising":
        return DenoisingProblem()

    if cfg.problem == "inpaint_center":
        return MaskProjectionProblem(
            name="inpaint_center",
            mask=make_center_mask(clean, cfg.inpaint_fraction),
            parameters={"missing_side_fraction": cfg.inpaint_fraction},
        )

    if cfg.problem == "random_missing":
        mask_seed = cfg.seed + 1_000_003 * (image_index + 1)
        return MaskProjectionProblem(
            name="random_missing",
            mask=make_random_mask(clean, cfg.random_keep_fraction, mask_seed, device),
            parameters={
                "keep_fraction": cfg.random_keep_fraction,
                "mask_seed": mask_seed,
            },
        )

    if cfg.problem == "block_superres":
        return BlockSuperResolutionProblem(factor=cfg.superres_factor)

    if cfg.problem == "fourier_lowpass":
        mask = make_fourier_lowpass_mask(clean, cfg.lowpass_fraction, device)
        retained = float(mask.mean().item())
        return FourierLowpassProblem(mask=mask, retained_fraction=retained)

    raise KeyError(f"Unsupported problem: {cfg.problem}")


def save_clamped(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_tensor_image(tensor.detach().cpu().clamp(0.0, 1.0), path)


def make_gif_frame(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clamp(0.0, 1.0).clone()


def image_metrics(clean: torch.Tensor, x: torch.Tensor) -> tuple[float, float]:
    return float(tensor_psnr(clean, x)), float(tensor_ssim(clean, x))


def tensor_rmse(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x).item()) / math.sqrt(x.numel())


def mean_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean, "std": 0.0}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return {"mean": mean, "std": math.sqrt(variance)}


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for field in NUMERIC_AGGREGATE_FIELDS:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        aggregate[field] = mean_std(values)
    aggregate["stop_reason_counts"] = dict(
        Counter(str(row["stop_reason"]) for row in rows)
    )
    return aggregate


def write_per_image_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sampler_direction_sigma(direction: torch.Tensor) -> float:
    return tensor_rmse(direction)


def evaluate_output(
    *,
    clean: torch.Tensor,
    raw: torch.Tensor,
    problem: DenoisingProblem | ProjectionProblem,
    observed_projection: torch.Tensor | None,
) -> tuple[torch.Tensor, float, float, float | None, float | None]:
    evaluated = problem.final_output(raw, observed_projection)
    psnr_db, ssim_value = image_metrics(clean, evaluated)
    data_rmse = problem.data_rmse(evaluated, observed_projection)
    raw_data_rmse = problem.data_rmse(raw, observed_projection)
    return evaluated, psnr_db, ssim_value, data_rmse, raw_data_rmse


def run_one_image(
    *,
    cfg: SamplerConfig,
    model: torch.nn.Module,
    image_path: Path,
    image_index: int,
    device: torch.device,
    images_dir: Path,
    gifs_dir: Path | None,
    sigma_stop_threshold: float,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(cfg.seed + image_index)

    clean = load_grayscale_tensor(image_path).to(device=device, dtype=torch.float32)
    problem = build_problem(cfg, clean, image_index, device)
    state = problem.make_state(clean, cfg, generator, device)
    y = state.initial

    observed_psnr, observed_ssim = image_metrics(clean, state.observed_image)
    (
        initial_eval,
        initial_psnr,
        initial_ssim,
        initial_data_rmse,
        initial_raw_data_rmse,
    ) = evaluate_output(
        clean=clean,
        raw=y,
        problem=problem,
        observed_projection=state.observed_projection,
    )

    records: list[StepRecord] = [
        StepRecord(
            step=0,
            h=0.0,
            sigma_eff=float("nan"),
            gamma=0.0,
            psnr_db=initial_psnr,
            ssim=initial_ssim,
            data_rmse=initial_data_rmse,
            raw_data_rmse=initial_raw_data_rmse,
        )
    ]

    gif_frames: list[torch.Tensor] = []
    last_gif_step: int | None = None
    if cfg.save_gifs:
        gif_frames.append(make_gif_frame(initial_eval))
        last_gif_step = 0

    stem = safe_path_token(image_path.stem)
    if cfg.save_images:
        save_clamped(
            state.observed_image,
            images_dir / f"{stem}_{problem.observed_label}.png",
        )

    stop_reason = "completed"
    stopped_early = False
    final_step = 0

    with torch.inference_mode():
        denoised_initial = model(y)
        one_pass = problem.one_pass(denoised_initial, state.observed_projection)
        one_pass_psnr, one_pass_ssim = image_metrics(clean, one_pass)
        one_pass_data_rmse = problem.data_rmse(one_pass, state.observed_projection)
        if cfg.save_images:
            save_clamped(one_pass, images_dir / f"{stem}_one_pass.png")

        step_iter = tqdm(
            range(1, cfg.num_steps + 1),
            desc=image_path.name,
            unit="step",
            leave=False,
            disable=not cfg.show_step_progress,
        )
        for step in step_iter:
            denoised = model(y)
            direction = problem.direction(y, denoised, state.observed_projection)
            sigma_eff = sampler_direction_sigma(direction)

            if sigma_stop_threshold > 0.0 and sigma_eff <= sigma_stop_threshold:
                stopped_early = True
                stop_reason = "sigma_stop"
                break

            h = effective_h(step, cfg)
            gamma = gamma_from_schedule(h, resolved_beta(cfg), sigma_eff)
            y = y + h * direction + gamma * randn_like_cpu_seeded(y, generator, device)

            (
                eval_y,
                psnr_db,
                ssim_value,
                data_rmse,
                raw_data_rmse,
            ) = evaluate_output(
                clean=clean,
                raw=y,
                problem=problem,
                observed_projection=state.observed_projection,
            )
            final_step = step
            records.append(
                StepRecord(
                    step=step,
                    h=h,
                    sigma_eff=sigma_eff,
                    gamma=gamma,
                    psnr_db=psnr_db,
                    ssim=ssim_value,
                    data_rmse=data_rmse,
                    raw_data_rmse=raw_data_rmse,
                )
            )

            if cfg.save_gifs and step % cfg.gif_every == 0:
                gif_frames.append(make_gif_frame(eval_y))
                last_gif_step = step

            postfix: dict[str, str] = {
                "psnr": f"{psnr_db:.2f}",
                "sigma_eff": f"{sigma_eff:.3e}",
            }
            if data_rmse is not None:
                postfix["data"] = f"{data_rmse * 255.0:.2f}"
            if raw_data_rmse is not None:
                postfix["raw_data"] = f"{raw_data_rmse * 255.0:.2f}"
            step_iter.set_postfix(postfix)

    with torch.inference_mode():
        final_denoised = model(y)
        final_direction = problem.direction(y, final_denoised, state.observed_projection)
        final_sigma_eff = sampler_direction_sigma(final_direction)
    (
        final,
        final_psnr,
        final_ssim,
        final_data_rmse,
        final_raw_data_rmse,
    ) = evaluate_output(
        clean=clean,
        raw=y,
        problem=problem,
        observed_projection=state.observed_projection,
    )
    final_record = StepRecord(
        step=final_step,
        h=0.0,
        sigma_eff=final_sigma_eff,
        gamma=0.0,
        psnr_db=final_psnr,
        ssim=final_ssim,
        data_rmse=final_data_rmse,
        raw_data_rmse=final_raw_data_rmse,
    )
    if records[-1].step == final_step:
        records[-1] = final_record
    else:
        records.append(final_record)
    if cfg.save_images:
        save_clamped(final, images_dir / f"{stem}_final.png")

    gif_path = None
    if cfg.save_gifs:
        if last_gif_step != final_step:
            gif_frames.append(make_gif_frame(final))
        if gifs_dir is None:
            raise ValueError("gifs_dir is required when save_gifs=True.")
        gif_path = gifs_dir / f"{stem}_trajectory.gif"
        save_tensor_gif(
            gif_frames,
            gif_path,
            duration_ms=cfg.gif_duration_ms,
        )

    best_record = max(records, key=lambda record: record.psnr_db)
    return {
        "image": image_path.name,
        "problem": problem.name,
        "noise_seed": cfg.seed + image_index,
        "observed_psnr_db": observed_psnr,
        "observed_ssim": observed_ssim,
        "initial_psnr_db": initial_psnr,
        "initial_ssim": initial_ssim,
        "one_pass_psnr_db": one_pass_psnr,
        "one_pass_ssim": one_pass_ssim,
        "final_psnr_db": final_record.psnr_db,
        "final_ssim": final_record.ssim,
        "best_psnr_db": best_record.psnr_db,
        "best_ssim": best_record.ssim,
        "final_minus_best_psnr_db": final_record.psnr_db - best_record.psnr_db,
        "final_minus_one_pass_psnr_db": final_record.psnr_db - one_pass_psnr,
        "best_minus_one_pass_psnr_db": best_record.psnr_db - one_pass_psnr,
        "initial_data_rmse": initial_data_rmse,
        "one_pass_data_rmse": one_pass_data_rmse,
        "final_data_rmse": final_data_rmse,
        "best_data_rmse": best_record.data_rmse,
        "initial_raw_data_rmse": initial_raw_data_rmse,
        "final_raw_data_rmse": final_raw_data_rmse,
        "best_raw_data_rmse": best_record.raw_data_rmse,
        "final_step": final_step,
        "best_step": best_record.step,
        "final_sigma_eff": final_sigma_eff,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "gif_path": str(gif_path) if gif_path is not None else None,
    }


def run_sampler(cfg: SamplerConfig) -> dict[str, Any]:
    validate_config(cfg)
    device = resolve_device(cfg.device)
    output_dir = resolve_output_dir(cfg)
    images_dir = output_dir / "images"
    gifs_dir = output_dir / "gifs"
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_images:
        images_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_gifs:
        gifs_dir.mkdir(parents=True, exist_ok=True)

    image_paths = resolve_image_paths(cfg)
    if not image_paths:
        raise FileNotFoundError(f"No images found for config: {cfg}")

    model_log_dir = resolve_model_log_dir(cfg.model_key)
    loaded = load_model_for_eval(model_log_dir, device)
    model = loaded.model.eval()
    sigma_stop_threshold = resolved_sigma_stop(cfg)

    rows: list[dict[str, Any]] = []
    image_progress = tqdm(image_paths, desc="Set12 images", unit="image")
    for image_index, image_path in enumerate(image_progress):
        row = run_one_image(
            cfg=cfg,
            model=model,
            image_path=image_path,
            image_index=image_index,
            device=device,
            images_dir=images_dir,
            gifs_dir=gifs_dir if cfg.save_gifs else None,
            sigma_stop_threshold=sigma_stop_threshold,
        )
        rows.append(row)
        image_progress.set_postfix(
            final=f"{row['final_psnr_db']:.2f}",
            best=f"{row['best_psnr_db']:.2f}@{row['best_step']}",
            gap=f"{row['final_minus_best_psnr_db']:.2f}",
        )

    aggregate = aggregate_rows(rows)
    payload = {
        "config": config_payload(cfg, output_dir),
        "device": str(device),
        "model": {
            "key": cfg.model_key,
            "log_dir": str(loaded.log_dir),
            "checkpoint": str(loaded.checkpoint_path),
            "arch": loaded.cfg.model,
            "train_objective": loaded.cfg.train_objective,
            "wrapper_mode": loaded.cfg.model_cfg.wrapper_mode,
            "pred_mode": loaded.cfg.model_cfg.pred_mode,
            "train_min_noise_8bit": loaded.cfg.min_noise,
            "train_max_noise_8bit": loaded.cfg.max_noise,
        },
        "problem": {
            "name": cfg.problem,
            "supported": SUPPORTED_PROBLEMS,
            "tag": problem_tag(cfg),
            "algorithm": "Algorithm 1" if cfg.problem == "denoising" else "Algorithm 2",
        },
        "num_images": len(rows),
        "sigma_normalized": cfg.sigma_8bit / 255.0,
        "initial_sigma_8bit": resolved_initial_sigma_8bit(cfg),
        "initial_sigma_normalized": resolved_initial_sigma_8bit(cfg) / 255.0,
        "sigma_stop_normalized": sigma_stop_threshold,
        "stop_reason_explanation": {
            reason: stop_reason_explanation(reason, cfg, sigma_stop_threshold)
            for reason in sorted(aggregate["stop_reason_counts"])
        },
        "aggregate": aggregate,
        "per_image": rows,
        "outputs": {
            "per_image_csv": str(output_dir / "per_image_metrics.csv"),
            "summary_json": str(output_dir / "summary.json"),
            "images_dir": str(images_dir) if cfg.save_images else None,
            "gifs_dir": str(gifs_dir) if cfg.save_gifs else None,
        },
        "notes": (
            "For constrained problems the sampler direction is "
            "d_t = (I - P) f(y_t) + P x - P y_t, where P = M M^T. "
            "Trajectory PSNR/SSIM and data_rmse are evaluated after projecting "
            "each raw sampler state onto the measurement affine set; raw_data_rmse "
            "logs the unprojected sampler state's measurement error. "
            "Only observed/noisy, one-pass constrained denoiser, and final sampler "
            "PNGs are saved by default. When save_gifs is enabled, per-image "
            "trajectory GIFs save the projected initial state, every gif_every "
            "projected sampler steps, and the projected final state. Best trajectory "
            "metrics are oracle diagnostics computed "
            "from clean Set12 targets; they do not affect the update or stopping."
        ),
    }

    write_per_image_csv(rows, output_dir / "per_image_metrics.csv")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    return payload


def metric_mean(payload: dict[str, Any], key: str) -> float:
    value = payload["aggregate"][key]["mean"]
    if value is None:
        return float("nan")
    return float(value)


def maybe_metric_mean(payload: dict[str, Any], key: str) -> float | None:
    value = payload["aggregate"][key]["mean"]
    if value is None:
        return None
    return float(value)


def main(config: SamplerConfig = DEFAULT_CONFIG) -> None:
    payload = run_sampler(config)
    aggregate = payload["aggregate"]

    print(f"Saved outputs to: {payload['config']['output_dir']}")
    print(f"Problem: {payload['problem']['name']} ({payload['problem']['algorithm']})")
    print(f"Model checkpoint: {payload['model']['checkpoint']}")
    print(
        "Initial sampler sigma: "
        f"{payload['initial_sigma_8bit']:.3f} in 8-bit units "
        f"({payload['initial_sigma_normalized']:.6f} normalized)"
    )
    print(f"Images: {payload['num_images']}")
    print(
        "Mean PSNR (observed / initial / one-pass / final / best): "
        f"{metric_mean(payload, 'observed_psnr_db'):.3f} / "
        f"{metric_mean(payload, 'initial_psnr_db'):.3f} / "
        f"{metric_mean(payload, 'one_pass_psnr_db'):.3f} / "
        f"{metric_mean(payload, 'final_psnr_db'):.3f} / "
        f"{metric_mean(payload, 'best_psnr_db'):.3f} dB"
    )
    print(
        "Mean stability gaps (final-best / final-one-pass / best-one-pass): "
        f"{metric_mean(payload, 'final_minus_best_psnr_db'):.3f} / "
        f"{metric_mean(payload, 'final_minus_one_pass_psnr_db'):.3f} / "
        f"{metric_mean(payload, 'best_minus_one_pass_psnr_db'):.3f} dB"
    )
    final_data = maybe_metric_mean(payload, "final_data_rmse")
    if final_data is not None:
        print(
            "Mean data RMSE in 8-bit units (initial / one-pass / final): "
            f"{metric_mean(payload, 'initial_data_rmse') * 255.0:.3f} / "
            f"{metric_mean(payload, 'one_pass_data_rmse') * 255.0:.3f} / "
            f"{final_data * 255.0:.3f}"
        )
        print(
            "Mean raw data RMSE in 8-bit units (initial / final): "
            f"{metric_mean(payload, 'initial_raw_data_rmse') * 255.0:.3f} / "
            f"{metric_mean(payload, 'final_raw_data_rmse') * 255.0:.3f}"
        )
    print(
        "Mean steps (final / best): "
        f"{metric_mean(payload, 'final_step'):.2f} / "
        f"{metric_mean(payload, 'best_step'):.2f}"
    )
    print(f"Stop reasons: {aggregate['stop_reason_counts']}")
    print(f"Per-image CSV: {payload['outputs']['per_image_csv']}")
    print(f"Summary JSON: {payload['outputs']['summary_json']}")
    if payload["outputs"]["gifs_dir"] is not None:
        print(f"GIFs: {payload['outputs']['gifs_dir']}")


if __name__ == "__main__":
    main()
