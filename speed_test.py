"""
Minimal speed benchmark for denoising architectures.

It instantiates fresh models, measures backward and inference time on a fixed
random batch, prints a small table, and saves the results under
./artifacts/speed_tests/.

Default targets cover scale-equivariant and norm-equivariant FDnCNN variants on
GPU and CPU.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import fmean, pstdev
from time import perf_counter
from typing import Iterable

import torch

from se.configs import ModelConfig, PROJECT_ROOT, TrainConfig
from se.models import build_model


# TARGETS = [
#     "gpu-wne=wne_fdncnn@cuda",
#     "cpu-wne=wne_fdncnn@cpu",
#     "gpu-baseline=b_fdncnn@cuda",
#     "cpu-baseline=b_fdncnn@cpu",
#     "gpu-norm-equiv=ne_fdncnn@cuda",
#     "cpu-norm-equiv=ne_fdncnn@cpu",
# ]

TARGETS = [
    "gpu-wne=wne_swinir@cuda",
    "cpu-wne=wne_swinir@cpu",
    "gpu-baseline=b_swinir@cuda",
    "cpu-baseline=b_swinir@cpu",
]


@dataclass
class TargetSpec:
    key: str
    device: str
    label: str


@dataclass(frozen=True)
class ModelPreset:
    model: str
    model_cfg: ModelConfig
    description: str = ""


@dataclass
class BenchSettings:
    batch_size: int
    height: int
    width: int
    warmup: int
    steps: int
    skip_backward: bool


@dataclass
class BenchmarkResult:
    label: str
    key: str
    device: str
    model_name: str
    model_mode: str
    wrapper_mode: str
    pred_mode: str
    log_dir: str | None
    checkpoint: str | None
    weights_loaded: bool
    backward_s: float | None
    backward_std_s: float | None
    inference_s: float
    inference_std_s: float


MODEL_PRESETS: dict[str, ModelPreset] = {
    "se_fdncnn": ModelPreset(
        model="fdncnn",
        model_cfg=ModelConfig(
            model_mode="scale-equiv",
            wrapper_mode="idem",
            pred_mode="direct",
        ),
        description="Scale-equivariant FDnCNN (direct prediction)",
    ),
    "ne_fdncnn": ModelPreset(
        model="fdncnn",
        model_cfg=ModelConfig(
            model_mode="norm-equiv",
            wrapper_mode="idem",
            pred_mode="direct",
        ),
        description="Norm-equivariant FDnCNN (direct prediction)",
    ),
    "wne_fdncnn": ModelPreset(
        model="fdncnn",
        model_cfg=ModelConfig(
            model_mode="ordinary",
            wrapper_mode="norm-equiv",
            pred_mode="direct",
        ),
        description="Wrapper norm-equivariant FDnCNN (direct prediction)",
    ),
    "b_fdncnn": ModelPreset(
        model="fdncnn",
        model_cfg=ModelConfig(
            model_mode="ordinary",
            wrapper_mode="idem",
            pred_mode="direct",
        ),
        description="Baseline FDnCNN (direct prediction)",
    ),
    "wne_swinir": ModelPreset(
        model="swinir",
        model_cfg=ModelConfig(
            model_mode="ordinary",
            wrapper_mode="norm-equiv",
            pred_mode="direct",
        ),
        description="Wrapper norm-equivariant SwinIR (direct prediction)",
    ),
    "b_swinir": ModelPreset(
        model="swinir",
        model_cfg=ModelConfig(
            model_mode="ordinary",
            wrapper_mode="idem",
            pred_mode="direct",
        ),
        description="Baseline SwinIR (direct prediction)",
    ),
}


@dataclass
class SpeedTestConfig:
    targets: list[str] = field(default_factory=lambda: TARGETS.copy())
    batch_size: int = 32
    height: int = 64
    width: int = 64
    warmup: int = 3
    steps: int = 10
    skip_backward: bool = False
    save_name: str | None = None

    def bench_settings(self) -> BenchSettings:
        return BenchSettings(
            batch_size=self.batch_size,
            height=self.height,
            width=self.width,
            warmup=self.warmup,
            steps=self.steps,
            skip_backward=self.skip_backward,
        )


def parse_target(raw: str) -> TargetSpec:
    """Accepts forms like 'label=se_fdncnn@cuda' or 'se_fdncnn@cuda'."""
    if "=" in raw:
        label_part, remainder = raw.split("=", 1)
    else:
        label_part, remainder = None, raw

    if "@" in remainder:
        key, device = remainder.split("@", 1)
    else:
        key, device = remainder, "cuda"

    label = label_part or key
    return TargetSpec(key=key.strip(), device=device.strip(), label=label.strip())


def ensure_device(device_str: str) -> torch.device:
    requested = device_str.strip()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"Requested {requested} but CUDA is unavailable; falling back to CPU.",
            flush=True,
        )
        requested = "cpu"
    return torch.device(requested)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def normalize_key(key: str) -> str:
    """Drop trailing noise-level suffixes like '_50' if present."""
    parts = key.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return key


def resolve_preset(key: str) -> tuple[str, ModelPreset]:
    normalized = normalize_key(key)
    preset = MODEL_PRESETS.get(normalized)
    if preset is None:
        known = ", ".join(sorted(MODEL_PRESETS))
        raise KeyError(f"Unknown model key '{key}'. Available: {known}.")
    return normalized, preset


def build_fresh_model(target: TargetSpec, device: torch.device):
    normalized_key, preset = resolve_preset(target.key)

    cfg = TrainConfig(
        model=preset.model,
        model_cfg=deepcopy(preset.model_cfg),
        min_noise=0.0,
        max_noise=0.0,
    )
    model = build_model(cfg)
    model.to(device)
    model.eval()
    return model, cfg, normalized_key


def benchmark_backward(
    model: torch.nn.Module,
    device: torch.device,
    shape: tuple[int, int, int, int],
    warmup: int,
    steps: int,
) -> tuple[float, float]:
    model.train()
    base = torch.randn(shape, device=device)

    def step() -> None:
        inp = base.clone().detach().requires_grad_(True)
        out = model(inp)
        loss = out.mean()
        loss.backward()
        model.zero_grad(set_to_none=True)

    with torch.enable_grad():
        for _ in range(warmup):
            step()

    sync_device(device)
    times: list[float] = []
    for _ in range(steps):
        start = perf_counter()
        with torch.enable_grad():
            step()
        sync_device(device)
        times.append(perf_counter() - start)

    return fmean(times), pstdev(times)


def benchmark_inference(
    model: torch.nn.Module,
    device: torch.device,
    shape: tuple[int, int, int, int],
    warmup: int,
    steps: int,
) -> tuple[float, float]:
    model.eval()
    inp = torch.randn(shape, device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(inp)

    sync_device(device)
    times: list[float] = []
    with torch.inference_mode():
        for _ in range(steps):
            start = perf_counter()
            model(inp)
            sync_device(device)
            times.append(perf_counter() - start)

    return fmean(times), pstdev(times)


def run_benchmark(target: TargetSpec, settings: BenchSettings) -> BenchmarkResult:
    device = ensure_device(target.device)
    model, cfg, normalized_key = build_fresh_model(target, device)

    shape = (settings.batch_size, 1, settings.height, settings.width)
    backward_time, backward_std = (
        (None, None)
        if settings.skip_backward
        else benchmark_backward(model, device, shape, settings.warmup, settings.steps)
    )
    inference_time, inference_std = benchmark_inference(
        model, device, shape, settings.warmup, settings.steps
    )

    return BenchmarkResult(
        label=target.label,
        key=target.key,
        device=str(device),
        model_name=normalized_key,
        model_mode=cfg.model_cfg.model_mode,
        wrapper_mode=cfg.model_cfg.wrapper_mode,
        pred_mode=cfg.model_cfg.pred_mode,
        log_dir=None,
        checkpoint=None,
        weights_loaded=False,
        backward_s=backward_time,
        backward_std_s=backward_std,
        inference_s=inference_time,
        inference_std_s=inference_std,
    )


def format_table(results: Iterable[BenchmarkResult]) -> str:
    header = (
        f"{'label':18s} {'device':8s} {'bwd mean (s)':14s} {'bwd std (s)':13s} "
        f"{'inf mean (s)':14s} {'inf std (s)':13s}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        bwd = f"{r.backward_s:.4f}" if r.backward_s is not None else "skipped"
        bwd_std = (
            f"{r.backward_std_s:.4f}" if r.backward_std_s is not None else "skipped"
        )
        inf = f"{r.inference_s:.4f}"
        inf_std = f"{r.inference_std_s:.4f}"
        lines.append(
            f"{r.label:18s} {r.device:8s} {bwd:14s} {bwd_std:13s} "
            f"{inf:14s} {inf_std:13s}"
        )
    return "\n".join(lines)


def save_results(
    results: list[BenchmarkResult],
    settings: BenchSettings,
    save_name: str | None,
) -> Path:
    artifacts_dir = Path(PROJECT_ROOT) / "artifacts" / "speed_tests"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = save_name or f"speed_test_{timestamp}"

    table_text = format_table(results)
    text_path = artifacts_dir / f"{stem}.txt"
    text_path.write_text(
        table_text + "\n\n" + json.dumps(asdict(settings), indent=2) + "\n",
        encoding="utf-8",
    )

    payload = {
        "timestamp": timestamp,
        "settings": asdict(settings),
        "results": [asdict(r) for r in results],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    json_path = artifacts_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return json_path


def main(config: SpeedTestConfig | None = None) -> None:
    cfg = config or SpeedTestConfig()

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    targets = [parse_target(t) for t in cfg.targets]
    settings = cfg.bench_settings()

    print(
        f"Running speed test on batch={settings.batch_size} x 1 x "
        f"{settings.height} x {settings.width}, steps={settings.steps}, "
        f"warmup={settings.warmup}."
    )
    results = [run_benchmark(t, settings) for t in targets]

    table_text = format_table(results)
    print(table_text)
    save_path = save_results(results, settings, cfg.save_name)
    print(f"Saved artifacts to {save_path.parent} (table + JSON).")


if __name__ == "__main__":
    main()
