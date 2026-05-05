import math
from typing import Iterable

import torch

from se.configs import NoiseType

VALID_NOISE_TYPES: set[str] = {"gaussian", "laplace", "uniform", "rayleigh"}


def _normalized_noise_type(noise_type: str | NoiseType) -> NoiseType:
    candidate = str(noise_type).lower()
    if candidate not in VALID_NOISE_TYPES:
        raise ValueError(
            f"Unsupported noise_type '{noise_type}'. "
            f"Expected one of {sorted(VALID_NOISE_TYPES)}."
        )
    return candidate  # type: ignore[return-value]


def _sample_unit_noise(data: torch.Tensor, noise_type: NoiseType) -> torch.Tensor:
    """Return noise with unit standard deviation for the requested distribution."""
    if noise_type == "gaussian":
        return torch.randn_like(data)

    if noise_type == "laplace":
        dist = torch.distributions.Laplace(
            loc=torch.zeros((), device=data.device, dtype=data.dtype),
            scale=torch.ones((), device=data.device, dtype=data.dtype),
        )
        return dist.rsample(data.shape) / math.sqrt(2.0)  # std=1

    if noise_type == "uniform":
        # Uniform in [-sqrt(3), sqrt(3)] has std=1
        return (torch.rand_like(data) * 2.0 - 1.0) * math.sqrt(3.0)

    # Rayleigh(scale=1) has std = sqrt((4 - pi) / 2); sample via inverse CDF and scale to unit std.
    u = torch.rand_like(data).clamp_min(torch.finfo(data.dtype).tiny)
    rayleigh_scale1 = torch.sqrt(-2.0 * torch.log1p(-u))
    return rayleigh_scale1 / math.sqrt((4.0 - math.pi) / 2.0)


def _resolve_scales(
    min_noise: float, max_noise: float, batch_shape: Iterable[int], device, dtype
) -> torch.Tensor:
    """
    Returns per-sample scales (std-deviations) of shape [B,1,1,...] in the same
    device/dtype as the input.
    """
    scales = torch.rand(tuple(batch_shape), device=device, dtype=dtype)
    return min_noise + (max_noise - min_noise) * scales


def _resolve_noise_scales_for_data(
    data: torch.Tensor,
    min_noise: float,
    max_noise: float,
) -> torch.Tensor:
    if abs(max_noise - min_noise) < 2 / 255.0:
        noise_std = (min_noise + max_noise) / 2.0
        batch_shape = (data.shape[0], *[1] * (data.ndim - 1))
        return torch.full(
            batch_shape,
            fill_value=noise_std,
            device=data.device,
            dtype=data.dtype,
        )

    batch_shape = (data.shape[0], *[1] * (data.ndim - 1))
    return _resolve_scales(
        min_noise=min_noise,
        max_noise=max_noise,
        batch_shape=batch_shape,
        device=data.device,
        dtype=data.dtype,
    )


def get_noise(
    data: torch.Tensor,
    min_noise: float = 5.0 / 255.0,
    max_noise: float = 55.0 / 255.0,
    noise_type: NoiseType = "gaussian",
) -> torch.Tensor:
    """
    Draw additive noise with user-selected distribution and variance range.

    - If |max_noise - min_noise| < 2/255, a single sigma is used for the batch ("S" mode).
    - Otherwise, each sample draws its own sigma uniformly in [min_noise, max_noise] ("B" mode).

    Args:
        data: Reference tensor to match shape/device/dtype.
        min_noise: Lower bound on target standard deviation (image domain).
        max_noise: Upper bound on target standard deviation (image domain).
        noise_type: Distribution family: gaussian | laplace | uniform | rayleigh.
    """
    resolved_type = _normalized_noise_type(noise_type)
    base = _sample_unit_noise(data, resolved_type)

    scales = _resolve_noise_scales_for_data(
        data=data,
        min_noise=min_noise,
        max_noise=max_noise,
    )
    return base * scales


def get_noise_pair(
    data: torch.Tensor,
    min_noise: float = 5.0 / 255.0,
    max_noise: float = 55.0 / 255.0,
    noise_type: NoiseType = "gaussian",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Draw two independent additive noise samples that share the same per-sample
    standard deviation. This is the canonical synthetic N2N setup for fixed- or
    range-based sigma training.
    """
    resolved_type = _normalized_noise_type(noise_type)
    scales = _resolve_noise_scales_for_data(
        data=data,
        min_noise=min_noise,
        max_noise=max_noise,
    )
    noise_first = _sample_unit_noise(data, resolved_type) * scales
    noise_second = _sample_unit_noise(data, resolved_type) * scales
    return noise_first, noise_second
