from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch


FloatImageArray = NDArray[np.float32]
GifDuration = int | Sequence[int]


def load_grayscale_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def tensor_to_image_array(tensor: torch.Tensor) -> FloatImageArray:
    image = tensor.detach().cpu().clamp(0.0, 1.0)
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got shape {tuple(image.shape)}.")
        image = image.squeeze(0)

    if image.ndim == 3:
        if image.shape[0] == 1:
            array = image.squeeze(0).numpy()
            return np.asarray(array, dtype=np.float32)
        if image.shape[0] == 3:
            array = image.permute(1, 2, 0).numpy()
            return np.asarray(array, dtype=np.float32)
        raise ValueError(f"Expected 1 or 3 channels, got shape {tuple(image.shape)}.")

    if image.ndim == 2:
        return np.asarray(image.numpy(), dtype=np.float32)

    raise ValueError(f"Unsupported tensor shape {tuple(image.shape)}.")


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    tensor_to_pil_image(tensor).save(path)


def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    array = (tensor_to_image_array(tensor) * 255.0).round().astype(np.uint8)
    if array.ndim == 2:
        return Image.fromarray(array, mode="L")
    if array.ndim == 3 and array.shape[2] == 3:
        return Image.fromarray(array, mode="RGB")
    raise ValueError(f"Unsupported image array shape {array.shape}.")


def save_tensor_gif(
    frames: list[torch.Tensor],
    path: Path,
    *,
    duration_ms: GifDuration,
    loop: int = 0,
) -> None:
    if not frames:
        raise ValueError("Cannot save a GIF without at least one frame.")

    images = [tensor_to_pil_image(frame) for frame in frames]
    save_pil_gif(images, path, duration_ms=duration_ms, loop=loop)


def save_image_files_gif(
    image_paths: Sequence[Path],
    path: Path,
    *,
    duration_ms: GifDuration,
    loop: int = 0,
) -> None:
    if not image_paths:
        raise ValueError("Cannot save a GIF without at least one frame.")

    images: list[Image.Image] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.copy())
    save_pil_gif(images, path, duration_ms=duration_ms, loop=loop)


def save_pil_gif(
    images: list[Image.Image],
    path: Path,
    *,
    duration_ms: GifDuration,
    loop: int = 0,
) -> None:
    if not images:
        raise ValueError("Cannot save a GIF without at least one frame.")

    duration = validate_gif_duration(duration_ms, len(images))
    path.parent.mkdir(parents=True, exist_ok=True)
    first, *rest = images
    first.save(
        path,
        save_all=True,
        append_images=rest,
        duration=duration,
        loop=loop,
    )


def validate_gif_duration(duration_ms: GifDuration, frame_count: int) -> int | list[int]:
    if isinstance(duration_ms, int):
        if duration_ms < 1:
            raise ValueError("duration_ms must be at least 1.")
        return duration_ms

    durations = list(duration_ms)
    if len(durations) != frame_count:
        raise ValueError(
            f"Expected {frame_count} GIF durations, got {len(durations)}."
        )
    if any(duration < 1 for duration in durations):
        raise ValueError("All GIF durations must be at least 1.")
    return durations
