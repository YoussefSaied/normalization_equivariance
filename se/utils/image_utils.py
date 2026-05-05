from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch


FloatImageArray = NDArray[np.float32]


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
    array = (tensor_to_image_array(tensor) * 255.0).round().astype(np.uint8)
    if array.ndim == 2:
        Image.fromarray(array, mode="L").save(path)
        return
    if array.ndim == 3 and array.shape[2] == 3:
        Image.fromarray(array, mode="RGB").save(path)
        return
    raise ValueError(f"Unsupported image array shape {array.shape}.")


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
    duration_ms: int,
    loop: int = 0,
) -> None:
    if not frames:
        raise ValueError("Cannot save a GIF without at least one frame.")
    if duration_ms < 1:
        raise ValueError("duration_ms must be at least 1.")

    path.parent.mkdir(parents=True, exist_ok=True)
    images = [tensor_to_pil_image(frame) for frame in frames]
    first, *rest = images
    first.save(
        path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=loop,
    )
