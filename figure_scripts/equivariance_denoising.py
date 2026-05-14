"""
Render a SwinIR denoising comparison between a base model and its
norm-equivariant (wrapper) counterpart. The script crops the configured
image to a square, adds training-level noise and a heavier test noise,
denoises with both models, and writes the four panels as separate PNG images.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
from PIL import Image
import torch

from model_logs import models_log
from se.configs import PROJECT_ROOT
from se.utils.eval_utils import load_model_for_eval

model_name = "swinir"
eq_type = "wne"  # norm-equivariant wrapper
TRAIN_SIGMA_8BIT = 10
BASE_KEY = f"b_{model_name}_{TRAIN_SIGMA_8BIT}"
EQ_KEY = f"{eq_type}_{model_name}_{TRAIN_SIGMA_8BIT}"

comparison_name = f"{BASE_KEY}_vs_{EQ_KEY}"
# IMAGE_PATH = Path(f"{PROJECT_ROOT}/data/Set68/test053.png")
IMAGE_PATH = Path(f"{PROJECT_ROOT}/data/Set12/07.png")
OUTPUT_DIR = Path(f"{PROJECT_ROOT}/artifacts/{comparison_name}_comparison_pngs")
TEST_SIGMA_8BIT = 90  # stronger than training noise to stress generalization


def build_model_from_key(key: str, device: torch.device):
    loaded = load_model_for_eval(PROJECT_ROOT / models_log[key], device)
    return loaded.model, loaded.cfg, loaded.checkpoint_path


def square_crop_top_right(arr: np.ndarray) -> np.ndarray:
    """Crop to a square using (W-H, W) x (0, H) if W>H, mirrored if H>W."""
    h, w = arr.shape[:2]
    if w >= h:
        start_x = w - h
        return arr[0:h, start_x : start_x + h]
    start_y = h - w
    return arr[start_y : start_y + w, 0:w]


def load_clean_tensor(path: Path, device: torch.device) -> torch.Tensor:
    """Load image via OpenCV, crop square, convert to [1,1,H,W] tensor in [0,1]."""
    img_bgr = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    cropped = square_crop_top_right(img_bgr)
    grayscale = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    clean = torch.from_numpy(grayscale.astype("float32")) / 255.0
    return clean.unsqueeze(0).unsqueeze(0).to(device)


def add_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    # Match psnr_plot: additive AWGN without clamping before denoising
    return img + torch.randn_like(img) * sigma


def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    return t.squeeze(0).squeeze(0).cpu().numpy()


def save_grayscale_png(img: np.ndarray, path: Path) -> None:
    if img.ndim != 2:
        raise ValueError(f"Expected a grayscale 2D image, got shape {img.shape}.")
    array = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def save_panels() -> tuple[Path, dict]:
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, base_cfg, base_ckpt = build_model_from_key(BASE_KEY, device)
    eq_model, eq_cfg, eq_ckpt = build_model_from_key(EQ_KEY, device)

    clean = load_clean_tensor(IMAGE_PATH, device)
    train_sigma_8bit = int(max(base_cfg.min_noise, base_cfg.max_noise))
    train_sigma = train_sigma_8bit / 255.0
    test_sigma = TEST_SIGMA_8BIT / 255.0

    noisy_train = add_noise(clean, train_sigma)
    noisy_test = add_noise(clean, test_sigma)

    with torch.inference_mode():
        denoised_base = base_model(noisy_test).clamp(0.0, 1.0)
        denoised_wne = eq_model(noisy_test).clamp(0.0, 1.0)

    panels = [
        (
            f"noisy_train_sigma{train_sigma_8bit}.png",
            tensor_to_img(noisy_train),
        ),
        (f"noisy_test_sigma{TEST_SIGMA_8BIT}.png", tensor_to_img(noisy_test)),
        ("denoised_swinir.png", tensor_to_img(denoised_base)),
        ("denoised_swinir_wne.png", tensor_to_img(denoised_wne)),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_outputs = {}
    for filename, img in panels:
        path = OUTPUT_DIR / filename
        save_grayscale_png(img, path)
        panel_outputs[filename] = str(path)

    metadata = {
        "device": str(device),
        "base_key": BASE_KEY,
        "wne_key": EQ_KEY,
        "base_log": str(models_log[BASE_KEY]),
        "wne_log": str(models_log[EQ_KEY]),
        "base_checkpoint": base_ckpt.name,
        "wne_checkpoint": eq_ckpt.name,
        "train_sigma_8bit": train_sigma_8bit,
        "test_sigma_8bit": TEST_SIGMA_8BIT,
        "image": str(IMAGE_PATH),
        "output_dir": str(OUTPUT_DIR),
        "panel_outputs": panel_outputs,
    }
    return OUTPUT_DIR, metadata


if __name__ == "__main__":
    out, meta = save_panels()
    print(f"Saved panel PNGs to: {out}")
    for k, v in meta.items():
        print(f"{k}: {v}")
