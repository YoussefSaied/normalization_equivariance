"""
Denoise a Set12 image with two invariance-aware DnCNN variants.

The script loads Set12 image `10`, adds σ=10 AWGN, denoises with both the WNE
wrapper and the input-normalized WNEI wrapper ("DnCNN-IN"), and writes a
three-panel PDF comparing the noisy input, DnCNN-WNE output, and DnCNN-IN
output.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from model_logs import models_log
from se.configs import PROJECT_ROOT
from se.models import build_model
from se.utils.eval_utils import load_train_config, resolve_checkpoint_path
from se.utils.psnr_plot import to_tensor01

# Serif font, LaTeX disabled for portability
plt.rcParams.update(
    {
        # "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
    }
)

MODEL_NAME = "dncnn"
SIGMA_8BIT = 10
WNE_KEY = f"wne_{MODEL_NAME}_{SIGMA_8BIT}"
WNEI_KEY = f"wnei_{MODEL_NAME}_{SIGMA_8BIT}"
COMPARISON_NAME = f"{WNE_KEY}_vs_{WNEI_KEY}"
IMAGE_PATH = Path(f"{PROJECT_ROOT}/data/Set12/10.png")
OUTPUT_PATH = Path(f"{PROJECT_ROOT}/artifacts/{COMPARISON_NAME}_invariance.pdf")
DATASET_MODE = "s"  # grayscale conversion consistent with training/eval utils


def build_model_from_key(key: str, device: torch.device):
    log_dir = PROJECT_ROOT / models_log[key]
    cfg = load_train_config(log_dir / "config.json")
    ckpt = resolve_checkpoint_path(log_dir, epoch=None, explicit=None)

    model = build_model(cfg)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, cfg, ckpt


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
    clean = to_tensor01(cropped, mode=DATASET_MODE)
    return clean.to(device)


def add_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    # Add AWGN without clamping before denoising
    return img + torch.randn_like(img) * sigma


def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    return t.squeeze(0).squeeze(0).cpu().numpy()


def make_figure() -> tuple[Path, dict]:
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wne_model, wne_cfg, wne_ckpt = build_model_from_key(WNE_KEY, device)
    wnei_model, _wnei_cfg, wnei_ckpt = build_model_from_key(WNEI_KEY, device)

    clean = load_clean_tensor(IMAGE_PATH, device)
    train_sigma_8bit = int(max(wne_cfg.min_noise, wne_cfg.max_noise))
    noise_sigma = train_sigma_8bit / 255.0

    noisy = add_noise(clean, noise_sigma)

    with torch.inference_mode():
        denoised_wne = wne_model(noisy).clamp(0.0, 1.0)
        denoised_wnei = wnei_model(noisy).clamp(0.0, 1.0)

    panels = [
        (tensor_to_img(noisy), f"Noisy image\nσ = {train_sigma_8bit}"),
        (tensor_to_img(denoised_wne), "DnCNN-WNE"),
        (tensor_to_img(denoised_wnei), "DnCNN-IN"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 2.6))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")
        ax.text(
            0.5,
            -0.06,
            title,
            ha="center",
            va="top",
            fontsize=10,
            transform=ax.transAxes,
        )

    plt.tight_layout(rect=(0, 0.02, 1, 1))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "device": str(device),
        "wne_key": WNE_KEY,
        "wnei_key": WNEI_KEY,
        "wne_log": str(models_log[WNE_KEY]),
        "wnei_log": str(models_log[WNEI_KEY]),
        "wne_checkpoint": wne_ckpt.name,
        "wnei_checkpoint": wnei_ckpt.name,
        "sigma_8bit": train_sigma_8bit,
        "image": str(IMAGE_PATH),
        "output": str(OUTPUT_PATH),
    }
    return OUTPUT_PATH, metadata


if __name__ == "__main__":
    out, meta = make_figure()
    print(f"Saved figure to: {out}")
    for k, v in meta.items():
        print(f"{k}: {v}")
