"""
Render a SwinIR denoising comparison between a base model and its
norm-equivariant (wrapper) counterpart. The script crops the Set68
`test014` image to a square, adds training-level noise and a heavier
test noise, denoises with both models, and writes a four-panel PDF.
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

# Use a bundled serif font; keep LaTeX disabled to avoid external dependency
plt.rcParams.update(
    {
        # "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
    }
)

model_name = "swinir"
eq_type = "wne"  # norm-equivariant wrapper
TRAIN_SIGMA_8BIT = 10
BASE_KEY = f"b_{model_name}_{TRAIN_SIGMA_8BIT}"
EQ_KEY = f"{eq_type}_{model_name}_{TRAIN_SIGMA_8BIT}"

comparison_name = f"{BASE_KEY}_vs_{EQ_KEY}"
# IMAGE_PATH = Path(f"{PROJECT_ROOT}/data/Set68/test053.png")
IMAGE_PATH = Path(f"{PROJECT_ROOT}/data/Set12/07.png")
OUTPUT_PATH = Path(f"{PROJECT_ROOT}/artifacts/{comparison_name}_comparison.pdf")
TEST_SIGMA_8BIT = 90  # stronger than training noise to stress generalization
DATASET_MODE = "s"  # grayscale conversion consistent with psnr_plot


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
    # Match psnr_plot: additive AWGN without clamping before denoising
    return img + torch.randn_like(img) * sigma


def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    return t.squeeze(0).squeeze(0).cpu().numpy()


def make_figure() -> tuple[Path, dict]:
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, base_cfg, base_ckpt = build_model_from_key(BASE_KEY, device)
    eq_model, eq_cfg, eq_ckpt = build_model_from_key(EQ_KEY, device)

    clean = load_clean_tensor(IMAGE_PATH, device)
    print(clean.shape)
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
            tensor_to_img(noisy_train),
            f"Noisy training image\nσ = {train_sigma_8bit}",
        ),
        (tensor_to_img(noisy_test), f"Noisy test image\nσ = {TEST_SIGMA_8BIT}"),
        (tensor_to_img(denoised_base), "Test image denoised by\nSwinIR"),
        (tensor_to_img(denoised_wne), "Test image denoised by\nSwinIR-WNE (ours)"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(9, 2.6))
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
        "base_key": BASE_KEY,
        "wne_key": EQ_KEY,
        "base_log": str(models_log[BASE_KEY]),
        "wne_log": str(models_log[EQ_KEY]),
        "base_checkpoint": base_ckpt.name,
        "wne_checkpoint": eq_ckpt.name,
        "train_sigma_8bit": train_sigma_8bit,
        "test_sigma_8bit": TEST_SIGMA_8BIT,
        "image": str(IMAGE_PATH),
        "output": str(OUTPUT_PATH),
    }
    return OUTPUT_PATH, metadata


if __name__ == "__main__":
    out, meta = make_figure()
    print(f"Saved figure to: {out}")
    for k, v in meta.items():
        print(f"{k}: {v}")
