"""
Replicate the adaptive-filter figure comparing SwinIR and its WNE wrapper.

Two noise levels (σ=25 train, σ=5 test) are applied to a cropped Set12 image.
For each level we show: noisy input, SwinIR denoised, SwinIR filters,
WNE-SwinIR denoised, and WNE filters. Each denoised panel reports PSNR and ρ
(filter literalness: ||f(y) - J_f(y) y|| / ||f(y)||). Filter panels show the
backpropagated Jacobian rows for two pixels (i, j) and (j, i), along with Σ (sum
of coefficients) for each.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import numpy as np
import torch

from model_logs import models_log
from se.configs import PROJECT_ROOT
from se.models import build_model
from se.utils.eval_utils import load_train_config, resolve_checkpoint_path
from se.utils.psnr_plot import psnr_from_mse, to_tensor01

# Serif font with LaTeX rendering for publication-ready text
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.unicode_minus": False,
    }
)

BORDER_W = 1.5
WSPACE = 0.10
HSPACE = -0.25
MODEL_PAD_MULTIPLIER = 2.0  # widen gaps around model pairs
BASE_KEY = "b_swinir_25"
WNE_KEY = "wne_swinir_25"
BASE_MODEL_LABEL = r"\textsc{Baseline}"
WNE_MODEL_LABEL = r"\textsc{WNE}"
NOISY_LABEL = r"\textsc{Noisy}"
# Vertical offset for group labels to separate them from column titles
GROUP_LABEL_Y_OFFSET = 0.06
IMAGE_PATH = Path(f"{PROJECT_ROOT}/data/Set12/02.png")
OUTPUT_PATH = Path(f"{PROJECT_ROOT}/artifacts/adaptive_filters_swinir.pdf")
NOISE_LEVELS_8BIT = (25, 5)
PIXELS: tuple[tuple[int, int], tuple[int, int]] = (
    (20, 80),
    (80, 20),
)  # (i,j) and (j,i)
DATASET_MODE = "s"  # grayscale conversion consistent with training/eval utils


def build_model_from_key(key: str, device: torch.device) -> torch.nn.Module:
    log_dir = PROJECT_ROOT / models_log[key]
    cfg = load_train_config(log_dir / "config.json")
    ckpt = resolve_checkpoint_path(log_dir, epoch=None, explicit=None)

    model = build_model(cfg)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_clean_tensor(path: Path, device: torch.device) -> torch.Tensor:
    """Load image via OpenCV, crop to 96x96 (top-left offset 50), to [1,1,H,W] in [0,1]."""
    img_bgr = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    cropped = img_bgr[50 : 50 + 96, 50 : 50 + 96]
    clean = to_tensor01(cropped, mode=DATASET_MODE)
    return clean.to(device)


def add_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    return img + torch.randn_like(img) * sigma


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    return t.squeeze(0).squeeze(0).detach().cpu().numpy()


def psnr_db(output: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((output - target) ** 2)
    return float(psnr_from_mse(mse))


def compute_adaptive_filter(
    model: torch.nn.Module, img_noise: torch.Tensor, i: int, j: int
) -> torch.Tensor:
    """Jacobian row: d output[i,j] / d input."""
    img_noisy = img_noise.clone().requires_grad_(True)
    model.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    den: torch.Tensor = model(img_noisy)
    den[0, 0, i, j].backward()
    grad = img_noisy.grad  # [1,1,H,W]
    assert isinstance(grad, torch.Tensor)
    return grad.detach()


def filter_literalness(
    model: torch.nn.Module, noisy: torch.Tensor
) -> tuple[float, torch.Tensor]:
    """
    ρ(y) = ||f(y) - J_f(y) y|| / ||f(y)||.
    Returns (rho, f(y)).
    """

    def fn(inp: torch.Tensor) -> torch.Tensor:
        return model(inp)

    inp = noisy.clone().requires_grad_(True)
    with torch.enable_grad():
        f_val, jvp = torch.autograd.functional.jvp(
            fn, (inp,), (inp,), create_graph=False, strict=True
        )
    num = torch.linalg.norm(
        (f_val - jvp).reshape(-1)  # type: ignore
    )  # ||f(y) - J_f(y) y||
    denom = torch.linalg.norm(f_val.reshape(-1)).clamp_min(1e-12)  # type: ignore
    return float((num / denom).item()), f_val.detach()  # type: ignore


def combine_filters(filters: Iterable[torch.Tensor]) -> np.ndarray:
    """
    Merge multiple filters into a single map by averaging.
    Input filters are [1,1,H,W]; output is H x W.
    """
    stacked = torch.stack(filters, dim=0)  # [N,1,H,W] # type: ignore
    merged = stacked.mean(dim=0)
    return tensor_to_np(merged)


def plot_filters(
    ax: Axes,
    filt_panel: np.ndarray,
    sums: list[float],
    pixels: Iterable[tuple[int, int]],
    cmap: str = "coolwarm",
) -> None:
    vmax = np.max(np.abs(filt_panel))
    if vmax == 0:
        vmax = 1.0
    h, w = filt_panel.shape
    ax.imshow(
        filt_panel,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        origin="upper",
        extent=(0, w, h, 0),
        interpolation="nearest",
    )
    ax.plot([0, w], [0, h], "k--", linewidth=1.2)
    # place sigma annotations near their corresponding pixel locations
    px_list = list(pixels)
    label_pos = [(0.2, 0.88), (0.4, 0.18)]  # top-left, bottom-right-ish (axes coords)
    for (x, y), s_val in zip(label_pos, sums):
        ax.text(
            x,
            y,
            rf"$\Sigma = {s_val:.2f}$",
            ha="left",
            va="center",
            fontsize=10,
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2),
        )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.margins(0)
    ax.axis("off")
    ax.set_aspect("equal")


def make_figure() -> Path:
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = build_model_from_key(BASE_KEY, device)
    wne_model = build_model_from_key(WNE_KEY, device)
    clean = load_clean_tensor(IMAGE_PATH, device)

    fig, axes = plt.subplots(len(NOISE_LEVELS_8BIT), 5, figsize=(12.0, 6.0))
    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        top=0.92,
        bottom=0.06,
        wspace=WSPACE,
        hspace=HSPACE,
    )
    for ax in axes.flat:
        ax.set_box_aspect(1.0)
    column_titles = [
        "",
        r"\textbf{\textit{Denoised}}",
        r"\textbf{\textit{Adaptive filters}}",
        r"\textbf{\textit{Denoised}}",
        r"\textbf{\textit{Adaptive filters}}",
    ]
    for ax, title in zip(axes[0], column_titles):
        ax.set_title(title, fontsize=12, pad=8)

    # Increase spacing between noisy → model 1 and model 1 → model 2 blocks
    base_positions = [axes[0, c].get_position() for c in range(5)]
    base_width = base_positions[0].width
    base_gap = base_positions[1].x0 - base_positions[0].x1
    gap_ratio = base_gap / base_width
    left_margin = fig.subplotpars.left
    right_margin = fig.subplotpars.right
    avail_width = right_margin - left_margin
    # Two default gaps and two widened gaps per row
    gap_weight = 2 + 2 * MODEL_PAD_MULTIPLIER
    new_width = avail_width / (5 + gap_ratio * gap_weight)
    new_gap = gap_ratio * new_width
    gap_factors = [0.0, MODEL_PAD_MULTIPLIER, 1.0, MODEL_PAD_MULTIPLIER, 1.0]
    for row_axes in axes:
        x_cursor = left_margin
        for col_idx, ax in enumerate(row_axes):
            if col_idx > 0:
                x_cursor += new_gap * gap_factors[col_idx]
            bbox = ax.get_position()
            ax.set_position([x_cursor, bbox.y0, new_width, bbox.height])
            x_cursor += new_width

    def add_group_label(left_ax: Axes, right_ax: Axes, text: str) -> None:
        left_pos = left_ax.get_position()
        right_pos = right_ax.get_position()
        x_center = (left_pos.x0 + right_pos.x1) / 2
        y = min(0.995, left_pos.y1 + GROUP_LABEL_Y_OFFSET)
        fig.text(
            x_center,
            y,
            text,
            ha="center",
            va="bottom",
            fontsize=12,
            transform=fig.transFigure,
        )

    def add_single_label(ax: Axes, text: str) -> None:
        pos = ax.get_position()
        x_center = (pos.x0 + pos.x1) / 2
        y = min(0.995, pos.y1 + GROUP_LABEL_Y_OFFSET)
        fig.text(
            x_center,
            y,
            text,
            ha="center",
            va="bottom",
            fontsize=12,
            transform=fig.transFigure,
        )

    add_group_label(axes[0, 1], axes[0, 2], BASE_MODEL_LABEL)
    add_group_label(axes[0, 3], axes[0, 4], WNE_MODEL_LABEL)
    add_single_label(axes[0, 0], NOISY_LABEL)

    def add_border(ax: Axes, color: str = "black", width: float = BORDER_W) -> None:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(width)

        ax.add_patch(
            Rectangle(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                fill=False,
                lw=width,
                edgecolor=color,
                zorder=10,
                clip_on=False,
            )
        )
        ax.set_facecolor("white")

    for row_idx, sigma_8 in enumerate(NOISE_LEVELS_8BIT):
        sigma = sigma_8 / 255.0
        noisy = add_noise(clean, sigma)

        # Baseline
        base_rho, base_denoised = filter_literalness(base_model, noisy)
        base_psnr = psnr_db(base_denoised, clean)
        base_filters = [
            compute_adaptive_filter(base_model, noisy, i, j) for i, j in PIXELS
        ]
        base_sums = [float(f.sum().item()) for f in base_filters]

        # WNE
        wne_rho, wne_denoised = filter_literalness(wne_model, noisy)
        wne_psnr = psnr_db(wne_denoised, clean)
        wne_filters = [
            compute_adaptive_filter(wne_model, noisy, i, j) for i, j in PIXELS
        ]
        wne_sums = [float(f.sum().item()) for f in wne_filters]

        row_axes = axes[row_idx]
        noisy_ax, base_img_ax, base_f_ax, wne_img_ax, wne_f_ax = row_axes
        noisy_ax.text(
            -0.10,
            0.5,
            rf"$\sigma = {sigma_8}$",
            rotation=90,
            transform=noisy_ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
        )

        # Noisy panel
        noisy_ax.imshow(tensor_to_np(noisy), cmap="gray", vmin=0.0, vmax=1.0)
        noisy_ax.axis("off")
        add_border(noisy_ax)
        # noisy_ax.text(
        #     0.02,
        #     0.98,
        #     f"σ = {sigma_8}",
        #     transform=noisy_ax.transAxes,
        #     ha="left",
        #     va="top",
        #     fontsize=10,
        #     bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
        # )

        def annotate_image(ax: Axes, img: torch.Tensor, psnr: float, rho: float):
            ax.imshow(tensor_to_np(img), cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            add_border(ax)
            ax.text(
                0.02,
                0.98,
                (rf"PSNR: {psnr:.2f}\,dB" "\n" rf"$\rho$: {rho:.3f}"),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
            )
            # mark the two pixels
            xs = [p[1] for p in PIXELS]
            ys = [p[0] for p in PIXELS]
            ax.scatter(xs, ys, s=30, c="#0096FF", marker="s")

        annotate_image(base_img_ax, base_denoised, base_psnr, base_rho)
        annotate_image(wne_img_ax, wne_denoised, wne_psnr, wne_rho)

        # Filter panels
        base_panel = -combine_filters(base_filters)
        wne_panel = -combine_filters(wne_filters)
        plot_filters(base_f_ax, base_panel, base_sums, PIXELS)
        plot_filters(wne_f_ax, wne_panel, wne_sums, PIXELS)
        for f_ax in (base_f_ax, wne_f_ax):
            add_border(f_ax)
            f_ax.set_aspect("equal")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return OUTPUT_PATH


if __name__ == "__main__":
    out_path = make_figure()
    print(f"Saved figure to: {out_path}")
