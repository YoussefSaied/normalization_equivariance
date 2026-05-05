from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity
import torch
from torch.utils.data import DataLoader

from se.utils.runtime_utils import autocast_context


def tensor_to_uint8_batch(images: torch.Tensor) -> np.ndarray:
    images = images.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    return images.permute(0, 2, 3, 1).cpu().numpy()


def compute_psnr_uint8_batch(pred_uint8: np.ndarray, gt_uint8: np.ndarray) -> np.ndarray:
    pred = pred_uint8.astype(np.float32)
    gt = gt_uint8.astype(np.float32)
    mse = np.mean((pred - gt) ** 2, axis=(1, 2, 3))
    mse = np.maximum(mse, 1e-12)
    return 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)


def compute_ssim_uint8_batch(pred_uint8: np.ndarray, gt_uint8: np.ndarray) -> np.ndarray:
    values = [
        structural_similarity(
            gt_image,
            pred_image,
            data_range=255,
            channel_axis=-1,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
        for pred_image, gt_image in zip(pred_uint8, gt_uint8)
    ]
    return np.asarray(values, dtype=np.float64)


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    psnr_values: list[np.ndarray] = []
    ssim_values: list[np.ndarray] = []

    for noisy, gt in dataloader:
        noisy = noisy.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        with autocast_context(device, amp_dtype):
            pred = model(noisy)

        pred_uint8 = tensor_to_uint8_batch(pred)
        gt_uint8 = tensor_to_uint8_batch(gt)
        psnr_values.append(compute_psnr_uint8_batch(pred_uint8, gt_uint8))
        ssim_values.append(compute_ssim_uint8_batch(pred_uint8, gt_uint8))

    if was_training:
        model.train()

    psnr_all = np.concatenate(psnr_values, axis=0)
    ssim_all = np.concatenate(ssim_values, axis=0)
    return {
        "psnr": float(psnr_all.mean()),
        "ssim": float(ssim_all.mean()),
        "num_images": float(psnr_all.shape[0]),
    }
