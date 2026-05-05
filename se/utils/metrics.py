import torch
import numpy as np
import numpy.typing as npt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


FloatArray = npt.NDArray[np.float32]


def _to_skimage_image(img: FloatArray) -> FloatArray:
    if img.ndim != 3:
        raise ValueError(f"Expected a CHW image, got shape {img.shape}.")
    if img.shape[0] == 1:
        return np.asarray(img[0], dtype=np.float32)
    return np.asarray(np.moveaxis(img, 0, -1), dtype=np.float32)


def ssim(clean: torch.Tensor, noisy: torch.Tensor, normalized=True):
    """Use skimage.meamsure.compare_ssim to calculate SSIM
    Args:
        clean (Tensor): (B, C, H, W)
        noisy (Tensor): (B, C, H, W)
        normalized (bool): If True, the range of tensors are [0., 1.] else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    if normalized:
        clean = clean.mul(255).clamp(0, 255)
        noisy = noisy.mul(255).clamp(0, 255)

    clean_np: FloatArray = np.asarray(clean.cpu().detach().numpy(), dtype=np.float32)
    noisy_np: FloatArray = np.asarray(noisy.cpu().detach().numpy(), dtype=np.float32)
    values = []
    for c, n in zip(clean_np, noisy_np):
        c_img = _to_skimage_image(np.asarray(c, dtype=np.float32))
        n_img = _to_skimage_image(np.asarray(n, dtype=np.float32))
        channel_axis = -1 if c_img.ndim == 3 else None
        values.append(
            structural_similarity(
                c_img,
                n_img,
                data_range=255,
                channel_axis=channel_axis,
            )
        )
    return float(np.mean(values))


def psnr(clean: torch.Tensor, noisy: torch.Tensor, normalized=True):
    """
    Args:
        clean (Tensor): (B, C, H, W)
        noisy (Tensor): (B, C, H, W)
        normalized (bool): If True, the range of tensors are [0., 1.]
            else [0, 255]
    Returns:
        PSNR per image: (B, )
    """
    if normalized:
        clean = clean.mul(255).clamp(0, 255)
        noisy = noisy.mul(255).clamp(0, 255)

    clean_np: FloatArray = np.asarray(clean.cpu().detach().numpy(), dtype=np.float32)
    noisy_np: FloatArray = np.asarray(noisy.cpu().detach().numpy(), dtype=np.float32)
    values = []
    for c, n in zip(clean_np, noisy_np):
        c_img = _to_skimage_image(np.asarray(c, dtype=np.float32))
        n_img = _to_skimage_image(np.asarray(n, dtype=np.float32))
        values.append(peak_signal_noise_ratio(c_img, n_img, data_range=255))
    return float(np.mean(values))
