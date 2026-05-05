from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from se.configs import PROJECT_ROOT
from se.data import image_to_tensor, load_image_array


PNG_EXTENSIONS = {".png"}
SIDD_ROOT = Path(PROJECT_ROOT) / "data" / "SIDD"
SIDD_MEDIUM_DATA_DIR = SIDD_ROOT / "Data"
SIDD_PATCHES_DIR = SIDD_ROOT / "patches_png"
SIDD_VAL_NOISY_MAT = SIDD_ROOT / "ValidationNoisyBlocksSrgb.mat"
SIDD_VAL_GT_MAT = SIDD_ROOT / "ValidationGtBlocksSrgb.mat"


def read_rgb_image(path: str | Path) -> np.ndarray:
    return load_image_array(path, image_mode="rgb")


def find_sidd_medium_srgb_pairs(data_dir: str | Path) -> list[tuple[Path, Path]]:
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"SIDD data directory not found: {root}")

    gt_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in PNG_EXTENSIONS
        and "GT_SRGB" in path.name
    )
    if not gt_files:
        raise FileNotFoundError(f"No '*GT_SRGB*.png' files found under {root}")

    pairs: list[tuple[Path, Path]] = []
    for gt_path in gt_files:
        noisy_name = gt_path.name.replace("GT_SRGB", "NOISY_SRGB")
        noisy_path = gt_path.with_name(noisy_name)
        if not noisy_path.is_file():
            raise FileNotFoundError(f"Missing noisy pair for {gt_path}: {noisy_path}")
        pairs.append((noisy_path, gt_path))
    return pairs


def resolve_patch_dirs(
    patches_dir: str | Path | None = None,
    input_dir: str | Path | None = None,
    gt_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    if patches_dir is not None:
        root = Path(patches_dir)
        input_path = root / "input"
        gt_path = root / "gt"
    else:
        if input_dir is None or gt_dir is None:
            raise ValueError("Provide either patches_dir or both input_dir and gt_dir.")
        input_path = Path(input_dir)
        gt_path = Path(gt_dir)

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input patch directory not found: {input_path}")
    if not gt_path.is_dir():
        raise FileNotFoundError(f"Ground-truth patch directory not found: {gt_path}")
    return input_path, gt_path


def _list_png_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in PNG_EXTENSIONS
    )


def _extract_mat_blocks(mat_path: str | Path, preferred_keys: tuple[str, ...]) -> np.ndarray:
    from einops import rearrange
    from scipy.io import loadmat

    contents = loadmat(mat_path)

    for key in preferred_keys:
        if key in contents:
            blocks = np.asarray(contents[key])
            break
    else:
        blocks = None
        for key, value in contents.items():
            if key.startswith("__"):
                continue
            value = np.asarray(value)
            if value.ndim in {4, 5} and value.shape[-1] == 3:
                blocks = value
                break
        if blocks is None:
            keys = sorted(k for k in contents if not k.startswith("__"))
            raise KeyError(f"Could not find SIDD block array in {mat_path}. Keys: {keys}")

    if blocks.ndim == 5:
        blocks = rearrange(blocks, "a b h w c -> (a b) h w c")
    elif blocks.ndim != 4:
        raise ValueError(f"Unexpected block shape {blocks.shape} in {mat_path}")

    if blocks.shape[1:] != (256, 256, 3):
        raise ValueError(f"Unexpected block shape {blocks.shape} in {mat_path}")

    return np.asarray(blocks)


class SIDDTrainDataset(Dataset):
    def __init__(
        self,
        patches_dir: str | Path | None = None,
        *,
        input_dir: str | Path | None = None,
        gt_dir: str | Path | None = None,
        crop_size: int = 256,
    ) -> None:
        super().__init__()
        self.input_dir, self.gt_dir = resolve_patch_dirs(
            patches_dir=patches_dir,
            input_dir=input_dir,
            gt_dir=gt_dir,
        )
        self.crop_size = crop_size

        input_files = _list_png_files(self.input_dir)
        if not input_files:
            raise FileNotFoundError(f"No PNG patches found in {self.input_dir}")

        self.pairs: list[tuple[Path, Path]] = []
        for noisy_path in input_files:
            gt_path = self.gt_dir / noisy_path.name
            if not gt_path.is_file():
                raise FileNotFoundError(
                    f"Missing ground-truth patch for {noisy_path.name}: {gt_path}"
                )
            self.pairs.append((noisy_path, gt_path))

    def __len__(self) -> int:
        return len(self.pairs)

    def _random_crop(self, noisy: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        crop = self.crop_size
        height, width = noisy.shape[:2]
        if height < crop or width < crop:
            raise ValueError(
                f"Crop size {crop} exceeds patch shape {noisy.shape} for SIDD training patch."
            )
        top = random.randint(0, height - crop)
        left = random.randint(0, width - crop)
        noisy = noisy[top : top + crop, left : left + crop]
        gt = gt[top : top + crop, left : left + crop]
        return noisy, gt

    def _augment(self, noisy: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            noisy = np.flip(noisy, axis=1)
            gt = np.flip(gt, axis=1)
        if random.random() < 0.5:
            noisy = np.flip(noisy, axis=0)
            gt = np.flip(gt, axis=0)
        rotation_k = random.randint(0, 3)
        if rotation_k:
            noisy = np.rot90(noisy, k=rotation_k)
            gt = np.rot90(gt, k=rotation_k)
        return np.ascontiguousarray(noisy), np.ascontiguousarray(gt)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        noisy_path, gt_path = self.pairs[index]
        noisy = read_rgb_image(noisy_path)
        gt = read_rgb_image(gt_path)
        if noisy.shape != gt.shape:
            raise ValueError(
                f"Mismatched pair shapes for {noisy_path.name}: {noisy.shape} vs {gt.shape}"
            )

        noisy, gt = self._random_crop(noisy, gt)
        noisy, gt = self._augment(noisy, gt)
        return image_to_tensor(noisy), image_to_tensor(gt)


class SIDDValDataset(Dataset):
    def __init__(self, noisy_mat_path: str | Path, gt_mat_path: str | Path) -> None:
        super().__init__()
        noisy_mat_path = Path(noisy_mat_path)
        gt_mat_path = Path(gt_mat_path)
        if not noisy_mat_path.is_file():
            raise FileNotFoundError(f"SIDD validation MAT file not found: {noisy_mat_path}")
        if not gt_mat_path.is_file():
            raise FileNotFoundError(f"SIDD validation MAT file not found: {gt_mat_path}")

        self.noisy_blocks = _extract_mat_blocks(
            noisy_mat_path,
            preferred_keys=("ValidationNoisyBlocksSrgb", "ValidationNoisyBlocksRaw"),
        )
        self.gt_blocks = _extract_mat_blocks(
            gt_mat_path,
            preferred_keys=("ValidationGtBlocksSrgb", "ValidationGtBlocksRaw"),
        )
        if self.noisy_blocks.shape != self.gt_blocks.shape:
            raise ValueError(
                "Validation noisy/gt block arrays differ: "
                f"{self.noisy_blocks.shape} vs {self.gt_blocks.shape}"
            )

    def __len__(self) -> int:
        assert self.noisy_blocks is not None
        return int(self.noisy_blocks.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.noisy_blocks is not None
        assert self.gt_blocks is not None
        noisy = self.noisy_blocks[index]
        gt = self.gt_blocks[index]
        return image_to_tensor(noisy), image_to_tensor(gt)


__all__ = [
    "SIDD_MEDIUM_DATA_DIR",
    "SIDD_PATCHES_DIR",
    "SIDD_ROOT",
    "SIDD_VAL_GT_MAT",
    "SIDD_VAL_NOISY_MAT",
    "SIDDTrainDataset",
    "SIDDValDataset",
    "find_sidd_medium_srgb_pairs",
    "image_to_tensor",
    "read_rgb_image",
    "resolve_patch_dirs",
]
