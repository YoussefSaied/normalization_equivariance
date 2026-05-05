import os
from os.path import isfile
from time import perf_counter

from PIL import Image, ImageOps
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from se.configs import DatasetConfig, TrainConfig, resolve_image_mode


def load_data_m(cfg: TrainConfig | DatasetConfig):
    """Load Mohan et al. (2020) bias-free denoising splits (train/valid)."""
    train_dataset = DatasetM(filename=os.path.join(cfg.train_path, "train.h5"))
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, num_workers=1, shuffle=True
    )
    valid_loader = _build_valid_loader(cfg)
    return train_loader, valid_loader


class DatasetM(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.h5f = h5py.File(filename, "r")
        self.keys = list(self.h5f.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        data = np.array(self.h5f[key])
        return torch.Tensor(data)


def list_image_files(
    in_folders: list[str], max_images: int | None = None
) -> list[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    files = [
        f"{in_folder}/{name}"
        for in_folder in in_folders
        for name in sorted(os.listdir(in_folder))
        if isfile(f"{in_folder}/{name}")
        and not name.startswith(".")
        and name.lower().endswith(exts)
    ]
    if max_images is not None:
        files = files[:max_images]
    return files


def load_image_array(path: str | os.PathLike[str], image_mode: str) -> np.ndarray:
    with Image.open(path) as image:
        if image_mode == "rgb":
            return np.array(image.convert("RGB")).astype(np.uint8)
        if image_mode == "m":
            rgb = np.array(image.convert("RGB")).astype(np.uint8)
            return rgb[..., 2]
        if image_mode in {"gray", "h"}:
            return np.array(ImageOps.grayscale(image)).astype(np.uint8)
    raise ValueError(f"Unsupported image mode '{image_mode}'.")


def load_images(
    in_folders: list[str], image_mode: str = "h", max_images: int | None = None
) -> list[np.ndarray]:
    """Load images as grayscale or RGB arrays."""
    start = perf_counter()
    files = list_image_files(in_folders, max_images=max_images)
    images = [load_image_array(f, image_mode=image_mode) for f in files]
    duration = perf_counter() - start
    print(f"load_images: loaded {len(images)} images in {duration:.3f}s", flush=True)
    return images


def image_to_tensor(img_np: np.ndarray) -> torch.Tensor:
    img_array = np.asarray(img_np)
    if img_array.dtype == np.uint8:
        img_float = img_array.astype(np.float32) / 255.0
    else:
        img_float = img_array.astype(np.float32)
    if img_float.ndim == 2:
        return torch.from_numpy(img_float).unsqueeze(0).float()
    img_chw = np.ascontiguousarray(np.transpose(img_float, (2, 0, 1)))
    return torch.from_numpy(img_chw).float()


def augmentation(x, k=0, inverse=False):
    k = k % 8
    if inverse:
        k = [0, 1, 6, 3, 4, 5, 2, 7][k]
    if k % 2 == 1:
        x = torch.flip(x, dims=[2])
    return torch.rot90(x, k=(k // 2) % 4, dims=[1, 2])


class DatasetH(Dataset):
    def __init__(
        self,
        in_folders: list[str],
        patch_size=70,
        samples_per_epoch=1000,
        image_mode: str = "h",
    ):
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch

        self.images_train = load_images(in_folders, image_mode=image_mode)
        self.number_of_images = len(self.images_train)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        img_np = self.images_train[np.random.choice(self.number_of_images)]
        h, w = img_np.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            raise ValueError(
                f"Patch size {self.patch_size} exceeds image shape {img_np.shape}."
            )
        i = np.random.randint(0, h - self.patch_size + 1)
        j = np.random.randint(0, w - self.patch_size + 1)
        patch = img_np[i : i + self.patch_size, j : j + self.patch_size]

        img_torch = image_to_tensor(patch)

        k = np.random.randint(8)
        img_torch = augmentation(img_torch, k)

        return img_torch


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        in_folders: list[str],
        image_mode: str,
        max_images: int | None = None,
    ):
        super().__init__()
        self.images = load_images(
            in_folders, image_mode=image_mode, max_images=max_images
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return image_to_tensor(self.images[index])


def _resolve_train_dirs(base_path: str, folders: list[str]) -> list[str]:
    resolved: list[str] = []
    for folder in folders:
        folder_path = (
            folder if os.path.isabs(folder) else os.path.join(base_path, folder)
        )
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")
        resolved.append(folder_path)
    return resolved


def _build_valid_loader(cfg: TrainConfig | DatasetConfig):
    if cfg.valid_path:
        valid_dirs = _resolve_train_dirs(cfg.train_path, cfg.valid_path)
        valid_dataset: Dataset = ImageFolderDataset(
            in_folders=valid_dirs,
            image_mode=resolve_image_mode(cfg),
            max_images=cfg.valid_max_images,
        )
    else:
        valid_dataset = DatasetM(filename=os.path.join(cfg.train_path, "valid.h5"))
    return DataLoader(valid_dataset, batch_size=1, num_workers=1, shuffle=False)


def load_data_h(cfg: TrainConfig | DatasetConfig):
    train_dirs = _resolve_train_dirs(cfg.train_path, cfg.train_image_dirs)
    assert (
        cfg.s_samples_per_epoch is not None
    ), "s_samples_per_epoch must be set for 'h' dataset"
    train_dataset = DatasetH(
        in_folders=train_dirs,
        patch_size=cfg.s_patch_size,
        samples_per_epoch=cfg.s_samples_per_epoch,
        image_mode=resolve_image_mode(cfg),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, num_workers=1, shuffle=True
    )
    valid_loader = _build_valid_loader(cfg)
    return train_loader, valid_loader


def build_sidd_validation_loader(
    noisy_mat_path: str,
    gt_mat_path: str,
    *,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    from data.sidd_dataset import SIDDValDataset

    dataset = SIDDValDataset(noisy_mat_path, gt_mat_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_sidd_loaders(
    train_dir: str,
    val_noisy_mat: str,
    val_gt_mat: str,
    *,
    crop_size: int = 256,
    batch_size: int = 32,
    val_batch_size: int = 16,
    num_workers: int = 0,
    val_num_workers: int = 0,
    pin_memory: bool = False,
    worker_init_fn=None,
    generator: torch.Generator | None = None,
) -> tuple[DataLoader, DataLoader]:
    from data.sidd_dataset import SIDDTrainDataset

    train_dataset = SIDDTrainDataset(train_dir, crop_size=crop_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    val_loader = build_sidd_validation_loader(
        val_noisy_mat,
        val_gt_mat,
        batch_size=val_batch_size,
        num_workers=val_num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


# %% data loaders
def build_loaders(cfg: TrainConfig | DatasetConfig):
    dataset_type = cfg.train_dataset_type.lower()
    if dataset_type == "h":
        return load_data_h(cfg)
    if dataset_type == "m":
        return load_data_m(cfg)
    raise ValueError(
        f"Unknown dataset type '{cfg.train_dataset_type}'. Use 'm' or 'h'."
    )
