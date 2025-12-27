from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import rasterio
from rasterio.errors import RasterioIOError

from config import IMAGE_MEAN, IMAGE_STD


@dataclass
class DatasetConfig:
    image_dir: str
    image_size: tuple[int, int]
    mode: str
    group_cols: Optional[list[str]]
    augment: bool


def _normalize_image(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGE_MEAN, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    std = torch.tensor(IMAGE_STD, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    return (img - mean) / std


def _resize_image(img: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    img = img.unsqueeze(0)
    img = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
    return img.squeeze(0)


def _maybe_augment(img: torch.Tensor) -> torch.Tensor:
    if torch.rand(1).item() < 0.5:
        img = torch.flip(img, dims=[2])
    if torch.rand(1).item() < 0.5:
        img = torch.flip(img, dims=[1])
    return img


class SwissImageDataset(Dataset):
    def __init__(
        self,
        df,
        config: DatasetConfig,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.config = config

        if self.config.mode not in {"image", "fusion"}:
            raise ValueError(f"Unsupported mode: {self.config.mode}")

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> torch.Tensor:
        with rasterio.open(path) as src:
            img = src.read(out_dtype="float32")

        # Expect shape (bands, H, W); keep first 3 channels.
        if img.shape[0] < 3:
            raise ValueError(f"Expected at least 3 bands, got {img.shape[0]} for {path}")
        img = img[:3, :, :]
        img = torch.from_numpy(img)

        # Scale to 0-1 if likely uint8 range.
        if img.max() > 1.5:
            img = img / 255.0

        img = _resize_image(img, self.config.image_size)
        if self.config.augment:
            img = _maybe_augment(img)
        img = _normalize_image(img)
        return img

    def _load_tabular(self, index: int) -> torch.Tensor:
        if not self.config.group_cols:
            raise ValueError("group_cols must be provided for fusion mode.")
        values = self.df.loc[index, self.config.group_cols].to_numpy(dtype=np.float32)
        return torch.from_numpy(values)

    def __getitem__(self, index: int):
        row = self.df.loc[index]
        if "img_path" in row:
            image_path = Path(row["img_path"])
        else:
            image_path = Path(self.config.image_dir) / f"{row['id']}.tif"
        try:
            image = self._load_image(str(image_path))
        except RasterioIOError:
            return None
        label = int(row["EUNIS_cls"])

        if self.config.mode == "image":
            return image, label

        tabular = self._load_tabular(index)
        return image, tabular, label
