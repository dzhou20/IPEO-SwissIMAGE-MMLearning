from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from config import CSV_PATH, IMAGE_DIR, IMAGE_SIZE
from sweco_group_of_variables import sweco_variables_dict
from src.data.dataset import DatasetConfig, SwissImageDataset


def _resolve_group_columns(df: pd.DataFrame, group: Optional[str]) -> Optional[list[str]]:
    if group is None:
        return None

    if group == "all":
        desired = []
        seen = set()
        for cols in sweco_variables_dict.values():
            for col in cols:
                if col not in seen:
                    desired.append(col)
                    seen.add(col)
        group = "__all_sweco__"

    if group not in sweco_variables_dict and group != "__all_sweco__":
        raise ValueError(f"Unknown group: {group}. Options: {list(sweco_variables_dict)} + ['all']")

    if group == "__all_sweco__":
        desired = desired
    else:
        desired = sweco_variables_dict[group]
    resolved = []
    for col in desired:
        if col in df.columns:
            resolved.append(col)
            continue

        # Handle duplicated column names (pandas appends .1, .2).
        matches = [c for c in df.columns if c.startswith(f"{col}.")]
        if matches:
            resolved.append(matches[0])
            continue

        raise KeyError(f"Column '{col}' not found in dataset_split.csv.")

    return resolved


def build_dataloaders(
    mode: str,
    group: Optional[str],
    batch_size: int,
    num_workers: int,
    image_dir: str = IMAGE_DIR,
    csv_path: str = CSV_PATH,
    image_size: tuple[int, int] = IMAGE_SIZE,
):
    df = pd.read_csv(csv_path)
    total_count = len(df)
    df["img_path"] = df["id"].apply(lambda x: str(Path(image_dir) / f"{x}.tif"))
    df["img_exists"] = df["img_path"].apply(lambda p: Path(p).exists())
    missing_count = int((~df["img_exists"]).sum())
    if missing_count > 0:
        print(f"[warn] Missing images: {missing_count} rows will be skipped.")
    df = df[df["img_exists"]].drop(columns=["img_exists"]).reset_index(drop=True)
    used_count = len(df)
    print(f"[info] Dataset rows: total={total_count}, used={used_count}, missing={missing_count}")

    group_cols = _resolve_group_columns(df, group)
    if group_cols is not None:
        print(f"[info] SWECO variables used: {len(group_cols)}")

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_cfg = DatasetConfig(
        image_dir=image_dir,
        image_size=image_size,
        mode=mode,
        group_cols=group_cols,
        augment=True,
    )
    eval_cfg = DatasetConfig(
        image_dir=image_dir,
        image_size=image_size,
        mode=mode,
        group_cols=group_cols,
        augment=False,
    )

    train_ds = SwissImageDataset(train_df, train_cfg)
    val_ds = SwissImageDataset(val_df, eval_cfg)
    test_ds = SwissImageDataset(test_df, eval_cfg)

    def collate_fn(batch):
        filtered = [item for item in batch if item is not None]
        if len(filtered) != len(batch):
            dropped = len(batch) - len(filtered)
            print(f"[warn] Dropped {dropped} corrupt samples in a batch.")
        if not filtered:
            return None
        return default_collate(filtered)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, group_cols
