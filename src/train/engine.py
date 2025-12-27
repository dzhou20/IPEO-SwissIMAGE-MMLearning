from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from src.utils.metrics import classification_metrics, per_class_f1


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mode: str,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        if batch is None:
            continue
        optimizer.zero_grad()

        if mode == "image":
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
        else:
            images, tabular, labels = batch
            images = images.to(device)
            tabular = tabular.to(device)
            labels = labels.to(device)
            logits = model(images, tabular)

        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    num_classes: int,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        if batch is None:
            continue
        if mode == "image":
            images, labels = batch
            images = images.to(device)
            logits = model(images)
        else:
            images, tabular, labels = batch
            images = images.to(device)
            tabular = tabular.to(device)
            logits = model(images, tabular)

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    metrics = classification_metrics(y_true, y_pred)
    per_class = per_class_f1(y_true, y_pred, num_classes=num_classes)
    return metrics, torch.from_numpy(y_true), torch.from_numpy(y_pred), torch.from_numpy(per_class)
