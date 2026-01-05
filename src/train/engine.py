from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from src.utils.metrics import classification_metrics, per_class_f1


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    criterion_l1: torch.nn.Module | None,
    device: torch.device,
    mode: str,
    two_stage: bool = False,
    level1_weight: float = 0.3,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    num_samples = 0
    correct = 0

    for batch in loader:
        if batch is None:
            continue
            
        optimizer.zero_grad()

        if mode == "image":
            if two_stage:
                images, labels_l2, labels_l1 = batch
                images = images.to(device)
                labels_l2 = labels_l2.to(device)
                labels_l1 = labels_l1.to(device)
                logits_l1, logits_l2 = model(images)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
        elif mode == "tabular":
            if two_stage:
                tabular, labels_l2, labels_l1 = batch
                tabular = tabular.to(device)
                labels_l2 = labels_l2.to(device)
                labels_l1 = labels_l1.to(device)
                logits_l1, logits_l2 = model(tabular)
            else:
                tabular, labels = batch
                tabular = tabular.to(device)
                labels = labels.to(device)
                logits = model(tabular)
        else:
            if two_stage:
                images, tabular, labels_l2, labels_l1 = batch
                images = images.to(device)
                tabular = tabular.to(device)
                labels_l2 = labels_l2.to(device)
                labels_l1 = labels_l1.to(device)
                logits_l1, logits_l2 = model(images, tabular)
            else:
                images, tabular, labels = batch
                images = images.to(device)
                tabular = tabular.to(device)
                labels = labels.to(device)
                logits = model(images, tabular)

        if two_stage:
            loss_l2 = criterion(logits_l2, labels_l2)
            loss_l1 = (criterion_l1 or criterion)(logits_l1, labels_l1)
            loss = loss_l2 * (1.0 - level1_weight) + loss_l1 * level1_weight
            preds = torch.argmax(logits_l2, dim=1)
            labels_for_acc = labels_l2
        else:
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            labels_for_acc = labels
        loss.backward()
        optimizer.step()
        
        # loss      
        batch_size = labels_for_acc.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

        # accuracy 
        correct += (preds == labels_for_acc).sum().item()

    train_loss = running_loss / max(num_samples, 1)
    train_acc = correct / max(num_samples, 1)

    return train_loss, train_acc


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    criterion_l1: torch.nn.Module | None,
    device: torch.device,
    mode: str,
    num_classes: int,
    two_stage: bool = False,
    num_level1_classes: int | None = None,
    level1_id_to_level2_ids: dict[int, list[int]] | None = None,
    level1_weight: float = 0.3,
) -> Tuple[Dict[str, float], float, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()

    all_preds = []
    all_labels = []
    all_preds_l1 = []
    all_labels_l1 = []

    running_loss = 0.0
    num_samples = 0
    level1_mask = None

    if two_stage:
        if num_level1_classes is None or level1_id_to_level2_ids is None:
            raise ValueError("num_level1_classes and level1_id_to_level2_ids are required for two_stage.")
        level1_mask = torch.full(
            (num_level1_classes, num_classes),
            float("-inf"),
            device=device,
        )
        for level1_id, level2_ids in level1_id_to_level2_ids.items():
            level1_mask[level1_id, level2_ids] = 0.0

    for batch in loader:
        if batch is None:
            continue

        if mode == "image":
            if two_stage:
                images, labels_l2, labels_l1 = batch
                images = images.to(device)
                labels_l2 = labels_l2.to(device)
                labels_l1 = labels_l1.to(device)
                logits_l1, logits_l2 = model(images)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
        elif mode == "tabular":
            if two_stage:
                tabular, labels_l2, labels_l1 = batch
                tabular = tabular.to(device)
                labels_l2 = labels_l2.to(device)
                labels_l1 = labels_l1.to(device)
                logits_l1, logits_l2 = model(tabular)
            else:
                tabular, labels = batch
                tabular = tabular.to(device)
                labels = labels.to(device)
                logits = model(tabular)
        else:
            if two_stage:
                images, tabular, labels_l2, labels_l1 = batch
                images = images.to(device)
                tabular = tabular.to(device)
                labels_l2 = labels_l2.to(device)
                labels_l1 = labels_l1.to(device)
                logits_l1, logits_l2 = model(images, tabular)
            else:
                images, tabular, labels = batch
                images = images.to(device)
                tabular = tabular.to(device)
                labels = labels.to(device)
                logits = model(images, tabular)

        if two_stage:
            loss_l2 = criterion(logits_l2, labels_l2)
            loss_l1 = (criterion_l1 or criterion)(logits_l1, labels_l1)
            loss = loss_l2 * (1.0 - level1_weight) + loss_l1 * level1_weight
            preds_l1 = torch.argmax(logits_l1, dim=1)
            masked_logits_l2 = logits_l2 + level1_mask[preds_l1]
            preds = torch.argmax(masked_logits_l2, dim=1)
            labels_for_metrics = labels_l2
        else:
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            labels_for_metrics = labels

        batch_size = labels_for_metrics.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

        all_preds.append(preds.cpu())
        all_labels.append(labels_for_metrics.cpu())
        if two_stage:
            all_preds_l1.append(preds_l1.cpu())
            all_labels_l1.append(labels_l1.cpu())

    val_loss = running_loss / max(num_samples, 1)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    metrics = classification_metrics(y_true, y_pred)
    per_class = per_class_f1(y_true, y_pred, num_classes=num_classes)

    if not two_stage:
        return (
            metrics,
            val_loss,
            torch.from_numpy(y_true),
            torch.from_numpy(y_pred),
            torch.from_numpy(per_class),
        )

    y_pred_l1 = torch.cat(all_preds_l1).numpy()
    y_true_l1 = torch.cat(all_labels_l1).numpy()
    metrics_l1 = classification_metrics(y_true_l1, y_pred_l1)
    per_class_l1 = per_class_f1(y_true_l1, y_pred_l1, num_classes=num_level1_classes)

    return (
        metrics,
        metrics_l1,
        val_loss,
        torch.from_numpy(y_true),
        torch.from_numpy(y_pred),
        torch.from_numpy(y_true_l1),
        torch.from_numpy(y_pred_l1),
        torch.from_numpy(per_class),
        torch.from_numpy(per_class_l1),
    )
