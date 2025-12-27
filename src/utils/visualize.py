from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(
    y_true,
    y_pred,
    labels: List[str],
    output_dir: str,
    filename_prefix: str = "confusion_matrix",
) -> np.ndarray:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    csv_path = output_path / f"{filename_prefix}.csv"
    np.savetxt(csv_path, cm, delimiter=",", fmt="%d")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path / f"{filename_prefix}.png", dpi=300)
    plt.close()

    return cm


def save_normalized_confusion_matrix(
    y_true,
    y_pred,
    labels: List[str],
    output_dir: str,
    filename_prefix: str = "confusion_matrix_normalized",
) -> np.ndarray:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    csv_path = output_path / f"{filename_prefix}.csv"
    np.savetxt(csv_path, cm_norm, delimiter=",", fmt="%.6f")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=False,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0.0,
        vmax=1.0,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path / f"{filename_prefix}.png", dpi=300)
    plt.close()

    return cm_norm
