from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
    }


def per_class_f1(y_true, y_pred, num_classes: int) -> np.ndarray:
    return f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)))
