"""?????????????????"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from horizontal_xgb.tree import HorizontalXGBModel


def predict_proba(
    model: HorizontalXGBModel,
    features: ArrayLike,
    feature_names: Sequence[str] | None = None,
) -> NDArray[np.float64]:
    """?????????????"""
    return model.predict_proba(features, feature_names)


def evaluate_binary_classification(
    labels: ArrayLike,
    probabilities: ArrayLike,
    threshold: float = 0.5,
) -> dict[str, float]:
    """?? AUC?Accuracy ? LogLoss?????????"""
    y_true = np.asarray(labels, dtype=np.float64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or y_prob.shape != y_true.shape or y_true.size == 0:
        raise ValueError("labels ? probabilities ??????????????")
    if not np.all(np.isin(y_true, (0.0, 1.0))) or np.unique(y_true).size != 2:
        raise ValueError("???????? 0 ? 1 ?????")
    if not np.all(np.isfinite(y_prob)) or np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("??????? [0, 1] ???????")
    if not 0.0 < threshold < 1.0:
        raise ValueError("???????? (0, 1)?")
    predictions = (y_prob >= threshold).astype(np.int64)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "logloss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }
