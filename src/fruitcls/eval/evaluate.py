from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    confusion_matrix: np.ndarray
    class_names: List[str]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    macro_precision: float
    macro_recall: float


def _confusion_matrix_and_preds(
    model: tf.keras.Model,
    ds: tf.data.Dataset,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_all = []
    y_pred_all = []

    for x, y in ds:
        probs = model.predict(x, verbose=0)
        y_pred = np.argmax(probs, axis=1)

        y_true_all.append(y.numpy())
        y_pred_all.append(y_pred)

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)

    cm = tf.math.confusion_matrix(y_true_all, y_pred_all, num_classes=num_classes).numpy()
    return cm, y_true_all, y_pred_all


def _precision_recall_from_cm(cm: np.ndarray, class_names: List[str]) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
    eps = 1e-12
    tp = np.diag(cm).astype(np.float64)

    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    per_class_precision = {name: float(p) for name, p in zip(class_names, precision)}
    per_class_recall = {name: float(r) for name, r in zip(class_names, recall)}

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))

    return per_class_precision, per_class_recall, macro_precision, macro_recall


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: List[str],
) -> EvalResult:
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in test_ds:
        probs = model(x, training=False)
        batch_loss = loss_fn(y, probs).numpy()
        total_loss += batch_loss * x.shape[0]

        y_pred = tf.argmax(probs, axis=1)
        total_correct += int(tf.reduce_sum(tf.cast(y_pred == tf.cast(y, tf.int64), tf.int32)).numpy())
        total += x.shape[0]

    loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)

    cm, _, _ = _confusion_matrix_and_preds(model, test_ds, num_classes=len(class_names))
    per_prec, per_rec, macro_prec, macro_rec = _precision_recall_from_cm(cm, class_names)

    return EvalResult(
        loss=float(loss),
        accuracy=float(acc),
        confusion_matrix=cm,
        class_names=class_names,
        per_class_precision=per_prec,
        per_class_recall=per_rec,
        macro_precision=macro_prec,
        macro_recall=macro_rec,
    )