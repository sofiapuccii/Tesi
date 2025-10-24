"""
Evaluation utilities: compute metrics, confusion matrix, and save to JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None, requested: List[str]) -> Dict[str, float]:
    met: Dict[str, float] = {}
    y_true_int = y_true.astype(int)
    y_pred_int = y_pred.astype(int)
    if "accuracy" in requested:
        met["accuracy"] = float(accuracy_score(y_true_int, y_pred_int))
    if "precision" in requested:
        met["precision"] = float(precision_score(y_true_int, y_pred_int, average="weighted", zero_division=0))
    if "recall" in requested:
        met["recall"] = float(recall_score(y_true_int, y_pred_int, average="weighted", zero_division=0))
    if "f1" in requested:
        met["f1"] = float(f1_score(y_true_int, y_pred_int, average="weighted", zero_division=0))
    if "roc_auc" in requested and y_prob is not None:
        # One-vs-rest macro AUC
        try:
            met["roc_auc"] = float(roc_auc_score(y_true_int, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            met["roc_auc"] = float("nan")
    return met


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, metrics: List[str]) -> Dict[str, Any]:
    preds = model.predict(X_test)
    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba(X_test)
        except Exception:
            prob = None
    met = compute_metrics(y_test.to_numpy(), preds, prob, metrics)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    return {"metrics": met, "confusion_matrix": cm.tolist(), "report": report, "y_true": y_test.tolist(), "y_pred": preds.tolist()}


def save_metrics(results: Dict[str, Any], metrics_dir: Path) -> Path:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return path


