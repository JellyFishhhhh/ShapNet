"""Per-WSI ranking and binary metrics with validation-only thresholds."""
import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                              precision_recall_fscore_support)

from .streaming import select_indices


def ranking_metrics(scores, targets, ratio=0.3):
    def finite_or_none(value):
        return float(value) if np.isfinite(value) else None
    if len(scores) < 2 or scores.std() == 0 or targets.std() == 0:
        spearman, kendall = None, None
    else:
        spearman = finite_or_none(spearmanr(scores.numpy(), targets.numpy()).statistic)
        kendall = finite_or_none(kendalltau(scores.numpy(), targets.numpy()).statistic)
    pred = set(select_indices(scores, ratio, min_patches=1).tolist())
    truth = set(select_indices(targets, ratio, min_patches=1).tolist())
    return {"spearman": spearman, "kendall": kendall,
            "topk_overlap": len(pred & truth) / len(truth),
            "mae": float(torch.mean(torch.abs(scores - targets)))}


def macro_ranking(rows):
    out = {}
    for key in ("spearman", "kendall", "topk_overlap", "mae"):
        valid = [r[key] for r in rows if r[key] is not None]
        out[key] = float(np.mean(valid)) if valid else None
        out[key + "_valid_bags"] = len(valid)
    return out


def choose_threshold(labels, probabilities):
    if set(labels) != {0, 1}:
        raise ValueError("Threshold calibration needs both validation classes")
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    valid = np.isfinite(thresholds)
    return float(thresholds[valid][np.argmax((tpr - fpr)[valid])])


def binary_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if not np.isfinite(probabilities).all() or not np.isfinite(threshold):
        raise ValueError("Nonfinite predictions or threshold")
    predicted = probabilities >= threshold
    p, r, f, _ = precision_recall_fscore_support(labels, predicted, average="binary", zero_division=0)
    return {"auc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None,
            "accuracy": float(accuracy_score(labels, predicted)), "precision": float(p),
            "recall": float(r), "f1": float(f), "threshold": float(threshold),
            "n_slides": len(labels)}
