from typing import Dict, Iterable, List, Sequence

import numpy as np


def _safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def confusion_matrix(labels: Sequence[int], predictions: Sequence[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, prediction in zip(labels, predictions):
        matrix[int(label), int(prediction)] += 1
    return matrix


def per_class_f1(matrix: np.ndarray) -> List[float]:
    scores = []
    for idx in range(matrix.shape[0]):
        tp = matrix[idx, idx]
        fp = matrix[:, idx].sum() - tp
        fn = matrix[idx, :].sum() - tp
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        scores.append(_safe_div(2.0 * precision * recall, precision + recall))
    return scores


def balanced_accuracy(matrix: np.ndarray) -> float:
    recalls = []
    for idx in range(matrix.shape[0]):
        support = matrix[idx, :].sum()
        recalls.append(_safe_div(matrix[idx, idx], support))
    return float(np.mean(recalls)) if recalls else 0.0


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 15) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == labels).astype(np.float32)
    ece = 0.0
    for bin_idx in range(n_bins):
        lower = bin_idx / n_bins
        upper = (bin_idx + 1) / n_bins
        if bin_idx == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not mask.any():
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.mean()) * abs(float(bin_acc - bin_conf))
    return float(ece)


def uncertainty_risk_auc(labels: np.ndarray, probabilities: np.ndarray, uncertainty: np.ndarray) -> float:
    predictions = probabilities.argmax(axis=1)
    errors = (predictions != labels).astype(np.float32)
    order = np.argsort(uncertainty)
    sorted_errors = errors[order]
    cumulative_risk = np.cumsum(sorted_errors) / np.arange(1, len(sorted_errors) + 1)
    coverage = np.arange(1, len(sorted_errors) + 1) / max(1, len(sorted_errors))
    return float(np.trapezoid(cumulative_risk, coverage)) if len(sorted_errors) else 0.0


def compute_fer_metrics(
    labels: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    uncertainty: Sequence[float],
    class_names: Sequence[str],
) -> Dict[str, object]:
    labels_np = np.asarray(labels, dtype=np.int64)
    probs_np = np.asarray(probabilities, dtype=np.float64)
    uncertainty_np = np.asarray(uncertainty, dtype=np.float64)
    predictions = probs_np.argmax(axis=1)
    matrix = confusion_matrix(labels_np, predictions, len(class_names))
    f1_scores = per_class_f1(matrix)
    accuracy = float((predictions == labels_np).mean()) if len(labels_np) else 0.0
    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy(matrix),
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "per_class_f1": {name: float(score) for name, score in zip(class_names, f1_scores)},
        "confusion_matrix": matrix.tolist(),
        "ece": expected_calibration_error(labels_np, probs_np),
        "uncertainty_risk_auc": uncertainty_risk_auc(labels_np, probs_np, uncertainty_np),
    }
    return metrics
