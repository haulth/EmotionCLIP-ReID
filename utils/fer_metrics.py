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


def adaptive_calibration_error(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 15) -> float:
    if len(labels) == 0:
        return 0.0
    confidences = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == labels).astype(np.float64)
    bins = np.array_split(np.argsort(confidences), min(n_bins, len(labels)))
    return float(
        sum(
            (len(indices) / len(labels))
            * abs(float(correct[indices].mean() - confidences[indices].mean()))
            for indices in bins
            if len(indices)
        )
    )


def classwise_calibration_error(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 15) -> float:
    if len(labels) == 0:
        return 0.0
    errors = []
    for class_idx in range(probabilities.shape[1]):
        class_labels = (labels == class_idx).astype(np.float64)
        class_probs = probabilities[:, class_idx]
        error = 0.0
        for bin_idx in range(n_bins):
            lower, upper = bin_idx / n_bins, (bin_idx + 1) / n_bins
            mask = (class_probs >= lower) & (class_probs <= upper if bin_idx == n_bins - 1 else class_probs < upper)
            if mask.any():
                error += mask.mean() * abs(float(class_labels[mask].mean() - class_probs[mask].mean()))
        errors.append(error)
    return float(np.mean(errors))


def negative_log_likelihood(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    true_probabilities = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(true_probabilities, 1e-12, 1.0)).mean())


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    return float(np.square(probabilities - one_hot).sum(axis=1).mean())


def uncertainty_risk_auc(labels: np.ndarray, probabilities: np.ndarray, uncertainty: np.ndarray) -> float:
    predictions = probabilities.argmax(axis=1)
    errors = (predictions != labels).astype(np.float32)
    order = np.argsort(uncertainty)
    sorted_errors = errors[order]
    cumulative_risk = np.cumsum(sorted_errors) / np.arange(1, len(sorted_errors) + 1)
    coverage = np.arange(1, len(sorted_errors) + 1) / max(1, len(sorted_errors))
    return float(np.trapezoid(cumulative_risk, coverage)) if len(sorted_errors) else 0.0


def excess_aurc(labels: np.ndarray, probabilities: np.ndarray, uncertainty: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    predictions = probabilities.argmax(axis=1)
    errors = (predictions != labels).astype(np.float64)
    model_aurc = uncertainty_risk_auc(labels, probabilities, uncertainty)
    oracle_uncertainty = errors
    oracle_aurc = uncertainty_risk_auc(labels, probabilities, oracle_uncertainty)
    return float(max(0.0, model_aurc - oracle_aurc))


def risk_at_coverage(labels: np.ndarray, probabilities: np.ndarray, uncertainty: np.ndarray, coverage: float) -> float:
    if len(labels) == 0:
        return 0.0
    count = max(1, int(np.ceil(float(coverage) * len(labels))))
    retained = np.argsort(uncertainty)[:count]
    errors = probabilities.argmax(axis=1) != labels
    return float(errors[retained].mean())


def _binary_ranking_metrics(targets: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        return 0.0, 0.0

    order = np.argsort(-scores, kind="mergesort")
    sorted_targets = targets[order]
    sorted_scores = scores[order]
    threshold_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    tp = np.cumsum(sorted_targets)[threshold_ends].astype(np.float64)
    fp = (threshold_ends + 1).astype(np.float64) - tp
    tpr = np.r_[0.0, tp / positives]
    fpr = np.r_[0.0, fp / negatives]
    auroc = float(np.trapezoid(tpr, fpr))

    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1.0)
    previous_recall = np.r_[0.0, recall[:-1]]
    aupr = float(np.sum((recall - previous_recall) * precision))
    return auroc, aupr


def ood_detection_metrics(
    in_distribution_uncertainty: Sequence[float],
    ood_uncertainty: Sequence[float],
) -> Dict[str, float]:
    """Evaluate uncertainty as an OOD score (larger means more likely OOD)."""
    id_scores = np.asarray(in_distribution_uncertainty, dtype=np.float64)
    ood_scores = np.asarray(ood_uncertainty, dtype=np.float64)
    if len(id_scores) == 0 or len(ood_scores) == 0:
        raise ValueError("both in-distribution and OOD uncertainty scores are required")
    targets = np.r_[np.zeros(len(id_scores), dtype=np.int64), np.ones(len(ood_scores), dtype=np.int64)]
    scores = np.r_[id_scores, ood_scores]
    auroc, aupr = _binary_ranking_metrics(targets, scores)

    thresholds = np.unique(scores)[::-1]
    fpr_at_95 = 1.0
    for threshold in thresholds:
        selected = scores >= threshold
        tpr = float(selected[targets == 1].mean())
        if tpr >= 0.95:
            fpr_at_95 = float(selected[targets == 0].mean())
            break
    return {"ood_auroc": auroc, "ood_aupr": aupr, "ood_fpr95": fpr_at_95}


def compute_fer_metrics(
    labels: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    uncertainty: Sequence[float],
    class_names: Sequence[str],
) -> Dict[str, object]:
    labels_np = np.asarray(labels, dtype=np.int64)
    probs_np = np.asarray(probabilities, dtype=np.float64)
    uncertainty_np = np.asarray(uncertainty, dtype=np.float64)
    if len(labels_np) == 0:
        raise ValueError("compute_fer_metrics requires at least one sample")
    if probs_np.ndim != 2 or probs_np.shape[0] != len(labels_np):
        raise ValueError("probabilities must have shape (num_samples, num_classes)")
    if uncertainty_np.shape != (len(labels_np),):
        raise ValueError("uncertainty must have shape (num_samples,)")
    predictions = probs_np.argmax(axis=1)
    matrix = confusion_matrix(labels_np, predictions, len(class_names))
    f1_scores = per_class_f1(matrix)
    accuracy = float((predictions == labels_np).mean()) if len(labels_np) else 0.0
    errors = (predictions != labels_np).astype(np.int64)
    error_auroc, error_aupr = _binary_ranking_metrics(errors, uncertainty_np)
    aurc = uncertainty_risk_auc(labels_np, probs_np, uncertainty_np)
    entropy = -np.sum(probs_np * np.log(np.clip(probs_np, 1e-12, 1.0)), axis=1)
    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy(matrix),
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "per_class_f1": {name: float(score) for name, score in zip(class_names, f1_scores)},
        "confusion_matrix": matrix.tolist(),
        "ece": expected_calibration_error(labels_np, probs_np),
        "adaptive_ece": adaptive_calibration_error(labels_np, probs_np),
        "classwise_ece": classwise_calibration_error(labels_np, probs_np),
        "nll": negative_log_likelihood(labels_np, probs_np),
        "brier": brier_score(labels_np, probs_np),
        "aurc": aurc,
        "eaurc": excess_aurc(labels_np, probs_np, uncertainty_np),
        "uncertainty_risk_auc": aurc,
        "risk_at_coverage_50": risk_at_coverage(labels_np, probs_np, uncertainty_np, 0.5),
        "risk_at_coverage_80": risk_at_coverage(labels_np, probs_np, uncertainty_np, 0.8),
        "risk_at_coverage_90": risk_at_coverage(labels_np, probs_np, uncertainty_np, 0.9),
        "error_auroc": error_auroc,
        "error_aupr": error_aupr,
        "avg_entropy": float(entropy.mean()),
    }
    return metrics
