"""Notebook helpers for EmotionCLIP metric discovery and plotting."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


MetricRow = Dict[str, Any]
MetricBundle = Dict[str, Any]


def metric_epoch(path: Path) -> int:
    match = re.search(r"metrics_epoch_(\d+)\.json$", Path(path).name)
    return int(match.group(1)) if match else -1


def unique_paths(*paths: object) -> List[Path]:
    seen = set()
    result: List[Path] = []
    for raw_path in paths:
        if raw_path is None:
            continue
        path = Path(raw_path).expanduser()
        try:
            key = str(path.resolve()) if path.exists() else str(path)
        except OSError:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().rstrip(",")
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    if text.endswith("s"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_existing(paths: Iterable[object]) -> Optional[Path]:
    for path in unique_paths(*paths):
        if path.exists():
            return path
    return None


def _best_epoch_from_uncertainty(paths: Iterable[object]) -> Optional[int]:
    path = _first_existing(paths)
    if path is None:
        return None
    try:
        payload = _read_json(path)
    except json.JSONDecodeError:
        return None
    value = payload.get("best_checkpoint_epoch")
    return int(value) if value is not None else None


def _finalize_metric_bundle(
    result_dir: Path,
    metric_files: Sequence[Path],
    metric_history: List[MetricRow],
    metrics_by_epoch: Dict[int, Dict[str, Any]],
    source: Optional[Path],
    source_kind: str,
) -> MetricBundle:
    metric_history.sort(key=lambda row: int(row["epoch"]))
    if metric_history:
        latest_epoch = int(metric_history[-1]["epoch"])
        best_metric = max(metric_history, key=lambda row: row.get("macro_f1") or -1)
        metrics_latest = metrics_by_epoch.get(latest_epoch, metric_history[-1])
        metrics_best = metrics_by_epoch.get(int(best_metric["epoch"]), best_metric)
    else:
        latest_epoch = None
        best_metric = None
        metrics_latest = {}
        metrics_best = {}

    return {
        "result_dir": result_dir,
        "metric_files": list(metric_files),
        "metric_history": metric_history,
        "metrics_by_epoch": metrics_by_epoch,
        "latest_epoch": latest_epoch,
        "best_metric": best_metric,
        "metrics_latest": metrics_latest,
        "metrics_best": metrics_best,
        "source": source,
        "source_kind": source_kind,
    }


def _load_json_metrics(candidate_output_dirs: Sequence[object]) -> Optional[MetricBundle]:
    for candidate in unique_paths(*candidate_output_dirs):
        files = sorted(candidate.glob("metrics_epoch_*.json"), key=metric_epoch) if candidate.exists() else []
        if not files:
            continue
        metric_history: List[MetricRow] = []
        metrics_by_epoch: Dict[int, Dict[str, Any]] = {}
        for path in files:
            epoch = metric_epoch(path)
            payload = _read_json(path)
            payload["epoch"] = epoch
            metrics_by_epoch[epoch] = payload
            row: MetricRow = {"epoch": epoch}
            for key in [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "avg_uncertainty",
                "avg_confidence",
                "ece",
                "uncertainty_risk_auc",
                "num_samples",
            ]:
                row[key] = payload.get(key)
            metric_history.append(row)
        return _finalize_metric_bundle(candidate, files, metric_history, metrics_by_epoch, files[-1], "json")
    return None


def _load_csv_metrics(
    candidate_output_dirs: Sequence[object],
    csv_candidates: Sequence[object],
    summary_candidates: Sequence[object],
    uncertainty_candidates: Sequence[object],
) -> MetricBundle:
    csv_path = _first_existing(csv_candidates)
    metric_history: List[MetricRow] = []
    metrics_by_epoch: Dict[int, Dict[str, Any]] = {}

    if csv_path is not None:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for raw_row in csv.DictReader(handle):
                epoch_value = raw_row.get("epoch")
                if epoch_value in {None, ""}:
                    continue
                epoch = int(float(epoch_value))
                row = metrics_by_epoch.setdefault(epoch, {"epoch": epoch})
                for csv_key, metric_key in [
                    ("accuracy", "accuracy"),
                    ("balanced_acc", "balanced_accuracy"),
                    ("balanced_accuracy", "balanced_accuracy"),
                    ("macro_f1", "macro_f1"),
                    ("ece", "ece"),
                    ("uncertainty_risk_auc", "uncertainty_risk_auc"),
                    ("avg_confidence", "avg_confidence"),
                    ("avg_uncertainty", "avg_uncertainty"),
                    ("num_samples", "num_samples"),
                ]:
                    value = _to_float(raw_row.get(csv_key))
                    if value is not None:
                        row[metric_key] = value
        metric_history = [metrics_by_epoch[epoch] for epoch in sorted(metrics_by_epoch)]

    summary_path = _first_existing(summary_candidates)
    if summary_path is not None:
        try:
            summary = _read_json(summary_path)
        except json.JSONDecodeError:
            summary = {}
        summary_epoch = _best_epoch_from_uncertainty(uncertainty_candidates)
        if summary_epoch is None and metric_history:
            summary_epoch = int(max(metric_history, key=lambda row: row.get("macro_f1") or -1)["epoch"])
        if summary_epoch is not None:
            row = metrics_by_epoch.setdefault(summary_epoch, {"epoch": summary_epoch})
            row.update(summary)
            if row not in metric_history:
                metric_history.append(row)

    fallback_dir = csv_path.parent if csv_path is not None else unique_paths(*candidate_output_dirs)[0]
    return _finalize_metric_bundle(fallback_dir, [], metric_history, metrics_by_epoch, csv_path, "csv")


def load_validation_metrics(
    candidate_output_dirs: Sequence[object],
    csv_candidates: Sequence[object] = (),
    summary_candidates: Sequence[object] = (),
    uncertainty_candidates: Sequence[object] = (),
) -> MetricBundle:
    """Load validation metrics from per-epoch JSON first, then CSV fallbacks."""
    json_bundle = _load_json_metrics(candidate_output_dirs)
    if json_bundle is not None:
        return json_bundle
    return _load_csv_metrics(candidate_output_dirs, csv_candidates, summary_candidates, uncertainty_candidates)


def print_validation_summary(bundle: MetricBundle) -> None:
    print("Using RESULT_DIR:", bundle["result_dir"])
    print("Metric source:", bundle.get("source") or "none", f"({bundle.get('source_kind')})")
    metric_files = bundle.get("metric_files") or []
    if metric_files:
        print("Metric files:", [path.name for path in metric_files[-5:]])

    metric_history = bundle.get("metric_history") or []
    if not metric_history:
        print("Chưa có metrics_epoch_*.json/validation_metrics.csv. Hãy chạy train trước, hoặc kiểm tra OUTPUT_DIR.")
        return

    metrics_latest = bundle.get("metrics_latest") or {}
    metrics_best = bundle.get("metrics_best") or {}
    best_metric = bundle.get("best_metric") or {}
    latest_epoch = bundle.get("latest_epoch")
    print(f"Latest epoch: {latest_epoch}")
    for key in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "avg_uncertainty",
        "avg_confidence",
        "ece",
        "uncertainty_risk_auc",
        "num_samples",
    ]:
        if key in metrics_latest:
            print(key, metrics_latest.get(key))
    if best_metric:
        print(f"Best macro_f1 epoch: {best_metric['epoch']} ({best_metric.get('macro_f1', 0):.4f})")
    per_class_payload = _pick_payload(bundle, "per_class_f1")
    if per_class_payload:
        print("Per-class F1:", per_class_payload.get("per_class_f1"))


def _finite_xy(metric_history: Sequence[Mapping[str, Any]], key: str) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for row in metric_history:
        value = _to_float(row.get(key))
        if value is None or not np.isfinite(value):
            continue
        xs.append(float(row["epoch"]))
        ys.append(value)
    return xs, ys


def _format_epoch_axis(ax: Any, epochs: np.ndarray) -> None:
    from matplotlib.ticker import MaxNLocator

    if len(epochs) == 1:
        ax.set_xlim(float(epochs[0]) - 0.5, float(epochs[0]) + 0.5)
    elif len(epochs) > 1:
        ax.margins(x=0.04)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _plot_metric_lines(
    ax: Any,
    metric_history: Sequence[Mapping[str, Any]],
    series: Sequence[Tuple[str, str, Optional[str]]],
    title: str,
    ylabel: str,
    epochs: np.ndarray,
    ylim: Optional[Tuple[float, float]] = None,
    best_epoch: Optional[int] = None,
    best_label: Optional[str] = None,
    empty_text: str = "No per-epoch values",
) -> None:
    has_line = False
    for key, label, color in series:
        xs, ys = _finite_xy(metric_history, key)
        if not xs:
            continue
        kwargs = {"marker": "o", "linewidth": 1.8, "label": label}
        if color:
            kwargs["color"] = color
        ax.plot(xs, ys, **kwargs)
        has_line = True

    if has_line and best_epoch is not None:
        ax.axvline(best_epoch, color="#555555", linestyle="--", linewidth=1, label=best_label)
    if not has_line:
        ax.text(0.5, 0.5, empty_text, ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    _format_epoch_axis(ax, epochs)
    ax.grid(alpha=0.25)
    if has_line:
        ax.legend()


def _pick_payload(bundle: MetricBundle, *required_keys: str) -> Dict[str, Any]:
    candidates: List[Mapping[str, Any]] = [
        bundle.get("metrics_latest") or {},
        bundle.get("metrics_best") or {},
    ]
    metrics_by_epoch = bundle.get("metrics_by_epoch") or {}
    for epoch in sorted(metrics_by_epoch, reverse=True):
        candidates.append(metrics_by_epoch[epoch])
    for payload in candidates:
        if all(payload.get(key) for key in required_keys):
            return dict(payload)
    return {}


def plot_validation_metric_curves(
    bundle: MetricBundle,
    dataset_name: str,
    file_prefix: str,
    show: bool = True,
) -> Optional[Path]:
    import matplotlib.pyplot as plt

    metric_history = bundle.get("metric_history") or []
    if not metric_history:
        print("Chưa có metric_history để vẽ. Hãy chạy cell metrics ở trên sau khi train/eval.")
        return None

    epochs = np.array([row["epoch"] for row in metric_history], dtype=float)
    best_metric = bundle.get("best_metric") or {}
    best_epoch = int(best_metric["epoch"]) if best_metric else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    _plot_metric_lines(
        axes[0, 0],
        metric_history,
        [
            ("accuracy", "Accuracy", "#1f77b4"),
            ("balanced_accuracy", "Balanced acc", "#ff7f0e"),
            ("macro_f1", "Macro F1", "#2ca02c"),
        ],
        "Validation scores",
        "Score",
        epochs,
        ylim=(0, 1.02),
        best_epoch=best_epoch,
        best_label="Best macro F1",
    )
    _plot_metric_lines(
        axes[0, 1],
        metric_history,
        [
            ("avg_confidence", "Avg confidence", "#4c78a8"),
            ("avg_uncertainty", "Avg uncertainty", "#e45756"),
        ],
        "Confidence vs uncertainty",
        "Value",
        epochs,
        ylim=(0, 1.02),
        empty_text="No confidence/uncertainty history",
    )
    _plot_metric_lines(
        axes[1, 0],
        metric_history,
        [
            ("ece", "ECE", "#f58518"),
            ("uncertainty_risk_auc", "Uncertainty-risk AUC", "#72b7b2"),
        ],
        "Calibration and risk",
        "Lower is better",
        epochs,
        ylim=(0, 1.02),
        empty_text="No calibration/risk history",
    )

    f1_payload = _pick_payload(bundle, "per_class_f1")
    latest_f1 = f1_payload.get("per_class_f1", {})
    ax = axes[1, 1]
    if latest_f1:
        names = list(latest_f1.keys())
        values = [float(latest_f1[name]) for name in names]
        y = np.arange(len(names))
        ax.barh(y, values, color="#6a9f58")
        ax.set_yticks(y, names)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_xlabel("F1")
        epoch = f1_payload.get("epoch", bundle.get("latest_epoch"))
        ax.set_title(f"{dataset_name} per-class F1 at epoch {epoch}")
        ax.grid(axis="x", alpha=0.25)
        for idx, value in enumerate(values):
            ax.text(min(value + 0.015, 0.98), idx, f"{value:.2f}", va="center", fontsize=9)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No per-class F1", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    out_path = Path(bundle["result_dir"]) / f"{file_prefix}_validation_metric_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print("Saved:", out_path)
    if show:
        plt.show()
    return out_path


def _parse_log_training_history(path: Path) -> List[MetricRow]:
    rows: List[MetricRow] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.search(r"\bStage\s*([12])\s+epoch=(\d+)\s+done\b", line)
        if not match:
            continue
        fields = dict(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line))
        row: MetricRow = {"stage": int(match.group(1)), "epoch": int(match.group(2))}
        for key in ["loss", "cls", "align", "unc_loss", "pred_unc", "conf", "acc", "lr", "time"]:
            value = _to_float(fields.get(key))
            if value is None:
                continue
            row["time_sec" if key == "time" else key] = value
        rows.append(row)
    return rows


def _parse_csv_training_history(path: Path) -> List[MetricRow]:
    rows: List[MetricRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            if not raw_row.get("stage") or not raw_row.get("epoch"):
                continue
            row: MetricRow = {
                "stage": int(float(raw_row["stage"])),
                "epoch": int(float(raw_row["epoch"])),
            }
            for key in ["loss", "cls", "align", "unc_loss", "pred_unc", "conf", "acc", "lr"]:
                value = _to_float(raw_row.get(key))
                if value is not None:
                    row[key] = value
            time_value = _to_float(raw_row.get("time_sec", raw_row.get("time")))
            if time_value is not None:
                row["time_sec"] = time_value
            rows.append(row)
    return rows


def load_training_history(
    candidate_dirs: Sequence[object],
    log_names: Sequence[str] = (
        "train.log",
        "notebook_console.log",
        "training_notebook_raw.log",
        "training_events.log",
        "training_notebook_clean.log",
    ),
    csv_candidates: Sequence[object] = (),
) -> Tuple[List[MetricRow], Optional[Path]]:
    log_candidates: List[Path] = []
    for base in unique_paths(*candidate_dirs):
        log_candidates.extend(base / name for name in log_names)

    for path in unique_paths(*log_candidates):
        if not path.exists():
            continue
        rows = _parse_log_training_history(path)
        if rows:
            return sorted(rows, key=lambda row: (row["stage"], row["epoch"])), path

    for path in unique_paths(*csv_candidates):
        if not path.exists():
            continue
        rows = _parse_csv_training_history(path)
        if rows:
            return sorted(rows, key=lambda row: (row["stage"], row["epoch"])), path

    return [], None


def plot_training_metric_curves(
    training_history: Sequence[Mapping[str, Any]],
    training_source: Optional[Path],
    result_dir: object,
    file_prefix: str,
    show: bool = True,
) -> Optional[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if not training_history:
        print("Chưa tìm thấy log/CSV training để vẽ loss. Sau khi chạy train, mở lại cell này.")
        return None

    print("Training source:", training_source)
    rows = [dict(row) for row in training_history]
    stage_offsets: Dict[int, int] = {}
    offset = 0
    stages = sorted({int(row["stage"]) for row in rows})
    for stage in stages:
        stage_offsets[stage] = offset
        offset += max(int(row["epoch"]) for row in rows if int(row["stage"]) == stage)
    for row in rows:
        row["global_epoch"] = stage_offsets[int(row["stage"])] + int(row["epoch"])

    plot_keys = [
        ("loss", "Loss"),
        ("cls", "Classification loss"),
        ("align", "Alignment loss"),
        ("unc_loss", "Uncertainty loss"),
        ("acc", "Train accuracy"),
        ("conf", "Confidence"),
        ("pred_unc", "Predicted uncertainty"),
        ("time_sec", "Epoch time (sec)"),
    ]
    plot_keys = [(key, label) for key, label in plot_keys if any(key in row for row in rows)]
    if not plot_keys:
        print("Training history không có metric số để vẽ.")
        return None

    cols = 2
    subplot_rows = int(np.ceil(len(plot_keys) / cols))
    fig, axes = plt.subplots(subplot_rows, cols, figsize=(13, 3.8 * subplot_rows), squeeze=False)
    axes = axes.ravel()

    for ax, (key, label) in zip(axes, plot_keys):
        for stage in stages:
            stage_rows = [row for row in rows if int(row["stage"]) == stage and key in row]
            if not stage_rows:
                continue
            x = [row["global_epoch"] for row in stage_rows]
            y = [row[key] for row in stage_rows]
            ax.plot(x, y, marker="o", linewidth=1.8, label=f"Stage {stage}")
        ax.set_title(label)
        ax.set_xlabel("Global epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25)
        ax.legend()

    for ax in axes[len(plot_keys):]:
        ax.axis("off")

    plt.tight_layout()
    out_path = Path(result_dir) / f"{file_prefix}_training_metric_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print("Saved:", out_path)
    if show:
        plt.show()
    return out_path


def plot_confusion_matrix_and_f1(
    bundle: MetricBundle,
    dataset_name: str,
    file_prefix: str,
    show: bool = True,
) -> Optional[Path]:
    import matplotlib.pyplot as plt

    payload = _pick_payload(bundle, "confusion_matrix", "per_class_f1")
    if not payload:
        print("Chưa có confusion_matrix hoặc per_class_f1 trong metrics.")
        return None

    matrix = np.asarray(payload.get("confusion_matrix", []), dtype=float)
    class_names = list((payload.get("per_class_f1") or {}).keys())
    if matrix.size == 0 or not class_names:
        print("metrics không có confusion_matrix hoặc per_class_f1.")
        return None
    if matrix.shape[0] != len(class_names):
        class_names = [str(idx) for idx in range(matrix.shape[0])]

    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.15, 1.0]})
    im = axes[0].imshow(matrix_norm, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_xticks(np.arange(len(class_names)), class_names, rotation=35, ha="right")
    axes[0].set_yticks(np.arange(len(class_names)), class_names)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Ground truth")
    epoch = payload.get("epoch", bundle.get("latest_epoch"))
    axes[0].set_title(f"{dataset_name} normalized confusion matrix, epoch {epoch}")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix_norm[row, col]
            count = int(matrix[row, col])
            color = "white" if value > 0.55 else "black"
            axes[0].text(col, row, f"{value:.2f}\n{count}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    per_class_f1 = payload.get("per_class_f1") or {}
    f1_values = [float(per_class_f1[name]) for name in class_names]
    y = np.arange(len(class_names))
    axes[1].barh(y, f1_values, color="#6a9f58")
    axes[1].set_yticks(y, class_names)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("F1")
    axes[1].set_title("Per-class F1")
    axes[1].grid(axis="x", alpha=0.25)
    for idx, value in enumerate(f1_values):
        axes[1].text(min(value + 0.015, 0.98), idx, f"{value:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    out_path = Path(bundle["result_dir"]) / f"{file_prefix}_confusion_matrix_per_class_f1.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print("Saved:", out_path)
    if show:
        plt.show()
    return out_path
