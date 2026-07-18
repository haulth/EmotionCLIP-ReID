"""Audit landmark coverage, pose, jitter, and geometry signal for a manifest."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.anatomy import (  # noqa: E402
    ANATOMY_REGIONS,
    MAX_GEOMETRY_FEATURES,
    MEDIAPIPE_GROUPS,
    REGION_FEATURE_NAMES,
    anatomy_to_model_inputs,
    load_anatomy_artifact,
)
from datasets.emotion_manifest import load_emotion_manifest  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root-dir", default="")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--output", required=True, help="Output audit JSON")
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--minimum-detection-rate", type=float, default=0.0)
    return parser.parse_args()


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "mean": None, "std": None, "p05": None, "p50": None, "p95": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _landmark_valid(landmark: Any) -> bool:
    if not isinstance(landmark, dict) or not bool(landmark.get("valid", True)):
        return False
    try:
        return math.isfinite(float(landmark["x"])) and math.isfinite(float(landmark["y"]))
    except (KeyError, TypeError, ValueError):
        return False


def _signal_to_jitter(
    values: np.ndarray,
    validity: np.ndarray,
    labels: Sequence[str],
    feature_jitter: np.ndarray,
    feature_jitter_validity: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    labels = np.asarray(labels)
    report: Dict[str, Dict[str, Any]] = {}
    for region_index, region in enumerate(ANATOMY_REGIONS):
        region_report: Dict[str, Any] = {}
        for feature_index, feature_name in enumerate(REGION_FEATURE_NAMES[region]):
            mask = validity[:, region_index, feature_index]
            selected = values[mask, region_index, feature_index]
            if selected.size:
                sample_median = float(np.median(selected))
                sample_variation = float(
                    1.4826 * np.median(np.abs(selected - sample_median))
                )
                class_medians = []
                for label in np.unique(labels):
                    class_mask = mask & (labels == label)
                    if class_mask.any():
                        class_medians.append(
                            float(np.median(values[class_mask, region_index, feature_index]))
                        )
                if len(class_medians) >= 2:
                    class_medians_array = np.asarray(class_medians)
                    class_center = float(np.median(class_medians_array))
                    signal = float(
                        1.4826 * np.median(np.abs(class_medians_array - class_center))
                    )
                else:
                    signal = None
            else:
                sample_variation = signal = None

            jitter_mask = feature_jitter_validity[:, region_index, feature_index]
            selected_jitter = feature_jitter[jitter_mask, region_index, feature_index]
            jitter = float(np.median(selected_jitter)) if selected_jitter.size else None
            ratio = (
                signal / max(jitter, 1e-6)
                if signal is not None and jitter is not None
                else None
            )
            region_report[feature_name] = {
                "valid_count": int(selected.size),
                "valid_rate": float(mask.mean()) if mask.size else 0.0,
                "robust_sample_mad": sample_variation,
                "robust_between_class_median_mad": signal,
                "feature_jitter_count": int(selected_jitter.size),
                "median_feature_jitter_std": jitter,
                "signal_to_jitter": ratio,
                "units": "geometry_feature_units",
            }
        report[region] = region_report
    return report


def main() -> None:
    args = _parse_args()
    split: Optional[str] = None if args.split == "all" else args.split
    samples = load_emotion_manifest(
        args.manifest,
        root_dir=args.root_dir or None,
        split=split,
    )
    detected = 0
    missing_artifacts = 0
    group_valid = Counter()
    group_total = Counter()
    yaw: List[float] = []
    pitch: List[float] = []
    roll: List[float] = []
    mean_jitter: List[float] = []
    jitter_success_rate: List[float] = []
    region_quality: List[np.ndarray] = []
    geometry_values: List[np.ndarray] = []
    geometry_validity: List[np.ndarray] = []
    geometry_feature_jitter: List[np.ndarray] = []
    geometry_feature_jitter_validity: List[np.ndarray] = []
    emotion_labels: List[str] = []
    class_counts = Counter()

    for sample in samples:
        class_counts[sample.emotion] += 1
        emotion_labels.append(sample.emotion)
        try:
            artifact = sample.anatomy or load_anatomy_artifact(sample.landmark_path)
        except (OSError, json.JSONDecodeError, ValueError):
            artifact = None
            missing_artifacts += 1
        detector = (artifact or {}).get("detector") or {}
        is_detected = bool((artifact or {}).get("landmarks")) and bool(detector.get("detected", True))
        detected += int(is_detected)
        landmarks = (artifact or {}).get("landmarks") or []
        if is_detected:
            for group, indices in MEDIAPIPE_GROUPS.items():
                group_total[group] += len(indices)
                group_valid[group] += sum(
                    index < len(landmarks) and _landmark_valid(landmarks[index])
                    for index in indices
                )

        pose = (artifact or {}).get("pose") or {}
        for target, key in ((yaw, "yaw"), (pitch, "pitch"), (roll, "roll")):
            try:
                value = float(pose[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                target.append(value)
        jitter = (artifact or {}).get("jitter") or {}
        try:
            mean_jitter.append(float(jitter["mean_uncertainty"]))
        except (KeyError, TypeError, ValueError):
            pass
        requested = int(jitter.get("requested_repeats", 0) or 0)
        successful = int(jitter.get("successful_repeats", 0) or 0)
        if requested:
            jitter_success_rate.append(successful / requested)
        feature_std = np.asarray(jitter.get("geometry_feature_std", []), dtype=np.float32)
        feature_valid = np.asarray(jitter.get("geometry_feature_valid", []), dtype=np.bool_)
        # Runtime descriptors currently have 12 slots. Legacy artifacts have no
        # feature-level jitter and remain explicitly invalid for this metric.
        expected_shape = (len(ANATOMY_REGIONS), MAX_GEOMETRY_FEATURES)
        if feature_std.shape != expected_shape or feature_valid.shape != expected_shape:
            feature_std = np.zeros(expected_shape, dtype=np.float32)
            feature_valid = np.zeros(expected_shape, dtype=np.bool_)
        geometry_feature_jitter.append(feature_std)
        geometry_feature_jitter_validity.append(feature_valid)

        image_size = (artifact or {}).get("image_size") or (args.image_width, args.image_height)
        inputs = anatomy_to_model_inputs(
            artifact,
            original_size=(int(image_size[0]), int(image_size[1])),
            output_size=(args.image_height, args.image_width),
        )
        region_quality.append(inputs["region_quality"].numpy())
        geometry_values.append(inputs["geometry_features"].numpy())
        geometry_validity.append(inputs["geometry_validity"].numpy())

    quality_array = np.stack(region_quality)
    value_array = np.stack(geometry_values)
    validity_array = np.stack(geometry_validity)
    feature_jitter_array = np.stack(geometry_feature_jitter)
    feature_jitter_validity_array = np.stack(geometry_feature_jitter_validity)
    detection_rate = detected / len(samples)
    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "split": args.split,
        "sample_count": len(samples),
        "class_counts": dict(sorted(class_counts.items())),
        "detection": {
            "detected": detected,
            "rate": detection_rate,
            "failure_rate": 1.0 - detection_rate,
            "missing_or_invalid_artifact": missing_artifacts,
        },
        "missing_landmark_rate": {
            "conditional_on_detection": True,
            "evaluated_detected_samples": detected,
            "by_group": {
                group: (
                    1.0 - group_valid[group] / group_total[group]
                    if group_total[group]
                    else None
                )
                for group in MEDIAPIPE_GROUPS
            },
        },
        "pose_degrees": {
            "yaw": {**_summary(yaw), "absolute_ge_30_rate": float(np.mean(np.abs(yaw) >= 30.0)) if yaw else None},
            "pitch": {**_summary(pitch), "absolute_ge_30_rate": float(np.mean(np.abs(pitch) >= 30.0)) if pitch else None},
            "roll": _summary(roll),
        },
        "augmentation_jitter": {
            "mean_landmark_uncertainty": _summary(mean_jitter),
            "successful_repeat_rate": _summary(jitter_success_rate),
        },
        "region_quality": {
            region: _summary(quality_array[:, index]) for index, region in enumerate(ANATOMY_REGIONS)
        },
        "signal_to_jitter": _signal_to_jitter(
            value_array,
            validity_array,
            emotion_labels,
            feature_jitter_array,
            feature_jitter_validity_array,
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote={output}")
    print(f"samples={len(samples)} detected={detected} detection_rate={detection_rate:.4f}")
    if detection_rate < args.minimum_detection_rate:
        raise SystemExit(
            f"Detection rate {detection_rate:.4f} is below required minimum {args.minimum_detection_rate:.4f}"
        )


if __name__ == "__main__":
    main()
