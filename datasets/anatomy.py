"""Landmark artifact contract and mask-aware facial geometry features.

The runtime representation is detector agnostic.  A detector artifact may keep
all of its native landmarks, while this module projects them into three stable
regions (upper, middle, lower), fixed-width geometry descriptors, and explicit
validity/uncertainty tensors.  Missing points are never imputed.
"""

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


ANATOMY_REGIONS = ("upper", "middle", "lower")
ANATOMY_ARTIFACT_SCHEMA_VERSION = 3
ANATOMY_DESCRIPTOR_VERSION = 3
NUM_ANATOMY_REGIONS = len(ANATOMY_REGIONS)
MAX_GEOMETRY_FEATURES = 12
MAX_REGION_LANDMARKS = 64


# Stable MediaPipe Face Mesh/Face Landmarker indices.  Left/right subgroups are
# retained for self-occlusion checks; the model consumes their three unions.
MEDIAPIPE_GROUPS = {
    "left_eye": (33, 160, 158, 133, 153, 144),
    "right_eye": (362, 385, 387, 263, 373, 380),
    "left_brow": (70, 63, 105, 66, 107),
    "right_brow": (336, 296, 334, 293, 300),
    "nose": (1, 2, 4, 5, 6, 19, 94, 98, 168, 195, 197, 327),
    "left_cheek": (50, 101, 118, 117, 123, 147, 187, 205),
    "right_cheek": (280, 330, 347, 346, 352, 376, 411, 425),
    "outer_lip": (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78),
    "inner_lip": (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95),
    "jaw": (234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454),
}

REGION_GROUPS = {
    "upper": ("left_eye", "right_eye", "left_brow", "right_brow"),
    "middle": ("nose", "left_cheek", "right_cheek"),
    "lower": ("outer_lip", "inner_lip", "jaw"),
}

REGION_FEATURE_NAMES = {
    "upper": (
        "left_ear",
        "right_ear",
        "left_brow_eye_distance",
        "right_brow_eye_distance",
        "left_brow_slope",
        "right_brow_slope",
        "eye_asymmetry",
        "brow_eye_asymmetry",
        "left_brow_curvature",
        "right_brow_curvature",
    ),
    "middle": (
        "nose_width",
        "nose_height",
        "eye_mouth_distance",
        "mid_face_width",
        "left_cheek_distance",
        "right_cheek_distance",
        "cheek_asymmetry",
    ),
    "lower": (
        "mar",
        "mouth_width",
        "mouth_opening",
        "left_corner_elevation",
        "right_corner_elevation",
        "lip_curvature",
        "jaw_opening",
        "mouth_asymmetry",
        "jaw_asymmetry",
    ),
}

# Geometry feature slots whose semantic left/right roles exchange after an
# image-space horizontal flip.  The features are first recomputed from mirrored
# coordinates (so signed slopes also change sign), then these slots are swapped.
HORIZONTAL_FLIP_FEATURE_PAIRS = {
    "upper": ((0, 1), (2, 3), (4, 5), (8, 9)),
    "middle": ((4, 5),),
    "lower": ((3, 4),),
}


def geometry_feature_definition_mask() -> torch.Tensor:
    """Return the non-padding feature slots for each anatomy region."""
    mask = torch.zeros(NUM_ANATOMY_REGIONS, MAX_GEOMETRY_FEATURES, dtype=torch.bool)
    for region_index, region in enumerate(ANATOMY_REGIONS):
        mask[region_index, : len(REGION_FEATURE_NAMES[region])] = True
    return mask


def apply_horizontal_flip_to_geometry(
    values: np.ndarray,
    validity: np.ndarray,
    uncertainty: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Swap semantic left/right descriptor slots after mirroring coordinates."""
    values = np.asarray(values).copy()
    validity = np.asarray(validity).copy()
    uncertainty = np.asarray(uncertainty).copy()
    for region_index, region in enumerate(ANATOMY_REGIONS):
        for left_index, right_index in HORIZONTAL_FLIP_FEATURE_PAIRS[region]:
            for array in (values, validity, uncertainty):
                array[region_index, [left_index, right_index]] = array[
                    region_index, [right_index, left_index]
                ].copy()
    return values, validity, uncertainty


@lru_cache(maxsize=32768)
def _load_anatomy_artifact_path(
    resolved_path: str,
    modified_time_ns: int,
    file_size: int,
) -> Dict[str, Any]:
    del modified_time_ns, file_size  # Included in the cache key for safe invalidation.
    with Path(resolved_path).open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Anatomy artifact {resolved_path} must contain a JSON object")
    return loaded


def load_anatomy_artifact(value: Any) -> Optional[Dict[str, Any]]:
    """Load an inline artifact mapping or a JSON artifact path."""
    if value is None or value == "":
        return None
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(str(value)).resolve()
    stat = path.stat()
    return _load_anatomy_artifact_path(str(path), stat.st_mtime_ns, stat.st_size)


def _coerce_score(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _artifact_arrays(artifact: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    landmarks = artifact.get("landmarks") or []
    detector = artifact.get("detector") or {}
    raw_detector_confidence = detector.get("confidence", artifact.get("detector_confidence"))
    detector_confidence_valid = raw_detector_confidence is not None
    detector_confidence = _coerce_score(raw_detector_confidence, 1.0)
    coords = []
    visibility = []
    visibility_validity = []
    confidence = []
    confidence_validity = []
    uncertainty = []
    validity = []
    for landmark in landmarks:
        if isinstance(landmark, Mapping):
            x = _coerce_score(landmark.get("x"), float("nan"))
            y = _coerce_score(landmark.get("y"), float("nan"))
            z = _coerce_score(landmark.get("z"), 0.0)
            visibility_valid = bool(
                landmark.get("visibility_valid", landmark.get("visibility") is not None)
            )
            presence_valid = bool(
                landmark.get("presence_valid", landmark.get("presence") is not None)
            )
            visible = _coerce_score(landmark.get("visibility"), 1.0)
            present = _coerce_score(landmark.get("presence"), visible)
            explicit_confidence_valid = landmark.get("confidence") is not None
            confidence_valid = bool(
                landmark.get(
                    "confidence_valid",
                    explicit_confidence_valid
                    or presence_valid
                    or visibility_valid
                    or detector_confidence_valid,
                )
            )
            conf = _coerce_score(
                landmark.get("confidence"),
                min(present, detector_confidence),
            )
            unc = max(0.0, _coerce_score(landmark.get("uncertainty", landmark.get("jitter")), 0.0))
            valid = bool(landmark.get("valid", True))
        else:
            values = list(landmark)
            x = _coerce_score(values[0] if len(values) > 0 else None, float("nan"))
            y = _coerce_score(values[1] if len(values) > 1 else None, float("nan"))
            z = _coerce_score(values[2] if len(values) > 2 else None, 0.0)
            visible = 1.0
            conf = detector_confidence
            visibility_valid = False
            confidence_valid = detector_confidence_valid
            unc = 0.0
            valid = True
        coords.append((x, y, z))
        visibility.append(np.clip(visible, 0.0, 1.0))
        visibility_validity.append(visibility_valid)
        confidence.append(np.clip(conf, 0.0, 1.0))
        confidence_validity.append(confidence_valid)
        uncertainty.append(unc)
        validity.append(valid and math.isfinite(x) and math.isfinite(y))

    coordinate_space = str(artifact.get("coordinate_space", "normalized")).lower()
    coords_array = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
    if coordinate_space in {"pixel", "pixels", "absolute"} and coords_array.size:
        width, height = artifact.get("image_size", (None, None))
        if not width or not height:
            raise ValueError("Pixel-coordinate anatomy artifacts require image_size=[width, height]")
        coords_array[:, 0] /= float(width)
        coords_array[:, 1] /= float(height)
        coords_array[:, 2] /= float(width)

    return {
        "coords": coords_array,
        "visibility": np.asarray(visibility, dtype=np.float32),
        "visibility_valid": np.asarray(visibility_validity, dtype=np.bool_),
        "confidence": np.asarray(confidence, dtype=np.float32),
        "confidence_valid": np.asarray(confidence_validity, dtype=np.bool_),
        "uncertainty": np.asarray(uncertainty, dtype=np.float32),
        "valid": np.asarray(validity, dtype=np.bool_),
    }


def transform_normalized_landmarks(
    coords: np.ndarray,
    valid: np.ndarray,
    original_size: Tuple[int, int],
    output_size: Tuple[int, int],
    horizontal_flip: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the exact center-crop geometry used by ``ImageOps.fit``."""
    coords = np.asarray(coords, dtype=np.float32).copy()
    valid = np.asarray(valid, dtype=np.bool_).copy()
    if coords.size == 0:
        return coords.reshape(-1, 3), valid

    source_w, source_h = map(float, original_size)
    output_h, output_w = map(float, output_size)
    scale = max(output_w / max(source_w, 1.0), output_h / max(source_h, 1.0))
    resized_w = source_w * scale
    resized_h = source_h * scale
    crop_left = 0.5 * (resized_w - output_w)
    crop_top = 0.5 * (resized_h - output_h)
    coords[:, 0] = (coords[:, 0] * source_w * scale - crop_left) / max(output_w, 1.0)
    coords[:, 1] = (coords[:, 1] * source_h * scale - crop_top) / max(output_h, 1.0)
    coords[:, 2] = coords[:, 2] * source_w * scale / max(output_w, 1.0)
    if horizontal_flip:
        coords[:, 0] = 1.0 - coords[:, 0]
        coords[:, 2] = -coords[:, 2]
    in_frame = (
        np.isfinite(coords[:, 0])
        & np.isfinite(coords[:, 1])
        & (coords[:, 0] >= 0.0)
        & (coords[:, 0] <= 1.0)
        & (coords[:, 1] >= 0.0)
        & (coords[:, 1] <= 1.0)
    )
    return coords, valid & in_frame


def _distance(coords: np.ndarray, first: int, second: int) -> float:
    # MediaPipe z is in the same approximate normalized scale as x.  Artifacts
    # without depth use z=0 and naturally reduce to the 2D measurement.
    return float(np.linalg.norm(coords[first, :3] - coords[second, :3]))


def _region_indices(region: str, landmark_count: int) -> Tuple[int, ...]:
    indices = []
    for group in REGION_GROUPS[region]:
        indices.extend(index for index in MEDIAPIPE_GROUPS[group] if index < landmark_count)
    return tuple(dict.fromkeys(indices))


def _feature(
    coords: np.ndarray,
    valid: np.ndarray,
    uncertainty: np.ndarray,
    indices: Iterable[int],
    fn,
) -> Tuple[float, bool, float]:
    indices = tuple(indices)
    if not indices or any(index >= len(valid) or not valid[index] for index in indices):
        return 0.0, False, 1.0
    try:
        value = float(fn())
    except (FloatingPointError, ZeroDivisionError, ValueError):
        return 0.0, False, 1.0
    if not math.isfinite(value):
        return 0.0, False, 1.0
    return value, True, float(np.mean(uncertainty[list(indices)]))


def _polyline_curvature(coords: np.ndarray, indices: Sequence[int], scale: float) -> float:
    points = coords[list(indices), :2]
    start, end = points[0], points[-1]
    chord = end - start
    chord_length = max(float(np.linalg.norm(chord)), 1e-6)
    offsets = points[1:-1] - start
    perpendicular_distance = np.abs(
        offsets[:, 0] * chord[1] - offsets[:, 1] * chord[0]
    ) / chord_length
    return float(perpendicular_distance.mean() / max(float(scale), 1e-6))


def _geometry_features(
    coords: np.ndarray,
    valid: np.ndarray,
    uncertainty: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros((NUM_ANATOMY_REGIONS, MAX_GEOMETRY_FEATURES), dtype=np.float32)
    masks = np.zeros_like(values, dtype=np.bool_)
    feature_uncertainty = np.ones_like(values, dtype=np.float32)
    if len(coords) <= 454:
        return values, masks, feature_uncertainty

    eps = 1e-6
    face_width = max(_distance(coords, 234, 454), eps) if valid[234] and valid[454] else 1.0
    face_height = max(_distance(coords, 10, 152), eps) if valid[10] and valid[152] else 1.0
    left_eye_center = coords[[33, 133], :2].mean(axis=0)
    right_eye_center = coords[[362, 263], :2].mean(axis=0)
    mouth_center = coords[[13, 14], :2].mean(axis=0)

    upper_specs = [
        ((33, 160, 144, 158, 153, 133), lambda: (_distance(coords, 160, 144) + _distance(coords, 158, 153)) / (2.0 * max(_distance(coords, 33, 133), eps))),
        ((362, 385, 380, 387, 373, 263), lambda: (_distance(coords, 385, 380) + _distance(coords, 387, 373)) / (2.0 * max(_distance(coords, 362, 263), eps))),
        ((70, 63, 105, 66, 107, 33, 133, 10, 152), lambda: abs(float(coords[[70, 63, 105, 66, 107], 1].mean() - left_eye_center[1])) / face_height),
        ((336, 296, 334, 293, 300, 362, 263, 10, 152), lambda: abs(float(coords[[336, 296, 334, 293, 300], 1].mean() - right_eye_center[1])) / face_height),
        ((70, 107), lambda: float((coords[107, 1] - coords[70, 1]) / max(abs(coords[107, 0] - coords[70, 0]), eps))),
        ((336, 300), lambda: float((coords[300, 1] - coords[336, 1]) / max(abs(coords[300, 0] - coords[336, 0]), eps))),
        ((33, 160, 144, 158, 153, 133, 362, 385, 380, 387, 373, 263), lambda: abs(((_distance(coords, 160, 144) + _distance(coords, 158, 153)) / (2.0 * max(_distance(coords, 33, 133), eps))) - ((_distance(coords, 385, 380) + _distance(coords, 387, 373)) / (2.0 * max(_distance(coords, 362, 263), eps))))),
        ((70, 63, 105, 66, 107, 33, 133, 336, 296, 334, 293, 300, 362, 263, 10, 152), lambda: abs(abs(float(coords[[70, 63, 105, 66, 107], 1].mean() - left_eye_center[1])) - abs(float(coords[[336, 296, 334, 293, 300], 1].mean() - right_eye_center[1]))) / face_height),
        ((70, 63, 105, 66, 107, 10, 152), lambda: _polyline_curvature(coords, (70, 63, 105, 66, 107), face_height)),
        ((336, 296, 334, 293, 300, 10, 152), lambda: _polyline_curvature(coords, (336, 296, 334, 293, 300), face_height)),
    ]
    middle_specs = [
        ((98, 327, 234, 454), lambda: _distance(coords, 98, 327) / face_width),
        ((168, 2, 10, 152), lambda: _distance(coords, 168, 2) / face_height),
        ((33, 133, 362, 263, 13, 14, 10, 152), lambda: abs(float(0.5 * (left_eye_center[1] + right_eye_center[1]) - mouth_center[1])) / face_height),
        ((50, 280, 234, 454), lambda: _distance(coords, 50, 280) / face_width),
        ((50, 1, 234, 454), lambda: _distance(coords, 50, 1) / face_width),
        ((280, 1, 234, 454), lambda: _distance(coords, 280, 1) / face_width),
        ((50, 280, 1, 234, 454), lambda: abs(_distance(coords, 50, 1) - _distance(coords, 280, 1)) / face_width),
    ]
    lower_specs = [
        ((13, 14, 78, 308), lambda: _distance(coords, 13, 14) / max(_distance(coords, 78, 308), eps)),
        ((61, 291, 234, 454), lambda: _distance(coords, 61, 291) / face_width),
        ((13, 14, 10, 152), lambda: _distance(coords, 13, 14) / face_height),
        ((61, 13, 14, 10, 152), lambda: float(mouth_center[1] - coords[61, 1]) / face_height),
        ((291, 13, 14, 10, 152), lambda: float(mouth_center[1] - coords[291, 1]) / face_height),
        ((61, 291, 13, 14, 10, 152), lambda: float(mouth_center[1] - 0.5 * (coords[61, 1] + coords[291, 1])) / face_height),
        ((17, 152, 10), lambda: _distance(coords, 17, 152) / face_height),
        ((61, 291, 13, 14, 10, 152), lambda: abs(float(coords[61, 1] - coords[291, 1])) / face_height),
        ((234, 454, 152), lambda: abs(_distance(coords, 234, 152) - _distance(coords, 454, 152)) / face_width),
    ]

    for region_index, specs in enumerate((upper_specs, middle_specs, lower_specs)):
        for feature_index, (indices, fn) in enumerate(specs):
            value, is_valid, feature_u = _feature(coords, valid, uncertainty, indices, fn)
            values[region_index, feature_index] = value
            masks[region_index, feature_index] = is_valid
            feature_uncertainty[region_index, feature_index] = feature_u
    return values, masks, feature_uncertainty


def _post_transform_crop_quality(coords: np.ndarray, valid: np.ndarray) -> float:
    """Measure retained face coverage after the runtime center crop."""
    if not valid.size or not valid.any():
        return 0.0
    valid_rate = float(valid.mean())
    valid_xy = coords[valid, :2]
    edge_margin = float(
        min(
            valid_xy[:, 0].min(),
            1.0 - valid_xy[:, 0].max(),
            valid_xy[:, 1].min(),
            1.0 - valid_xy[:, 1].max(),
        )
    )
    margin_quality = float(np.clip(edge_margin / 0.03, 0.0, 1.0))
    return float(np.clip(valid_rate * (0.5 + 0.5 * margin_quality), 0.0, 1.0))


def _artifact_feature_uncertainty(
    artifact: Mapping[str, Any],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    jitter = artifact.get("jitter") or {}
    values = np.asarray(jitter.get("geometry_feature_std", []), dtype=np.float32)
    valid = np.asarray(jitter.get("geometry_feature_valid", []), dtype=np.bool_)
    expected = (NUM_ANATOMY_REGIONS, MAX_GEOMETRY_FEATURES)
    if values.shape != expected or valid.shape != expected:
        return None
    valid = valid & np.isfinite(values) & (values >= 0.0)
    return np.nan_to_num(values, nan=1.0, posinf=1.0, neginf=1.0), valid


def empty_anatomy_inputs() -> Dict[str, torch.Tensor]:
    return {
        "region_landmarks": torch.zeros(NUM_ANATOMY_REGIONS, MAX_REGION_LANDMARKS, 2, dtype=torch.float32),
        "region_landmark_weights": torch.zeros(NUM_ANATOMY_REGIONS, MAX_REGION_LANDMARKS, dtype=torch.float32),
        "region_landmark_uncertainty": torch.ones(NUM_ANATOMY_REGIONS, MAX_REGION_LANDMARKS, dtype=torch.float32),
        "region_landmark_mask": torch.zeros(NUM_ANATOMY_REGIONS, MAX_REGION_LANDMARKS, dtype=torch.bool),
        "geometry_features": torch.zeros(NUM_ANATOMY_REGIONS, MAX_GEOMETRY_FEATURES, dtype=torch.float32),
        "geometry_validity": torch.zeros(NUM_ANATOMY_REGIONS, MAX_GEOMETRY_FEATURES, dtype=torch.bool),
        "geometry_uncertainty": torch.ones(NUM_ANATOMY_REGIONS, MAX_GEOMETRY_FEATURES, dtype=torch.float32),
        "region_quality": torch.zeros(NUM_ANATOMY_REGIONS, dtype=torch.float32),
        "pose_quality": torch.tensor(0.0, dtype=torch.float32),
        "pose_valid": torch.tensor(False),
        "crop_quality": torch.tensor(0.0, dtype=torch.float32),
        "anatomy_available": torch.tensor(False),
    }


def anatomy_to_model_inputs(
    artifact: Optional[Mapping[str, Any]],
    original_size: Tuple[int, int],
    output_size: Tuple[int, int],
    horizontal_flip: bool = False,
    uncertainty_tau: float = 0.05,
) -> Dict[str, torch.Tensor]:
    if not artifact or not artifact.get("landmarks"):
        return empty_anatomy_inputs()
    arrays = _artifact_arrays(artifact)
    coords, valid = transform_normalized_landmarks(
        arrays["coords"],
        arrays["valid"],
        original_size=original_size,
        output_size=output_size,
        horizontal_flip=horizontal_flip,
    )
    visibility = arrays["visibility"]
    visibility_valid = arrays["visibility_valid"]
    confidence = arrays["confidence"]
    confidence_valid = arrays["confidence_valid"]
    uncertainty = arrays["uncertainty"]
    landmark_weights = visibility * confidence * np.exp(-uncertainty / max(float(uncertainty_tau), 1e-6))
    landmark_weights = landmark_weights * valid.astype(np.float32)

    result = empty_anatomy_inputs()
    pose = artifact.get("pose") or {}
    pose_values_present = all(pose.get(key) is not None for key in ("yaw", "pitch", "roll"))
    pose_valid = bool(
        pose.get(
            "valid",
            pose.get("transformation_matrix") is not None
            or pose_values_present
            or pose.get("quality") is not None,
        )
    )
    pose_quality = (
        np.clip(
            _coerce_score(pose.get("quality", artifact.get("pose_quality")), 0.0),
            0.0,
            1.0,
        )
        if pose_valid
        else 0.0
    )
    artifact_crop_quality = np.clip(
        _coerce_score(artifact.get("crop_quality"), 1.0),
        0.0,
        1.0,
    )
    crop_quality = min(float(artifact_crop_quality), _post_transform_crop_quality(coords, valid))
    region_quality = np.zeros(NUM_ANATOMY_REGIONS, dtype=np.float32)
    for region_index, region in enumerate(ANATOMY_REGIONS):
        indices = _region_indices(region, len(coords))[:MAX_REGION_LANDMARKS]
        if not indices:
            continue
        count = len(indices)
        region_coords = coords[list(indices), :2]
        region_valid = valid[list(indices)]
        region_weights = landmark_weights[list(indices)]
        region_uncertainty = uncertainty[list(indices)]
        result["region_landmarks"][region_index, :count] = torch.from_numpy(region_coords)
        result["region_landmark_weights"][region_index, :count] = torch.from_numpy(region_weights)
        result["region_landmark_uncertainty"][region_index, :count] = torch.from_numpy(region_uncertainty)
        result["region_landmark_mask"][region_index, :count] = torch.from_numpy(region_valid)
        if region_valid.any():
            selected = np.asarray(indices)[region_valid]
            valid_rate = float(region_valid.mean())
            selected_visibility_valid = visibility_valid[selected]
            selected_confidence_valid = confidence_valid[selected]
            visible_quality = (
                float(visibility[selected][selected_visibility_valid].mean())
                if selected_visibility_valid.any()
                else 1.0
            )
            confidence_quality = (
                float(confidence[selected][selected_confidence_valid].mean())
                if selected_confidence_valid.any()
                else 1.0
            )
            measurement_valid = selected_visibility_valid | selected_confidence_valid
            measurement_coverage = (
                float(measurement_valid.mean()) if measurement_valid.any() else 1.0
            )
            jitter_quality = float(np.exp(-uncertainty[selected].mean() / max(float(uncertainty_tau), 1e-6)))
            region_quality[region_index] = (
                valid_rate
                * visible_quality
                * confidence_quality
                * measurement_coverage
                * jitter_quality
            )

    geometry, geometry_valid, geometry_uncertainty = _geometry_features(coords, valid, uncertainty)
    artifact_feature_uncertainty = _artifact_feature_uncertainty(artifact)
    if horizontal_flip:
        geometry, geometry_valid, geometry_uncertainty = apply_horizontal_flip_to_geometry(
            geometry,
            geometry_valid,
            geometry_uncertainty,
        )
        if artifact_feature_uncertainty is not None:
            feature_uncertainty, feature_uncertainty_valid = artifact_feature_uncertainty
            feature_uncertainty, feature_uncertainty_valid, _ = apply_horizontal_flip_to_geometry(
                feature_uncertainty,
                feature_uncertainty_valid,
                np.zeros_like(feature_uncertainty),
            )
            artifact_feature_uncertainty = feature_uncertainty, feature_uncertainty_valid
    if artifact_feature_uncertainty is not None:
        feature_uncertainty, feature_uncertainty_valid = artifact_feature_uncertainty
        usable_uncertainty = geometry_valid & feature_uncertainty_valid
        geometry_uncertainty = np.where(usable_uncertainty, feature_uncertainty, 1.0).astype(
            np.float32
        )
    result["geometry_features"] = torch.from_numpy(geometry)
    result["geometry_validity"] = torch.from_numpy(geometry_valid)
    result["geometry_uncertainty"] = torch.from_numpy(geometry_uncertainty)
    result["region_quality"] = torch.from_numpy(region_quality * pose_quality * crop_quality)
    result["pose_quality"] = torch.tensor(float(pose_quality), dtype=torch.float32)
    result["pose_valid"] = torch.tensor(pose_valid)
    result["crop_quality"] = torch.tensor(float(crop_quality), dtype=torch.float32)
    result["anatomy_available"] = torch.tensor(bool(valid.any()))
    return result


def fit_class_geometry_statistics(
    geometry_features: torch.Tensor,
    geometry_validity: torch.Tensor,
    geometry_uncertainty: torch.Tensor,
    region_quality: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    minimum_samples: int = 8,
) -> Dict[str, torch.Tensor]:
    """Fit train-only median/MAD statistics without imputing invalid features."""
    if geometry_features.ndim != 3:
        raise ValueError("geometry_features must have shape (N, regions, features)")
    medians = torch.zeros(num_classes, *geometry_features.shape[1:], dtype=torch.float32)
    scales = torch.zeros_like(medians)
    valid_rate = torch.zeros_like(medians)
    median_uncertainty = torch.ones_like(medians)
    class_quality = torch.zeros(num_classes, geometry_features.shape[1], dtype=torch.float32)
    sample_count = torch.zeros(num_classes, dtype=torch.long)
    features = geometry_features.detach().float().cpu()
    masks = geometry_validity.detach().bool().cpu()
    uncertainties = geometry_uncertainty.detach().float().cpu()
    qualities = region_quality.detach().float().cpu()
    labels = labels.detach().long().cpu()
    definition_mask = geometry_feature_definition_mask()
    if definition_mask.shape != masks.shape[1:]:
        raise ValueError(
            f"geometry feature shape {tuple(masks.shape[1:])} does not match the declared "
            f"anatomy descriptor {tuple(definition_mask.shape)}"
        )

    for class_index in range(num_classes):
        selected = labels == class_index
        sample_count[class_index] = int(selected.sum())
        if not selected.any():
            continue
        class_masks = masks[selected]
        valid_rate[class_index] = class_masks.float().mean(dim=0)
        for region_index in range(features.shape[1]):
            q = qualities[selected, region_index]
            count_factor = min(1.0, float(selected.sum()) / max(float(minimum_samples), 1.0))
            declared = definition_mask[region_index]
            declared_masks = class_masks[:, region_index, declared]
            declared_uncertainty = uncertainties[selected][:, region_index, declared]
            valid_uncertainty = declared_uncertainty[declared_masks]
            mean_uncertainty = (
                valid_uncertainty.mean() if valid_uncertainty.numel() else torch.tensor(1.0)
            )
            uncertainty_factor = torch.exp(-mean_uncertainty.nan_to_num(1.0))
            class_quality[class_index, region_index] = (
                q.mean() * declared_masks.float().mean() * uncertainty_factor * count_factor
            ).clamp(0.0, 1.0)
            for feature_index in range(features.shape[2]):
                feature_mask = class_masks[:, region_index, feature_index]
                if not feature_mask.any():
                    continue
                values = features[selected, region_index, feature_index][feature_mask]
                feature_uncertainties = uncertainties[
                    selected,
                    region_index,
                    feature_index,
                ][feature_mask]
                median = values.median()
                mad = (values - median).abs().median()
                medians[class_index, region_index, feature_index] = median
                scales[class_index, region_index, feature_index] = 1.4826 * mad
                median_uncertainty[class_index, region_index, feature_index] = (
                    feature_uncertainties.median().nan_to_num(1.0).clamp_min(0.0)
                )
    return {
        "median": medians,
        "scale": scales,
        "quality": class_quality,
        "valid_rate": valid_rate,
        "uncertainty": median_uncertainty,
        "sample_count": sample_count,
    }
