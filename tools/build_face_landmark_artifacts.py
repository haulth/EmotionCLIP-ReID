"""Build detector-agnostic anatomy artifacts for an emotion JSONL manifest.

MediaPipe is deliberately isolated in this offline tool.  Training and
inference consume the JSON contract in ``datasets.anatomy`` and therefore do
not depend on MediaPipe or on a particular detector implementation.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.anatomy import (  # noqa: E402
    ANATOMY_ARTIFACT_SCHEMA_VERSION,
    anatomy_to_model_inputs,
)


ARTIFACT_SCHEMA_VERSION = ANATOMY_ARTIFACT_SCHEMA_VERSION


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Input emotion JSONL manifest")
    parser.add_argument("--output-manifest", required=True, help="JSONL manifest with landmark_path fields")
    parser.add_argument("--artifact-root", required=True, help="Directory for one anatomy JSON file per image")
    parser.add_argument("--model-path", required=True, help="MediaPipe face_landmarker.task model")
    parser.add_argument("--root-dir", default="", help="Root used to resolve relative image_path fields")
    parser.add_argument("--jitter-repeats", type=int, default=4, help="Extra translated detections used for uncertainty")
    parser.add_argument("--jitter-pixels", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_mediapipe():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is required only for artifact generation. Install the emotion environment "
            "or run `pip install mediapipe`, then retry."
        ) from exc
    return mp, mp_python, vision


def _resolve_image_path(record: Dict[str, Any], manifest: Path, root_dir: str) -> Path:
    value = Path(str(record["image_path"]))
    if value.is_absolute():
        return value
    base = Path(root_dir) if root_dir else manifest.parent
    return (base / value).resolve()


def _artifact_path(image_path: Path, artifact_root: Path) -> Path:
    digest = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:12]
    safe_stem = "".join(character if character.isalnum() or character in "-_" else "_" for character in image_path.stem)
    return artifact_root / f"{safe_stem}_{digest}.json"


def _shift_image(image: Image.Image, dx: int, dy: int) -> Image.Image:
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1.0, 0.0, -float(dx), 0.0, 1.0, -float(dy)),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )


def _detect(landmarker, mp, image: Image.Image) -> Tuple[Optional[List[Any]], Optional[np.ndarray]]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return None, None
    matrix = None
    matrices = getattr(result, "facial_transformation_matrixes", None)
    if matrices is not None and len(matrices) > 0:
        candidate = np.asarray(matrices[0], dtype=np.float32)
        if candidate.size >= 16:
            matrix = candidate.reshape(4, 4)
    return list(result.face_landmarks[0]), matrix


def _pose_from_matrix(matrix: Optional[np.ndarray]) -> Dict[str, Any]:
    if matrix is None:
        return {"yaw": None, "pitch": None, "roll": None, "quality": 0.0, "valid": False}
    rotation = matrix[:3, :3]
    sy = math.sqrt(float(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(-rotation[2, 0]), sy)
        roll = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        pitch = math.atan2(float(-rotation[1, 2]), float(rotation[1, 1]))
        yaw = math.atan2(float(-rotation[2, 0]), sy)
        roll = 0.0
    yaw, pitch, roll = (math.degrees(value) for value in (yaw, pitch, roll))
    quality = math.exp(-((abs(yaw) / 45.0) ** 2 + (abs(pitch) / 35.0) ** 2))
    return {
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "quality": float(np.clip(quality, 0.0, 1.0)),
        "valid": True,
        "transformation_matrix": matrix.tolist(),
    }


def _score(landmark: Any, name: str, default: float) -> float:
    value = getattr(landmark, name, None)
    if value is None:
        return float(default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _score_with_validity(landmark: Any, name: str, default: float) -> Tuple[float, bool]:
    value = getattr(landmark, name, None)
    if value is None:
        return float(default), False
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default), False
    if not math.isfinite(value):
        return float(default), False
    return value, True


def _jitter_offsets(repeats: int, pixels: int, rng: np.random.Generator) -> Iterable[Tuple[int, int]]:
    for _ in range(max(0, repeats)):
        dx = int(rng.integers(-pixels, pixels + 1)) if pixels > 0 else 0
        dy = int(rng.integers(-pixels, pixels + 1)) if pixels > 0 else 0
        if pixels > 0 and dx == 0 and dy == 0:
            dx = pixels
        yield dx, dy


def _feature_jitter_from_observations(
    observations: Sequence[np.ndarray],
    landmarks: Sequence[Dict[str, Any]],
    image_size: Tuple[int, int],
) -> Dict[str, Any]:
    """Propagate repeated detector observations into geometry feature units."""
    values = []
    validity = []
    width, height = image_size
    for observation in observations:
        observation_landmarks = []
        for index, landmark in enumerate(landmarks):
            item = dict(landmark)
            x, y = map(float, observation[index])
            item["x"] = x
            item["y"] = y
            item["valid"] = bool(
                math.isfinite(x)
                and math.isfinite(y)
                and 0.0 <= x <= 1.0
                and 0.0 <= y <= 1.0
            )
            observation_landmarks.append(item)
        inputs = anatomy_to_model_inputs(
            {
                "coordinate_space": "normalized",
                "image_size": [width, height],
                "detector": {"detected": True, "confidence": 1.0},
                "landmarks": observation_landmarks,
                "pose": {"quality": 1.0},
                "crop_quality": 1.0,
            },
            original_size=(width, height),
            output_size=(height, width),
        )
        values.append(inputs["geometry_features"].numpy())
        validity.append(inputs["geometry_validity"].numpy())

    value_array = np.stack(values)
    validity_array = np.stack(validity)
    observation_count = validity_array.sum(axis=0)
    feature_std = np.zeros(value_array.shape[1:], dtype=np.float32)
    feature_valid = observation_count >= 2
    for region_index, feature_index in zip(*np.nonzero(feature_valid)):
        selected = value_array[
            validity_array[:, region_index, feature_index],
            region_index,
            feature_index,
        ]
        feature_std[region_index, feature_index] = float(np.std(selected))
    return {
        "geometry_feature_std": feature_std.tolist(),
        "geometry_feature_valid": feature_valid.tolist(),
        "geometry_feature_observation_count": observation_count.tolist(),
    }


def _build_artifact(
    image: Image.Image,
    image_path: Path,
    landmarker,
    mp,
    jitter_repeats: int,
    jitter_pixels: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    image = image.convert("RGB")
    width, height = image.size
    base, matrix = _detect(landmarker, mp, image)
    if not base:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "image_path": str(image_path),
            "image_size": [width, height],
            "coordinate_space": "normalized",
            "detector": {"name": "mediapipe_face_landmarker", "detected": False},
            "landmarks": [],
            "pose": {
                "yaw": None,
                "pitch": None,
                "roll": None,
                "quality": 0.0,
                "valid": False,
            },
            "crop_quality": 0.0,
            "jitter": {"requested_repeats": jitter_repeats, "successful_repeats": 0},
        }

    observations = [
        np.asarray([[_score(point, "x", np.nan), _score(point, "y", np.nan)] for point in base], dtype=np.float32)
    ]
    successful_jitters = 0
    for dx, dy in _jitter_offsets(jitter_repeats, jitter_pixels, rng):
        shifted, _ = _detect(landmarker, mp, _shift_image(image, dx, dy))
        if not shifted or len(shifted) != len(base):
            continue
        mapped = np.asarray(
            [[_score(point, "x", np.nan) - dx / width, _score(point, "y", np.nan) - dy / height] for point in shifted],
            dtype=np.float32,
        )
        observations.append(mapped)
        successful_jitters += 1

    observation_array = np.stack(observations, axis=0)
    coordinate_std = np.nanstd(observation_array, axis=0)
    uncertainty = np.sqrt(np.square(coordinate_std).sum(axis=-1))
    miss_fraction = (jitter_repeats - successful_jitters) / max(1, jitter_repeats)
    uncertainty = uncertainty + 0.05 * miss_fraction

    landmarks = []
    xy = observations[0]
    for index, point in enumerate(base):
        visibility, visibility_valid = _score_with_validity(point, "visibility", 1.0)
        visibility = float(np.clip(visibility, 0.0, 1.0))
        presence, presence_valid = _score_with_validity(point, "presence", visibility)
        presence = float(np.clip(presence, 0.0, 1.0))
        confidence_valid = visibility_valid or presence_valid
        confidence = min(
            visibility if visibility_valid else 1.0,
            presence if presence_valid else 1.0,
        )
        x, y = map(float, xy[index])
        valid = math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        landmarks.append(
            {
                "x": x,
                "y": y,
                "z": _score(point, "z", 0.0),
                "visibility": visibility,
                "visibility_valid": visibility_valid,
                "presence": presence,
                "presence_valid": presence_valid,
                "confidence": confidence,
                "confidence_valid": confidence_valid,
                "uncertainty": float(uncertainty[index]),
                "valid": valid,
            }
        )
    feature_jitter = _feature_jitter_from_observations(
        observations,
        landmarks,
        image_size=(width, height),
    )

    finite = np.isfinite(xy).all(axis=-1)
    in_frame = finite & (xy[:, 0] >= 0.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= 0.0) & (xy[:, 1] <= 1.0)
    valid_rate = float(in_frame.mean())
    valid_xy = xy[in_frame]
    if valid_xy.size:
        edge_margin = float(min(valid_xy[:, 0].min(), 1.0 - valid_xy[:, 0].max(), valid_xy[:, 1].min(), 1.0 - valid_xy[:, 1].max()))
        margin_quality = float(np.clip(edge_margin / 0.03, 0.0, 1.0))
    else:
        margin_quality = 0.0
    crop_quality = valid_rate * (0.5 + 0.5 * margin_quality)

    valid_confidences = [
        item["confidence"] for item in landmarks if item["confidence_valid"]
    ]
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "image_path": str(image_path),
        "image_size": [width, height],
        "coordinate_space": "normalized",
        "detector": {
            "name": "mediapipe_face_landmarker",
            "detected": True,
            "confidence": (
                float(np.mean(valid_confidences)) if valid_confidences else None
            ),
            "confidence_valid": bool(valid_confidences),
            "confidence_source": (
                "mean_available_landmark_visibility_presence"
                if valid_confidences
                else "unavailable"
            ),
            "landmark_count": len(landmarks),
        },
        "landmarks": landmarks,
        "pose": _pose_from_matrix(matrix),
        "crop_quality": float(np.clip(crop_quality, 0.0, 1.0)),
        "jitter": {
            "requested_repeats": jitter_repeats,
            "successful_repeats": successful_jitters,
            "translation_pixels": jitter_pixels,
            "mean_uncertainty": float(np.mean(uncertainty)),
            **feature_jitter,
        },
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_artifact_reference(target: Path, manifest: Path) -> str:
    try:
        return Path(os.path.relpath(target, manifest.parent)).as_posix()
    except ValueError:
        # Windows cannot express a relative path across drive letters.
        return str(target)


def main() -> None:
    args = _parse_args()
    manifest = Path(args.manifest).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    model_path = Path(args.model_path).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    mp, mp_python, vision = _load_mediapipe()
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=float(args.min_detection_confidence),
        min_face_presence_confidence=float(args.min_presence_confidence),
        min_tracking_confidence=float(args.min_tracking_confidence),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
    lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    detected = 0
    rng = np.random.default_rng(args.seed)
    provenance = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "detector": "mediapipe_face_landmarker",
        "model_sha256": _file_sha256(model_path),
        "jitter_repeats": int(args.jitter_repeats),
        "jitter_pixels": int(args.jitter_pixels),
        "seed": int(args.seed),
        "min_detection_confidence": float(args.min_detection_confidence),
        "min_presence_confidence": float(args.min_presence_confidence),
        "min_tracking_confidence": float(args.min_tracking_confidence),
    }
    landmarker = vision.FaceLandmarker.create_from_options(options)
    try:
        with output_manifest.open("w", encoding="utf-8") as output:
            for index, line in enumerate(lines, 1):
                record = json.loads(line)
                image_path = _resolve_image_path(record, manifest, args.root_dir)
                target = _artifact_path(image_path, artifact_root)
                if target.exists() and not args.overwrite:
                    artifact = json.loads(target.read_text(encoding="utf-8"))
                    if artifact.get("provenance") != provenance:
                        raise RuntimeError(
                            f"Stale or incompatible anatomy artifact {target}. "
                            "Regenerate with --overwrite so detector settings and feature jitter match."
                        )
                else:
                    with Image.open(image_path) as image:
                        artifact = _build_artifact(
                            image,
                            image_path,
                            landmarker,
                            mp,
                            args.jitter_repeats,
                            args.jitter_pixels,
                            rng,
                        )
                    artifact["provenance"] = provenance
                    target.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
                detected += int(bool((artifact.get("detector") or {}).get("detected")))
                record["landmark_path"] = _portable_artifact_reference(target, output_manifest)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                if index % 100 == 0 or index == len(lines):
                    print(f"processed={index}/{len(lines)} detected={detected} rate={detected / index:.3f}")
    finally:
        landmarker.close()


if __name__ == "__main__":
    main()
