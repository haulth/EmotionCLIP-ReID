import numpy as np

from datasets.anatomy import MAX_GEOMETRY_FEATURES, NUM_ANATOMY_REGIONS
from tools.audit_anatomy_geometry import _signal_to_jitter
from tools.build_face_landmark_artifacts import _feature_jitter_from_observations


def _landmarks():
    landmarks = [
        {
            "x": 0.5,
            "y": 0.5,
            "z": 0.0,
            "visibility": 1.0,
            "confidence": 1.0,
            "uncertainty": 0.0,
            "valid": True,
        }
        for _ in range(478)
    ]
    landmarks[10]["y"], landmarks[152]["y"] = 0.2, 0.8
    landmarks[234]["x"], landmarks[454]["x"] = 0.2, 0.8
    landmarks[33]["x"], landmarks[133]["x"] = 0.4, 0.45
    landmarks[362]["x"], landmarks[263]["x"] = 0.55, 0.6
    landmarks[61]["x"], landmarks[291]["x"] = 0.4, 0.6
    landmarks[13]["y"], landmarks[14]["y"] = 0.48, 0.52
    return landmarks


def test_artifact_builder_propagates_jitter_into_feature_units():
    landmarks = _landmarks()
    base = np.asarray([[item["x"], item["y"]] for item in landmarks], dtype=np.float32)
    shifted = base.copy()
    shifted[14, 1] += 0.02
    result = _feature_jitter_from_observations(
        [base, shifted],
        landmarks,
        image_size=(100, 100),
    )
    feature_std = np.asarray(result["geometry_feature_std"])
    feature_valid = np.asarray(result["geometry_feature_valid"])

    assert feature_std.shape == (NUM_ANATOMY_REGIONS, MAX_GEOMETRY_FEATURES)
    assert feature_valid[2, 0]
    assert feature_std[2, 0] > 0.0


def test_signal_to_jitter_compares_quantities_in_geometry_feature_units():
    values = np.zeros((4, NUM_ANATOMY_REGIONS, MAX_GEOMETRY_FEATURES), dtype=np.float32)
    validity = np.zeros_like(values, dtype=np.bool_)
    values[:, 0, 0] = [0.0, 0.0, 2.0, 2.0]
    validity[:, 0, 0] = True
    feature_jitter = np.zeros_like(values)
    feature_jitter[:, 0, 0] = 0.1
    feature_jitter_validity = np.zeros_like(validity)
    feature_jitter_validity[:, 0, 0] = True

    report = _signal_to_jitter(
        values,
        validity,
        ["a", "a", "b", "b"],
        feature_jitter,
        feature_jitter_validity,
    )
    left_ear = report["upper"]["left_ear"]
    assert left_ear["units"] == "geometry_feature_units"
    assert np.isclose(left_ear["robust_between_class_median_mad"], 1.4826)
    assert np.isclose(left_ear["median_feature_jitter_std"], 0.1)
    assert np.isclose(left_ear["signal_to_jitter"], 14.826)
