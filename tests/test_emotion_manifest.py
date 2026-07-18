import json

import torch
from PIL import Image

from tools.convert_fer2013_to_emotion_jsonl import split_from_usage
from tools.convert_rafdb_to_emotion_jsonl import select_stratified_validation_images
from tools.download_hf_emotion_dataset import split_name
from datasets.emotion_manifest import (
    CANONICAL_EMOTIONS,
    EMOTION_TO_ID,
    EmotionManifestDataset,
    FaceSafeTransform,
    load_emotion_manifest,
    make_emotion_dataloaders,
    normalize_emotion,
    validate_split_leakage,
)


def _write_image(path):
    Image.new("RGB", (16, 16), color=(120, 80, 40)).save(path)


def test_normalize_emotion_aliases():
    assert normalize_emotion("happy") == ("happiness", EMOTION_TO_ID["happiness"])
    assert normalize_emotion("fearful") == ("fear", EMOTION_TO_ID["fear"])
    assert normalize_emotion(6) == ("neutral", 6)
    assert CANONICAL_EMOTIONS[3] == "happiness"


def test_manifest_uses_string_label_when_id_mismatches(tmp_path):
    image_path = tmp_path / "train" / "000001.jpg"
    image_path.parent.mkdir()
    _write_image(image_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "image_path": "train/000001.jpg",
                "emotion": "happiness",
                "emotion_id": 0,
                "split": "train",
                "au_text": ["raises the cheeks"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_emotion_manifest(str(manifest), root_dir=str(tmp_path), split="train")
    assert samples[0].emotion == "happiness"
    assert samples[0].emotion_id == EMOTION_TO_ID["happiness"]
    assert samples[0].au_text == ["raises the cheeks"]

    dataset = EmotionManifestDataset(
        str(manifest),
        root_dir=str(tmp_path),
        split="train",
        transform=FaceSafeTransform(size=(16, 16), train=False),
    )
    item = dataset[0]
    assert item["image"].shape == (3, 16, 16)
    assert int(item["label"]) == EMOTION_TO_ID["happiness"]


def test_split_leakage_detects_video_overlap(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    lines = []
    for split, image_name in [("train", "a.jpg"), ("val", "b.jpg")]:
        _write_image(tmp_path / image_name)
        lines.append(
            json.dumps(
                {
                    "image_path": image_name,
                    "emotion": "neutral",
                    "split": split,
                    "video_id": "video-1",
                }
            )
        )
    manifest.write_text("\n".join(lines), encoding="utf-8")
    samples = load_emotion_manifest(str(manifest), root_dir=str(tmp_path))
    leaks = validate_split_leakage(samples)
    assert leaks == {"video_id": ["video-1"]}


def test_dataloaders_keep_validation_and_test_separate(tmp_path):
    records = []
    for split in ("train", "val", "test"):
        image_name = f"{split}.jpg"
        _write_image(tmp_path / image_name)
        records.append({"image_path": image_name, "emotion": "neutral", "split": split})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    cfg = {
        "DATASETS": {
            "MANIFEST": str(manifest),
            "ROOT_DIR": str(tmp_path),
            "STRICT_SPLIT_LEAKAGE": True,
            "REQUIRE_VAL": True,
            "REQUIRE_TEST": True,
        },
        "INPUT": {"SIZE_TRAIN": [16, 16], "SIZE_TEST": [16, 16]},
        "DATALOADER": {"NUM_WORKERS": 0, "PIN_MEMORY": False},
        "SOLVER": {"STAGE1": {"IMS_PER_BATCH": 1}, "STAGE2": {"IMS_PER_BATCH": 1}},
        "TEST": {"IMS_PER_BATCH": 1, "EVALUATE_AFTER_TRAIN": False},
    }

    train_loader, stage1_loader, val_loader, test_loader, _ = make_emotion_dataloaders(cfg)

    assert train_loader.dataset.samples[0].split == "train"
    assert train_loader.dataset is not stage1_loader.dataset
    assert train_loader.dataset.transform.train
    assert not stage1_loader.dataset.transform.train
    assert stage1_loader.dataset.transform.hflip_prob == 0.0
    assert stage1_loader.dataset.transform.color_jitter == 0.0
    assert val_loader.dataset.samples[0].split == "val"
    assert test_loader.dataset.samples[0].split == "test"
    assert val_loader.dataset.samples[0].image_path != test_loader.dataset.samples[0].image_path


def test_fer2013_official_splits_are_not_merged():
    assert split_name("train") == "train"
    assert split_name("publicTest") == "val"
    assert split_name("privateTest") == "test"
    assert split_from_usage("Training") == "train"
    assert split_from_usage("PublicTest") == "val"
    assert split_from_usage("PrivateTest") == "test"


def test_rafdb_dev_split_is_deterministic_stratified_and_train_only():
    rows = []
    for label_id in (1, 2, 3):
        rows.extend((f"train_{label_id}_{index}.jpg", label_id, "train") for index in range(10))
        rows.extend((f"test_{label_id}_{index}.jpg", label_id, "test") for index in range(3))

    first = select_stratified_validation_images(rows, val_ratio=0.2, seed=1234)
    second = select_stratified_validation_images(rows, val_ratio=0.2, seed=1234)

    assert first == second
    assert len(first) == 6
    assert all(name.startswith("train_") for name in first)
    for label_id in (1, 2, 3):
        assert sum(name.startswith(f"train_{label_id}_") for name in first) == 2


def test_rafdb_full_official_train_mode_has_no_validation_selection():
    rows = [("train_0001.jpg", 1, "train"), ("test_0001.jpg", 1, "test")]
    assert select_stratified_validation_images(rows, val_ratio=0.0, seed=1234) == set()


def _face_mesh_artifact():
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
    landmarks[33]["x"] = 0.4
    landmarks[133]["x"] = 0.45
    landmarks[362]["x"] = 0.55
    landmarks[263]["x"] = 0.6
    landmarks[10]["y"] = 0.2
    landmarks[152]["y"] = 0.8
    landmarks[13]["y"] = 0.55
    landmarks[14]["y"] = 0.6
    landmarks[61]["x"] = 0.4
    landmarks[291]["x"] = 0.6
    landmarks[234]["x"] = 0.2
    landmarks[454]["x"] = 0.8
    return {
        "coordinate_space": "normalized",
        "image_size": [200, 100],
        "detector": {"confidence": 1.0, "detected": True},
        "landmarks": landmarks,
        "pose": {"quality": 1.0},
        "crop_quality": 1.0,
    }


def test_landmarks_follow_center_crop_and_horizontal_flip():
    image = Image.new("RGB", (200, 100), color=(120, 80, 40))
    transform = FaceSafeTransform(
        size=(100, 100),
        train=True,
        hflip_prob=1.0,
        color_jitter=0.0,
    )
    tensor, anatomy = transform(image, anatomy=_face_mesh_artifact())

    assert tensor.shape == (3, 100, 100)
    assert anatomy["region_landmarks"].shape == (3, 64, 2)
    assert anatomy["geometry_features"].shape == (3, 12)
    assert bool(anatomy["anatomy_available"])
    # Point 33 is the first upper landmark: center crop maps .4 -> .3, flip maps .3 -> .7.
    torch.testing.assert_close(anatomy["region_landmarks"][0, 0, 0], torch.tensor(0.7))
    assert torch.all(anatomy["region_quality"] > 0)


def test_horizontal_flip_swaps_left_right_geometry_semantics():
    artifact = _face_mesh_artifact()
    landmarks = artifact["landmarks"]
    landmarks[160]["y"], landmarks[144]["y"] = 0.44, 0.56
    landmarks[158]["y"], landmarks[153]["y"] = 0.45, 0.55
    landmarks[385]["y"], landmarks[380]["y"] = 0.49, 0.51
    landmarks[387]["y"], landmarks[373]["y"] = 0.49, 0.51
    feature_std = [[0.0] * 12 for _ in range(3)]
    feature_valid = [[False] * 12 for _ in range(3)]
    feature_std[0][0], feature_std[0][1] = 0.1, 0.2
    feature_valid[0][0] = feature_valid[0][1] = True
    artifact["jitter"] = {
        "geometry_feature_std": feature_std,
        "geometry_feature_valid": feature_valid,
    }
    image = Image.new("RGB", (200, 100), color=(120, 80, 40))
    _, normal = FaceSafeTransform(
        size=(100, 100),
        train=False,
        color_jitter=0.0,
    )(image, anatomy=artifact)
    _, flipped = FaceSafeTransform(
        size=(100, 100),
        train=True,
        hflip_prob=1.0,
        color_jitter=0.0,
    )(image, anatomy=artifact)

    assert normal["geometry_features"][0, 0] > normal["geometry_features"][0, 1]
    torch.testing.assert_close(
        flipped["geometry_features"][0, :2],
        normal["geometry_features"][0, [1, 0]],
    )
    torch.testing.assert_close(
        flipped["geometry_validity"][0, :2],
        normal["geometry_validity"][0, [1, 0]],
    )
    torch.testing.assert_close(
        flipped["geometry_uncertainty"][0, :2],
        normal["geometry_uncertainty"][0, [1, 0]],
    )


def test_normalized_geometry_is_invalid_when_scale_landmarks_are_missing():
    artifact = _face_mesh_artifact()
    artifact["landmarks"][10]["valid"] = False
    image = Image.new("RGB", (200, 100), color=(120, 80, 40))
    _, anatomy = FaceSafeTransform(size=(100, 100), train=False)(
        image,
        anatomy=artifact,
    )

    # Brow-eye distances and their asymmetry all divide by face height.
    assert not anatomy["geometry_validity"][0, 2]
    assert not anatomy["geometry_validity"][0, 3]
    assert not anatomy["geometry_validity"][0, 7]


def test_dataset_missing_anatomy_uses_explicit_zero_quality_fallback(tmp_path):
    image_path = tmp_path / "sample.jpg"
    _write_image(image_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"image_path": str(image_path), "emotion": "neutral", "split": "train"}) + "\n",
        encoding="utf-8",
    )
    dataset = EmotionManifestDataset(
        str(manifest),
        split="train",
        transform=FaceSafeTransform(size=(16, 16), train=False),
    )
    anatomy = dataset[0]["anatomy"]
    assert not bool(anatomy["anatomy_available"])
    torch.testing.assert_close(anatomy["region_quality"], torch.zeros(3))
    assert not anatomy["geometry_validity"].any()
