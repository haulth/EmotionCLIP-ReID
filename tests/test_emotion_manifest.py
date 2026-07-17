import json

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

    train_loader, _, val_loader, test_loader, _ = make_emotion_dataloaders(cfg)

    assert train_loader.dataset.samples[0].split == "train"
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
