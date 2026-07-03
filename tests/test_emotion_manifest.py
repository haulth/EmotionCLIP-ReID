import json

from PIL import Image

from datasets.emotion_manifest import (
    CANONICAL_EMOTIONS,
    EMOTION_TO_ID,
    EmotionManifestDataset,
    FaceSafeTransform,
    load_emotion_manifest,
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
