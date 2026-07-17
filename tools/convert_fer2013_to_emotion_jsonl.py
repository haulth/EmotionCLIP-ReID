import argparse
import csv
import json
import shutil
from pathlib import Path


FER2013_ID_TO_EMOTION = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise",
    6: "neutral",
}

CANONICAL_EMOTIONS = (
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
    "neutral",
)

EMOTION_TO_ID = {name: idx for idx, name in enumerate(CANONICAL_EMOTIONS)}

EMOTION_ALIASES = {
    "angry": "anger",
    "anger": "anger",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "afraid": "fear",
    "happy": "happiness",
    "happiness": "happiness",
    "joy": "happiness",
    "sad": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "surprised": "surprise",
    "neutral": "neutral",
}

USAGE_TO_SPLIT = {
    "training": "train",
    "train": "train",
    "publictest": "val",
    "validation": "val",
    "valid": "val",
    "val": "val",
    "privatetest": "test",
    "test": "test",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def row_value(row, *names):
    lowered = {key.lower(): value for key, value in row.items() if key is not None}
    for name in names:
        value = lowered.get(name.lower())
        if value not in {None, ""}:
            return value
    return None


def split_from_usage(value, default="train"):
    if value is None:
        return default
    compact = str(value).strip().replace("_", "").replace(" ", "").lower()
    if compact not in USAGE_TO_SPLIT:
        raise ValueError(f"Unsupported FER2013 Usage/split value: {value!r}")
    return USAGE_TO_SPLIT[compact]


def emotion_from_label(value):
    if value is None:
        raise ValueError("Missing emotion label")
    text = str(value).strip()
    if text.isdigit():
        label = int(text)
        if label not in FER2013_ID_TO_EMOTION:
            raise ValueError(f"Unknown FER2013 emotion id {label}; expected 0-6")
        emotion = FER2013_ID_TO_EMOTION[label]
    else:
        normalized = text.lower().strip().replace("-", "_").replace(" ", "_")
        if normalized not in EMOTION_ALIASES:
            raise ValueError(f"Unknown emotion label {value!r}")
        emotion = EMOTION_ALIASES[normalized]
    return emotion


def save_pixel_image(pixel_text, output_path):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to convert FER2013 CSV pixels into images") from exc

    pixels = [int(value) for value in str(pixel_text).split()]
    side = int(len(pixels) ** 0.5)
    if side * side != len(pixels):
        raise ValueError(f"Expected square FER2013 pixel vector, got {len(pixels)} values")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (side, side))
    image.putdata(pixels)
    image.convert("RGB").save(output_path)


def convert_csv(csv_path, images_root, output_path, default_split):
    records = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            emotion = emotion_from_label(row_value(row, "emotion", "label", "target"))
            split = split_from_usage(row_value(row, "Usage", "usage", "split"), default=default_split)
            rel_path = Path(split) / f"{index:06d}.png"
            save_pixel_image(row_value(row, "pixels", "pixel"), images_root / rel_path)
            records.append(
                {
                    "image_path": rel_path.as_posix(),
                    "emotion": emotion,
                    "emotion_id": EMOTION_TO_ID[emotion],
                    "split": split,
                    "source": "FER2013",
                }
            )
    write_records(records, output_path)
    return records


def iter_split_dirs(image_dir, directory_test_split):
    if directory_test_split != "test":
        raise ValueError("FER2013 test directories must remain sealed as split='test'")
    split_names = {"train", "training", "val", "valid", "validation", "test"}
    for split_dir in sorted(path for path in image_dir.iterdir() if path.is_dir()):
        if split_dir.name.lower() not in split_names:
            continue
        source_split = split_from_usage(split_dir.name)
        target_split = directory_test_split if source_split == "test" else source_split
        yield split_dir, target_split


def convert_image_dir(image_dir, output_path, copy_images_root, directory_test_split):
    records = []
    for split_dir, split in iter_split_dirs(image_dir, directory_test_split):
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            emotion = emotion_from_label(class_dir.name)
            for source_path in sorted(class_dir.rglob("*")):
                if not source_path.is_file() or source_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                rel_source = source_path.relative_to(image_dir)
                rel_path = rel_source.as_posix()
                if copy_images_root is not None:
                    rel_path = (Path(split) / emotion / source_path.name).as_posix()
                    target_path = copy_images_root / rel_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, target_path)
                records.append(
                    {
                        "image_path": rel_path,
                        "emotion": emotion,
                        "emotion_id": EMOTION_TO_ID[emotion],
                        "split": split,
                        "source": "FER2013",
                    }
                )
    write_records(records, output_path)
    return records


def write_records(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FER2013 CSV or class-folder data to the EmotionCLIP JSONL manifest."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="Path to fer2013.csv or icml_face_data.csv.")
    source.add_argument(
        "--image-dir",
        type=Path,
        help="Directory with split/class folders, for example train/angry/*.jpg and test/happy/*.jpg.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("data/fer2013/images"),
        help="Where CSV pixel images are written. Use this as DATASETS.ROOT_DIR.",
    )
    parser.add_argument("--default-split", default="train", choices=["train", "val", "test"])
    parser.add_argument(
        "--directory-test-split",
        default="test",
        choices=["test"],
        help="Keep a class-folder test directory sealed as split='test'.",
    )
    parser.add_argument(
        "--copy-images-root",
        type=Path,
        default=None,
        help="Optional target root for class-folder images. Use as DATASETS.ROOT_DIR if set.",
    )
    args = parser.parse_args()

    if args.csv:
        records = convert_csv(args.csv, args.images_root, args.output, args.default_split)
        root_dir = args.images_root
    else:
        records = convert_image_dir(args.image_dir, args.output, args.copy_images_root, args.directory_test_split)
        root_dir = args.copy_images_root or args.image_dir

    split_counts = {}
    for record in records:
        split_counts[record["split"]] = split_counts.get(record["split"], 0) + 1
    print(f"Wrote {len(records)} records to {args.output}")
    print(f"Split counts: {split_counts}")
    print(f"Use DATASETS.ROOT_DIR {root_dir}")


if __name__ == "__main__":
    main()
