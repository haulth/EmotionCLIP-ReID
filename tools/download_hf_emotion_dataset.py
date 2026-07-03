import argparse
import json
import shutil
from pathlib import Path


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

LABEL_TO_EMOTION = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise",
    6: "neutral",
}

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


def normalize_emotion(value, label_feature=None):
    if label_feature is not None and hasattr(label_feature, "int2str"):
        try:
            value = label_feature.int2str(int(value))
        except (TypeError, ValueError):
            pass
    if isinstance(value, int):
        if value not in LABEL_TO_EMOTION:
            raise ValueError(f"Unknown emotion id {value}; expected 0-6")
        return LABEL_TO_EMOTION[value]

    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text not in EMOTION_ALIASES:
        raise ValueError(f"Unknown emotion label {value!r}")
    return EMOTION_ALIASES[text]


def split_name(source_split, test_as):
    source_split = str(source_split).lower()
    if source_split in {"train", "training"}:
        return "train"
    if source_split in {"validation", "valid", "val"}:
        return "val"
    if source_split == "publictest":
        return "val"
    if source_split in {"test", "privatetest"}:
        return test_as
    return source_split


def save_image(image_obj, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image_obj, dict) and image_obj.get("path"):
        shutil.copy2(image_obj["path"], output_path)
        return
    image = image_obj.convert("RGB")
    image.save(output_path)


def iter_limited(dataset_split, max_samples):
    if max_samples is None or max_samples < 0:
        yield from enumerate(dataset_split)
        return
    for idx, sample in enumerate(dataset_split):
        if idx >= max_samples:
            break
        yield idx, sample


def main():
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face image emotion dataset and export EmotionCLIP manifest/images."
    )
    parser.add_argument("--dataset", default="Aaryan333/fer2013_train_publicTest_privateTest")
    parser.add_argument("--output-root", type=Path, default=Path("data/hf_fer2013"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--test-as", default="val", choices=["val", "test"])
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=-1,
        help="Use a small value such as 200 for a quick Jupyter smoke run. -1 exports all samples.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable Hugging Face streaming. By default samples are streamed so smoke runs do not download all files.",
    )
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the Hugging Face datasets package first: pip install datasets pillow") from exc

    output_root = args.output_root
    images_root = output_root / "images"
    manifest_path = args.manifest or output_root / "manifest.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, streaming=not args.no_streaming)
    records = []
    counts = {}
    for hf_split, split_data in dataset.items():
        split = split_name(hf_split, args.test_as)
        if args.shuffle_buffer > 0:
            try:
                split_data = split_data.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
            except TypeError:
                split_data = split_data.shuffle(seed=args.seed)
        label_feature = getattr(split_data, "features", {}).get(args.label_column)
        for row_idx, sample in iter_limited(split_data, args.max_samples_per_split):
            emotion = normalize_emotion(sample[args.label_column], label_feature=label_feature)
            rel_path = Path(split) / emotion / f"{hf_split}_{row_idx:06d}.jpg"
            save_image(sample[args.image_column], images_root / rel_path)
            record = {
                "image_path": rel_path.as_posix(),
                "emotion": emotion,
                "emotion_id": EMOTION_TO_ID[emotion],
                "split": split,
                "source": args.dataset,
                "hf_split": hf_split,
                "hf_index": row_idx,
            }
            records.append(record)
            counts[split] = counts.get(split, 0) + 1

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {manifest_path}")
    print(f"Images root: {images_root}")
    print(f"Split counts: {counts}")


if __name__ == "__main__":
    main()
