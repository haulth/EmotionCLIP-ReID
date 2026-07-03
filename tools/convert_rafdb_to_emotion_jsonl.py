import argparse
import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path


RAF_BASIC_LABELS = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happiness",
    5: "sadness",
    6: "anger",
    7: "neutral",
}

EMOTION_TO_ID = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happiness": 3,
    "sadness": 4,
    "surprise": 5,
    "neutral": 6,
}

DEFAULT_LABEL_NAMES = (
    "list_patition_label.txt",
    "list_partition_label.txt",
    "EmoLabel/list_patition_label.txt",
    "EmoLabel/list_partition_label.txt",
    "basic/EmoLabel/list_patition_label.txt",
    "basic/EmoLabel/list_partition_label.txt",
)

IMAGE_DIR_HINTS = (
    "Image/aligned",
    "Image/original",
    "basic/Image/aligned",
    "basic/Image/original",
    "aligned",
    "original",
    "images",
    "",
)


def _extract_archive(archive_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / ".extracted_from"
    if marker.exists() and marker.read_text(encoding="utf-8") == str(archive_path.resolve()):
        return output_dir

    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(output_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            archive.extractall(output_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    marker.write_text(str(archive_path.resolve()), encoding="utf-8")
    return output_dir


def _find_first_existing(root: Path, relative_names) -> Path:
    for relative_name in relative_names:
        candidate = root / relative_name
        if candidate.exists():
            return candidate
    matches = sorted(root.rglob("list_patition_label.txt")) + sorted(root.rglob("list_partition_label.txt"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Could not find RAF-DB label file. Expected one of: " + ", ".join(DEFAULT_LABEL_NAMES)
    )


def _image_candidates(image_name: str):
    path = Path(image_name)
    stem = path.stem
    suffix = path.suffix or ".jpg"
    names = [image_name]
    if not stem.endswith("_aligned"):
        names.append(f"{stem}_aligned{suffix}")
    if path.name != image_name:
        names.append(path.name)
    return names


def _resolve_image(root: Path, image_name: str, preferred_aligned: bool) -> Path:
    candidates = []
    names = _image_candidates(image_name) if preferred_aligned else [image_name] + _image_candidates(image_name)
    for hint in IMAGE_DIR_HINTS:
        base = root / hint
        for name in names:
            candidates.append(base / name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    stem = Path(image_name).stem
    glob_patterns = [f"{stem}_aligned.*", f"{stem}.*"]
    for pattern in glob_patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Could not resolve RAF-DB image for label entry '{image_name}'")


def _official_split_from_name(image_name: str) -> str:
    lowered = Path(image_name).name.lower()
    if lowered.startswith("train"):
        return "train"
    if lowered.startswith("test"):
        return "test"
    raise ValueError(f"RAF-DB image name should start with train/test, got: {image_name}")


def _read_label_file(label_file: Path):
    rows = []
    with label_file.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{label_file}:{line_no} expected '<image_name> <label_id>'")
            image_name = parts[0].replace("\\", "/")
            label_id = int(parts[1])
            if label_id not in RAF_BASIC_LABELS:
                raise ValueError(f"{label_file}:{line_no} has unsupported RAF-DB basic label {label_id}")
            rows.append((image_name, label_id))
    return rows


def convert_rafdb(
    raf_root: Path,
    output: Path,
    root_dir: Path,
    label_file: Path,
    test_split_name: str,
    preferred_aligned: bool,
    allow_missing: bool,
):
    rows = _read_label_file(label_file)
    records = []
    missing = []

    for image_name, label_id in rows:
        official_split = _official_split_from_name(image_name)
        split = "train" if official_split == "train" else test_split_name
        emotion = RAF_BASIC_LABELS[label_id]
        try:
            image_path = _resolve_image(raf_root, image_name, preferred_aligned)
        except FileNotFoundError:
            if allow_missing:
                missing.append(image_name)
                continue
            raise

        records.append(
            {
                "image_path": os.path.relpath(image_path, root_dir).replace("\\", "/"),
                "emotion": emotion,
                "emotion_id": EMOTION_TO_ID[emotion],
                "split": split,
                "source": "RAF-DB",
                "split_protocol": "rafdb_official_train_test_no_random_split",
                "rafdb_label_id": label_id,
                "official_split": official_split,
                "official_image_name": image_name,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = {}
    for record in records:
        key = (record["split"], record["emotion"])
        counts[key] = counts.get(key, 0) + 1

    print(f"Wrote {len(records)} RAF-DB records to {output}")
    print("Split protocol: RAF-DB official train/test split; no random split is created.")
    print(f"ROOT_DIR for training: {root_dir}")
    print(f"Label file: {label_file}")
    if missing:
        print(f"Skipped {len(missing)} missing images because --allow-missing was set")
    for split in ("train", test_split_name):
        split_total = sum(count for (record_split, _), count in counts.items() if record_split == split)
        print(f"{split}: {split_total}")
        for emotion in ("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"):
            print(f"  {emotion}: {counts.get((split, emotion), 0)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert RAF-DB Basic official train/test split to EmotionCLIP JSONL."
    )
    parser.add_argument("--raf-root", help="Path to extracted RAF-DB root.")
    parser.add_argument("--archive", help="Path to official RAF-DB zip/tar archive. Extracted under --extract-dir.")
    parser.add_argument("--extract-dir", default="data/RAF-DB", help="Where --archive should be extracted.")
    parser.add_argument("--label-file", help="Override path to list_patition_label.txt/list_partition_label.txt.")
    parser.add_argument("--output", default="data/RAF-DB/manifest.jsonl")
    parser.add_argument("--root-dir", help="Root used to make image_path relative. Defaults to RAF root.")
    parser.add_argument(
        "--test-split-name",
        default="val",
        choices=["val", "test"],
        help="The repo trains against split='val'; keep 'val' for normal training/eval, use 'test' for export only.",
    )
    parser.add_argument("--original", action="store_true", help="Prefer original images over aligned images.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip label rows whose images are missing.")
    args = parser.parse_args()

    if not args.raf_root and not args.archive:
        raise SystemExit("Provide --raf-root for extracted RAF-DB or --archive for an official RAF-DB archive.")

    if args.archive:
        raf_root = _extract_archive(Path(args.archive), Path(args.extract_dir))
    else:
        raf_root = Path(args.raf_root)
    raf_root = raf_root.resolve()
    root_dir = Path(args.root_dir).resolve() if args.root_dir else raf_root
    label_file = Path(args.label_file).resolve() if args.label_file else _find_first_existing(raf_root, DEFAULT_LABEL_NAMES)

    convert_rafdb(
        raf_root=raf_root,
        output=Path(args.output),
        root_dir=root_dir,
        label_file=label_file,
        test_split_name=args.test_split_name,
        preferred_aligned=not args.original,
        allow_missing=args.allow_missing,
    )


if __name__ == "__main__":
    main()
