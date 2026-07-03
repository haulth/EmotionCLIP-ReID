import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.emotion_manifest import EMOTION_TO_ID, normalize_emotion


def parse_au_fields(row):
    au_labels = {}
    for key, value in row.items():
        if key and key.upper().startswith("AU") and value not in {None, ""}:
            try:
                au_labels[key.upper()] = int(float(value))
            except ValueError:
                au_labels[key.upper()] = value
    return au_labels or None


def main():
    parser = argparse.ArgumentParser(description="Convert RAF-AU style CSV annotations to EmotionCLIP JSONL.")
    parser.add_argument("--csv", required=True, help="CSV with image_path, emotion or emotion_id, and split columns.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--default-split", default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    records = []
    with open(args.csv, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_no, row in enumerate(reader, 2):
            image_path = row.get("image_path") or row.get("path") or row.get("file") or row.get("filename")
            if not image_path:
                raise ValueError(f"CSV row {row_no} missing image path column")

            if row.get("emotion"):
                emotion, emotion_id = normalize_emotion(row["emotion"])
            elif row.get("emotion_id"):
                emotion, emotion_id = normalize_emotion(int(float(row["emotion_id"])))
            else:
                raise ValueError(f"CSV row {row_no} missing emotion or emotion_id")

            record = {
                "image_path": image_path.replace("\\", "/"),
                "emotion": emotion,
                "emotion_id": EMOTION_TO_ID[emotion],
                "split": (row.get("split") or args.default_split).lower(),
                "source": "RAF-AU",
            }
            for key in ("subject_id", "video_id", "frame_id"):
                if row.get(key):
                    record[key] = row[key]
            au_labels = parse_au_fields(row)
            if au_labels:
                record["au_labels"] = au_labels
            records.append(record)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
