import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.emotion_manifest import CANONICAL_EMOTIONS, EMOTION_TO_ID


AFFWILD2_EXPR_TO_CANONICAL = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise",
}


def iter_label_file(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for idx, raw_line in enumerate(handle, 1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            if raw_line.lower().startswith("frame"):
                continue
            parts = [part.strip() for part in raw_line.replace(",", " ").split()]
            if len(parts) == 1:
                frame_id = idx
                label = int(float(parts[0]))
            else:
                frame_id = int(float(parts[0]))
                label = int(float(parts[-1]))
            yield frame_id, label


def main():
    parser = argparse.ArgumentParser(description="Convert Aff-Wild2 expression annotations to EmotionCLIP JSONL.")
    parser.add_argument("--annotations-dir", required=True, help="Directory containing official split annotation txt/csv files.")
    parser.add_argument("--frames-root", required=True, help="Root directory containing pre-extracted frames.")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"], help="Split to write into JSONL records.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--frame-pattern",
        default="{video_id}/{frame_id:05d}.jpg",
        help="Relative frame path pattern under frames-root. Uses video_id and integer frame_id.",
    )
    args = parser.parse_args()

    annotations_dir = Path(args.annotations_dir)
    records = []
    for label_file in sorted(annotations_dir.glob("*")):
        if label_file.suffix.lower() not in {".txt", ".csv"}:
            continue
        video_id = label_file.stem
        for frame_id, raw_label in iter_label_file(label_file):
            emotion = AFFWILD2_EXPR_TO_CANONICAL.get(raw_label)
            if emotion is None:
                continue
            rel_path = args.frame_pattern.format(video_id=video_id, frame_id=frame_id)
            records.append(
                {
                    "image_path": rel_path.replace("\\", "/"),
                    "emotion": emotion,
                    "emotion_id": EMOTION_TO_ID[emotion],
                    "split": args.split,
                    "video_id": video_id,
                    "frame_id": str(frame_id),
                    "source": "Aff-Wild2",
                }
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {args.output}")
    print(f"Canonical class order: {', '.join(CANONICAL_EMOTIONS)}")


if __name__ == "__main__":
    main()
