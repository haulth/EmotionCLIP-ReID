import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset


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
ID_TO_EMOTION = {idx: name for name, idx in EMOTION_TO_ID.items()}

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


@dataclass(frozen=True)
class EmotionSample:
    image_path: str
    emotion: str
    emotion_id: int
    split: str
    video_id: Optional[str] = None
    subject_id: Optional[str] = None
    frame_id: Optional[str] = None
    au_labels: Optional[Dict[str, Any]] = None
    au_text: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


def normalize_emotion(value: Any) -> Tuple[str, int]:
    if isinstance(value, int):
        if value not in ID_TO_EMOTION:
            raise ValueError(f"Unknown emotion id {value}; expected 0-{len(CANONICAL_EMOTIONS) - 1}")
        name = ID_TO_EMOTION[value]
        return name, value

    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text not in EMOTION_ALIASES:
        raise ValueError(f"Unknown emotion '{value}'. Expected one of: {', '.join(CANONICAL_EMOTIONS)}")
    name = EMOTION_ALIASES[text]
    return name, EMOTION_TO_ID[name]


def _resolve_image_path(path: str, root_dir: Optional[str]) -> str:
    if os.path.isabs(path):
        return path
    if root_dir:
        return os.path.normpath(os.path.join(root_dir, path))
    return os.path.normpath(path)


def _sample_from_record(
    record: Dict[str, Any],
    manifest_path: str,
    root_dir: Optional[str],
    line_no: int,
    logger: logging.Logger,
) -> EmotionSample:
    if "image_path" not in record:
        raise ValueError(f"{manifest_path}:{line_no} missing required field 'image_path'")
    if "emotion" not in record and "emotion_id" not in record:
        raise ValueError(f"{manifest_path}:{line_no} missing required field 'emotion' or 'emotion_id'")

    if "emotion" in record:
        emotion, emotion_id = normalize_emotion(record["emotion"])
        provided_id = record.get("emotion_id")
        if provided_id is not None and int(provided_id) != emotion_id:
            logger.warning(
                "%s:%s has emotion_id=%s but emotion='%s' maps to %s; using canonical string mapping",
                manifest_path,
                line_no,
                provided_id,
                record["emotion"],
                emotion_id,
            )
    else:
        emotion, emotion_id = normalize_emotion(int(record["emotion_id"]))

    split = str(record.get("split", "train")).strip().lower()
    if split not in {"train", "val", "valid", "validation", "test"}:
        raise ValueError(f"{manifest_path}:{line_no} has unsupported split '{split}'")
    if split in {"valid", "validation"}:
        split = "val"

    metadata = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "image_path",
            "emotion",
            "emotion_id",
            "split",
            "video_id",
            "subject_id",
            "frame_id",
            "au_labels",
            "au_text",
        }
    }

    return EmotionSample(
        image_path=_resolve_image_path(str(record["image_path"]), root_dir),
        emotion=emotion,
        emotion_id=emotion_id,
        split=split,
        video_id=None if record.get("video_id") is None else str(record.get("video_id")),
        subject_id=None if record.get("subject_id") is None else str(record.get("subject_id")),
        frame_id=None if record.get("frame_id") is None else str(record.get("frame_id")),
        au_labels=record.get("au_labels"),
        au_text=record.get("au_text"),
        metadata=metadata,
    )


def load_emotion_manifest(
    manifest_path: str,
    root_dir: Optional[str] = None,
    split: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> List[EmotionSample]:
    logger = logger or logging.getLogger("emotionclip.data")
    samples: List[EmotionSample] = []
    split = split.lower() if split else None
    if split in {"valid", "validation"}:
        split = "val"

    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            sample = _sample_from_record(record, manifest_path, root_dir, line_no, logger)
            if split is None or sample.split == split:
                samples.append(sample)

    if not samples:
        raise ValueError(f"No samples found in {manifest_path} for split={split!r}")
    return samples


def validate_split_leakage(
    samples: Sequence[EmotionSample],
    group_keys: Sequence[str] = ("image_path", "video_id", "subject_id"),
) -> Dict[str, List[str]]:
    leaks: Dict[str, List[str]] = {}
    for key in group_keys:
        groups: Dict[str, set] = {}
        for sample in samples:
            value = getattr(sample, key)
            if value:
                groups.setdefault(value, set()).add(sample.split)
        leaked = sorted(group for group, splits in groups.items() if len(splits) > 1)
        if leaked:
            leaks[key] = leaked
    return leaks


class FaceSafeTransform:
    def __init__(
        self,
        size: Tuple[int, int] = (224, 224),
        train: bool = False,
        hflip_prob: float = 0.5,
        color_jitter: float = 0.05,
        mean: Sequence[float] = (0.48145466, 0.4578275, 0.40821073),
        std: Sequence[float] = (0.26862954, 0.26130258, 0.27577711),
    ):
        self.size = tuple(size)
        self.train = train
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def _jitter(self, image: Image.Image) -> Image.Image:
        if self.color_jitter <= 0:
            return image
        amount = float(np.random.uniform(1.0 - self.color_jitter, 1.0 + self.color_jitter))
        image = ImageEnhance.Brightness(image).enhance(amount)
        amount = float(np.random.uniform(1.0 - self.color_jitter, 1.0 + self.color_jitter))
        image = ImageEnhance.Contrast(image).enhance(amount)
        return image

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        image = ImageOps.fit(image, self.size[::-1], method=Image.BICUBIC, centering=(0.5, 0.5))
        if self.train and self.hflip_prob > 0 and float(np.random.rand()) < self.hflip_prob:
            image = ImageOps.mirror(image)
        if self.train:
            image = self._jitter(image)

        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self.mean) / self.std


class EmotionManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        root_dir: Optional[str] = None,
        split: str = "train",
        transform: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
        samples: Optional[Sequence[EmotionSample]] = None,
    ):
        self.samples = (
            list(samples)
            if samples is not None
            else load_emotion_manifest(manifest_path, root_dir=root_dir, split=split, logger=logger)
        )
        if not self.samples:
            raise ValueError(f"No samples supplied for split={split!r}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image) if self.transform is not None else FaceSafeTransform()(image)

        return {
            "image": tensor,
            "label": torch.tensor(sample.emotion_id, dtype=torch.long),
            "emotion": sample.emotion,
            "image_path": sample.image_path,
            "video_id": sample.video_id,
            "subject_id": sample.subject_id,
            "frame_id": sample.frame_id,
            "au_labels": sample.au_labels,
            "au_text": sample.au_text,
            "metadata": sample.metadata or {},
        }


def emotion_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "labels": torch.stack([item["label"] for item in batch], dim=0),
        "emotions": [item["emotion"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "metadata": [
            {
                "video_id": item["video_id"],
                "subject_id": item["subject_id"],
                "frame_id": item["frame_id"],
                "au_labels": item["au_labels"],
                "au_text": item["au_text"],
                **item["metadata"],
            }
            for item in batch
        ],
    }


def make_emotion_dataloaders(cfg: Any):
    data_cfg = cfg["DATASETS"]
    input_cfg = cfg["INPUT"]
    loader_cfg = cfg["DATALOADER"]
    solver_cfg = cfg["SOLVER"]
    test_cfg = cfg["TEST"]

    manifest = data_cfg["MANIFEST"]
    root_dir = data_cfg.get("ROOT_DIR") or None
    image_size = tuple(input_cfg.get("SIZE_TRAIN", [224, 224]))
    mean = input_cfg.get("PIXEL_MEAN", [0.48145466, 0.4578275, 0.40821073])
    std = input_cfg.get("PIXEL_STD", [0.26862954, 0.26130258, 0.27577711])

    train_transform = FaceSafeTransform(
        size=image_size,
        train=True,
        hflip_prob=float(input_cfg.get("PROB", 0.5)),
        color_jitter=float(input_cfg.get("COLOR_JITTER", 0.05)),
        mean=mean,
        std=std,
    )
    val_transform = FaceSafeTransform(
        size=tuple(input_cfg.get("SIZE_TEST", image_size)),
        train=False,
        hflip_prob=0.0,
        color_jitter=0.0,
        mean=mean,
        std=std,
    )

    all_samples = load_emotion_manifest(manifest, root_dir=root_dir, split=None)
    leaks = validate_split_leakage(all_samples)
    if leaks and data_cfg.get("STRICT_SPLIT_LEAKAGE", True):
        summary = {key: values[:10] for key, values in leaks.items()}
        raise ValueError(f"Detected group leakage across splits: {summary}")

    samples_by_split = {
        split: [sample for sample in all_samples if sample.split == split]
        for split in ("train", "val", "test")
    }
    if not samples_by_split["train"]:
        raise ValueError(f"Manifest {manifest!r} has no train samples")
    if data_cfg.get("REQUIRE_VAL", True) and not samples_by_split["val"]:
        raise ValueError(
            f"Manifest {manifest!r} has no validation samples. Use a development manifest, "
            "or set DATASETS.REQUIRE_VAL false only for a locked fixed-epoch final run."
        )
    require_test = bool(data_cfg.get("REQUIRE_TEST", False) or test_cfg.get("EVALUATE_AFTER_TRAIN", False))
    if require_test and not samples_by_split["test"]:
        raise ValueError(f"Manifest {manifest!r} has no sealed test samples")

    train_set = EmotionManifestDataset(
        manifest,
        split="train",
        transform=train_transform,
        samples=samples_by_split["train"],
    )
    val_set = (
        EmotionManifestDataset(manifest, split="val", transform=val_transform, samples=samples_by_split["val"])
        if samples_by_split["val"]
        else None
    )
    test_set = (
        EmotionManifestDataset(manifest, split="test", transform=val_transform, samples=samples_by_split["test"])
        if samples_by_split["test"]
        else None
    )

    train_loader = DataLoader(
        train_set,
        batch_size=int(solver_cfg["STAGE2"].get("IMS_PER_BATCH", 32)),
        shuffle=True,
        num_workers=int(loader_cfg.get("NUM_WORKERS", 0)),
        collate_fn=emotion_collate_fn,
        pin_memory=bool(loader_cfg.get("PIN_MEMORY", False)),
    )
    stage1_loader = DataLoader(
        train_set,
        batch_size=int(solver_cfg["STAGE1"].get("IMS_PER_BATCH", 64)),
        shuffle=False,
        num_workers=int(loader_cfg.get("NUM_WORKERS", 0)),
        collate_fn=emotion_collate_fn,
        pin_memory=bool(loader_cfg.get("PIN_MEMORY", False)),
    )
    def make_eval_loader(dataset):
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=int(test_cfg.get("IMS_PER_BATCH", 64)),
            shuffle=False,
            num_workers=int(loader_cfg.get("NUM_WORKERS", 0)),
            collate_fn=emotion_collate_fn,
            pin_memory=bool(loader_cfg.get("PIN_MEMORY", False)),
        )

    val_loader = make_eval_loader(val_set)
    test_loader = make_eval_loader(test_set)
    return train_loader, stage1_loader, val_loader, test_loader, CANONICAL_EMOTIONS
