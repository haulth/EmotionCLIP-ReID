import argparse
import csv
import json
import logging
import os
import random
from datetime import datetime
from typing import Any, Iterable

import numpy as np
import torch

from config.emotion_defaults import load_emotion_cfg
from datasets.emotion_manifest import make_emotion_dataloaders
from model.emotionclip_model import EmotionCLIPModel
from processor.processor_emotionclip import (
    do_train_emotion_stage1,
    do_train_emotion_stage2,
    load_emotion_checkpoint,
    log_run_config,
    log_training_event,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "train.log"), encoding="utf-8"),
        ],
    )


def _flatten_config(prefix: str, value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_config(child_prefix, child)
    else:
        yield prefix, value


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def save_train_config_csv(cfg, config_file: str = "", opts: Iterable[str] = ()):
    os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)
    saved_at = datetime.now().astimezone()
    suffix = saved_at.strftime("%Y%m%d_%H%M%S")
    cfg.setdefault("TRAIN", {})
    cfg["TRAIN"]["RUN_ID"] = suffix
    cfg["TRAIN"]["RUN_SAVED_AT"] = saved_at.isoformat(timespec="seconds")
    cfg["TRAIN"]["RUN_HISTORY_CSV"] = os.path.join(cfg["OUTPUT_DIR"], f"train_history_{suffix}.csv")
    cfg["TRAIN"]["TRAINING_EPOCH_CSV"] = os.path.join(cfg["OUTPUT_DIR"], "training_epoch_losses.csv")
    cfg["TRAIN"]["VALIDATION_CSV"] = os.path.join(cfg["OUTPUT_DIR"], "validation_metrics.csv")
    for csv_path in [
        cfg["TRAIN"]["RUN_HISTORY_CSV"],
        cfg["TRAIN"]["TRAINING_EPOCH_CSV"],
        cfg["TRAIN"]["VALIDATION_CSV"],
    ]:
        if os.path.exists(csv_path):
            os.remove(csv_path)
    path = os.path.join(cfg["OUTPUT_DIR"], f"train_config_{suffix}.csv")
    cfg["TRAIN"]["CONFIG_CSV"] = path
    rows = [
        ("RUN.saved_at", saved_at.isoformat(timespec="seconds")),
        ("RUN.config_file", config_file),
        ("RUN.opts", list(opts or [])),
        ("RUN.history_csv", cfg["TRAIN"]["RUN_HISTORY_CSV"]),
        ("RUN.training_epoch_csv", cfg["TRAIN"]["TRAINING_EPOCH_CSV"]),
        ("RUN.validation_csv", cfg["TRAIN"]["VALIDATION_CSV"]),
    ]
    rows.extend(_flatten_config("", cfg))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "value", "type"])
        for key, value in rows:
            writer.writerow([key, _csv_value(value), type(value).__name__])
    return path


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def configure_device(cfg, gpu_id=None):
    requested_device = str(cfg["MODEL"].get("DEVICE", "cpu")).strip()
    if gpu_id is not None:
        if gpu_id < 0:
            raise ValueError(f"CUDA device index must be >= 0, got {gpu_id}")
        requested_device = f"cuda:{gpu_id}"
        cfg["MODEL"]["DEVICE"] = requested_device

    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            cfg["MODEL"]["DEVICE"] = "cpu"
            return torch.device("cpu"), "CUDA requested but current PyTorch is CPU-only; falling back to CPU"

        device = torch.device(requested_device)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device.index} was requested, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible"
            )

        if device.index is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            cfg["MODEL"]["DEVICE"] = str(device)
        torch.cuda.set_device(device.index)
        return device, None

    device = torch.device(requested_device)
    cfg["MODEL"]["DEVICE"] = str(device)
    return device, None


def main():
    parser = argparse.ArgumentParser(description="EmotionCLIP-ReID FER training")
    parser.add_argument("--config_file", default="configs/emotion/vit_b16_emotionclip.yml", type=str)
    parser.add_argument("--gpu", type=int, default=None, help="CUDA GPU index to use, for example --gpu 1")
    parser.add_argument(
        "--no_progress",
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars in console output",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_emotion_cfg(args.config_file, args.opts)
    if args.no_progress:
        cfg["TRAIN"]["PROGRESS_BAR"] = False
    setup_logging(cfg["OUTPUT_DIR"])
    logger = logging.getLogger("emotionclip.train")

    device, device_warning = configure_device(cfg, gpu_id=args.gpu)
    if device_warning:
        logger.warning(device_warning)
    if device.type == "cuda":
        log_training_event(
            logger,
            "CUDA device selected",
            device=device,
            name=torch.cuda.get_device_name(device),
            visible_devices=torch.cuda.device_count(),
        )

    train_config_csv = save_train_config_csv(cfg, config_file=args.config_file, opts=args.opts)
    log_training_event(logger, "Train config saved", path=train_config_csv)
    log_run_config(logger, cfg, config_file=args.config_file, opts=args.opts)
    set_seed(int(cfg["SOLVER"].get("SEED", 1234)))
    log_training_event(logger, "Seed set", seed=cfg["SOLVER"].get("SEED", 1234))

    log_training_event(logger, "Building dataloaders")
    train_loader, train_loader_stage1, val_loader, class_names = make_emotion_dataloaders(cfg)
    log_training_event(
        logger,
        "Dataloaders ready",
        train_samples=len(train_loader.dataset) if hasattr(train_loader, "dataset") else "unknown",
        val_samples=len(val_loader.dataset) if hasattr(val_loader, "dataset") else "unknown",
        stage1_batches=len(train_loader_stage1),
        stage2_batches=len(train_loader),
        val_batches=len(val_loader),
        classes=",".join(class_names),
    )
    model_cfg = cfg["MODEL"]["EMOTION"]
    log_training_event(logger, "Building model", model=cfg["MODEL"]["NAME"])
    model = EmotionCLIPModel(
        class_names=class_names,
        backbone_name=cfg["MODEL"]["NAME"],
        image_size=cfg["INPUT"]["SIZE_TRAIN"],
        stride_size=cfg["MODEL"]["STRIDE_SIZE"],
        n_ctx=int(model_cfg["N_CTX"]),
        adapter_dim=int(model_cfg["ADAPTER_DIM"]),
        adapter_dropout=float(model_cfg["ADAPTER_DROPOUT"]),
        topk_patches=int(model_cfg["TOPK_PATCHES"]),
        global_weight=float(model_cfg["GLOBAL_WEIGHT"]),
        local_weight=float(model_cfg["LOCAL_WEIGHT"]),
        classifier_weight=float(model_cfg["CLASSIFIER_WEIGHT"]),
        train_last_blocks=int(model_cfg["TRAIN_LAST_BLOCKS"]),
    )

    model.to(device)
    log_training_event(
        logger,
        "Model ready",
        device=device,
        adapter_count=getattr(model, "adapter_count", "unknown"),
        num_classes=len(class_names),
    )

    stage1_weight = model_cfg.get("STAGE1_WEIGHT") or ""
    if stage1_weight:
        logger.info("Loading Stage 1 prompt checkpoint: %s", stage1_weight)
        load_emotion_checkpoint(model, stage1_weight, strict=False)

    if cfg["TRAIN"].get("RUN_STAGE1", True):
        model.set_train_stage(1)
        optimizer = torch.optim.AdamW(
            trainable_parameters(model),
            lr=float(cfg["SOLVER"]["STAGE1"]["BASE_LR"]),
            weight_decay=float(cfg["SOLVER"]["STAGE1"]["WEIGHT_DECAY"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(cfg["SOLVER"]["STAGE1"]["MAX_EPOCHS"]))
        )
        do_train_emotion_stage1(cfg, model, train_loader_stage1, optimizer, scheduler)

    if cfg["TRAIN"].get("RUN_STAGE2", True):
        model.set_train_stage(2)
        optimizer = torch.optim.AdamW(
            trainable_parameters(model),
            lr=float(cfg["SOLVER"]["STAGE2"]["BASE_LR"]),
            weight_decay=float(cfg["SOLVER"]["STAGE2"]["WEIGHT_DECAY"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(cfg["SOLVER"]["STAGE2"]["MAX_EPOCHS"]))
        )
        best_metrics = do_train_emotion_stage2(cfg, model, train_loader, val_loader, optimizer, scheduler)
        logger.info("Best metrics: %s", best_metrics)


if __name__ == "__main__":
    main()
