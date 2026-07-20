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
    EmotionDataParallel,
    evaluate_sealed_test,
    load_emotion_checkpoint,
    log_run_config,
    log_training_event,
    unwrap_model,
)
from utils.run_artifacts import initialize_immutable_run


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
    suffix = cfg["TRAIN"]["RUN_ID"]
    cfg.setdefault("TRAIN", {})
    cfg["TRAIN"]["RUN_SAVED_AT"] = saved_at.isoformat(timespec="seconds")
    cfg["TRAIN"]["RUN_HISTORY_CSV"] = os.path.join(cfg["OUTPUT_DIR"], f"train_history_{suffix}.csv")
    cfg["TRAIN"]["TRAINING_EPOCH_CSV"] = os.path.join(cfg["OUTPUT_DIR"], "training_epoch_losses.csv")
    cfg["TRAIN"]["VALIDATION_CSV"] = os.path.join(cfg["OUTPUT_DIR"], "validation_metrics.csv")
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


def parse_gpu_ids(gpus: str):
    if not str(gpus or "").strip():
        return []
    try:
        gpu_ids = [int(item.strip()) for item in str(gpus).split(",")]
    except ValueError as exc:
        raise ValueError(f"--gpus must be a comma-separated list of CUDA indices, got {gpus!r}") from exc
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(f"CUDA device indices must be >= 0, got {gpu_ids}")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"CUDA device indices must be unique, got {gpu_ids}")
    return gpu_ids


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
        "--gpus",
        default="",
        help="Comma-separated CUDA indices for single-process DataParallel, for example --gpus 0,1",
    )
    parser.add_argument("--run-id", default="", help="Unique immutable artifact run id")
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
    gpu_ids = parse_gpu_ids(args.gpus)
    if args.gpu is not None and gpu_ids:
        raise ValueError("Use either --gpu or --gpus, not both")
    device, device_warning = configure_device(cfg, gpu_id=gpu_ids[0] if gpu_ids else args.gpu)
    if gpu_ids and device.type != "cuda":
        gpu_ids = []
    if gpu_ids:
        unavailable = [gpu_id for gpu_id in gpu_ids if gpu_id >= torch.cuda.device_count()]
        if unavailable:
            raise ValueError(
                f"CUDA device(s) {unavailable} were requested, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible"
            )
    cfg["TRAIN"]["GPU_IDS"] = gpu_ids or ([device.index] if device.type == "cuda" else [])
    cfg["TRAIN"]["PARALLEL_MODE"] = "data_parallel" if len(gpu_ids) > 1 else "single_device"
    initialize_immutable_run(cfg, run_id=args.run_id)
    setup_logging(cfg["OUTPUT_DIR"])
    logger = logging.getLogger("emotionclip.train")

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
    train_loader, train_loader_stage1, val_loader, test_loader, class_names = make_emotion_dataloaders(cfg)
    log_training_event(
        logger,
        "Dataloaders ready",
        train_samples=len(train_loader.dataset) if hasattr(train_loader, "dataset") else "unknown",
        val_samples=len(val_loader.dataset) if val_loader is not None else 0,
        test_samples=len(test_loader.dataset) if test_loader is not None else 0,
        stage1_batches=len(train_loader_stage1),
        stage2_batches=len(train_loader),
        val_batches=len(val_loader) if val_loader is not None else 0,
        test_batches=len(test_loader) if test_loader is not None else 0,
        classes=",".join(class_names),
    )
    model_cfg = cfg["MODEL"]["EMOTION"]
    uncertainty_cfg = cfg["MODEL"].get("UNCERTAINTY", {})
    fusion_cfg = cfg["MODEL"].get("FUSION", {})
    prompt_geometry_cfg = cfg["MODEL"].get("ANATOMY_PROMPT", {})
    routing_cfg = cfg["MODEL"].get("ROUTING", {})
    geometry_cfg = cfg["MODEL"].get("GEOMETRY", {})
    disagreement_cfg = cfg["MODEL"].get("REGION_DISAGREEMENT", {})
    if uncertainty_cfg.get("MODE", "decoupled") != "decoupled":
        raise ValueError("MODEL.UNCERTAINTY.MODE currently supports only 'decoupled'")
    log_training_event(logger, "Building model", model=cfg["MODEL"]["NAME"])
    model = EmotionCLIPModel(
        class_names=class_names,
        backbone_name=cfg["MODEL"]["NAME"],
        image_size=cfg["INPUT"]["SIZE_TRAIN"],
        stride_size=cfg["MODEL"]["STRIDE_SIZE"],
        n_ctx=int(model_cfg["N_CTX"]),
        prompt_geometry_mode=str(prompt_geometry_cfg.get("MODE", "quality")),
        prompt_geometry_hidden_dim=int(prompt_geometry_cfg.get("HIDDEN_DIM", 32)),
        prompt_geometry_gate_init=float(prompt_geometry_cfg.get("GATE_INIT", -4.0)),
        adapter_dim=int(model_cfg["ADAPTER_DIM"]),
        adapter_dropout=float(model_cfg["ADAPTER_DROPOUT"]),
        topk_patches=int(model_cfg["TOPK_PATCHES"]),
        global_weight=float(model_cfg["GLOBAL_WEIGHT"]),
        local_weight=float(model_cfg["LOCAL_WEIGHT"]),
        classifier_weight=float(model_cfg["CLASSIFIER_WEIGHT"]),
        train_last_blocks=int(model_cfg["TRAIN_LAST_BLOCKS"]),
        fusion_gate_mode=str(fusion_cfg.get("GATE_MODE", "fixed")),
        fusion_scale_mode=str(fusion_cfg.get("SCALE_MODE", "temperature")),
        fusion_gate_hidden_dim=int(fusion_cfg.get("GATE_HIDDEN_DIM", 128)),
        fusion_gate_dropout=float(fusion_cfg.get("GATE_DROPOUT", 0.1)),
        min_branch_temperature=float(fusion_cfg.get("MIN_TEMPERATURE", 0.05)),
        max_branch_temperature=float(fusion_cfg.get("MAX_TEMPERATURE", 20.0)),
        initial_branch_temperatures=fusion_cfg.get("INITIAL_TEMPERATURES", [0.1, 1.0, 1.0]),
        learn_branch_temperatures=bool(fusion_cfg.get("LEARN_TEMPERATURES", True)),
        routing_mode=str(routing_cfg.get("MODE", "hybrid")),
        routing_sigma=float(routing_cfg.get("SIGMA", 0.08)),
        geometry_hidden_dim=int(geometry_cfg.get("HIDDEN_DIM", 64)),
        region_importance_hidden_dim=int(geometry_cfg.get("IMPORTANCE_HIDDEN_DIM", 128)),
        geometry_gate_init=float(geometry_cfg.get("GATE_INIT", -4.0)),
        geometry_enabled=bool(geometry_cfg.get("ENABLED", True)),
        geometry_fusion_mode=str(geometry_cfg.get("FUSION_MODE", "gated_residual")),
        disagreement_quality_threshold=float(disagreement_cfg.get("QUALITY_THRESHOLD", 0.5)),
        disagreement_min_regions=int(disagreement_cfg.get("MIN_REGIONS", 2)),
        reliability_hidden_dim=int(uncertainty_cfg.get("HIDDEN_DIM", 128)),
        reliability_dropout=float(uncertainty_cfg.get("DROPOUT", 0.1)),
        detach_class_prob=bool(uncertainty_cfg.get("DETACH_CLASS_PROB", True)),
        reliability_detach_visual_feature=bool(
            uncertainty_cfg.get("DETACH_VISUAL_FEATURE", True)
        ),
        max_strength=uncertainty_cfg.get("MAX_STRENGTH", 100.0),
        reliability_use_anatomy_quality=bool(uncertainty_cfg.get("USE_ANATOMY_QUALITY", True)),
    )

    model.to(device)
    if len(gpu_ids) > 1:
        model = EmotionDataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
        log_training_event(
            logger,
            "DataParallel enabled",
            gpu_ids=",".join(str(gpu_id) for gpu_id in gpu_ids),
            primary_device=device,
            global_batch_size=cfg["SOLVER"]["STAGE2"].get("IMS_PER_BATCH", "unknown"),
        )
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
        load_emotion_checkpoint(
            model,
            stage1_weight,
            strict=False,
            allow_config_mismatch=True,
            allow_untrained_stage2=True,
        )

    if cfg["TRAIN"].get("RUN_STAGE1", True):
        unwrap_model(model).set_train_stage(1)
        optimizer = torch.optim.AdamW(
            trainable_parameters(model),
            lr=float(cfg["SOLVER"]["STAGE1"]["BASE_LR"]),
            weight_decay=float(cfg["SOLVER"]["STAGE1"]["WEIGHT_DECAY"]),
        )
        stage1_cfg = cfg["SOLVER"]["STAGE1"]
        stage1_mode = str(stage1_cfg.get("MODE", "base")).lower()
        stage1_epochs = int(stage1_cfg.get("MAX_EPOCHS", 20))
        if stage1_mode == "both":
            stage1_epochs = int(stage1_cfg.get("BASE_EPOCHS", stage1_epochs)) + int(
                stage1_cfg.get("GEOMETRY_EPOCHS", 0)
            )
        elif stage1_mode == "geometry":
            stage1_epochs = int(stage1_cfg.get("GEOMETRY_EPOCHS", stage1_epochs))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, stage1_epochs))
        do_train_emotion_stage1(cfg, model, train_loader_stage1, optimizer, scheduler)

    if cfg["TRAIN"].get("RUN_STAGE2", True):
        unwrap_model(model).set_train_stage(2)
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

    if cfg.get("TEST", {}).get("EVALUATE_AFTER_TRAIN", False):
        test_weight = str(cfg["TEST"].get("WEIGHT") or "")
        if not test_weight:
            checkpoint_name = "best_emotionclip.pth" if val_loader is not None else "last_emotionclip.pth"
            test_weight = os.path.join(cfg["OUTPUT_DIR"], checkpoint_name)
        if not os.path.exists(test_weight):
            raise FileNotFoundError(
                f"Cannot run sealed test evaluation: checkpoint not found at {test_weight!r}. "
                "Run Stage 2 or set TEST.WEIGHT explicitly."
            )
        selection_split = "val" if val_loader is not None else "fixed_epoch_no_validation"
        log_training_event(
            logger,
            "Sealed test evaluation",
            checkpoint=test_weight,
            selection_split=selection_split,
        )
        test_metrics = evaluate_sealed_test(
            cfg,
            model,
            test_loader,
            checkpoint_path=test_weight,
            selection_split=selection_split,
        )
        log_training_event(
            logger,
            "Test result",
            accuracy=test_metrics["accuracy"],
            balanced_acc=test_metrics["balanced_accuracy"],
            macro_f1=test_metrics["macro_f1"],
            ece=test_metrics["ece"],
            samples=test_metrics["num_samples"],
        )


if __name__ == "__main__":
    main()
