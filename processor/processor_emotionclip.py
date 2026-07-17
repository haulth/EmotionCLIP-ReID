import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from loss.emotion_losses import emotion_stage2_loss
from utils.fer_metrics import compute_fer_metrics

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None


def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = dict(batch)
    moved["images"] = batch["images"].to(device, non_blocking=True)
    moved["labels"] = batch["labels"].to(device, non_blocking=True)
    return moved


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "disable", "disabled"}:
        return False
    return None


def _progress_enabled(cfg: Optional[Dict[str, Any]] = None) -> bool:
    if _as_bool(os.getenv("TQDM_DISABLE")) is True:
        return False

    setting = "auto"
    if cfg is not None:
        setting = cfg.get("TRAIN", {}).get("PROGRESS_BAR", setting)
    env_value = os.getenv("EMOTIONCLIP_PROGRESS")
    if env_value is not None:
        setting = env_value

    parsed = _as_bool(setting)
    if parsed is not None:
        return parsed
    return sys.stderr.isatty()


def _progress(iterable, cfg: Optional[Dict[str, Any]] = None, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, dynamic_ncols=True, leave=False, disable=not _progress_enabled(cfg), **kwargs)


def _batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().detach().cpu())


def _batch_confidence(probabilities: torch.Tensor) -> float:
    return float(probabilities.max(dim=1).values.mean().detach().cpu())


def _batch_uncertainty(uncertainty: torch.Tensor) -> float:
    return float(uncertainty.mean().detach().cpu())


def _format_log_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def log_training_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    details = " ".join(f"{key}={_format_log_value(value)}" for key, value in fields.items())
    logger.info("%s%s", event, f" {details}" if details else "")


_RUN_HISTORY_COLUMNS = [
    "timestamp",
    "run_id",
    "event",
    "stage",
    "epoch",
    "epoch_total",
    "split",
    "loss",
    "cls",
    "align",
    "unc_loss",
    "gate_loss",
    "temperature_loss",
    "pred_unc",
    "conf",
    "acc",
    "fusion_gate_classifier",
    "fusion_gate_global",
    "fusion_gate_local",
    "fusion_gate_classifier_std",
    "fusion_gate_global_std",
    "fusion_gate_local_std",
    "gate_dominant_classifier_rate",
    "gate_dominant_global_rate",
    "gate_dominant_local_rate",
    "temperature_classifier",
    "temperature_global",
    "temperature_local",
    "classifier_logit_abs_mean",
    "global_logit_abs_mean",
    "local_logit_abs_mean",
    "classifier_logit_mean",
    "global_logit_mean",
    "local_logit_mean",
    "classifier_logit_std",
    "global_logit_std",
    "local_logit_std",
    "scaled_classifier_logit_abs_mean",
    "scaled_global_logit_abs_mean",
    "scaled_local_logit_abs_mean",
    "gate_entropy",
    "gate_collapse_rate",
    "images_per_second",
    "peak_vram_allocated_mb",
    "peak_vram_reserved_mb",
    "lr",
    "time_sec",
    "accuracy",
    "balanced_acc",
    "macro_f1",
    "avg_unc",
    "avg_conf",
    "avg_uncertainty",
    "avg_confidence",
    "avg_strength",
    "avg_entropy",
    "ece",
    "adaptive_ece",
    "classwise_ece",
    "nll",
    "brier",
    "aurc",
    "eaurc",
    "error_auroc",
    "error_aupr",
    "uncertainty_risk_auc",
    "samples",
    "num_samples",
]

_TRAIN_EPOCH_COLUMNS = [
    "run_id",
    "stage",
    "epoch",
    "epoch_total",
    "loss",
    "cls",
    "align",
    "unc_loss",
    "gate_loss",
    "temperature_loss",
    "pred_unc",
    "conf",
    "acc",
    "fusion_gate_classifier",
    "fusion_gate_global",
    "fusion_gate_local",
    "fusion_gate_classifier_std",
    "fusion_gate_global_std",
    "fusion_gate_local_std",
    "gate_dominant_classifier_rate",
    "gate_dominant_global_rate",
    "gate_dominant_local_rate",
    "temperature_classifier",
    "temperature_global",
    "temperature_local",
    "classifier_logit_abs_mean",
    "global_logit_abs_mean",
    "local_logit_abs_mean",
    "classifier_logit_mean",
    "global_logit_mean",
    "local_logit_mean",
    "classifier_logit_std",
    "global_logit_std",
    "local_logit_std",
    "scaled_classifier_logit_abs_mean",
    "scaled_global_logit_abs_mean",
    "scaled_local_logit_abs_mean",
    "gate_entropy",
    "gate_collapse_rate",
    "images_per_second",
    "peak_vram_allocated_mb",
    "peak_vram_reserved_mb",
    "lr",
    "time_sec",
]

_VALIDATION_COLUMNS = [
    "run_id",
    "epoch",
    "epoch_total",
    "accuracy",
    "balanced_acc",
    "balanced_accuracy",
    "macro_f1",
    "avg_unc",
    "avg_conf",
    "avg_uncertainty",
    "avg_confidence",
    "avg_strength",
    "avg_entropy",
    "ece",
    "adaptive_ece",
    "classwise_ece",
    "nll",
    "brier",
    "aurc",
    "eaurc",
    "error_auroc",
    "error_aupr",
    "uncertainty_risk_auc",
    "samples",
    "num_samples",
]


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return value


def _append_csv_row(path: Optional[str], columns: list, row: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _artifact_path(cfg: Dict[str, Any], key: str, default_name: str) -> str:
    return cfg.get("TRAIN", {}).get(key) or os.path.join(cfg["OUTPUT_DIR"], default_name)


def _record_training_epoch(cfg: Dict[str, Any], row: Dict[str, Any]) -> None:
    row = dict(row)
    row.setdefault("timestamp", datetime.now().astimezone().isoformat(timespec="seconds"))
    row.setdefault("run_id", cfg.get("TRAIN", {}).get("RUN_ID", ""))
    row.setdefault("split", "train")
    _append_csv_row(_artifact_path(cfg, "RUN_HISTORY_CSV", "train_history.csv"), _RUN_HISTORY_COLUMNS, row)
    _append_csv_row(
        _artifact_path(cfg, "TRAINING_EPOCH_CSV", "training_epoch_losses.csv"),
        _TRAIN_EPOCH_COLUMNS,
        row,
    )


def _record_validation_epoch(cfg: Dict[str, Any], row: Dict[str, Any]) -> None:
    row = dict(row)
    row.setdefault("timestamp", datetime.now().astimezone().isoformat(timespec="seconds"))
    row.setdefault("run_id", cfg.get("TRAIN", {}).get("RUN_ID", ""))
    row.setdefault("event", "Validation")
    row.setdefault("stage", 2)
    row.setdefault("split", "val")
    _append_csv_row(_artifact_path(cfg, "RUN_HISTORY_CSV", "train_history.csv"), _RUN_HISTORY_COLUMNS, row)
    _append_csv_row(_artifact_path(cfg, "VALIDATION_CSV", "validation_metrics.csv"), _VALIDATION_COLUMNS, row)


def log_run_config(logger: logging.Logger, cfg: Dict[str, Any], config_file: str = "", opts: Optional[list] = None) -> None:
    model_cfg = cfg["MODEL"]
    emotion_cfg = model_cfg["EMOTION"]
    fusion_cfg = model_cfg.get("FUSION", {})
    solver_cfg = cfg["SOLVER"]
    stage1_cfg = solver_cfg["STAGE1"]
    stage2_cfg = solver_cfg["STAGE2"]
    log_training_event(
        logger,
        "Run config",
        config_file=config_file or "<default>",
        device=model_cfg["DEVICE"],
        model=model_cfg["NAME"],
        manifest=cfg["DATASETS"]["MANIFEST"],
        root_dir=cfg["DATASETS"].get("ROOT_DIR") or "<none>",
        output_dir=cfg["OUTPUT_DIR"],
        size_train=cfg["INPUT"]["SIZE_TRAIN"],
        size_test=cfg["INPUT"]["SIZE_TEST"],
        run_stage1=cfg["TRAIN"].get("RUN_STAGE1", True),
        run_stage2=cfg["TRAIN"].get("RUN_STAGE2", True),
        progress_bar=cfg["TRAIN"].get("PROGRESS_BAR", "auto"),
    )
    log_training_event(
        logger,
        "Model config",
        n_ctx=emotion_cfg["N_CTX"],
        adapter_dim=emotion_cfg["ADAPTER_DIM"],
        topk_patches=emotion_cfg["TOPK_PATCHES"],
        train_last_blocks=emotion_cfg["TRAIN_LAST_BLOCKS"],
        classifier_weight=emotion_cfg["CLASSIFIER_WEIGHT"],
        global_weight=emotion_cfg["GLOBAL_WEIGHT"],
        local_weight=emotion_cfg["LOCAL_WEIGHT"],
        fusion_gate_mode=fusion_cfg.get("GATE_MODE", "fixed"),
        fusion_scale_mode=fusion_cfg.get("SCALE_MODE", "temperature"),
        learn_temperatures=fusion_cfg.get("LEARN_TEMPERATURES", True),
        initial_temperatures=fusion_cfg.get("INITIAL_TEMPERATURES", [0.1, 1.0, 1.0]),
    )
    log_training_event(
        logger,
        "Solver config",
        seed=solver_cfg.get("SEED", 1234),
        stage1_epochs=stage1_cfg["MAX_EPOCHS"],
        stage1_batch=stage1_cfg["IMS_PER_BATCH"],
        stage1_lr=stage1_cfg["BASE_LR"],
        stage1_log_period=stage1_cfg.get("LOG_PERIOD", 20),
        stage2_epochs=stage2_cfg["MAX_EPOCHS"],
        stage2_batch=stage2_cfg["IMS_PER_BATCH"],
        stage2_lr=stage2_cfg["BASE_LR"],
        stage2_log_period=stage2_cfg.get("LOG_PERIOD", 20),
        beta_align=stage2_cfg.get("BETA_ALIGN", 0.5),
        lambda_unc=stage2_cfg.get("LAMBDA_UNC", 0.05),
    )
    if opts:
        log_training_event(logger, "CLI opts", opts=" ".join(str(item) for item in opts))


def _checkpoint_payload(model, epoch: int, stage: int, metrics: Optional[Dict[str, Any]] = None):
    return {
        "stage": stage,
        "epoch": epoch,
        "model": model.state_dict(),
        "class_names": model.class_names,
        "metrics": metrics or {},
    }


def parameter_report(model) -> Dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    groups: Dict[str, int] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".emotion_adapter." in name:
            group = "adapters"
        elif name.startswith("classifier."):
            group = "classifier"
        elif name.startswith("reliability_head."):
            group = "reliability"
        elif "temperature" in name:
            group = "temperatures"
        elif "fusion" in name:
            group = "fusion"
        elif "image_encoder" in name:
            group = "unfrozen_backbone"
        else:
            group = "other"
        groups[group] = groups.get(group, 0) + parameter.numel()
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_percent": 100.0 * trainable / max(total, 1),
        "groups": groups,
    }


def save_checkpoint(model, output_dir: str, name: str, epoch: int, stage: int, metrics: Optional[Dict[str, Any]] = None):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    torch.save(_checkpoint_payload(model, epoch, stage, metrics), path)
    return path


def load_emotion_checkpoint(model, checkpoint_path: str, strict: bool = True):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict and any(key.startswith("reliability_head.") for key in incompatible.missing_keys):
        logging.getLogger("emotionclip.checkpoint").warning(
            "Checkpoint %s predates the decoupled reliability head; classification remains usable, "
            "but strength/uncertainty is untrained until Stage 2 is retrained.",
            checkpoint_path,
        )
    if not strict and "fusion_weights" in incompatible.unexpected_keys:
        logging.getLogger("emotionclip.checkpoint").warning(
            "Checkpoint %s uses unconstrained fusion_weights; they were replaced by the configured simplex "
            "fusion prior and must be retrained for calibrated branch temperatures.",
            checkpoint_path,
        )
    return checkpoint


def precompute_stage1_features(model, loader, device: torch.device) -> Dict[str, torch.Tensor]:
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            image_features = model(images=batch["images"], get_image=True)
            features.append(image_features.detach().cpu())
            labels.append(batch["labels"].detach().cpu())
    return {"features": torch.cat(features, dim=0), "labels": torch.cat(labels, dim=0)}


def do_train_emotion_stage1(cfg, model, train_loader_stage1, optimizer, scheduler=None):
    logger = logging.getLogger("emotionclip.train")
    device = torch.device(cfg["MODEL"]["DEVICE"])
    output_dir = cfg["OUTPUT_DIR"]
    stage_cfg = cfg["SOLVER"]["STAGE1"]
    max_epochs = int(stage_cfg["MAX_EPOCHS"])
    log_period = int(stage_cfg.get("LOG_PERIOD", 20))

    model.to(device)
    model.set_train_stage(1)
    cached = precompute_stage1_features(model, train_loader_stage1, device)
    features = cached["features"].to(device)
    labels = cached["labels"].to(device)
    batch_size = int(stage_cfg.get("IMS_PER_BATCH", 64))
    log_training_event(
        logger,
        "Stage1 start",
        epoch_total=max_epochs,
        samples=labels.numel(),
        batch_size=batch_size,
        steps_per_epoch=max(1, (labels.shape[0] + batch_size - 1) // batch_size),
        lr=optimizer.param_groups[0]["lr"],
    )

    for epoch in range(1, max_epochs + 1):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_conf = 0.0
        steps = 0
        permutation = torch.randperm(labels.shape[0], device=device)
        epoch_starts = range(0, labels.shape[0], batch_size)
        progress = _progress(epoch_starts, cfg=cfg, desc=f"Stage1 {epoch}/{max_epochs}", total=len(epoch_starts))
        for start in progress:
            indices = permutation[start : start + batch_size]
            batch_features = features[indices]
            batch_labels = labels[indices]
            optimizer.zero_grad()
            text_features = model(get_text=True)
            logits = model.logit_scale.exp().float() * batch_features @ text_features.t()
            loss = F.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()

            batch_loss = float(loss.detach().cpu())
            batch_acc = _batch_accuracy(logits, batch_labels)
            batch_conf = _batch_confidence(F.softmax(logits.detach(), dim=1))
            total_loss += batch_loss
            total_acc += batch_acc
            total_conf += batch_conf
            steps += 1
            if tqdm is not None:
                progress.set_postfix(
                    loss=f"{total_loss / max(steps, 1):.4f}",
                    acc=f"{total_acc / max(steps, 1):.3f}",
                    conf=f"{total_conf / max(steps, 1):.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )
            if log_period > 0 and steps % log_period == 0:
                log_training_event(
                    logger,
                    "Stage1 train",
                    epoch=f"{epoch}/{max_epochs}",
                    step=f"{steps}/{max(1, (labels.shape[0] + batch_size - 1) // batch_size)}",
                    loss=total_loss / max(steps, 1),
                    acc=total_acc / max(steps, 1),
                    conf=total_conf / max(steps, 1),
                    lr=optimizer.param_groups[0]["lr"],
                )

        if scheduler is not None:
            scheduler.step()
        avg_loss = total_loss / max(steps, 1)
        epoch_metrics = {
            "event": "Stage1 done",
            "stage": 1,
            "epoch": epoch,
            "epoch_total": max_epochs,
            "loss": avg_loss,
            "acc": total_acc / max(steps, 1),
            "conf": total_conf / max(steps, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": time.time() - start_time,
        }
        log_training_event(
            logger,
            "Stage1 done",
            epoch=f"{epoch}/{max_epochs}",
            loss=epoch_metrics["loss"],
            acc=epoch_metrics["acc"],
            conf=epoch_metrics["conf"],
            lr=epoch_metrics["lr"],
            time_sec=epoch_metrics["time_sec"],
        )
        _record_training_epoch(cfg, epoch_metrics)

        save_checkpoint(model, output_dir, "last_emotionclip_stage1.pth", epoch, stage=1)


def evaluate_emotion_model(cfg, model, data_loader, text_features: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    device = torch.device(cfg["MODEL"]["DEVICE"])
    labels = []
    probabilities = []
    uncertainties = []
    strengths = []
    image_paths = []
    model.eval()
    with torch.no_grad():
        if text_features is None:
            text_features = model(get_text=True)
        text_features = text_features.to(device)
        for batch in data_loader:
            batch = _batch_to_device(batch, device)
            outputs = model(images=batch["images"], text_features=text_features)
            labels.extend(batch["labels"].detach().cpu().tolist())
            probabilities.extend(outputs["probabilities"].detach().cpu().tolist())
            uncertainties.extend(outputs["uncertainty"].detach().cpu().tolist())
            strengths.extend(outputs["strength"].detach().cpu().tolist())
            image_paths.extend(batch["image_paths"])
    metrics = compute_fer_metrics(labels, probabilities, uncertainties, model.class_names)
    metrics["num_samples"] = len(labels)
    metrics["image_paths"] = image_paths
    if probabilities:
        probabilities_tensor = torch.tensor(probabilities, dtype=torch.float32)
        metrics["avg_confidence"] = float(probabilities_tensor.max(dim=1).values.mean())
    else:
        metrics["avg_confidence"] = 0.0
    metrics["avg_uncertainty"] = float(sum(uncertainties) / max(len(uncertainties), 1))
    metrics["avg_strength"] = float(sum(strengths) / max(len(strengths), 1))
    return metrics


def evaluate_sealed_test(
    cfg,
    model,
    test_loader,
    checkpoint_path: str = "",
    selection_split: str = "val",
) -> Dict[str, Any]:
    """Evaluate the held-out test split once and write a test-specific artifact."""
    if test_loader is None:
        raise ValueError("Sealed test evaluation requested, but the manifest has no split='test'")
    if checkpoint_path:
        load_emotion_checkpoint(model, checkpoint_path, strict=True)
    metrics = evaluate_emotion_model(cfg, model, test_loader)
    metrics.update(
        {
            "evaluation_split": "test",
            "selection_split": selection_split,
            "checkpoint": checkpoint_path or "in_memory_model",
        }
    )
    output_name = cfg.get("TEST", {}).get("OUTPUT_FILE", "test_metrics.json")
    output_path = os.path.join(cfg["OUTPUT_DIR"], output_name)
    os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def do_train_emotion_stage2(cfg, model, train_loader, val_loader, optimizer, scheduler=None):
    logger = logging.getLogger("emotionclip.train")
    device = torch.device(cfg["MODEL"]["DEVICE"])
    output_dir = cfg["OUTPUT_DIR"]
    stage_cfg = cfg["SOLVER"]["STAGE2"]
    max_epochs = int(stage_cfg["MAX_EPOCHS"])
    log_period = int(stage_cfg.get("LOG_PERIOD", 20))
    eval_period = max(1, int(stage_cfg.get("EVAL_PERIOD", 1)))
    beta_align = float(stage_cfg.get("BETA_ALIGN", 0.5))
    configured_reliability_weight = stage_cfg.get("LAMBDA_RELIABILITY")
    lambda_unc = float(
        stage_cfg.get("LAMBDA_UNC", 0.05)
        if configured_reliability_weight is None
        else configured_reliability_weight
    )
    reliability_target = str(stage_cfg.get("RELIABILITY_TARGET", "correctness")).lower()
    lambda_gate = float(stage_cfg.get("LAMBDA_GATE", 0.0))
    lambda_temperature = float(stage_cfg.get("LAMBDA_TEMPERATURE", 0.0))
    anneal_epochs = max(
        1,
        int(stage_cfg.get("RELIABILITY_WARMUP_EPOCHS", stage_cfg.get("EDL_ANNEALING_EPOCHS", 10))),
    )

    model.to(device)
    model.set_train_stage(2)
    params = parameter_report(model)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "trainable_parameters.json"), "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)
    best_macro_f1 = -1.0
    best_metrics = None
    last_metrics = None
    log_training_event(
        logger,
        "Stage2 start",
        epoch_total=max_epochs,
        samples=len(train_loader.dataset) if hasattr(train_loader, "dataset") else "unknown",
        batch_size=stage_cfg.get("IMS_PER_BATCH", "unknown"),
        steps_per_epoch=len(train_loader),
        lr=optimizer.param_groups[0]["lr"],
        beta_align=beta_align,
        lambda_unc=lambda_unc,
        reliability_target=reliability_target,
        trainable_params=params["trainable_params"],
        trainable_percent=params["trainable_percent"],
        parameter_groups=params["groups"],
        selection_split="val" if val_loader is not None else "fixed_epoch_no_validation",
    )
    for epoch in range(1, max_epochs + 1):
        start_time = time.time()
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        text_features = model(get_text=True).detach()
        total_loss = 0.0
        total_cls = 0.0
        total_align = 0.0
        total_unc = 0.0
        total_gate_loss = 0.0
        total_temperature_loss = 0.0
        total_pred_unc = 0.0
        total_conf = 0.0
        total_acc = 0.0
        total_gate = torch.zeros(3, dtype=torch.float64)
        total_gate_std = torch.zeros(3, dtype=torch.float64)
        total_gate_dominant = torch.zeros(3, dtype=torch.float64)
        total_temperatures = torch.zeros(3, dtype=torch.float64)
        total_raw_abs = torch.zeros(3, dtype=torch.float64)
        total_raw_mean = torch.zeros(3, dtype=torch.float64)
        total_raw_std = torch.zeros(3, dtype=torch.float64)
        total_scaled_abs = torch.zeros(3, dtype=torch.float64)
        total_gate_entropy = 0.0
        total_gate_collapse = 0.0
        samples_seen = 0
        steps = 0
        anneal = min(1.0, epoch / anneal_epochs)
        progress = _progress(train_loader, cfg=cfg, desc=f"Stage2 {epoch}/{max_epochs}", total=len(train_loader))
        for batch in progress:
            batch = _batch_to_device(batch, device)
            samples_seen += int(batch["labels"].shape[0])
            optimizer.zero_grad()
            outputs = model(images=batch["images"], text_features=text_features)
            losses = emotion_stage2_loss(
                outputs,
                batch["labels"],
                beta_align=beta_align,
                lambda_unc=lambda_unc,
                edl_annealing=anneal,
                reliability_target=reliability_target,
                lambda_gate=lambda_gate,
                lambda_temperature=lambda_temperature,
            )
            losses["loss"].backward()
            optimizer.step()
            batch_loss = float(losses["loss"].detach().cpu())
            batch_cls = float(losses["classification"].detach().cpu())
            batch_align = float(losses["alignment"].detach().cpu())
            batch_unc = float(losses["uncertainty"].detach().cpu())
            batch_gate_loss = float(losses["gate"].detach().cpu())
            batch_temperature_loss = float(losses["temperature"].detach().cpu())
            batch_pred_unc = _batch_uncertainty(outputs["uncertainty"])
            batch_conf = _batch_confidence(outputs["probabilities"])
            batch_acc = _batch_accuracy(outputs["logits"], batch["labels"])
            total_loss += batch_loss
            total_cls += batch_cls
            total_align += batch_align
            total_unc += batch_unc
            total_gate_loss += batch_gate_loss
            total_temperature_loss += batch_temperature_loss
            total_pred_unc += batch_pred_unc
            total_conf += batch_conf
            total_acc += batch_acc
            if "fusion_gate" in outputs:
                gate = outputs["fusion_gate"].detach().float().cpu()
                total_gate += gate.mean(dim=0).double()
                total_gate_std += gate.std(dim=0, unbiased=False).double()
                total_gate_dominant += F.one_hot(gate.argmax(dim=-1), num_classes=3).float().mean(dim=0).double()
                total_gate_entropy += float(outputs["gate_entropy"].detach().mean().cpu())
                total_gate_collapse += float((gate.max(dim=-1).values > 0.95).float().mean())
                total_temperatures += outputs["branch_temperatures"].detach().float().cpu().double()
                raw_branches = torch.stack(
                    (outputs["classifier_logits"], outputs["global_logits"], outputs["local_logits"]), dim=1
                )
                total_raw_abs += raw_branches.detach().abs().mean(dim=(0, 2)).cpu().double()
                total_raw_mean += raw_branches.detach().mean(dim=(0, 2)).cpu().double()
                total_raw_std += raw_branches.detach().std(dim=(0, 2), unbiased=False).cpu().double()
                total_scaled_abs += (
                    outputs["scaled_branch_logits"].detach().abs().mean(dim=(0, 2)).cpu().double()
                )
            steps += 1
            if tqdm is not None:
                progress.set_postfix(
                    loss=f"{total_loss / max(steps, 1):.4f}",
                    acc=f"{total_acc / max(steps, 1):.3f}",
                    unc=f"{total_pred_unc / max(steps, 1):.3f}",
                    conf=f"{total_conf / max(steps, 1):.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )
            if log_period > 0 and steps % log_period == 0:
                log_training_event(
                    logger,
                    "Stage2 train",
                    epoch=f"{epoch}/{max_epochs}",
                    step=f"{steps}/{len(train_loader)}",
                    loss=total_loss / max(steps, 1),
                    cls=total_cls / max(steps, 1),
                    align=total_align / max(steps, 1),
                    unc_loss=total_unc / max(steps, 1),
                    pred_unc=total_pred_unc / max(steps, 1),
                    conf=total_conf / max(steps, 1),
                    acc=total_acc / max(steps, 1),
                    anneal=anneal,
                    lr=optimizer.param_groups[0]["lr"],
                )

        if scheduler is not None:
            scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_time = time.time() - start_time
        peak_vram_allocated_mb = (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        )
        peak_vram_reserved_mb = (
            torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else 0.0
        )
        avg_loss = total_loss / max(steps, 1)
        fusion_gate_mean = (total_gate / max(steps, 1)).tolist()
        fusion_gate_std = (total_gate_std / max(steps, 1)).tolist()
        fusion_gate_dominant = (total_gate_dominant / max(steps, 1)).tolist()
        branch_temperature_mean = (total_temperatures / max(steps, 1)).tolist()
        branch_raw_abs_mean = (total_raw_abs / max(steps, 1)).tolist()
        branch_raw_mean = (total_raw_mean / max(steps, 1)).tolist()
        branch_raw_std = (total_raw_std / max(steps, 1)).tolist()
        branch_scaled_abs_mean = (total_scaled_abs / max(steps, 1)).tolist()
        epoch_metrics = {
            "event": "Stage2 done",
            "stage": 2,
            "epoch": epoch,
            "epoch_total": max_epochs,
            "loss": avg_loss,
            "cls": total_cls / max(steps, 1),
            "align": total_align / max(steps, 1),
            "unc_loss": total_unc / max(steps, 1),
            "gate_loss": total_gate_loss / max(steps, 1),
            "temperature_loss": total_temperature_loss / max(steps, 1),
            "pred_unc": total_pred_unc / max(steps, 1),
            "conf": total_conf / max(steps, 1),
            "acc": total_acc / max(steps, 1),
            "fusion_gate_classifier": fusion_gate_mean[0],
            "fusion_gate_global": fusion_gate_mean[1],
            "fusion_gate_local": fusion_gate_mean[2],
            "fusion_gate_classifier_std": fusion_gate_std[0],
            "fusion_gate_global_std": fusion_gate_std[1],
            "fusion_gate_local_std": fusion_gate_std[2],
            "gate_dominant_classifier_rate": fusion_gate_dominant[0],
            "gate_dominant_global_rate": fusion_gate_dominant[1],
            "gate_dominant_local_rate": fusion_gate_dominant[2],
            "temperature_classifier": branch_temperature_mean[0],
            "temperature_global": branch_temperature_mean[1],
            "temperature_local": branch_temperature_mean[2],
            "classifier_logit_abs_mean": branch_raw_abs_mean[0],
            "global_logit_abs_mean": branch_raw_abs_mean[1],
            "local_logit_abs_mean": branch_raw_abs_mean[2],
            "classifier_logit_mean": branch_raw_mean[0],
            "global_logit_mean": branch_raw_mean[1],
            "local_logit_mean": branch_raw_mean[2],
            "classifier_logit_std": branch_raw_std[0],
            "global_logit_std": branch_raw_std[1],
            "local_logit_std": branch_raw_std[2],
            "scaled_classifier_logit_abs_mean": branch_scaled_abs_mean[0],
            "scaled_global_logit_abs_mean": branch_scaled_abs_mean[1],
            "scaled_local_logit_abs_mean": branch_scaled_abs_mean[2],
            "gate_entropy": total_gate_entropy / max(steps, 1),
            "gate_collapse_rate": total_gate_collapse / max(steps, 1),
            "images_per_second": samples_seen / max(epoch_time, 1e-12),
            "peak_vram_allocated_mb": peak_vram_allocated_mb,
            "peak_vram_reserved_mb": peak_vram_reserved_mb,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": epoch_time,
        }
        log_training_event(
            logger,
            "Stage2 done",
            epoch=f"{epoch}/{max_epochs}",
            loss=epoch_metrics["loss"],
            cls=epoch_metrics["cls"],
            align=epoch_metrics["align"],
            unc_loss=epoch_metrics["unc_loss"],
            pred_unc=epoch_metrics["pred_unc"],
            conf=epoch_metrics["conf"],
            acc=epoch_metrics["acc"],
            fusion_gate=fusion_gate_mean,
            fusion_gate_std=fusion_gate_std,
            gate_dominant_rate=fusion_gate_dominant,
            branch_temperatures=branch_temperature_mean,
            raw_branch_abs_mean=branch_raw_abs_mean,
            raw_branch_mean=branch_raw_mean,
            raw_branch_std=branch_raw_std,
            scaled_branch_abs_mean=branch_scaled_abs_mean,
            gate_entropy=epoch_metrics["gate_entropy"],
            gate_collapse_rate=epoch_metrics["gate_collapse_rate"],
            images_per_second=epoch_metrics["images_per_second"],
            peak_vram_allocated_mb=epoch_metrics["peak_vram_allocated_mb"],
            lr=epoch_metrics["lr"],
            time_sec=epoch_metrics["time_sec"],
        )
        _record_training_epoch(cfg, epoch_metrics)

        metrics = None
        if val_loader is not None and (epoch % eval_period == 0 or epoch == max_epochs):
            metrics = evaluate_emotion_model(cfg, model, val_loader)
            metrics["evaluation_split"] = "val"
            metrics["selection_split"] = "val"
            metrics_path = os.path.join(output_dir, f"metrics_epoch_{epoch}.json")
            os.makedirs(output_dir, exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
            log_training_event(
                logger,
                "Validation",
                epoch=f"{epoch}/{max_epochs}",
                accuracy=metrics["accuracy"],
                balanced_acc=metrics["balanced_accuracy"],
                macro_f1=metrics["macro_f1"],
                avg_unc=metrics["avg_uncertainty"],
                avg_conf=metrics["avg_confidence"],
                avg_strength=metrics["avg_strength"],
                avg_entropy=metrics["avg_entropy"],
                ece=metrics["ece"],
                nll=metrics["nll"],
                brier=metrics["brier"],
                aurc=metrics["aurc"],
                eaurc=metrics["eaurc"],
                uncertainty_risk_auc=metrics["uncertainty_risk_auc"],
                samples=metrics["num_samples"],
            )
            _record_validation_epoch(
                cfg,
                {
                    "epoch": epoch,
                    "epoch_total": max_epochs,
                    "accuracy": metrics["accuracy"],
                    "balanced_acc": metrics["balanced_accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "avg_unc": metrics["avg_uncertainty"],
                    "avg_conf": metrics["avg_confidence"],
                    "avg_uncertainty": metrics["avg_uncertainty"],
                    "avg_confidence": metrics["avg_confidence"],
                    "avg_strength": metrics["avg_strength"],
                    "avg_entropy": metrics["avg_entropy"],
                    "ece": metrics["ece"],
                    "adaptive_ece": metrics["adaptive_ece"],
                    "classwise_ece": metrics["classwise_ece"],
                    "nll": metrics["nll"],
                    "brier": metrics["brier"],
                    "aurc": metrics["aurc"],
                    "eaurc": metrics["eaurc"],
                    "error_auroc": metrics["error_auroc"],
                    "error_aupr": metrics["error_aupr"],
                    "uncertainty_risk_auc": metrics["uncertainty_risk_auc"],
                    "samples": metrics["num_samples"],
                    "num_samples": metrics["num_samples"],
                },
            )
            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                best_metrics = metrics
                save_checkpoint(model, output_dir, "best_emotionclip.pth", epoch, stage=2, metrics=metrics)
            last_metrics = metrics

        save_checkpoint(model, output_dir, "last_emotionclip.pth", epoch, stage=2, metrics=last_metrics)

    return best_metrics
