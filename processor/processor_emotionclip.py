import json
import logging
import os
import time
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from loss.emotion_losses import emotion_stage2_loss
from utils.fer_metrics import compute_fer_metrics

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None


def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = dict(batch)
    moved["images"] = batch["images"].to(device, non_blocking=True)
    moved["labels"] = batch["labels"].to(device, non_blocking=True)
    return moved


def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, dynamic_ncols=True, leave=False, **kwargs)


def _batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().detach().cpu())


def _batch_confidence(probabilities: torch.Tensor) -> float:
    return float(probabilities.max(dim=1).values.mean().detach().cpu())


def _batch_uncertainty(uncertainty: torch.Tensor) -> float:
    return float(uncertainty.mean().detach().cpu())


def _checkpoint_payload(model, epoch: int, stage: int, metrics: Optional[Dict[str, Any]] = None):
    return {
        "stage": stage,
        "epoch": epoch,
        "model": model.state_dict(),
        "class_names": model.class_names,
        "metrics": metrics or {},
    }


def save_checkpoint(model, output_dir: str, name: str, epoch: int, stage: int, metrics: Optional[Dict[str, Any]] = None):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    torch.save(_checkpoint_payload(model, epoch, stage, metrics), path)
    return path


def load_emotion_checkpoint(model, checkpoint_path: str, strict: bool = True):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=strict)
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
    checkpoint_period = int(stage_cfg.get("CHECKPOINT_PERIOD", max_epochs))

    model.to(device)
    model.set_train_stage(1)
    cached = precompute_stage1_features(model, train_loader_stage1, device)
    features = cached["features"].to(device)
    labels = cached["labels"].to(device)
    batch_size = int(stage_cfg.get("IMS_PER_BATCH", 64))
    logger.info(
        "Stage1 start epochs=%s samples=%s batch_size=%s steps_per_epoch=%s lr=%.6g",
        max_epochs,
        labels.numel(),
        batch_size,
        max(1, (labels.shape[0] + batch_size - 1) // batch_size),
        optimizer.param_groups[0]["lr"],
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
        progress = _progress(epoch_starts, desc=f"Stage1 {epoch}/{max_epochs}", total=len(epoch_starts))
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
            if steps % log_period == 0:
                logger.info(
                    "Stage1 epoch=%s/%s step=%s/%s loss=%.4f acc=%.4f conf=%.4f lr=%.6g",
                    epoch,
                    max_epochs,
                    steps,
                    max(1, (labels.shape[0] + batch_size - 1) // batch_size),
                    total_loss / max(steps, 1),
                    total_acc / max(steps, 1),
                    total_conf / max(steps, 1),
                    optimizer.param_groups[0]["lr"],
                )

        if scheduler is not None:
            scheduler.step()
        avg_loss = total_loss / max(steps, 1)
        logger.info(
            "Stage1 epoch=%s/%s done loss=%.4f acc=%.4f conf=%.4f lr=%.6g time=%.1fs",
            epoch,
            max_epochs,
            avg_loss,
            total_acc / max(steps, 1),
            total_conf / max(steps, 1),
            optimizer.param_groups[0]["lr"],
            time.time() - start_time,
        )

        if epoch % checkpoint_period == 0 or epoch == max_epochs:
            save_checkpoint(model, output_dir, f"{model.backbone_name}_emotion_stage1_{epoch}.pth", epoch, stage=1)


def evaluate_emotion_model(cfg, model, val_loader, text_features: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    device = torch.device(cfg["MODEL"]["DEVICE"])
    labels = []
    probabilities = []
    uncertainties = []
    image_paths = []
    model.eval()
    with torch.no_grad():
        if text_features is None:
            text_features = model(get_text=True)
        text_features = text_features.to(device)
        for batch in val_loader:
            batch = _batch_to_device(batch, device)
            outputs = model(images=batch["images"], text_features=text_features)
            labels.extend(batch["labels"].detach().cpu().tolist())
            probabilities.extend(outputs["probabilities"].detach().cpu().tolist())
            uncertainties.extend(outputs["uncertainty"].detach().cpu().tolist())
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
    return metrics


def do_train_emotion_stage2(cfg, model, train_loader, val_loader, optimizer, scheduler=None):
    logger = logging.getLogger("emotionclip.train")
    device = torch.device(cfg["MODEL"]["DEVICE"])
    output_dir = cfg["OUTPUT_DIR"]
    stage_cfg = cfg["SOLVER"]["STAGE2"]
    max_epochs = int(stage_cfg["MAX_EPOCHS"])
    log_period = int(stage_cfg.get("LOG_PERIOD", 20))
    checkpoint_period = int(stage_cfg.get("CHECKPOINT_PERIOD", max_epochs))
    eval_period = int(stage_cfg.get("EVAL_PERIOD", 1))
    beta_align = float(stage_cfg.get("BETA_ALIGN", 0.5))
    lambda_unc = float(stage_cfg.get("LAMBDA_UNC", 0.05))
    anneal_epochs = max(1, int(stage_cfg.get("EDL_ANNEALING_EPOCHS", 10)))

    model.to(device)
    model.set_train_stage(2)
    best_macro_f1 = -1.0
    best_metrics = None
    logger.info(
        "Stage2 start epochs=%s samples=%s batch_size=%s steps_per_epoch=%s lr=%.6g beta_align=%.3f lambda_unc=%.3f",
        max_epochs,
        len(train_loader.dataset) if hasattr(train_loader, "dataset") else "unknown",
        stage_cfg.get("IMS_PER_BATCH", "unknown"),
        len(train_loader),
        optimizer.param_groups[0]["lr"],
        beta_align,
        lambda_unc,
    )
    for epoch in range(1, max_epochs + 1):
        start_time = time.time()
        model.train()
        text_features = model(get_text=True).detach()
        total_loss = 0.0
        total_cls = 0.0
        total_align = 0.0
        total_unc = 0.0
        total_pred_unc = 0.0
        total_conf = 0.0
        total_acc = 0.0
        steps = 0
        anneal = min(1.0, epoch / anneal_epochs)
        progress = _progress(train_loader, desc=f"Stage2 {epoch}/{max_epochs}", total=len(train_loader))
        for batch in progress:
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad()
            outputs = model(images=batch["images"], text_features=text_features)
            losses = emotion_stage2_loss(
                outputs,
                batch["labels"],
                beta_align=beta_align,
                lambda_unc=lambda_unc,
                edl_annealing=anneal,
            )
            losses["loss"].backward()
            optimizer.step()
            batch_loss = float(losses["loss"].detach().cpu())
            batch_cls = float(losses["classification"].detach().cpu())
            batch_align = float(losses["alignment"].detach().cpu())
            batch_unc = float(losses["uncertainty"].detach().cpu())
            batch_pred_unc = _batch_uncertainty(outputs["uncertainty"])
            batch_conf = _batch_confidence(outputs["probabilities"])
            batch_acc = _batch_accuracy(outputs["logits"], batch["labels"])
            total_loss += batch_loss
            total_cls += batch_cls
            total_align += batch_align
            total_unc += batch_unc
            total_pred_unc += batch_pred_unc
            total_conf += batch_conf
            total_acc += batch_acc
            steps += 1
            if tqdm is not None:
                progress.set_postfix(
                    loss=f"{total_loss / max(steps, 1):.4f}",
                    acc=f"{total_acc / max(steps, 1):.3f}",
                    unc=f"{total_pred_unc / max(steps, 1):.3f}",
                    conf=f"{total_conf / max(steps, 1):.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )
            if steps % log_period == 0:
                logger.info(
                    "Stage2 epoch=%s/%s step=%s/%s loss=%.4f cls=%.4f align=%.4f unc_loss=%.4f pred_unc=%.4f conf=%.4f acc=%.4f anneal=%.3f lr=%.6g",
                    epoch,
                    max_epochs,
                    steps,
                    len(train_loader),
                    total_loss / max(steps, 1),
                    total_cls / max(steps, 1),
                    total_align / max(steps, 1),
                    total_unc / max(steps, 1),
                    total_pred_unc / max(steps, 1),
                    total_conf / max(steps, 1),
                    total_acc / max(steps, 1),
                    anneal,
                    optimizer.param_groups[0]["lr"],
                )

        if scheduler is not None:
            scheduler.step()
        avg_loss = total_loss / max(steps, 1)
        logger.info(
            "Stage2 epoch=%s/%s done loss=%.4f cls=%.4f align=%.4f unc_loss=%.4f pred_unc=%.4f conf=%.4f acc=%.4f lr=%.6g time=%.1fs",
            epoch,
            max_epochs,
            avg_loss,
            total_cls / max(steps, 1),
            total_align / max(steps, 1),
            total_unc / max(steps, 1),
            total_pred_unc / max(steps, 1),
            total_conf / max(steps, 1),
            total_acc / max(steps, 1),
            optimizer.param_groups[0]["lr"],
            time.time() - start_time,
        )

        metrics = None
        if epoch % eval_period == 0 or epoch == max_epochs:
            metrics = evaluate_emotion_model(cfg, model, val_loader)
            metrics_path = os.path.join(output_dir, f"metrics_epoch_{epoch}.json")
            os.makedirs(output_dir, exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
            logger.info(
                "Validation epoch=%s/%s accuracy=%.4f balanced_acc=%.4f macro_f1=%.4f avg_unc=%.4f avg_conf=%.4f ece=%.4f uncertainty_risk_auc=%.4f samples=%s",
                epoch,
                max_epochs,
                metrics["accuracy"],
                metrics["balanced_accuracy"],
                metrics["macro_f1"],
                metrics["avg_uncertainty"],
                metrics["avg_confidence"],
                metrics["ece"],
                metrics["uncertainty_risk_auc"],
                metrics["num_samples"],
            )
            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                best_metrics = metrics
                save_checkpoint(model, output_dir, "best_emotionclip.pth", epoch, stage=2, metrics=metrics)

        if epoch % checkpoint_period == 0 or epoch == max_epochs:
            save_checkpoint(
                model,
                output_dir,
                f"{model.backbone_name}_emotion_stage2_{epoch}.pth",
                epoch,
                stage=2,
                metrics=metrics,
            )

    return best_metrics
