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
from torch.nn.parallel.scatter_gather import gather as parallel_gather

from datasets.anatomy import ANATOMY_DESCRIPTOR_VERSION, fit_class_geometry_statistics
from loss.emotion_losses import emotion_stage2_loss
from utils.fer_metrics import compute_fer_metrics


EMOTION_CHECKPOINT_SCHEMA_VERSION = 2
_DERIVED_PROMPT_BUFFER_SUFFIXES = (
    "prompt_learner.token_prefix",
    "prompt_learner.token_suffix",
    "prompt_learner.tokenized_prompts",
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None


def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = dict(batch)
    moved["images"] = batch["images"].to(device, non_blocking=True)
    moved["labels"] = batch["labels"].to(device, non_blocking=True)
    if "anatomy" in batch:
        moved["anatomy"] = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch["anatomy"].items()
        }
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


def _tensor_diagnostic(tensor: torch.Tensor) -> Dict[str, Any]:
    detached = tensor.detach().float()
    finite = torch.isfinite(detached)
    values = detached[finite]
    return {
        "shape": list(detached.shape),
        "finite": bool(finite.all()),
        "nonfinite_count": int((~finite).sum().cpu()),
        "min": float(values.min().cpu()) if values.numel() else None,
        "max": float(values.max().cpu()) if values.numel() else None,
        "mean": float(values.mean().cpu()) if values.numel() else None,
    }


def _model_output_error(outputs: Dict[str, Any], model=None) -> Optional[str]:
    """Return a precise invariant violation before an invalid output reaches a loss."""
    for name in ("logits", "alignment_logits", "probabilities", "raw_strength", "uncertainty"):
        value = outputs.get(name)
        if not torch.is_tensor(value):
            return f"missing tensor output {name!r}"
        if not bool(torch.isfinite(value).all()):
            return f"non-finite model output {name!r}"

    logits = outputs["logits"]
    probabilities = outputs["probabilities"]
    if probabilities.shape != logits.shape:
        return (
            f"probabilities shape {tuple(probabilities.shape)} does not match "
            f"logits shape {tuple(logits.shape)}"
        )
    tolerance = 1e-4
    if bool((probabilities < -tolerance).any()) or bool((probabilities > 1.0 + tolerance).any()):
        return (
            "probabilities outside [0, 1]: "
            f"min={float(probabilities.min().detach().cpu()):.6g} "
            f"max={float(probabilities.max().detach().cpu()):.6g}"
        )
    row_sums = probabilities.float().sum(dim=-1)
    if not bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)):
        return (
            "probability rows do not sum to one: "
            f"min_sum={float(row_sums.min().detach().cpu()):.6g} "
            f"max_sum={float(row_sums.max().detach().cpu()):.6g}"
        )
    uncertainty = outputs["uncertainty"]
    if bool((uncertainty < -tolerance).any()) or bool((uncertainty > 1.0 + tolerance).any()):
        return (
            "uncertainty outside [0, 1]: "
            f"min={float(uncertainty.min().detach().cpu()):.6g} "
            f"max={float(uncertainty.max().detach().cpu()):.6g}"
        )

    raw_strength_unbounded = outputs.get("raw_strength_unbounded")
    if torch.is_tensor(raw_strength_unbounded):
        if not bool(torch.isfinite(raw_strength_unbounded).all()):
            return "non-finite model output 'raw_strength_unbounded'"
        core_model = unwrap_model(model) if model is not None else None
        max_abs = float(getattr(core_model, "max_abs_raw_strength", 20.0))
        observed = float(raw_strength_unbounded.detach().abs().max().cpu())
        if observed > 10.0 * max_abs:
            return (
                "unbounded reliability logit exceeded the safety envelope: "
                f"abs_max={observed:.6g} limit={10.0 * max_abs:.6g}"
            )

    temperatures = outputs.get("branch_temperatures")
    if torch.is_tensor(temperatures):
        if not bool(torch.isfinite(temperatures).all()) or bool((temperatures <= 0).any()):
            return "branch temperatures must be finite and positive"
    return None


def _write_training_failure(
    output_dir: str,
    *,
    epoch: int,
    batch_index: int,
    reason: str,
    losses: Optional[Dict[str, Any]] = None,
    outputs: Optional[Dict[str, Any]] = None,
    model=None,
    batch: Optional[Dict[str, Any]] = None,
) -> str:
    """Write a compact, JSON-safe forensic snapshot before failing closed."""
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "reason": reason,
        "losses": {},
        "outputs": {},
        "nonfinite_gradients": [],
        "nonfinite_parameters": [],
    }
    for name, value in (losses or {}).items():
        if torch.is_tensor(value):
            payload["losses"][name] = _tensor_diagnostic(value)
    for name in (
        "logits",
        "alignment_logits",
        "classifier_logits",
        "global_logits",
        "local_logits",
        "scaled_branch_logits",
        "probabilities",
        "alpha",
        "raw_strength",
        "raw_strength_unbounded",
        "strength",
        "uncertainty",
        "branch_temperatures",
        "routing_loss",
    ):
        value = (outputs or {}).get(name)
        if torch.is_tensor(value):
            payload["outputs"][name] = _tensor_diagnostic(value)
    if model is not None:
        for name, parameter in unwrap_model(model).named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                payload["nonfinite_gradients"].append(name)
            if not bool(torch.isfinite(parameter).all()):
                payload["nonfinite_parameters"].append(name)
    if batch is not None:
        paths = batch.get("image_paths", [])
        payload["image_paths"] = [str(path) for path in list(paths)[:32]]
    if torch.cuda.is_available():
        payload["cuda"] = {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "training_failure.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def _losses_are_finite(losses: Dict[str, Any]) -> bool:
    return all(
        bool(torch.isfinite(value).all())
        for value in losses.values()
        if torch.is_tensor(value)
    )


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older supported PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _nonfinite_parameter_names(model):
    return [
        name
        for name, parameter in unwrap_model(model).named_parameters()
        if parameter.requires_grad and not bool(torch.isfinite(parameter).all())
    ]


def _forward_batch(model, batch: Dict[str, Any], text_features: Optional[torch.Tensor] = None):
    kwargs = {"images": batch["images"]}
    if text_features is not None:
        kwargs["text_features"] = text_features
    if "anatomy" in batch:
        kwargs["anatomy"] = batch["anatomy"]
    return model(**kwargs)


def corrupt_images_for_reliability(
    images: torch.Tensor,
    noise_std: float = 0.08,
    occlusion_ratio: float = 0.2,
    return_occlusion_mask: bool = False,
):
    """Create a clean-corrupted pair and optionally return its pixel occlusion mask."""
    corrupted = images.detach().clone()
    occlusion_mask = torch.zeros(
        images.shape[0],
        images.shape[-2],
        images.shape[-1],
        device=images.device,
        dtype=torch.bool,
    )
    if noise_std > 0:
        corrupted = corrupted + torch.randn_like(corrupted) * float(noise_std)
    ratio = float(occlusion_ratio)
    if ratio > 0:
        height, width = corrupted.shape[-2:]
        side_h = max(1, int(height * ratio))
        side_w = max(1, int(width * ratio))
        for index in range(corrupted.shape[0]):
            top = int(torch.randint(0, max(1, height - side_h + 1), (), device=corrupted.device))
            left = int(torch.randint(0, max(1, width - side_w + 1), (), device=corrupted.device))
            corrupted[index, :, top : top + side_h, left : left + side_w] = 0.0
            occlusion_mask[index, top : top + side_h, left : left + side_w] = True
    if return_occlusion_mask:
        return corrupted, occlusion_mask
    return corrupted


def corrupt_anatomy_for_reliability(
    anatomy: Optional[Dict[str, Any]],
    occlusion_mask: torch.Tensor,
) -> Optional[Dict[str, Any]]:
    """Invalidate anatomy evidence covered by the synthetic image occlusion."""
    if anatomy is None or "region_landmarks" not in anatomy:
        return anatomy
    corrupted = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in anatomy.items()
    }
    coordinates = corrupted["region_landmarks"]
    landmark_mask = corrupted["region_landmark_mask"].bool()
    height, width = occlusion_mask.shape[-2:]
    x = (coordinates[..., 0] * width).floor().long().clamp(0, max(width - 1, 0))
    y = (coordinates[..., 1] * height).floor().long().clamp(0, max(height - 1, 0))
    batch_index = torch.arange(coordinates.shape[0], device=coordinates.device).view(-1, 1, 1)
    covered = occlusion_mask[batch_index, y, x] & landmark_mask
    retained_mask = landmark_mask & ~covered
    original_count = landmark_mask.sum(dim=-1)
    retained_count = retained_mask.sum(dim=-1)
    retained_fraction = torch.where(
        original_count > 0,
        retained_count.to(coordinates.dtype) / original_count.clamp_min(1).to(coordinates.dtype),
        torch.zeros_like(original_count, dtype=coordinates.dtype),
    )
    affected_regions = covered.any(dim=-1)

    corrupted["region_landmark_mask"] = retained_mask
    corrupted["region_landmark_weights"] = corrupted["region_landmark_weights"] * retained_mask.to(
        corrupted["region_landmark_weights"].dtype
    )
    corrupted["region_quality"] = corrupted["region_quality"] * retained_fraction.to(
        corrupted["region_quality"].dtype
    )
    if "geometry_validity" in corrupted:
        corrupted["geometry_validity"] = corrupted["geometry_validity"] & ~affected_regions.unsqueeze(-1)
    if "geometry_uncertainty" in corrupted:
        corrupted["geometry_uncertainty"] = torch.where(
            affected_regions.unsqueeze(-1),
            torch.ones_like(corrupted["geometry_uncertainty"]),
            corrupted["geometry_uncertainty"],
        )
    if "anatomy_available" in corrupted:
        corrupted["anatomy_available"] = retained_mask.flatten(1).any(dim=-1)
    return corrupted


def _format_log_value(value: Any) -> str:
    if isinstance(value, float):
        if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
            return f"{value:.4e}"
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
    "optimizer_steps",
    "gradient_accumulation_steps",
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
    routing_cfg = model_cfg.get("ROUTING", {})
    prompt_geometry_cfg = model_cfg.get("ANATOMY_PROMPT", {})
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
        initial_temperatures=fusion_cfg.get("INITIAL_TEMPERATURES", [1.0, 1.0, 1.0]),
        routing_mode=routing_cfg.get("MODE", "hybrid"),
        routing_sigma=routing_cfg.get("SIGMA", 0.08),
        anatomy_prompt_mode=prompt_geometry_cfg.get("MODE", "quality"),
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
        stage2_gradient_accumulation=stage2_cfg.get("GRADIENT_ACCUMULATION_STEPS", 1),
        stage2_effective_batch=(
            int(stage2_cfg["IMS_PER_BATCH"])
            * int(stage2_cfg.get("GRADIENT_ACCUMULATION_STEPS", 1))
        ),
        stage2_amp=stage2_cfg.get("AMP_ENABLED", True),
        stage2_max_grad_norm=stage2_cfg.get("MAX_GRAD_NORM", 1.0),
        stage2_corruption_ranking=stage2_cfg.get("CORRUPTION", {}).get("LAMBDA_RANKING", 0.0),
        stage2_lr=stage2_cfg["BASE_LR"],
        stage2_log_period=stage2_cfg.get("LOG_PERIOD", 20),
        beta_align=stage2_cfg.get("BETA_ALIGN", 0.5),
        lambda_unc=stage2_cfg.get("LAMBDA_UNC", 0.05),
    )
    if opts:
        log_training_event(logger, "CLI opts", opts=" ".join(str(item) for item in opts))


def unwrap_model(model):
    """Return the underlying EmotionCLIP model for parallel wrappers."""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def get_shared_text_features(model, **kwargs):
    """Build class text descriptors once instead of once per data-parallel replica.

    Text descriptors are indexed by class rather than by image batch. Calling
    ``DataParallel.forward(get_text=True)`` therefore makes every replica return
    the complete class table and the default gather concatenates those tables.
    """
    core_model = unwrap_model(model)
    if not hasattr(core_model, "get_text_features"):
        raise AttributeError("Emotion model must define get_text_features()")
    return core_model.get_text_features(**kwargs)


class EmotionDataParallel(torch.nn.DataParallel):
    """DataParallel adapter for EmotionCLIP's batch and shared output tensors.

    The default DataParallel scatter would split ``text_features`` by class and
    the default gather would concatenate shared tensors such as the three branch
    temperatures. Both operations are incorrect for this model: image-shaped
    tensors are batch-parallel, while text descriptors and regularizers are
    shared across every replica.
    """

    _SHARED_IDENTICAL_KEYS = {
        "branch_temperatures",
        "text_features",
    }
    _SHARED_WEIGHTED_MEAN_KEYS = {
        "gate_regularization",
        "temperature_regularization",
        "routing_loss",
    }

    def scatter(self, inputs, kwargs, device_ids):
        shared_text_features = kwargs.get("text_features")
        scatter_kwargs = dict(kwargs)
        if torch.is_tensor(shared_text_features):
            scatter_kwargs.pop("text_features", None)
        scattered_inputs, scattered_kwargs = super().scatter(inputs, scatter_kwargs, device_ids)
        if torch.is_tensor(shared_text_features):
            for index, replica_kwargs in enumerate(scattered_kwargs):
                target = torch.device("cuda", device_ids[index])
                replica_kwargs["text_features"] = shared_text_features.to(
                    target,
                    non_blocking=True,
                )
        return scattered_inputs, scattered_kwargs

    @staticmethod
    def _output_device(output_device, values):
        if isinstance(output_device, int):
            if torch.cuda.is_available():
                return torch.device("cuda", output_device)
            return values[0].device
        return torch.device(output_device)

    def gather(self, outputs, output_device):
        if not outputs or not isinstance(outputs[0], dict):
            return super().gather(outputs, output_device)

        batch_sizes = []
        for output in outputs:
            logits = output.get("logits")
            batch_sizes.append(
                int(logits.shape[0])
                if torch.is_tensor(logits) and logits.ndim > 0
                else None
            )

        def merge(key, values):
            if not values or not all(torch.is_tensor(value) for value in values):
                return values[0]
            target = self._output_device(output_device, values)
            # Resolve known shared tensors by semantic key before shape-based
            # detection. A local batch of three must not make the three branch
            # temperatures look like a batch tensor.
            if key in self._SHARED_IDENTICAL_KEYS:
                moved = [value.to(target, non_blocking=True) for value in values]
                reference = moved[0]
                if not all(
                    value.shape == reference.shape
                    and torch.allclose(value, reference, atol=1e-5, rtol=1e-5)
                    for value in moved[1:]
                ):
                    raise RuntimeError(
                        f"DataParallel replicas produced inconsistent shared output {key!r}"
                    )
                return moved[0]
            is_batch_tensor = all(
                batch_size is not None
                and value.ndim > 0
                and value.shape[0] == batch_size
                for value, batch_size in zip(values, batch_sizes)
            )
            if is_batch_tensor:
                if all(value.is_cuda for value in values):
                    return parallel_gather(values, output_device, dim=self.dim)
                moved = [value.to(target, non_blocking=True) for value in values]
                return torch.cat(moved, dim=0)
            moved = [value.to(target, non_blocking=True) for value in values]
            if key in self._SHARED_WEIGHTED_MEAN_KEYS:
                valid_sizes = [size for size in batch_sizes if size is not None]
                if len(valid_sizes) == len(moved) and sum(valid_sizes) > 0:
                    total = sum(valid_sizes)
                    return sum(value * (size / total) for value, size in zip(moved, valid_sizes))
                return torch.stack(moved, dim=0).mean(dim=0)
            # Keep a conservative fallback for future shared metadata tensors.
            if all(value.shape == moved[0].shape for value in moved):
                return moved[0]
            return torch.cat(moved, dim=0)

        keys = outputs[0].keys()
        return {key: merge(key, [output[key] for output in outputs]) for key in keys}


def _model_signature(model) -> Dict[str, Any]:
    model = unwrap_model(model)
    prompt_learner = getattr(model, "prompt_learner", None)
    anatomy_fusion = getattr(model, "anatomy_fusion", None)
    return {
        "anatomy_descriptor_version": ANATOMY_DESCRIPTOR_VERSION,
        "class_names": list(getattr(model, "class_names", ())),
        "backbone_name": getattr(model, "backbone_name", None),
        "routing_mode": getattr(model, "routing_mode", None),
        "geometry_enabled": getattr(anatomy_fusion, "geometry_enabled", None),
        "geometry_fusion_mode": getattr(anatomy_fusion, "fusion_mode", None),
        "reliability_use_anatomy_quality": getattr(
            model,
            "reliability_use_anatomy_quality",
            None,
        ),
        "reliability_detach_visual_feature": getattr(
            model,
            "reliability_detach_visual_feature",
            None,
        ),
        "max_abs_raw_strength": getattr(model, "max_abs_raw_strength", None),
        "prompt_n_ctx": getattr(prompt_learner, "n_ctx", None),
        "prompt_prefix": getattr(prompt_learner, "prompt_prefix", None),
        "prompt_suffix_template": getattr(prompt_learner, "prompt_suffix_template", None),
    }


def _checkpoint_payload(model, epoch: int, stage: int, metrics: Optional[Dict[str, Any]] = None):
    model = unwrap_model(model)
    return {
        "schema_version": EMOTION_CHECKPOINT_SCHEMA_VERSION,
        "stage": stage,
        "epoch": epoch,
        "model": model.state_dict(),
        "class_names": model.class_names,
        "model_signature": _model_signature(model),
        "metrics": metrics or {},
    }


def parameter_report(model) -> Dict[str, Any]:
    model = unwrap_model(model)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    groups: Dict[str, int] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".emotion_adapter." in name:
            group = "adapters"
        elif name.startswith("anatomy_fusion."):
            group = "anatomy"
        elif name.startswith("prompt_learner.geometry_residual."):
            group = "prompt_geometry"
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


def load_emotion_checkpoint(
    model,
    checkpoint_path: str,
    strict: bool = True,
    *,
    allow_config_mismatch: bool = False,
    allow_untrained_stage2: bool = False,
):
    model = unwrap_model(model)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} must contain a state-dict mapping")
    schema_version = int(checkpoint.get("schema_version", 0) or 0)
    if schema_version > EMOTION_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} uses schema {schema_version}, newer than supported "
            f"schema {EMOTION_CHECKPOINT_SCHEMA_VERSION}"
        )
    checkpoint_classes = checkpoint.get("class_names")
    model_classes = tuple(getattr(model, "class_names", ()))
    if checkpoint_classes and model_classes and tuple(checkpoint_classes) != model_classes:
        raise RuntimeError(
            f"Checkpoint class order {tuple(checkpoint_classes)!r} does not match model "
            f"class order {model_classes!r}"
        )
    signature = checkpoint.get("model_signature") or {}
    if signature and not allow_config_mismatch:
        current_signature = _model_signature(model)
        mismatches = [
            f"{key}: checkpoint={value!r} current={current_signature.get(key)!r}"
            for key, value in signature.items()
            if value is not None
            and current_signature.get(key) is not None
            and value != current_signature.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} is incompatible with the current experiment config: "
                + "; ".join(mismatches)
            )
    if (
        int(checkpoint.get("stage", 0) or 0) == 1
        and not allow_untrained_stage2
    ):
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is Stage 1 only and cannot be used for Stage 2 inference. "
            "Train Stage 2 first or explicitly load it as a training initialization."
        )
    state_dict = checkpoint.get("model", checkpoint)
    dropped_shape_mismatches = []
    dropped_derived_buffers = []
    if not strict:
        current_state = model.state_dict()
        filtered_state = {}
        for key, value in state_dict.items():
            if key.endswith(_DERIVED_PROMPT_BUFFER_SUFFIXES):
                dropped_derived_buffers.append(key)
                continue
            if key in current_state and current_state[key].shape != value.shape:
                dropped_shape_mismatches.append(
                    f"{key}: checkpoint={tuple(value.shape)} current={tuple(current_state[key].shape)}"
                )
                continue
            filtered_state[key] = value
        state_dict = filtered_state
        missing_anatomy = sorted(
            key
            for key in current_state
            if key.startswith("anatomy_fusion.") and key not in state_dict
        )
        routing_mode = str(getattr(model, "routing_mode", "topk")).lower()
        if missing_anatomy and routing_mode != "topk" and not allow_untrained_stage2:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} does not contain a complete trained anatomy_fusion "
                f"for routing_mode={routing_mode!r}. Use a Stage 2 checkpoint, switch explicitly "
                "to MODEL.ROUTING.MODE=topk, or load only as a Stage 2 training initialization."
            )
        if missing_anatomy and allow_untrained_stage2:
            logging.getLogger("emotionclip.checkpoint").warning(
                "Checkpoint %s initializes Stage 2 with %s anatomy tensors absent; these modules "
                "must be trained before inference.",
                checkpoint_path,
                len(missing_anatomy),
            )
        missing_runtime = sorted(
            key
            for key in current_state
            if key.startswith(("classifier.", "fusion.", "reliability_head."))
            and key not in state_dict
        )
        if missing_runtime and not allow_untrained_stage2:
            missing_modules = sorted({key.split(".", 1)[0] for key in missing_runtime})
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} is missing trained Stage 2 runtime modules: "
                f"{', '.join(missing_modules)}. Use a complete Stage 2 checkpoint or load only "
                "as a training initialization."
            )
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if dropped_derived_buffers:
        logging.getLogger("emotionclip.checkpoint").info(
            "Regenerated prompt token buffers from the current prompt template instead of loading: %s",
            ", ".join(dropped_derived_buffers),
        )
    if dropped_shape_mismatches:
        logging.getLogger("emotionclip.checkpoint").warning(
            "Checkpoint %s has shape-mismatched tensors skipped during migration: %s",
            checkpoint_path,
            "; ".join(dropped_shape_mismatches),
        )
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
    geometry = {key: [] for key in ("geometry_features", "geometry_validity", "geometry_uncertainty", "region_quality")}
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            image_features = model(images=batch["images"], get_image=True)
            features.append(image_features.detach().cpu())
            labels.append(batch["labels"].detach().cpu())
            if "anatomy" in batch:
                for key in geometry:
                    geometry[key].append(batch["anatomy"][key].detach().cpu())
    result = {"features": torch.cat(features, dim=0), "labels": torch.cat(labels, dim=0)}
    if geometry["geometry_features"]:
        result.update({key: torch.cat(values, dim=0) for key, values in geometry.items()})
    return result


def evaluate_stage1_prompt_model(
    cfg, model, cached: Dict[str, torch.Tensor], *, use_geometry: bool = False
) -> Dict[str, Any]:
    """Evaluate only the Stage 1 contract: frozen image features vs learned prompts."""
    model.eval()
    device = torch.device(cfg["MODEL"]["DEVICE"])
    core_model = unwrap_model(model)
    with torch.no_grad():
        try:
            text_features = core_model.get_text_features(use_geometry=use_geometry)
        except TypeError:
            text_features = core_model.get_text_features()
        logits = core_model.logit_scale.exp().float() * cached["features"].to(device) @ text_features.t()
        probabilities = F.softmax(logits, dim=1)
    labels = cached["labels"].detach().cpu().tolist()
    probs = probabilities.detach().cpu().tolist()
    # Stage 1 has no trained reliability head; uncertainty is deliberately neutral.
    metrics = compute_fer_metrics(
        labels, probs, [0.0] * len(labels), core_model.class_names
    )
    metrics["stage"] = 1
    metrics["evaluation_split"] = "val"
    metrics["selection_split"] = "val"
    metrics["num_samples"] = len(labels)
    metrics["loss"] = float(F.cross_entropy(logits, cached["labels"].to(device)).detach().cpu())
    return metrics


def do_train_emotion_stage1(cfg, model, train_loader_stage1, optimizer, scheduler=None, val_loader=None):
    logger = logging.getLogger("emotionclip.train")
    device = torch.device(cfg["MODEL"]["DEVICE"])
    output_dir = cfg["OUTPUT_DIR"]
    stage_cfg = cfg["SOLVER"]["STAGE1"]
    log_period = int(stage_cfg.get("LOG_PERIOD", 20))

    model.to(device)
    core_model = unwrap_model(model)
    core_model.set_train_stage(1)
    initial_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    cached = precompute_stage1_features(model, train_loader_stage1, device)
    val_cached = precompute_stage1_features(model, val_loader, device) if val_loader is not None else None
    features = cached["features"].to(device)
    labels = cached["labels"].to(device)
    statistics_available = False
    if "geometry_features" in cached and hasattr(core_model, "set_class_geometry_statistics"):
        statistics = fit_class_geometry_statistics(
            cached["geometry_features"],
            cached["geometry_validity"],
            cached["geometry_uncertainty"],
            cached["region_quality"],
            cached["labels"],
            num_classes=core_model.num_classes,
            minimum_samples=int(stage_cfg.get("MIN_GEOMETRY_SAMPLES", 8)),
        )
        core_model.set_class_geometry_statistics(statistics)
        statistics_available = bool(statistics["quality"].sum() > 0)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(statistics, os.path.join(output_dir, "stage1_geometry_statistics.pt"))

    mode = str(stage_cfg.get("MODE", "base")).lower()
    if mode not in {"base", "geometry", "both"}:
        raise ValueError("SOLVER.STAGE1.MODE must be 'base', 'geometry', or 'both'")
    legacy_epochs = int(stage_cfg.get("MAX_EPOCHS", 20))
    base_epochs = int(stage_cfg.get("BASE_EPOCHS", legacy_epochs))
    geometry_epochs = int(stage_cfg.get("GEOMETRY_EPOCHS", legacy_epochs if mode == "geometry" else 0))
    phases = []
    if mode in {"base", "both"} and base_epochs > 0:
        phases.append(("base", base_epochs))
    if mode in {"geometry", "both"} and geometry_epochs > 0:
        if statistics_available:
            phases.append(("geometry", geometry_epochs))
        else:
            message = "Stage1 geometry phase requested but train split has no reliable anatomy statistics"
            if not bool(cfg.get("DATASETS", {}).get("ALLOW_ANATOMY_FALLBACK", False)):
                raise RuntimeError(
                    message
                    + ". Build compatible landmark artifacts or explicitly enable anatomy fallback."
                )
            logger.warning("%s; explicit anatomy fallback enabled, so Stage1B is skipped", message)
    total_epochs = sum(epochs for _, epochs in phases)

    batch_size = int(stage_cfg.get("IMS_PER_BATCH", 64))
    log_training_event(
        logger,
        "Stage1 start",
        epoch_total=total_epochs,
        samples=labels.numel(),
        batch_size=batch_size,
        steps_per_epoch=max(1, (labels.shape[0] + batch_size - 1) // batch_size),
        lr=optimizer.param_groups[0]["lr"],
        mode=mode,
        geometry_statistics=statistics_available,
    )

    global_epoch = 0
    selection_metric = str(stage_cfg.get("SELECTION_METRIC", "macro_f1"))
    valid_selection_metrics = {"macro_f1", "balanced_accuracy", "accuracy", "loss"}
    if selection_metric not in valid_selection_metrics:
        raise ValueError(
            f"Unsupported SOLVER.STAGE1.SELECTION_METRIC={selection_metric!r}; "
            f"expected one of {sorted(valid_selection_metrics)}"
        )
    best_metric = float("inf") if selection_metric == "loss" else float("-inf")
    best_metrics = None
    early_stopping_patience = int(stage_cfg.get("EARLY_STOPPING_PATIENCE", 0))
    for phase, phase_epochs in phases:
        phase_epochs_without_improvement = 0
        # In MODE=both, geometry must start from the best validated base prompt,
        # never from an overfit last-base state.
        if phase == "geometry" and best_metrics is not None:
            load_emotion_checkpoint(
                model,
                os.path.join(output_dir, "best_emotionclip_stage1.pth"),
                strict=False,
                allow_untrained_stage2=True,
            )
            # The restored best-base weights must not inherit Adam moments or
            # a decayed LR schedule from the last-base trajectory.
            optimizer.state.clear()
            for group, initial_lr in zip(optimizer.param_groups, initial_lrs):
                group["lr"] = initial_lr
            if scheduler is not None:
                scheduler.base_lrs = list(initial_lrs)
                scheduler.last_epoch = -1
                scheduler._step_count = 0
                if hasattr(scheduler, "T_max"):
                    scheduler.T_max = max(1, phase_epochs)
        if hasattr(core_model, "set_stage1_phase"):
            core_model.set_stage1_phase(phase)
        base_text_features = None
        if phase == "geometry" and hasattr(core_model, "get_text_features"):
            with torch.no_grad():
                base_text_features = core_model.get_text_features(use_geometry=False).detach()
        for phase_epoch in range(1, phase_epochs + 1):
            global_epoch += 1
            start_time = time.time()
            model.train()
            total_loss = 0.0
            total_acc = 0.0
            total_conf = 0.0
            total_shift = 0.0
            total_semantic = 0.0
            steps = 0
            permutation = torch.randperm(labels.shape[0], device=device)
            epoch_starts = range(0, labels.shape[0], batch_size)
            progress = _progress(
                epoch_starts,
                cfg=cfg,
                desc=f"Stage1-{phase} {phase_epoch}/{phase_epochs}",
                total=len(epoch_starts),
            )
            for start in progress:
                indices = permutation[start : start + batch_size]
                batch_features = features[indices]
                batch_labels = labels[indices]
                optimizer.zero_grad()
                if hasattr(core_model, "prompt_learner") and hasattr(core_model.prompt_learner, "geometry_residual"):
                    text_features = core_model.get_text_features(use_geometry=phase == "geometry")
                else:
                    text_features = get_shared_text_features(model)
                logits = core_model.logit_scale.exp().float() * batch_features @ text_features.t()
                classification_loss = F.cross_entropy(logits, batch_labels)
                shift_loss = classification_loss.new_zeros(())
                semantic_loss = classification_loss.new_zeros(())
                if phase == "geometry":
                    shift_loss = core_model.prompt_learner.geometry_residual.regularization()
                    semantic_loss = (1.0 - F.cosine_similarity(
                        text_features, base_text_features, dim=-1
                    ).mean()).clamp_min(0.0)
                loss = (
                    classification_loss
                    + float(stage_cfg.get("LAMBDA_SHIFT", 0.01)) * shift_loss
                    + float(stage_cfg.get("LAMBDA_SEMANTIC", 0.1)) * semantic_loss
                )
                if not bool(torch.isfinite(loss)):
                    failure_path = _write_training_failure(
                        output_dir,
                        epoch=global_epoch,
                        batch_index=steps + 1,
                        reason=f"non-finite Stage 1 {phase} loss",
                        losses={
                            "loss": loss,
                            "classification": classification_loss,
                            "shift": shift_loss,
                            "semantic": semantic_loss,
                        },
                        outputs={"logits": logits},
                        model=model,
                    )
                    raise FloatingPointError(
                        f"Non-finite Stage 1 loss in phase={phase} epoch={phase_epoch}; "
                        f"diagnostic={failure_path}"
                    )
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    _trainable_parameters(model),
                    max_norm=float(stage_cfg.get("MAX_GRAD_NORM", 1.0)),
                    error_if_nonfinite=False,
                )
                if not bool(torch.isfinite(grad_norm)):
                    failure_path = _write_training_failure(
                        output_dir,
                        epoch=global_epoch,
                        batch_index=steps + 1,
                        reason=f"non-finite Stage 1 {phase} gradient norm",
                        losses={"loss": loss},
                        outputs={"logits": logits},
                        model=model,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"Non-finite Stage 1 gradients in phase={phase} epoch={phase_epoch}; "
                        f"diagnostic={failure_path}"
                    )
                optimizer.step()
                nonfinite_parameters = _nonfinite_parameter_names(model)
                if nonfinite_parameters:
                    failure_path = _write_training_failure(
                        output_dir,
                        epoch=global_epoch,
                        batch_index=steps + 1,
                        reason=f"non-finite Stage 1 {phase} parameters after optimizer step",
                        losses={"loss": loss},
                        outputs={"logits": logits},
                        model=model,
                    )
                    raise FloatingPointError(
                        f"Non-finite Stage 1 parameters after optimizer step: "
                        f"{nonfinite_parameters[:5]}; diagnostic={failure_path}"
                    )

                total_loss += float(loss.detach().cpu())
                total_acc += _batch_accuracy(logits, batch_labels)
                total_conf += _batch_confidence(F.softmax(logits.detach(), dim=1))
                total_shift += float(shift_loss.detach().cpu())
                total_semantic += float(semantic_loss.detach().cpu())
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
                        phase=phase,
                        epoch=f"{phase_epoch}/{phase_epochs}",
                        step=f"{steps}/{max(1, (labels.shape[0] + batch_size - 1) // batch_size)}",
                        loss=total_loss / max(steps, 1),
                        acc=total_acc / max(steps, 1),
                        conf=total_conf / max(steps, 1),
                        lr=optimizer.param_groups[0]["lr"],
                    )

            if scheduler is not None:
                scheduler.step()
            epoch_metrics = {
                "event": "Stage1 done",
                "stage": 1,
                "epoch": global_epoch,
                "epoch_total": total_epochs,
                "loss": total_loss / max(steps, 1),
                "acc": total_acc / max(steps, 1),
                "conf": total_conf / max(steps, 1),
                "lr": optimizer.param_groups[0]["lr"],
                "time_sec": time.time() - start_time,
            }
            log_training_event(
                logger,
                "Stage1 done",
                phase=phase,
                epoch=f"{phase_epoch}/{phase_epochs}",
                loss=epoch_metrics["loss"],
                shift=f"{total_shift / max(steps, 1):.6e}",
                semantic=f"{total_semantic / max(steps, 1):.6e}",
                acc=epoch_metrics["acc"],
                conf=epoch_metrics["conf"],
                lr=epoch_metrics["lr"],
                time_sec=epoch_metrics["time_sec"],
            )
            _record_training_epoch(cfg, epoch_metrics)
            save_checkpoint(model, output_dir, "last_emotionclip_stage1.pth", global_epoch, stage=1)
            if val_cached is not None and (
                global_epoch % max(1, int(stage_cfg.get("EVAL_PERIOD", 1))) == 0
                or global_epoch == total_epochs
            ):
                metrics = evaluate_stage1_prompt_model(
                    cfg, model, val_cached, use_geometry=(phase == "geometry")
                )
                metrics["epoch"] = global_epoch
                metrics["phase"] = phase
                metrics_path = os.path.join(output_dir, f"stage1_metrics_epoch_{global_epoch}.json")
                with open(metrics_path, "w", encoding="utf-8") as handle:
                    json.dump(metrics, handle, indent=2)
                _append_csv_row(
                    _artifact_path(cfg, "STAGE1_VALIDATION_CSV", "stage1_validation_metrics.csv"),
                    _VALIDATION_COLUMNS,
                    {**metrics, "epoch": global_epoch, "epoch_total": total_epochs},
                )
                score = float(metrics[selection_metric])
                is_better = (
                    score < best_metric - float(stage_cfg.get("MIN_DELTA", 0.0))
                    if selection_metric == "loss"
                    else score > best_metric + float(stage_cfg.get("MIN_DELTA", 0.0))
                )
                if is_better:
                    best_metric = score
                    best_metrics = metrics
                    phase_epochs_without_improvement = 0
                    save_checkpoint(
                        model, output_dir, "best_emotionclip_stage1.pth", global_epoch, stage=1, metrics=metrics
                    )
                else:
                    phase_epochs_without_improvement += 1
                if (
                    early_stopping_patience > 0
                    and phase_epochs_without_improvement >= early_stopping_patience
                ):
                    log_training_event(
                        logger,
                        "Stage1 early stop",
                        phase=phase,
                        epoch=phase_epoch,
                        patience=early_stopping_patience,
                        selection_metric=selection_metric,
                        best_value=best_metric,
                    )
                    break

    if val_cached is not None and best_metrics is not None:
        load_emotion_checkpoint(
            model,
            os.path.join(output_dir, "best_emotionclip_stage1.pth"),
            strict=False,
            allow_untrained_stage2=True,
        )
        with open(os.path.join(output_dir, "stage1_selection.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "stage": 1,
                    "selection_split": "val",
                    "selection_metric": selection_metric,
                    "best_epoch": best_metrics.get("epoch"),
                    "best_phase": best_metrics.get("phase"),
                    "best_value": best_metric,
                    "checkpoint": "best_emotionclip_stage1.pth",
                },
                handle,
                indent=2,
            )

    if hasattr(core_model, "save_stage1_descriptors"):
        core_model.save_stage1_descriptors(os.path.join(output_dir, "stage1_text_descriptors.pth"))
    return best_metrics


def evaluate_emotion_model(
    cfg,
    model,
    data_loader,
    text_features: Optional[torch.Tensor] = None,
    include_analysis_outputs: bool = False,
) -> Dict[str, Any]:
    device = torch.device(cfg["MODEL"]["DEVICE"])
    labels = []
    probabilities = []
    uncertainties = []
    strengths = []
    class_ambiguities = []
    region_disagreements = []
    region_disagreement_valid = []
    region_qualities = []
    image_paths = []
    model.eval()
    with torch.no_grad():
        if text_features is None:
            text_features = get_shared_text_features(model)
        text_features = text_features.to(device)
        for batch in data_loader:
            batch = _batch_to_device(batch, device)
            outputs = _forward_batch(model, batch, text_features=text_features)
            labels.extend(batch["labels"].detach().cpu().tolist())
            probabilities.extend(outputs["probabilities"].detach().cpu().tolist())
            uncertainties.extend(outputs["uncertainty"].detach().cpu().tolist())
            strengths.extend(outputs["strength"].detach().cpu().tolist())
            if "class_ambiguity" in outputs:
                class_ambiguities.extend(outputs["class_ambiguity"].detach().cpu().tolist())
            if "region_disagreement" in outputs:
                region_disagreements.extend(outputs["region_disagreement"].detach().cpu().tolist())
                region_disagreement_valid.extend(
                    outputs["region_disagreement_valid"].detach().cpu().tolist()
                )
                region_qualities.extend(outputs["region_quality"].detach().cpu().tolist())
            image_paths.extend(batch["image_paths"])
    metrics = compute_fer_metrics(labels, probabilities, uncertainties, unwrap_model(model).class_names)
    metrics["num_samples"] = len(labels)
    metrics["image_paths"] = image_paths
    if probabilities:
        probabilities_tensor = torch.tensor(probabilities, dtype=torch.float32)
        metrics["avg_confidence"] = float(probabilities_tensor.max(dim=1).values.mean())
    else:
        metrics["avg_confidence"] = 0.0
    metrics["avg_uncertainty"] = float(sum(uncertainties) / max(len(uncertainties), 1))
    metrics["avg_strength"] = float(sum(strengths) / max(len(strengths), 1))
    metrics["avg_class_ambiguity"] = float(
        sum(class_ambiguities) / max(len(class_ambiguities), 1)
    )
    valid_disagreements = [
        value for value, valid in zip(region_disagreements, region_disagreement_valid) if valid
    ]
    metrics["avg_region_disagreement"] = float(
        sum(valid_disagreements) / max(len(valid_disagreements), 1)
    )
    metrics["region_disagreement_valid_rate"] = float(
        sum(bool(value) for value in region_disagreement_valid)
        / max(len(region_disagreement_valid), 1)
    )
    if include_analysis_outputs:
        predictions = torch.tensor(probabilities).argmax(dim=-1).tolist() if probabilities else []
        metrics["analysis_outputs"] = [
            {
                "image_path": path,
                "label": int(label),
                "prediction": int(prediction),
                "class_ambiguity": float(class_ambiguity),
                "region_disagreement": float(region_disagreement),
                "region_disagreement_valid": bool(disagreement_valid),
                "extrinsic_unreliability": float(extrinsic_unreliability),
                "strength": float(strength),
                "region_quality": [float(value) for value in quality],
            }
            for path, label, prediction, class_ambiguity, region_disagreement, disagreement_valid,
            extrinsic_unreliability, strength, quality in zip(
                image_paths,
                labels,
                predictions,
                class_ambiguities,
                region_disagreements,
                region_disagreement_valid,
                uncertainties,
                strengths,
                region_qualities,
            )
        ]
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
    metrics = evaluate_emotion_model(
        cfg,
        model,
        test_loader,
        include_analysis_outputs=bool(cfg.get("TEST", {}).get("SAVE_ANALYSIS_OUTPUTS", True)),
    )
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
    lambda_routing = float(stage_cfg.get("LAMBDA_ROUTING", 0.0))
    core_model = unwrap_model(model)
    routing_mode = str(getattr(core_model, "routing_mode", "topk")).lower()
    if routing_mode != "hybrid" and lambda_routing != 0.0:
        logger.info(
            "Disabling routing loss for pure %s routing ablation; "
            "anatomy-supervised free attention is only defined for hybrid routing.",
            routing_mode,
        )
        lambda_routing = 0.0
    corruption_cfg = stage_cfg.get("CORRUPTION", {})
    lambda_reliability_ranking = float(corruption_cfg.get("LAMBDA_RANKING", 0.0))
    reliability_ranking_warmup_epochs = max(0, int(corruption_cfg.get("WARMUP_EPOCHS", 0)))
    accumulation_steps = max(1, int(stage_cfg.get("GRADIENT_ACCUMULATION_STEPS", 1)))
    amp_enabled = bool(stage_cfg.get("AMP_ENABLED", True)) and device.type == "cuda"
    max_grad_norm = float(stage_cfg.get("MAX_GRAD_NORM", 1.0))
    fail_on_nonfinite = bool(stage_cfg.get("FAIL_ON_NONFINITE", True))
    scaler = _make_grad_scaler(amp_enabled)
    anneal_epochs = max(
        1,
        int(stage_cfg.get("RELIABILITY_WARMUP_EPOCHS", stage_cfg.get("EDL_ANNEALING_EPOCHS", 10))),
    )

    model.to(device)
    core_model.set_train_stage(2)
    params = parameter_report(model)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "trainable_parameters.json"), "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)
    best_macro_f1 = -1.0
    best_metrics = None
    last_metrics = None
    early_stopping_patience = int(stage_cfg.get("EARLY_STOPPING_PATIENCE", 0))
    epochs_without_improvement = 0
    log_training_event(
        logger,
        "Stage2 start",
        epoch_total=max_epochs,
        samples=len(train_loader.dataset) if hasattr(train_loader, "dataset") else "unknown",
        batch_size=stage_cfg.get("IMS_PER_BATCH", "unknown"),
        gradient_accumulation_steps=accumulation_steps,
        effective_batch_size=int(stage_cfg.get("IMS_PER_BATCH", 1)) * accumulation_steps,
        amp_enabled=amp_enabled,
        max_grad_norm=max_grad_norm,
        steps_per_epoch=len(train_loader),
        lr=optimizer.param_groups[0]["lr"],
        beta_align=beta_align,
        lambda_unc=lambda_unc,
        lambda_routing=lambda_routing,
        lambda_reliability_ranking=lambda_reliability_ranking,
        reliability_ranking_warmup_epochs=reliability_ranking_warmup_epochs,
        max_abs_raw_strength=getattr(core_model, "max_abs_raw_strength", "unknown"),
        numeric_heads_dtype="float32",
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
        text_features = get_shared_text_features(model).detach()
        total_loss = 0.0
        total_cls = 0.0
        total_align = 0.0
        total_unc = 0.0
        total_gate_loss = 0.0
        total_temperature_loss = 0.0
        total_routing_loss = 0.0
        total_reliability_ranking = 0.0
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
        optimizer_steps = 0
        last_grad_norm = 0.0
        anneal = min(1.0, epoch / anneal_epochs)
        effective_reliability_ranking = (
            lambda_reliability_ranking if epoch > reliability_ranking_warmup_epochs else 0.0
        )
        progress = _progress(train_loader, cfg=cfg, desc=f"Stage2 {epoch}/{max_epochs}", total=len(train_loader))
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(progress, start=1):
            batch = _batch_to_device(batch, device)
            samples_seen += int(batch["labels"].shape[0])
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = _forward_batch(model, batch, text_features=text_features)
                corrupted_outputs = None
                if (
                    effective_reliability_ranking > 0
                    and float(torch.rand((), device=batch["images"].device))
                    < float(corruption_cfg.get("PROBABILITY", 1.0))
                ):
                    corrupted_batch = dict(batch)
                    corrupted_images, occlusion_mask = corrupt_images_for_reliability(
                        batch["images"],
                        noise_std=float(corruption_cfg.get("NOISE_STD", 0.08)),
                        occlusion_ratio=float(corruption_cfg.get("OCCLUSION_RATIO", 0.2)),
                        return_occlusion_mask=True,
                    )
                    corrupted_batch["images"] = corrupted_images
                    corrupted_anatomy = corrupt_anatomy_for_reliability(
                        batch.get("anatomy"),
                        occlusion_mask,
                    )
                    if corrupted_anatomy is not None:
                        corrupted_batch["anatomy"] = corrupted_anatomy
                    corrupted_outputs = _forward_batch(model, corrupted_batch, text_features=text_features)
                output_error = _model_output_error(outputs, model=model)
                if output_error is None and corrupted_outputs is not None:
                    corrupted_error = _model_output_error(corrupted_outputs, model=model)
                    if corrupted_error is not None:
                        output_error = f"corrupted forward: {corrupted_error}"
                if output_error is None:
                    losses = emotion_stage2_loss(
                        outputs,
                        batch["labels"],
                        beta_align=beta_align,
                        lambda_unc=lambda_unc,
                        edl_annealing=anneal,
                        reliability_target=reliability_target,
                        lambda_gate=lambda_gate,
                        lambda_temperature=lambda_temperature,
                        lambda_routing=lambda_routing,
                        corrupted_outputs=corrupted_outputs,
                        lambda_reliability_ranking=effective_reliability_ranking,
                        reliability_ranking_margin=float(corruption_cfg.get("RANKING_MARGIN", 1.0)),
                    )
            if output_error is not None:
                failure_path = _write_training_failure(
                    output_dir,
                    epoch=epoch,
                    batch_index=batch_index,
                    reason=output_error,
                    outputs=corrupted_outputs if output_error.startswith("corrupted forward:") else outputs,
                    model=model,
                    batch=batch,
                )
                optimizer.zero_grad(set_to_none=True)
                message = (
                    f"Invalid Stage 2 output at epoch={epoch} batch={batch_index}: "
                    f"{output_error}; diagnostic={failure_path}"
                )
                if fail_on_nonfinite:
                    raise FloatingPointError(message)
                logger.error("%s; batch skipped", message)
                continue
            if not _losses_are_finite(losses):
                failure_path = _write_training_failure(
                    output_dir,
                    epoch=epoch,
                    batch_index=batch_index,
                    reason="non-finite loss",
                    losses=losses,
                    outputs=outputs,
                    model=model,
                    batch=batch,
                )
                optimizer.zero_grad(set_to_none=True)
                message = f"Non-finite Stage 2 loss at epoch={epoch} batch={batch_index}; diagnostic={failure_path}"
                if fail_on_nonfinite:
                    raise FloatingPointError(message)
                logger.error("%s; batch skipped", message)
                continue

            window_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps + 1
            window_size = min(accumulation_steps, len(train_loader) - window_start + 1)
            scaler.scale(losses["loss"] / window_size).backward()
            should_step = batch_index % accumulation_steps == 0 or batch_index == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    _trainable_parameters(model),
                    max_norm=max_grad_norm,
                    error_if_nonfinite=False,
                )
                if not bool(torch.isfinite(grad_norm)):
                    failure_path = _write_training_failure(
                        output_dir,
                        epoch=epoch,
                        batch_index=batch_index,
                        reason="non-finite gradient norm",
                        losses=losses,
                        outputs=outputs,
                        model=model,
                        batch=batch,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    message = (
                        f"Non-finite Stage 2 gradients at epoch={epoch} batch={batch_index}; "
                        f"diagnostic={failure_path}"
                    )
                    if fail_on_nonfinite:
                        raise FloatingPointError(message)
                    logger.error("%s; optimizer step skipped", message)
                    scaler.update()
                    continue
                scaler.step(optimizer)
                nonfinite_parameters = _nonfinite_parameter_names(model)
                if nonfinite_parameters:
                    failure_path = _write_training_failure(
                        output_dir,
                        epoch=epoch,
                        batch_index=batch_index,
                        reason="non-finite parameters after optimizer step",
                        losses=losses,
                        outputs=outputs,
                        model=model,
                        batch=batch,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"Non-finite Stage 2 parameters after optimizer step: "
                        f"{nonfinite_parameters[:5]}; diagnostic={failure_path}"
                    )
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                last_grad_norm = float(grad_norm.detach().cpu())
            batch_loss = float(losses["loss"].detach().cpu())
            batch_cls = float(losses["classification"].detach().cpu())
            batch_align = float(losses["alignment"].detach().cpu())
            batch_unc = float(losses["uncertainty"].detach().cpu())
            batch_gate_loss = float(losses["gate"].detach().cpu())
            batch_temperature_loss = float(losses["temperature"].detach().cpu())
            batch_routing_loss = float(losses["routing"].detach().cpu())
            batch_reliability_ranking = float(losses["reliability_ranking"].detach().cpu())
            batch_pred_unc = _batch_uncertainty(outputs["uncertainty"])
            batch_conf = _batch_confidence(outputs["probabilities"])
            batch_acc = _batch_accuracy(outputs["logits"], batch["labels"])
            total_loss += batch_loss
            total_cls += batch_cls
            total_align += batch_align
            total_unc += batch_unc
            total_gate_loss += batch_gate_loss
            total_temperature_loss += batch_temperature_loss
            total_routing_loss += batch_routing_loss
            total_reliability_ranking += batch_reliability_ranking
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
                logits_for_log = outputs["logits"].detach().float()
                raw_strength_for_log = outputs["raw_strength"].detach().float()
                raw_strength_unbounded_for_log = outputs.get("raw_strength_unbounded")
                probabilities_for_log = outputs["probabilities"].detach().float()
                probability_sums_for_log = probabilities_for_log.sum(dim=-1)
                log_training_event(
                    logger,
                    "Stage2 train",
                    epoch=f"{epoch}/{max_epochs}",
                    step=f"{steps}/{len(train_loader)}",
                    loss=total_loss / max(steps, 1),
                    cls=total_cls / max(steps, 1),
                    align=total_align / max(steps, 1),
                    unc_loss=total_unc / max(steps, 1),
                    routing_loss=batch_routing_loss,
                    temperature_loss=batch_temperature_loss,
                    reliability_ranking_loss=batch_reliability_ranking,
                    reliability_ranking_weight=effective_reliability_ranking,
                    pred_unc=total_pred_unc / max(steps, 1),
                    conf=total_conf / max(steps, 1),
                    acc=total_acc / max(steps, 1),
                    anneal=anneal,
                    logit_min=float(logits_for_log.min().cpu()),
                    logit_max=float(logits_for_log.max().cpu()),
                    raw_strength_min=float(raw_strength_for_log.min().cpu()),
                    raw_strength_max=float(raw_strength_for_log.max().cpu()),
                    raw_strength_unbounded_min=(
                        float(raw_strength_unbounded_for_log.detach().float().min().cpu())
                        if torch.is_tensor(raw_strength_unbounded_for_log)
                        else "unavailable"
                    ),
                    raw_strength_unbounded_max=(
                        float(raw_strength_unbounded_for_log.detach().float().max().cpu())
                        if torch.is_tensor(raw_strength_unbounded_for_log)
                        else "unavailable"
                    ),
                    probability_min=float(probabilities_for_log.min().cpu()),
                    probability_max=float(probabilities_for_log.max().cpu()),
                    probability_sum_min=float(probability_sums_for_log.min().cpu()),
                    probability_sum_max=float(probability_sums_for_log.max().cpu()),
                    branch_temperatures=outputs.get("branch_temperatures", []).detach().float().cpu().tolist()
                    if torch.is_tensor(outputs.get("branch_temperatures"))
                    else [],
                    grad_norm=last_grad_norm,
                    optimizer_step=optimizer_steps,
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
            "routing_loss": total_routing_loss / max(steps, 1),
            "reliability_ranking_loss": total_reliability_ranking / max(steps, 1),
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
            "optimizer_steps": optimizer_steps,
            "gradient_accumulation_steps": accumulation_steps,
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
            optimizer_steps=epoch_metrics["optimizer_steps"],
            gradient_accumulation_steps=epoch_metrics["gradient_accumulation_steps"],
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
                epochs_without_improvement = 0
                save_checkpoint(model, output_dir, "best_emotionclip.pth", epoch, stage=2, metrics=metrics)
            else:
                epochs_without_improvement += 1
            last_metrics = metrics

        save_checkpoint(model, output_dir, "last_emotionclip.pth", epoch, stage=2, metrics=last_metrics)
        if (
            val_loader is not None
            and early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            log_training_event(
                logger,
                "Stage2 early stop",
                epoch=epoch,
                patience=early_stopping_patience,
                best_macro_f1=best_macro_f1,
            )
            break

    return best_metrics
