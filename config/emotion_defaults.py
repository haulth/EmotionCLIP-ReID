import copy
from typing import Any, Dict, Iterable, List, MutableMapping

import torch
import yaml


_DEFAULT = {
    "MODEL": {
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "NAME": "ViT-B-16",
        "STRIDE_SIZE": [16, 16],
        "EMOTION": {
            "N_CTX": 4,
            "ADAPTER_DIM": 64,
            "ADAPTER_DROPOUT": 0.0,
            "TOPK_PATCHES": 5,
            "GLOBAL_WEIGHT": 1.0,
            "LOCAL_WEIGHT": 0.5,
            "CLASSIFIER_WEIGHT": 1.0,
            "TRAIN_LAST_BLOCKS": 0,
            "STAGE1_WEIGHT": "",
        },
        "ANATOMY_PROMPT": {
            # Anatomy is the default contract; explicit ablation configs may opt out.
            "MODE": "legacy",
            "HIDDEN_DIM": 32,
            "GATE_INIT": -4.0,
        },
        "ROUTING": {
            "MODE": "hybrid",
            "SIGMA": 0.08,
        },
        "GEOMETRY": {
            "ENABLED": True,
            "FUSION_MODE": "gated_residual",
            "HIDDEN_DIM": 64,
            "IMPORTANCE_HIDDEN_DIM": 128,
            "GATE_INIT": -4.0,
        },
        "REGION_DISAGREEMENT": {
            "MIN_REGION_QUALITY": 0.2,
            "MIN_REGIONS": 2,
            "MIN_EFFECTIVE_REGIONS": 1.5,
        },
        "UNCERTAINTY": {
            "MODE": "decoupled",
            "USE_ANATOMY_QUALITY": True,
            "HIDDEN_DIM": 128,
            "DROPOUT": 0.1,
            "DETACH_CLASS_PROB": True,
            "DETACH_VISUAL_FEATURE": True,
            "MAX_STRENGTH": 100.0,
        },
        "FUSION": {
            "GATE_MODE": "fixed",
            "SCALE_MODE": "temperature",
            "GATE_HIDDEN_DIM": 128,
            "GATE_DROPOUT": 0.1,
            "MIN_TEMPERATURE": 0.05,
            "MAX_TEMPERATURE": 20.0,
            "INITIAL_TEMPERATURES": [0.1, 1.0, 1.0],
            "LEARN_TEMPERATURES": True,
        },
    },
    "INPUT": {
        "SIZE_TRAIN": [224, 224],
        "SIZE_TEST": [224, 224],
        "PROB": 0.5,
        "COLOR_JITTER": 0.05,
        "PIXEL_MEAN": [0.48145466, 0.4578275, 0.40821073],
        "PIXEL_STD": [0.26862954, 0.26130258, 0.27577711],
    },
    "DATASETS": {
        "MANIFEST": "",
        "ROOT_DIR": "",
        "STRICT_SPLIT_LEAKAGE": True,
        "REQUIRE_VAL": True,
        "REQUIRE_TEST": False,
        "REQUIRE_ANATOMY": True,
        "MIN_ANATOMY_COVERAGE": 0.8,
        "ALLOW_ANATOMY_FALLBACK": False,
    },
    "DATALOADER": {
        "NUM_WORKERS": 0,
        "PIN_MEMORY": False,
    },
    "SOLVER": {
        "SEED": 1234,
        "STAGE1": {
            "IMS_PER_BATCH": 64,
            "MAX_EPOCHS": 20,
            "MODE": "base",
            "BASE_EPOCHS": 20,
            "GEOMETRY_EPOCHS": 0,
            "MIN_GEOMETRY_SAMPLES": 8,
            "LAMBDA_SHIFT": 0.01,
            "LAMBDA_SEMANTIC": 0.1,
            "BASE_LR": 3.5e-4,
            "WEIGHT_DECAY": 1.0e-4,
            "CHECKPOINT_PERIOD": 10,
            "LOG_PERIOD": 20,
            "EVAL_PERIOD": 1,
            "SELECTION_METRIC": "macro_f1",
            "MIN_DELTA": 0.0,
        },
        "STAGE2": {
            "IMS_PER_BATCH": 32,
            "MAX_EPOCHS": 30,
            "BASE_LR": 5.0e-6,
            "WEIGHT_DECAY": 1.0e-4,
            "CHECKPOINT_PERIOD": 10,
            "EVAL_PERIOD": 1,
            "LOG_PERIOD": 20,
            "BETA_ALIGN": 0.5,
            "LAMBDA_UNC": 0.05,
            "EDL_ANNEALING_EPOCHS": 10,
            "LAMBDA_RELIABILITY": None,
            "RELIABILITY_WARMUP_EPOCHS": 10,
            "RELIABILITY_TARGET": "correctness",
            "LAMBDA_GATE": 0.01,
            "LAMBDA_TEMPERATURE": 0.001,
            "LAMBDA_ROUTING": 0.0,
            "CORRUPTION": {
                "LAMBDA_RANKING": 0.05,
                "RANKING_MARGIN": 1.0,
                "PROBABILITY": 0.5,
                "NOISE_STD": 0.08,
                "OCCLUSION_RATIO": 0.2,
            },
        },
    },
    "TEST": {
        "IMS_PER_BATCH": 64,
        "WEIGHT": "",
        "EVALUATE_AFTER_TRAIN": False,
        "OUTPUT_FILE": "test_metrics.json",
        "SAVE_ANALYSIS_OUTPUTS": True,
    },
    "TRAIN": {
        "RUN_ID": "",
        "RUN_STAGE1": True,
        "RUN_STAGE2": True,
        "PROGRESS_BAR": "auto",
    },
    "OUTPUT_DIR": "outputs/emotionclip",
}


_CLASS_GEOMETRY_PROMPT_MODES = {"median", "median_mad", "quality", "shuffled"}


def anatomy_requirement_reasons(cfg: MutableMapping[str, Any]) -> List[str]:
    """Return the configured features that require usable anatomy evidence."""
    reasons: List[str] = []
    model_cfg = cfg.get("MODEL", {})
    data_cfg = cfg.get("DATASETS", {})
    train_cfg = cfg.get("TRAIN", {})
    stage1_cfg = cfg.get("SOLVER", {}).get("STAGE1", {})
    routing_mode = str(model_cfg.get("ROUTING", {}).get("MODE", "topk")).lower()
    prompt_mode = str(model_cfg.get("ANATOMY_PROMPT", {}).get("MODE", "legacy")).lower()

    if bool(data_cfg.get("REQUIRE_ANATOMY", False)):
        reasons.append("DATASETS.REQUIRE_ANATOMY")
    if routing_mode in {"anatomy", "hybrid"}:
        reasons.append(f"MODEL.ROUTING.MODE={routing_mode}")
    if routing_mode != "topk" and bool(model_cfg.get("GEOMETRY", {}).get("ENABLED", False)):
        reasons.append("MODEL.GEOMETRY.ENABLED")
    if bool(model_cfg.get("UNCERTAINTY", {}).get("USE_ANATOMY_QUALITY", False)):
        reasons.append("MODEL.UNCERTAINTY.USE_ANATOMY_QUALITY")

    stage1_mode = str(stage1_cfg.get("MODE", "base")).lower()
    if (
        bool(train_cfg.get("RUN_STAGE1", True))
        and stage1_mode in {"geometry", "both"}
        and prompt_mode in _CLASS_GEOMETRY_PROMPT_MODES
    ):
        reasons.append(f"Stage1 {prompt_mode} class geometry statistics")
    return list(dict.fromkeys(reasons))


def get_default_emotion_cfg() -> Dict[str, Any]:
    return copy.deepcopy(_DEFAULT)


def _deep_update(base: MutableMapping[str, Any], update: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in update.items():
        if isinstance(value, MutableMapping) and isinstance(base.get(key), MutableMapping):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _set_by_dotted_key(cfg: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def validate_emotion_cfg(cfg: MutableMapping[str, Any]) -> None:
    data_cfg = cfg.get("DATASETS", {})
    minimum_coverage = float(data_cfg.get("MIN_ANATOMY_COVERAGE", 0.8))
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("DATASETS.MIN_ANATOMY_COVERAGE must be between 0 and 1")
    stage1_mode = str(cfg.get("SOLVER", {}).get("STAGE1", {}).get("MODE", "base")).lower()
    if stage1_mode not in {"base", "geometry", "both"}:
        raise ValueError("SOLVER.STAGE1.MODE must be 'base', 'geometry', or 'both'")
    disagreement = cfg.get("MODEL", {}).get("REGION_DISAGREEMENT", {})
    min_region_quality = float(disagreement.get("MIN_REGION_QUALITY", 0.2))
    if not 0.0 <= min_region_quality <= 1.0:
        raise ValueError("MODEL.REGION_DISAGREEMENT.MIN_REGION_QUALITY must be between 0 and 1")
    if int(disagreement.get("MIN_REGIONS", 2)) < 2:
        raise ValueError("MODEL.REGION_DISAGREEMENT.MIN_REGIONS must be at least 2")
    if float(disagreement.get("MIN_EFFECTIVE_REGIONS", 1.5)) < 1.0:
        raise ValueError("MODEL.REGION_DISAGREEMENT.MIN_EFFECTIVE_REGIONS must be at least 1")


def load_emotion_cfg(config_file: str = "", opts: Iterable[str] = ()) -> Dict[str, Any]:
    cfg = get_default_emotion_cfg()
    if config_file:
        with open(config_file, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        _deep_update(cfg, loaded)

    opts = list(opts or [])
    if len(opts) % 2 != 0:
        raise ValueError("Command-line opts must be KEY VALUE pairs")
    for key, value in zip(opts[0::2], opts[1::2]):
        _set_by_dotted_key(cfg, key, _coerce_value(value))
    validate_emotion_cfg(cfg)
    return cfg
