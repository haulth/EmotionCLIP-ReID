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
            "TRAIN_LAST_BLOCKS": 2,
            "STAGE1_WEIGHT": "",
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
            "BASE_LR": 3.5e-4,
            "WEIGHT_DECAY": 1.0e-4,
            "CHECKPOINT_PERIOD": 10,
            "LOG_PERIOD": 20,
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
        },
    },
    "TEST": {
        "IMS_PER_BATCH": 64,
        "WEIGHT": "",
    },
    "TRAIN": {
        "RUN_STAGE1": True,
        "RUN_STAGE2": True,
        "PROGRESS_BAR": "auto",
    },
    "OUTPUT_DIR": "outputs/emotionclip",
}


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
    return cfg
