import argparse
import logging
import os
import random

import numpy as np
import torch

from config.emotion_defaults import load_emotion_cfg
from datasets.emotion_manifest import make_emotion_dataloaders
from model.emotionclip_model import EmotionCLIPModel
from processor.processor_emotionclip import (
    do_train_emotion_stage1,
    do_train_emotion_stage2,
    load_emotion_checkpoint,
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


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def main():
    parser = argparse.ArgumentParser(description="EmotionCLIP-ReID FER training")
    parser.add_argument("--config_file", default="configs/emotion/vit_b16_emotionclip.yml", type=str)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_emotion_cfg(args.config_file, args.opts)
    setup_logging(cfg["OUTPUT_DIR"])
    logger = logging.getLogger("emotionclip.train")
    logger.info("Loaded config: %s", cfg)

    if cfg["MODEL"]["DEVICE"] == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but current PyTorch is CPU-only; falling back to CPU")
        cfg["MODEL"]["DEVICE"] = "cpu"
    set_seed(int(cfg["SOLVER"].get("SEED", 1234)))

    train_loader, train_loader_stage1, val_loader, class_names = make_emotion_dataloaders(cfg)
    model_cfg = cfg["MODEL"]["EMOTION"]
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

    device = torch.device(cfg["MODEL"]["DEVICE"])
    model.to(device)

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
