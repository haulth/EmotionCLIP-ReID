import argparse
import json

import torch
from PIL import Image

from config.emotion_defaults import load_emotion_cfg
from datasets.emotion_manifest import CANONICAL_EMOTIONS, FaceSafeTransform
from model.emotionclip_model import EmotionCLIPModel
from processor.processor_emotionclip import load_emotion_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Run EmotionCLIP-ReID FER inference on one image.")
    parser.add_argument("--config_file", default="configs/emotion/vit_b16_emotionclip.yml")
    parser.add_argument("--weight", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_emotion_cfg(args.config_file, args.opts)
    if cfg["MODEL"]["DEVICE"] == "cuda" and not torch.cuda.is_available():
        cfg["MODEL"]["DEVICE"] = "cpu"
    model_cfg = cfg["MODEL"]["EMOTION"]
    uncertainty_cfg = cfg["MODEL"].get("UNCERTAINTY", {})
    fusion_cfg = cfg["MODEL"].get("FUSION", {})
    model = EmotionCLIPModel(
        class_names=CANONICAL_EMOTIONS,
        backbone_name=cfg["MODEL"]["NAME"],
        image_size=cfg["INPUT"]["SIZE_TEST"],
        stride_size=cfg["MODEL"]["STRIDE_SIZE"],
        n_ctx=int(model_cfg["N_CTX"]),
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
        reliability_hidden_dim=int(uncertainty_cfg.get("HIDDEN_DIM", 128)),
        reliability_dropout=float(uncertainty_cfg.get("DROPOUT", 0.1)),
        detach_class_prob=bool(uncertainty_cfg.get("DETACH_CLASS_PROB", True)),
        max_strength=uncertainty_cfg.get("MAX_STRENGTH", 100.0),
    )
    load_emotion_checkpoint(model, args.weight, strict=False)
    device = torch.device(cfg["MODEL"]["DEVICE"])
    model.to(device)
    model.eval()

    transform = FaceSafeTransform(
        size=tuple(cfg["INPUT"]["SIZE_TEST"]),
        train=False,
        mean=cfg["INPUT"]["PIXEL_MEAN"],
        std=cfg["INPUT"]["PIXEL_STD"],
    )
    with Image.open(args.image) as image:
        tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(images=tensor)
    probabilities = outputs["probabilities"][0].detach().cpu()
    prediction = int(probabilities.argmax())
    result = {
        "emotion": CANONICAL_EMOTIONS[prediction],
        "emotion_id": prediction,
        "probabilities": {name: float(probabilities[idx]) for idx, name in enumerate(CANONICAL_EMOTIONS)},
        "uncertainty": float(outputs["uncertainty"][0].detach().cpu()),
        "strength": float(outputs["strength"][0].detach().cpu()),
        "fusion_gate": outputs["fusion_gate"][0].detach().cpu().tolist(),
        "branch_temperatures": outputs["branch_temperatures"].detach().cpu().tolist(),
        "dirichlet_mean": {
            name: float(outputs["dirichlet_mean"][0, idx].detach().cpu())
            for idx, name in enumerate(CANONICAL_EMOTIONS)
        },
        "descriptor_similarity": {
            name: float(outputs["alignment_logits"][0, idx].detach().cpu())
            for idx, name in enumerate(CANONICAL_EMOTIONS)
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
