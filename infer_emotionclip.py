import argparse
import json

import torch
from PIL import Image

from config.emotion_defaults import anatomy_requirement_reasons, load_emotion_cfg
from datasets.anatomy import ANATOMY_ARTIFACT_SCHEMA_VERSION, load_anatomy_artifact
from datasets.emotion_manifest import CANONICAL_EMOTIONS, FaceSafeTransform
from model.emotionclip_model import EmotionCLIPModel
from processor.processor_emotionclip import load_emotion_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Run EmotionCLIP-ReID FER inference on one image.")
    parser.add_argument("--config_file", default="configs/emotion/vit_b16_emotionclip.yml")
    parser.add_argument("--weight", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--landmark", default="", help="Optional anatomy artifact JSON for this image")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_emotion_cfg(args.config_file, args.opts)
    anatomy_reasons = anatomy_requirement_reasons(cfg)
    if (
        anatomy_reasons
        and not args.landmark
        and not bool(cfg["DATASETS"].get("ALLOW_ANATOMY_FALLBACK", False))
    ):
        raise RuntimeError(
            "This inference config requires anatomy evidence for "
            + ", ".join(anatomy_reasons)
            + ". Supply --landmark or use an explicit anatomy-free baseline."
        )
    if cfg["MODEL"]["DEVICE"] == "cuda" and not torch.cuda.is_available():
        cfg["MODEL"]["DEVICE"] = "cpu"
    model_cfg = cfg["MODEL"]["EMOTION"]
    uncertainty_cfg = cfg["MODEL"].get("UNCERTAINTY", {})
    fusion_cfg = cfg["MODEL"].get("FUSION", {})
    prompt_geometry_cfg = cfg["MODEL"].get("ANATOMY_PROMPT", {})
    routing_cfg = cfg["MODEL"].get("ROUTING", {})
    geometry_cfg = cfg["MODEL"].get("GEOMETRY", {})
    disagreement_cfg = cfg["MODEL"].get("REGION_DISAGREEMENT", {})
    model = EmotionCLIPModel(
        class_names=CANONICAL_EMOTIONS,
        backbone_name=cfg["MODEL"]["NAME"],
        image_size=cfg["INPUT"]["SIZE_TEST"],
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
        initial_branch_temperatures=fusion_cfg.get("INITIAL_TEMPERATURES", [1.0, 1.0, 1.0]),
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
        max_abs_raw_strength=float(uncertainty_cfg.get("MAX_ABS_RAW_STRENGTH", 20.0)),
        reliability_use_anatomy_quality=bool(uncertainty_cfg.get("USE_ANATOMY_QUALITY", True)),
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
        if args.landmark:
            artifact = load_anatomy_artifact(args.landmark)
            schema_version = int((artifact or {}).get("schema_version", 0) or 0)
            if schema_version != ANATOMY_ARTIFACT_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Anatomy artifact schema {schema_version} is incompatible with required "
                    f"schema {ANATOMY_ARTIFACT_SCHEMA_VERSION}; regenerate the artifact."
                )
            tensor, anatomy = transform(image, anatomy=artifact)
            anatomy = {key: value.unsqueeze(0).to(device) for key, value in anatomy.items()}
        else:
            tensor = transform(image)
            anatomy = None
        tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(images=tensor, anatomy=anatomy)
    probabilities = outputs["probabilities"][0].detach().cpu()
    prediction = int(probabilities.argmax())
    result = {
        "emotion": CANONICAL_EMOTIONS[prediction],
        "emotion_id": prediction,
        "probabilities": {name: float(probabilities[idx]) for idx, name in enumerate(CANONICAL_EMOTIONS)},
        "uncertainty": float(outputs["uncertainty"][0].detach().cpu()),
        "class_ambiguity": float(outputs["class_ambiguity"][0].detach().cpu()),
        "region_disagreement": float(outputs["region_disagreement"][0].detach().cpu()),
        "region_disagreement_valid": bool(outputs["region_disagreement_valid"][0].detach().cpu()),
        "region_quality": outputs["region_quality"][0].detach().cpu().tolist(),
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
