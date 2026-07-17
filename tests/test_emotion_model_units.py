import torch

from datasets.emotion_manifest import CANONICAL_EMOTIONS
from model.emotionclip_model import (
    BranchFusion,
    EmotionCLIPModel,
    EmotionPromptLearner,
    ExpressionAdapter,
    validate_train_last_blocks,
)


def test_emotion_prompt_learner_shapes():
    token_embedding = torch.nn.Embedding(50000, 512)
    learner = EmotionPromptLearner(
        CANONICAL_EMOTIONS,
        dtype=torch.float32,
        token_embedding=token_embedding,
        n_ctx=4,
    )
    prompts, tokenized = learner()
    assert prompts.shape == (len(CANONICAL_EMOTIONS), 77, 512)
    assert tokenized.shape == (len(CANONICAL_EMOTIONS), 77)

    labels = torch.tensor([0, 3])
    prompts, tokenized = learner(labels)
    assert prompts.shape == (2, 77, 512)
    assert tokenized.shape == (2, 77)


def test_expression_adapter_zero_initialized_residual():
    adapter = ExpressionAdapter(dim=8, bottleneck_dim=4)
    x = torch.randn(3, 2, 8)
    y = adapter(x)
    torch.testing.assert_close(y, torch.zeros_like(y))


def test_fixed_fusion_has_simplex_gate_bounded_temperatures_and_gradients():
    fusion = BranchFusion(
        feature_dim=8,
        prior_weights=(1.0, 1.0, 0.5),
        gate_mode="fixed",
        scale_mode="temperature",
    )
    features = torch.randn(4, 8)
    branches = [torch.randn(4, 7, requires_grad=True) for _ in range(3)]
    outputs = fusion(*branches, features)

    expected_gate = torch.tensor([0.4, 0.4, 0.2]).expand(4, -1)
    torch.testing.assert_close(outputs["fusion_gate"], expected_gate)
    torch.testing.assert_close(outputs["fusion_gate"].sum(dim=-1), torch.ones(4))
    assert torch.all(outputs["fusion_gate"] >= 0)
    assert outputs["logits"].shape == (4, 7)
    assert torch.all(torch.isfinite(outputs["branch_temperatures"]))
    assert torch.all(outputs["branch_temperatures"] >= 0.05)
    assert torch.all(outputs["branch_temperatures"] <= 20.0)

    outputs["logits"].sum().backward()
    assert fusion.branch_raw_temperatures.grad is not None
    assert not hasattr(fusion, "fusion_gate_logits")


def test_simplex_and_sample_dependent_fusion_are_valid_ablation_modes():
    features = torch.randn(5, 8)
    branches = [torch.randn(5, 7) for _ in range(3)]
    for mode in ("simplex", "sample_dependent"):
        fusion = BranchFusion(
            feature_dim=8,
            prior_weights=(1.0, 1.0, 0.5),
            gate_mode=mode,
            scale_mode="rms",
        )
        outputs = fusion(*branches, features)
        assert outputs["fusion_gate"].shape == (5, 3)
        torch.testing.assert_close(outputs["fusion_gate"].sum(dim=-1), torch.ones(5))
        torch.testing.assert_close(outputs["fusion_gate"][0], torch.tensor([0.4, 0.4, 0.2]))
        outputs["logits"].sum().backward()
        if mode == "simplex":
            assert fusion.fusion_gate_logits.grad is not None
        else:
            assert fusion.fusion_gate[-1].weight.grad is not None


def test_train_last_blocks_validation():
    assert validate_train_last_blocks(0, 12) == 0
    assert validate_train_last_blocks(12, 12) == 12
    for invalid in (-1, 13):
        try:
            validate_train_last_blocks(invalid, 12)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected train_last_blocks={invalid} to fail")


class _DummyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.original = torch.nn.Linear(4, 4)
        self.emotion_adapter = ExpressionAdapter(4, bottleneck_dim=2)


def _stage_test_model(last_blocks):
    model = EmotionCLIPModel.__new__(EmotionCLIPModel)
    torch.nn.Module.__init__(model)
    model.train_last_blocks = last_blocks
    model.prompt_learner = torch.nn.Linear(4, 4)
    model.classifier = torch.nn.Linear(4, 3)
    model.reliability_head = torch.nn.Linear(4, 1)
    model.fusion = BranchFusion(4, (1.0, 1.0, 0.5))
    model.logit_scale = torch.nn.Parameter(torch.tensor(1.0))
    transformer = torch.nn.Module()
    transformer.resblocks = torch.nn.ModuleList([_DummyBlock(), _DummyBlock()])
    model.image_encoder = torch.nn.Module()
    model.image_encoder.transformer = transformer
    return model


def test_adapter_only_stage2_trainable_parameter_whitelist():
    model = _stage_test_model(last_blocks=0)
    model.set_train_stage(2)
    for name, parameter in model.named_parameters():
        if name.startswith("image_encoder"):
            if ".emotion_adapter." in name:
                assert parameter.requires_grad, name
            else:
                assert not parameter.requires_grad, name
    assert not model.logit_scale.requires_grad
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert all(parameter.requires_grad for parameter in model.reliability_head.parameters())
    assert model.fusion.branch_raw_temperatures.requires_grad
