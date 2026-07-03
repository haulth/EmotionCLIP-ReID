import torch

from datasets.emotion_manifest import CANONICAL_EMOTIONS
from model.emotionclip_model import EmotionPromptLearner, ExpressionAdapter


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
