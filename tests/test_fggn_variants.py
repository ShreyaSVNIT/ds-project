import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardio.fggn_variants import (
    CategoricalEmbeddingFGGN,
    CrossGroupInteractionFGGN,
    FeatureAttentionFGGN,
    FGGNPlusPlus,
    FocalLoss,
    HierarchicalGatingFGGN,
    ResidualFGGN,
    get_loss_function,
)
from cardio.torch_models import FEATURE_GROUPS


@pytest.fixture
def sample_feature_names():
    features = []
    for names in FEATURE_GROUPS.values():
        features.extend(names)
    return features


@pytest.fixture
def sample_batch(sample_feature_names):
    torch.manual_seed(42)
    # 8 samples, 21 features
    return torch.randn(8, len(sample_feature_names))


def test_focal_loss():
    logits = torch.randn(10, requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=5.0)
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None


def test_residual_fggn(sample_feature_names, sample_batch):
    model = ResidualFGGN(sample_feature_names, hidden_dim=16)
    logits, weights = model.forward_with_gates(sample_batch)
    assert logits.shape == (8,)
    assert weights.shape == (8, 4)
    assert torch.allclose(weights.sum(dim=1), torch.ones(8), atol=1e-5)
    logits.sum().backward()


def test_feature_attention_fggn(sample_feature_names, sample_batch):
    model = FeatureAttentionFGGN(sample_feature_names, embed_dim=8, hidden_dim=16)
    logits, weights = model.forward_with_gates(sample_batch)
    assert logits.shape == (8,)
    assert weights.shape == (8, 4)
    logits.sum().backward()


def test_cross_group_interaction_fggn(sample_feature_names, sample_batch):
    model = CrossGroupInteractionFGGN(sample_feature_names, hidden_dim=16)
    logits, weights = model.forward_with_gates(sample_batch)
    assert logits.shape == (8,)
    assert weights.shape == (8, 4)
    logits.sum().backward()


def test_hierarchical_gating_fggn(sample_feature_names, sample_batch):
    model = HierarchicalGatingFGGN(sample_feature_names, hidden_dim=16, initial_temperature=1.5)
    logits, weights = model.forward_with_gates(sample_batch)
    assert logits.shape == (8,)
    assert weights.shape == (8, 4)
    assert abs(model.temperature - 1.5) < 1e-4
    logits.sum().backward()


def test_categorical_embedding_fggn(sample_feature_names, sample_batch):
    model = CategoricalEmbeddingFGGN(sample_feature_names, embed_dim=8, hidden_dim=16)
    logits, weights = model.forward_with_gates(sample_batch)
    assert logits.shape == (8,)
    assert weights.shape == (8, 4)
    logits.sum().backward()


def test_fggn_plus_plus(sample_feature_names, sample_batch):
    model = FGGNPlusPlus(sample_feature_names, embed_dim=8, hidden_dim=16, num_heads=2)
    logits, weights = model.forward_with_gates(sample_batch)
    assert logits.shape == (8,)
    assert weights.shape == (8, 4)
    logits.sum().backward()
