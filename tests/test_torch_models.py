import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cardio.torch_models import (
    FEATURE_GROUPS,
    EnsCVDDNet,
    FeatureGroupGatedNet,
    GRUClassifier,
    LeNet1D,
    MetaMLP,
    UniformGroupNet,
)


def test_torch_model_forward_shapes():
    batch_size = 8
    n_features = 21
    x = torch.randn(batch_size, n_features)

    # 1. LeNet1D
    lenet = LeNet1D(n_features)
    out_lenet = lenet(x)
    assert out_lenet.shape == (batch_size,)

    # 2. GRUClassifier
    gru = GRUClassifier(hidden_size=17, num_layers=2)
    out_gru = gru(x)
    assert out_gru.shape == (batch_size,)

    # 3. EnsCVDDNet
    ens = EnsCVDDNet(n_features)
    out_ens = ens(x)
    assert out_ens.shape == (batch_size,)

    # 4. MetaMLP
    meta = MetaMLP(n_inputs=2)
    meta_in = torch.rand(batch_size, 2)
    out_meta = meta(meta_in)
    assert out_meta.shape == (batch_size,)


def test_feature_group_gated_net_and_uniform():
    batch_size = 12
    # All 21 features across the 4 groups
    all_features = [f for group in FEATURE_GROUPS.values() for f in group]
    assert len(all_features) == 21
    x = torch.randn(batch_size, 21)

    # Test FGGN
    fggn = FeatureGroupGatedNet(all_features, hidden_dim=16)
    logits, weights = fggn.forward_with_gates(x)
    assert logits.shape == (batch_size,)
    assert weights.shape == (batch_size, 4)
    # Check that softmax gate weights sum to 1.0 for every sample
    assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size), atol=1e-5)

    # Test gradient propagation through gates
    loss = logits.sum()
    loss.backward()
    assert fggn.gate.weight.grad is not None

    # Test UniformGroupNet ablation
    uniform_net = UniformGroupNet(all_features, hidden_dim=16)
    out_uniform = uniform_net(x)
    assert out_uniform.shape == (batch_size,)
