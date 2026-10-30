from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from cardio.torch_models import FEATURE_GROUPS


# ---------------------------------------------------------------------------
# Feature Categorization Constants
# ---------------------------------------------------------------------------
CATEGORICAL_METADATA = {
    "Diabetes": {"cardinality": 4, "offset": 0},      # raw values 0, 1, 2
    "GenHlth": {"cardinality": 6, "offset": 1},       # raw values 1..5
    "Age": {"cardinality": 15, "offset": 1},          # raw values 1..13
    "Education": {"cardinality": 8, "offset": 1},     # raw values 1..6
    "Income": {"cardinality": 10, "offset": 1},       # raw values 1..8
}

CONTINUOUS_FEATURE_NAMES = ["BMI", "MentHlth", "PhysHlth"]

BINARY_FEATURE_NAMES = [
    "HighBP", "HighChol", "CholCheck", "Stroke", "DiffWalk",
    "Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
    "AnyHealthcare", "NoDocbcCost", "Sex"
]


# ---------------------------------------------------------------------------
# Phase 2 Variant 6: Imbalance-Aware Loss Suite
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal Loss for binary classification on imbalanced datasets.
    
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    Reduces the relative loss for well-classified examples and puts more focus
    on hard, misclassified instances near the decision boundary.
    """
    def __init__(
        self,
        alpha: Optional[float] = 0.25,
        gamma: float = 2.0,
        pos_weight: Optional[float] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        pw_tensor = (
            torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
            if self.pos_weight is not None
            else None
        )
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw_tensor, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        p_t = torch.clamp(p_t, min=1e-7, max=1.0 - 1e-7)
        modulating_factor = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * modulating_factor * bce_loss
        else:
            loss = modulating_factor * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_function(loss_name: str, pos_weight: Optional[float] = None, gamma: float = 2.0, alpha: float = 0.25) -> nn.Module:
    """Factory for standard BCE, weighted BCE, and Focal Loss."""
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "weighted_bce":
        if pos_weight is None:
            raise ValueError("pos_weight must be provided for weighted_bce")
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    elif loss_name == "focal":
        return FocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss_name: {loss_name!r}. Must be 'bce', 'weighted_bce', or 'focal'.")


# ---------------------------------------------------------------------------
# Core Building Blocks: Residual MLP Block with LayerNorm & GELU
# ---------------------------------------------------------------------------
class ResidualMLPBlock(nn.Module):
    """Pre-LayerNorm Residual MLP block with GELU and Dropout."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = self.linear1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.linear2(h)
        h = self.drop2(h)
        return residual + h


# ---------------------------------------------------------------------------
# Phase 2 Variant 1: Residual FGGN
# ---------------------------------------------------------------------------
class ResidualFGGN(nn.Module):
    """FGGN variant replacing plain ReLU MLPs with Residual LayerNorm GELU blocks."""
    def __init__(
        self,
        feature_names: List[str],
        hidden_dim: int = 32,
        dropout: float = 0.1,
        feature_groups: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__()
        self.feature_names = list(feature_names)
        self.feature_groups = feature_groups if feature_groups is not None else FEATURE_GROUPS
        name_to_idx = {name: i for i, name in enumerate(self.feature_names)}
        self.group_names = []
        self.group_indices = []
        for group_name, names in self.feature_groups.items():
            indices = [name_to_idx[name] for name in names if name in name_to_idx]
            if indices:
                self.group_names.append(group_name)
                self.group_indices.append(indices)

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(indices), hidden_dim),
                nn.GELU(),
                ResidualMLPBlock(hidden_dim, dropout=dropout),
                nn.LayerNorm(hidden_dim),
            )
            for indices in self.group_indices
        ])
        total_dim = hidden_dim * len(self.group_indices)
        self.gate = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(total_dim // 2, len(self.group_indices))
        )
        self.classifier = nn.Sequential(
            ResidualMLPBlock(hidden_dim, dropout=dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward_with_gates(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = [encoder(x[:, indices]) for encoder, indices in zip(self.encoders, self.group_indices)]
        joint = torch.cat(embeddings, dim=1)
        weights = torch.softmax(self.gate(joint), dim=1)
        stacked = torch.stack(embeddings, dim=1)  # [batch, groups, hidden_dim]
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        logits = self.classifier(fused).squeeze(-1)
        return logits, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_gates(x)
        return logits


# ---------------------------------------------------------------------------
# Phase 2 Variant 2: Feature-wise Attention FGGN
# ---------------------------------------------------------------------------
class IntraGroupFeatureAttention(nn.Module):
    """Computes multi-head self-attention across individual features within a domain group."""
    def __init__(self, n_features: int, embed_dim: int = 16, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.embed_dim = embed_dim
        # Project each scalar feature into an embedding token
        self.feature_projections = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(n_features)
        ])
        self.pos_emb = nn.Parameter(torch.zeros(1, n_features, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        # Learnable pooling query to aggregate feature tokens into single group vector
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_attn = nn.Linear(embed_dim, 1)

    def forward(self, x_group: torch.Tensor) -> torch.Tensor:
        batch_size = x_group.shape[0]
        # x_group shape: [batch, n_features]
        tokens = [proj(x_group[:, i : i + 1]) for i, proj in enumerate(self.feature_projections)]
        h = torch.stack(tokens, dim=1) + self.pos_emb  # [batch, n_features, embed_dim]
        # Self-attention over features
        h_norm = self.norm1(h)
        attn_out, _ = self.mha(h_norm, h_norm, h_norm)
        h = h + attn_out
        # Feed-forward
        h = h + self.mlp(self.norm2(h))
        # Attentive pooling into single group representation
        weights = torch.softmax(self.pool_attn(h), dim=1)  # [batch, n_features, 1]
        pooled = torch.sum(h * weights, dim=1)  # [batch, embed_dim]
        return pooled


class FeatureAttentionFGGN(nn.Module):
    """FGGN variant introducing intra-group multi-head attention before group fusion."""
    def __init__(
        self,
        feature_names: List[str],
        embed_dim: int = 16,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        feature_groups: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__()
        self.feature_names = list(feature_names)
        self.feature_groups = feature_groups if feature_groups is not None else FEATURE_GROUPS
        name_to_idx = {name: i for i, name in enumerate(self.feature_names)}
        self.group_names = []
        self.group_indices = []
        for group_name, names in self.feature_groups.items():
            indices = [name_to_idx[name] for name in names if name in name_to_idx]
            if indices:
                self.group_names.append(group_name)
                self.group_indices.append(indices)

        self.encoders = nn.ModuleList([
            IntraGroupFeatureAttention(len(indices), embed_dim=embed_dim, num_heads=2, dropout=dropout)
            for indices in self.group_indices
        ])
        self.group_proj = nn.ModuleList([
            nn.Linear(embed_dim, hidden_dim) for _ in self.group_indices
        ])
        total_dim = hidden_dim * len(self.group_indices)
        self.gate = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(total_dim // 2, len(self.group_indices))
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward_with_gates(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = [
            proj(enc(x[:, indices]))
            for enc, proj, indices in zip(self.encoders, self.group_proj, self.group_indices)
        ]
        joint = torch.cat(embeddings, dim=1)
        weights = torch.softmax(self.gate(joint), dim=1)
        stacked = torch.stack(embeddings, dim=1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        logits = self.classifier(fused).squeeze(-1)
        return logits, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_gates(x)
        return logits


# ---------------------------------------------------------------------------
# Phase 2 Variant 3: Cross-Group Interaction FGGN
# ---------------------------------------------------------------------------
class CrossGroupInteractionModule(nn.Module):
    """Computes second-order nonlinear interactions across domain group representations."""
    def __init__(self, num_groups: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.num_groups = num_groups
        self.hidden_dim = hidden_dim
        # Multi-head attention across the domain tokens
        self.norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=2, dropout=dropout, batch_first=True
        )
        # Pairwise interaction projection
        num_pairs = (num_groups * (num_groups - 1)) // 2
        self.pair_proj = nn.Sequential(
            nn.Linear(num_pairs * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, stacked_groups: torch.Tensor) -> torch.Tensor:
        # stacked_groups: [batch, num_groups, hidden_dim]
        norm_groups = self.norm(stacked_groups)
        attn_groups, _ = self.cross_attn(norm_groups, norm_groups, norm_groups)
        enriched_groups = stacked_groups + attn_groups  # [batch, num_groups, hidden_dim]

        # Compute pairwise Hadamard products
        pair_products = []
        for i in range(self.num_groups):
            for j in range(i + 1, self.num_groups):
                pair_products.append(enriched_groups[:, i] * enriched_groups[:, j])
        all_pairs = torch.cat(pair_products, dim=1)  # [batch, num_pairs * hidden_dim]
        interaction_vec = self.pair_proj(all_pairs)    # [batch, hidden_dim]
        return interaction_vec


class CrossGroupInteractionFGGN(nn.Module):
    """FGGN variant introducing an explicit cross-group interaction module."""
    def __init__(
        self,
        feature_names: List[str],
        hidden_dim: int = 32,
        dropout: float = 0.1,
        feature_groups: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__()
        self.feature_names = list(feature_names)
        self.feature_groups = feature_groups if feature_groups is not None else FEATURE_GROUPS
        name_to_idx = {name: i for i, name in enumerate(self.feature_names)}
        self.group_names = []
        self.group_indices = []
        for group_name, names in self.feature_groups.items():
            indices = [name_to_idx[name] for name in names if name in name_to_idx]
            if indices:
                self.group_names.append(group_name)
                self.group_indices.append(indices)

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(indices), hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            for indices in self.group_indices
        ])
        self.interaction_module = CrossGroupInteractionModule(
            num_groups=len(self.group_indices), hidden_dim=hidden_dim, dropout=dropout
        )
        total_dim = hidden_dim * len(self.group_indices)
        self.gate = nn.Linear(total_dim, len(self.group_indices))
        # Classifier receives both fused representation and cross-group interactions
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward_with_gates(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = [encoder(x[:, indices]) for encoder, indices in zip(self.encoders, self.group_indices)]
        stacked = torch.stack(embeddings, dim=1)  # [batch, num_groups, hidden_dim]
        joint = torch.cat(embeddings, dim=1)
        weights = torch.softmax(self.gate(joint), dim=1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)  # [batch, hidden_dim]

        # Cross-group interaction
        interaction = self.interaction_module(stacked)  # [batch, hidden_dim]
        combined = torch.cat([fused, interaction], dim=1)  # [batch, hidden_dim * 2]
        logits = self.classifier(combined).squeeze(-1)
        return logits, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_gates(x)
        return logits


# ---------------------------------------------------------------------------
# Phase 2 Variant 4: Hierarchical Gating FGGN
# ---------------------------------------------------------------------------
class HierarchicalGatingFGGN(nn.Module):
    """FGGN variant combining feature-level attention with temperature-scaled group gating."""
    def __init__(
        self,
        feature_names: List[str],
        hidden_dim: int = 32,
        initial_temperature: float = 1.0,
        dropout: float = 0.1,
        feature_groups: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__()
        self.feature_names = list(feature_names)
        self.feature_groups = feature_groups if feature_groups is not None else FEATURE_GROUPS
        name_to_idx = {name: i for i, name in enumerate(self.feature_names)}
        self.group_names = []
        self.group_indices = []
        for group_name, names in self.feature_groups.items():
            indices = [name_to_idx[name] for name in names if name in name_to_idx]
            if indices:
                self.group_names.append(group_name)
                self.group_indices.append(indices)

        # Feature-level attention gates inside each group
        self.feature_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(indices), len(indices)),
                nn.Sigmoid(),
            )
            for indices in self.group_indices
        ])
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(indices), hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            for indices in self.group_indices
        ])
        # Temperature-calibrated group gate
        total_dim = hidden_dim * len(self.group_indices)
        self.gate_mlp = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(total_dim // 2, len(self.group_indices)),
        )
        self.log_temperature = nn.Parameter(torch.tensor(math.log(initial_temperature)))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temperature).item())

    def forward_with_gates(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = []
        for f_gate, enc, indices in zip(self.feature_gates, self.encoders, self.group_indices):
            x_g = x[:, indices]
            # Level 1: Feature-level gating
            gated_x_g = x_g * f_gate(x_g)
            embeddings.append(enc(gated_x_g))

        stacked = torch.stack(embeddings, dim=1)
        joint = torch.cat(embeddings, dim=1)

        # Level 2: Group-level temperature-scaled routing
        temp = torch.clamp(torch.exp(self.log_temperature), min=0.1, max=10.0)
        gate_logits = self.gate_mlp(joint) / temp
        weights = torch.softmax(gate_logits, dim=1)

        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        logits = self.classifier(fused).squeeze(-1)
        return logits, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_gates(x)
        return logits


# ---------------------------------------------------------------------------
# Phase 2 Variant 5: Categorical Embedding FGGN
# ---------------------------------------------------------------------------
class HeterogeneousFeatureTokenizer(nn.Module):
    """Maps categorical/ordinal features to entity embeddings and continuous/binary features to linear projections."""
    def __init__(
        self,
        feature_names: List[str],
        embed_dim: int = 16,
        scaler_mean: Optional[List[float]] = None,
        scaler_scale: Optional[List[float]] = None,
    ):
        super().__init__()
        self.feature_names = list(feature_names)
        self.embed_dim = embed_dim

        # Scaler parameters for exact inverse transform of standardized tensors
        if scaler_mean is not None and scaler_scale is not None:
            self.register_buffer("scaler_mean", torch.tensor(scaler_mean, dtype=torch.float32))
            self.register_buffer("scaler_scale", torch.tensor(scaler_scale, dtype=torch.float32))
        else:
            self.scaler_mean = None
            self.scaler_scale = None

        self.categorical_indices: Dict[str, int] = {}
        self.cat_embeddings = nn.ModuleDict()
        self.cat_offsets: Dict[str, int] = {}

        for name, meta in CATEGORICAL_METADATA.items():
            if name in self.feature_names:
                idx = self.feature_names.index(name)
                self.categorical_indices[name] = idx
                self.cat_offsets[name] = meta["offset"]
                self.cat_embeddings[name] = nn.Embedding(meta["cardinality"] + 1, embed_dim)

        self.continuous_indices: List[int] = []
        for name in CONTINUOUS_FEATURE_NAMES + BINARY_FEATURE_NAMES:
            if name in self.feature_names:
                self.continuous_indices.append(self.feature_names.index(name))

        self.num_projections = nn.ModuleList([
            nn.Sequential(nn.Linear(1, embed_dim), nn.LayerNorm(embed_dim))
            for _ in self.continuous_indices
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Inverse standardize categorical features if scaler buffers exist
        x_raw = x
        if self.scaler_mean is not None and self.scaler_scale is not None:
            x_raw = x * self.scaler_scale + self.scaler_mean

        token_dict: Dict[str, torch.Tensor] = {}

        # Embed categorical/ordinal features
        for name, idx in self.categorical_indices.items():
            offset = self.cat_offsets[name]
            # Discretize safely into 0-indexed long
            cat_vals = torch.round(x_raw[:, idx]).long() - offset
            cat_vals = torch.clamp(cat_vals, min=0, max=CATEGORICAL_METADATA[name]["cardinality"])
            token_dict[name] = self.cat_embeddings[name](cat_vals)

        # Project continuous and binary features
        for idx, proj in zip(self.continuous_indices, self.num_projections):
            name = self.feature_names[idx]
            token_dict[name] = proj(x[:, idx : idx + 1])

        return token_dict


class CategoricalEmbeddingFGGN(nn.Module):
    """FGGN variant featuring entity embeddings for ordinal/categorical variables and continuous projections."""
    def __init__(
        self,
        feature_names: List[str],
        embed_dim: int = 16,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        scaler_mean: Optional[List[float]] = None,
        scaler_scale: Optional[List[float]] = None,
        feature_groups: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__()
        self.feature_names = list(feature_names)
        self.feature_groups = feature_groups if feature_groups is not None else FEATURE_GROUPS
        self.tokenizer = HeterogeneousFeatureTokenizer(
            feature_names, embed_dim=embed_dim, scaler_mean=scaler_mean, scaler_scale=scaler_scale
        )
        self.group_names = []
        self.group_feature_lists = []
        for group_name, names in self.feature_groups.items():
            valid_names = [n for n in names if n in self.feature_names]
            if valid_names:
                self.group_names.append(group_name)
                self.group_feature_lists.append(valid_names)

        # Each group aggregates its embedded tokens
        self.group_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(feat_list) * embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            for feat_list in self.group_feature_lists
        ])
        total_dim = hidden_dim * len(self.group_names)
        self.gate = nn.Linear(total_dim, len(self.group_names))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward_with_gates(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(x)
        embeddings = []
        for feat_list, encoder in zip(self.group_feature_lists, self.group_encoders):
            group_tokens = torch.cat([tokens[name] for name in feat_list], dim=1)
            embeddings.append(encoder(group_tokens))

        stacked = torch.stack(embeddings, dim=1)
        joint = torch.cat(embeddings, dim=1)
        weights = torch.softmax(self.gate(joint), dim=1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        logits = self.classifier(fused).squeeze(-1)
        return logits, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_gates(x)
        return logits


# ---------------------------------------------------------------------------
# Phase 2: Unified Flagship Architecture — FGGN++
# ---------------------------------------------------------------------------
class FGGNPlusPlus(nn.Module):
    """Unified Flagship Architecture: FGGN++
    
    Synthesizes:
    1. Heterogeneous Feature Tokenization (Entity Embeddings + Continuous Projections).
    2. Intra-Group Multi-Head Self-Attention across feature tokens.
    3. Deep Residual MLP Encoders with Pre-LayerNorm and GELU activations.
    4. Cross-Group Multi-Head Interaction Module capturing comorbidity x lifestyle x demographic interplay.
    5. Hierarchical Gating with learnable temperature calibration.
    6. Residual classification head.
    """
    def __init__(
        self,
        feature_names: List[str],
        embed_dim: int = 16,
        hidden_dim: int = 64,
        num_heads: int = 2,
        initial_temperature: float = 1.0,
        dropout: float = 0.15,
        scaler_mean: Optional[List[float]] = None,
        scaler_scale: Optional[List[float]] = None,
    ):
        super().__init__()
        self.feature_names = list(feature_names)
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # 1. Tokenizer
        self.tokenizer = HeterogeneousFeatureTokenizer(
            feature_names, embed_dim=embed_dim, scaler_mean=scaler_mean, scaler_scale=scaler_scale
        )

        self.group_names = []
        self.group_feature_lists = []
        for group_name, names in FEATURE_GROUPS.items():
            valid_names = [n for n in names if n in self.feature_names]
            if valid_names:
                self.group_names.append(group_name)
                self.group_feature_lists.append(valid_names)

        num_groups = len(self.group_names)

        # 2. Intra-Group Feature Attention & Residual Encoders
        self.group_mhas = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            for _ in self.group_feature_lists
        ])
        self.group_mha_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in self.group_feature_lists
        ])
        self.group_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(feat_list) * embed_dim, hidden_dim),
                nn.GELU(),
                ResidualMLPBlock(hidden_dim, dropout=dropout),
                nn.LayerNorm(hidden_dim),
            )
            for feat_list in self.group_feature_lists
        ])

        # 3. Cross-Group Interaction Module
        self.cross_group_norm = nn.LayerNorm(hidden_dim)
        self.cross_group_mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        num_pairs = (num_groups * (num_groups - 1)) // 2
        self.cross_pair_proj = nn.Sequential(
            nn.Linear(num_pairs * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        # 4. Hierarchical Temperature-Scaled Gating
        total_dim = hidden_dim * num_groups
        self.gate_mlp = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(total_dim // 2, num_groups),
        )
        self.log_temperature = nn.Parameter(torch.tensor(math.log(initial_temperature)))

        # 5. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            ResidualMLPBlock(hidden_dim, dropout=dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temperature).item())

    def forward_with_gates(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(x)

        # Process each domain group with intra-group attention
        group_embeddings = []
        for feat_list, mha, norm, proj in zip(
            self.group_feature_lists, self.group_mhas, self.group_mha_norms, self.group_projectors
        ):
            # Stack tokens: [batch, n_features_in_group, embed_dim]
            t_stack = torch.stack([tokens[name] for name in feat_list], dim=1)
            t_norm = norm(t_stack)
            attn_out, _ = mha(t_norm, t_norm, t_norm)
            t_enriched = t_stack + attn_out
            flat_tokens = t_enriched.reshape(t_enriched.shape[0], -1)
            group_embeddings.append(proj(flat_tokens))

        # Stack group embeddings: [batch, num_groups, hidden_dim]
        stacked = torch.stack(group_embeddings, dim=1)
        num_groups = stacked.shape[1]

        # Cross-group interaction
        cross_norm = self.cross_group_norm(stacked)
        cross_attn_out, _ = self.cross_group_mha(cross_norm, cross_norm, cross_norm)
        enriched_stacked = stacked + cross_attn_out

        # Pairwise second-order products
        pairs = []
        for i in range(num_groups):
            for j in range(i + 1, num_groups):
                pairs.append(enriched_stacked[:, i] * enriched_stacked[:, j])
        pair_tensor = torch.cat(pairs, dim=1)
        interaction_vec = self.cross_pair_proj(pair_tensor)  # [batch, hidden_dim]

        # Gating with learnable temperature
        joint = torch.cat(group_embeddings, dim=1)
        temp = torch.clamp(torch.exp(self.log_temperature), min=0.1, max=10.0)
        weights = torch.softmax(self.gate_mlp(joint) / temp, dim=1)
        fused = torch.sum(enriched_stacked * weights.unsqueeze(-1), dim=1)  # [batch, hidden_dim]

        # Final prediction
        fused_and_interacted = torch.cat([fused, interaction_vec], dim=1)  # [batch, hidden_dim * 2]
        logits = self.classifier(fused_and_interacted).squeeze(-1)
        return logits, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_gates(x)
        return logits
