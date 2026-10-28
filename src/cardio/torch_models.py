from __future__ import annotations

import torch
from torch import nn


class LeNet1D(nn.Module):
    """Small 1-D CNN inspired by the paper's modified LeNet for tabular columns."""
    def __init__(self, n_features: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.Tanh(), nn.MaxPool1d(2),
            nn.Conv1d(32, 8, kernel_size=5, padding=2), nn.Tanh(), nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=1), nn.Tanh(), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh(),
            nn.Linear(16, 4), nn.Tanh(),
        )
        self.classifier = nn.Linear(4, 1)

    def features(self, x):
        return self.head(self.conv(x.unsqueeze(1)))

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(-1)


class GRUClassifier(nn.Module):
    """GRU that treats the fixed feature-column order as a sequence of scalar inputs."""
    def __init__(self, hidden_size: int = 17, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=0.1 if num_layers > 1 else 0.0)
        self.classifier = nn.Linear(hidden_size, 1)

    def features(self, x):
        output, _ = self.gru(x.unsqueeze(-1))
        return output[:, -1, :]

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(-1)


class EnsCVDDNet(nn.Module):
    """Parallel LeNet/GRU hybrid; branch features are concatenated before classification."""
    def __init__(self, n_features: int):
        super().__init__()
        self.lenet = LeNet1D(n_features)
        self.gru = GRUClassifier()
        self.classifier = nn.Sequential(nn.Linear(4 + 17, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, x):
        z = torch.cat([self.lenet.features(x), self.gru.features(x)], dim=1)
        return self.classifier(z).squeeze(-1)


class MetaMLP(nn.Module):
    """Meta-learner operating on base-model probabilities (2 inputs in the paper-style blend)."""
    def __init__(self, n_inputs: int = 2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_inputs, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


FEATURE_GROUPS = {
    "clinical_history": ["HighBP", "HighChol", "CholCheck", "Stroke", "Diabetes", "DiffWalk"],
    "lifestyle": ["BMI", "Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"],
    "self_reported_health": ["GenHlth", "MentHlth", "PhysHlth"],
    "access_demographics": ["AnyHealthcare", "NoDocbcCost", "Sex", "Age", "Education", "Income"],
}


class FeatureGroupGatedNet(nn.Module):
    """Domain-aware tabular network with group-specific encoders and learned softmax gates."""
    def __init__(self, feature_names: list[str], hidden_dim: int = 16):
        super().__init__()
        self.feature_names = list(feature_names)
        name_to_idx = {name: i for i, name in enumerate(self.feature_names)}
        self.group_names = []
        self.group_indices = []
        for group_name, names in FEATURE_GROUPS.items():
            indices = [name_to_idx[name] for name in names if name in name_to_idx]
            if indices:
                self.group_names.append(group_name)
                self.group_indices.append(indices)
        if not self.group_indices:
            raise ValueError("No recognized feature names were provided to FeatureGroupGatedNet.")
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(len(indices), hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            for indices in self.group_indices
        ])
        self.gate = nn.Linear(hidden_dim * len(self.group_indices), len(self.group_indices))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, 1)
        )

    def forward_with_gates(self, x):
        embeddings = [encoder(x[:, indices]) for encoder, indices in zip(self.encoders, self.group_indices)]
        joint = torch.cat(embeddings, dim=1)
        weights = torch.softmax(self.gate(joint), dim=1)
        stacked = torch.stack(embeddings, dim=1)  # [batch, groups, hidden_dim]
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        return self.classifier(fused).squeeze(-1), weights

    def forward(self, x):
        logits, _ = self.forward_with_gates(x)
        return logits


class UniformGroupNet(nn.Module):
    """Ablation network: uses identical feature group encoders as FGGN but averages embeddings uniformly (no learned gates)."""
    def __init__(self, feature_names: list[str], hidden_dim: int = 16):
        super().__init__()
        self.feature_names = list(feature_names)
        name_to_idx = {name: i for i, name in enumerate(self.feature_names)}
        self.group_names = []
        self.group_indices = []
        for group_name, names in FEATURE_GROUPS.items():
            indices = [name_to_idx[name] for name in names if name in name_to_idx]
            if indices:
                self.group_names.append(group_name)
                self.group_indices.append(indices)
        if not self.group_indices:
            raise ValueError("No recognized feature names were provided to UniformGroupNet.")
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(len(indices), hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            for indices in self.group_indices
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        embeddings = [encoder(x[:, indices]) for encoder, indices in zip(self.encoders, self.group_indices)]
        stacked = torch.stack(embeddings, dim=1)  # [batch, groups, hidden_dim]
        fused = torch.mean(stacked, dim=1)  # Uniform averaging across groups
        return self.classifier(fused).squeeze(-1)
