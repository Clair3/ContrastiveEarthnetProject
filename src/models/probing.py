import torch
import torch.nn as nn
from .encoders import TimeSeriesTransformerEncoder


class RegressionHead(nn.Module):
    def __init__(self, d_model, out_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, seq):
        cls = seq[:, 0, :]
        return self.mlp(cls)


class CLSHead(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        seq = self.encoder(x)
        return seq[:, 0, :]
