import torch
import torch.nn as nn
import math
import numpy as np

import torch
import torch.nn as nn
import numpy as np


class SeasonalPositionalEncoding(nn.Module):
    def __init__(self, dim_model, temporal_resolution=5):
        """
        Args:
            d_model: embedding size
            seq_len: number of timesteps (≈73)
            min_period: shortest time scale (days)
            max_period: longest time scale (days)
        """
        super().__init__()
        max_period = 365  # days
        seq_len = max_period // temporal_resolution  # number of time steps
        days = torch.arange(seq_len) * temporal_resolution
        min_period = (
            temporal_resolution * 2
        )  # temporal resolution (days) * 2 Nyquist constraint of Nyquist–Shannon sampling theorem

        # Log-spaced frequencies between 1/max_period and 1/min_period
        num_freqs = dim_model // 2
        periods = torch.logspace(
            np.log10(min_period), np.log10(max_period), num_freqs
        )  # Creates periods: [10, 15, 23, 35, 53, 81, 123, 187, 284, 365] days (example)
        freqs = 2 * np.pi / periods  # angular frequency (radians/day)

        positional_encoder = torch.zeros(seq_len, dim_model)
        positional_encoder[:, 0::2] = torch.sin(days.unsqueeze(1) * freqs)
        positional_encoder[:, 1::2] = torch.cos(days.unsqueeze(1) * freqs)

        self.register_buffer(
            "positional_encoder", positional_encoder.unsqueeze(0)
        )  # [1, seq_len, d_model]

    def forward(self, x):
        return x + self.positional_encoder[:, : x.size(1)]


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class PositionalEncodingHybrid(nn.Module):

    def __init__(self, dim_model: int, temporal_resolution: int = 5):
        """
        Args:
            dim_model: embedding dimension of the transformer
            seq_len: number of time steps per sample (≈ 73 for 5-day data)
            period_days: annual period in days (default 365)
        """
        super().__init__()
        sequence_length = int(365 / temporal_resolution)

        # Convert time steps to real day-of-year values (every 5 days)
        days = torch.linspace(0, 365, sequence_length, dtype=torch.float32)

        # Multi-frequency encoding
        div_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-np.log(10000.0) / dim_model)
        )
        positional_encoder = torch.zeros(sequence_length, dim_model)
        positional_encoder[:, 0::2] = torch.sin(days.unsqueeze(1) * div_term)
        positional_encoder[:, 1::2] = torch.cos(days.unsqueeze(1) * div_term)

        # Explicit day-of-year periodic embedding (low-frequency annual cycle)
        # day_norm = 2 * np.pi * days / 365  # normalize to [0, 2pi]
        # doy_embedding = torch.stack([torch.sin(day_norm), torch.cos(day_norm)], dim=-1)
        # self.annual_projection = nn.Linear(2, d_model, bias=False)
        # annual = self.annual_projection(doy_embedding)

        # Combine both encodings
        positional_encoder = positional_encoder  # + annual

        # Register as buffer (not learnable)
        self.register_buffer(
            "positional_encoder", positional_encoder.unsqueeze(0)
        )  # [1, seq_len, d_model]

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape [B, T, D]
        Returns:
            x + positional encoding [:, :T, :]
        """
        T = x.size(1)
        return x + self.positional_encoder[:, :T]
