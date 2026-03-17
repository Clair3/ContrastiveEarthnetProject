import torch
import torch.nn as nn
import math
import numpy as np

import torch
import torch.nn as nn
import numpy as np


class SeasonalPositionalEncoding(nn.Module):
    def __init__(self, d_model, sequence_length=23):
        """
        Args:
            d_model: embedding size
            seq_len: number of timesteps (≈73)
            min_period: shortest time scale (days)
            max_period: longest time scale (days)
        """
        super().__init__()
        max_period = 365  # days
        temporal_resolution = max_period // sequence_length  # days between time steps
        days = torch.arange(sequence_length) * temporal_resolution
        min_period = (
            temporal_resolution * 2
        )  # temporal resolution (days) * 2 Nyquist constraint of Nyquist–Shannon sampling theorem

        # Log-spaced frequencies between 1/max_period and 1/min_period
        num_freqs = d_model // 2
        periods = torch.logspace(
            np.log10(min_period), np.log10(max_period), num_freqs
        )  # Creates periods of eg: [10, 15, 23, 35, 53, 81, 123, 187, 284, 365] days
        freqs = 2 * np.pi / periods  # angular frequency (radians/day)

        positional_encoder = torch.zeros(sequence_length, d_model)
        positional_encoder[:, 0::2] = torch.sin(days.unsqueeze(1) * freqs)
        positional_encoder[:, 1::2] = torch.cos(days.unsqueeze(1) * freqs)

        self.register_buffer(
            "positional_encoder", positional_encoder.unsqueeze(0)
        )  # [1, seq_len, d_model]

    def forward(self, x):
        return x + self.positional_encoder[:, : x.size(1), :]


class PositionalEncoding(nn.Module):
    """Classical Positional encoding for transformer"""

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
