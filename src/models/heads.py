import torch
import torch.nn as nn
from .encoders import TimeSeriesTransformerEncoder


class ForecastHead(nn.Module):
    def forward(self, encoded):
        raise NotImplementedError
