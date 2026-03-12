import torch
import torch.nn as nn
from encoders import TimeSeriesTransformerEncoder


class PersistenceBaseline(nn.Module):

    def forward(self, batch):
        veg_history = batch["vegetation_history"]
        return veg_history


class SeasonalBaseline(nn.Module):
    pass


class LSTM(nn.Module):
    # Todo: maybe auto encoder with 4 LTSM layers, then a linear head to predict the next time step

    def __init__(self, veg_dim, weather_dim, hidden_dim):

        super().__init__()

        self.lstm = nn.LSTM(
            veg_dim + weather_dim,
            hidden_dim,
            batch_first=True,
        )

        self.head = nn.Linear(hidden_dim, veg_dim)

    def forward(self, batch):
        veg = batch["vegetation_history"]
        weather = batch["weather_history"]
        x = torch.cat([veg, weather], dim=-1)
        out, _ = self.lstm(x)
        return self.head(out)


class TransformerForecast(nn.Module):

    def __init__(self, veg_dim, weather_dim, d_model=128):

        super().__init__()

        self.encoder = TimeSeriesTransformerEncoder(
            input_dim=veg_dim + weather_dim,
            sequence_length=23,
            d_model=d_model,
        )

        self.head = nn.Linear(d_model, veg_dim)

    def forward(self, batch):

        veg = batch["vegetation_history"]
        weather = batch["weather_history"]

        x = torch.cat([veg, weather], dim=-1)

        embedding = self.encoder(x)

        return self.head(embedding)


# todo add memba + transformer head with contrastive embeddings
