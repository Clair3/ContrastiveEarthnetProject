import torch
import torch.nn as nn
from .encoders import TimeSeriesTransformerEncoder


class PersistenceBaseline(nn.Module):
    def __init__(self, veg_dim, weather_dim, config):
        super().__init__()

    def forward(self, batch):
        veg_history = batch["vegetation_history"]
        return veg_history


class SeasonalBaseline(nn.Module):
    pass


class LSTM(nn.Module):
    def __init__(self, veg_dim, weather_dim, config):

        super().__init__()
        hidden_dim = config.hidden_dim

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

        veg_history = batch["vegetation_history"]
        weather_history = batch["weather_history"]
        weather_future = batch["weather_future"]
        emb_veg = self.encoder(veg)
        emb_weather = self.encoder(weather)

        x = torch.cat([veg, weather], dim=-1)

        embedding = self.encoder(x)

        return self.head(embedding)


# todo add memba + transformer head with contrastive embeddings
