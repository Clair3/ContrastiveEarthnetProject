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


class LinearRegressionBaseline(nn.Module):
    def __init__(self, data_config, config=None):

        super().__init__()
        self.veg_dim = len(data_config["vegetation"]["variables"])
        self.weather_dim = len(data_config["weather"]["variables"])
        self.veg_seq_len = data_config["vegetation"]["sequence_length"]
        self.weather_seq_len = data_config["weather"]["sequence_length"]
        input_dim = (
            self.veg_seq_len * self.veg_dim
            + 2 * self.weather_seq_len * self.weather_dim
        )
        self.linear = nn.Linear(input_dim, self.veg_seq_len * self.veg_dim)

    def forward(self, batch):
        veg = batch["vegetation_history"]
        weather = batch["weather_history"]
        forecast = batch["weather_forecast"]

        # Replace NaNs with zero
        veg = torch.nan_to_num(veg, nan=0.0)
        weather = torch.nan_to_num(weather, nan=0.0)
        forecast = torch.nan_to_num(forecast, nan=0.0)

        x = torch.cat(
            [veg.flatten(1), weather.flatten(1), forecast.flatten(1)],
            dim=-1,
        )
        out = self.linear(x)
        return out.view(veg.shape[0], self.veg_seq_len, self.veg_dim)


class MLP(nn.Module):
    def __init__(self, data_config, config=None):

        super().__init__()
        self.veg_dim = len(data_config["vegetation"]["variables"])
        self.weather_dim = len(data_config["weather"]["variables"])
        self.veg_seq_len = data_config["vegetation"]["sequence_length"]
        self.weather_seq_len = data_config["weather"]["sequence_length"]

        hidden_dim = config.hidden_dim if config is not None else 128

        input_dim = (
            self.veg_seq_len * self.veg_dim
            + 2 * self.weather_seq_len * self.weather_dim
        )
        output_dim = self.veg_seq_len * self.veg_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch):
        veg = batch["vegetation_history"]  # [B, T_veg, C_veg]
        weather = batch["weather_history"]  # [B, T_w, C_w]
        forecast = batch["weather_forecast"]  # [B, T_w, C_w]

        # Replace NaNs with zero
        veg = torch.nan_to_num(veg, nan=0.0)
        weather = torch.nan_to_num(weather, nan=0.0)
        forecast = torch.nan_to_num(forecast, nan=0.0)

        x = torch.cat(
            [veg.flatten(1), weather.flatten(1), forecast.flatten(1)],
            dim=-1,
        )
        out = self.mlp(x)
        return out.view(
            veg.shape[0], self.veg_seq_len, self.veg_dim
        )  # reshape to [B, T_veg, C_veg]


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
        emb_veg_history = self.encoder(veg_history)
        emb_weather_history = self.encoder(weather_history)
        emb_weather_future = self.encoder(weather_future)

        latent_history = torch.cat(
            [emb_veg_history, emb_weather_history], dim=-1
        )  # see cross-attention in transformer decoder for how to do this better

        return self.head(embedding)


def forward(self, batch):

    veg_hist = batch["vegetation_history"]
    weather_hist = batch["weather_history"]
    weather_future = batch["weather_future"]

    # Encode sequences
    veg_tokens = self.encoder_veg(veg_hist)  # [B, T_v, d]
    weather_tokens = self.encoder_weather(weather_hist)  # [B, T_w, d]
    weather_future_tokens = self.encoder_weather(weather_future)

    # Build memory
    memory = torch.cat([veg_tokens, weather_tokens, weather_future_tokens], dim=1)

    # Build query (future timesteps)
    B, T_future, _ = weather_future.shape
    queries = torch.zeros(B, T_future, self.d_model, device=veg_hist.device)
    queries = self.positional_encoding(queries)

    # Decode
    out = self.decoder(tgt=queries, memory=memory)

    # Predict vegetation
    pred = self.head(out)

    return pred


# todo add memba + transformer head with contrastive embeddings
