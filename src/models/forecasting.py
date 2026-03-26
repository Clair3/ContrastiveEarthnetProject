from matplotlib.pyplot import flag
import torch
import torch.nn as nn
from .encoders import TimeSeriesTransformerEncoder


class PersistenceBaseline(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        return batch["vegetation_history"]


class SeasonalBaseline(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        return batch["mean_seasonal_cycle"]


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
        return self.linear(x)


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
        return self.mlp(x)


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
    def __init__(self, data_config, config=None):
        super().__init__()
        veg_dim = len(data_config["vegetation"]["variables"])
        weather_dim = len(data_config["weather"]["variables"])
        veg_seq_len = data_config["vegetation"]["sequence_length"]
        weather_seq_len = data_config["weather"]["sequence_length"]

        if veg_seq_len != weather_seq_len:
            raise ValueError(
                f"Weather and vegetation sequence lengths must match: {veg_seq_len} vs {weather_seq_len}"
            )

        sequence_length = veg_seq_len * 2  # Usually 2 years / temporal resolution
        input_dim = veg_dim + weather_dim + 1  # +1 for year flag

        self.encoder = TimeSeriesTransformerEncoder(
            input_dim=input_dim,
            sequence_length=sequence_length,  # veg history + veg forecast
            d_model=config.d_model,
            seasonal_positional_encoding=False,
        )
        self.head = nn.Linear(config.d_model, veg_seq_len)  # predict only veg forecast
        self.model = nn.Sequential(self.encoder, self.head)

    def forward(self, batch):
        B, T, C_veg = batch["vegetation_history"].shape
        C_weather = batch["weather_history"].shape[2]

        # 1. Prepare vegetation: history + placeholder for forecast
        veg_forecast_placeholder = torch.full(
            (B, T, C_veg), float("nan"), device=batch["vegetation_history"].device
        )
        veg_seq = torch.cat(
            [batch["vegetation_history"], veg_forecast_placeholder], dim=1
        )  # [B, 2*T, C_veg]

        # Prepare weather: history + forecast
        weather_seq = torch.cat(
            [batch["weather_history"], batch["weather_forecast"]], dim=1
        )  # [B, 2*T, C_weather]

        # Year flag: 0 for history, 1 for forecast
        year_flag = torch.cat(
            [
                torch.zeros(B, T, 1, device=weather_seq.device),
                torch.ones(B, T, 1, device=weather_seq.device),
            ],
            dim=1,
        )

        weather_seq = torch.cat(
            [weather_seq, year_flag], dim=-1
        )  # # [B, 2*T, C_weather+1]

        x = torch.cat([veg_seq, weather_seq], dim=-1)
        return self.model(x)


class TransformerForecastOld(nn.Module):

    def __init__(self, data_config, config=None):
        super().__init__()
        veg_dim = len(data_config["vegetation"]["variables"])
        weather_dim = len(data_config["weather"]["variables"])
        veg_seq_len = data_config["vegetation"]["sequence_length"]
        weather_seq_len = data_config["weather"]["sequence_length"]

        if weather_seq_len != veg_seq_len:
            raise ValueError(
                f"Weather and vegetation resolution should be identical resolution for this model: {veg_seq_len} vs {weather_seq_len}"
            )

        super().__init__()
        self.encoder = TimeSeriesTransformerEncoder(
            input_dim=veg_dim + weather_dim + 1,  # +1 for year flag
            sequence_length=veg_seq_len
            * 2,  # veg history + weather history + weather forecast
            d_model=config.d_model,
            seasonal_positional_encoding=False,
            causal_mask=True,
        )
        self.head = nn.Linear(config.d_model, veg_seq_len)
        self.model = nn.Sequential(self.encoder, self.head)

    def forward(self, batch):
        veg_hist = batch["vegetation_history"]  # [B, T_veg, C_veg]
        weather_hist = batch["weather_history"]  # [B, T_w, C_w]
        forecast = batch["weather_forecast"]  # [B, T_w, C_w]

        veg_forecast = (
            torch.zeros(veg_hist.shape[0], veg_hist.shape[1], veg_hist.shape[2])
            * torch.nan
        )  # [B, T_veg, C_veg]
        veg_time_series = torch.cat(
            [veg_hist, veg_forecast], dim=1
        )  # [B, T_veg*2, C_veg]
        weather_time_series = torch.cat(
            [weather_hist, forecast], dim=1
        )  # [B, T_w*2, C_w]
        year_flag = torch.cat(
            [torch.zeros(weather_hist), torch.ones(forecast)]
        )  # [year0 + year1]
        year_flag = (
            year_flag.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)
        )  # [B, T_w*2, 1]
        weather_time_series = torch.cat([weather_time_series, year_flag], dim=-1)

        x = torch.cat(
            [veg_time_series, weather_time_series],
            dim=-1,
        )
        return self.model(x)
