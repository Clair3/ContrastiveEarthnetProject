import torch
import torch.nn as nn
from .encoders import TimeSeriesTransformerEncoder
import torch.nn.functional as F


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
        out = self.mlp(x)
        if out.ndim == 2:  # [B, T]
            out = out.unsqueeze(-1)
        return out


class LSTM(nn.Module):
    def __init__(self, data_config, config=None):
        super().__init__()
        veg_dim = len(data_config["vegetation"]["variables"])
        weather_dim = len(data_config["weather"]["variables"])

        input_dim = veg_dim + weather_dim
        hidden_dim = config.hidden_dim if config is not None else 128
        num_layers = config.num_layers if config is not None else 1
        dropout = config.dropout if config is not None else 0.0
        print(
            f"Initializing LSTM with hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}"
        )
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Prediction head: maps hidden state to vegetation
        self.head = nn.Linear(hidden_dim, veg_dim)

    def teacher_forcing_ratio(self, step=None, total_steps=None):
        # Compute teacher forcing probability using a sigmoid schedule
        if step is not None and total_steps is not None:
            print("YEEH")
            # Start mostly teacher forcing, decay over time
            return 1 / (1 + torch.exp(5 * (step / total_steps - 0.5)))  # sigmoid
        else:
            return 0.5

    def forward(self, batch, step=None, total_steps=None):
        """
        step: current training step (int)
        total_steps: total number of training steps (int)
        The teacher forcing ratio will be computed with a sigmoid schedule.
        """
        veg_hist = batch["vegetation_history"]
        weather_hist = batch["weather_history"]
        weather_forecast = batch["weather_forecast"]

        # Smooth NaNs in vegetation history
        smoothed = F.avg_pool1d(
            torch.nan_to_num(veg_hist, nan=0.0).transpose(1, 2),
            kernel_size=5,
            stride=1,
            padding=5 // 2,
        ).transpose(1, 2)

        # Only replace NaNs, keep original values intact
        veg_hist = torch.where(torch.isnan(veg_hist), smoothed, veg_hist)

        if self.training:
            veg_forecast = batch["vegetation_forecast"]

        B, T_future, _ = weather_forecast.shape

        # History encoding
        x_hist = torch.cat([veg_hist, weather_hist], dim=-1)
        _, (h, c) = self.lstm(x_hist)

        input_veg = veg_hist[:, -1, :]
        preds = []

        # print(
        #     torch.isnan(weather_hist).any(),
        #     torch.isnan(weather_forecast).any(),
        #     torch.isnan(veg_hist).any(),
        # )

        for t in range(T_future):
            weather_step = weather_forecast[:, t, :]
            lstm_input = torch.cat([input_veg, weather_step], dim=-1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_input, (h, c))
            pred_step = self.head(out.squeeze(1))
            preds.append(pred_step.unsqueeze(1))

            # Decide teacher forcing per batch element
            if self.training:
                tf_prob = self.teacher_forcing_ratio(step=step, total_steps=total_steps)
                mask = (torch.rand(B, device=pred_step.device) < tf_prob).unsqueeze(1)

                # Only use non-NaN values for teacher forcing
                veg_step = veg_forecast[:, t, :]

                # valid teacher forcing positions
                valid_tf = mask & (~torch.isnan(veg_step))

                input_veg = torch.where(valid_tf, veg_step, pred_step)
            else:
                input_veg = pred_step  # no teacher forcing at inference
        return torch.cat(preds, dim=1)


class TransformerBaseline(nn.Module):
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

        d_model = config.d_model if config is not None else 128
        num_layers = config.num_layers if config is not None else 2
        dropout = config.dropout if config is not None else 0.2
        num_heads = config.num_heads if config is not None else 2

        print(
            f"Initializing Transformer with d_model={d_model}, num_layers={num_layers}, dropout={dropout}, num_heads={num_heads}"
        )

        sequence_length = veg_seq_len * 2  # Usually 2 years / temporal resolution
        input_dim = veg_dim + weather_dim + 1  # +1 for year flag

        self.encoder = TimeSeriesTransformerEncoder(
            input_dim=input_dim,
            sequence_length=sequence_length,  # veg history + veg forecast
            d_model=d_model,
            nhead=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            seasonal_positional_encoding=False,
        )
        self.head = nn.Linear(config.d_model, veg_seq_len)  # predict only veg forecast
        self.model = nn.Sequential(self.encoder, self.head)

    def forward(self, batch):
        B, T, C_veg = batch["vegetation_history"].shape

        # Prepare vegetation: history + placeholder for forecast
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
        )  # [B, 2*T, C_weather+1]

        x = torch.cat([veg_seq, weather_seq], dim=-1)
        out = self.model(x)
        if out.ndim == 2:  # [B, T]
            out = out.unsqueeze(-1)
        return out
