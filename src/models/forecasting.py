import torch
import torch.nn as nn
from .encoders import TimeSeriesTransformerEncoder
from .positional_encoding import ClassicPositionalEncoding
import torch.nn.functional as F
import copy


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
        # print(input_dim, output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch):
        veg_hist = batch["vegetation_history"]  # [B, T_veg, C_veg]
        weather = batch["weather_history"]  # [B, T_w, C_w]
        forecast = batch["weather_forecast"]  # [B, T_w, C_w]

        smoothed = F.avg_pool1d(
            torch.nan_to_num(veg_hist, nan=0.0).transpose(1, 2),
            kernel_size=5,
            stride=1,
            padding=5 // 2,
        ).transpose(1, 2)

        # Only replace NaNs, keep original values intact
        veg_hist = torch.where(torch.isnan(veg_hist), smoothed, veg_hist)

        # Replace NaNs with zero
        veg = torch.nan_to_num(veg_hist, nan=0.0)
        weather = torch.nan_to_num(weather, nan=0.0)
        forecast = torch.nan_to_num(forecast, nan=0.0)

        x = torch.cat(
            [veg.flatten(1), weather.flatten(1), forecast.flatten(1)],
            dim=-1,
        )
        # print(x.shape)
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
        # if step is not None and total_steps is not None:
        #     # Start mostly teacher forcing, decay over time
        #     return 1 / (1 + torch.exp(5 * (step / total_steps - 0.5)))  # sigmoid
        # else:
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
        veg_hist = torch.where(torch.isnan(veg_hist), 0, veg_hist)

        if self.training:
            veg_forecast = batch["vegetation_forecast"]

        B, T_future, _ = weather_forecast.shape

        # History encoding
        x_hist = torch.cat([veg_hist, weather_hist], dim=-1)
        _, (h, c) = self.lstm(x_hist)

        input_veg = veg_hist[:, -1, :]
        preds = []

        for t in range(T_future):
            weather_step = weather_forecast[:, t, :]
            lstm_input = torch.cat([input_veg, weather_step], dim=-1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_input, (h, c))
            # Predict only the difference from previous step
            pred_step = input_veg + self.head(out.squeeze(1))
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
    def __init__(self, data_config, config, pretrained_encoders=None):
        super().__init__()
        veg_dim = len(data_config["vegetation"]["variables"])
        weather_dim = len(data_config["weather"]["variables"])
        T_veg = data_config["vegetation"]["sequence_length"]
        T_weather = data_config["weather"]["sequence_length"]
        d_model = config.d_model

        if pretrained_encoders is not None:
            print("Using pretrained encoders for forecasting...")
            self.veg_encoder = pretrained_encoders["veg"]
            self.weather_encoder = pretrained_encoders["weather"]
            self.weather_query_encoder = copy.deepcopy(pretrained_encoders["weather"])
            print("Loaded pretrained encoders for forecasting.")

        else:
            print(
                f"Initializing Transformer with d_model={config.d_model}, num_layers={config.num_layers}, dropout={config.dropout}, num_heads={config.num_heads}"
            )
            self.veg_encoder = TimeSeriesTransformerEncoder(
                input_dim=veg_dim,
                sequence_length=T_veg,
                d_model=d_model,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                dropout=config.dropout,
                use_cls=config.use_cls,
                seasonal_positional_encoding=config.seasonal_positional_encoding,
            )
            self.weather_encoder = TimeSeriesTransformerEncoder(
                input_dim=weather_dim,
                sequence_length=T_weather,
                d_model=d_model,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                dropout=config.dropout,
                use_cls=config.use_cls,
                seasonal_positional_encoding=config.seasonal_positional_encoding,
            )

        # --- Decoder: weather_forecast attends to past context ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # pre-norm: more stable with small data
        )
        # causal_mask = self.generate_causal_mask(T, x.device)
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_layers,
        )

        # --- Head ---
        self.head = nn.Linear(d_model, veg_dim)

        self.veg_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.weather_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, batch):
        # Encode past: veg and weather history → sequence of tokens
        veg_mem = self.veg_encoder(batch["vegetation_history"])  # [B, T_veg, d]
        weather_mem = self.weather_encoder(batch["weather_history"])  # [B, T_w,   d]
        veg_mem = veg_mem + self.veg_token
        weather_mem = weather_mem + self.weather_token

        # Past memory = concat along sequence dim (not feature dim)
        memory = torch.cat([veg_mem, weather_mem], dim=1)  # [B, T_veg+T_w, d]

        # Decode: weather forecast queries into past memory
        forecast_query = self.weather_encoder(batch["weather_forecast"])  # [B, T_w, d]
        out = self.decoder(tgt=forecast_query, memory=memory)  # [B, T_w, d]

        # Project to vegetation space
        out = self.head(out)  # [B, T_w, veg_dim]
        return out


class TransformerMaxEVI(nn.Module):
    def __init__(self, data_config, config, pretrained_encoders=None):
        super().__init__()
        veg_dim = len(data_config["vegetation"]["variables"])
        weather_dim = len(data_config["weather"]["variables"])
        T_veg = data_config["vegetation"]["sequence_length"]
        T_weather = data_config["weather"]["sequence_length"]
        d_model = config.d_model

        if pretrained_encoders is not None:
            print("Using pretrained encoders for forecasting...")
            self.veg_encoder = pretrained_encoders["veg"]
            self.weather_encoder = pretrained_encoders["weather"]
            self.weather_query_encoder = copy.deepcopy(pretrained_encoders["weather"])
            print("Loaded pretrained encoders for forecasting.")

        else:
            print(
                f"Initializing Transformer with d_model={config.d_model}, num_layers={config.num_layers}, dropout={config.dropout}, num_heads={config.num_heads}"
            )
            self.veg_encoder = TimeSeriesTransformerEncoder(
                input_dim=veg_dim,
                sequence_length=T_veg,
                d_model=d_model,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                dropout=config.dropout,
                use_cls=config.use_cls,
                seasonal_positional_encoding=config.seasonal_positional_encoding,
            )
            self.weather_encoder = TimeSeriesTransformerEncoder(
                input_dim=weather_dim,
                sequence_length=T_weather,
                d_model=d_model,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                dropout=config.dropout,
                use_cls=config.use_cls,
                seasonal_positional_encoding=config.seasonal_positional_encoding,
            )

        # --- Decoder: weather_forecast attends to past context ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # pre-norm: more stable with small data
        )
        # causal_mask = self.generate_causal_mask(T, x.device)
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_layers,
        )

        # --- Head ---
        self.head = nn.Linear(d_model, veg_dim)
        self.pool = nn.Linear(d_model, 1)

    def forward(self, batch):
        # Encode past: veg and weather history → sequence of tokens
        veg_mem = self.veg_encoder(batch["vegetation_history"])  # [B, T_veg, d]
        weather_mem = self.weather_encoder(batch["weather_history"])  # [B, T_w,   d]

        # Past memory = concat along sequence dim (not feature dim)
        memory = torch.cat([veg_mem, weather_mem], dim=1)  # [B, T_veg+T_w, d]

        # Decode: weather forecast queries into past memory
        forecast_query = self.weather_encoder(batch["weather_forecast"])  # [B, T_w, d]
        out = self.decoder(tgt=forecast_query, memory=memory)  # [B, T_w, d]

        # Project to vegetation space
        out = self.head(out)  # [B, T_w, veg_dim]
        weights = torch.softmax(self.pool(out), dim=1)  # [B, T, 1]
        pooled = (out * weights).sum(dim=1)
        return pooled

        # return out[:, 0, :]  # predict max EVI prediction for the forecast period
