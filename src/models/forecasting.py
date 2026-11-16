import torch
import torch.nn as nn

import torch.nn.functional as F
import copy
import math
import torch
import torch.nn as nn

from data.datasets import days_to_steps
from .encoders import TimeSeriesTransformerEncoder, build_encoder
from .positional_encoding import ClassicPositionalEncoding


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


class TransformerBaseline1(nn.Module):
    def __init__(self, data_config, config, pretrained_checkpoint=None):
        super().__init__()
        veg_dim = len(data_config["vegetation"]["variables"])
        weather_dim = len(data_config["weather"]["variables"])

        d_model = config.d_model

        print(
            f"Initializing Transformer with d_model={config.d_model}, num_layers={config.num_layers}, dropout={config.dropout}, num_heads={config.num_heads}"
        )
        self.veg_encoder = TimeSeriesTransformerEncoder(
            input_dim=veg_dim + 1,
            sequence_length=config.context_length,
            d_model=d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_cls=config.use_cls,
            seasonal_positional_encoding=config.seasonal_positional_encoding,
        )
        self.weather_encoder = TimeSeriesTransformerEncoder(
            input_dim=weather_dim * 2,
            sequence_length=config.context_length,
            d_model=d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_cls=config.use_cls,
            seasonal_positional_encoding=config.seasonal_positional_encoding,
        )
        self.weather_forecast_encoder = TimeSeriesTransformerEncoder(
            input_dim=weather_dim,
            sequence_length=config.prediction_length,
            d_model=d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_cls=config.use_cls,
            seasonal_positional_encoding=config.seasonal_positional_encoding,
        )

        if pretrained_checkpoint is not None:
            self.load_pretrained(pretrained_checkpoint)

        if pretrained_checkpoint is not None:
            print("Using pretrained encoders for forecasting...")
            ckpt = torch.load(pretrained_checkpoint, map_location="cpu")

            state_dict = ckpt["state_dict"]

            # self.veg_encoder.load_state_dict(
            #     extract_encoder_weights(
            #         state_dict,
            #         "encoder_veg"
            #     )
            # )
            #
            # self.weather_encoder.load_state_dict(
            #     extract_encoder_weights(
            #         state_dict,
            #         "encoder_weather"
            #     )
            # )
            # ckpt_path = CHECKPOINT_DIR / run_name / "best.ckpt"
            # model = self.build_model()
            # ckpt = torch.load(ckpt_path, map_location="cpu")
            # model.load_state_dict(ckpt["state_dict"])
            # self.veg_encoder = pretrained_encoders["veg"]
            # self.weather_encoder = pretrained_encoders["weather"]
            # self.weather_query_encoder = copy.deepcopy(pretrained_encoders["weather"])
            # for param in self.veg_encoder.parameters():
            #     param.requires_grad = False
            #
            # for param in self.weather_encoder.parameters():
            #     param.requires_grad = False
            #
            # for param in self.weather_query_encoder.parameters():
            #     param.requires_grad = False
            print("Loaded pretrained encoders for forecasting.")

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

    def forward(self, batch):

        # Encode past: veg and weather history → sequence of tokens
        # batch["msc"] = torch.zeros_like(batch["msc"])
        # veg_input = batch["vegetation_history"]
        veg_input = torch.cat(
            [
                batch["vegetation_history"],
                batch["msc"],
            ],
            dim=-1,
        )
        veg_mem = self.veg_encoder(veg_input)  # [B, T_veg, d]
        weather_mem = self.weather_encoder(batch["weather_history"])  # [B, T_w,   d]

        # Past memory = concat along sequence dim (not feature dim)
        memory = torch.cat([veg_mem, weather_mem], dim=1)  # [B, T_veg+T_w, d]

        # Decode: weather forecast queries into past memory
        forecast_query = self.weather_forecast_encoder(
            batch["weather_forecast"]
        )  # [B, T_w, d]
        out = self.decoder(tgt=forecast_query, memory=memory)  # [B, T_w, d]

        # Project to vegetation space
        out = self.head(out)  # [B, T_w, veg_dim]
        return out


class TransformerBaseline(nn.Module):
    def __init__(self, data_config, config, pretrained_checkpoint=None):
        super().__init__()

        veg_dim = len(data_config["vegetation"]["variables"])
        weather_dim = len(data_config["weather"]["variables"])
        climate_dim = veg_dim + weather_dim

        d_model = config.d_model

        print(
            f"Initializing Transformer with "
            f"d_model={config.d_model}, "
            f"num_layers={config.num_layers}, "
            f"dropout={config.dropout}, "
            f"num_heads={config.num_heads}"
        )

        # ------------------------------------------------------------------
        # Hist encoder
        # ------------------------------------------------------------------
        self.hist_encoder = TimeSeriesTransformerEncoder(
            input_dim=climate_dim,
            sequence_length=config.context_length,  # // 5,
            d_model=d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_cls=config.use_cls,
            seasonal_positional_encoding=False,  # config.seasonal_positional_encoding,
        )

        # ------------------------------------------------------------------
        # Climate encoder
        # ------------------------------------------------------------------
        self.climate_encoder = TimeSeriesTransformerEncoder(
            input_dim=climate_dim,
            sequence_length=365,  # // 5,
            d_model=d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_cls=config.use_cls,
            seasonal_positional_encoding=False,
        )

        # ------------------------------------------------------------------
        # Forecast weather encoder
        # ------------------------------------------------------------------
        self.weather_forecast_encoder = TimeSeriesTransformerEncoder(
            input_dim=weather_dim,
            sequence_length=config.prediction_length,
            d_model=d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_cls=config.use_cls,
            seasonal_positional_encoding=False,  # config.seasonal_positional_encoding,
        )

        if pretrained_checkpoint is not None:
            self.load_pretrained(pretrained_checkpoint)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_layers,
        )

        self.head = nn.Linear(d_model, veg_dim)

    def forward(self, batch):

        # -------------------------------------------------------
        # Climate input
        # [B,365,veg_dim+weather_dim]
        # -------------------------------------------------------
        climate = torch.cat(
            [
                batch["msc"],
                batch["weather_msc"],
            ],
            dim=-1,
        )

        hist = torch.cat(
            [
                batch["vegetation_history"],
                batch["weather_history"],
            ],
            dim=-1,
        )

        # -------------------------------------------------------
        # Encode memories
        # -------------------------------------------------------
        hist_mem = self.hist_encoder(hist)

        climate_mem = self.climate_encoder(climate)

        # -------------------------------------------------------
        # Memory for decoder
        # -------------------------------------------------------
        memory = torch.cat(
            [
                hist_mem,
                climate_mem,
            ],
            dim=1,
        )

        # -------------------------------------------------------
        # Forecast queries
        # -------------------------------------------------------
        forecast_query = self.weather_forecast_encoder(batch["weather_forecast"])

        # -------------------------------------------------------
        # Decode
        # -------------------------------------------------------
        out = self.decoder(
            tgt=forecast_query,
            memory=memory,
        )

        return self.head(out)


class DayOfYearEncoding(nn.Module):
    """Stateless: turns integer day-of-year (0-364) into 2 channels (sin, cos)."""

    def __init__(self, period: int = 365):
        super().__init__()
        self.period = period

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        # doy: [B, T] integer or float day-of-year
        angle = 2 * math.pi * doy.float() / self.period
        return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)  # [B, T, 2]


class TransformerDecoder(nn.Module):
    def __init__(self, data_config, config, pretrained_checkpoint=None):
        super().__init__()

        veg_dim = len(data_config["vegetation"]["variables"])
        weather_dim = len(data_config["weather"]["variables"])
        climate_dim = veg_dim + weather_dim

        d_model = config.d_model
        self.prediction_length = config.prediction_length

        print(
            f"Initializing Transformer with "
            f"d_model={config.d_model}, "
            f"num_layers={config.num_layers}, "
            f"dropout={config.dropout}, "
            f"num_heads={config.num_heads}"
        )

        self.doy_encoding = DayOfYearEncoding(period=365)

        # ------------------------------------------------------------------
        # Hist encoder (past veg + weather residuals) — variable length,
        # supports > 1 year via max_context_length + padding mask
        # ------------------------------------------------------------------
        self.hist_encoder = TimeSeriesTransformerEncoder(
            input_dim=climate_dim + 2,  # +2 for sin/cos doy channels
            sequence_length=config.context_length,  #  config.max_context_length,
            d_model=d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_cls=False,
            seasonal_positional_encoding=False,
        )

        # ------------------------------------------------------------------
        # Climate encoder (mean seasonal cycle, veg + weather, 365 days)
        # ------------------------------------------------------------------
        self.climate_encoder = TimeSeriesTransformerEncoder(
            input_dim=climate_dim + 2,
            sequence_length=365,
            d_model=d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_cls=False,
            seasonal_positional_encoding=False,
        )

        # ------------------------------------------------------------------
        # Forecast weather encoder — becomes the basis of the query
        # ------------------------------------------------------------------
        self.weather_forecast_encoder = TimeSeriesTransformerEncoder(
            input_dim=weather_dim + 2,
            sequence_length=config.prediction_length,
            d_model=d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_cls=False,
            seasonal_positional_encoding=False,
        )

        if pretrained_checkpoint is not None:
            self.load_pretrained(pretrained_checkpoint)

        # ------------------------------------------------------------------
        # Type embeddings: tag memory tokens by source (0=hist, 1=climate)
        # ------------------------------------------------------------------
        self.type_embed = nn.Embedding(2, d_model)

        # Learnable per-lead-time query prior, added to weather-forecast
        # encoding so the query isn't *purely* a function of weather content
        self.query_pos = nn.Parameter(
            torch.randn(1, config.prediction_length, d_model) * 0.02
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_layers
        )

        self.head = nn.Linear(d_model, veg_dim)

    def _with_doy(self, x: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
        """Append sin/cos day-of-year channels to a [B, T, C] input."""
        return torch.cat([x, self.doy_encoding(doy)], dim=-1)

    def forward(self, batch):
        B = batch["vegetation_history"].shape[0]

        # -------------------------------------------------------
        # History (variable length; may span > 1 year)
        # batch["hist_doy"]: [B, T_hist] calendar day-of-year per step
        # batch["hist_mask"]: [B, T_hist] True where PADDED (optional)
        # -------------------------------------------------------
        hist = torch.cat(
            [batch["vegetation_history"], batch["weather_history"]], dim=-1
        )
        hist = self._with_doy(hist, batch["hist_doy"])
        hist_mem = self.hist_encoder(
            hist
        )  # , src_key_padding_mask=batch.get("hist_mask"))
        hist_mem = hist_mem + self.type_embed.weight[0].view(1, 1, -1)

        # -------------------------------------------------------
        # Climatology (365 days, shared calendar across batch)
        # -------------------------------------------------------
        climate = torch.cat([batch["msc"], batch["weather_msc"]], dim=-1)
        climate = self._with_doy(climate, batch["climate_doy"])
        climate_mem = self.climate_encoder(climate)
        climate_mem = climate_mem + self.type_embed.weight[1].view(1, 1, -1)

        # -------------------------------------------------------
        # Context memory: history + climatology, type-tagged
        # -------------------------------------------------------
        memory = torch.cat([hist_mem, climate_mem], dim=1)

        # -------------------------------------------------------
        # Query: forecast weather (known future condition) + learnable
        # per-lead-time prior. batch["forecast_doy"]: [B, prediction_length]
        # -------------------------------------------------------
        forecast_weather = self._with_doy(
            batch["weather_forecast"], batch["forecast_doy"]
        )
        forecast_mem = self.weather_forecast_encoder(forecast_weather)
        query = forecast_mem + self.query_pos.expand(B, -1, -1)

        # -------------------------------------------------------
        # Decode
        # -------------------------------------------------------
        out = self.decoder(tgt=query, memory=memory)

        return self.head(out)


# ----------------------------------------------------------------------
# Simple TimeMixer block
# ----------------------------------------------------------------------
class MixerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        self.time_norm = nn.LayerNorm(d_model)
        self.feature_norm = nn.LayerNorm(d_model)

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

        self.feature_mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        # TODO: maybe something weird here to check
        # temporal mixing
        y = self.time_norm(x)
        y = y.transpose(1, 2)
        y = self.time_mlp(y.transpose(1, 2))
        x = x + y

        # feature mixing
        y = self.feature_norm(x)
        y = self.feature_mlp(y)
        x = x + y

        return x


# ----------------------------------------------------------------------
# Forecast model
# ----------------------------------------------------------------------
class TransformerTimeMixer(nn.Module):

    def __init__(self, data_config, config):

        super().__init__()

        veg_dim = len(data_config["vegetation"]["variables"])
        weather_dim = len(data_config["weather"]["variables"])
        climate_dim = veg_dim + weather_dim

        d_model = config.d_model

        self.doy_encoding = DayOfYearEncoding()

        # --------------------------------------------------------------
        # Encoders
        # --------------------------------------------------------------
        self.hist_encoder = TimeSeriesTransformerEncoder(
            input_dim=climate_dim + 2,
            sequence_length=config.context_length,
            d_model=d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_cls=False,
            seasonal_positional_encoding=False,
        )

        self.weather_encoder = TimeSeriesTransformerEncoder(
            input_dim=weather_dim + 2,
            sequence_length=config.prediction_length,
            d_model=d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_cls=False,
            seasonal_positional_encoding=False,
        )

        # self.climate_encoder = TimeSeriesTransformerEncoder(
        #     input_dim=climate_dim + 2,
        #     sequence_length=365,
        #     d_model=d_model,
        #     num_heads=config.num_heads,
        #     num_layers=config.num_layers,
        #     dropout=config.dropout,
        #     use_cls=False,
        #     seasonal_positional_encoding=False,
        # )

        # --------------------------------------------------------------
        # Single cross-attention
        # --------------------------------------------------------------
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # --------------------------------------------------------------
        # Fusion
        # --------------------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.mixers = nn.Sequential(
            *[
                MixerBlock(
                    d_model=d_model,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.head = nn.Linear(d_model, veg_dim)

    def _with_doy(self, x, doy):
        return torch.cat(
            [x, self.doy_encoding(doy)],
            dim=-1,
        )

    def forward(self, batch):

        # --------------------------------------------------------------
        # History
        # --------------------------------------------------------------
        hist = torch.cat(
            [
                batch["vegetation_history"],
                batch["weather_history"],
            ],
            dim=-1,
        )

        hist = self._with_doy(hist, batch["hist_doy"])

        hist_features = self.hist_encoder(hist)

        # --------------------------------------------------------------
        # Future weather
        # --------------------------------------------------------------
        weather = self._with_doy(
            batch["weather_forecast"],
            batch["forecast_doy"],
        )

        weather_features = self.weather_encoder(weather)

        # --------------------------------------------------------------
        # Climate
        # --------------------------------------------------------------
        # climate = torch.cat(
        #     [
        #         batch["msc"],
        #         batch["weather_msc"],
        #     ],
        #     dim=-1,
        # )

        # climate = self._with_doy(
        #    climate,
        #    batch["climate_doy"],
        # )
        #
        # climate_features = self.climate_encoder(climate)

        # --------------------------------------------------------------
        # Align climatology
        #
        # forecast_doy contains values in [0,364]
        # --------------------------------------------------------------
        # idx = batch["forecast_doy"].long()
        #
        # idx = idx.unsqueeze(-1).expand(
        #     -1,
        #     -1,
        #     climate_features.shape[-1],
        # )
        #
        # aligned_climate = torch.gather(
        #     climate_features,
        #     dim=1,
        #     index=idx,
        # )

        # --------------------------------------------------------------
        # Retrieve relevant history
        # --------------------------------------------------------------
        history_context, _ = self.cross_attention(
            query=weather_features,
            key=hist_features,
            value=hist_features,
        )

        # --------------------------------------------------------------
        # Fuse everything
        # --------------------------------------------------------------
        x = torch.cat(
            [
                weather_features,
                history_context,
                # aligned_climate,
            ],
            dim=-1,
        )

        x = self.fusion(x)

        # --------------------------------------------------------------
        # Mixer
        # --------------------------------------------------------------
        x = self.mixers(x)

        return self.head(x)



class SingleStreamForecastModel(nn.Module):
    def __init__(self, data_config, config):
        super().__init__()

        veg_dim = len(data_config["vegetation"]["variables"])
        weather_dim = len(data_config["weather"]["variables"])

        context_length = days_to_steps(
            config.context_days,
            data_config["sampling_days"],
        )
        prediction_length = days_to_steps(
            config.prediction_days,
            data_config["sampling_days"],
        )

        self.doy_encoding = DayOfYearEncoding()

        # EVI + meteorology + EVI missing flags + 2 DOY features
        input_dim = 2 * veg_dim + weather_dim + 2

        self.encoder = build_encoder(
            name="transformer_nans_flags",
            input_dim=input_dim,
            sequence_length=context_length + prediction_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        self.head = nn.Linear(config.d_model, veg_dim)

    def forward(self, batch):

        vegetation_history = batch["vegetation_history"]
        weather_history = batch["weather_history"]
        weather_future = batch["weather_forecast"]

        B, H, _ = weather_future.shape

        # This implementation assumes complete meteorological inputs.
        # if (
        #     torch.isnan(weather_history).any()
        #     or torch.isnan(weather_future).any()
        # ):
        #     raise ValueError("Meteorological inputs contain NaNs.")

        # --------------------------------------------------
        # Historical sequence
        # --------------------------------------------------

        veg_missing = torch.isnan(vegetation_history)

        veg_filled = torch.nan_to_num(
            vegetation_history, nan=0.0
        )

        history = torch.cat(
            [
                veg_filled,
                weather_history,
                veg_missing.to(vegetation_history.dtype),
                self.doy_encoding(batch["hist_doy"]),
            ],
            dim=-1,
        )

        # --------------------------------------------------
        # Future sequence
        # --------------------------------------------------

        # Future EVI is unknown, not observed.
        future_veg = vegetation_history.new_zeros(
            B, H, vegetation_history.shape[-1]
        )

        # Flag every future EVI value as missing.
        future_flags = torch.ones_like(future_veg)

        future = torch.cat(
            [
                future_veg,
                weather_future,
                future_flags,
                self.doy_encoding(batch["forecast_doy"]),
            ],
            dim=-1,
        )

        # --------------------------------------------------
        # Single sequence: past followed by future
        # --------------------------------------------------

        sequence = torch.cat(
            [history, future],
            dim=1,
        )

        # --------------------------------------------------
        # Transformer
        # --------------------------------------------------

        features = self.encoder(sequence)

        # --------------------------------------------------
        # Predict only the future EVI
        # --------------------------------------------------

        future_features = features[:, -H:, :]

        prediction = self.head(future_features)

        return prediction


class GatedFusion(nn.Module):
    """
    Gated residual fusion between future features
    and historical context.
    """

    def __init__(self, d_model, dropout):
        super().__init__()

        self.projection = nn.Linear(2 * d_model, d_model)
        self.gate = nn.Linear(2 * d_model, d_model)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, future, history):

        combined = torch.cat([future, history], dim=-1)

        candidate = self.activation(self.projection(combined))
        gate = torch.sigmoid(self.gate(combined))

        # Interpolate between fused information and
        # the original future representation.
        x = gate * candidate + (1.0 - gate) * future

        return self.dropout(x)


class TemporalConvBlock(nn.Module):
    """
    Lightweight residual temporal CNN.
    Preserves [B, T, D].
    """

    def __init__(self, d_model, dropout):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)

        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=3,
            padding=1,
            groups=d_model,
        )

        self.pointwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=1,
        )

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.norm(x)
        x = x.transpose(1, 2)

        x = self.depthwise(x)
        x = self.activation(x)
        x = self.pointwise(x)

        x = x.transpose(1, 2)

        return residual + self.dropout(x)


import torch
import torch.nn as nn


class ForecastModelOld(nn.Module):

    ABLATIONS = {
        "full": {},
        "separate_history": {
            "history_mode": "separate",
        },
        "linear_future_encoder": {
            "future_encoder_mode": "linear",
        },
        "mean_history": {
            "history_retrieval": "mean",
        },
        "no_mixer": {
            "decoder_mode": "none",
        },
        "no_hist_weather": {
            "use_hist_weather": False,
        },
        "no_future_weather": {
            "use_future_weather": False,
        },
        "gated_fusion": {
            "fusion_mode": "gated",
        },
        "cnn_decoder": {
            "decoder_mode": "cnn",
        },
        "no_fusion": {"fusion_mode": "none"},
    }

    def __init__(self, data_config, config):
        super().__init__()

        # --------------------------------------------------
        # Ablation configuration
        # --------------------------------------------------

        self.ablation = config.get("ablation", "full")

        if self.ablation not in self.ABLATIONS:
            raise ValueError(
                f"Unknown ablation: {self.ablation}. "
                f"Available: {list(self.ABLATIONS)}"
            )

        options = {
            "history_mode": "shared",
            "future_encoder_mode": "temporal",
            "history_retrieval": "cross_attention",
            "fusion_mode": "concat",
            "decoder_mode": "mixer",
            "use_hist_weather": True,
            "use_future_weather": True,
        }

        options.update(self.ABLATIONS[self.ablation])

        self.history_mode = options["history_mode"]
        self.future_encoder_mode = options["future_encoder_mode"]
        self.history_retrieval = options["history_retrieval"]
        self.fusion_mode = options["fusion_mode"]
        self.decoder_mode = options["decoder_mode"]
        self.use_hist_weather = options["use_hist_weather"]
        self.use_future_weather = options["use_future_weather"]

        # --------------------------------------------------
        # Dimensions
        # --------------------------------------------------

        veg_dim = len(data_config["vegetation"]["variables"])

        weather_dim = len(data_config["weather"]["variables"])

        context_length = days_to_steps(
            config.context_days,
            data_config["sampling_days"],
        )

        prediction_length = days_to_steps(
            config.prediction_days,
            data_config["sampling_days"],
        )

        d_model = config.d_model
        dropout = config.dropout

        self.doy_encoding = DayOfYearEncoding()

        # Original implementation assumes 2 DOY features.
        doy_dim = 2

        # --------------------------------------------------
        # Encoder configuration
        # --------------------------------------------------

        encoder_kwargs = {
            "num_layers": config.num_layers,
        }

        if config.encoder_name == "transformer":

            encoder_kwargs.update(
                {
                    "num_heads": config.num_heads,
                    "dropout": config.dropout,
                    "seasonal_positional_encoding": config.seasonal_positional_encoding,
                    "use_cls": False,
                }
            )

        elif config.encoder_name == "mamba":

            encoder_kwargs.update(
                {
                    "mamba_d_state": config.mamba_d_state,
                    "mamba_d_conv": config.mamba_d_conv,
                    "mamba_expand": config.mamba_expand,
                    "mamba_headdim": config.mamba_headdim,
                }
            )

        # --------------------------------------------------
        # Historical encoder
        # --------------------------------------------------

        if self.history_mode == "shared":

            hist_input_dim = veg_dim + doy_dim

            if self.use_hist_weather:
                hist_input_dim += weather_dim

            self.hist_encoder = build_encoder(
                name=config.encoder_name,
                input_dim=hist_input_dim,
                sequence_length=context_length,
                d_model=d_model,
                **encoder_kwargs,
            )

        elif self.history_mode == "separate":

            if not self.use_hist_weather:
                raise ValueError(
                    "Separate history encoders require " "historical weather."
                )

            self.veg_hist_encoder = build_encoder(
                name=config.encoder_name,
                input_dim=veg_dim + doy_dim,
                sequence_length=context_length,
                d_model=d_model,
                **encoder_kwargs,
            )

            self.weather_hist_encoder = build_encoder(
                name=config.encoder_name,
                input_dim=weather_dim + doy_dim,
                sequence_length=context_length,
                d_model=d_model,
                **encoder_kwargs,
            )

            self.hist_fusion = nn.Linear(
                2 * d_model,
                d_model,
            )

        # --------------------------------------------------
        # Future weather encoder
        # --------------------------------------------------

        future_input_dim = doy_dim

        if self.use_future_weather:
            future_input_dim += weather_dim

        if self.future_encoder_mode == "linear":

            self.weather_encoder = nn.Linear(
                future_input_dim,
                d_model,
            )

        elif self.future_encoder_mode == "temporal":

            self.weather_encoder = build_encoder(
                name=config.encoder_name,
                input_dim=future_input_dim,
                sequence_length=prediction_length,
                d_model=d_model,
                **encoder_kwargs,
            )

        # --------------------------------------------------
        # Historical retrieval
        # --------------------------------------------------

        if self.history_retrieval == "cross_attention":

            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=config.num_heads,
                dropout=dropout,
                batch_first=True,
            )

        # --------------------------------------------------
        # Fusion
        # --------------------------------------------------

        if self.fusion_mode == "concat":

            self.fusion = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        elif self.fusion_mode == "gated":

            self.fusion = GatedFusion(
                d_model=d_model,
                dropout=dropout,
            )

        # --------------------------------------------------
        # Forecast decoder
        # --------------------------------------------------

        if self.decoder_mode == "mixer":

            self.mixers = nn.Sequential(
                *[
                    MixerBlock(
                        d_model=d_model,
                        dropout=dropout,
                    )
                    for _ in range(config.num_layers)
                ]
            )

        elif self.decoder_mode == "cnn":

            cnn_layers = config.get(
                "cnn_layers",
                config.num_layers,
            )

            self.mixers = nn.Sequential(
                *[
                    TemporalConvBlock(
                        d_model=d_model,
                        dropout=dropout,
                    )
                    for _ in range(cnn_layers)
                ]
            )

        elif self.decoder_mode == "none":

            self.mixers = nn.Identity()

        # --------------------------------------------------
        # Prediction head
        # --------------------------------------------------

        self.head = nn.Linear(
            d_model,
            veg_dim,
        )

    def _with_doy(self, x, doy):

        return torch.cat(
            [x, self.doy_encoding(doy)],
            dim=-1,
        )

    def _encode_history(self, batch):

        hist_doy = batch["hist_doy"]

        # --------------------------------------------------
        # Shared historical encoder
        # --------------------------------------------------

        if self.history_mode == "shared":

            hist_inputs = [
                batch["vegetation_history"],
            ]

            if self.use_hist_weather:
                hist_inputs.append(batch["weather_history"])

            hist = torch.cat(
                hist_inputs,
                dim=-1,
            )

            hist = self._with_doy(
                hist,
                hist_doy,
            )

            return self.hist_encoder(hist)

        # --------------------------------------------------
        # Separate historical encoders
        # --------------------------------------------------

        veg_hist = self._with_doy(
            batch["vegetation_history"],
            hist_doy,
        )

        weather_hist = self._with_doy(
            batch["weather_history"],
            hist_doy,
        )

        veg_features = self.veg_hist_encoder(veg_hist)

        weather_features = self.weather_hist_encoder(weather_hist)

        hist_features = torch.cat(
            [veg_features, weather_features],
            dim=-1,
        )

        return self.hist_fusion(hist_features)

    def _encode_future(self, batch):

        forecast_doy = batch["forecast_doy"]

        # Always retain known calendar information.
        future_inputs = [self.doy_encoding(forecast_doy)]

        if self.use_future_weather:

            future_inputs.insert(
                0,
                batch["weather_forecast"],
            )

        future = torch.cat(
            future_inputs,
            dim=-1,
        )

        return self.weather_encoder(future)

    def _retrieve_history(
        self,
        hist_features,
        weather_features,
    ):

        # --------------------------------------------------
        # Global historical context
        # --------------------------------------------------

        if self.history_retrieval == "mean":

            context = hist_features.mean(
                dim=1,
                keepdim=True,
            )

            return context.expand(
                -1,
                weather_features.shape[1],
                -1,
            )

        # --------------------------------------------------
        # Horizon-dependent cross-attention
        # --------------------------------------------------

        context, _ = self.cross_attention(
            query=weather_features,
            key=hist_features,
            value=hist_features,
            need_weights=False,
        )

        return context

    def forward(self, batch):

        # [B, context_length, d_model]
        hist_features = self._encode_history(batch)

        # [B, prediction_length, d_model]
        weather_features = self._encode_future(batch)

        # [B, prediction_length, d_model]
        history_context = self._retrieve_history(
            hist_features,
            weather_features,
        )

        # --------------------------------------------------
        # Fuse history and future information
        # --------------------------------------------------

        if self.fusion_mode == "concat":

            x = torch.cat(
                [weather_features, history_context],
                dim=-1,
            )

            x = self.fusion(x)
        elif self.fusion_mode == "None":
            x = history_context
        else:

            x = self.fusion(
                weather_features,
                history_context,
            )

        # --------------------------------------------------
        # Temporal forecasting
        # --------------------------------------------------

        x = self.mixers(x)

        # [B, prediction_length, veg_dim]
        return self.head(x)


class ForecastModelWithNaNsFlags(nn.Module):

    def __init__(self, data_config, config):

        super().__init__()

        veg_dim = len(data_config["vegetation"]["variables"])

        weather_dim = len(data_config["weather"]["variables"])

        context_length = days_to_steps(
            config.context_days,
            data_config["sampling_days"],
        )

        prediction_length = days_to_steps(
            config.prediction_days,
            data_config["sampling_days"],
        )

        climate_dim = veg_dim + weather_dim

        d_model = config.d_model

        # --------------------------------------------------------------
        # Day-of-year encoding
        # --------------------------------------------------------------

        self.doy_encoding = DayOfYearEncoding()

        encoder_kwargs = {
            "num_layers": config.num_layers,
        }

        print("encoder_name:", config.encoder_name)
        print("encoder_kwargs:", encoder_kwargs)

        if config.encoder_name == "transformer":
            encoder_kwargs.update(
                {
                    "num_heads": config.num_heads,
                    "dropout": config.dropout,
                    "seasonal_positional_encoding": config.seasonal_positional_encoding,
                }
            )

        elif config.encoder_name == "mamba":
            encoder_kwargs.update(
                {
                    "mamba_d_state": config.mamba_d_state,
                    "mamba_d_conv": config.mamba_d_conv,
                    "mamba_expand": config.mamba_expand,
                    "mamba_headdim": config.mamba_headdim,
                }
            )
        print("encoder_name:", config.encoder_name)
        print("encoder_kwargs:", encoder_kwargs)
        # --------------------------------------------------------------
        # History
        # --------------------------------------------------------------

        self.hist_encoder = build_encoder(
            name=config.encoder_name,
            input_dim=climate_dim + veg_dim + 2,
            sequence_length=context_length,
            d_model=d_model,
            **encoder_kwargs,
        )

        # --------------------------------------------------------------
        # Future weather
        # --------------------------------------------------------------

        self.weather_encoder = build_encoder(
            name=config.encoder_name,
            input_dim=weather_dim + 2,
            sequence_length=prediction_length,
            d_model=d_model,
            **encoder_kwargs,
        )

        # --------------------------------------------------------------
        # Cross-attention
        # --------------------------------------------------------------

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # --------------------------------------------------------------
        # Fusion
        # --------------------------------------------------------------

        self.fusion = nn.Sequential(
            nn.Linear(
                2 * d_model,
                d_model,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # --------------------------------------------------------------
        # Prediction head
        # --------------------------------------------------------------

        self.head = nn.Linear(
            d_model,
            veg_dim,
        )

    def _with_doy(self, x, doy):

        return torch.cat(
            [
                x,
                self.doy_encoding(doy),
            ],
            dim=-1,
        )

    def forward(self, batch):

        # ==============================================================
        # History
        # ==============================================================

        vegetation_history = batch["vegetation_history"]

        weather_history = batch["weather_history"]

        # Identify missing EVI observations
        veg_missing = torch.isnan(vegetation_history)

        # Replace missing EVI with zero in normalized space
        veg_filled = torch.nan_to_num(vegetation_history, nan=0.0)

        # --------------------------------------------------------------
        # Concatenate normalized history
        # --------------------------------------------------------------

        # Concatenate EVI, weather and missingness indicator
        hist = torch.cat(
            [
                veg_filled,
                weather_history,
                veg_missing.to(vegetation_history.dtype),
            ],
            dim=-1,
        )

        hist = self._with_doy(
            hist,
            batch["hist_doy"],
        )

        hist_features = self.hist_encoder(hist)

        # ==============================================================
        # Future weather
        # ==============================================================

        weather = batch["weather_forecast"]

        weather = self._with_doy(
            weather,
            batch["forecast_doy"],
        )

        weather_features = self.weather_encoder(weather)

        # ==============================================================
        # Retrieve relevant history
        # ==============================================================

        history_context, _ = self.cross_attention(
            query=weather_features,
            key=hist_features,
            value=hist_features,
        )

        # ==============================================================
        # Fusion
        # ==============================================================

        x = torch.cat(
            [
                weather_features,
                history_context,
                # aligned_climate,
            ],
            dim=-1,
        )

        x = self.fusion(x)

        # ==============================================================
        # Forecast in normalized vegetation space
        # ==============================================================

        prediction = self.head(x)

        return prediction



class ForecastModel(nn.Module):

    def __init__(self, data_config, config):

        super().__init__()

        veg_dim = len(data_config["vegetation"]["variables"])

        weather_dim = len(data_config["weather"]["variables"])

        context_length = days_to_steps(
            config.context_days,
            data_config["sampling_days"],
        )

        prediction_length = days_to_steps(
            config.prediction_days,
            data_config["sampling_days"],
        )

        climate_dim = veg_dim + weather_dim

        d_model = config.d_model

        # --------------------------------------------------------------
        # Day-of-year encoding
        # --------------------------------------------------------------

        self.doy_encoding = DayOfYearEncoding()

        encoder_kwargs = {
            "num_layers": config.num_layers,
        }

        print("encoder_name:", config.encoder_name)
        print("encoder_kwargs:", encoder_kwargs)

        if config.encoder_name == "transformer":
            encoder_kwargs.update(
                {
                    "num_heads": config.num_heads,
                    "dropout": config.dropout,
                    "seasonal_positional_encoding": config.seasonal_positional_encoding,
                    "use_cls": (
                        config.get("use_cls", False)
                        if isinstance(config, dict)
                        else config.use_cls
                    ),
                }
            )

        elif config.encoder_name == "mamba":
            encoder_kwargs.update(
                {
                    "mamba_d_state": config.mamba_d_state,
                    "mamba_d_conv": config.mamba_d_conv,
                    "mamba_expand": config.mamba_expand,
                    "mamba_headdim": config.mamba_headdim,
                }
            )
        print("encoder_name:", config.encoder_name)
        print("encoder_kwargs:", encoder_kwargs)
        # --------------------------------------------------------------
        # History
        # --------------------------------------------------------------

        self.hist_encoder = build_encoder(
            name=config.encoder_name,
            input_dim=climate_dim + 2,
            sequence_length=context_length,
            d_model=d_model,
            **encoder_kwargs,
        )

        # --------------------------------------------------------------
        # Future weather
        # --------------------------------------------------------------

        self.weather_encoder = build_encoder(
            name=config.encoder_name,
            input_dim=weather_dim + 2,
            sequence_length=prediction_length,
            d_model=d_model,
            **encoder_kwargs,
        )

        # --------------------------------------------------------------
        # Cross-attention
        # --------------------------------------------------------------

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # --------------------------------------------------------------
        # Fusion
        # --------------------------------------------------------------

        self.fusion = nn.Sequential(
            nn.Linear(
                2 * d_model,
                d_model,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # --------------------------------------------------------------
        # Prediction head
        # --------------------------------------------------------------

        self.head = nn.Linear(
            d_model,
            veg_dim,
        )

    def _with_doy(self, x, doy):

        return torch.cat(
            [
                x,
                self.doy_encoding(doy),
            ],
            dim=-1,
        )

    def forward(self, batch):

        # ==============================================================
        # History
        # ==============================================================

        vegetation_history = batch["vegetation_history"]

        weather_history = batch["weather_history"]

        # --------------------------------------------------------------
        # Concatenate normalized history
        # --------------------------------------------------------------

        hist = torch.cat(
            [
                vegetation_history,
                weather_history,
            ],
            dim=-1,
        )

        hist = self._with_doy(
            hist,
            batch["hist_doy"],
        )

        hist_features = self.hist_encoder(hist)

        # ==============================================================
        # Future weather
        # ==============================================================

        weather = batch["weather_forecast"]

        weather = self._with_doy(
            weather,
            batch["forecast_doy"],
        )

        weather_features = self.weather_encoder(weather)

        # ==============================================================
        # Retrieve relevant history
        # ==============================================================

        history_context, _ = self.cross_attention(
            query=weather_features,
            key=hist_features,
            value=hist_features,
        )

        # ==============================================================
        # Fusion
        # ==============================================================

        # x = torch.cat(
        #     [
        #         weather_features,
        #         history_context,
        #     ],
        #     dim=-1,
        # )

        x = history_context #self.fusion(x)

        # ==============================================================
        # Forecast in normalized vegetation space
        # ==============================================================

        prediction = self.head(x)

        return prediction



class RAFT(nn.Module):
    """
    RAFT baseline adapted to the project's ForecastingDataModule interface.

    Input:
        batch["vegetation_history"]:
            [B, context_length, vegetation_dim]

        batch["retrieval"]:
            [B, topm, prediction_length, vegetation_dim]

        batch["retrieval_weights"]:
            [B, topm]

    Output:
        [B, prediction_length, vegetation_dim]
    """

    def __init__(self, data_config, config):
        super().__init__()

        veg_dim = len(data_config["vegetation"]["variables"])

        context_length = days_to_steps(
            config.context_days,
            data_config["sampling_days"],
        )

        prediction_length = days_to_steps(
            config.prediction_days,
            data_config["sampling_days"],
        )

        self.veg_dim = veg_dim
        self.context_length = context_length
        self.prediction_length = prediction_length

        # --------------------------------------------------
        # Direct forecasting branch
        # --------------------------------------------------

        self.linear_x = nn.Linear(
            context_length,
            prediction_length,
        )

        # --------------------------------------------------
        # Retrieval branch
        #
        # The retrieved future already has the correct
        # prediction length, but RAFT learns a transformation
        # before combining it with the direct forecast.
        # --------------------------------------------------

        self.linear_retrieval = nn.Linear(
            prediction_length,
            prediction_length,
        )

        # --------------------------------------------------
        # Combine direct + retrieval forecasts
        # --------------------------------------------------

        self.linear_pred = nn.Linear(
            2 * prediction_length,
            prediction_length,
        )

    def forward(self, batch):

        # ==================================================
        # Input
        # ==================================================

        x = batch["vegetation_history"]

        # [B, context_length, veg_dim]

        retrieval = batch["retrieval"]

        # [B, topm, prediction_length, veg_dim]

        weights = batch["retrieval_weights"]

        # [B, topm]

        # ==================================================
        # Direct forecasting branch
        # ==================================================

        # Last observed EVI value.
        #
        # This is the same offset-style treatment used
        # in the original RAFT implementation.

        x_offset = x[:, -1:, :].detach()

        x_norm = x - x_offset

        # [B, context_length, C]
        #       ↓
        # [B, C, context_length]
        #       ↓
        # [B, C, prediction_length]
        #       ↓
        # [B, prediction_length, C]

        x_pred = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)

        # ==================================================
        # Retrieval branch
        # ==================================================

        # Weighted combination of the top-k retrieved
        # future trajectories.
        #
        # retrieval:
        #     [B, K, P, C]
        #
        # weights:
        #     [B, K]
        #
        # result:
        #     [B, P, C]

        retrieval_pred = (retrieval * weights[:, :, None, None]).sum(dim=1)

        # ==================================================
        # Retrieval transformation
        # ==================================================

        retrieval_pred = self.linear_retrieval(retrieval_pred.permute(0, 2, 1)).permute(
            0, 2, 1
        )

        # ==================================================
        # Combine
        # ==================================================

        # [B, P, C] + [B, P, C]
        #
        # concatenate along the temporal dimension
        # because the original RAFT implementation applies
        # its final linear layer channel-wise.

        pred = torch.cat(
            [
                x_pred,
                retrieval_pred,
            ],
            dim=1,
        )

        # [B, 2P, C]
        #      ↓
        # [B, C, 2P]
        #      ↓
        # [B, C, P]
        #      ↓
        # [B, P, C]

        pred = self.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1)

        # Restore original EVI level.

        pred = pred + x_offset

        return pred
