import torch
import torch.nn as nn

from .positional_encoding import SeasonalPositionalEncoding, ClassicPositionalEncoding


class TimeSeriesTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        sequence_length,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        use_cls=True,
        seasonal_positional_encoding=True,
    ):
        super().__init__()
        self.use_cls = use_cls
        self.linear_layer = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.positional_encoding = (
            SeasonalPositionalEncoding(d_model, sequence_length=sequence_length)
            if seasonal_positional_encoding
            else ClassicPositionalEncoding(d_model, max_len=sequence_length)
        )

        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

        if self.use_cls:
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, d_model)
            )  # Classification token

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.linear_layer.weight)
        if self.linear_layer.bias is not None:
            nn.init.zeros_(self.linear_layer.bias)

        # Initialize CLS token with small random values
        if self.use_cls:
            nn.init.normal_(self.cls_token, mean=0, std=0.02)

    def forward(self, x):
        """
        x: [B, T, input_dim] - input time series
        """
        # Replace NaNs with zeros before linear layer (they'll be masked out anyway)
        B, _, _ = x.shape
        # PyTorch transformer expects: True = IGNORE, False = ATTEND
        padding_mask = torch.isnan(x).any(dim=-1)  # [B, T]
        x = torch.nan_to_num(x, nan=0.0)

        x = self.linear_layer(x)  # [B, T, d_model]
        x = self.positional_encoding(x)

        if self.use_cls:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, d_model]

            # Extend mask for CLS token (CLS is always valid)
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # [B, T+1]

        x = self.transformer(x, src_key_padding_mask=padding_mask)  # [B, T, d_model]
        return self.norm(x[:, 0, :]) if self.use_cls else self.norm(x)
