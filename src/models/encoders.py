import torch
import torch.nn as nn

from .positional_encoding import SeasonalPositionalEncoding, ClassicPositionalEncoding


class TimeSeriesTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        sequence_length,
        use_cls,
        seasonal_positional_encoding,
        d_model,
        num_heads,
        num_layers,
        dropout,
    ):
        super().__init__()
        # print(
        #     input_dim,
        #     sequence_length,
        #     use_cls,
        #     seasonal_positional_encoding,
        #     d_model,
        #     num_heads,
        #     num_layers,
        # )
        self.use_cls = use_cls
        self.linear_layer = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.positional_encoding = (
            SeasonalPositionalEncoding(d_model, sequence_length=sequence_length)
            if seasonal_positional_encoding
            else ClassicPositionalEncoding(d_model, max_len=sequence_length)
        )

        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

        # if self.use_cls:
        #     self.cls_token = nn.Parameter(
        #         torch.randn(1, 1, d_model)
        #     )  # Classification token

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.linear_layer.weight)
        if self.linear_layer.bias is not None:
            nn.init.zeros_(self.linear_layer.bias)

        # Initialize CLS token with small random values
        # if self.use_cls:
        #     nn.init.normal_(self.cls_token, mean=0, std=0.02)

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

        return (
            self.transformer(x, src_key_padding_mask=padding_mask),
            padding_mask,
        )  # [B, T, d_model]

    #         if self.use_cls:
    #             mask = ~padding_mask
    #             mask = mask.unsqueeze(-1)
    #             x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
    #
    #         return self.norm(x)


class ContrastiveHead(nn.Module):
    def __init__(self, d_model, projection_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim),
        )

    def forward(self, x, padding_mask):
        # x: (B, T, D)

        mask = ~padding_mask
        print(mask.shape, x.shape)
        mask = mask.unsqueeze(-1)

        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        print(pooled.shape)
        z = self.proj(pooled)
        print(z.shape)
        z = nn.functional.normalize(z, dim=-1)
        return z


class ContrastiveTransformer(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        seq, mask = self.encoder(x)
        z = self.head(seq, mask)
        return seq[:, 0, :]


# class ContrastiveTransformer(nn.Module):
#     def __init__(
#         self, encoder_veg, encoder_weather, head_veg, head_weather, lr, temperature
#     ):
#         super().__init__()
#         self.encoder_veg = encoder_veg
#         self.encoder_weather = encoder_weather
#         self.head_veg = head_veg
#         self.head_weather = head_weather
#         self.temperature = temperature
#
#     def forward(self, veg, weather):
#         # encoder returns (seq, mask)
#         seq_v, mask_v = self.encoder_veg(veg)
#         # heads do pooling + projection
#         z_v = self.head_veg(seq_v, mask_v)
#         z_w = self.head_weather(seq_w, mask_w)
#         return z_v, z_w
#

# self.norm(x[:, 0, :]) if self.use_cls else self.norm(x)
# if self.use_cls:
#     cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
#     x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, d_model]
#
#     # Extend mask for CLS token (CLS is always valid)
#     cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
#     padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # [B, T+1]
