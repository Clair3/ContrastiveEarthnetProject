class InterpretableDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt

        # Self-attention
        tgt2, self_attn = self.self_attn(
            x,
            x,
            x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = x + self.dropout1(tgt2)
        x = self.norm1(x)

        # Cross-attention (IMPORTANT)
        tgt2, cross_attn = self.multihead_attn(
            x,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = x + self.dropout2(tgt2)
        x = self.norm2(x)

        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout3(tgt2)
        x = self.norm3(x)

        return x, self_attn, cross_attn

class InterpretableDecoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, tgt, memory):
        self_attns = []
        cross_attns = []

        x = tgt
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, memory)
            self_attns.append(self_attn)
            cross_attns.append(cross_attn)

        return x, self_attns, cross_attns
    

    class TransformerBaselineInterpretable(nn.Module):
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
        decoder_layer = InterpretableDecoderLayer(
                d_model=d_model,
                nhead=config.num_heads,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True
        )
        self.decoder = InterpretableDecoder(decoder_layer, config.num_layers)
        # causal_mask = self.generate_causal_mask(T, x.device)


        # --- Head ---
        self.head = nn.Linear(d_model, veg_dim)

    def forward(self, batch):
        # Encode past: veg and weather history → sequence of tokens
        veg_mem, _ = self.veg_encoder(batch["vegetation_history"])  # [B, T_veg, d]
        weather_mem, _ = self.weather_encoder(batch["weather_history"])  # [B, T_w,   d]

        # Past memory = concat along sequence dim (not feature dim)
        memory = torch.cat([veg_mem, weather_mem], dim=1)  # [B, T_veg+T_w, d]

        # Decode: weather forecast queries into past memory

        forecast_query, _ = self.weather_encoder(
            batch["weather_forecast"]
        )  # [B, T_w, d]
        # print(memory.shape, forecast_query.shape)
        out, self_attns, cross_attns = self.decoder(
            tgt=forecast_query,
            memory=memory
        )

        # Project to vegetation space
        out = self.head(out)  # [B, T_w, veg_dim]
        return out, {
                "self_attn": self_attns,
                "cross_attn": cross_attns
            }

