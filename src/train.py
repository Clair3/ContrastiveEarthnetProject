from torch import optim
from pytorch_lightning import LightningModule
import torch
from models import TimeSeriesTransformerEncoder
from loss import info_nce_loss


class ContrastiveTrainingModule(LightningModule):
    def __init__(self, encoder_veg, encoder_weather, lr=1e-3):
        super().__init__()
        self.encoder_veg = encoder_veg
        self.encoder_weather = encoder_weather
        self.lr = lr

    def forward(self, vegetation, weather):
        veg_emb = self.encoder_veg(vegetation)
        weather_emb = self.encoder_weather(weather)
        return veg_emb, weather_emb

    def training_step(self, batch, batch_idx):
        if batch == None:
            self.log(
                "train_loss", float("nan"), on_step=True, on_epoch=True, prog_bar=True
            )
            return None  # skip this batch

        vegetation = batch["vegetation"]
        weather = batch["weather"]

        veg_emb, weather_emb = self(vegetation, weather)
        loss = info_nce_loss(veg_emb, weather_emb)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=vegetation.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if batch == None:
            self.log(
                "val_loss", float("nan"), on_step=True, on_epoch=True, prog_bar=True
            )
            return None  # skip this batch

        vegetation = batch["vegetation"]
        weather = batch["weather"]
        veg_emb, weather_emb = self(vegetation, weather)
        loss = info_nce_loss(veg_emb, weather_emb)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=vegetation.shape[0],
        )
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            list(self.encoder_veg.parameters())
            + list(self.encoder_weather.parameters()),
            lr=self.lr,
        )
        return optimizer
