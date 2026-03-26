from torch import optim
from pytorch_lightning import LightningModule
import torch
from loss import info_nce_loss
import torch.nn.functional as F
import torch.nn as nn


class ContrastiveTrainingModule(LightningModule):
    def __init__(
        self,
        encoder_veg,
        encoder_weather,
        lr=3e-4,
        temperature=0.07,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder_veg", "encoder_weather"])
        self.encoder_veg = encoder_veg
        self.encoder_weather = encoder_weather
        self.lr = lr
        self.loss_fn = lambda veg, weather: info_nce_loss(
            veg, weather, temperature=temperature
        )

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
        loss = self.loss_fn(veg_emb, weather_emb)

        self.log("veg_emb_std", veg_emb.std(), on_step=True)
        self.log("weather_emb_std", weather_emb.std(), on_step=True)
        self.log("veg_emb_mean", veg_emb.mean().abs(), on_step=True)
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
            return None  # skip this batch

        vegetation = batch["vegetation"]
        weather = batch["weather"]
        veg_emb, weather_emb = self(vegetation, weather)
        loss = self.loss_fn(veg_emb, weather_emb)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=vegetation.shape[0],
        )
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return optim.Adam(
            list(self.encoder_veg.parameters())
            + list(self.encoder_weather.parameters()),
            lr=self.lr,
        )


class ForecastingTrainModule(LightningModule):
    """
    LightningModule for vegetation forecasting.
    """

    def __init__(self, model, lr=3e-4):
        super().__init__()
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        if batch is None:
            self.log(
                "train_loss", float("nan"), on_step=True, on_epoch=True, prog_bar=True
            )
            return None

        y_pred = self(batch)
        y_true = batch["vegetation_forecast"].squeeze(-1)
        print(y_pred.shape, y_true.shape)
        mask = ~torch.isnan(y_true)

        loss = self.loss_fn(y_pred[mask], y_true[mask])

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=y_true.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            print("not here")
            return None
        print("here")

        y_pred = self(batch)
        y_true = batch["vegetation_forecast"]
        loss = self.loss_fn(y_true, y_pred)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=y_true.shape[0],
        )
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return optim.Adam(
            list(self.model.parameters()),
            lr=self.lr,
        )
