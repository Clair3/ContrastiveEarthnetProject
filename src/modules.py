import torch
import os

import numpy as np
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from pytorch_lightning import LightningModule

from loss import info_nce_loss


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

    def __init__(self, model, output_dir, lr=3e-4):
        super().__init__()
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.output_dir = output_dir
        self.test_outputs = []
        os.makedirs(self.output_dir, exist_ok=True)

    def forward(self, batch):
        return self.model(batch)

    def _step(self, batch, stage: str):
        if batch is None:
            if stage == "train":
                self.log(
                    f"{stage}_loss",
                    float("nan"),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                )
            return None

        y_pred = self(batch)
        y_true = batch["vegetation_forecast"]

        # Mask NaNs consistently
        mask = ~torch.isnan(y_true)

        loss = self.loss_fn(y_pred[mask], y_true[mask])

        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=y_true.shape[0],
        )
        if stage == "test":
            self.test_outputs.append(
                {
                    "preds": y_pred.detach().cpu(),
                    "targets": y_true.detach().cpu(),
                    "mask": mask.detach().cpu(),
                    "locations": [x.detach().cpu() for x in batch["location"]],
                }
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")
        return

    def on_test_epoch_end(self):
        preds = torch.cat([x["preds"] for x in self.test_outputs])
        targets = torch.cat([x["targets"] for x in self.test_outputs])
        mask = torch.cat([x["mask"] for x in self.test_outputs])
        lons = torch.cat([x["locations"][0] for x in self.test_outputs])
        lats = torch.cat([x["locations"][1] for x in self.test_outputs])
        locations = torch.stack([lons, lats], dim=1)

        # Save predictions
        pred_path = os.path.join(self.output_dir, f"{self.logger.experiment.id}.npz")
        print(f"Saving predictions to {pred_path}")

        np.savez_compressed(
            pred_path,
            preds=preds.numpy(),
            targets=targets.numpy(),
            mask=mask.numpy(),
            locations=locations.numpy(),
        )

        self.test_outputs.clear()

    def configure_optimizers(self):
        return optim.Adam(
            list(self.model.parameters()),
            lr=self.lr,
        )
