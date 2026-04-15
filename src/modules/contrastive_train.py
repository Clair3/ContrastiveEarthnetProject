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

    # def configure_optimizers2(self):
    #    lr = float(self.config.lr)
    #    total_steps = self.trainer.estimated_stepping_batches
    #    print(total_steps)
    #    optimizer = torch.optim.AdamW(
    #        self.model.parameters(), lr=lr, weight_decay=self.config.weight_decay
    #    )
    #    warmup_steps = int(self.config.warmup_fraction * total_steps)
    #    warmup = LinearLR(
    #        optimizer,
    #        start_factor=1e-6,
    #        end_factor=1.0,
    #        total_iters=warmup_steps,
    #    )#
    #    cosine = CosineAnnealingLR(
    #        optimizer,
    #        T_max=total_steps - warmup_steps,
    #        eta_min=lr * 0.05,
    #    )#
    #    scheduler = SequentialLR(
    #        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    #    )#
    #    return {"optimizer": optimizer, "lr_scheduler": scheduler}
