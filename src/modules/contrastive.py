import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch
import os
from torch import optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

# from .models import RegressionHead
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from loss import info_nce_loss


class ContrastiveModule(LightningModule):
    def __init__(
        self,
        encoder_veg,
        encoder_weather,
        cfg,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder_veg", "encoder_weather"])
        self.encoder_veg = encoder_veg
        self.encoder_weather = encoder_weather
        self.config = cfg
        print("Initialized ContrastiveModule with config:", self.config)
        self.loss_fn = lambda veg, weather: info_nce_loss(
            veg, weather, temperature=self.config["temperature"]
        )
        self.probing_data = {
            "veg_emb": [],
            "weather_emb": [],
            "max_evi": [],
            "sum_precip": [],
        }

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
        self.probing_step(veg_emb, weather_emb, batch)
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

    def probing_step(self, veg_emb, weather_emb, batch):
        self.probing_data["veg_emb"].append(veg_emb.detach().cpu())
        self.probing_data["weather_emb"].append(weather_emb.detach().cpu())
        self.probing_data["max_evi"].append(batch["max_evi"].detach().cpu())
        self.probing_data["sum_precip"].append(batch["sum_precip"].detach().cpu())

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        self.probing()

    # def configure_optimizers(self):
    #     return optim.Adam(
    #         list(self.encoder_veg.parameters())
    #         + list(self.encoder_weather.parameters()),
    #         lr=self.config.lr,
    #     )

    def configure_optimizers(self):
        lr = float(self.config["lr"])
        warmup_fraction = getattr(self.config, "warmup_fraction", 0)

        total_steps = self.trainer.estimated_stepping_batches
        optimizer = torch.optim.AdamW(
            list(self.encoder_veg.parameters())
            + list(self.encoder_weather.parameters()),
            lr=lr,
            weight_decay=self.config["weight_decay"],  # L2 norm parameters
        )
        warmup_steps = int(warmup_fraction * total_steps)
        warmup = LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=lr * 0.05,
        )

        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def probing(self):
        veg_emb = torch.cat(self.probing_data["veg_emb"]).numpy()
        weather_emb = torch.cat(self.probing_data["weather_emb"]).numpy()
        max_evi = torch.cat(self.probing_data["max_evi"]).numpy()
        precip = torch.cat(self.probing_data["sum_precip"]).numpy()

        # ---- split ----
        veg_X_train, veg_X_test, precip_y_train, precip_y_test = train_test_split(
            veg_emb, precip, test_size=0.3, random_state=42
        )

        weather_X_train, weather_X_test, evi_y_train, evi_y_test = train_test_split(
            weather_emb, max_evi, test_size=0.3, random_state=42
        )

        # ---- probe 1: veg → precipitation ----
        probe_precip = LinearRegression()
        probe_precip.fit(veg_X_train, precip_y_train)
        score_precip = probe_precip.score(veg_X_test, precip_y_test)

        # ---- probe 2: weather → EVI ----
        probe_evi = LinearRegression()
        probe_evi.fit(weather_X_train, evi_y_train)
        score_max_evi = probe_evi.score(weather_X_test, evi_y_test)

        self.log("probe_precip_r2", score_precip, prog_bar=True)
        self.log("probe_evi_r2", score_max_evi, prog_bar=True)
        print("probe_precip_r2", score_precip, "probe_max_evi_r2", score_max_evi)

        # reset
        for key in self.probing_data:
            self.probing_data[key].clear()
