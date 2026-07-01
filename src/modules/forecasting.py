import torch
import os

import numpy as np
import xarray as xr
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from pathlib import Path
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
from pytorch_lightning import LightningModule


class ForecastingModule(LightningModule):
    """
    LightningModule for vegetation forecasting.
    """

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = nn.MSELoss()
        self.test_outputs = []

    def forward(self, batch):
        return self.model(batch)

    def _compute_extremes_loss(self, y_pred, y_true, percentiles, mask):
        percentiles = percentiles[mask]
        extremes_mask = percentiles <= 0.1

        if not extremes_mask.any():
            return None

        return self.loss_fn(
            y_pred[mask][extremes_mask],
            y_true[mask][extremes_mask],
        )

    def _compute_anomalies_loss(self, y_pred, y_true, msc, mask):
        percentiles = percentiles[mask]
        extremes_mask = percentiles <= 0.1

        if not extremes_mask.any():
            return None

        return self.loss_fn(
            y_pred[mask][extremes_mask],
            y_true[mask][extremes_mask],
        )

    def training_step(self, batch, batch_idx):
        if batch is None:
            self.log(
                f"train_loss",
                float("nan"),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            return None

        y_pred = self(batch)
        y_true = batch["vegetation_forecast"]

        # Shape should match y_true
        # msc = batch["msc"]

        # w = torch.abs(torch.diff(msc, dim=1, prepend=msc[:, :1]))
        # weight = abs(dmsc)
        # w = 0.2 + abs(dmsc) / abs(dmsc).max()
        # # Normalize per sample
        # w_min = w.amin(dim=1, keepdim=True)
        # w_max = w.amax(dim=1, keepdim=True)
        ##
        ## # Normalize MSC to [0, 1]
        # w = (w - w_min) / (w_max - w_min + 1e-8)
        #
        # # Optional: avoid giving almost zero weight to winter
        # w = 0.2 + 0.8 * w

        mask = ~torch.isnan(y_true)

        # w = w[:, : mask.size(1), :][mask]
        y_pred = y_pred[mask]
        y_true = y_true[mask]

        # loss_fn must have reduction="none"
        loss = self.loss_fn(y_pred, y_true)

        # loss = (w * loss).mean()

        # # Normalize MSC to [0, 1]
        # w = (msc - msc.min()) / (msc.max() - msc.min() + 1e-8)
        #
        # # Optional: avoid giving almost zero weight to winter
        # w = 0.5 + 0.5 * w
        #
        # mask = ~torch.isnan(y_true)
        #
        # y_pred = y_pred[mask]
        # y_true = y_true[mask]
        # w = w[mask]
        #
        # # loss_fn must have reduction="none"
        # loss = self.loss_fn(y_pred, y_true)
        #
        # loss = (w * loss).mean()

        # y_pred = self(batch)
        # y_true = batch["vegetation_forecast"]
        # msc  = batch["msc"]
        # w = msc / msc.max() - msc.min()
        #
        # mask = ~torch.isnan(y_true)
        # y_pred, y_true = y_pred[mask], y_true[mask]
        # loss = self.loss_fn(y_pred, y_true)

        # mask = torch.rand_like(y_true) < 0.2
        # loss = self.loss_fn(y_pred[mask], y_true[mask])
        #
        # pred_monthly = F.adaptive_avg_pool1d(y_pred.transpose(1, 2), 12).transpose(1, 2)
        #
        # target_monthly = F.adaptive_avg_pool1d(y_true.transpose(1, 2), 12).transpose(
        #     1, 2
        # )
        # loss_monthly = self.loss_fn(pred_monthly, target_monthly)
        #
        # loss_seasonal = self.loss_fn(y_pred.mean(dim=1), y_true.mean(dim=1))
        #
        # loss = 0.5 * loss_daily + 0.3 * loss_monthly + 0.2 * loss_seasonal

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
            return None

        y_pred = self(batch)
        y_true = batch["vegetation_forecast"]

        # Shape should match y_true
        msc = batch["msc"]
        # Normalize per sample
        # msc_min = msc.amin(dim=1, keepdim=True)
        # msc_max = msc.amax(dim=1, keepdim=True)
        #
        # # Normalize MSC to [0, 1]
        # w = (msc - msc_min) / (msc_max - msc_min + 1e-8)
        #
        # # Optional: avoid giving almost zero weight to winter
        # w = 0.2 + 0.8 * w

        mask = ~torch.isnan(y_true)

        # w = w[:, : mask.size(1), :][mask]
        y_pred = y_pred[mask]
        y_true = y_true[mask]

        # loss_fn must have reduction="none"
        loss = self.loss_fn(y_pred, y_true)

        # loss = (w * loss).mean()

        # y_pred = self(batch)
        # y_true = batch["vegetation_forecast"]
        #
        # mask = ~torch.isnan(y_true)
        #
        # loss = self.loss_fn(y_pred[mask], y_true[mask])

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=y_true.shape[0],
        )

        if "percentiles_forecast" in batch:
            extremes_loss = self._compute_extremes_loss(
                y_pred,
                y_true,
                batch["percentiles_forecast"],
                mask,
            )

            if extremes_loss is not None:
                self.log(
                    "val_extremes_loss",
                    extremes_loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=y_true.shape[0],
                )

        return loss

    def test_step(self, batch, batch_idx):
        if batch is None:
            return None
        y_pred = self(batch)
        y_true = batch["vegetation_forecast"]
        persistence = batch["vegetation_history"][:, -1, :]

        mask = ~torch.isnan(y_true)

        loss = self.loss_fn(y_pred[mask], y_true[mask])

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=y_true.shape[0],
        )

        if "percentiles_forecast" in batch:
            extremes_loss = self._compute_extremes_loss(
                y_pred,
                y_true,
                batch["percentiles_forecast"],
                mask,
            )
            if extremes_loss is not None:
                self.log("test_extremes_loss", extremes_loss)

        self.test_outputs.append(
            {
                "preds": y_pred.detach().cpu(),
                "targets": y_true.detach().cpu(),
                "mask": mask.detach().cpu(),
                "forecast_origin": batch["forecast_origin"].cpu(),
                "persistence": persistence.detach().cpu(),
                "sample_id": batch["sample_id"].cpu(),
            }
        )
        return loss

    def on_test_epoch_end(self):

        predictions = (
            torch.cat([x["preds"] for x in self.test_outputs]).squeeze(-1).numpy()
        )

        targets = (
            torch.cat([x["targets"] for x in self.test_outputs]).squeeze(-1).numpy()
        )

        masks = torch.cat([x["mask"] for x in self.test_outputs]).squeeze(-1).numpy()

        forecast_origins = torch.cat(
            [x["forecast_origin"] for x in self.test_outputs]
        ).numpy()

        persistence = (
            torch.cat([x["persistence"] for x in self.test_outputs]).squeeze(-1).numpy()
        )

        sample_ids = torch.cat([x["sample_id"] for x in self.test_outputs]).numpy()

        dataset = self.trainer.datamodule.test_dataset

        lats = np.array([dataset.locations[int(sid)][0] for sid in sample_ids])

        lons = np.array([dataset.locations[int(sid)][1] for sid in sample_ids])

        ds = xr.Dataset(
            data_vars={
                "predictions": (
                    ("window", "lead_time"),
                    predictions,
                ),
                "targets": (
                    ("window", "lead_time"),
                    targets,
                ),
                "masks": (
                    ("window", "lead_time"),
                    masks,
                ),
                "persistence": (
                    ("window",),
                    persistence,
                ),
            },
            coords={
                "window": np.arange(len(sample_ids)),
                "sample_id": ("window", sample_ids),
                "latitude": ("window", lats),
                "longitude": ("window", lons),
                "forecast_date": (
                    "window",
                    dataset.dataset.time_veg.values[forecast_origins],
                ),
                "lead_time": np.arange(self.config.prediction_length),
            },
        )

        ds.attrs["time_reference"] = "forecast_origin indexes dataset.time_veg"
        ds.attrs["prediction_length"] = self.config.prediction_length

        print("Saving path: ", Path(self.config.output_dir) / "predictions.zarr")
        ds.to_zarr(
            Path(self.config.output_dir) / "predictions.zarr",
            mode="w",
        )

        self.test_outputs.clear()

    def configure_optimizers(self):
        lr = float(self.config.lr)
        warmup_fraction = getattr(self.config, "warmup_fraction", 0)

        total_steps = self.trainer.estimated_stepping_batches
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.config.weight_decay,  # L2 norm parameters
        )
        warmup_steps = int(warmup_fraction * total_steps)
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
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
