import torch
import os

import numpy as np
import xarray as xr
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

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

        mask = ~torch.isnan(y_true)
        y_pred, y_true = y_pred[mask], y_true[mask]
        mask = torch.rand_like(y_true) < 0.2
        # loss = self.loss_fn(y_pred[mask], y_true[mask])
        loss = self.loss_fn(y_pred, y_true)
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
        y_pred = self(batch)
        y_true = batch["vegetation_forecast"]

        mask = ~torch.isnan(y_true)

        loss = self.loss_fn(y_pred[mask], y_true[mask])

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
        y_pred = self(batch)
        y_true = batch["vegetation_forecast"]

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
                "year": batch["year"].cpu() if "year" in batch else None,
                "sample_id": batch["sample_id"].cpu() if "sample_id" in batch else None,
            }
        )
        return loss

    def on_test_epoch_end(self):
        predictions = torch.cat([x["preds"] for x in self.test_outputs])
        targets = torch.cat([x["targets"] for x in self.test_outputs])
        masks = torch.cat([x["mask"] for x in self.test_outputs])

        years = torch.cat([x["year"] for x in self.test_outputs]).numpy()
        sample_ids = torch.cat([x["sample_id"] for x in self.test_outputs]).numpy()

        dataset = self.trainer.datamodule.test_dataset

        unique_years = np.unique(years)

        for year in unique_years:

            idx = years == year

            preds_year = predictions[idx]
            targets_year = targets[idx]
            masks_year = masks[idx]

            sample_ids_year = sample_ids[idx]

            lats = np.array([dataset.locations[sid][0] for sid in sample_ids_year])

            lons = np.array([dataset.locations[sid][1] for sid in sample_ids_year])

            ds = xr.Dataset(
                data_vars={
                    "predictions": (
                        ("sample", "time"),
                        preds_year.squeeze(-1).numpy(),
                    ),
                    "targets": (
                        ("sample", "time"),
                        targets_year.squeeze(-1).numpy(),
                    ),
                    "masks": (
                        ("sample", "time"),
                        masks_year.squeeze(-1).numpy(),
                    ),
                },
                coords={
                    "time": dataset.year_times[int(year)],
                    "latitude": ("sample", lats),
                    "longitude": ("sample", lons),
                    "sample_id": ("sample", sample_ids_year),
                },
            )

            try:
                pred_path = os.path.join(
                    self.config.output_dir,
                    f"{self.logger.experiment.id}_{int(year)}.zarr",
                )
            except Exception:
                pred_path = os.path.join(
                    self.config.output_dir,
                    f"prediction_{int(year)}.zarr",
                )

            print(f"Saving predictions to {pred_path}")

            ds.to_zarr(
                pred_path,
                mode="w",
                encoding={
                    "predictions": {
                        "chunks": (
                            min(1000, preds_year.shape[0]),
                            preds_year.shape[1],
                        )
                    },
                    "targets": {
                        "chunks": (
                            min(1000, targets_year.shape[0]),
                            targets_year.shape[1],
                        )
                    },
                    "masks": {
                        "chunks": (
                            min(1000, masks_year.shape[0]),
                            masks_year.shape[1],
                        )
                    },
                },
            )

        self.test_outputs.clear()

    # def on_test_epoch_end(self):
    #     predictions = torch.cat([x["preds"] for x in self.test_outputs])
    #     targets = torch.cat([x["targets"] for x in self.test_outputs])
    #     masks = torch.cat([x["mask"] for x in self.test_outputs])
    #     years = torch.cat([x["year"] for x in self.test_outputs])
    #     sample_ids = torch.cat([x["sample_id"] for x in self.test_outputs]).numpy()
    #
    #     ds_kwargs = {
    #         "data_vars": {
    #             "predictions": (
    #                 ("sample", "time"),
    #                 predictions.squeeze(-1).cpu().numpy(),
    #             ),
    #             "targets": (("sample", "time"), targets.squeeze(-1).cpu().numpy()),
    #             "masks": (("sample", "time"), masks.squeeze(-1).cpu().numpy()),
    #         }
    #     }
    #
    #     if self.test_outputs[0]["location"] is not None:
    #         lats = torch.cat([x["location"][0] for x in self.test_outputs])
    #         lons = torch.cat([x["location"][1] for x in self.test_outputs])
    #         locations = torch.stack([lons, lats], dim=1)
    #         ds_kwargs["coords"] = {
    #             "longitude": ("sample", locations[:, 0].cpu().numpy()),
    #             "latitude": ("sample", locations[:, 1].cpu().numpy()),
    #             # "time":
    #         }
    #         # time_array = self.test_outputs[0]["time"].cpu().numpy().astype("datetime64[s]")
    #
    #     ds = xr.Dataset(**ds_kwargs)
    #     # Zarr path
    #     try:
    #         pred_path = os.path.join(
    #             self.config.output_dir, f"{self.logger.experiment.id}.zarr"
    #         )
    #     except:
    #         pred_path = os.path.join(self.config.output_dir, f"prediction.zarr")
    #
    #     print(f"Saving predictions to {pred_path}")
    #
    #     ds.to_zarr(
    #         pred_path,
    #         mode="w",
    #         encoding={
    #             "predictions": {"chunks": (1000, predictions.shape[1])},
    #             "targets": {"chunks": (1000, targets.shape[1])},
    #             "masks": {"chunks": (1000, masks.shape[1])},
    #             "longitude": {"chunks": (1000,)},
    #             "latitude": {"chunks": (1000,)},
    #         },
    #     )
    #
    #     self.test_outputs.clear()

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
