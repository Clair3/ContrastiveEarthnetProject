import torch
import os

import numpy as np
import xarray as xr
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from pytorch_lightning import LightningModule


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
        predictions = torch.cat([x["preds"] for x in self.test_outputs])
        targets = torch.cat([x["targets"] for x in self.test_outputs])
        masks = torch.cat([x["mask"] for x in self.test_outputs])
        lons = torch.cat([x["locations"][0] for x in self.test_outputs])
        lats = torch.cat([x["locations"][1] for x in self.test_outputs])
        locations = torch.stack([lons, lats], dim=1)

        ds = xr.Dataset(
            {
                "predictions": (
                    ("sample", "time"),
                    predictions.squeeze(-1).cpu().numpy(),
                ),
                "targets": (("sample", "time"), targets.squeeze(-1).cpu().numpy()),
                "masks": (("sample", "time"), masks.squeeze(-1).cpu().numpy()),
            },
            coords={
                "longitude": ("sample", locations[:, 0].cpu().numpy()),
                "latitude": ("sample", locations[:, 1].cpu().numpy()),
            },
        )
        # Zarr path
        pred_path = os.path.join(self.output_dir, f"{self.logger.experiment.id}.zarr")
        print(f"Saving predictions to {pred_path}")

        ds.to_zarr(
            pred_path,
            mode="w",
            encoding={
                "predictions": {"chunks": (1000, predictions.shape[1])},
                "targets": {"chunks": (1000, targets.shape[1])},
                "masks": {"chunks": (1000, masks.shape[1])},
                "longitude": {"chunks": (1000,)},
                "latitude": {"chunks": (1000,)},
            },
        )

        self.test_outputs.clear()

    def configure_optimizers(self):
        return optim.Adam(
            list(self.model.parameters()),
            lr=self.lr,
        )
