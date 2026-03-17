import xarray as xr
import logging
import numpy as np
import torch

import pandas as pd
from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader

from .datasets import ContrastiveDataset, ForecastingDataset
from .batch_sampler import BatchSampler


class ContrastiveDataModule(LightningDataModule):
    """
    DataModule for temporal contrastive learning with a held-out test year.
    """

    def __init__(
        self,
        data_config,
        batch_size=16,
        num_workers=16,
    ):
        """
        test_year: index of the held-out year in Zarr array
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_path = Path(data_config["dataset"]["path"])
        self.sentinel2_vars = data_config["vegetation"]["variables"]
        self.era5_vars = data_config["weather"]["variables"]

    def _build_dataset(self, split):
        return ContrastiveDataset(
            dataset_path=self.dataset_path / f"{split}.zarr",
            sentinel2_vars=self.sentinel2_vars,
            era5_vars=self.era5_vars,
        )

    def setup(self, stage=None):
        self.train_dataset = self._build_dataset("train")
        self.val_dataset = self._build_dataset("train")
        self.test_dataset = self._build_dataset("train")

    def _build_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            # batch_sampler=BatchSampler(
            #     dataset=self.train_dataset,
            #     shuffle=True,
            # ),
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=safe_collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._build_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._build_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._build_dataloader(self.test_dataset)


class ForecastingDataModule(ContrastiveDataModule):
    def __init__(self, data_config, batch_size=16, num_workers=16):
        super().__init__(data_config, batch_size, num_workers)

    def _build_dataset(self, split):
        return ForecastingDataset(
            dataset_path=self.dataset_path / f"{split}.zarr",
            sentinel2_vars=self.sentinel2_vars,
            era5_vars=self.era5_vars,
        )


def safe_collate(batch):
    # Remove any None entries
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)
