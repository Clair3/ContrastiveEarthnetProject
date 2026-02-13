from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import default_collate
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import logging
import pandas as pd
from pathlib import Path
import numpy as np
import random

from .batch_sampler import BatchSampler


class ContrastiveDataset(Dataset):
    """
    Contrastive dataset for Sentinel-2 + ERA5 time series.
    Each sample returns:
        - anchor: one (location, year)
        - positive: temporal crop of same (location, year)
        - negative: another year of same location
    """

    def __init__(self, dataset_path):
        self.dataset = xr.open_zarr(dataset_path)
        samples_ids = self.dataset.sample.values.tolist()
        self.years = self.dataset.time_veg.dt.year.values.tolist()
        self.training_pairs = self._list_pairs(samples_ids, self.years)[:100]
        self.sentinel2_variables = [
            "evi",  # Enhanced Vegetation Index
        ]
        self.era5_variables = [
            "t2m_mean",  # 2-meter air temperature
            # "tp_mean",  # Total precipitation
            # "t2m_min",
            # "tp_min",
            # "t2m_max",
            "tp_max",
            # "pev_mean",  # Potential evapotranspiration
            # "ssr_mean",  # Surface solar radiation
            # "pev_min",
            # "ssr_min",
            # "pev_max",
            # "ssr_max",
        ]
        self.temporal_resolution_veg = 16
        self.temporal_resolution_weather = 5

    def _list_pairs(self, sample_ids, years):
        # Precompute valid (sample, year) pairs
        pairs = []
        for sample_id in sample_ids:
            for year in years:
                pairs.append((sample_id, year))
        return pairs

    def __len__(self):
        return len(self.training_pairs)

    def _to_tensor(self, data):
        """Convert xarray data to a PyTorch tensor."""
        arr = data.to_array().to_numpy()  # shape [variables, time, H, W] maybe
        arr = torch.as_tensor(arr, dtype=torch.float32)

        if torch.isnan(arr).all() or torch.isinf(arr).any():
            raise ValueError("Tensor contains only NaNs or infs")
        return arr.permute(1, 0)  # now [T, C]

    def __getitem__(self, idx: int) -> dict | None:
        sample_id, year = self.training_pairs[idx]
        print(f"Loading sample {sample_id} for year {year}")

        try:
            sample = self.dataset.sel(
                sample=sample_id,
            )
            vegetation = sample[self.sentinel2_variables].sel(
                time_veg=sample.time_veg.dt.year == int(year)
            )
            weather = sample[self.era5_variables].sel(
                time_weather=sample.time_weather.dt.year == int(year)
            )

            vegetation_location = (
                sample.latitude.values.item(),
                sample.longitude.values.item(),
            )

            data = {
                "vegetation": self._to_tensor(vegetation),
                "weather": self._to_tensor(weather),
                "location": vegetation_location,
            }

            return data

        except Exception as e:
            logging.warning(f"Skipping {sample_id}: {e}")
            return None


class ContrastiveDataModule(LightningDataModule):
    """
    DataModule for temporal contrastive learning with a held-out test year.
    """

    def __init__(
        self,
        dataset_path,
        batch_size=8,
        num_workers=16,
    ):
        """
        test_year: index of the held-out year in Zarr array
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_path = Path(dataset_path)

    def setup(self, stage=None):

        self.train_dataset = ContrastiveDataset(
            dataset_path=self.dataset_path / "train.zarr",
        )
        self.val_dataset = ContrastiveDataset(
            dataset_path=self.dataset_path / "validation.zarr",
        )

        self.test_dataset = ContrastiveDataset(
            dataset_path=self.dataset_path / "test.zarr",
        )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_sampler=BatchSampler(
                dataset=self.train_dataset,
                shuffle=True,
            ),
            collate_fn=safe_collate,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return loader  # NonNoneDataLoader(loader)

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_sampler=BatchSampler(
                dataset=self.val_dataset,
                shuffle=True,
            ),
            num_workers=self.num_workers,
            collate_fn=safe_collate,
            pin_memory=True,
        )
        return loader  # NonNoneDataLoader(loader)

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_sampler=BatchSampler(
                dataset=self.test_dataset,
                shuffle=True,
            ),
            num_workers=self.num_workers,
            collate_fn=safe_collate,
            pin_memory=True,
        )
        return loader  # NonNoneDataLoader(loader)


class NonNoneDataLoader:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for batch in self.loader:
            if batch is not None:
                yield batch

    def __len__(self):
        return len(self.loader)


def safe_collate(batch):
    # Remove any None entries
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # whole batch invalid TÓDO improve
    return default_collate(batch)
