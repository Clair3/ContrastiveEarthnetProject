import xarray as xr
import logging
from pathlib import Path
import numpy as np
import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from .batch_sampler import BatchSampler


class ContrastiveDataset(Dataset):
    """
    Contrastive dataset for Sentinel-2 + ERA5 time series.
    Each sample returns:
        - anchor: one (location, year)
        - positive: temporal crop of same (location, year)
        - negative: another year of same location
    """

    def __init__(self, dataset_path: str, sentinel2_vars: list, era5_vars: list):
        self.dataset = xr.open_zarr(dataset_path, consolidated=True)
        print(self.dataset)
        print(self.dataset.chunks)
        self.sentinel2_vars = sentinel2_vars
        self.era5_vars = era5_vars

        # Create training pairs
        self.training_pairs = self._create_training_pairs()

        # Pre-compute year masks once during init
        self._precompute_year_indices()

    def _create_training_pairs(self):
        """Create list of (sample_id, year) pairs."""
        pairs = []
        years = np.unique(self.dataset.time_veg.dt.year.values)
        for sample_id in range(len(self.dataset.sample[:100])):
            for year in years:
                pairs.append((sample_id, year))

        return pairs

    def _precompute_year_indices(self):
        """Pre-compute which time indices belong to each year."""

        # Extract year for each timestamp
        veg_years = pd.DatetimeIndex(self.dataset.time_veg.values).year

        # Create a dictionary mapping year → array of indices
        self.veg_year_masks = {}
        for year in np.unique(veg_years):
            self.veg_year_masks[year] = np.where(veg_years == year)[0]

        # Same for weather data
        weather_years = pd.DatetimeIndex(self.dataset.time_weather.values).year
        self.weather_year_masks = {}
        for year in np.unique(weather_years):
            self.weather_year_masks[year] = np.where(weather_years == year)[0]

    def __len__(self):
        return len(self.training_pairs)

    def _to_tensor(self, data):
        """Convert xarray data to a PyTorch tensor."""
        arr = data.to_array().to_numpy()  # shape [variables, time, H, W] maybe
        arr = torch.as_tensor(arr, dtype=torch.float32)

        if torch.isnan(arr).all() or torch.isinf(arr).any():
            raise ValueError("Tensor contains only NaNs or infs")
        return arr.permute(1, 0)  # shape [time, variables]

    def __getitem__(self, idx: int) -> dict | None:
        sample_id, year = self.training_pairs[idx]
        try:
            sample = self.dataset.isel(
                sample=sample_id,
            )

            veg_indices = self.veg_year_masks[year]
            weather_indices = self.weather_year_masks[year]

            vegetation = ds[self.sentinel2_vars].isel(
                sample=sample_id, time_veg=veg_indices
            )

            weather = ds[self.era5_vars].isel(
                sample=sample_id, time_weather=weather_indices
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
            # logging.warning(f"Skipping {sample_id}: {e}")
            return None


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
        self.data_config = data_config

    def setup(self, stage=None):
        self.train_dataset = ContrastiveDataset(
            dataset_path=self.dataset_path / "train_old.zarr",
            sentinel2_vars=self.data_config["vegetation"]["variables"],
            era5_vars=self.data_config["weather"]["variables"],
        )
        self.val_dataset = ContrastiveDataset(
            dataset_path=self.dataset_path / "validation_old.zarr",
            sentinel2_vars=self.data_config["vegetation"]["variables"],
            era5_vars=self.data_config["weather"]["variables"],
        )

        self.test_dataset = ContrastiveDataset(
            dataset_path=self.dataset_path / "test_old.zarr",
            sentinel2_vars=self.data_config["vegetation"]["variables"],
            era5_vars=self.data_config["weather"]["variables"],
        )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            # batch_sampler=BatchSampler(
            #     dataset=self.train_dataset,
            #     shuffle=True,
            # ),
            collate_fn=safe_collate,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

        return loader  # NonNoneDataLoader(loader)

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            # batch_sampler=BatchSampler(
            #     dataset=self.val_dataset,
            #     shuffle=True,
            # ),
            num_workers=self.num_workers,
            collate_fn=safe_collate,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            # batch_sampler=BatchSampler(
            #     dataset=self.test_dataset,
            #     shuffle=True,
            # ),
            num_workers=self.num_workers,
            collate_fn=safe_collate,
            pin_memory=True,
        )
        return loader


def safe_collate(batch):
    # Remove any None entries
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)
