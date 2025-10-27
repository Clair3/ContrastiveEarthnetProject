import numpy as np
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from pathlib import Path
import logging
import pandas as pd

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(level=logging.WARNING)

from .preprocessing import (
    Sentinel2Preprocessing,
    weather_normalization,
    select_year,
)
from .batch_sampler import BatchSampler


class ContrastiveDataset(Dataset):
    """
    Contrastive dataset for Sentinel-2 + ERA5 time series.
    Each sample returns:
        - anchor: one (location, year)
        - positive: temporal crop of same (location, year)
        - negative: another year of same location
    """

    def __init__(self, dataset_path, train_years):
        sample_paths = pd.read_csv("valid_sample_paths.csv")[
            "sample_path"
        ].tolist()  ##[str(p) for p in Path(dataset_path).glob("*/*.zarr")]
        self.train_years = train_years
        self.training_pairs = self._list_pairs(sample_paths, train_years)

        self.temporal_resolution_veg = 16
        self.temporal_resolution_weather = 5

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

    def _list_pairs(self, sample_paths, years):
        # Precompute valid (sample, year) pairs
        pairs = []
        for sample_path in sample_paths:
            for year in years:
                pairs.append((sample_path, year))
        return pairs

    def __len__(self):
        return len(self.training_pairs)

    def _to_tensor(self, data):
        """Convert xarray data to a PyTorch tensor."""
        arr = data.to_array().to_numpy()  # shape [variables, time, H, W] maybe
        arr = torch.as_tensor(arr, dtype=torch.float32)
        return arr.permute(1, 0)  # now [T, C]

    def __getitem__(self, idx: int) -> dict | None:
        sample_path, year = self.training_pairs[idx]

        try:
            ds = xr.open_zarr(sample_path)
        except Exception as e:
            logging.warning(f"Failed to open Zarr {sample_path}: {e}")
            return None

        try:
            vegetation = self._process_vegetation(ds, year)
            weather = self._process_weather(ds, year)
            vegetation_location = (
                vegetation.location.latitude.item(),
                vegetation.location.longitude.item(),
            )

            return {
                "vegetation": self._to_tensor(vegetation),
                "weather": self._to_tensor(weather),
                "path": sample_path,
                "location": vegetation_location,
            }

        except Exception as e:
            logging.warning(f"Skipping {sample_path}: {e}")
            return None

    # --- helper functions ---
    def _process_vegetation(self, ds: xr.Dataset, year: int) -> xr.DataArray:
        veg_index = Sentinel2Preprocessing(
            temporal_resolution=self.temporal_resolution_veg
        ).generate_masked_vegetation_index(ds)
        return select_year(
            veg_index,
            temporal_resolution=self.temporal_resolution_veg,
            selected_year=year,
        )

    def _process_weather(self, ds: xr.Dataset, year: int) -> xr.DataArray:
        weather = ds[self.era5_variables]
        # Keep only training years
        weather = weather.sel(time=weather["time.year"].isin(self.train_years))
        weather = weather_normalization(weather)
        return select_year(
            weather,
            temporal_resolution=self.temporal_resolution_weather,
            selected_year=year,
        )


class ContrastiveDataModule(LightningDataModule):
    """
    DataModule for temporal contrastive learning with a held-out test year.
    """

    def __init__(
        self,
        dataset_path,
        batch_size=8,
        num_workers=4,
        years=range(2017, 2023),
        test_year=2020,
        transform=None,
    ):
        """
        test_year: index of the held-out year in Zarr array
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_path = dataset_path
        self.test_year = test_year
        self.training_years = [year for year in years if year != self.test_year]

    def setup(self, stage=None):

        self.train_dataset = ContrastiveDataset(
            dataset_path=self.dataset_path,
            train_years=self.training_years,
        )
        # self.test_dataset = ContrastiveDataset(
        #     dataset_path=self.dataset_path,
        #     train_years=[self.test_year],
        #     transform=None,
        # )

        # Test: held-out year, full year per location
        # self.test_dataset = FullYearDataset(
        #     dataset_path=self.dataset_path, year=self.test_year, transform=self.transform
        # )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            batch_sampler=BatchSampler(
                dataset=self.train_dataset,
                shuffle=True,
            ),
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
