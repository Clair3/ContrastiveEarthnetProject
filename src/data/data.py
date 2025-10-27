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

from .preprocessing import Sentinel2Preprocessing, weather_normalization


class ContrastiveDataset(Dataset):
    """
    Contrastive dataset for Sentinel-2 + ERA5 time series.
    Each sample returns:
        - anchor: one (location, year)
        - positive: temporal crop of same (location, year)
        - negative: another year of same location
    """

    def __init__(self, sample_paths, train_years, transform=None):
        self.sample_paths = sample_paths
        self.train_years = train_years
        self.max_retries = 5
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

    def __len__(self):
        return len(self.sample_paths)

    def _to_tensor(self, data):
        """Convert xarray data to a PyTorch tensor."""
        arr = data.to_array().to_numpy()  # shape [variables, time, H, W] maybe
        arr = torch.as_tensor(arr, dtype=torch.float32)
        return arr.permute(1, 2, 0)  # now [B, T, C]

    def fill_year(self, data, temporal_resolution=16, years=None):
        """
        Pads each year to a full 365-day (or 366 for leap years) coverage
        Returns an xarray with dims (year, doy_period, ...), keeping order.
        """
        # --- Define expected day-of-year bins ---
        doy_bins = np.arange(1, 367, temporal_resolution)
        n_periods = len(doy_bins)

        padded_years = []

        for year in years:
            start_date = datetime(year, 1, 1)
            expected_times = [start_date + timedelta(days=int(d - 1)) for d in doy_bins]

            # Select this year's data
            data_year = data.sel(time=pd.to_datetime(data.time).year == year)

            # Reindex to fill missing periods with NaN
            data_year = data_year.reindex(time=pd.to_datetime(expected_times))

            # Assign a simple DOY index for model input
            data_year = data_year.assign_coords(
                doy=("time", np.arange(1, n_periods + 1))
            )

            data_year = data_year.swap_dims({"time": "doy"}).drop_vars("time")
            padded_years.append(data_year)

        # --- Concatenate along year dimension ---
        data_padded = xr.concat(padded_years, dim=pd.Index(years, name="year"))
        return data_padded

    def __getitem__(self, idx):
        for _ in range(self.max_retries):
            sample_path = self.sample_paths[idx]
            try:
                # Load data
                ds = xr.open_zarr(sample_path)
                vegetation = Sentinel2Preprocessing(
                    temporal_resolution=self.temporal_resolution_veg
                ).generate_masked_vegetation_index(ds)
                vegetation_location = (
                    vegetation.location.latitude.item(),
                    vegetation.location.longitude.item(),
                )
                weather = ds[self.era5_variables]

                # Select only the training years
                vegetation = vegetation.sel(
                    time=vegetation["time.year"].isin(self.train_years)
                ).isel(
                    location=0
                )  # remove location dim
                weather = weather.sel(time=weather["time.year"].isin(self.train_years))

                # Normalize
                weather = weather_normalization(weather)
                vegetation_per_year = self.fill_and_stack_year(
                    vegetation,
                    temporal_resolution=self.temporal_resolution_veg,
                    years=self.train_years,
                )
                weather_per_year = self.fill_and_stack_year(
                    weather,
                    temporal_resolution=self.temporal_resolution_weather,
                    years=self.train_years,
                )

                vegetation_t = self._to_tensor(vegetation_per_year)
                weather_t = self._to_tensor(weather_per_year)
                print(vegetation_t.shape, weather_t.shape)
                return {
                    "vegetation": vegetation_t,
                    "weather": weather_t,
                    "path": sample_path,
                    "location": vegetation_location,
                }
            except Exception as e:
                logging.warning(f"Skipping {sample_path}: {e}")
                idx = np.random.randint(0, len(self))

        print(f"[ERROR] Failed after {self.max_retries} retries.")
        return  # self._dummy_sample()


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
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_year = test_year
        self.years = years
        self.transform = transform

    def setup(self, stage=None):
        paths = [str(p) for p in Path(self.dataset_path).glob("*/*.zarr")]
        training_years = [
            year for year in self.years if year != self.test_year
        ]  # example

        self.train_dataset = ContrastiveDataset(paths, training_years)
        self.test_dataset = ContrastiveDataset(paths, [self.test_year])

        # Test: held-out year, full year per location
        # self.test_dataset = FullYearDataset(
        #     dataset_path=self.dataset_path, year=self.test_year, transform=self.transform
        # )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
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
