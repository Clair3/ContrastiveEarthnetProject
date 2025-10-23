import numpy as np
import zarr
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import os
from pathlib import Path
import time
import sys
import logging
import traceback
import random

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

    def __init__(self, sample_paths, years, transform=None):
        self.samples = self._list_pairs(sample_paths, years)
        self.years = years
        self.transform = transform
        self.max_retries = 5

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
        for path in sample_paths:
            for year in years:
                pairs.append((path, year))

        return pairs

    def __len__(self):
        return len(self.samples)

    def _to_tensor(self, data):
        """Convert xarray data to a PyTorch tensor."""
        arr = data.to_array().to_numpy()  # shape [variables, time, H, W] maybe
        return torch.as_tensor(arr, dtype=torch.float32).transpose(
            1, 2
        )  # now [B, T, C]

    def __getitem__(self, idx):
        for attempt in range(self.max_retries):
            sample_path, positive_year = self.samples[idx]
            try:
                # Load data
                ds = xr.open_zarr(sample_path)
                vegetation = Sentinel2Preprocessing().generate_masked_vegetation_index(
                    ds
                )
                vegetation_location = (
                    vegetation.location.latitude.item(),
                    vegetation.location.longitude.item(),
                )

                weather = ds[self.era5_variables]

                # Select only the training years
                vegetation = vegetation.sel(
                    time=vegetation["time.year"].isin(self.years)
                ).isel(
                    location=0
                )  # only one location per zarr
                weather = weather.sel(time=weather["time.year"].isin(self.years))

                # Normalize
                weather = weather_normalization(weather)

                # Positive pair
                positive_vegetation = vegetation.sel(
                    time=vegetation["time.year"] == positive_year
                )
                positive_weather = weather.sel(
                    time=weather["time.year"] == positive_year
                )
                positive_vegetation_t = self._to_tensor(positive_vegetation)
                positive_weather_t = self._to_tensor(positive_weather)

                #  Negatives pairs
                negatives = []
                negative_years = [y for y in self.years if y != positive_year]
                # print(negative_years, positive_year)
                for year in negative_years:
                    # Vegetation from another year
                    negative_vegetation = vegetation.sel(
                        time=vegetation["time.year"] == year
                    )
                    negative_vegetation_t = self._to_tensor(negative_vegetation)
                    negatives.append((negative_vegetation_t, positive_weather_t))

                    # Weather from another year
                    negative_weather = weather.sel(time=weather["time.year"] == year)
                    negative_weather_t = self._to_tensor(negative_weather)
                    negatives.append((positive_vegetation_t, negative_weather_t))

                return {
                    "positive": (
                        positive_vegetation_t,
                        positive_weather_t,
                    ),  # Tensor [C, T, H, W] or similar
                    "negatives": negatives,  # List of (veg_tensor, weather_tensor)
                    "positive_year": positive_year,
                    "path": sample_path,
                    "location": vegetation_location,
                }
            except Exception as e:
                logging.warning(f"Skipping {sample_path}: {e}")
                idx = np.random.randint(0, len(self))

        print(f"[ERROR] Failed after {self.max_retries} retries.")
        return  # self._dummy_sample()


# -----------------------------
# 2Dataset for test/evaluation (full year)
# -----------------------------
# class FullYearDataset(Dataset):
#     """
#     Returns full year time series for evaluation.
#     """
#
#     def __init__(self, dataset_path, year, transform=None):
#         self.store = zarr.open(dataset_path, mode="r")
#         self.sentinel = self.store["sentinel2"]
#         self.era5 = self.store["era5"]
#         self.num_locations = self.sentinel.shape[0]
#         self.year = year
#         self.transform = transform
#
#     def __len__(self):
#         return self.num_locations
#
#     def __getitem__(self, idx):
#         sentinel = np.asarray(self.sentinel[idx, self.year], dtype=np.float32)
#         era5 = np.asarray(self.era5[idx, self.year], dtype=np.float32)
#
#         if self.transform:
#             sentinel, era5 = self.transform(sentinel, era5)
#
#         sentinel = torch.from_numpy(sentinel).permute(0, 3, 1, 2)
#         era5 = torch.from_numpy(era5)
#         return {
#             "sentinel": sentinel,  # (T, C, H, W)
#             "era5": era5,
#             "loc": idx,
#         }


# -----------------------------
# 3️⃣ DataModule
# -----------------------------
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
