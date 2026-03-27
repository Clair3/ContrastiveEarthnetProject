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


class BaseDataset(Dataset):
    """
    Contrastive dataset for Sentinel-2 + ERA5 time series.
    Each sample returns:
        - anchor: one (location, year)
        - positive: temporal crop of same (location, year)
        - negative: another year of same location
    """

    def __init__(
        self, dataset_path: str, sentinel2_vars: list, era5_vars: list, years=list
    ):
        self.dataset = self.create_split(dataset_path=dataset_path, years=years)
        self.sentinel2_vars = sentinel2_vars
        self.era5_vars = era5_vars

        # Pre-compute year masks once during init
        self._precompute_year_indices()

    def create_split(self, dataset_path: str, years: list[int]) -> xr.Dataset:
        """
        Create a dataset split containing only data from specified years.

        Args:
            dataset: Input dataset with time_veg and time_weather dimensions
            years: List of years to include in this split

        Returns:
            Dataset filtered to only include samples from the specified years
        """
        dataset = xr.open_zarr(dataset_path)
        return dataset.sel(
            time_veg=pd.DatetimeIndex(dataset.time_veg.values).year.isin(years),
            time_weather=pd.DatetimeIndex(dataset.time_weather.values).year.isin(years),
        )

    def _create_training_pairs(self, years):
        """Create list of (sample_id, year) pairs."""
        pairs = []
        for sample_id in range(len(self.dataset.sample)):
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

    def _load_year(self, sample, year):
        """Return vegetation + weather tensors for a given year."""
        veg_idx = self.veg_year_masks[year]
        weather_idx = self.weather_year_masks[year]

        vegetation = sample[self.sentinel2_vars].isel(time_veg=veg_idx)
        weather = sample[self.era5_vars].isel(time_weather=weather_idx)

        return (
            self._to_tensor(vegetation),
            self._to_tensor(weather),
        )

    def __len__(self):
        if self.training_pairs is None:
            raise ValueError("training_pairs must be defined in subclass")
        return len(self.training_pairs)

    def _to_tensor(self, data):
        """Convert xarray data to a PyTorch tensor."""
        arr = data.to_array().to_numpy()  # shape [variables, time, H, W] maybe
        arr = torch.as_tensor(arr, dtype=torch.float32)

        if torch.isnan(arr).all() or torch.isinf(arr).any():
            raise ValueError("Tensor contains only NaNs or infs")
        return arr.permute(1, 0)  # shape [time, variables]


class ContrastiveDataset(BaseDataset):

    def __init__(self, dataset_path, sentinel2_vars, era5_vars, years):

        super().__init__(dataset_path, sentinel2_vars, era5_vars, years)
        # Create training pairs
        self.training_pairs = self._create_training_pairs(years=years)

    def __getitem__(self, idx) -> dict | None:
        sample_id, year = self.training_pairs[idx]
        try:
            sample = self.dataset.isel(sample=sample_id)
            location = (
                sample.latitude.values.item(),
                sample.longitude.values.item(),
            )
            vegetation, weather = self._load_year(sample, year)

            data = {
                "vegetation": vegetation,
                "weather": weather,
                "location": location,
            }
            return data
        except Exception as e:
            # logging.warning(f"Skipping {sample_id}: {e}")
            return None


class ForecastingDataset(BaseDataset):

    def __init__(self, dataset_path, sentinel2_vars, era5_vars, years):

        super().__init__(dataset_path, sentinel2_vars, era5_vars, years)
        self.training_pairs = self._create_training_pairs(
            years=years[:-1]
        )  # year + 1 need to be available for forecasting

    def __getitem__(self, idx):

        sample_id, year = self.training_pairs[idx]
        sample = self.dataset.isel(sample=sample_id)

        veg_hist, weather_hist = self._load_year(sample, year)
        veg_forecast, weather_forecast = self._load_year(sample, year + 1)
        return {
            "vegetation_history": veg_hist,
            "weather_history": weather_hist,
            "vegetation_forecast": veg_forecast,
            "weather_forecast": weather_forecast,
        }
        # except Exception as e:


#
#     logging.warning(f"Skipping {(sample_id, year)}: {e}")
#     return None
