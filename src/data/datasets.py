import xarray as xr
import numpy as np
import torch

import pandas as pd
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    BaseDataset for Sentinel-2 + ERA5 time series.
    Preloads entire dataset into RAM as PyTorch tensors.
    """

    def __init__(
        self, dataset_path: str, sentinel2_vars: list, era5_vars: list, years=list
    ):
        self.dataset = self.dataset_split(dataset_path=dataset_path, years=years)
        self.num_samples = len(self.dataset.sample)
        print(f"Dataset loaded with {self.num_samples} samples.")
        self.sentinel2_vars = sentinel2_vars
        self.era5_vars = era5_vars

        # Pre-compute year masks once during init
        self._precompute_year_indices()
        print("Loading all samples into memory.")
        self.samples = [self._preload_sample(i, years) for i in range(self.num_samples)]
        print("Preloaded all samples into memory.")

    def dataset_split(self, dataset_path: str, years: list[int]) -> xr.Dataset:
        """
        Create a dataset split containing only data from specified years.

        Args:
            dataset: Input dataset with time_veg and time_weather dimensions
            years: List of years to include in this split

        Returns:
            Dataset filtered to only include samples from the specified years
        """
        print(f"Loading dataset from {dataset_path} and filtering for years {years}")
        dataset = xr.open_zarr(dataset_path)
        return dataset.sel(
            time_veg=pd.DatetimeIndex(dataset.time_veg.values).year.isin(years),
            time_weather=pd.DatetimeIndex(dataset.time_weather.values).year.isin(years),
        ).load()

    def _create_training_pairs(self, years):
        """Create list of (idx_sample, year) pairs."""
        self.training_pairs = [
            (idx_sample, year)
            for idx_sample in range(self.num_samples)
            for year in years
        ]

    def _precompute_year_indices(self):
        """Pre-compute which time indices belong to each year."""
        #   Vegetation
        # Extract year for each timestamp
        veg_years = pd.DatetimeIndex(self.dataset.time_veg.values).year
        # Create a dictionary mapping year → array of indices
        self.veg_year_masks = {
            year: (veg_years == year).nonzero()[0] for year in pd.unique(veg_years)
        }

        #   Weather
        weather_years = pd.DatetimeIndex(self.dataset.time_weather.values).year
        self.weather_year_masks = {
            year: (weather_years == year).nonzero()[0]
            for year in pd.unique(weather_years)
        }

    def _load_year_tensor(self, sample, year):
        veg_idx = self.veg_year_masks[year]
        weather_idx = self.weather_year_masks[year]

        veg_arr = torch.as_tensor(
            np.stack(
                [sample[var].values[veg_idx] for var in self.sentinel2_vars], axis=1
            ),
            dtype=torch.float32,
        )
        weather_arr = torch.as_tensor(
            np.stack(
                [sample[var].values[weather_idx] for var in self.era5_vars], axis=1
            ),
            dtype=torch.float32,
        )

        return veg_arr, weather_arr

    def _preload_sample(self, sample_id, years):
        """Precompute tensors for all years of a given sample."""
        sample = self.dataset.isel(sample=sample_id)
        sample_cache = {}
        for year in years:
            sample_cache[year] = self._load_year_tensor(sample, year)
        return sample_cache

    def __len__(self):
        if self.training_pairs is None:
            raise ValueError("training_pairs must be defined in subclass")
        return len(self.training_pairs)


class ContrastiveDataset(BaseDataset):

    def __init__(self, dataset_path, sentinel2_vars, era5_vars, years):

        super().__init__(dataset_path, sentinel2_vars, era5_vars, years)
        self._create_training_pairs(years=years)

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
        self._create_training_pairs(
            years=years[1:]
        )  # year + 1 need to be available for forecasting

    def __getitem__(self, idx):

        sample_id, year = self.training_pairs[idx]
        try:
            sample = self.samples[sample_id]

            veg_hist, weather_hist = sample[year - 1]
            veg_forecast, weather_forecast = sample[year]

            return {
                "vegetation_history": veg_hist,
                "weather_history": weather_hist,
                "vegetation_forecast": veg_forecast,
                "weather_forecast": weather_forecast,
            }
        except Exception as e:
            # logging.warning(f"Skipping {(sample_id, year)}: {e}")
            return None
