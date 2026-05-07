from matplotlib.pylab import sample
import xarray as xr
import numpy as np
import torch
import logging
import pandas as pd
from torch.utils.data import Dataset
import copy


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
        print(
            f"Dataset loaded with {self.num_samples} samples. years: {[i for i in years]}"
        )
        self.sentinel2_vars = sentinel2_vars
        self.era5_vars = era5_vars

        # Pre-compute year masks once during init
        self._precompute_year_indices()
        self.locations = list(
            zip(self.dataset.latitude.values, self.dataset.longitude.values)
        )

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

    def _compute_max_evi(self, veg_arr):
        nan_mask = torch.isnan(veg_arr)
        veg_arr = copy.deepcopy(veg_arr)
        veg_arr = torch.where(nan_mask, torch.tensor(float("-inf")), veg_arr)
        return torch.max(veg_arr)

    def _compute_sum_precip(self, weather_arr):
        # precipitation is either "tp_max" or "P"
        if "tp_max" in self.era5_vars:
            precip_idx = self.era5_vars.index("tp_max")
        elif "P" in self.era5_vars:
            precip_idx = self.era5_vars.index("P")
        else:
            raise ValueError(
                "No precipitation variable found in era5_vars. Must include either 'tp_max' or 'P'."
            )
        return torch.sum(weather_arr[:, precip_idx])

    def _get_msc(self, sample):
        msc = sample["msc"]
        msc_arr = torch.as_tensor(
            msc,
            dtype=torch.float32,
        )
        return msc_arr

    def _preload_sample(self, sample_id, years):
        """Precompute tensors for all years of a given sample."""
        sample = self.dataset.isel(sample=sample_id)
        sample_cache = {}
        for year in years:
            sample_cache[year] = self._load_year_tensor(sample, year)
        return sample_cache

    def _validate_tensors(self, *tensors):
        for t in tensors:
            if torch.isnan(t).all() or torch.isinf(t).any():
                raise ValueError("Tensor contains only NaNs or infs")

    def __len__(self):
        if self.training_pairs is None:
            raise ValueError("training_pairs must be defined in subclass")
        return len(self.training_pairs)


class ContrastiveDataset(BaseDataset):

    def __init__(self, dataset_path, sentinel2_vars, era5_vars, years):

        super().__init__(dataset_path, sentinel2_vars, era5_vars, years)
        self._create_training_pairs(years=years)
        self.samples = [self._preload_sample(i, years) for i in range(self.num_samples)]

    def __getitem__(self, idx) -> dict | None:
        sample_id, year = self.training_pairs[idx]
        try:
            vegetation, weather = self.samples[sample_id][year]
            self._validate_tensors(vegetation, weather)

            max_evi = self._compute_max_evi(vegetation)
            sum_precip = self._compute_sum_precip(weather)

            data = {
                "vegetation": vegetation,
                "weather": weather,
                "max_evi": max_evi,
                "sum_precip": sum_precip,
                "location": self.locations[sample_id],
            }
            return data
        except Exception as e:
            # logging.warning(f"Skipping {sample_id}: {e}")
            return None


class ForecastingTrainDatasetOld(BaseDataset):

    def __init__(
        self,
        dataset_path,
        sentinel2_vars,
        era5_vars,
        years,
    ):
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

            self._validate_tensors(
                veg_hist, weather_hist, veg_forecast, weather_forecast
            )

            return {
                "vegetation_history": veg_hist,
                "weather_history": weather_hist,
                "vegetation_forecast": veg_forecast,
                "weather_forecast": weather_forecast,
            }
        except Exception as e:
            # logging.warning(f"Skipping {(sample_id, year)}: {e}")
            return None


class ForecastingValDatasetOld(BaseDataset):

    def __init__(
        self,
        dataset_path,
        percentiles_path,
        sentinel2_vars,
        era5_vars,
        years,
    ):
        super().__init__(dataset_path, sentinel2_vars, era5_vars, years)
        self._create_training_pairs(
            years=years[1:]
        )  # year + 1 need to be available for forecasting
        self.extremes_dataset = self.additional_dataset(  # could technically download only years[1:], but veg_idx might be an issue
            dataset_path=percentiles_path, years=years
        )
        self.extremes = [
            self._preload_additional_variable(i, years) for i in range(self.num_samples)
        ]

    def additional_dataset(self, dataset_path: str, years: list[int]) -> xr.Dataset:
        """
        Create a dataset split containing only data from specified years.

        Args:
            dataset: Input dataset with time_veg and time_weather dimensions
            years: List of years to include in this split

        Returns:
            Dataset filtered to only include samples from the specified years
        """
        dataset = xr.open_zarr(dataset_path)
        dataset = dataset.chunk({"location": 1000, "time": 365})

        if "location" in dataset.dims:
            dataset = dataset.reset_index("location")
            dataset = dataset.rename(location="sample")
        if "time" in dataset.dims:
            dataset = dataset.sel(
                time=pd.DatetimeIndex(dataset.time.values).year.isin(years),
            )
            dataset = dataset.rename(time="time_veg")
        return dataset.load()

    def _load_year_additional_tensor(self, sample, year, variables):
        veg_idx = self.veg_year_masks[year]
        veg_arr = torch.as_tensor(
            np.stack([sample[var].values[veg_idx] for var in variables], axis=1),
            dtype=torch.float32,
        )
        return veg_arr

    def _preload_additional_variable(self, sample_id, years):
        """Precompute tensors for all years of a given sample."""
        sample = self.extremes_dataset.isel(sample=sample_id)
        sample_cache = {}
        for year in years:
            sample_cache[year] = self._load_year_additional_tensor(
                sample, year, ["extremes"]
            )
        return sample_cache

    def __getitem__(self, idx):

        sample_id, year = self.training_pairs[idx]
        try:
            sample = self.samples[sample_id]

            veg_hist, weather_hist = sample[year - self.memory_length : year - 1]
            veg_forecast, weather_forecast = sample[year]

            # msc = self._get_msc(sample)
            max_evi = self._compute_max_evi(veg_forecast)
            sum_precip = self._compute_sum_precip(weather_forecast)

            # percentiles_forecast = self.extremes[sample_id][year]

            self._validate_tensors(
                veg_hist, weather_hist, veg_forecast, weather_forecast
            )
            return {
                "vegetation_history": veg_hist,
                "weather_history": weather_hist,
                "vegetation_forecast": veg_forecast,
                "weather_forecast": weather_forecast,
                # "percentiles_forecast": percentiles_forecast,
                "max_evi": max_evi,
                "sum_precip": sum_precip,
                "location": self.locations[sample_id],
                "time": torch.tensor(
                    self.dataset.time_veg.values[self.veg_year_masks[year]]
                    .astype("datetime64[s]")
                    .astype(np.int64)
                ),
            }
        except Exception as e:
            # logging.warning(f"Skipping {(sample_id, year)}: {e}")
            return None


class ForecastingTrainDataset(BaseDataset):

    def __init__(
        self,
        dataset_path,
        sentinel2_vars,
        era5_vars,
        years,
        memory_length,
    ):
        super().__init__(
            dataset_path,
            sentinel2_vars,
            era5_vars,
            years=list(range(years[0] - memory_length, years[-1] + 1)),
        )
        self.memory_length = memory_length
        self._create_training_pairs(years=years)
        self.samples = [self._preload_sample(i, years) for i in range(self.num_samples)]

    def __getitem__(self, idx):

        sample_id, year = self.training_pairs[idx]
        # try:
        sample = self.samples[sample_id]
        past_years = list(range(year - self.memory_length, year))
        # if len(past_years) > 1:
        veg_hist = torch.cat([sample[y][0] for y in past_years], dim=0)
        weather_hist = torch.cat([sample[y][1] for y in past_years], dim=0)
        # else:
        #     veg_hist, weather_hist = sample[year - 1]
        veg_forecast, weather_forecast = sample[year]
        self._validate_tensors(veg_hist, weather_hist, veg_forecast, weather_forecast)

        return {
            "vegetation_history": veg_hist,
            "weather_history": weather_hist,
            "vegetation_forecast": veg_forecast,
            "weather_forecast": weather_forecast,
        }
        # except Exception as e:
        #     logging.warning(f"Skipping {(sample_id, year)}: {e}")
        #     return None


class ForecastingValDataset(BaseDataset):

    def __init__(
        self,
        dataset_path,
        percentiles_path,
        sentinel2_vars,
        era5_vars,
        years,
        memory_length,
    ):
        super().__init__(
            dataset_path,
            sentinel2_vars,
            era5_vars,
            years=list(range(years[0] - memory_length, years[-1] + 1)),
        )

        self.memory_length = memory_length
        self._create_training_pairs(
            years=years
        )  # year + 1 need to be available for forecasting
        # self.extremes_dataset = self.additional_dataset(  # could technically download only years[1:], but veg_idx might be an issue
        #     dataset_path=percentiles_path, years=years
        # )
        # self.extremes = [
        #     self._preload_additional_variable(i, years) for i in range(self.num_samples)
        # ]

    def additional_dataset(self, dataset_path: str, years: list[int]) -> xr.Dataset:
        """
        Create a dataset split containing only data from specified years.

        Args:
            dataset: Input dataset with time_veg and time_weather dimensions
            years: List of years to include in this split

        Returns:
            Dataset filtered to only include samples from the specified years
        """
        dataset = xr.open_zarr(dataset_path)
        dataset = dataset.chunk({"location": 1000, "time": 365})

        if "location" in dataset.dims:
            dataset = dataset.reset_index("location")
            dataset = dataset.rename(location="sample")
        if "time" in dataset.dims:
            dataset = dataset.sel(
                time=pd.DatetimeIndex(dataset.time.values).year.isin(years),
            )
            dataset = dataset.rename(time="time_veg")
        return dataset.load()

    def _load_year_additional_tensor(self, sample, year, variables):
        veg_idx = self.veg_year_masks[year]
        veg_arr = torch.as_tensor(
            np.stack([sample[var].values[veg_idx] for var in variables], axis=1),
            dtype=torch.float32,
        )
        return veg_arr

    def _preload_additional_variable(self, sample_id, years):
        """Precompute tensors for all years of a given sample."""
        sample = self.extremes_dataset.isel(sample=sample_id)
        sample_cache = {}
        for year in years:
            sample_cache[year] = self._load_year_additional_tensor(
                sample, year, ["extremes"]
            )
        return sample_cache

    def __getitem__(self, idx):

        sample_id, year = self.training_pairs[idx]
        try:
            sample = self.samples[sample_id]
            past_years = list(range(year - self.memory_length, year))
            # if len(past_years) > 1:
            veg_hist = torch.cat([sample[y][0] for y in past_years], dim=0)
            weather_hist = torch.cat([sample[y][1] for y in past_years], dim=0)
            # else:
            #     veg_hist, weather_hist = sample[year - 1]
            veg_forecast, weather_forecast = sample[year]

            msc = self._get_msc(sample)
            max_evi = self._compute_max_evi(veg_forecast)
            sum_precip = self._compute_sum_precip(weather_forecast)

            # percentiles_forecast = self.extremes[sample_id][year]

            self._validate_tensors(
                veg_hist, weather_hist, veg_forecast, weather_forecast
            )
            return {
                "vegetation_history": veg_hist,
                "weather_history": weather_hist,
                "vegetation_forecast": veg_forecast,
                "weather_forecast": weather_forecast,
                # "percentiles_forecast": percentiles_forecast,
                "max_evi": max_evi,
                "sum_precip": sum_precip,
                "location": self.locations[sample_id],
                "time": torch.tensor(
                    self.dataset.time_veg.values[self.veg_year_masks[year]]
                    .astype("datetime64[s]")
                    .astype(np.int64)
                ),
            }
        except Exception as e:
            # logging.warning(f"Skipping {(sample_id, year)}: {e}")
            return None


class ForecastingAnomTrainDataset(BaseDataset):

    def __init__(
        self,
        dataset_path,
        sentinel2_vars,
        era5_vars,
        years,
        memory_length,
    ):
        years = list(range(years[0] - memory_length, years[-1] + 1))
        super().__init__(
            dataset_path,
            sentinel2_vars,
            era5_vars,
            years=years,
        )
        self._create_training_pairs(years=years[1:])
        self.msc = [self._preload_msc(i) for i in range(self.num_samples)]
        self.samples = [self._preload_sample(i, years) for i in range(self.num_samples)]

    def _preload_sample(self, sample_id, years):
        """Precompute tensors for all years of a given sample."""
        sample = self.dataset.isel(sample=sample_id)
        sample_cache = {}
        for year in years:
            sample_cache[year] = self._load_year_tensor(sample, year)
        return sample_cache

    def _preload_msc(self, sample_id):
        msc = self.dataset.isel(sample=sample_id).msc
        if msc.size == 366:
            msc = msc.drop_isel(dayofyear=59)  # leap year
        msc_arr = torch.as_tensor(
            msc.values,
            dtype=torch.float32,
        ).unsqueeze(-1)
        return msc_arr

    def _load_year_tensor(self, sample, year):
        veg_idx = self.veg_year_masks[year]
        weather_idx = self.weather_year_masks[year]
        anom_arr = torch.as_tensor(
            sample["anomalies"].values[veg_idx],
            dtype=torch.float32,
        ).unsqueeze(
            -1
        )  # T, C=1
        weather_arr = torch.as_tensor(
            np.stack(
                [sample[var].values[weather_idx] for var in self.era5_vars], axis=1
            ),
            dtype=torch.float32,
        )

        return anom_arr, weather_arr

    def __getitem__(self, idx):
        sample_id, year = self.training_pairs[idx]
        try:
            sample = self.samples[sample_id]
            msc = self.msc[sample_id]
            anom_hist, weather_hist = sample[year - 1]
            anom_forecast, weather_forecast = sample[year]
            self._validate_tensors(anom_hist, anom_forecast, weather_forecast)

            return {
                "vegetation_history": anom_hist,
                "weather_history": weather_hist,
                "vegetation_forecast": anom_forecast,
                "weather_forecast": weather_forecast,
                "location": self.locations[sample_id],
            }
        except Exception as e:
            # logging.warning(f"Skipping {(sample_id, year)}: {e}")
            return None
