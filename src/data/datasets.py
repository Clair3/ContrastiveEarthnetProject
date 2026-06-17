from os import times
from weakref import ref

from matplotlib.pylab import sample
import xarray as xr
import numpy as np
import torch
import logging
import pandas as pd
from torch.utils.data import Dataset
import copy
import random


class BaseDataset(Dataset):

    def __init__(
        self,
        dataset_path,
        vegetation_indices,
        weather_vars,
        years,
        context_length,
        prediction_length,
    ):
        self.dataset = self.dataset_split(
            dataset_path=dataset_path,
            years=years,
        )

        self.vegetation_indices = vegetation_indices
        self.weather_vars = weather_vars

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = len(self.dataset.sample)
        self.stride = 32

        self._create_training_pairs()
        self._preload_data()
        self.locations = {
            sample_id: (
                float(self.dataset.latitude.values[sample_id]),
                float(self.dataset.longitude.values[sample_id]),
            )
            for sample_id in range(self.num_samples)
        }

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

    def _preload_data(self):

        self.vegetation = []
        self.weather = []

        for sample_id in range(self.num_samples):

            sample = self.dataset.isel(sample=sample_id)

            veg = torch.as_tensor(
                np.stack(
                    [sample[var].values for var in self.vegetation_indices],
                    axis=1,
                ),
                dtype=torch.float32,
            )

            weather = torch.as_tensor(
                np.stack(
                    [sample[var].values for var in self.weather_vars],
                    axis=1,
                ),
                dtype=torch.float32,
            )

            self.vegetation.append(veg)
            self.weather.append(weather)

    def _create_training_pairs(self):

        n_time = len(self.dataset.time_veg)

        origins = range(
            self.context_length,
            n_time - self.prediction_length + 1,
            self.stride,
        )
        self.training_pairs = [
            (sample_id, t0) for sample_id in range(self.num_samples) for t0 in origins
        ]

        print(
            f"n_time={n_time}, "
            f"context={self.context_length}, "
            f"prediction={self.prediction_length}"
        )

        print(
            f"num_samples={self.num_samples}, "
            f"num_windows={len(self.training_pairs)}"
        )

    def __len__(self):
        return len(self.training_pairs)

    def _validate_tensors(self, *tensors):
        for t in tensors:
            if torch.isnan(t).all() or torch.isinf(t).any():
                raise ValueError("Tensor contains only NaNs or infs")

    # def __len__(self):
    #     return self.num_samples * self.n_windows


class BaseDataset2(Dataset):
    """
    BaseDataset for Sentinel-2 + ERA5 time series.
    Preloads entire dataset into RAM as PyTorch tensors.
    """

    def __init__(
        self,
        dataset_path: str,
        vegetation_indices: list,
        weather_vars: list,
        years=list[int],
        start_doy=1,
    ):
        self.dataset = self.dataset_split(dataset_path=dataset_path, years=years)
        self.num_samples = len(self.dataset.sample)
        print(
            f"Dataset loaded with {self.num_samples} samples. years: {[i for i in years]}"
        )
        self.vegetation_indices = vegetation_indices
        self.weather_vars = weather_vars

        # Pre-compute year masks once during init
        self._precompute_year_indices(start_doy=start_doy)

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
            (idx_sample, year, doy)
            for idx_sample in range(self.num_samples)
            for year in years
            for doy in range(1, 366)
        ]

    def _precompute_year_indices(self, start_doy=1):
        """Pre-compute which time indices belong to each year."""

        def _get_custom_year(times, start_doy):
            ref = pd.Timestamp("2001-01-01") + pd.Timedelta(days=start_doy - 1)
            start_month = ref.month
            start_day = ref.day

            dt = pd.DatetimeIndex(times)

            before_start = (dt.month < start_month) | (
                (dt.month == start_month) & (dt.day < start_day)
            )

            return np.where(before_start, dt.year - 1, dt.year)

        #   Vegetation
        # Extract year for each timestamp
        veg_years = _get_custom_year(self.dataset.time_veg.values, start_doy=start_doy)
        # Create a dictionary mapping year → array of indices
        self.veg_year_masks = {
            year: (veg_years == year).nonzero()[0]
            for year in pd.unique(veg_years)  # skip the first year if shiffted
        }
        #   Weather
        weather_years = _get_custom_year(
            self.dataset.time_weather.values, start_doy=start_doy
        )
        self.weather_year_masks = {
            year: (weather_years == year).nonzero()[0]
            for year in pd.unique(weather_years)
        }

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
        veg_arr = torch.as_tensor(
            np.stack(
                [sample[var].values[veg_idx] for var in self.vegetation_indices], axis=1
            ),
            dtype=torch.float32,
        )
        weather_arr = torch.as_tensor(
            np.stack(
                [sample[var].values[weather_idx] for var in self.weather_vars], axis=1
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
        if "tp_max" in self.weather_vars:
            precip_idx = self.weather_vars.index("tp_max")
        elif "P" in self.weather_vars:
            precip_idx = self.weather_vars.index("P")
        else:
            raise ValueError(
                "No precipitation variable found in weather_vars. Must include either 'tp_max' or 'P'."
            )
        return torch.sum(weather_arr[:, precip_idx])

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

    def __init__(self, dataset_path, vegetation_indices, weather_vars, years):

        super().__init__(dataset_path, vegetation_indices, weather_vars, years)
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


class ForecastingTrainDataset(BaseDataset):

    def __init__(
        self,
        dataset_path,
        vegetation_indices,
        weather_vars,
        years,
        memory_length,
    ):
        super().__init__(
            dataset_path,
            vegetation_indices,
            weather_vars,
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
        vegetation_indices,
        weather_vars,
        years,
        memory_length,
    ):
        super().__init__(
            dataset_path,
            vegetation_indices,
            weather_vars,
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

            msc = self.msc(sample)
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

    def __getitem__(self, idx):
        try:
            sample_id, t0 = self.training_pairs[idx]

            veg = self.vegetation[sample_id]
            weather = self.weather[sample_id]

            hist_slice = slice(
                t0 - self.context_length,
                t0,
            )

            forecast_slice = slice(
                t0,
                t0 + self.prediction_length,
            )

            vegetation_history = veg[hist_slice]
            weather_history = weather[hist_slice]

            vegetation_forecast = veg[forecast_slice]
            weather_forecast = weather[forecast_slice]

            self._validate_tensors(
                vegetation_history, vegetation_forecast, weather_forecast
            )

            # msc = self.msc[sample_id]

            return {
                "vegetation_history": vegetation_history,
                "weather_history": weather_history,
                "vegetation_forecast": vegetation_forecast,
                "weather_forecast": weather_forecast,
                # "msc": msc,
            }
        except Exception as e:
            # logging.warning(f"Skipping {(sample_id, year)}: {e}")
            return None


class ForecastingAnomValDataset(BaseDataset):

    def __getitem__(self, idx):
        try:
            sample_id, t0 = self.training_pairs[idx]

            veg = self.vegetation[sample_id]
            weather = self.weather[sample_id]

            hist_slice = slice(
                t0 - self.context_length,
                t0,
            )

            forecast_slice = slice(
                t0,
                t0 + self.prediction_length,
            )

            vegetation_history = veg[hist_slice]
            weather_history = weather[hist_slice]

            vegetation_forecast = veg[forecast_slice]
            weather_forecast = weather[forecast_slice]
            self._validate_tensors(
                vegetation_history, vegetation_forecast, weather_forecast
            )

            # msc = self.msc[sample_id]

            return {
                "vegetation_history": vegetation_history,
                "weather_history": weather_history,
                "vegetation_forecast": vegetation_forecast,
                "weather_forecast": weather_forecast,
                # "msc": msc,
                "sample_id": torch.tensor(sample_id),
                "forecast_origin": torch.tensor(t0),
                "location": self.locations[sample_id],
            }
        except Exception as e:
            # logging.warning(f"Skipping {(sample_id, year)}: {e}")
            return None


class ForecastingAnomTrainDataset_old(BaseDataset):

    def __init__(
        self,
        dataset_path,
        vegetation_indices,
        weather_vars,
        years,
        memory_length,
        start_doy=1,
    ):
        super().__init__(
            dataset_path,
            vegetation_indices,
            weather_vars,
            years=list(range(years[0] - memory_length, years[-1] + 1 + 1)),
            start_doy=start_doy,
        )
        print("training years:", years)
        self.memory_length = memory_length
        self._create_training_pairs(years=years)
        self.msc = [self._preload_msc(i) for i in range(self.num_samples)]
        self.samples = [
            self._preload_sample(
                i, list(range(years[0] - memory_length, years[-1] + 1))
            )
            for i in range(self.num_samples)
        ]

        self.locations = {
            sample_id: (
                float(self.dataset.latitude.values[sample_id]),
                float(self.dataset.longitude.values[sample_id]),
            )
            for sample_id in range(self.num_samples)
        }

        self.year_times = {
            year: self.dataset.time_veg.values[self.veg_year_masks[year]]
            for year in self.veg_year_masks
        }

    def _preload_sample(self, sample_id, years):
        """Precompute tensors for all years of a given sample."""
        sample = self.dataset.isel(sample=sample_id)
        sample_cache = {}
        for year in years:
            sample_cache[year] = self._load_year_tensor(sample, year)
        return sample_cache

    def __getitem__(self, idx):
        sample_id, year = self.training_pairs[idx]
        try:
            sample = self.samples[sample_id]
            past_years = list(range(year - self.memory_length, year))
            anom_hist = torch.cat([sample[y][0] for y in past_years], dim=0)
            weather_hist = torch.cat([sample[y][1] for y in past_years], dim=0)
            anom_forecast, weather_forecast = sample[year]
            msc = self.msc[sample_id]
            self._validate_tensors(anom_hist, anom_forecast, weather_forecast)

            # forecast_len = anom_forecast.shape[0]
            # window_size = random.randint(30, 180)
            # start = random.randint(0, forecast_len - window_size)
            # end = start + window_size
            # forecast_mask = torch.full((forecast_len,), float("nan"))
            # forecast_mask[0:90] = 1
            return {
                "vegetation_history": anom_hist,
                "weather_history": weather_hist,
                "vegetation_forecast": anom_forecast,  # * forecast_mask,
                "weather_forecast": weather_forecast,  # * forecast_mask,
                "msc": msc,
            }
        except Exception as e:
            # logging.warning(f"Skipping {(sample_id, year)}: {e}")
            return None


class ForecastingAnomValDataset_old(ForecastingAnomTrainDataset_old):

    def __getitem__(self, idx):
        sample_id, year = self.training_pairs[idx]
        try:
            sample = self.samples[sample_id]
            past_years = list(range(year - self.memory_length, year))
            anom_hist = torch.cat([sample[y][0] for y in past_years], dim=0)
            weather_hist = torch.cat([sample[y][1] for y in past_years], dim=0)
            anom_forecast, weather_forecast = sample[year]
            msc = self.msc[sample_id]
            self._validate_tensors(anom_hist, anom_forecast, weather_forecast)
            return {
                "vegetation_history": anom_hist,
                "weather_history": weather_hist,
                "vegetation_forecast": anom_forecast,
                "weather_forecast": weather_forecast,
                "msc": msc,
                # "location": self.locations[sample_id],
                "sample_id": sample_id,
                "year": year,
                # "time": self.dataset.time_veg.values[self.veg_year_masks[year]],
            }
        except Exception as e:
            # logging.warning(
            #     f"Skipping {(sample_id, year)} {type(sample_id), type(year)}: {e}"
            # )
            return None
