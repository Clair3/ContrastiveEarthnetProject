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

from torchvision.transforms import Compose

from .transform import RandomFeatureMask, RandomGaussianJitter


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
        self.vegetation_indices = vegetation_indices
        self.weather_vars = weather_vars

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = 16

        self.dataset = self.dataset_split(
            dataset_path=dataset_path,
            years=years,
        )
        self.num_samples = len(self.dataset.sample)

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
        dataset = xr.open_zarr(dataset_path)
        print(f"Loading dataset from {dataset_path} and filtering for years {years}")

        dataset = dataset.sel(
            time_veg=pd.DatetimeIndex(dataset.time_veg.values).year.isin(years),
            time_weather=pd.DatetimeIndex(dataset.time_weather.values).year.isin(years),
        )

        # weather_vars = [
        #     v for v, da in dataset.data_vars.items() if "time_weather" in da.dims
        # ]
        # veg_vars = [v for v, da in dataset.data_vars.items() if "time_veg" in da.dims]
        #
        # veg = (
        #     dataset[veg_vars]
        #     .rename(time_veg="time")
        #     .groupby("time.year")
        #     .map(lambda ds: ds.coarsen(time=5, boundary="trim").mean())
        #     .rename(time="time_veg")
        # )
        #
        # weather = (
        #     dataset[weather_vars]
        #     .rename(time_weather="time")
        #     .groupby("time.year")
        #     .map(lambda ds: ds.coarsen(time=5, boundary="trim").mean())
        #     .rename(time="time_weather")
        # )
        # out = dataset.drop_vars(veg_vars + weather_vars)
        # dataset = xr.merge([out, veg, weather])

        return dataset.load()

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

    def _preload_data(self):

        self.vegetation = []
        self.weather = []
        self.msc = []
        self.weather_msc = []
        self._build_doy_lookup()

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
                    [sample[f"{var}_anom"].values for var in self.weather_vars],
                    axis=1,
                ),
                dtype=torch.float32,
            )

            msc = self._preload_msc(sample_id)
            weather_msc = self._preload_weather_msc(sample_id)

            # Expand climatology onto the time axis
            msc_time = msc[self.doy_lookup]
            weather_msc_time = weather_msc[self.doy_lookup]

            self.vegetation.append(veg)
            self.weather.append(weather)
            self.msc.append(msc_time)
            self.weather_msc.append(weather_msc_time)

    def _preload_msc(self, sample_id):
        msc = self.dataset.isel(sample=sample_id).msc
        if msc.size == 366:
            msc = msc.drop_isel(dayofyear=59)  # leap year
        msc_arr = torch.as_tensor(
            msc.values,
            dtype=torch.float32,
        ).unsqueeze(-1)
        return msc_arr

    def _preload_weather_msc(self, sample_id):

        sample = self.dataset.isel(sample=sample_id)

        weather_msc = np.stack(
            [sample[f"{var}_msc"].values for var in self.weather_vars],
            axis=1,
        )

        # Remove Feb 29 if present
        if weather_msc.shape[0] == 366:
            weather_msc = np.delete(weather_msc, 59, axis=0)

        return torch.as_tensor(
            weather_msc,
            dtype=torch.float32,
        )

    def _build_doy_lookup(self):

        time = self.dataset.time_veg

        doy = time.dt.dayofyear.values.copy()
        is_leap = time.dt.is_leap_year.values
        # if doy.shape[0] == 366:
        # Match the MSC where Feb 29 has been removed
        doy[(doy == 60) & is_leap] = 59
        doy[(doy > 60) & is_leap] -= 1

        # Convert 1..365 -> 0..364
        self.doy_lookup = torch.as_tensor(
            doy - 1,
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.training_pairs)

    def _validate_tensors(self, *tensors):
        for t in tensors:
            if torch.isnan(t).all() or torch.isinf(t).any():
                raise ValueError("Tensor contains only NaNs or infs")

    def _get_sample(self, sample_id, t0, include_metadata=False):

        veg = self.vegetation[sample_id]
        weather = self.weather[sample_id]
        weather_msc = self.weather_msc[sample_id]
        msc = self.msc[sample_id]

        msc_slice = slice(
            t0 - 365,
            t0,
        )

        hist_slice = slice(
            t0 - self.context_length,
            t0,
        )

        forecast_slice = slice(
            t0,
            t0 + self.prediction_length,
        )

        weather_msc = weather_msc[msc_slice]
        msc = msc[msc_slice]

        vegetation_history = veg[hist_slice]
        weather_history = weather[hist_slice]

        vegetation_forecast = veg[forecast_slice]
        weather_forecast = weather[forecast_slice]

        hist_doy = self.doy_lookup[hist_slice]
        climate_doy = self.doy_lookup[msc_slice]
        forecast_doy = self.doy_lookup[forecast_slice]

        self._validate_tensors(
            vegetation_history,
            vegetation_forecast,
            weather_forecast,
        )
        # vegetation_history[-30:] = torch.nan

        sample = {
            "msc": msc,
            "weather_msc": weather_msc,
            "vegetation_history": vegetation_history,
            "weather_history": weather_history,
            "vegetation_forecast": vegetation_forecast,
            "weather_forecast": weather_forecast,
            "hist_doy": hist_doy,
            "climate_doy": climate_doy,
            "forecast_doy": forecast_doy,
        }

        if include_metadata:
            sample.update(
                {
                    "sample_id": torch.tensor(sample_id),
                    "location": self.locations[sample_id],
                    "forecast_origin": torch.tensor(t0),
                }
            )

        return sample


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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transform = Compose(
            [
                RandomFeatureMask(
                    keys=[
                        "weather_history",
                        "vegetation_history",
                        "msc",
                        "weather_msc",
                    ],
                    p_feature=0.2,
                ),
                RandomGaussianJitter(
                    keys=[
                        "weather_history",
                        "vegetation_history",
                        "msc",
                        "weather_msc",
                    ],
                    p_feature=0.5,
                    sigma=0.05,
                ),
            ]
        )

    def __getitem__(self, idx):
        try:
            sample_id, t0 = self.training_pairs[idx]
            sample = self._get_sample(sample_id, t0)

            if self.transform is not None:
                sample = self.transform(sample)

            return sample

        except Exception as e:
            # print(f"Skipping sample {idx} due to {e}.")
            return None


class ForecastingValDataset(BaseDataset):

    def __getitem__(self, idx):
        try:
            sample_id, t0 = self.training_pairs[idx]
            return self._get_sample(
                sample_id,
                t0,
                include_metadata=True,
            )
        except Exception as e:
            # print(f"Skipping sample {idx} due to {e}.")
            return None
