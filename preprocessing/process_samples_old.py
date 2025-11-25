import xarray as xr
import pandas as pd
import numpy as np
import torch
import xarray as xr
import os
import logging
from .process_vegetation import Sentinel2Preprocessing
from .process_weather import weather_normalization
from pathlib import Path
from multiprocessing import Pool, cpu_count
from abc import ABC, abstractmethod


class ProcessTrainDataset(ABC):
    def __init__(
        self,
        dataset_path,
        train_years,
        temporal_resolution_veg,
        temporal_resolution_weather,
        output_dir,
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.sample_paths = [str(p) for p in Path(dataset_path).glob("*/*.zarr")]
        self.train_years = train_years

        self.temporal_resolution_veg = temporal_resolution_veg
        self.temporal_resolution_weather = temporal_resolution_weather

        self.era5_variables = [
            "t2m_mean",  # 2-meter air temperature
            "tp_mean",  # Total precipitation
            "t2m_min",
            "tp_min",
            "t2m_max",
            "tp_max",
            "pev_mean",  # Potential evapotranspiration
            "ssr_mean",  # Surface solar radiation
            "pev_min",
            "ssr_min",
            "pev_max",
            "ssr_max",
        ]

    def __len__(self):
        return len(self.sample_paths)

    def build_multi_year_zarr(self, sample_path: str) -> xr.Dataset:
        ds = xr.open_zarr(sample_path)

        year_datasets = []
        for year in self.train_years:
            vegetation = self._process_vegetation(ds, year)
            weather = self._process_weather(ds, year)

            lat = vegetation.location.latitude.item()
            lon = vegetation.location.longitude.item()

            ds_year = xr.Dataset(
                {
                    "vegetation": vegetation,
                    "weather": weather,
                },
                coords={
                    "latitude": lat,
                    "longitude": lon,
                    "time": vegetation.time,
                    "year": year,
                },
            )
            year_datasets.append(ds_year)

        combined = xr.concat(year_datasets, dim="year")
        return combined

    def process_one_sample(self, path):
        combined = self.build_multi_year_zarr(path)
        lat = combined.latitude.item()
        lon = combined.longitude.item()
        out = f"{self.output_dir}/{path.stem}_{lat}_{lon}.zarr"
        combined.to_zarr(out, mode="w")

    # --- helper functions ---
    def _process_vegetation(self, ds: xr.Dataset, year: int) -> xr.DataArray:
        veg_index = Sentinel2Preprocessing(
            temporal_resolution=self.temporal_resolution_veg
        ).generate_masked_vegetation_index(ds)
        if veg_index is None:
            raise ValueError("Vegetation index computation failed")
        return self.select_year(
            veg_index,
            temporal_resolution=self.temporal_resolution_veg,
            selected_year=year,
        )

    def _process_weather(self, ds: xr.Dataset, year: int) -> xr.DataArray:
        weather = ds[self.era5_variables]
        # Keep only training years
        weather = weather.sel(time=weather["time.year"].isin(self.train_years))
        weather = weather_normalization(weather)
        return self.select_year(
            weather,
            temporal_resolution=self.temporal_resolution_weather,
            selected_year=year,
        )

    # Helper function to select and fill data for a specific year
    def select_year(data, selected_year, temporal_resolution=16):
        """
        Pads each year to a full 365-day coverage
        Returns an xarray with dims (year, doy_period, ...), keeping order.
        """
        # --- Define expected day-of-year bins on a non leap year
        canonical_times = pd.date_range(
            f"2019-01-01", f"2019-12-31", freq=f"{temporal_resolution}D"
        )
        expected_times = canonical_times.map(lambda t: t.replace(year=selected_year))
        # Select this year's data
        data_year = data.sel(time=pd.to_datetime(data.time).year == selected_year)
        # Reindex to fill missing periods with NaN
        data_year = data_year.reindex(
            time=pd.to_datetime(expected_times),
            method="nearest",
            tolerance=np.timedelta64(temporal_resolution // 2, "D"),
        )
        return data_year


if __name__ == "__main__":
    dataset_path = "/Net/Groups/BGI/work_2/scratch/DeepExtremes/dx-minicubes/full/"
    temporal_resolution_veg = 16
    temporal_resolution_weather = 5
    training_years = (range(2016, 2023),)
    val_years = [2020]
    test_years = [2021]
    train_years = [
        year
        for year in training_years
        if year not in test_years and year not in val_years
    ]

    # 1. Create your original dataset (the slow one)
    train_dataset = ProcessTrainDataset(
        dataset_path=dataset_path,
        train_years=train_years,
        temporal_resolution_veg=temporal_resolution_veg,
        temporal_resolution_weather=temporal_resolution_weather,
        output_dir="preprocessed/train/samples/",
    )
