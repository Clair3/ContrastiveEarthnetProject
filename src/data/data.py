from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import default_collate
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(
    filename="bad_paths.log",
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

from .preprocessing import (
    Sentinel2Preprocessing,
    weather_normalization,
    select_year,
)
from .batch_sampler import BatchSampler


class TrainContrastiveDataset(Dataset):
    """
    Contrastive dataset for Sentinel-2 + ERA5 time series.
    Each sample returns:
        - anchor: one (location, year)
        - positive: temporal crop of same (location, year)
        - negative: another year of same location
    """

    def __init__(self, dataset_path, train_years):
        sample_paths = [str(p) for p in Path(dataset_path).glob("*/*.zarr")]
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
            # "ssr_min",bad_paths_csv
            # "pev_max",
            # "ssr_max",
        ]
        self.bad_paths_csv = "unvalid_paths_deepextremes.csv"
        try:
            self.bad_paths = set(pd.read_csv("unvalid_paths.csv")["bad_path"].tolist())
        except FileNotFoundError:
            self.bad_paths = set()

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

        if torch.isnan(arr).all() or torch.isinf(arr).any():
            raise ValueError("Tensor contains only NaNs or infs")
        return arr.permute(1, 0)  # now [T, C]

    def __getitem__(self, idx: int) -> dict | None:
        sample_path, year = self.training_pairs[idx]

        try:
            ds = xr.open_zarr(sample_path)
        except Exception as e:
            logging.warning(f"Failed to open Zarr {sample_path}: {e}")
            self._record_bad_path(sample_path)
            return None

        try:
            vegetation = self._process_vegetation(ds, year)
            weather = self._process_weather(ds, year)
            vegetation_location = (
                vegetation.location.latitude.item(),
                vegetation.location.longitude.item(),
            )

            data = {
                "vegetation": self._to_tensor(vegetation),
                "weather": self._to_tensor(weather),
                "path": sample_path,
                "location": vegetation_location,
            }
            return data

        except Exception as e:
            logging.warning(f"Skipping {sample_path}: {e}")
            self._record_bad_path(sample_path)
            return None

    def _record_bad_path(self, path):
        """Append bad path to CSV if not already recorded"""
        if path not in self.bad_paths:
            self.bad_paths.add(path)
            pd.DataFrame({"bad_path": [path]}).to_csv(
                self.bad_paths_csv,
                mode="a",
                header=not Path(self.bad_paths_csv).exists(),
                index=False,
            )

    # --- helper functions ---
    def _process_vegetation(self, ds: xr.Dataset, year: int) -> xr.DataArray:
        veg_index = Sentinel2Preprocessing(
            temporal_resolution=self.temporal_resolution_veg
        ).generate_masked_vegetation_index(ds)
        if veg_index is None:
            raise ValueError("Vegetation index computation failed")
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


class ValContrastiveDataset(TrainContrastiveDataset):

    def __init__(self, dataset_path, train_years, heldout_years):
        sample_paths = [str(p) for p in Path(dataset_path).glob("*/*.zarr")]
        self.train_years = train_years
        self.val_years = heldout_years
        self.training_pairs = self._list_pairs(sample_paths, heldout_years)
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
            # "ssr_min",bad_paths_csv
            # "pev_max",
            # "ssr_max",
        ]
        self.bad_paths_csv = "unvalid_paths_deepextremes.csv"
        try:
            self.bad_paths = set(pd.read_csv("unvalid_paths.csv")["bad_path"].tolist())
        except FileNotFoundError:
            self.bad_paths = set()

    def _process_weather(self, ds: xr.Dataset, year: int) -> xr.DataArray:
        weather = ds[self.era5_variables]
        # Keep only training years
        weather_train = weather.sel(time=weather["time.year"].isin(self.train_years))
        weather_train = weather_normalization(weather_train)
        weather = weather.sel(time=weather["time.year"].isin(year))
        return select_year(
            weather,
            temporal_resolution=self.temporal_resolution_weather,
            selected_year=year,
        )

    def __getitem__(self, idx: int) -> dict | None:
        sample_path, year = self.training_pairs[idx]

        try:
            ds = xr.open_zarr(sample_path)
        except Exception as e:
            logging.warning(f"Failed to open Zarr {sample_path}: {e}")
            self._record_bad_path(sample_path)
            return None

        try:
            vegetation = self._process_vegetation(ds, year)
            weather = self._process_weather(ds, year)
            vegetation_location = (
                vegetation.location.latitude.item(),
                vegetation.location.longitude.item(),
            )
            print(vegetation.evi.values)

            data = {
                "vegetation": self._to_tensor(vegetation),
                "weather": self._to_tensor(weather),
                "path": sample_path,
                "location": vegetation_location,
            }
            print(data["vegetation"])
            print(data["weather"])
            return data

        except Exception as e:
            logging.warning(f"Skipping {sample_path}: {e}")
            self._record_bad_path(sample_path)
            return None


class ContrastiveDataModule(LightningDataModule):
    """
    DataModule for temporal contrastive learning with a held-out test year.
    """

    def __init__(
        self,
        dataset_path,
        batch_size=8,
        num_workers=4,
        training_years=range(2017, 2023),
        val_years=[2021],
        test_years=[2020],
    ):
        """
        test_year: index of the held-out year in Zarr array
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_path = dataset_path
        self.val_years = val_years if isinstance(val_years, list) else [test_years]
        self.test_years = test_years if isinstance(test_years, list) else [test_years]
        self.train_years = [
            year
            for year in training_years
            if year not in self.test_years and year not in self.val_years
        ]

    def setup(self, stage=None):

        self.train_dataset = TrainContrastiveDataset(
            dataset_path=self.dataset_path,
            train_years=self.train_years,
        )
        self.val_dataset = ValContrastiveDataset(
            dataset_path=self.dataset_path,
            train_years=self.train_years,
            heldout_years=self.val_years,
        )

        self.test_dataset = ValContrastiveDataset(
            dataset_path=self.dataset_path,
            train_years=self.train_years,
            heldout_years=self.test_years,
        )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_sampler=BatchSampler(
                dataset=self.train_dataset,
                shuffle=True,
            ),
            collate_fn=safe_collate,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return loader  # NonNoneDataLoader(loader)

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_sampler=BatchSampler(
                dataset=self.val_dataset,
                shuffle=True,
            ),
            num_workers=self.num_workers,
            collate_fn=safe_collate,
            pin_memory=True,
        )
        return loader  # NonNoneDataLoader(loader)

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_sampler=BatchSampler(
                dataset=self.test_dataset,
                shuffle=True,
            ),
            num_workers=self.num_workers,
            collate_fn=safe_collate,
            pin_memory=True,
        )
        return NonNoneDataLoader(loader)


class NonNoneDataLoader:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for batch in self.loader:
            if batch is not None:
                yield batch

    def __len__(self):
        return len(self.loader)


def safe_collate(batch):
    # Remove any None entries
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # whole batch invalid TÓDO improve
    return default_collate(batch)
