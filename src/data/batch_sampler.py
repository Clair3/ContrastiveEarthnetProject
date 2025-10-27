from torch.utils.data.sampler import Sampler
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import pandas as pd


def _list_pairs(self, sample_paths, years):
    # Precompute valid (sample, year) pairs
    pairs = []
    for path in sample_paths:
        for year in years:
            pairs.append((path, year))
    return pairs


def fill_and_stack_year(self, data, temporal_resolution=16, years=None):
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
        data_year = data_year.assign_coords(doy=("time", np.arange(1, n_periods + 1)))

        data_year = data_year.swap_dims({"time": "doy"}).drop_vars("time")
        padded_years.append(data_year)

    # --- Concatenate along year dimension ---
    data_padded = xr.concat(padded_years, dim=pd.Index(years, name="year"))
    return data_padded


class BatchSampler(Sampler):
    r"""Yield a mini-batch of indices.

    Args:
        data: Dataset for building sampling logic.
        batch_size: Size of mini-batch.
    """

    def __init__(self, data, batch_size):
        # build data for sampling here
        self.batch_size = batch_size
        self.data = data

    def __iter__(self):
        # implement logic of sampling here
        batch = []
        for i, item in enumerate(self.data):
            batch.append(i)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

    def __len__(self):
        return len(self.data)


class BatchSampler(Sampler):
    def __init__(self, batch_to_indices):
        self.batch_to_indices = batch_to_indices
        self.batch_ids = list(batch_to_indices.keys())

    def __iter__(self):
        for batch_id in self.batch_ids:
            yield self.batch_to_indices[batch_id]

    def __len__(self):
        return len(self.batch_to_indices)
