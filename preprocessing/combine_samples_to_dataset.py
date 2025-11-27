import xarray as xr
from glob import glob
import os
import pandas as pd
from tqdm import tqdm  # <-- progress bar

input_dir = "datasets/train/samples"
output_store = "datasets/train.zarr"

paths = sorted(glob(os.path.join(input_dir, "*.zarr")))
datasets = []

# Wrap the iterator with tqdm
for i, p in enumerate(tqdm(paths, desc="Loading samples")):
    ds = xr.open_zarr(p)

    # Expand datasets along a new dimension "sample"
    ds = ds.expand_dims(sample=[i])

    # Make lon/lat proper sample coordinates
    lon = float(ds.longitude)
    lat = float(ds.latitude)

    ds = ds.assign_coords(longitude=("sample", [lon]), latitude=("sample", [lat]))
    datasets.append(ds)

# Concatenate along sample dimension
combined = xr.concat(datasets, dim="sample")

years = pd.DatetimeIndex(combined.time.values).year

val_mask = years == 2020
test_mask = years == 2021
train_mask = (years != 2020) & (years != 2021)

train = combined.sel(time=train_mask)
val = combined.sel(time=val_mask)
test = combined.sel(time=test_mask)

train.to_zarr("datasets/train.zarr", mode="w")
val.to_zarr("datasets/validation.zarr", mode="w")
test.to_zarr("datasets/test.zarr", mode="w")
