import xarray as xr
from glob import glob
import os
import pandas as pd
from tqdm import tqdm  # <-- progress bar

input_dir = "datasets/samples"

paths = sorted(glob(os.path.join(input_dir, "*.zarr")))
print("Found {} sample files.".format(len(paths)))
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
    if ds.evi.notnull().any().compute().item():
        # (ds.evi.notnull().groupby("time_veg.year").any(dim="time_veg").all().compute().item())
        datasets.append(ds)

# Concatenate along sample dimension
combined = xr.concat(datasets, dim="sample")

years_veg = pd.DatetimeIndex(combined.time_veg.values).year

val_mask_veg = years_veg == 2020
test_mask_veg = years_veg == 2021
train_mask_veg = (years_veg != 2020) & (years_veg != 2021)

years_weather = pd.DatetimeIndex(combined.time_weather.values).year

val_mask_weather = years_weather == 2020
test_mask_weather = years_weather == 2021
train_mask_weather = (years_weather != 2020) & (years_weather != 2021)


train = combined.sel(time_veg=train_mask_veg, time_weather=train_mask_weather)
val = combined.sel(time_veg=val_mask_veg, time_weather=val_mask_weather)
test = combined.sel(time_veg=test_mask_veg, time_weather=test_mask_weather)

train.to_zarr("datasets/train_10.zarr", mode="w")
val.to_zarr("datasets/validation_10.zarr", mode="w")
test.to_zarr("datasets/test_10.zarr", mode="w")
