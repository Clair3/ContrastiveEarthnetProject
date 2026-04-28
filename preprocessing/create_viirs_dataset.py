import re
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from pathlib import Path
import os
import subprocess


def extract_locations_from_paths(path_file):
    # --- 1. Extract lon/lat from file ---
    pattern = re.compile(r"mc_(-?\d+\.\d+)_(-?\d+\.\d+)")

    lons, lats = [], []

    with open(path_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                lon, lat = map(float, match.groups())
                lons.append(lon)
                lats.append(lat)

    # keep only first n points
    lons = np.array(lons)
    lats = np.array(lats)
    return lons, lats


def subset_era(ds, lons, lats):
    """
    Extract a subset of an xarray dataset at locations defined in a path file.

    Parameters
    ----------
    ds : xarray.Dataset
        The large ERA dataset
    path_file : str
        File containing paths with embedded lon/lat
    n_points : int
        Number of locations to use (default=10 for testing)

    Returns
    -------
    subset : xarray.Dataset
        Dataset subset at selected points
    """

    print(f"Using {len(lons)} locations")

    # --- 2. Build ERA grid (small, safe to load) ---
    lat_grid = (
        ds["latchunk"].values[:, None] + ds["sublat_era"].values[None, :]
    )  # (90, 9)

    lon_grid = (
        ds["lonchunk"].values[:, None] + ds["sublon_era"].values[None, :]
    )  # (180, 5)

    # flatten
    lat_flat = lat_grid.reshape(-1)  # 810
    lon_flat = lon_grid.reshape(-1)  # 1620

    # full mesh
    lat_full = np.repeat(lat_flat[:, None], len(lon_flat), axis=1).ravel()
    lon_full = np.repeat(lon_flat[None, :], len(lat_flat), axis=0).ravel()

    # --- 3. Nearest neighbor search ---
    tree = cKDTree(np.column_stack([lat_full, lon_full]))
    _, idx = tree.query(np.column_stack([lats, lons]))

    # --- 4. Convert to indices ---
    n_lon = len(lon_flat)

    lat_idx = idx // n_lon
    lon_idx = idx % n_lon

    sublon_era_res = ds["sublon_era"].size
    sublat_era_res = ds["sublon_era"].size

    latchunk_idx = lat_idx // sublat_era_res
    sublat_idx = lat_idx % sublat_era_res

    lonchunk_idx = lon_idx // sublon_era_res
    sublon_idx = lon_idx % sublon_era_res

    # --- 5. Extract subset ---
    subset = ds.isel(
        latchunk=xr.DataArray(latchunk_idx, dims="locations"),
        sublat_era=xr.DataArray(sublat_idx, dims="locations"),
        lonchunk=xr.DataArray(lonchunk_idx, dims="locations"),
        sublon_era=xr.DataArray(sublon_idx, dims="locations"),
    )

    # --- 6. Attach actual coordinates ---
    subset = subset.assign_coords(
        lat=("locations", lats),
        lon=("locations", lons),
    )

    return subset


def subset_viirs(ds, variable, vi_name, lons, lats):
    """
    Extract VIIRS time series at all locations from samples_paths.txt.

    Input dataset must have dims:
    (time, latchunk, latstep_modis, lonchunk, lonstep_modis)

    Returns:
        Dataset with dims (time, locations)
    """

    print(f"Loaded {len(lats)} locations")

    # --- 2. Build full VIIRS grid ---
    lat_chunk = ds["latchunk"].values
    lat_step = ds["latstep_modis"].values
    lon_chunk = ds["lonchunk"].values
    lon_step = ds["lonstep_modis"].values

    # build 4D mesh
    lat4d = lat_chunk[:, None, None, None] + lat_step[None, :, None, None]

    lon4d = lon_chunk[None, None, :, None] + lon_step[None, None, None, :]

    # broadcast to full grid
    lat_full = np.broadcast_to(lat4d, (90, 40, 180, 40))
    lon_full = np.broadcast_to(lon4d, (90, 40, 180, 40))

    # flatten
    lat_flat = lat_full.ravel()
    lon_flat = lon_full.ravel()

    # --- 3. KDTree nearest neighbor ---
    tree = cKDTree(np.column_stack([lat_flat, lon_flat]))
    _, idx = tree.query(np.column_stack([lats, lons]))

    # --- 4. Convert flat index → VIIRS indices ---
    n_lon_total = len(ds["lonchunk"]) * len(ds["lonstep_modis"])

    lat_idx = idx // n_lon_total
    lon_idx = idx % n_lon_total

    latstep_res = len(ds["latstep_modis"])
    lonstep_res = len(ds["lonstep_modis"])

    latchunk_idx = lat_idx // latstep_res
    latstep_idx = lat_idx % latstep_res

    lonchunk_idx = lon_idx // lonstep_res
    lonstep_idx = lon_idx % lonstep_res

    # --- 5. Extract VIIRS variable ---
    da = ds[variable].isel(
        latchunk=xr.DataArray(latchunk_idx, dims="locations"),
        latstep_modis=xr.DataArray(latstep_idx, dims="locations"),
        lonchunk=xr.DataArray(lonchunk_idx, dims="locations"),
        lonstep_modis=xr.DataArray(lonstep_idx, dims="locations"),
    )

    # --- 6. Convert to Dataset + attach coords ---
    subset = da.to_dataset(name=vi_name)

    subset = subset.assign_coords(
        lat=("locations", lats),
        lon=("locations", lons),
    )

    return subset


def normalize_weather_variables(
    dataset: xr.Dataset, weather_vars: list[str]
) -> xr.Dataset:
    """
    Normalize weather variables using z-score normalization.

    Precipitation variables (starting with 'tp_') are log1p transformed before normalization.
    All other variables use standard z-score: (x - mean) / std

    Args:
        Dataset
        weather_vars: List of weather variable names to normalize

    Returns:
        Dataset with normalized weather variables
    """
    print("\nNormalizing weather variables:")

    # Create copies to avoid modifying originals
    dataset_out = dataset.copy()

    for var in weather_vars:
        # Apply log1p transform to precipitation
        if var.startswith("tp_"):
            dataset_array = np.log1p(dataset[var])
            transform_type = "log1p + z-score"
        else:
            dataset_array = dataset[var]
            transform_type = "z-score"

        # Compute mean and std
        mean_dataset = float(dataset_array.mean())
        std_dataset = float(dataset_array.std())

        # Apply z-score normalization: (x - mean) / std
        dataset_out[var] = (dataset_array - mean_dataset) / (std_dataset + 1e-8)

        print(
            f"  {var:12s} ({transform_type:20s}): mean={mean_dataset:10.4f}, std={std_dataset:10.4f}"
        )

    return dataset_out


def main():
    """Main preprocessing pipeline."""

    # Configuration
    OUTPUT_DIR = "datasets/VIIRS_evi_daily_10.zarr"
    SCRATCH_DIR = Path(
        "/Net/Groups/BGI/tscratch/crobin/ContrastiveEarthnetProject/datasets"
    )

    SAMPLES_PATHS = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ContrastiveEarthnetProject/preprocessing/sample_paths_10.txt"

    VIIRS_PATH = "/Net/Groups/BGI/work_4/scratch/fluxcom/upscaling_inputs/VIIRS_gapfilled.zarr/EVIgapfilled_002_QCfix.zarr"  # VIIRS_gapfilled.zarr"
    VIIRS_VARIABLE = "EVIgapfilled_QCfix"

    ERA5_PATH = (
        "/Net/Groups/BGI/work_4/scratch/fluxcom/upscaling_inputs/ERA5_daily.zarr"
    )

    VEG_VARS = ["evi"]
    WEATHER_VARS = ["P", "SW_IN", "TA", "TA_max", "TA_min", "VPD", "VPD_min", "VPD_max"]

    print("\n=== Loading samples ===")
    lons, lats = extract_locations_from_paths(path_file=SAMPLES_PATHS)

    viirs_ds = xr.open_zarr(VIIRS_PATH)
    viirs = subset_viirs(
        viirs_ds, variable=VIIRS_VARIABLE, vi_name=VEG_VARS[0], lons=lons, lats=lats
    )

    era_ds = xr.open_zarr(ERA5_PATH)  # .mean("number")
    era = subset_era(era_ds, lons, lats)

    viirs, era = xr.align(viirs, era, join="inner")
    ds = xr.merge([viirs, era], compat="override")

    ds = ds.sel(time=slice("2012-01-01T12:00:00", "2025-12-31T12:00:00"))
    ds = ds.sel(time=~((ds["time"].dt.month == 2) & (ds["time"].dt.day == 29)))

    # ds["evi"] = ds.evi.groupby("time.year").map(
        # lambda x: x.resample(time="5D", origin="start").mean()
    # )
# 
    # # MSC (Jan 1 anchored bins)
    # ds["msc"] = (
        # ds.msc.groupby_bins("dayofyear", bins=np.arange(1, 367, 5), right=False)
        # .mean()
        # .rename({"dayofyear_bins": "dayofyear_5d"})
    # )

    ds = ds.rename({"locations": "sample"})
    ds = ds.rename({"lat": "latitude"})
    ds = ds.rename({"lon": "longitude"})

    ds = ds.transpose("sample", "time")

    ds_veg = ds[VEG_VARS].rename({"time": "time_veg"})
    ds_weather = ds[WEATHER_VARS].rename({"time": "time_weather"})

    print("\n=== Normalizing weather variables ===")
    ds_weather = normalize_weather_variables(ds_weather, WEATHER_VARS)

    doy = ds_veg["time_veg"].dt.dayofyear
    msc = ds_veg["evi"].groupby(doy).mean("time_veg")

    dataset = xr.merge([ds_veg, ds_weather, msc.rename("msc")])

    print("\n=== Chunking dataset ===")
    dataset = dataset.chunk({"sample": 1000, "time_weather": 365, "time_veg": 365})

    print("\n=== Saving datasets and copying to tscratch ===")
    scratch_path = SCRATCH_DIR / OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(scratch_path, exist_ok=True)

    if dataset is not None:
        dataset_path = Path(OUTPUT_DIR)
        dataset.to_zarr(dataset_path, mode="w", consolidated=True)
        subprocess.run(
            [
                "rsync",
                "-a",
                str(dataset_path),
                str(SCRATCH_DIR),
            ],
            check=True,
        )

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
