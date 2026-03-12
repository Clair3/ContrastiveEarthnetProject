import os
import subprocess
from glob import glob
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def load_and_prepare_samples(input_dir: str) -> xr.Dataset:
    """Load all sample zarr files and return list of (location, year) samples."""
    paths = sorted(glob(os.path.join(input_dir, "*.zarr")))
    print(f"Found {len(paths)} sample files.")
    if len(paths) == 0:
        raise ValueError(f"No zarr files found in {input_dir}")

    samples = []

    for path in tqdm(paths, desc="Loading samples"):
        sample = xr.open_zarr(path)
        if not sample.evi.notnull().any().compute().item():
            continue
        samples.append(sample)
    dataset = xr.concat(samples, dim="sample")
    print(f"Created dataset with {len(dataset.sample)} samples")
    return dataset


def create_split(dataset: xr.Dataset, years: list[int]) -> xr.Dataset:
    """
    Create a dataset split containing only data from specified years.

    Args:
        dataset: Input dataset with time_veg and time_weather dimensions
        years: List of years to include in this split

    Returns:
        Dataset filtered to only include samples from the specified years
    """
    return dataset.sel(
        time_veg=pd.DatetimeIndex(dataset.time_veg.values).year.isin(years),
        time_weather=pd.DatetimeIndex(dataset.time_weather.values).year.isin(years),
    )


def split_dataset(
    dataset: xr.Dataset, val_year: int = 2020, test_year: int = 2021
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Split dataset into train/val/test by year.
    Each (location, year) combination becomes a separate sample.

    Returns datasets with dimensions: (sample, time_veg, ...) and (sample, time_weather, ...)
    where sample represents unique (location, year) pairs.
    """

    # Get unique years from vegetation time series
    years_veg = pd.DatetimeIndex(dataset.time_veg.values).year.unique()
    # Determine which years go to which split

    train_years = [y for y in years_veg if y != val_year and y != test_year]
    val_years = [val_year] if val_year in years_veg else []
    test_years = [test_year] if test_year in years_veg else []

    print(f"\nSplitting:")
    print(f"  Train years: {train_years}")
    print(f"  Val years:   {val_years}")
    print(f"  Test years:  {test_years}")

    # Create splits
    train = create_split(dataset, train_years) if train_years else None
    val = create_split(dataset, val_years) if val_years else None
    test = create_split(dataset, test_years) if test_years else None
    return train, val, test


def normalize_weather_variables(
    train: xr.Dataset, val: xr.Dataset, test: xr.Dataset, weather_vars: list[str]
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Normalize weather variables using z-score normalization with training statistics.

    Precipitation variables (starting with 'tp_') are log1p transformed before normalization.
    All other variables use standard z-score: (x - mean) / std

    Args:
        train: Training dataset
        val: Validation dataset
        test: Test dataset
        weather_vars: List of weather variable names to normalize

    Returns:
        Tuple of (train, val, test) with normalized weather variables
    """
    print("\nNormalizing weather variables:")

    # Create copies to avoid modifying originals
    train_out = train.copy()
    val_out = val.copy() if val is not None else None
    test_out = test.copy() if test is not None else None

    for var in weather_vars:
        # Apply log1p transform to precipitation for all splits
        if var.startswith("tp_"):
            train_array = np.log1p(train[var])
            val_array = np.log1p(val[var]) if val is not None else None
            test_array = np.log1p(test[var]) if test is not None else None
            transform_type = "log1p + z-score"
        else:
            train_array = train[var]
            val_array = val[var] if val is not None else None
            test_array = test[var] if test is not None else None
            transform_type = "z-score"

        # Compute mean and std from training data only
        mean_val = float(train_array.mean())
        std_val = float(train_array.std())

        # Apply z-score normalization: (x - mean) / std
        train_out[var] = (train_array - mean_val) / (std_val + 1e-8)
        if val is not None:
            val_out[var] = (val_array - mean_val) / (std_val + 1e-8)
        if test is not None:
            test_out[var] = (test_array - mean_val) / (std_val + 1e-8)

        print(
            f"  {var:12s} ({transform_type:20s}): mean={mean_val:10.4f}, std={std_val:10.4f}"
        )

    return train_out, val_out, test_out


def main():
    """Main preprocessing pipeline."""

    # Configuration
    INPUT_DIR = "datasets/samples"
    OUTPUT_DIR = "datasets"
    SCRATCH_DIR = Path("/Net/Groups/BGI/tscratch/crobin/ContrastiveEarthnetProject")

    WEATHER_VARS = [
        "t2m_mean",
        "t2m_min",
        "t2m_max",
        "tp_mean",
        "tp_min",
        "tp_max",
        "pev_mean",
        "pev_min",
        "pev_max",
        "ssr_mean",
        "ssr_min",
        "ssr_max",
    ]

    # Load and combine samples
    print("\n=== Loading samples ===")
    combined = load_and_prepare_samples(INPUT_DIR)

    # Split by year (this expands locations into location-year samples)
    print("\n=== Splitting by year ===")
    train, val, test = split_dataset(combined)

    # Normalize weather variables
    print("\n=== Normalizing weather variables ===")
    train, val, test = normalize_weather_variables(train, val, test, WEATHER_VARS)

    print("\n=== Chunking datasets ===")
    train = train.chunk({"sample": 1, "time_weather": -1, "time_veg": -1})

    val = val.chunk({"sample": 1, "time_weather": -1, "time_veg": -1})

    test = test.chunk({"sample": 1, "time_weather": -1, "time_veg": -1})

    # Save splits
    print("\n=== Saving datasets and copying to tscratch ===")
    scratch_path = SCRATCH_DIR / Path(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(scratch_path, exist_ok=True)

    if train is not None:
        train_path = os.path.join(OUTPUT_DIR, "train.zarr")
        train.to_zarr(train_path, mode="w", consolidated=True)
        subprocess.run(
            [
                "rsync",
                "-a",
                str(train_path),
                str(os.path.join(SCRATCH_DIR, train_path)),
            ],
            check=True,
        )

    if val is not None:
        val_path = os.path.join(OUTPUT_DIR, "validation.zarr")
        val.to_zarr(val_path, mode="w", consolidated=True)
        subprocess.run(
            [
                "rsync",
                "-a",
                str(val_path),
                str(os.path.join(SCRATCH_DIR, val_path)),
            ],
            check=True,
        )
    if test is not None:
        test_path = os.path.join(OUTPUT_DIR, "test.zarr")
        test.to_zarr(test_path, mode="w", consolidated=True)
        subprocess.run(
            [
                "rsync",
                "-a",
                str(test_path),
                str(os.path.join(SCRATCH_DIR, test_path)),
            ],
            check=True,
        )

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
