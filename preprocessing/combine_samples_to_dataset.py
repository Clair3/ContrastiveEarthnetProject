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
    INPUT_DIR = "datasets/samples_aligned/"
    OUTPUT_DIR = "datasets/aligned.zarr"
    SCRATCH_DIR = Path("/Net/Groups/BGI/tscratch/crobin/ContrastiveEarthnetProject")

    WEATHER_VARS = [
        # "t2m_mean",
        "t2m_min",
        "t2m_max",
        # "tp_mean",
        # "tp_min",
        "tp_max",
        "pev_mean",
        # "pev_min",
        "pev_max",
        "ssr_mean",
        # "ssr_min",
        "ssr_max",
    ]

    # Load and combine samples
    print("\n=== Loading samples ===")
    combined = load_and_prepare_samples(INPUT_DIR)

    # Normalize weather variables
    print("\n=== Normalizing weather variables ===")
    dataset = normalize_weather_variables(combined, WEATHER_VARS)

    print("\n=== Chunking dataset ===")
    dataset = dataset.chunk({"sample": 1000, "time_weather": -1, "time_veg": -1})

    # Save splits
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
                str(os.path.join(SCRATCH_DIR, OUTPUT_DIR)),
            ],
            check=True,
        )

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
