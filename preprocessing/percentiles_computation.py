import xarray as xr
import numpy as np
from pathlib import Path
import cf_xarray as cfxr


def ensure_coordinates(ds):
    if "time_veg" in ds.dims:
        ds = ds.rename({"time_veg": "time"})
    if "location" in ds.dims:
        ds = cfxr.decode_compress_to_multi_index(ds, "location")
    if "sample" in ds.dims:
        ds = ds.set_index(sample=("latitude", "longitude"))
        ds = ds.rename(sample="location")

    return ds


def load_file(filepath, variable):
    try:
        ds = xr.open_zarr(Path(filepath)).astype(np.float32)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

    ds = ensure_coordinates(ds)
    return ds[variable]


def deseasonalize(data, msc):
    data = data.rename(time="dayofyear")
    deseasonalized = data - msc
    return deseasonalized


def create_quantile_masks(data, thresholds):
    q = thresholds["quantile"]
    lower_q = q[q <= 0.5]
    upper_q = q[q > 0.5]

    lower = thresholds.sel(quantile=lower_q)
    upper = thresholds.sel(quantile=upper_q)

    masks = [
        data < lower.isel(quantile=0),
        *[
            (data >= lower.isel(quantile=i - 1)) & (data < lower.isel(quantile=i))
            for i in range(1, len(lower_q))
        ],
        *[
            (data > upper.isel(quantile=i - 1)) & (data <= upper.isel(quantile=i))
            for i in range(1, len(upper_q))
        ],
        data > upper.isel(quantile=-1),
    ]
    return masks


def apply_thresholds(data, thresholds):
    masks = create_quantile_masks(data, thresholds)
    quantiles = thresholds["quantile"]

    extremes = xr.full_like(data.astype(float), np.nan)

    for i, mask in enumerate(masks):
        extremes = xr.where(mask, quantiles[i], extremes)

    return extremes


def compute_percentiles(
    vegetation_index_path,
    msc_path,
    thresholds_path,
    variable="evi",
):
    """
    Evaluates ML model outputs using non-trivial percentile binning.
    """
    vegetation_index = load_file(vegetation_index_path, variable)

    msc = load_file(msc_path, "msc").sel(location=vegetation_index.location)
    thresholds = load_file(thresholds_path, "thresholds").sel(
        location=vegetation_index.location
    )

    if vegetation_index is None or msc is None:
        raise ValueError("Failed to load input data")

    # 1. Deseasonalize
    data = deseasonalize(vegetation_index, msc)
    data = data.chunk({"location": 1000, "dayofyear": -1})
    thresholds = thresholds.chunk(
        {
            "location": 1000,
            "quantile": -1,
        }
    )
    # 2. Apply thresholds
    result = apply_thresholds(
        data=data,
        thresholds=thresholds,
    )

    return result


predicted_extremes = compute_percentiles(
    vegetation_index_path="/Net/Groups/BGI/scratch/crobin/PythonProjects/ContrastiveEarthnetProject/outputs/predictions/TransformerBaseline/2026-04-16_16-15-04/2019/5unz2x9j.zarr",
    msc_path="/Net/Groups/BGI/tscratch/crobin/ContrastiveEarthnetProject/datasets/percentiles/S2_evi_5d_eco_clusters/msc.zarr",
    thresholds_path="/Net/Groups/BGI/tscratch/crobin/ContrastiveEarthnetProject/datasets/percentiles/S2_evi_5d_eco_clusters/thresholds.zarr",
    variable="predictions",
)
print(predicted_extremes)

# F1 score
