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
        ds = ds.set_index(sample=("longitude", "latitude"))
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
    aligned_msc = msc.sel(dayofyear=data["time.dayofyear"])
    deseasonalized = data - aligned_msc
    return deseasonalized.reset_coords("dayofyear", drop=True)


def create_quantile_masks(data, thresholds):
    q = thresholds["quantile"]
    lower_q = q[q <= 0.5]
    upper_q = q[q > 0.5]

    lower = thresholds.sel(quantile=lower_q)
    upper = thresholds.sel(quantile=upper_q)

    masks = [
        data < lower[0],
        *[(data >= lower[i - 1]) & (data < lower[i]) for i in range(1, len(lower_q))],
        *[(data > upper[i - 1]) & (data <= upper[i]) for i in range(1, len(upper_q))],
        data > upper[-1],
    ]
    return masks


def apply_thresholds(data, thresholds):
    masks = create_quantile_masks(data, thresholds)
    quantiles = thresholds["quantile"]

    out = xr.full_like(data.astype(float), np.nan)

    for i, mask in enumerate(masks):
        out = xr.where(mask, quantiles[i], out)

    return out


def compute_percentiles(
    vegetation_index_path,
    msc_path,
    thresholds_path,
    variable="evi",
):
    """
    Evaluates ML model outputs using non-trivial percentile binning.
    """

    msc = load_file(msc_path, "msc")  # .sel(location=vegetation_index.location)
    thresholds = load_file(thresholds_path, "thresholds")  # .sel(
    #     location=vegetation_index.location
    # )
    vegetation_index = load_file(vegetation_index_path, variable).sel(
        location=msc.location
    )

    if vegetation_index is None or msc is None:
        raise ValueError("Failed to load input data")

    # 1. Deseasonalize
    data = deseasonalize(vegetation_index, msc)

    # 2. Apply thresholds
    result = apply_thresholds(
        data=data,
        thresholds=thresholds,
    )

    return result


result = compute_percentiles(
    vegetation_index_path="/Net/Groups/BGI/scratch/crobin/PythonProjects/ContrastiveEarthnetProject/outputs/predictions/TransformerBaseline/2026-04-14_10-40-03/2021/ueclsrgz.zarr",
    msc_path="/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2026-04-15_16:38:59_ML_aligned/EVI/msc.zarr",
    thresholds_path="/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2026-04-15_16:38:59_ML_aligned/EVI/thresholds.zarr",
    variable="predictions",
)
