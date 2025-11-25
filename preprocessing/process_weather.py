import numpy as np
import xarray as xr

# Helper function for weather normalization
def weather_normalization(ds):
    """Normalize specified weather variables to [0, 1] range."""
    normalized_data = {}
    for var in ds:
        if var.startswith("tp_"):
            data_array = np.log1p(ds[var])  # equivalent to log(1 + tp)
        else:
            data_array = ds[var]
        min_val = data_array.min()
        max_val = data_array.max()
        normalized = (data_array - min_val) / (max_val - min_val + 1e-8)
        normalized_data[var] = normalized
    return xr.Dataset(normalized_data)
