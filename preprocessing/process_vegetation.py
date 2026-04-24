import numpy as np
import xarray as xr
from pyproj import Transformer
import pandas as pd
from datetime import datetime, timedelta, date
import numpy as np
from abc import ABC
from scipy.signal import savgol_filter


class Sentinel2Preprocessing:
    def __init__(self, index="evi", temporal_resolution=16):
        self.index = index
        self.temporal_resolution = temporal_resolution
        self.noise_half_windows = [1]
        self.gapfill = False

    @staticmethod
    def circular_rolling_mean(arr, window_size=5, min_periods=1):
        """Apply a rolling mean to a numpy array with cyclic handling.
        Args:
            arr (numpy.ndarray): Input array for which the rolling mean is calculated.
            window_size (int): Number of elements in the rolling window. Default is 5.
            min_periods (int): Minimum valid (non-NaN) values required to compute the mean. Default is 1.
        Returns:
            numpy.ndarray: Array with rolling mean applied, handling NaN values gracefully.
        """
        n = arr.shape[0]  # Get the length of the input array
        result = np.full_like(arr, np.nan)  # Initialize result array with NaNs
        half_window = window_size // 2  # Determine half the window size for indexing
        for i in range(n):
            # Compute cyclic indices for the rolling window
            indices = [(i + j - half_window) % n for j in range(window_size)]
            # Extract valid (non-NaN) values from the array within the computed window
            valid_values = arr[indices][~np.isnan(arr[indices])]
            # Compute the mean only if enough valid values exist
            if len(valid_values) >= min_periods:
                result[i] = np.mean(valid_values)
        # Replace NaN values in the result where original array had NaNs
        return np.where(np.isnan(arr), result, arr)

    def generate_masked_vegetation_index(self, ds):
        # High-resolution computation
        ds = self._ensure_coordinates(ds)
        ds = self._get_random_vegetation_pixel_series(ds)
        if ds is None:
            return None
        evi = self._calculate_vegetation_index(ds)
        mask = self._compute_masks(ds, evi)
        masked_evi = evi * mask
        ds = xr.Dataset(
            data_vars={
                f"{self.index.lower()}": masked_evi,
            },
        )
        if self._has_excessive_nan(masked_evi):
            raise ValueError("Too many NaNs in masked EVI")

        ds = self.compute_max_per_period(ds, period_size=self.temporal_resolution)
        ds = NoiseRemovalHelper().cloudfree_timeseries(
            ds, noise_half_windows=self.noise_half_windows, gapfill=self.gapfill
        )

        msc = self.compute_msc(ds[f"{self.index.lower()}"])
        ds["msc"] = msc
        return ds.isel(location=0)  # remove location dim

    def _ensure_coordinates(self, ds):
        """Transforms UTM coordinates to latitude and longitude."""
        if "x20" in ds.dims:
            # Coarsen 10m bands to 20m resolution
            ds_10m = (
                ds[["B02", "B03", "B04", "B08"]]
                .coarsen(x=2, y=2, boundary="trim")
                .mean()
            )

            # Rename to match 20m grid
            ds_10m = ds_10m.rename({"x": "x20", "y": "y20"})

            # Select 20m bands
            ds_20m = ds[["B05", "B06", "B07", "B11", "B12", "B8A", "SCL"]]

            # Merge all bands
            ds = xr.merge([ds_10m, ds_20m])
            ds = ds.rename({"x20": "x", "y20": "y"})
        epsg = (
            ds.attrs.get("spatial_ref") or ds.attrs.get("EPSG") or ds.attrs.get("CRS")
        )

        # Transform UTM coordinates to latitude and longitude if EPSG is provided
        if epsg is not None:
            transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
            lon, lat = transformer.transform(ds.x.values, ds.y.values)
            ds = ds.assign_coords({"x": ("x", lon), "y": ("y", lat)})

        if "spatial_ref" in ds:
            ds = ds.drop_vars("spatial_ref")

        ds["time"] = ds["time"].dt.floor("D")
        ds = ds.sel(time=slice(date(2017, 1, 1), None))

        return ds.rename({"x": "longitude", "y": "latitude"})

    def _get_random_vegetation_pixel_series(self, ds):
        """
        Select a random vegetation pixel time series from the minicube based on SCL classification.
        Returns None if no eligible vegetation pixels exist.
        """
        lon_idx, lat_idx = self._choose_random_pixel(ds)
        if (lon_idx == None) and (lat_idx == None):
            return None
        selected_data = self._select_random_pixel(ds, lon_idx, lat_idx)

        # Stack spatial dimensions for simpler downstream processing
        return selected_data.stack(location=("longitude", "latitude"))

    def _find_eligible_vegetation_pixels(self, ds):
        """Return array of (lon, lat) indices with sufficient vegetation coverage."""
        vegetation_count = (ds.SCL == 4).sum(dim="time")
        years = np.unique(ds.time.dt.year)
        threshold = 0.25 * (366 / self.temporal_resolution) * len(years)
        mask = vegetation_count > threshold
        return np.argwhere(mask.values)

    def _choose_random_pixel(self, ds):
        """Randomly select one pixel index from the eligible vegetation pixels."""
        eligible_indices = self._find_eligible_vegetation_pixels(ds)
        if eligible_indices.size == 0:
            return (None, None)
        random_index = eligible_indices[np.random.choice(eligible_indices.shape[0])]
        return tuple(random_index)

    def _select_random_pixel(self, ds, lon_idx, lat_idx):
        """
        Select a single pixel from the dataset.
        """
        selected = ds.isel(longitude=lon_idx, latitude=lat_idx)
        return selected.expand_dims(
            longitude=[selected.longitude.values.item()],
            latitude=[selected.latitude.values.item()],
        )

    def _calculate_vegetation_index(self, ds):
        """Calculates the Vegetation Index."""
        if self.index in ("evi", "EVI"):
            return (2.5 * (ds.B8A - ds.B04)) / (
                ds.B8A + 6 * ds.B04 - 7.5 * ds.B02 + 1 + 10e-8
            )
        elif self.index in ("ndvi", "ndvi"):
            return (ds.B8A - ds.B04) / (ds.B8A + ds.B04 + 10e-8)
        else:
            raise ValueError(f"Unknown vegetation index: {self.index}")

    def _compute_masks(self, ds, evi=None):
        """
        Applies cloud and vegetation masks.
        If use_coarsen=True, masks are coarsened before being returned.
        """

        # Start with all valid
        mask = xr.ones_like(ds.B04)  # .stack(location=("latitude", "longitude"))

        # Apply vegetation mask if SCL exists
        if "SCL" in ds.data_vars:
            valid_scl = ds.SCL.isin([4, 5, 6, 7])
            valid_scl = valid_scl  # .stack(location=("latitude", "longitude"))
            mask = mask.where(valid_scl, np.nan)

            # drop timesteps where less than 90% valid pixels
            valid_ratio = valid_scl.sum(dim=["location"]) / valid_scl.count(
                dim=["location"]
            )
            invalid_time_steps = valid_ratio < 0.9
            mask = mask.where(~invalid_time_steps, np.nan)

        # Apply cloud mask if available
        if "cloudmask_en" in ds.data_vars:
            mask = mask.where(ds.cloudmask_en == 0, np.nan)
        return mask

    def _has_excessive_nan(self, data):
        """Checks if the masked data contains excessive NaN values."""
        nan_percentage = data.isnull().mean().values * 100
        return nan_percentage > 95

    def compute_msc(
        self,
        data: xr.DataArray,
        smoothing_window: int = 7,  # 9,
        poly_order: int = 2,
    ):

        # Step 1: Compute mean seasonal cycle
        mean_seasonal_cycle = data.groupby("time.dayofyear").mean("time", skipna=True)
        # Apply circular padding along the dayofyear axis before rolling
        # # edge case growing season during the change of year
        padded_values = np.pad(
            mean_seasonal_cycle.values,
            (
                (smoothing_window, smoothing_window),
                (0, 0),
            ),  # Pad along the dayofyear axis
            mode="wrap",  # Wrap-around to maintain continuity
        )

        padded_values = self.circular_rolling_mean(
            padded_values, window_size=2, min_periods=1
        )
        padded_values = self.circular_rolling_mean(
            padded_values, window_size=4, min_periods=1
        )

        padded_values = np.nan_to_num(padded_values, nan=0)

        # Step 5: Apply Savitzky-Golay smoothing
        smoothed_values = savgol_filter(
            padded_values, smoothing_window, poly_order, axis=0
        )
        mean_seasonal_cycle = mean_seasonal_cycle.copy(
            data=smoothed_values[smoothing_window:-smoothing_window]
        )
        # Step 6: Ensure all values are non-negative
        mean_seasonal_cycle = mean_seasonal_cycle.where(mean_seasonal_cycle > 0, 0)
        return mean_seasonal_cycle

    def compute_max_per_period(self, data, period_size=10):
        # Function to generate valid dates (time bins) for all years at once
        def get_time_periods(bin_size, years):
            periods = []
            for year in years:
                bins = np.arange(1, 367, bin_size)
                base_date = datetime(year, 1, 1).date()
                dates = [base_date + timedelta(days=int(d - 1)) for d in bins]
                periods.extend(dates)
            return periods

        # Define a function to map the timestamp to the corresponding period using searchsorted
        def get_period_from_timestamp(timestamp, periods):
            period_dates = pd.to_datetime(periods)  # Convert periods to datetime
            # Efficient way to find the period index using searchsorted
            period_index = np.searchsorted(period_dates, timestamp, side="right") - 1
            return period_index

        # Remove unrealistic values
        data = data.where((data >= 0) & (data <= 1), np.nan)

        # Prepare the periods list (one time-period list for all years in the dataset)
        years = pd.to_datetime(data.time).year.unique()
        periods = get_time_periods(period_size, years)
        # Map each timestamp to a period
        periods_assigned = [
            get_period_from_timestamp(t, periods) for t in pd.to_datetime(data.time)
        ]

        # Add period as a new dimension to the DataArray
        data.coords["period"] = ("time", periods_assigned)

        # Group by the 'period' and compute max per period
        data_grouped = data.groupby("period")

        # Compute max for each period
        max_per_period = data_grouped.max(dim="time")

        # Apply the transformation to convert periods back to midpoints in time
        start_period_times = [
            pd.to_datetime(periods[p]) for p in max_per_period.coords["period"].values
        ]

        # Update max_per_period with the transformed 'time' coordinates (midpoints)
        max_per_period.coords["time"] = ("period", start_period_times)
        max_per_period = max_per_period.swap_dims({"period": "time"}).drop_vars(
            "period"
        )
        max_per_period = max_per_period.set_index(location=["longitude", "latitude"])
        return max_per_period


class NoiseRemovalHelper(ABC):
    """Class for detecting and removing cloud-related noise in vegetation index time series."""

    def remove_cloud_noise(self, data, half_window=2, gapfill=True):
        """Detect and remove cloud noise using shifted mean values."""
        before_max = self._compute_shifted_max(data, half_window, direction=1)
        after_max = self._compute_shifted_max(data, half_window, direction=-1)

        is_cloud = (data + 0.05 < before_max) & (data + 0.05 < after_max)

        if gapfill:
            replacement_values = (before_max + after_max) / 2
            return xr.where(is_cloud, replacement_values, data)

        return xr.where(is_cloud, np.nan, data)

    def _compute_shifted_max(self, data, window_size, direction=1):
        """Helper function to compute shifted maximum values."""
        max_vals = xr.full_like(data, fill_value=np.nan)  # Initialize with NaNs

        for i in range(1, window_size + 1):
            shifted = data.shift(time=i * direction)

            # Compute element-wise max between max_vals and shifted
            max_vals = xr.where(
                np.isnan(max_vals),
                shifted,
                xr.where(np.isnan(shifted), max_vals, np.maximum(max_vals, shifted)),
            )

        return max_vals

    def cloudfree_timeseries(self, data, noise_half_windows=[1, 3], gapfill=False):
        data = data.where((data >= 0) & (data <= 1), np.nan)
        for half_window in noise_half_windows:
            data = self.remove_cloud_noise(
                data, half_window=half_window, gapfill=gapfill
            )
        return data
