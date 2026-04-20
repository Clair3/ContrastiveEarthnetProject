#!/usr/bin/env python3
"""
Download and harmonize C3S seasonal forecasts for the VIIRS sample locations.

This script is intentionally standalone so it can be run in a data-preparation
environment without touching the current training code. It supports:

1. Retrospective forecasts ("hindcasts") for training.
2. Real-time forecasts for inference.
3. Harmonization from CDS/SEAS5 variable names to the project's VIIRS weather
   names: P, SW_IN, TA, TA_max, TA_min, VPD, VPD_min, VPD_max.

Important caveats
-----------------
- ECMWF SEAS5 seasonal forecasts cover about 7 months for the regular monthly
  runs, not a full next calendar year from a single initialization.
- For training, hindcasts are the right target, not a literal "previous year"
  forecast file. If you need Jan-Dec coverage, use rolling monthly
  initializations and stitch them causally.
- VPD variables are approximate because the daily CDS product exposes
  dewpoint temperature but not daily dewpoint min/max. We therefore combine
  daily dewpoint with daily min/max temperature.
"""

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_CDS_TO_PROJECT = {
    "total_precipitation": "P",
    "surface_solar_radiation_downwards": "SW_IN",
    "2m_temperature": "TA",
    "maximum_2m_temperature_in_the_last_24_hours": "TA_max",
    "minimum_2m_temperature_in_the_last_24_hours": "TA_min",
    "2m_dewpoint_temperature": "TD",
}

DEFAULT_VARIABLES = list(DEFAULT_CDS_TO_PROJECT)


def _lazy_imports():
    try:
        import cdsapi  # type: ignore
        import pandas as pd  # type: ignore
        import xarray as xr  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency while starting the seasonal forecast downloader. "
            "Please install at least cdsapi, pandas, xarray, cfgrib, eccodes, zarr. "
            f"Original error: {exc}"
        ) from exc
    return cdsapi, pd, xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and harmonize seasonal forecasts for VIIRS locations."
    )
    parser.add_argument(
        "--sample-paths",
        type=Path,
        default=Path("preprocessing/sample_paths.txt"),
        help="Text file containing the VIIRS minicube paths with embedded lon/lat.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/seasonal_forecast_viirs"),
        help="Directory where compact point-level Zarr stores will be written.",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="Optional directory for raw downloads and intermediate files.",
    )
    parser.add_argument(
        "--mode",
        choices=["hindcast", "forecast"],
        default="hindcast",
        help="Use hindcasts for training and forecasts for inference.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        required=True,
        help="Initialization years to request, e.g. 1993 1994 ... 2016.",
    )
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        default=[1],
        help="Initialization months to request, e.g. 1 2 3 ... 12.",
    )
    parser.add_argument(
        "--day",
        type=int,
        default=1,
        help="Initialization day. SEAS5 standard monthly runs use day 1.",
    )
    parser.add_argument(
        "--originating-centre",
        default="ecmwf",
        help="CDS originating centre. Default: ecmwf.",
    )
    parser.add_argument(
        "--system",
        default="51",
        help="Seasonal system identifier. ECMWF SEAS5 currently uses system 51 in CDS.",
    )
    parser.add_argument(
        "--lead-days",
        type=int,
        default=215,
        help="Maximum daily lead to request. About 215 days covers the regular 7-month runs.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=DEFAULT_VARIABLES,
        help="CDS variables to download.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep downloaded GRIB files instead of deleting them after processing.",
    )
    parser.add_argument(
        "--normalize-from",
        type=Path,
        default=None,
        help=(
            "Optional raw reference Zarr with the same project variable names. "
            "If provided, z-score normalization is applied using that reference."
        ),
    )
    return parser.parse_args()


def extract_locations_from_paths(path_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    pattern = re.compile(r"mc_(-?\d+\.\d+)_(-?\d+\.\d+)")
    lons, lats = [], []
    with path_file.open("r") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                lon, lat = map(float, match.groups())
                lons.append(lon)
                lats.append(lat)
    if not lons:
        raise ValueError(f"No coordinates found in {path_file}")
    return np.asarray(lons), np.asarray(lats)


def build_area(lons: np.ndarray, lats: np.ndarray, pad: float = 1.0) -> List[float]:
    north = min(90.0, math.ceil(float(lats.max()) + pad))
    south = max(-90.0, math.floor(float(lats.min()) - pad))
    west = max(-180.0, math.floor(float(lons.min()) - pad))
    east = min(180.0, math.ceil(float(lons.max()) + pad))
    return [north, west, south, east]


def daily_leadtime_hours(max_days: int) -> List[str]:
    return [str(day * 24) for day in range(1, max_days + 1)]


def request_payload(
    *,
    variable: str,
    years: List[int],
    month: int,
    day: int,
    args: argparse.Namespace,
    area: List[float],
) -> dict:
    payload = {
        "originating_centre": args.originating_centre,
        "system": args.system,
        "variable": variable,
        "year": [f"{year:04d}" for year in years],
        "month": f"{month:02d}",
        "day": f"{day:02d}",
        "leadtime_hour": daily_leadtime_hours(args.lead_days),
        "area": area,
    }
    # The CDS client interface has changed over time. Try the modern key first,
    # then fall back to the legacy one if retrieval fails.
    payload["data_format"] = "grib"
    return payload


def retrieve_variable(
    client,
    *,
    variable: str,
    years: List[int],
    month: int,
    day: int,
    args: argparse.Namespace,
    area: List[float],
    raw_dir: Path,
) -> Path:
    target = raw_dir / f"{args.mode}_{variable}_m{month:02d}_{years[0]}_{years[-1]}.grib"
    if target.exists():
        return target

    payload = request_payload(
        variable=variable,
        years=years,
        month=month,
        day=day,
        args=args,
        area=area,
    )

    dataset_name = "seasonal-original-single-levels"
    try:
        client.retrieve(dataset_name, payload, str(target))
    except Exception:
        payload = dict(payload)
        payload.pop("data_format", None)
        payload["format"] = "grib"
        client.retrieve(dataset_name, payload, str(target))
    return target


def open_single_variable_dataset(grib_path: Path, xr):
    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    )
    data_vars = list(ds.data_vars)
    if len(data_vars) != 1:
        raise ValueError(
            f"Expected one data variable in {grib_path}, found {data_vars}"
        )
    return ds


def _maybe_rename_lonlat(ds):
    rename_map = {}
    if "latitude" not in ds.coords and "lat" in ds.coords:
        rename_map["lat"] = "latitude"
    if "longitude" not in ds.coords and "lon" in ds.coords:
        rename_map["lon"] = "longitude"
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


def _to_minus180_180(lon_values: np.ndarray) -> np.ndarray:
    return ((lon_values + 180.0) % 360.0) - 180.0


def subset_to_samples(ds, lons: np.ndarray, lats: np.ndarray, xr):
    ds = _maybe_rename_lonlat(ds)

    if "longitude" not in ds.coords or "latitude" not in ds.coords:
        raise ValueError("Dataset is missing latitude/longitude coordinates.")

    ds = ds.sortby("latitude")
    if float(ds.longitude.max()) > 180.0:
        ds = ds.assign_coords(longitude=_to_minus180_180(ds.longitude.values)).sortby(
            "longitude"
        )

    selection = ds.sel(
        latitude=xr.DataArray(lats, dims="sample"),
        longitude=xr.DataArray(lons, dims="sample"),
        method="nearest",
    )
    selection = selection.assign_coords(
        latitude=("sample", lats),
        longitude=("sample", lons),
    )
    return selection


def saturation_vapor_pressure_kpa(temp_c):
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def harmonize_units_and_names(ds, xr):
    data_var_names = list(ds.data_vars)
    rename_map = {}
    for source_name in data_var_names:
        target_name = DEFAULT_CDS_TO_PROJECT.get(source_name)
        if target_name is not None:
            rename_map[source_name] = target_name
    if rename_map:
        ds = ds.rename(rename_map)

    # Temperature: Kelvin -> Celsius
    for temp_name in ["TA", "TA_max", "TA_min", "TD"]:
        if temp_name in ds:
            ds[temp_name] = ds[temp_name] - 273.15
            ds[temp_name].attrs["units"] = "degC"

    # Precipitation: m/day -> mm/day
    if "P" in ds:
        ds["P"] = ds["P"] * 1000.0
        ds["P"].attrs["units"] = "mm day-1"

    # Radiation: J m-2 over day -> W m-2 daily mean
    if "SW_IN" in ds:
        ds["SW_IN"] = ds["SW_IN"] / 86400.0
        ds["SW_IN"].attrs["units"] = "W m-2"

    # Approximate daily VPD diagnostics from temperature and dewpoint.
    if "TD" in ds:
        ea = saturation_vapor_pressure_kpa(ds["TD"])

        if "TA" in ds:
            ds["VPD"] = xr.apply_ufunc(
                saturation_vapor_pressure_kpa,
                ds["TA"],
            ) - ea
            ds["VPD"].attrs["units"] = "kPa"

        if "TA_min" in ds:
            ds["VPD_min"] = xr.apply_ufunc(
                saturation_vapor_pressure_kpa,
                ds["TA_min"],
            ) - ea
            ds["VPD_min"].attrs["units"] = "kPa"

        if "TA_max" in ds:
            ds["VPD_max"] = xr.apply_ufunc(
                saturation_vapor_pressure_kpa,
                ds["TA_max"],
            ) - ea
            ds["VPD_max"].attrs["units"] = "kPa"

        ds = ds.drop_vars("TD")

    return ds


def compute_reference_stats(reference_path: Path, xr) -> Dict[str, Tuple[float, float]]:
    reference = xr.open_zarr(reference_path)
    stats = {}
    for var in ["P", "SW_IN", "TA", "TA_max", "TA_min", "VPD", "VPD_min", "VPD_max"]:
        if var in reference:
            stats[var] = (float(reference[var].mean()), float(reference[var].std()))
    return stats


def normalize_with_reference(ds, stats):
    for var, (mean_value, std_value) in stats.items():
        if var in ds:
            ds[var] = (ds[var] - mean_value) / (std_value + 1e-8)
            ds[var].attrs["normalization"] = "z-score"
            ds[var].attrs["reference_mean"] = mean_value
            ds[var].attrs["reference_std"] = std_value
    return ds


def add_time_coordinates(ds, pd, xr):
    if "step" in ds.coords:
        ds = ds.assign_coords(
            lead_day=("step", ds["step"].dt.total_seconds().values / 86400.0)
        )
    if "time" in ds.coords and "step" in ds.coords:
        valid_time = pd.to_datetime(ds["time"].values[:, None]) + pd.to_timedelta(
            ds["step"].values[None, :]
        )
        ds = ds.assign_coords(valid_time=(("time", "step"), valid_time))
    return ds


def merge_requested_variables(variable_datasets, xr):
    merged = xr.merge(variable_datasets, compat="override", combine_attrs="drop_conflicts")
    keep_vars = [
        var
        for var in ["P", "SW_IN", "TA", "TA_max", "TA_min", "VPD", "VPD_min", "VPD_max"]
        if var in merged
    ]
    return merged[keep_vars]


def process_month(
    *,
    month: int,
    years: List[int],
    args: argparse.Namespace,
    lons: np.ndarray,
    lats: np.ndarray,
    area: List[float],
    client,
    xr,
    pd,
    raw_dir: Path,
    stats: Optional[Dict[str, Tuple[float, float]]],
):
    variable_datasets = []

    for variable in args.variables:
        raw_path = retrieve_variable(
            client,
            variable=variable,
            years=years,
            month=month,
            day=args.day,
            args=args,
            area=area,
            raw_dir=raw_dir,
        )
        opened = open_single_variable_dataset(raw_path, xr)
        single_name = next(iter(opened.data_vars))
        opened = opened.rename({single_name: variable})
        subset = subset_to_samples(opened, lons=lons, lats=lats, xr=xr)
        variable_datasets.append(subset)

        if not args.keep_raw:
            raw_path.unlink(missing_ok=True)

    merged = merge_requested_variables(variable_datasets, xr)
    merged = harmonize_units_and_names(merged, xr)
    merged = add_time_coordinates(merged, pd, xr)

    if stats is not None:
        merged = normalize_with_reference(merged, stats)

    merged.attrs.update(
        {
            "source_dataset": "seasonal-original-single-levels",
            "originating_centre": args.originating_centre,
            "system": str(args.system),
            "request_mode": args.mode,
            "initialization_month": int(month),
            "initialization_day": int(args.day),
            "years_requested": ",".join(str(year) for year in years),
            "note": (
                "VPD diagnostics are approximate and derived from daily dewpoint "
                "plus daily min/max temperature."
            ),
        }
    )

    years_label = f"{years[0]}_{years[-1]}"
    out_path = (
        args.output_dir
        / f"{args.originating_centre}_system{args.system}_{args.mode}_m{month:02d}_{years_label}.zarr"
    )
    merged.chunk({"sample": 1000}).to_zarr(out_path, mode="w", consolidated=True)
    return out_path


def main():
    args = parse_args()
    cdsapi, pd, xr = _lazy_imports()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.scratch_dir or (args.output_dir / "_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    lons, lats = extract_locations_from_paths(args.sample_paths)
    area = build_area(lons=lons, lats=lats)
    client = cdsapi.Client()

    stats = None
    if args.normalize_from is not None:
        stats = compute_reference_stats(args.normalize_from, xr)

    written = []
    for month in args.months:
        if not 1 <= month <= 12:
            raise ValueError(f"Initialization month must be in 1..12, got {month}")
        out_path = process_month(
            month=month,
            years=args.years,
            args=args,
            lons=lons,
            lats=lats,
            area=area,
            client=client,
            xr=xr,
            pd=pd,
            raw_dir=raw_dir,
            stats=stats,
        )
        written.append(out_path)
        print(f"Wrote {out_path}")

    print("\nFinished seasonal forecast download + harmonization.")
    print("Outputs:")
    for path in written:
        print(path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
