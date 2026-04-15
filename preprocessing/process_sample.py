#!/usr/bin/env python3
"""
process_train_dataset.py

Usage (single sample, good for Slurm array):
    python process_train_dataset.py --input-list sample_paths.txt --task-id $SLURM_ARRAY_TASK_ID \
        --output-dir /path/to/out 

Usage (multiprocessing local):
    python process_train_dataset.py --input-dir /path/to/minicubes --mode multiproc \
        --n-jobs 16 --output-dir /path/to/out

Usage (single path):
    python process_train_dataset.py --input-path /path/to/minicube.zarr --output-dir ...
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Iterable, List

import numpy as np
import pandas as pd
import xarray as xr

# Local imports - adapt import paths as needed
from process_vegetation import Sentinel2Preprocessing

logger = logging.getLogger("process_train_dataset")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class ProcessTrainDataset:
    def __init__(
        self,
        temporal_resolution_veg: int,
        temporal_resolution_weather: int,
        era5_variables: List[str] | None = None,
        output_dir: str | Path = "preprocessed/samples_aligned",
    ):
        self.temporal_resolution_veg = int(temporal_resolution_veg)
        self.temporal_resolution_weather = int(temporal_resolution_weather)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.era5_variables = (
            era5_variables
            if era5_variables is not None
            else [
                "t2m_mean",
                "tp_mean",
                "t2m_min",
                "tp_min",
                "t2m_max",
                "tp_max",
                "pev_mean",
                "ssr_mean",
                "pev_min",
                "ssr_min",
                "pev_max",
                "ssr_max",
            ]
        )

    def process_variables(self, sample_path: Path) -> xr.Dataset:

        ds = xr.open_zarr(sample_path)
        vegetation = self._process_vegetation(ds).rename({"time": "time_veg"})
        # weather = self._process_weather(ds).rename({"time": "time_weather"})
        weather = ds[self.era5_variables].rename({"time": "time_weather"})

        vegetation = self.reindex_all_years(
            vegetation,
            temporal_resolution=self.temporal_resolution_veg,
            time_var="time_veg",
        )
        weather = self.reindex_all_years(
            weather,
            temporal_resolution=self.temporal_resolution_weather,
            time_var="time_weather",
        )

        lat, lon = vegetation.location.item()
        out = xr.Dataset(
            coords={
                "latitude": lat,
                "longitude": lon,
            }
        )

        # --- Loop through vegetation variables ---
        for v in vegetation.data_vars:
            if "time_veg" in vegetation[v].dims:
                out[v] = vegetation[v].chunk({"time_veg": -1})
            else:
                out[v] = vegetation[v].chunk({"dayofyear": -1})

        # --- Loop through weather variables ---
        for w in weather.data_vars:
            out[w] = weather[w].chunk({"time_weather": -1})
        out = out.drop_vars("location")
        return out

    def process_one_sample(self, sample_path: str | Path) -> Path | None:
        """
        Process one input minicube (Zarr path) and write a single Zarr store
        with all years. Returns output path or None on failure.
        """
        path = Path(sample_path)
        try:
            ds = self.process_variables(path)
            lat = ds.latitude.item()
            lon = ds.longitude.item()

            out_name = f"{path.stem}_{lat:.5f}_{lon:.5f}.zarr"
            out_path = self.output_dir / out_name
            logger.info(f"Writing {out_path}")
            ds.to_zarr(store=str(out_path), mode="w")
            return out_path
        except Exception as e:
            logger.exception(f"Failed processing {path}: {e}")
            return None

    # -------------------
    # Helpers
    # -------------------
    def _process_vegetation(self, ds: xr.Dataset) -> xr.DataArray:
        veg_index = Sentinel2Preprocessing(
            temporal_resolution=self.temporal_resolution_veg
        ).generate_masked_vegetation_index(ds)
        if veg_index is None:
            raise ValueError("Vegetation index computation failed")
        return veg_index

    def reindex_all_years(self, data, temporal_resolution=16, time_var="time"):
        # Get all unique years
        years = np.unique(data[time_var].dt.year.values)

        reindexed_years = []
        for y in years:
            # Reuse your function
            reindexed_year = select_year(
                data,
                selected_year=int(y),
                temporal_resolution=temporal_resolution,
                time_var=time_var,
            )
            if reindexed_year is not None:
                reindexed_years.append(reindexed_year)

        # Concatenate all years along the time dimension
        data_all_years = xr.concat(reindexed_years, dim=time_var)
        return data_all_years


@staticmethod
def select_year(
    data: xr.DataArray | xr.Dataset,
    selected_year: int,
    temporal_resolution: int = 16,
    time_var: str = "time",
):
    """
    Reindex/pad time to canonical non-leap-year bins at the given temporal resolution.
    Returns the same xarray object but limited to selected_year with gaps filled as NaN.
    """

    # canonical times on a non-leap year (2019 used as canonical)
    canonical_times = pd.date_range(
        "2019-01-01", "2019-12-31", freq=f"{int(temporal_resolution)}D"
    )
    expected_times = canonical_times.map(lambda t: t.replace(year=int(selected_year)))

    # select data for the chosen year
    data_year = data.sel({time_var: data[time_var].dt.year == int(selected_year)})

    # reindex onto canonical expected times
    tol = np.timedelta64(max(1, temporal_resolution // 2), "D")

    data_year = data_year.reindex(
        {time_var: pd.to_datetime(expected_times)},
        method="nearest",
        tolerance=tol,
    )
    if data_year[time_var].isnull().any():
        return None  # year could not be properly reindexed
    return data_year


def load_paths_from_dir(input_dir: str | Path) -> List[str]:
    p = Path(input_dir)
    paths = sorted([str(x) for x in p.glob("*/*.zarr")])
    return paths


def load_paths_from_file(list_path: str | Path) -> List[str]:
    with open(list_path, "r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    return lines


def run_multiprocessing(
    worker: ProcessTrainDataset, paths: List[str], n_jobs: int | None = None
):
    n_jobs = int(n_jobs or max(1, cpu_count() - 1))
    logger.info(f"Running multiprocessing with n_jobs={n_jobs} on {len(paths)} samples")
    with Pool(processes=n_jobs) as pool:
        for _ in pool.imap_unordered(worker.process_one_sample, paths):
            pass


def run_single_task(worker: ProcessTrainDataset, path: str):
    logger.info(f"Running single task for {path}")
    return worker.process_one_sample(path)


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing minicube zarr stores (glob */*.zarr)",
    )
    p.add_argument(
        "--input-list",
        type=str,
        default=None,
        help="Text file with one zarr path per line",
    )
    p.add_argument(
        "--input-path", type=str, default=None, help="Process a single zarr path"
    )
    p.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Index into input-list (0-based). Useful for Slurm arrays",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "multiproc"],
        help="Execution mode",
    )
    p.add_argument(
        "--n-jobs", type=int, default=None, help="Number of processes for multiproc"
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--temporal-resolution-veg", type=int, default=5)
    p.add_argument("--temporal-resolution-weather", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    worker = ProcessTrainDataset(
        temporal_resolution_veg=args.temporal_resolution_veg,
        temporal_resolution_weather=args.temporal_resolution_weather,
        output_dir=args.output_dir,
    )

    # gather input paths
    if args.input_path:
        paths = [args.input_path]
    elif args.input_list:
        paths = load_paths_from_file(args.input_list)
    elif args.input_dir:
        paths = load_paths_from_dir(args.input_dir)
    else:
        raise ValueError("Provide one of --input-path, --input-dir or --input-list")

    if args.mode == "multiproc":
        run_multiprocessing(worker, paths, n_jobs=args.n_jobs)
    else:
        # single mode: either run one specified task (task-id provided), or run all sequentially
        if args.task_id is not None:
            if args.task_id < 0 or args.task_id >= len(paths):
                raise IndexError("task-id out of range for provided input list")
            path = paths[args.task_id]
            run_single_task(worker, path)
        else:
            # process sequentially (useful for debugging)
            for path in paths:
                run_single_task(worker, path)


if __name__ == "__main__":
    main()
