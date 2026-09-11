"""
Diagnostics for retrieval-augmented vegetation forecasting.

Main questions:
1. Does history similarity predict future similarity?
2. Is retrieval driven mainly by temporal autocorrelation?
3. Does cross-location retrieval work?
4. Does multivariate EVI + weather improve analogue quality?

Designed to work with the same dataset structure as the current
EVIHistoryRetriever:
    dataset.vegetation[sample_id]
    dataset.weather[sample_id]       # if available
    dataset.training_pairs
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

DEFAULT_RETRIEVAL_DAYS = 90
DEFAULT_TOPK = 20
DEFAULT_MIN_TEMPORAL_GAP = 0


# ============================================================
# Basic distances
# ============================================================


def cosine_similarity(a, b):
    """
    a, b: [L, C]
    """
    a = torch.as_tensor(a, dtype=torch.float32).flatten()
    b = torch.as_tensor(b, dtype=torch.float32).flatten()

    a = F.normalize(a.unsqueeze(0), dim=-1)
    b = F.normalize(b.unsqueeze(0), dim=-1)

    return float((a * b).sum())


def correlation_similarity(a, b):
    """
    Pearson correlation after flattening.
    """
    a = torch.as_tensor(a, dtype=torch.float32).flatten()
    b = torch.as_tensor(b, dtype=torch.float32).flatten()

    a = a - a.mean()
    b = b - b.mean()

    denom = torch.sqrt((a**2).sum() * (b**2).sum())

    if denom < 1e-8:
        return 0.0

    return float((a * b).sum() / denom)


def normalized_euclidean_similarity(a, b):
    """
    Normalize each vector and return negative Euclidean distance.

    Larger = more similar.
    """
    a = torch.as_tensor(a, dtype=torch.float32).flatten()
    b = torch.as_tensor(b, dtype=torch.float32).flatten()

    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)

    distance = torch.sqrt(((a - b) ** 2).mean())

    return float(-distance)


def rmse(a, b):
    """
    RMSE between two trajectories.
    Smaller = more similar.
    """
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)

    return float(torch.sqrt(((a - b) ** 2).mean()))


# ============================================================
# Temporal distance
# ============================================================


def temporal_distance(t0_a, t0_b):
    return abs(int(t0_a) - int(t0_b))


# ============================================================
# Dataset extraction
# ============================================================


def get_window(series, t0, length):
    """
    Extract [t0-length, t0).
    """
    return series[t0 - length : t0]


def get_future(series, t0, prediction_length):
    """
    Extract [t0, t0+prediction_length).
    """
    return series[t0 : t0 + prediction_length]


# ============================================================
# Candidate generation
# ============================================================


def build_pairs(
    dataset,
    retrieval_length,
    prediction_length,
    max_candidates=None,
):
    """
    Build query/candidate pairs.

    IMPORTANT:
    We deliberately do NOT restrict to same location.

    This allows us to separately study:
        same-location
        cross-location

    We also exclude overlapping forecast periods.
    """

    pairs = []

    all_pairs = list(dataset.training_pairs)

    if max_candidates is not None:
        all_pairs = all_pairs[:max_candidates]

    for q_idx, (q_location, q_t0) in enumerate(all_pairs):

        q_future_start = q_t0
        q_future_end = q_t0 + prediction_length

        for c_idx, (c_location, c_t0) in enumerate(all_pairs):

            # Don't compare the query with itself.
            if q_idx == c_idx:
                continue

            c_future_start = c_t0
            c_future_end = c_t0 + prediction_length

            # ------------------------------------------------
            # Exclude overlapping future periods.
            # ------------------------------------------------
            overlap = c_future_start < q_future_end and c_future_end > q_future_start

            if overlap:
                continue

            pairs.append(
                {
                    "query_idx": q_idx,
                    "candidate_idx": c_idx,
                    "query_location": int(q_location),
                    "candidate_location": int(c_location),
                    "query_t0": int(q_t0),
                    "candidate_t0": int(c_t0),
                    "same_location": int(q_location == c_location),
                    "temporal_distance": temporal_distance(
                        q_t0,
                        c_t0,
                    ),
                }
            )

    return pairs


# ============================================================
# Evaluate retrieval metrics
# ============================================================


def evaluate_pairs(
    dataset,
    pairs,
    retrieval_length,
    prediction_length,
    use_weather=False,
):
    """
    Compute history similarity and future similarity for
    every query/candidate pair.
    """

    rows = []

    for pair in tqdm(pairs, desc="Evaluating pairs"):

        q_loc = pair["query_location"]
        c_loc = pair["candidate_location"]

        q_t0 = pair["query_t0"]
        c_t0 = pair["candidate_t0"]

        # ----------------------------------------------------
        # Vegetation
        # ----------------------------------------------------

        q_veg = dataset.vegetation[q_loc]

        c_veg = dataset.vegetation[c_loc]

        q_hist = get_window(
            q_veg,
            q_t0,
            retrieval_length,
        )

        c_hist = get_window(
            c_veg,
            c_t0,
            retrieval_length,
        )

        q_future = get_future(
            q_veg,
            q_t0,
            prediction_length,
        )

        c_future = get_future(
            c_veg,
            c_t0,
            prediction_length,
        )

        # ----------------------------------------------------
        # History metrics
        # ----------------------------------------------------

        evi_cosine = cosine_similarity(
            q_hist,
            c_hist,
        )

        evi_corr = correlation_similarity(
            q_hist,
            c_hist,
        )

        evi_euclidean = normalized_euclidean_similarity(
            q_hist,
            c_hist,
        )

        # ----------------------------------------------------
        # Future metrics
        # ----------------------------------------------------

        future_corr = correlation_similarity(
            q_future,
            c_future,
        )

        future_rmse = rmse(
            q_future,
            c_future,
        )

        # ----------------------------------------------------
        # Weather
        # ----------------------------------------------------

        weather_cosine = np.nan

        if use_weather and hasattr(dataset, "weather"):

            q_weather = dataset.weather[q_loc]

            c_weather = dataset.weather[c_loc]

            q_weather_hist = get_window(
                q_weather,
                q_t0,
                retrieval_length,
            )

            c_weather_hist = get_window(
                c_weather,
                c_t0,
                retrieval_length,
            )

            weather_cosine = cosine_similarity(
                q_weather_hist,
                c_weather_hist,
            )

        rows.append(
            {
                **pair,
                "evi_cosine": evi_cosine,
                "evi_correlation": evi_corr,
                "evi_euclidean": evi_euclidean,
                "future_correlation": future_corr,
                "future_rmse": future_rmse,
                "weather_cosine": weather_cosine,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Retrieval evaluation
# ============================================================


def evaluate_topk(
    df,
    similarity_column,
    topk=20,
):
    """
    For each query, retrieve top-k candidates and evaluate
    their actual future similarity.

    This is the most important diagnostic.
    """

    results = []

    for query_idx, group in df.groupby("query_idx"):

        group = group.sort_values(
            similarity_column,
            ascending=False,
        )

        top = group.head(topk)

        results.append(
            {
                "query_idx": query_idx,
                "future_corr_mean": top["future_correlation"].mean(),
                "future_corr_median": top["future_correlation"].median(),
                "future_rmse_mean": top["future_rmse"].mean(),
                "temporal_distance_mean": top["temporal_distance"].mean(),
                "same_location_fraction": top["same_location"].mean(),
            }
        )

    return pd.DataFrame(results)


# ============================================================
# Correlation diagnostics
# ============================================================


def correlation_report(df):
    """
    Compute correlations between history similarity and
    future similarity.

    This directly tests the main hypothesis.
    """

    metrics = [
        "evi_cosine",
        "evi_correlation",
        "evi_euclidean",
        "weather_cosine",
    ]

    rows = []

    for metric in metrics:

        if metric not in df:
            continue

        valid = df[[metric, "future_correlation"]].dropna()

        if len(valid) < 10:
            continue

        pearson = valid[metric].corr(
            valid["future_correlation"],
            method="pearson",
        )

        spearman = valid[metric].corr(
            valid["future_correlation"],
            method="spearman",
        )

        rows.append(
            {
                "history_metric": metric,
                "pearson": pearson,
                "spearman": spearman,
                "n": len(valid),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Temporal autocorrelation diagnostic
# ============================================================


def temporal_bins(df):
    """
    Show how retrieval quality changes with temporal distance.
    """

    bins = [
        0,
        10,
        20,
        40,
        80,
        160,
        320,
        np.inf,
    ]

    labels = [
        "0-10",
        "10-20",
        "20-40",
        "40-80",
        "80-160",
        "160-320",
        "320+",
    ]

    df = df.copy()

    df["temporal_bin"] = pd.cut(
        df["temporal_distance"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    result = (
        df.groupby("temporal_bin", observed=True)
        .agg(
            n=("future_correlation", "size"),
            history_similarity=("evi_cosine", "mean"),
            future_similarity=("future_correlation", "mean"),
            future_rmse=("future_rmse", "mean"),
        )
        .reset_index()
    )

    return result


# ============================================================
# Same vs cross location
# ============================================================


def location_report(df):

    result = (
        df.groupby("same_location")
        .agg(
            n=("future_correlation", "size"),
            history_similarity=("evi_cosine", "mean"),
            future_similarity=("future_correlation", "mean"),
            future_rmse=("future_rmse", "mean"),
            temporal_distance=("temporal_distance", "mean"),
        )
        .reset_index()
    )

    result["location_type"] = result["same_location"].map(
        {
            0: "cross-location",
            1: "same-location",
        }
    )

    return result


# ============================================================
# Main
# ============================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--retrieval_days",
        type=int,
        default=90,
    )

    parser.add_argument(
        "--prediction_days",
        type=int,
        default=92,
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--data_config",
        type=str,
        default="/Net/Groups/BGI/scratch/crobin/PythonProjects/ContrastiveEarthnetProject/configs/data_config_VIIRS_passage.yaml",
        # required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Net/Groups/BGI/tscratch/crobin/ContrastiveEarthnetProject/datasets/PassageDatasets/raft",
    )

    args = parser.parse_args()

    import yaml

    with open(args.data_config, "r") as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    from data.datasets import ForecastingValDataset

    train_dataset = ForecastingValDataset(
        dataset_path=config["path"],
        vegetation_indices=config["vegetation"]["variables"],
        weather_vars=config["weather"]["variables"],
        years=config["forecasting"]["train"],
        context_days=365,
        prediction_days=args.prediction_days,
        sampling_days=config["sampling_days"],
    )

    retrieval_length = round(args.retrieval_days / config["sampling_days"])

    prediction_length = train_dataset.prediction_length

    print()
    print("==========================================")
    print("Retrieval diagnostics")
    print("==========================================")
    print(f"Retrieval length: {retrieval_length}")
    print(f"Prediction length: {prediction_length}")
    print(f"Number of training samples: " f"{len(train_dataset.training_pairs)}")

    # --------------------------------------------------------
    # Build all candidate pairs
    # --------------------------------------------------------

    pairs = build_pairs(
        train_dataset,
        retrieval_length,
        prediction_length,
    )

    print(f"Candidate pairs: {len(pairs):,}")

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    df = evaluate_pairs(
        train_dataset,
        pairs,
        retrieval_length,
        prediction_length,
        use_weather=True,
    )

    # --------------------------------------------------------
    # Save raw results
    # --------------------------------------------------------

    df.to_csv(
        output_dir / "pairwise_diagnostics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 1. Correlation report
    # --------------------------------------------------------

    corr = correlation_report(df)

    print()
    print("==========================================")
    print("1. HISTORY → FUTURE RELATIONSHIP")
    print("==========================================")

    print(corr.to_string(index=False))

    corr.to_csv(
        output_dir / "history_future_correlation.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 2. Temporal distance
    # --------------------------------------------------------

    temporal = temporal_bins(df)

    print()
    print("==========================================")
    print("2. TEMPORAL AUTOCORRELATION")
    print("==========================================")

    print(temporal.to_string(index=False))

    temporal.to_csv(
        output_dir / "temporal_bins.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. Same vs cross location
    # --------------------------------------------------------

    locations = location_report(df)

    print()
    print("==========================================")
    print("3. LOCATION EFFECT")
    print("==========================================")

    print(locations.to_string(index=False))

    locations.to_csv(
        output_dir / "location_report.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 4. Top-k retrieval
    # --------------------------------------------------------

    for metric in [
        "evi_cosine",
        "evi_correlation",
        "evi_euclidean",
        "weather_cosine",
    ]:

        if metric not in df:
            continue

        result = evaluate_topk(
            df,
            metric,
            topk=args.topk,
        )

        print()
        print("==========================================")
        print(f"4. TOP-{args.topk}: {metric}")
        print("==========================================")

        print(result.mean(numeric_only=True))

        result.to_csv(
            output_dir / f"topk_{metric}.csv",
            index=False,
        )

    print()
    print("Done.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
