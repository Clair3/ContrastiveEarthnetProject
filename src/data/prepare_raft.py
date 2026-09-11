"""
Prepare a lightweight RAFT-style retrieval dataset for EVI forecasting.

Setup
-----
Forecasting context:
    1 year = 73 observations (5-day sampling)

Retrieval key:
    last 90 days = 18 observations

Retrieval value:
    following prediction period = 18 observations

Retrieval candidates:
    - training samples only
    - same location only
    - temporally non-overlapping with query
    - top-k by EVI-history cosine similarity

The forecasting context and retrieval context are deliberately different.

The resulting file contains:

    retrieval:
        [N_query, topm, pred_len, n_vegetation_channels]

    training_pairs:
        [(sample_id, t0), ...]

This is intended as a cheap retrieval-augmented forecasting baseline.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

DEFAULT_RETRIEVAL_DAYS = 90
DEFAULT_TOPM = 20
DEFAULT_TEMPERATURE = 0.1


def days_to_steps(days, sampling_days):
    return max(1, round(days / sampling_days))


# ============================================================
# Retrieval
# ============================================================


class EVIHistoryRetriever:

    def __init__(
        self,
        retrieval_length,
        prediction_length,
        topm=20,
        temperature=0.1,
        batch_size=512,
        device="cuda",
    ):

        self.retrieval_length = retrieval_length
        self.prediction_length = prediction_length

        self.topm = topm
        self.temperature = temperature

        self.batch_size = batch_size
        self.device = torch.device(device)

        self.train_keys = None
        self.train_values = None

        self.location_ids = None
        self.t0s = None

        self.n_train = None

    # --------------------------------------------------------
    # Build retrieval corpus
    # --------------------------------------------------------

    def prepare_corpus(self, dataset):

        keys = []
        values = []
        locations = []
        t0s = []

        print("Preparing retrieval corpus...")

        for sample_id, t0 in tqdm(dataset.training_pairs):

            veg = dataset.vegetation[sample_id]

            # ------------------------------------------------
            # Retrieval key:
            #
            # last 90 days before t0
            # ------------------------------------------------

            key = veg[t0 - self.retrieval_length : t0]

            # ------------------------------------------------
            # Retrieved value:
            #
            # future EVI after candidate t0
            # ------------------------------------------------

            value = veg[t0 : t0 + self.prediction_length]

            keys.append(key)
            values.append(value)

            locations.append(sample_id)
            t0s.append(t0)

        self.train_keys = torch.stack(keys).float()

        self.train_values = torch.stack(values).float()

        self.location_ids = torch.tensor(
            locations,
            dtype=torch.long,
        )

        self.t0s = torch.tensor(
            t0s,
            dtype=torch.long,
        )

        self.n_train = len(dataset.training_pairs)

        print(f"Retrieval keys: " f"{self.train_keys.shape}")

        print(f"Retrieval values: " f"{self.train_values.shape}")

        print(f"Number of candidates: " f"{self.n_train}")

    # --------------------------------------------------------
    # Retrieve one batch
    # --------------------------------------------------------

    @torch.no_grad()
    def retrieve_batch(
        self,
        query_keys,
        query_locations,
        query_t0s,
    ):
        """
        query_keys:
            [B, retrieval_length, channels]

        query_locations:
            [B]

        query_t0s:
            [B]

        Returns:
            [B, topm, prediction_length, channels]
        """

        query_keys = query_keys.to(self.device)

        query_locations = query_locations.to(self.device)

        query_t0s = query_t0s.to(self.device)

        B, L, C = query_keys.shape

        if L != self.retrieval_length:
            raise ValueError(
                f"Expected retrieval length " f"{self.retrieval_length}, " f"got {L}"
            )

        # ----------------------------------------------------
        # We retrieve independently for each location.
        #
        # Because training_pairs are grouped by location,
        # the number of candidates per location is small.
        # ----------------------------------------------------

        outputs = []

        unique_locations = query_locations.unique()

        train_keys = self.train_keys
        train_values = self.train_values
        train_locations = self.location_ids
        train_t0s = self.t0s

        for location in unique_locations:

            batch_mask = query_locations == location

            batch_indices = batch_mask.nonzero(as_tuple=False).squeeze(-1)

            q = query_keys[batch_indices]

            q_t0 = query_t0s[batch_indices]

            # ------------------------------------------------
            # Candidates from this location only
            # ------------------------------------------------

            candidate_mask = train_locations == location.cpu()

            candidate_indices = candidate_mask.nonzero(as_tuple=False).squeeze(-1)

            candidate_keys = train_keys[candidate_indices].to(self.device)

            candidate_values = train_values[candidate_indices].to(self.device)

            candidate_t0s = train_t0s[candidate_indices].to(self.device)

            # ------------------------------------------------
            # [B, N, L, C]
            # ------------------------------------------------

            N = candidate_keys.shape[0]

            q_expanded = q[:, None, :, :]

            c_expanded = candidate_keys[None, :, :, :]

            # ------------------------------------------------
            # Flatten temporal/channel dimensions
            # ------------------------------------------------

            q_flat = q_expanded.reshape(
                q.shape[0],
                1,
                -1,
            )

            c_flat = c_expanded.reshape(
                1,
                N,
                -1,
            )

            # ------------------------------------------------
            # Cosine similarity
            # ------------------------------------------------

            q_norm = F.normalize(
                q_flat,
                dim=-1,
            )

            c_norm = F.normalize(
                c_flat,
                dim=-1,
            )

            similarity = (q_norm * c_norm).sum(dim=-1)

            # [B, N]

            # ------------------------------------------------
            # Temporal overlap mask
            #
            # Query forecast:
            #
            #   [q_t0, q_t0 + pred_len)
            #
            # Candidate forecast:
            #
            #   [candidate_t0,
            #    candidate_t0 + pred_len)
            #
            # ------------------------------------------------

            candidate_start = candidate_t0s[None, :]

            candidate_end = candidate_t0s[None, :] + self.prediction_length

            query_start = q_t0[:, None]

            query_end = q_t0[:, None] + self.prediction_length

            overlap = (candidate_start < query_end) & (candidate_end > query_start)

            similarity = similarity.masked_fill(
                overlap,
                float("-inf"),
            )

            # ------------------------------------------------
            # Also exclude the exact same sample/window
            # automatically through temporal overlap.
            # ------------------------------------------------

            available = torch.isfinite(similarity).sum(dim=1)

            if torch.any(available < self.topm):

                minimum = int(available.min().item())

                raise RuntimeError(
                    f"Location {int(location)} "
                    f"has fewer than {self.topm} "
                    f"valid retrieval candidates. "
                    f"Minimum available: {minimum}"
                )

            # ------------------------------------------------
            # Top-k
            # ------------------------------------------------

            top_values, top_indices = torch.topk(
                similarity,
                self.topm,
                dim=1,
            )

            # ------------------------------------------------
            # Temperature weighting
            # ------------------------------------------------

            weights = F.softmax(
                top_values / self.temperature,
                dim=1,
            )

            # ------------------------------------------------
            # Retrieve future EVI
            # ------------------------------------------------

            retrieved_values = candidate_values[top_indices]

            # [B, topm, pred_len, C]

            # ------------------------------------------------
            # Weighted retrieved future
            #
            # We keep ALL top-m futures because this is useful
            # for the RAFT model.
            # ------------------------------------------------

            # Store weights separately.
            #
            # For now the output contains the actual analogue
            # futures and their weights.
            # ------------------------------------------------

            batch_output = retrieved_values.cpu()

            batch_weights = weights.cpu()

            outputs.append(
                (
                    batch_indices.cpu(),
                    batch_output,
                    batch_weights,
                )
            )

        # ----------------------------------------------------
        # Reconstruct original batch order
        # ----------------------------------------------------

        retrieval = torch.empty(
            B,
            self.topm,
            self.prediction_length,
            self.train_values.shape[-1],
        )

        weights = torch.empty(
            B,
            self.topm,
        )

        for (
            batch_indices,
            batch_output,
            batch_weights,
        ) in outputs:

            retrieval[batch_indices] = batch_output

            weights[batch_indices] = batch_weights

        return retrieval, weights

    # --------------------------------------------------------
    # Retrieve entire dataset
    # --------------------------------------------------------

    @torch.no_grad()
    def retrieve_dataset(
        self,
        dataset,
    ):

        N = len(dataset.training_pairs)

        all_retrievals = []
        all_weights = []

        print(f"Computing retrievals for " f"{N} samples...")

        for start in tqdm(
            range(
                0,
                N,
                self.batch_size,
            )
        ):

            end = min(
                start + self.batch_size,
                N,
            )

            pairs = dataset.training_pairs[start:end]

            keys = []
            locations = []
            t0s = []

            for sample_id, t0 in pairs:

                veg = dataset.vegetation[sample_id]

                key = veg[t0 - self.retrieval_length : t0]

                keys.append(key)
                locations.append(sample_id)
                t0s.append(t0)

            query_keys = torch.stack(keys).float()

            query_locations = torch.tensor(
                locations,
                dtype=torch.long,
            )

            query_t0s = torch.tensor(
                t0s,
                dtype=torch.long,
            )

            retrieval, weights = self.retrieve_batch(
                query_keys,
                query_locations,
                query_t0s,
            )

            all_retrievals.append(retrieval)

            all_weights.append(weights)

        return (
            torch.cat(
                all_retrievals,
                dim=0,
            ),
            torch.cat(
                all_weights,
                dim=0,
            ),
        )


# ============================================================
# Dataset
# ============================================================


def build_dataset(
    data_config,
    years,
    context_days,
    prediction_days,
):
    from data.datasets import ForecastingValDataset

    return ForecastingValDataset(
        dataset_path=data_config["path"],
        vegetation_indices=data_config["vegetation"]["variables"],
        weather_vars=data_config["weather"]["variables"],
        years=years,
        context_days=context_days,
        prediction_days=prediction_days,
        sampling_days=data_config["sampling_days"],
    )


def save_retrieval(
    output_file,
    retrieval,
    weights,
    dataset,
    retrieval_length,
    prediction_length,
    args,
    data_config,
):

    torch.save(
        {
            "retrieval": retrieval,
            "weights": weights,
            "training_pairs": dataset.training_pairs,
            "context_length": dataset.context_length,
            "retrieval_length": retrieval_length,
            "prediction_length": prediction_length,
            "topm": args.topm,
            "temperature": args.temperature,
            "vegetation_variables": (data_config["vegetation"]["variables"]),
            "retrieval_type": (
                "EVI history, same location, "
                "retrieved from training corpus, "
                "non-overlapping forecast periods"
            ),
        },
        output_file,
    )

    print(f"Saved {output_file}")


# ============================================================
# Main
# ============================================================


def main():

    parser = argparse.ArgumentParser()
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

    parser.add_argument(
        "--context_days",
        type=int,
        default=365,
    )

    parser.add_argument(
        "--prediction_days",
        type=int,
        default=92,
    )

    parser.add_argument(
        "--retrieval_days",
        type=int,
        default=90,
    )

    parser.add_argument(
        "--topm",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load YAML
    # --------------------------------------------------------

    import yaml

    with open(
        args.data_config,
        "r",
    ) as f:

        data_config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Build dataset
    # --------------------------------------------------------
    print("\nLoading training dataset...")

    train_dataset = build_dataset(
        data_config,
        years=data_config["forecasting"]["train"],
        context_days=args.context_days,
        prediction_days=args.prediction_days,
    )

    print("\nLoading validation dataset...")

    val_dataset = build_dataset(
        data_config,
        years=data_config["forecasting"]["validation"],
        context_days=args.context_days,
        prediction_days=args.prediction_days,
    )

    print("\nLoading test dataset...")

    test_dataset = build_dataset(
        data_config,
        years=data_config["forecasting"]["test"],
        context_days=args.context_days,
        prediction_days=args.prediction_days,
    )

    retrieval_length = days_to_steps(
        args.retrieval_days,
        data_config["sampling_days"],
    )

    prediction_length = train_dataset.prediction_length

    channels = len(data_config["vegetation"]["variables"])

    print("\nRetrieval configuration:")

    print(f"  forecasting context : " f"{train_dataset.context_length} steps")

    print(f"  retrieval context   : " f"{retrieval_length} steps")

    print(f"  prediction length   : " f"{prediction_length} steps")

    print(f"  channels            : " f"{channels}")

    print(f"  top-m               : " f"{args.topm}")

    print(f"  temperature         : " f"{args.temperature}")

    print("\nImportant:")

    print("Retrieval is EVI-only and same-location.")

    print("Candidate forecast periods may not " "overlap the query forecast period.")

    # --------------------------------------------------------
    # Retriever
    # --------------------------------------------------------
    retriever = EVIHistoryRetriever(
        retrieval_length=retrieval_length,
        prediction_length=train_dataset.prediction_length,
        topm=args.topm,
        temperature=args.temperature,
        batch_size=args.batch_size,
        device=args.device,
    )

    retriever.prepare_corpus(train_dataset)

    # --------------------------------------------------------
    # Compute retrievals
    # --------------------------------------------------------
    print("\nRetrieving training queries...")

    train_retrieval, train_weights = retriever.retrieve_dataset(train_dataset)

    print("\nRetrieving validation queries...")

    val_retrieval, val_weights = retriever.retrieve_dataset(val_dataset)

    print("\nRetrieving test queries...")

    test_retrieval, test_weights = retriever.retrieve_dataset(test_dataset)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_retrieval(
        output_dir / "raft_train.pt",
        train_retrieval,
        train_weights,
        train_dataset,
        retrieval_length,
        prediction_length,
        args,
        data_config,
    )

    save_retrieval(
        output_dir / "raft_val.pt",
        val_retrieval,
        val_weights,
        val_dataset,
        retrieval_length,
        prediction_length,
        args,
        data_config,
    )

    save_retrieval(
        output_dir / "raft_test.pt",
        test_retrieval,
        test_weights,
        test_dataset,
        retrieval_length,
        prediction_length,
        args,
        data_config,
    )

    output_file = output_dir / "raft_train.pt"

    print("\n========================================")

    print("RAFT retrieval preprocessing complete.")


if __name__ == "__main__":
    main()
