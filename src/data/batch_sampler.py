from torch.utils.data.sampler import Sampler
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import random
from collections import defaultdict


class BatchSampler(Sampler):
    r"""Yield a mini-batch of indices.

    Args:
        data: Dataset for building sampling logic.
        batch_size: Size of mini-batch.
    """

    def __init__(self, dataset, shuffle=True):
        # build data for sampling here
        self.shuffle = shuffle
        self.batch_to_indices = self.batch_years(sample_years=dataset.training_pairs)
        self.batch_ids = list(self.batch_to_indices.keys())

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batch_ids)
        for batch_id in self.batch_ids:
            yield self.batch_to_indices[batch_id]

    def __len__(self):
        return len(self.batch_ids)

    def batch_years(self, sample_years):
        batch_to_indices = defaultdict(list)
        for idx, (path, year) in enumerate(sample_years):
            batch_to_indices[path].append(idx)
        return batch_to_indices
