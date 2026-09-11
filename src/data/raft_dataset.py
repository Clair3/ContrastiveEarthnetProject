import torch
from torch.utils.data import Dataset


class RAFTDataset(Dataset):

    def __init__(self, base_dataset, retrieval_path):

        self.base_dataset = base_dataset

        raft = torch.load(
            retrieval_path,
        )

        self.retrieval = raft["retrieval"]
        self.weights = raft["weights"]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):

        sample = self.base_dataset[index]

        if sample is None:
            return None

        sample["retrieval"] = self.retrieval[index]
        sample["retrieval_weights"] = self.weights[index]

        return sample
