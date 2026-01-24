import torch
from torch.utils.data import Dataset
import glob

class SequenceDataset(Dataset):
    def __init__(self, pt_files):
        """
        pt_files: list of .pt files
        """
        self.samples = []

        for f in pt_files:
            data = torch.load(f)
            self.samples.extend(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s["sequence"],   # (T, D)
            s["label"],     # scalar
            s["mjj"],      # scalar
        )
