import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class PrecomputedJetDataset(Dataset):
    """
    Loads precomputed dense jet images from a single HDF5 file.
    Uses lazy file opening (_ensure_file) for safe multiprocessing.
    """
    def __init__(self, h5_path, indices=None):
        self.h5_path = h5_path
        self.indices = indices
        self._file = None

        with h5py.File(self.h5_path, "r") as f_temp:
             self._N_total = len(f_temp["images"])

    def _ensure_file(self):
        """Opens the HDF5 file lazily, once per process/thread."""
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
            self.imgs = self._file["images"]   # Shape (N, 3, 50, 50)
            self.mjj  = self._file["MJJ"]      # Shape (N,)
            self.lab  = self._file["labels"]   # Shape (N,)

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        else:
            return self._N_total

    def __getitem__(self, idx):
        self._ensure_file()
        
        # Use subset index if provided
        if self.indices is not None:
            idx = self.indices[idx]

        # Check for skipped event sentinel value (MJJ = -1.0)
        # Assumes images were pre-allocated with zeros for skipped events.
        if self.mjj[idx] == -1.0:
             return None 

        # Read data
        img  = self.imgs[idx]
        mjj  = self.mjj[idx]
        label  = self.lab[idx]

        # Convert to Tensors
        img = torch.from_numpy(img)                         # [C,H,W] (float32)
        mjj = torch.as_tensor(mjj, dtype=torch.float32)     # scalar float
        label = torch.as_tensor(label, dtype=torch.long)        # scalar int64 (for labels)

        # Returns (img, mjj, label)
        return img, mjj, label