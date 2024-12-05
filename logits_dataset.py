import numpy as np
import torch
from torch.utils.data import Dataset

class LogitsDataset(Dataset):
    """
    A dataset for loading precomputed logits and their corresponding labels.
    """
    def __init__(self, logits_path, labels_path):
        """
        Initialize the dataset by loading logits and labels from .npy files.

        Args:
            logits_path (str): Path to the NumPy file containing logits.
            labels_path (str): Path to the NumPy file containing labels.
        """
        self.logits = np.load(logits_path)  # Shape: (num_samples, sequence_length, hidden_dim)
        self.labels = np.load(labels_path)  # Shape: (num_samples,)

        # Ensure the number of samples in logits and labels match
        assert len(self.logits) == len(self.labels), (
            f"Mismatch between number of samples in logits ({len(self.logits)}) "
            f"and labels ({len(self.labels)})"
        )

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve a single sample's logits and label.

        Args:
            idx (int): Index of the sample.

        Returns:
            (torch.Tensor, torch.Tensor): Logits and corresponding label.
        """
        x = self.logits[idx]  # Logits for the sample (sequence_length, hidden_dim)
        y = self.labels[idx]  # Corresponding label (scalar)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)