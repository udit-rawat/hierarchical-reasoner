# src/dataset.py
"""
Synthetic Reasoning Dataset
---------------------------
A minimal dataset for hierarchical reasoning model testing.
Generates simple arithmetic tasks like addition and subtraction.

Each sample:
  Input: sequence of two numbers + operation token (encoded numerically)
  Target: numeric result
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random


class SyntheticReasoningDataset(Dataset):
    """Generates arithmetic reasoning samples (add / subtract)."""

    def __init__(self, size: int = 10000, seq_len: int = 3, vocab_size: int = 20):
        """
        Args:
            size: number of samples
            seq_len: sequence length (2 numbers + 1 op token)
            vocab_size: upper bound for random numbers
        """
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.samples = self._generate_samples()

    def _generate_samples(self):
        data = []
        for _ in range(self.size):
            a = random.randint(0, self.vocab_size - 1)
            b = random.randint(0, self.vocab_size - 1)
            op = random.choice(["add", "sub"])
            target = a + b if op == "add" else a - b
            input_tokens = torch.tensor(
                [a, b, 0 if op == "add" else 1], dtype=torch.float)
            data.append((input_tokens, torch.tensor(
                [target], dtype=torch.float)))
        return data

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.samples[idx]


def get_dataloaders(batch_size: int = 32, train_size: int = 8000, test_size: int = 2000):
    """Returns PyTorch dataloaders for train/test splits."""
    train_dataset = SyntheticReasoningDataset(size=train_size)
    test_dataset = SyntheticReasoningDataset(size=test_size)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    # Sanity check
    train_loader, _ = get_dataloaders()
    batch_x, batch_y = next(iter(train_loader))
    print("Input batch shape:", batch_x.shape)
    print("Target batch shape:", batch_y.shape)
    print("Example input:", batch_x[0])
    print("Example target:", batch_y[0])
