from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from medmnist import PneumoniaMNIST

from config import DATA_DIR, BATCH_SIZE


_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def get_datasets() -> tuple[PneumoniaMNIST, PneumoniaMNIST, PneumoniaMNIST]:
    # medmnist 3.x requires the root directory to already exist before it tries to download
    os.makedirs(DATA_DIR, exist_ok=True)
    kwargs = dict(root=DATA_DIR, download=True, transform=_TRANSFORM)
    train = PneumoniaMNIST(split="train", **kwargs)
    val   = PneumoniaMNIST(split="val",   **kwargs)
    test  = PneumoniaMNIST(split="test",  **kwargs)
    return train, val, test


def _labels_array(dataset) -> np.ndarray:
    return np.array([int(dataset[i][1]) for i in range(len(dataset))])


def iid_partition(dataset, num_clients: int) -> list[Subset]:
    """Randomly shuffle and split the dataset into equal-sized client shards."""
    indices = np.random.permutation(len(dataset))
    splits  = np.array_split(indices, num_clients)
    return [Subset(dataset, s.tolist()) for s in splits]


def non_iid_partition(
    dataset,
    num_clients: int,
    alpha: float = 0.5,
) -> list[Subset]:
    """Partition dataset using per-class Dirichlet draws to simulate data heterogeneity.

    For each class, proportions are sampled from Dir(alpha) and used to split that
    class's samples across clients. Lower alpha means more skewed partitions —
    at alpha=0.5 some clients end up with very few samples from one class, mimicking
    the real-world scenario where different hospitals serve different patient populations.
    """
    labels      = _labels_array(dataset)
    num_classes = int(labels.max()) + 1
    client_idx  = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        cls_indices = np.where(labels == cls)[0]
        np.random.shuffle(cls_indices)

        proportions = np.random.dirichlet(np.full(num_clients, alpha))
        counts      = (proportions * len(cls_indices)).astype(int)
        # rounding can leave a few samples unassigned; give the remainder to the largest bucket
        counts[np.argmax(counts)] += len(cls_indices) - counts.sum()

        ptr = 0
        for cid, count in enumerate(counts):
            client_idx[cid].extend(cls_indices[ptr : ptr + count].tolist())
            ptr += count

    # with small alpha values, integer truncation can leave a client with zero samples, which crashes the DataLoader. We steal one sample from the client with the highest sample instead.
    for cid in range(num_clients):
        while len(client_idx[cid]) == 0:
            donor = max(range(num_clients), key=lambda i: len(client_idx[i]))
            client_idx[cid].append(client_idx[donor].pop())

    return [Subset(dataset, idx) for idx in client_idx]


def get_dataloader(dataset, batch_size: int = BATCH_SIZE, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)
