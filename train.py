from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_criterion = nn.BCEWithLogitsLoss()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    proximal_mu: float = 0.0,
    global_params: list[torch.Tensor] | None = None,
) -> tuple[float, float]:
    """Run one pass over the local training data.

    When proximal_mu > 0 (FedProx mode), adds a penalty term that discourages the
    local model from drifting too far from the global weights received at the start
    of the round. The penalty is excluded from the returned loss value so that
    training curves remain directly comparable between FedAvg and FedProx runs.
    """
    model.train()
    total_loss = correct = total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().squeeze(1).to(device)

        optimizer.zero_grad()
        logits     = model(images).squeeze(1)
        task_loss  = _criterion(logits, labels)

        if proximal_mu > 0.0 and global_params is not None:
            prox = sum(
                ((p - g) ** 2).sum()
                for p, g in zip(model.parameters(), global_params)
            )
            loss = task_loss + (proximal_mu / 2.0) * prox
        else:
            loss = task_loss

        loss.backward()
        optimizer.step()

        # accumulate only the task loss so FedAvg and FedProx curves are on the same scale
        total_loss += task_loss.item() * len(labels)
        preds   = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total   += len(labels)

    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[float, float, list[int], list[int]]:
    """Evaluate the model on a dataloader, returning loss, accuracy, and raw predictions."""
    model.eval()
    total_loss = correct = total = 0
    all_preds: list[int]  = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().squeeze(1).to(device)

            logits = model(images).squeeze(1)
            loss   = _criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            preds   = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total   += len(labels)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.long().cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels
