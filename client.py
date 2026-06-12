from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import flwr as fl
from flwr.common import Context
from torch.utils.data import Dataset

from config import DEVICE, BATCH_SIZE, LEARNING_RATE, LOCAL_EPOCHS
from data.dataset import get_dataloader
from train import train_epoch, evaluate


class MedicalClient(fl.client.NumPyClient):
    """Flower client wrapping a local model and its training/evaluation logic.

    Each client holds one partition of the training set and the shared validation set.
    It receives global model weights from the server at the start of every round,
    trains locally, and sends the updated weights back.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: Dataset,
        val_data: Dataset,
    ) -> None:
        self.model        = model.to(DEVICE)
        self.train_loader = get_dataloader(train_data, BATCH_SIZE, shuffle=True)
        self.val_loader   = get_dataloader(val_data,   BATCH_SIZE, shuffle=False)
        self.optimizer    = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        state = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        )
        self.model.load_state_dict(state, strict=True)

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict,
    ) -> tuple[list[np.ndarray], int, dict]:
        self.set_parameters(parameters)
        epochs = int(config.get("local_epochs", LOCAL_EPOCHS))
        mu     = float(config.get("proximal_mu", 0.0))

        # snapshot the global weights before any local update so the proximal penalty
        # always measures distance from this round's starting point, not a previous one
        global_params = (
            [p.detach().clone() for p in self.model.parameters()]
            if mu > 0.0 else None
        )

        loss = acc = 0.0
        for _ in range(epochs):
            loss, acc = train_epoch(
                self.model, self.train_loader, self.optimizer, DEVICE,
                proximal_mu=mu, global_params=global_params,
            )

        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {"train_loss": float(loss), "train_accuracy": float(acc)},
        )

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict,
    ) -> tuple[float, int, dict]:
        self.set_parameters(parameters)
        loss, acc, _, _ = evaluate(self.model, self.val_loader, DEVICE)
        return float(loss), len(self.val_loader.dataset), {"val_accuracy": float(acc)}


def make_client_fn(
    client_datasets: list[Dataset],
    val_dataset: Dataset,
    model_class: type,
):
    """Return a client factory compatible with Flower 1.x ClientApp.

    Flower passes a Context to client_fn; the partition index is read from
    context.node_config["partition-id"], which Flower sets sequentially from
    0 to num_supernodes-1 so each client gets a unique data shard.
    """
    def client_fn(context: Context) -> fl.client.Client:
        partition_id = int(context.node_config["partition-id"])
        model  = model_class()
        client = MedicalClient(model, client_datasets[partition_id], val_dataset)
        return client.to_client()

    return client_fn
