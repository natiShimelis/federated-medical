"""
Usage:
    python server.py [--model cnn|resnet] [--partition iid|non_iid]
                     [--strategy fedavg|fedprox] [--num_clients N]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import torch
import flwr as fl
from flwr.common import Context, NDArrays, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx
from flwr.client import ClientApp
from flwr.simulation import run_simulation
from torch.utils.data import Dataset

from config import (
    BATCH_SIZE, DEVICE, DIRICHLET_ALPHA, FEDPROX_MU,
    LOCAL_EPOCHS, NUM_CLIENTS, NUM_ROUNDS, OUTPUT_DIR, RANDOM_SEED,
)
from data.dataset import get_datasets, get_dataloader, iid_partition, non_iid_partition
from models import SimpleCNN, ResNet18Binary
from train import evaluate as eval_fn
from client import MedicalClient, make_client_fn



# Metric aggregation helpers

def _weighted_avg(metrics: list[tuple[int, dict]]) -> dict:
    total = sum(n for n, _ in metrics)
    agg: dict[str, float] = {}
    for n, m in metrics:
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + float(v) * n / total
    return agg


# Centralised evaluation (server-side, full test set)

def make_evaluate_fn(
    model_class: type,
    test_dataset: Dataset,
    save_path: str,
    metrics_store: list,
):
    """Build the server-side evaluation function called by Flower after each round.

    Runs the current global model on the full held-out test set (centralised evaluation),
    which gives a single unbiased accuracy number per round that is independent of how
    the training data was partitioned. The final-round weights are also saved to disk.
    """
    def evaluate_fn(
        server_round: int,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, dict[str, Scalar]]:
        model = model_class()
        state = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
        )
        model.load_state_dict(state, strict=True)

        if server_round == NUM_ROUNDS:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  -> Global model saved to {save_path}")

        model = model.to(DEVICE)
        loader = get_dataloader(test_dataset, BATCH_SIZE, shuffle=False)
        loss, acc, _, _ = eval_fn(model, loader, DEVICE)

        metrics_store.append({"round": server_round, "loss": loss, "accuracy": acc})
        print(f"  [Round {server_round:>2}] central  loss={loss:.4f}  acc={acc:.4f}")
        return float(loss), {"accuracy": float(acc)}

    return evaluate_fn


# Main

def main(
    model_name: str = "cnn",
    partition: str = "iid",
    strategy_name: str = "fedavg",
    num_clients: int | None = None,
) -> None:
    # fix seeds so partition splits and model initialisations are reproducible
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_clients   = num_clients if num_clients is not None else NUM_CLIENTS
    model_class = SimpleCNN if model_name == "cnn" else ResNet18Binary

    tag = f"{model_name}_{partition}"
    if strategy_name == "fedprox":
        tag += "_fedprox"
    if num_clients is not None:
        tag += f"_{n_clients}clients"

    print(f"\n=== Federated training | model={model_name}  partition={partition}  "
          f"strategy={strategy_name}  clients={n_clients} ===")
    print(f"    rounds={NUM_ROUNDS}  device={DEVICE}\n")

    train_dataset, val_dataset, test_dataset = get_datasets()

    if partition == "iid":
        client_datasets = iid_partition(train_dataset, n_clients)
    else:
        client_datasets = non_iid_partition(train_dataset, n_clients, DIRICHLET_ALPHA)

    # Shared containers filled by callbacks during simulation
    metrics: dict[str, list] = {"fit": [], "evaluate": [], "centralized": []}

    def fit_metrics_fn(results: list[tuple[int, dict]]) -> dict:
        agg = _weighted_avg(results)
        metrics["fit"].append(agg)
        print(f"         fit_agg   "
              f"loss={agg.get('train_loss', 0):.4f}  "
              f"acc={agg.get('train_accuracy', 0):.4f}")
        return agg

    def eval_metrics_fn(results: list[tuple[int, dict]]) -> dict:
        agg = _weighted_avg(results)
        metrics["evaluate"].append(agg)
        print(f"         eval_agg  acc={agg.get('val_accuracy', 0):.4f}")
        return agg

    save_path   = os.path.join(OUTPUT_DIR, f"model_{tag}.pt")
    evaluate_fn = make_evaluate_fn(model_class, test_dataset, save_path, metrics["centralized"])

    strategy_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=fit_metrics_fn,
        evaluate_metrics_aggregation_fn=eval_metrics_fn,
        on_fit_config_fn=lambda rnd: {"local_epochs": LOCAL_EPOCHS},
    )

    if strategy_name == "fedprox":
        strategy = FedProx(proximal_mu=FEDPROX_MU, **strategy_kwargs)
        print(f"    FedProx mu={FEDPROX_MU}\n")
    else:
        strategy = FedAvg(**strategy_kwargs)

    # ServerApp
    def server_fn(context: Context) -> ServerAppComponents:
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=NUM_ROUNDS),
        )

    server_app = ServerApp(server_fn=server_fn)
    client_app = ClientApp(client_fn=make_client_fn(client_datasets, val_dataset, model_class))

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=n_clients,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    )

    out_path = os.path.join(OUTPUT_DIR, f"metrics_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     choices=["cnn", "resnet"],        default="cnn")
    parser.add_argument("--partition", choices=["iid", "non_iid"],       default="iid")
    parser.add_argument("--strategy",    choices=["fedavg", "fedprox"],  default="fedavg")
    parser.add_argument("--num_clients", type=int,                        default=None,
                        help="Override NUM_CLIENTS from config.py")
    args = parser.parse_args()
    main(args.model, args.partition, args.strategy, args.num_clients)
