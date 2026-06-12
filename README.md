# Federated Learning for Medical Image Classification

Federated learning experiments on the [PneumoniaMNIST](https://medmnist.com/) dataset using the [Flower](https://flower.ai/) framework. Compares CNN vs ResNet18, IID vs non-IID data partitioning, FedAvg vs FedProx aggregation strategies, and the effect of varying the number of clients.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments

```bash
# FedAvg
python server.py --model cnn --partition iid
python server.py --model cnn --partition non_iid
python server.py --model resnet --partition iid
python server.py --model resnet --partition non_iid

# FedProx (mu=0.1)
python server.py --model cnn --partition non_iid --strategy fedprox
python server.py --model resnet --partition non_iid --strategy fedprox

# Client count scaling
python server.py --model cnn --partition non_iid --num_clients 3
python server.py --model cnn --partition non_iid --num_clients 10
```

## Evaluation

```bash
# Training curves (loss, val accuracy, test accuracy per round)
python evaluate.py --metrics outputs/metrics_cnn_iid.json

# Curves + confusion matrix + F1
python evaluate.py --metrics outputs/metrics_cnn_iid.json \
                   --weights outputs/model_cnn_iid.pt \
                   --model cnn

# All comparison charts
python compare_plots.py
```

## Project Structure

```
.
├── config.py           # hyperparameters and paths
├── server.py           # simulation entry point (strategy, server-side eval)
├── client.py           # Flower client (local training, validation)
├── train.py            # train_epoch and evaluate functions
├── evaluate.py         # post-training curves, confusion matrix, F1
├── compare_plots.py    # cross-experiment comparison charts
├── models/
│   ├── cnn.py          # 3-layer CNN baseline
│   └── resnet.py       # ResNet18 adapted for 28x28 single-channel input
├── data/
│   └── dataset.py      # IID and non-IID (Dirichlet) partitioning
├── outputs/            # metrics JSON, plots, model weights
└── requirements.txt
```

## Key Results

| Experiment | Final Acc | F1 |
|---|---|---|
| CNN — IID FedAvg | 86.5% | 0.902 |
| CNN — non-IID FedAvg | 84.8% | 0.891 |
| CNN — non-IID FedProx | 84.3% | 0.888 |
| ResNet18 — IID FedAvg | 85.9% | 0.898 |
| ResNet18 — non-IID FedAvg | 76.1% | 0.840 |
| ResNet18 — non-IID FedProx | **81.9%** | 0.873 |

FedProx provides the largest benefit for ResNet18 under non-IID partitioning (+5.8pp over FedAvg), where client drift is most severe. For the lighter CNN, the two strategies perform nearly identically.

## Configuration

Key hyperparameters in `config.py`:

| Parameter | Value | Notes |
|---|---|---|
| `NUM_CLIENTS` | 5 | default; override with `--num_clients` |
| `NUM_ROUNDS` | 20 | communication rounds |
| `DIRICHLET_ALPHA` | 0.5 | lower = more heterogeneous partitions |
| `FEDPROX_MU` | 0.1 | proximal penalty (FedProx paper default) |
| `RANDOM_SEED` | 42 | fixed for reproducibility |
