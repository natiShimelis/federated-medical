"""Generate comparison charts from federated training metrics.

Usage:
    python compare_plots.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTPUT_DIR = "./outputs"
ROUNDS = list(range(1, 21))


def load_central_acc(tag: str) -> list[float]:
    path = Path(OUTPUT_DIR) / f"metrics_{tag}.json"
    data = json.loads(path.read_text())
    by_round = {e["round"]: e["accuracy"] for e in data["centralized"]}
    return [by_round[r] for r in ROUNDS]


def _style_ax(ax, title: str, xlabel: str = "Round", ylabel: str = "Test Accuracy") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(1, 20)
    ax.set_ylim(0.55, 1.00)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


# Chart 1 — FedAvg vs FedProx | CNN non-IID

def chart_fedavg_vs_fedprox_cnn() -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(ROUNDS, load_central_acc("cnn_non_iid"),        marker="o", ms=4,
            label="FedAvg",  color="#2196F3", linewidth=1.8)
    ax.plot(ROUNDS, load_central_acc("cnn_non_iid_fedprox"), marker="s", ms=4,
            label="FedProx (μ=0.1)", color="#FF5722", linewidth=1.8)

    _style_ax(ax, "FedAvg vs FedProx — CNN, Non-IID (α=0.5)")
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "compare_fedavg_fedprox_cnn_noniid.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# Chart 2 — FedAvg vs FedProx | ResNet non-IID

def chart_fedavg_vs_fedprox_resnet() -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(ROUNDS, load_central_acc("resnet_non_iid"),        marker="o", ms=4,
            label="FedAvg",  color="#2196F3", linewidth=1.8)
    ax.plot(ROUNDS, load_central_acc("resnet_non_iid_fedprox"), marker="s", ms=4,
            label="FedProx (μ=0.1)", color="#FF5722", linewidth=1.8)

    _style_ax(ax, "FedAvg vs FedProx — ResNet18, Non-IID (α=0.5)")
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "compare_fedavg_fedprox_resnet_noniid.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# Chart 3 — CNN vs ResNet | non-IID FedAvg

def chart_cnn_vs_resnet_noniid() -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(ROUNDS, load_central_acc("cnn_non_iid"),    marker="o", ms=4,
            label="CNN",     color="#4CAF50", linewidth=1.8)
    ax.plot(ROUNDS, load_central_acc("resnet_non_iid"), marker="s", ms=4,
            label="ResNet18", color="#9C27B0", linewidth=1.8)

    _style_ax(ax, "CNN vs ResNet18 — FedAvg, Non-IID (α=0.5)")
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "compare_cnn_vs_resnet_noniid_fedavg.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# Chart 4 — IID vs non-IID | CNN FedAvg

def chart_iid_vs_noniid_cnn() -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(ROUNDS, load_central_acc("cnn_iid"),     marker="o", ms=4,
            label="IID",     color="#009688", linewidth=1.8)
    ax.plot(ROUNDS, load_central_acc("cnn_non_iid"), marker="s", ms=4,
            label="Non-IID (α=0.5)", color="#F44336", linewidth=1.8)

    _style_ax(ax, "IID vs Non-IID — CNN, FedAvg")
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "compare_iid_vs_noniid_cnn_fedavg.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# Chart 5 — 3 vs 5 vs 10 clients | CNN non-IID FedAvg

def chart_client_count() -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    specs = [
        ("cnn_non_iid_3clients",  "3 clients",  "#E91E63", "o"),
        ("cnn_non_iid_5clients",  "5 clients",  "#FF9800", "s"),
        ("cnn_non_iid_10clients", "10 clients", "#3F51B5", "^"),
    ]
    for tag, label, color, marker in specs:
        ax.plot(ROUNDS, load_central_acc(tag), marker=marker, ms=4,
                label=label, color=color, linewidth=1.8)

    _style_ax(ax, "Effect of Client Count — CNN, Non-IID FedAvg (α=0.5)")
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "compare_client_count_cnn_noniid.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# Main

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    charts = [
        ("FedAvg vs FedProx — CNN non-IID",      chart_fedavg_vs_fedprox_cnn),
        ("FedAvg vs FedProx — ResNet non-IID",   chart_fedavg_vs_fedprox_resnet),
        ("CNN vs ResNet — non-IID FedAvg",        chart_cnn_vs_resnet_noniid),
        ("IID vs non-IID — CNN FedAvg",           chart_iid_vs_noniid_cnn),
        ("Client count: 3 vs 5 vs 10",            chart_client_count),
    ]

    for label, fn in charts:
        path = fn()
        print(f"  [{label}]\n    -> {path}")
