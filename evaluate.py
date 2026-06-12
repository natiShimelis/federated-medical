"""Post-training evaluation and visualisation.

Usage:
    # To Plot training curves:
    python evaluate.py --metrics outputs/metrics_cnn_iid.json

    # To Plot curves + test-set confusion matrix / F1:
    python evaluate.py --metrics outputs/metrics_cnn_iid.json \\
                       --weights outputs/model_cnn_iid.pt \\
                       --model   cnn
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
)

from config import BATCH_SIZE, DEVICE, OUTPUT_DIR
from data.dataset import get_datasets, get_dataloader
from models import ResNet18Binary, SimpleCNN
from train import evaluate as eval_fn


# Curve plots

def plot_curves(metrics_path: str, save_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(save_dir, exist_ok=True)
    with open(metrics_path) as f:
        history = json.load(f)

    fit_data    = history.get("fit", [])
    eval_data   = history.get("evaluate", [])
    central     = history.get("centralized", [])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # train loss (from client fit aggregation) 
    if fit_data and "train_loss" in fit_data[0]:
        rounds = range(1, len(fit_data) + 1)
        axes[0].plot(rounds, [m["train_loss"] for m in fit_data], marker="o", ms=4)
        axes[0].set_title("Train Loss (federated avg)")
        axes[0].set_xlabel("Round")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True)

    # val accuracy (from client evaluate aggregation)
    if eval_data and "val_accuracy" in eval_data[0]:
        rounds = range(1, len(eval_data) + 1)
        axes[1].plot(rounds, [m["val_accuracy"] for m in eval_data], marker="o", ms=4, color="tab:orange")
        axes[1].set_title("Val Accuracy (federated avg)")
        axes[1].set_xlabel("Round")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True)

    # centralised test accuracy 
    if central and "accuracy" in central[0]:
        rounds = [m["round"] for m in central]
        accs   = [m["accuracy"] for m in central]
        axes[2].plot(rounds, accs, marker="o", ms=4, color="tab:green")
        axes[2].set_title("Test Accuracy (centralised)")
        axes[2].set_xlabel("Round")
        axes[2].set_ylabel("Accuracy")
        axes[2].set_ylim(0, 1)
        axes[2].grid(True)

    plt.tight_layout()
    stem    = Path(metrics_path).stem
    out_png = os.path.join(save_dir, f"curves_{stem}.png")
    fig.savefig(out_png, dpi=150)
    plt.close()
    print(f"Curves saved -> {out_png}")


# Confusion matrix + F1

def evaluate_model(
    model: torch.nn.Module,
    tag: str = "model",
    save_dir: str = OUTPUT_DIR,
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    _, _, test_dataset = get_datasets()
    loader = get_dataloader(test_dataset, BATCH_SIZE, shuffle=False)

    loss, acc, all_preds, all_labels = eval_fn(model, loader, DEVICE)
    f1  = f1_score(all_labels, all_preds, average="binary")
    cm  = confusion_matrix(all_labels, all_preds)

    print(f"\n{'─'*40}")
    print(f"  Model tag  : {tag}")
    print(f"  Test Loss  : {loss:.4f}")
    print(f"  Test Acc   : {acc:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"{'─'*40}")

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Pneumonia"],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {tag}")
    plt.tight_layout()
    out_png = os.path.join(save_dir, f"confusion_{tag}.png")
    fig.savefig(out_png, dpi=150)
    plt.close()
    print(f"Confusion matrix saved -> {out_png}")

    return {"loss": loss, "accuracy": acc, "f1": f1, "confusion_matrix": cm.tolist()}


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics",   type=str, required=True,
                        help="Path to metrics JSON (e.g. outputs/metrics_cnn_iid.json)")
    parser.add_argument("--weights",   type=str, default=None,
                        help="Path to saved .pt model weights for test evaluation")
    parser.add_argument("--model",     choices=["cnn", "resnet"], default="cnn")
    parser.add_argument("--save_dir",  type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    plot_curves(args.metrics, args.save_dir)

    if args.weights:
        model_class = SimpleCNN if args.model == "cnn" else ResNet18Binary
        model       = model_class()
        state       = torch.load(args.weights, map_location=DEVICE)
        model.load_state_dict(state, strict=True)
        model       = model.to(DEVICE)

        tag = Path(args.weights).stem
        evaluate_model(model, tag=tag, save_dir=args.save_dir)
