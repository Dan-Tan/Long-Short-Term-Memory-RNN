#!/usr/bin/env python3
"""Extended Training & High-Quality Plot Generator for LSTM from Scratch."""

import os
import multiprocessing

# Automatically utilize all CPU cores for parallel matrix operations
_num_cores = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = _num_cores
os.environ["MKL_NUM_THREADS"] = _num_cores
os.environ["OPENBLAS_NUM_THREADS"] = _num_cores

import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.lstm import SequentialLSTM, generate_sine_wave_data, mean_squared_error


def parse_args():
    parser = argparse.ArgumentParser(description="Full training pipeline with high-resolution plot generation")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.03, help="Initial learning rate (default: 0.03)")
    parser.add_argument("--lr-decay", type=float, default=0.95, help="Learning rate decay factor")
    parser.add_argument("--hidden-dim", type=int, default=32, help="LSTM hidden units")
    parser.add_argument("--seq-len", type=int, default=20, help="Sequence length")
    return parser.parse_args()


def generate_publication_plots(history, model, X_test, y_test):
    """Generate high-resolution, publication-quality plots."""
    os.makedirs("assets", exist_ok=True)
    epochs = len(history["loss"])
    epoch_axis = range(1, epochs + 1)

    # 1. Training & Validation Loss Plot
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.plot(epoch_axis, history["loss"], label="Train MSE Loss", color="#2563eb", linewidth=2.5, marker="o", markersize=5)
    if history["val_loss"]:
        ax.plot(epoch_axis, history["val_loss"], label="Val MSE Loss", color="#dc2626", linewidth=2.2, linestyle="--", marker="s", markersize=5)

    ax.set_title("LSTM Loss Convergence (Mean Squared Error)", fontsize=13, fontweight="bold", pad=12, color="#0f172a")
    ax.set_xlabel("Epoch", fontsize=11, fontweight="medium", color="#334155")
    ax.set_ylabel("MSE Loss", fontsize=11, fontweight="medium", color="#334155")
    ax.set_xticks(epoch_axis)
    ax.grid(True, linestyle=":", alpha=0.6, color="#94a3b8")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=10)
    ax.set_facecolor("#f8fafc")

    plt.tight_layout()
    loss_path = os.path.join("assets", "training_curves.png")
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved publication-quality training curves to: {loss_path}")

    # 2. Sequence Prediction Comparison Plot
    preds = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)

    sample_count = min(150, len(y_test))
    ax.plot(range(sample_count), y_test[:sample_count].ravel(), label="Ground Truth Signal", color="#0f172a", linewidth=2.0, alpha=0.85)
    ax.plot(range(sample_count), preds[:sample_count].ravel(), label="LSTM Predicted Forecast", color="#2563eb", linewidth=2.0, linestyle="--")

    ax.set_title("Sequence Forecasting: Ground Truth vs LSTM Predictions", fontsize=13, fontweight="bold", pad=12, color="#0f172a")
    ax.set_xlabel("Time Step / Test Sample", fontsize=11, fontweight="medium", color="#334155")
    ax.set_ylabel("Signal Amplitude", fontsize=11, fontweight="medium", color="#334155")
    ax.grid(True, linestyle=":", alpha=0.6, color="#94a3b8")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=10)
    ax.set_facecolor("#f8fafc")

    plt.tight_layout()
    pred_path = os.path.join("assets", "sequence_predictions.png")
    plt.savefig(pred_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved sequence forecast predictions plot to: {pred_path}")


def main():
    args = parse_args()

    print("==================================================")
    print(" LSTM Extended Training & Plot Generator ")
    print("==================================================")
    print(f"Hardware Threads: {multiprocessing.cpu_count()} (Multi-Threaded Parallel Execution)")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Initial LR: {args.lr}")
    print(f"Hidden Dimension: {args.hidden_dim} | Sequence Length: {args.seq_len}")
    print("--------------------------------------------------")

    print("Generating synthetic sequence dataset...")
    X_train, y_train = generate_sine_wave_data(num_samples=2000, seq_length=args.seq_len)
    X_test, y_test = generate_sine_wave_data(num_samples=500, seq_length=args.seq_len, seed=123)

    model = SequentialLSTM(input_dim=1, hidden_dim=args.hidden_dim, output_dim=1, learning_rate=args.lr)

    history = {
        "loss": [],
        "val_loss": [],
    }

    current_lr = args.lr
    print("\nStarting training loop...")
    for epoch in range(1, args.epochs + 1):
        print(f"--- Epoch {epoch}/{args.epochs} (Learning Rate: {current_lr:.5f}) ---")
        model.set_learning_rate(current_lr)

        ep_hist = model.fit(
            X_train,
            y_train,
            epochs=1,
            batch_size=args.batch_size,
            X_val=X_test,
            y_val=y_test,
            verbose=True,
        )

        history["loss"].append(ep_hist["loss"][0])
        history["val_loss"].append(ep_hist["val_loss"][0])

        # Apply learning rate decay
        current_lr *= args.lr_decay

    final_loss = model.evaluate(X_test, y_test)
    print("--------------------------------------------------")
    print(f"Final Test Evaluation MSE: {final_loss:.6f}")
    print("--------------------------------------------------")

    print("\nGenerating publication-quality plots...")
    generate_publication_plots(history, model, X_test, y_test)
    print("\nTraining and plot generation complete!")


if __name__ == "__main__":
    main()
