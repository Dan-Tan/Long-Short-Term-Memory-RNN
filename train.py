#!/usr/bin/env python3
"""Train LSTM model on synthetic sequence data."""

import os
import multiprocessing

# Automatically utilize CPU cores for parallel matrix operations
_num_cores = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = _num_cores
os.environ["MKL_NUM_THREADS"] = _num_cores
os.environ["OPENBLAS_NUM_THREADS"] = _num_cores

import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.lstm import SequentialLSTM, generate_sine_wave_data, durbin_watson_statistic


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM from scratch on sequence data")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.02, help="Initial learning rate")
    parser.add_argument("--lr-decay", type=float, default=0.95, help="Learning rate decay per epoch")
    parser.add_argument("--hidden-dim", type=int, default=32, help="LSTM hidden units")
    parser.add_argument("--seq-len", type=int, default=20, help="Sequence length")
    parser.add_argument("--save-plot", action="store_true", default=True, help="Save visualization plots")
    return parser.parse_args()


def save_plots(history, model, X_test, y_test):
    os.makedirs("assets", exist_ok=True)
    epochs = len(history["loss"])
    epoch_axis = range(1, epochs + 1)

    # 1. Loss plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epoch_axis, history["loss"], label="Train Loss", color="#2563eb", linewidth=2.0)
    if history["val_loss"]:
        ax.plot(epoch_axis, history["val_loss"], label="Val Loss", color="#dc2626", linewidth=2.0, linestyle="--")

    ax.set_title("LSTM Loss Convergence", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    loss_path = os.path.join("assets", "training_curves.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()

    # 2. Sequence Prediction plot
    preds = model.predict(X_test)
    dw_stat = durbin_watson_statistic(preds, y_test)
    fig, ax = plt.subplots(figsize=(10, 4.5))

    sample_count = min(150, len(y_test))
    ax.plot(range(sample_count), y_test[:sample_count, 0].ravel(), label="Ground Truth", color="#0f172a", linewidth=2.0, alpha=0.85)
    ax.plot(range(sample_count), preds[:sample_count, 0].ravel(), label="LSTM Forecast", color="#2563eb", linewidth=2.0, linestyle="--")

    ax.set_title(f"Sequence Forecast (DW={dw_stat:.3f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Signal Amplitude")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    pred_path = os.path.join("assets", "sequence_predictions.png")
    plt.savefig(pred_path, dpi=300)
    plt.close()


def main():
    args = parse_args()

    print(f"Training LSTM ({args.epochs} epochs, batch size {args.batch_size})...")

    X_train, y_train = generate_sine_wave_data(num_samples=1000, seq_length=args.seq_len)
    X_test, y_test = generate_sine_wave_data(num_samples=200, seq_length=args.seq_len, seed=123)

    model = SequentialLSTM(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        learning_rate=args.lr,
    )

    current_lr = args.lr
    history = {"loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        model.set_learning_rate(current_lr)

        ep_hist = model.fit(
            X_train,
            y_train,
            epochs=1,
            batch_size=args.batch_size,
            X_val=X_test,
            y_val=y_test,
            verbose=False,
        )

        loss = ep_hist["loss"][0]
        val_loss = ep_hist["val_loss"][0] if ep_hist["val_loss"] else 0.0

        history["loss"].append(loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch}/{args.epochs} - loss: {loss:.6f} - val_loss: {val_loss:.6f}")

        current_lr *= args.lr_decay

    final_loss = model.evaluate(X_test, y_test)
    print(f"\nFinal Test MSE Loss: {final_loss:.6f}")

    if args.save_plot:
        save_plots(history, model, X_test, y_test)
        print("Saved plots to assets/")


if __name__ == "__main__":
    main()
