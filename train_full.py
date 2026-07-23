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

from src.lstm import SequentialLSTM, generate_sine_wave_data, mean_squared_error, durbin_watson_statistic


def parse_args():
    parser = argparse.ArgumentParser(description="Full training pipeline with high-resolution plot generation")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default: 200)")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=0.015, help="Learning rate (default: 0.015)")
    parser.add_argument("--lr-decay", type=float, default=0.995, help="Learning rate decay factor (default: 0.995)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization weight decay")
    parser.add_argument("--dropout", type=float, default=0.1, help="Recurrent dropout rate")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Maximum gradient clipping norm")
    parser.add_argument("--hidden-dim", type=int, default=32, help="LSTM hidden units")
    parser.add_argument("--seq-len", type=int, default=20, help="Sequence length")
    # Autocorrelation Mitigation Options
    parser.add_argument("--use-residual", action="store_true", help="Enable residual skip connection for delta prediction")
    parser.add_argument("--stride", type=int, default=5, help="Sampling stride between sequence windows (reduces overlap autocorrelation)")
    parser.add_argument("--forecast-horizon", type=int, default=1, help="Direct multi-step forecast horizon dimension")
    parser.add_argument("--differencing", action="store_true", help="Apply first-differencing to sequence features")
    parser.add_argument("--autocorr-penalty", type=float, default=0.01, help="Autocorrelation loss penalty weight on model residuals")
    return parser.parse_args()


def generate_publication_plots(history, model, X_test, y_test):
    """Generate high-resolution, publication-quality plots."""
    os.makedirs("assets", exist_ok=True)
    epochs = len(history["loss"])
    epoch_axis = range(1, epochs + 1)

    # 1. Training & Validation Loss Plot
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.plot(epoch_axis, history["loss"], label="Train Loss", color="#2563eb", linewidth=2.0)
    if history["val_loss"]:
        ax.plot(epoch_axis, history["val_loss"], label="Val Loss", color="#dc2626", linewidth=2.0, linestyle="--")

    ax.set_title("LSTM Loss Convergence", fontsize=13, fontweight="bold", pad=12, color="#0f172a")
    ax.set_xlabel("Epoch", fontsize=11, fontweight="medium", color="#334155")
    ax.set_ylabel("Loss", fontsize=11, fontweight="medium", color="#334155")
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
    dw_stat = durbin_watson_statistic(preds, y_test)
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)

    sample_count = min(150, len(y_test))
    ax.plot(range(sample_count), y_test[:sample_count, 0].ravel(), label="Ground Truth Signal", color="#0f172a", linewidth=2.0, alpha=0.85)
    ax.plot(range(sample_count), preds[:sample_count, 0].ravel(), label="LSTM Forecast", color="#2563eb", linewidth=2.0, linestyle="--")

    ax.set_title(f"Sequence Forecasting (Durbin-Watson Residual DW={dw_stat:.3f})", fontsize=13, fontweight="bold", pad=12, color="#0f172a")
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
    print(f"Autocorrelation Controls -> Stride: {args.stride} | Horizon: {args.forecast_horizon} | Differencing: {args.differencing} | Residual: {args.use_residual} | Penalty: {args.autocorr_penalty}")
    print("--------------------------------------------------")

    print("Generating synthetic sequence dataset with autocorrelation controls...")
    X_train, y_train = generate_sine_wave_data(
        num_samples=2000,
        seq_length=args.seq_len,
        stride=args.stride,
        forecast_horizon=args.forecast_horizon,
        differencing=args.differencing,
    )
    X_test, y_test = generate_sine_wave_data(
        num_samples=500,
        seq_length=args.seq_len,
        stride=args.stride,
        forecast_horizon=args.forecast_horizon,
        differencing=args.differencing,
        seed=123,
    )

    model = SequentialLSTM(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        output_dim=args.forecast_horizon,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        grad_clip=args.grad_clip,
        use_residual=args.use_residual,
        autocorr_penalty=args.autocorr_penalty,
    )

    history = {
        "loss": [],
        "val_loss": [],
    }

    current_lr = args.lr
    print("\nStarting training loop...")
    for epoch in range(1, args.epochs + 1):
        model.set_learning_rate(current_lr)

        ep_hist = model.fit(
            X_train,
            y_train,
            epochs=1,
            batch_size=args.batch_size,
            X_val=X_test,
            y_val=y_test,
            verbose=(epoch % 20 == 0 or epoch == 1 or epoch == args.epochs),
        )

        history["loss"].append(ep_hist["loss"][0])
        history["val_loss"].append(ep_hist["val_loss"][0])

        current_lr *= args.lr_decay

    final_preds = model.predict(X_test)
    final_loss = model.evaluate(X_test, y_test)
    dw_stat = durbin_watson_statistic(final_preds, y_test)
    print("--------------------------------------------------")
    print(f"Final Test Evaluation MSE: {final_loss:.6f}")
    print(f"Durbin-Watson Residual Autocorrelation Stat: {dw_stat:.4f} (Ideal: ~2.00)")
    print("--------------------------------------------------")

    print("\nGenerating publication-quality plots...")
    generate_publication_plots(history, model, X_test, y_test)
    print("\nTraining and plot generation complete!")


if __name__ == "__main__":
    main()

