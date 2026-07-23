#!/usr/bin/env python3
"""Train LSTM model on synthetic sequence data."""

import os
import multiprocessing

# Automatically utilize all CPU cores for parallel matrix operations
_num_cores = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = _num_cores
os.environ["MKL_NUM_THREADS"] = _num_cores
os.environ["OPENBLAS_NUM_THREADS"] = _num_cores

import argparse
import numpy as np

from src.lstm import SequentialLSTM, generate_sine_wave_data, durbin_watson_statistic


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM from scratch with autocorrelation mitigations")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=32, help="LSTM hidden units")
    parser.add_argument("--seq-len", type=int, default=20, help="Sequence length")
    # Autocorrelation Mitigation Options
    parser.add_argument("--use-residual", action="store_true", help="Enable residual skip connection for delta prediction")
    parser.add_argument("--stride", type=int, default=5, help="Sampling stride between sequence windows (reduces overlap autocorrelation)")
    parser.add_argument("--forecast-horizon", type=int, default=1, help="Direct multi-step forecast horizon dimension")
    parser.add_argument("--differencing", action="store_true", help="Apply first-differencing to sequence features")
    parser.add_argument("--autocorr-penalty", type=float, default=0.01, help="Autocorrelation loss penalty weight on model residuals")
    return parser.parse_args()


def main():
    args = parse_args()

    print("==================================================")
    print(" LSTM from Scratch - Training Pipeline ")
    print("==================================================")
    print(f"Hardware Threads: {multiprocessing.cpu_count()} (Multi-Threaded Parallel Execution)")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"Hidden Dimension: {args.hidden_dim} | Sequence Length: {args.seq_len}")
    print(f"Autocorrelation Controls -> Stride: {args.stride} | Horizon: {args.forecast_horizon} | Differencing: {args.differencing} | Residual: {args.use_residual} | Penalty: {args.autocorr_penalty}")
    print("--------------------------------------------------")

    print("Generating sequence dataset with autocorrelation controls...")
    X_train, y_train = generate_sine_wave_data(
        num_samples=1000,
        seq_length=args.seq_len,
        stride=args.stride,
        forecast_horizon=args.forecast_horizon,
        differencing=args.differencing,
    )
    X_test, y_test = generate_sine_wave_data(
        num_samples=200,
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
        use_residual=args.use_residual,
        autocorr_penalty=args.autocorr_penalty,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        X_val=X_test,
        y_val=y_test,
        verbose=True,
    )

    preds_test = model.predict(X_test)
    final_loss = model.evaluate(X_test, y_test)
    dw_stat = durbin_watson_statistic(preds_test, y_test)

    print("--------------------------------------------------")
    print(f"Final Test MSE Loss: {final_loss:.6f}")
    print(f"Durbin-Watson Residual Autocorrelation Stat: {dw_stat:.4f} (Ideal: ~2.00)")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()

