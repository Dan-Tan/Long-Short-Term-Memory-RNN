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

from src.lstm import SequentialLSTM, generate_sine_wave_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM from scratch")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=32, help="LSTM hidden units")
    parser.add_argument("--seq-len", type=int, default=20, help="Sequence length")
    return parser.parse_args()


def main():
    args = parse_args()

    print("==================================================")
    print(" LSTM from Scratch - Training Pipeline ")
    print("==================================================")
    print(f"Hardware Threads: {multiprocessing.cpu_count()} (Multi-Threaded Parallel Execution)")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"Hidden Dimension: {args.hidden_dim} | Sequence Length: {args.seq_len}")
    print("--------------------------------------------------")

    print("Generating sequence dataset...")
    X_train, y_train = generate_sine_wave_data(num_samples=1000, seq_length=args.seq_len)
    X_test, y_test = generate_sine_wave_data(num_samples=200, seq_length=args.seq_len, seed=123)

    model = SequentialLSTM(input_dim=1, hidden_dim=args.hidden_dim, output_dim=1, learning_rate=args.lr)

    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        X_val=X_test,
        y_val=y_test,
        verbose=True,
    )

    final_loss = model.evaluate(X_test, y_test)
    print(f"\nFinal Test MSE Loss: {final_loss:.6f}")


if __name__ == "__main__":
    main()
