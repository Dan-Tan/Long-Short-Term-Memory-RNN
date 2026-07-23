#!/usr/bin/env python3
"""Automated Grid Search & Hyperparameter Optimization for LSTM Time-Series Forecasting."""

import os
import multiprocessing

# Utilize all available CPU cores for parallel matrix operations
_num_cores = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = _num_cores
os.environ["MKL_NUM_THREADS"] = _num_cores
os.environ["OPENBLAS_NUM_THREADS"] = _num_cores

import itertools
import numpy as np
from src.lstm import SequentialLSTM, generate_sine_wave_data, mean_squared_error


def run_hyperparameter_optimization():
    print("==================================================")
    print(" Automated LSTM Hyperparameter Optimization ")
    print("==================================================")
    print(f"Hardware Threads: {multiprocessing.cpu_count()} (Multi-Threaded Parallel Execution)")
    print("--------------------------------------------------")

    # Generate dataset with sequential train / validation split
    X_all, y_all = generate_sine_wave_data(num_samples=2500, seq_length=20, noise_level=0.01)
    split_idx = 2000
    X_train, y_train = X_all[:split_idx], y_all[:split_idx]
    X_val, y_val = X_all[split_idx:], y_all[split_idx:]

    print(f"Dataset Loaded -> Train: {X_train.shape[0]} samples | Val: {X_val.shape[0]} samples")

    # Grid search candidate space
    param_grid = {
        "lr": [0.01, 0.02, 0.03],
        "hidden_dim": [16, 32, 64],
        "batch_size": [16, 32],
        "weight_decay": [0.0, 1e-4],
        "dropout": [0.0, 0.05],
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Total Grid Combination Experiments: {len(combinations)}")
    print("--------------------------------------------------")

    results = []

    for i, params in enumerate(combinations, 1):
        # Set seed for fair evaluation across configurations
        np.random.seed(42)
        model = SequentialLSTM(
            input_dim=1,
            hidden_dim=params["hidden_dim"],
            output_dim=1,
            learning_rate=params["lr"],
            weight_decay=params["weight_decay"],
            dropout=params["dropout"],
            grad_clip=5.0,
        )

        # Train for 5 trial epochs
        model.fit(
            X_train,
            y_train,
            epochs=5,
            batch_size=params["batch_size"],
            verbose=False,
        )

        val_mse = model.evaluate(X_val, y_val)
        params["val_mse"] = val_mse
        results.append(params)

        print(
            f"Exp [{i:02d}/{len(combinations)}] -> LR: {params['lr']:.2f} | "
            f"Hidden: {params['hidden_dim']:02d} | Batch: {params['batch_size']:02d} | "
            f"L2: {params['weight_decay']} | Drop: {params['dropout']} => Val MSE: {val_mse:.6f}"
        )

    # Sort results by validation MSE ascending
    results.sort(key=lambda x: x["val_mse"])
    best = results[0]

    print("\n==================================================")
    print(" TOP 3 HYPERPARAMETER CONFIGURATIONS ")
    print("==================================================")
    for rank in range(min(3, len(results))):
        r = results[rank]
        print(
            f"Rank #{rank+1}: Val MSE = {r['val_mse']:.6f} | LR = {r['lr']} | "
            f"Hidden = {r['hidden_dim']} | Batch = {r['batch_size']} | "
            f"L2 = {r['weight_decay']} | Dropout = {r['dropout']}"
        )

    print("--------------------------------------------------")
    print("OPTIMAL CONFIGURATION FOUND:")
    print(f"  Learning Rate:  {best['lr']}")
    print(f"  Hidden Dim:     {best['hidden_dim']}")
    print(f"  Batch Size:     {best['batch_size']}")
    print(f"  Weight Decay:   {best['weight_decay']}")
    print(f"  Dropout Rate:   {best['dropout']}")
    print("==================================================")

    return best


if __name__ == "__main__":
    run_hyperparameter_optimization()
