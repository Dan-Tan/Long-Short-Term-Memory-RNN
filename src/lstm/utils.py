"""Dataset generators and utilities for time-series sequence tasks."""

import numpy as np
from typing import Tuple


def generate_sine_wave_data(
    num_samples: int = 1000,
    seq_length: int = 20,
    noise_level: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic sine wave sequence dataset for time-series forecasting.

    Args:
        num_samples: Number of sequence samples to generate.
        seq_length: Number of time steps per input sequence.
        noise_level: Standard deviation of Gaussian noise added to signals.
        seed: Random seed for reproducibility.

    Returns:
        X: Input array of shape (num_samples, seq_length, 1).
        y: Target array of shape (num_samples, 1).
    """
    np.random.seed(seed)
    total_steps = num_samples + seq_length + 10
    t = np.linspace(0, 50 * np.pi, total_steps)
    signal = np.sin(t) + 0.5 * np.sin(2.5 * t)
    if noise_level > 0.0:
        signal += np.random.normal(0, noise_level, size=signal.shape)

    X, y = [], []
    for i in range(num_samples):
        X.append(signal[i : i + seq_length])
        y.append(signal[i + seq_length])

    X_arr = np.array(X, dtype=np.float32)[..., np.newaxis]
    y_arr = np.array(y, dtype=np.float32)[..., np.newaxis]
    return X_arr, y_arr


def mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate Mean Squared Error (MSE)."""
    return float(np.mean((y_pred - y_true) ** 2))
