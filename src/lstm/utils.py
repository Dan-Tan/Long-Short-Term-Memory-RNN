"""Dataset generators and utilities for time-series sequence tasks."""

import numpy as np
from typing import Tuple


def generate_sine_wave_data(
    num_samples: int = 1000,
    seq_length: int = 20,
    stride: int = 1,
    forecast_horizon: int = 1,
    differencing: bool = False,
    noise_level: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic sine wave sequence dataset with autocorrelation controls.

    Args:
        num_samples: Number of sequence samples to generate.
        seq_length: Number of time steps per input sequence.
        stride: Step size between consecutive sequence windows (reduces overlap autocorrelation).
        forecast_horizon: Number of future time steps to predict (direct multi-step forecasting).
        differencing: Whether to apply first-differencing to sequence features.
        noise_level: Standard deviation of Gaussian noise added to signals.
        seed: Random seed for reproducibility.

    Returns:
        X: Input array of shape (num_samples, seq_length, 1).
        y: Target array of shape (num_samples, forecast_horizon).
    """
    np.random.seed(seed)
    total_steps = (num_samples - 1) * stride + seq_length + forecast_horizon + 20
    t = np.linspace(0, 50 * np.pi * (total_steps / 1000.0), total_steps)
    signal = np.sin(t) + 0.5 * np.sin(2.5 * t)
    if noise_level > 0.0:
        signal += np.random.normal(0, noise_level, size=signal.shape)

    X, y = [], []
    for k in range(num_samples):
        i = k * stride
        seq_x = signal[i : i + seq_length].copy()
        if differencing:
            seq_x = np.diff(seq_x, prepend=seq_x[0])

        if forecast_horizon == 1:
            seq_y = signal[i + seq_length]
        else:
            seq_y = signal[i + seq_length : i + seq_length + forecast_horizon]

        X.append(seq_x)
        y.append(seq_y)

    X_arr = np.array(X, dtype=np.float32)[..., np.newaxis]
    y_arr = np.array(y, dtype=np.float32)
    if forecast_horizon == 1:
        y_arr = y_arr[..., np.newaxis]
    return X_arr, y_arr


def mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate Mean Squared Error (MSE)."""
    return float(np.mean((y_pred - y_true) ** 2))


def durbin_watson_statistic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate Durbin-Watson statistic for model residuals (y_pred - y_true).

    DW ≈ 2.0 indicates zero residual autocorrelation (white noise).
    DW < 2.0 indicates positive autocorrelation.
    DW > 2.0 indicates negative autocorrelation.
    """
    residuals = (y_pred - y_true).ravel()
    if len(residuals) <= 1:
        return 2.0
    diff_sq = np.sum(np.diff(residuals) ** 2)
    rss = np.sum(residuals ** 2)
    if rss == 0:
        return 2.0
    return float(diff_sq / rss)

