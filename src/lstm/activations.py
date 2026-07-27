"""Activation functions and derivatives for LSTM networks."""

import numpy as np


class Sigmoid:
    """Sigmoid activation function: f(z) = 1 / (1 + exp(-z))."""

    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        z_clipped = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    @staticmethod
    def derivative(a: np.ndarray) -> np.ndarray:
        """Derivative wrt activated output a = sigmoid(z)."""
        return a * (1.0 - a)


class Tanh:
    """Hyperbolic Tangent activation function: f(z) = tanh(z)."""

    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    @staticmethod
    def derivative(a: np.ndarray) -> np.ndarray:
        """Derivative wrt activated output a = tanh(z)."""
        return 1.0 - a ** 2
