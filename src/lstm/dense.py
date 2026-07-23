"""Dense Linear Projection Layer for sequence outputs."""

import numpy as np


class DenseLayer:
    """Fully connected linear projection layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        learning_rate: float = 0.01,
    ) -> None:
        """Initialize DenseLayer parameters.

        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            learning_rate: Learning rate for parameters.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.learning_rate = learning_rate

        # Xavier uniform initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features)).astype(np.float32)
        self.b = np.zeros((out_features,), dtype=np.float32)

        self.x: np.ndarray = np.array([])

    def set_learning_rate(self, lr: float) -> None:
        """Set learning rate for parameters."""
        self.learning_rate = lr

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass Y = X * W + b.

        Args:
            x: Input array of shape (N, in_features).
            training: Whether layer is in training mode.

        Returns:
            Output array of shape (N, out_features).
        """
        if training:
            self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """Backward pass wrt linear projection layer.

        Args:
            dL_dout: Gradient wrt output (N, out_features).

        Returns:
            Gradient wrt input (N, in_features).
        """
        N = self.x.shape[0]
        dL_dW = np.dot(self.x.T, dL_dout) / N
        dL_db = np.sum(dL_dout, axis=0) / N

        dL_dx = np.dot(dL_dout, self.W.T)

        self.W -= self.learning_rate * dL_dW
        self.b -= self.learning_rate * dL_db

        return dL_dx
