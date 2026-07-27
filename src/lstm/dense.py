"""Dense Linear Projection Layer with L2 regularization and gradient clipping."""

import numpy as np


class DenseLayer:
    """Fully connected linear projection layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        grad_clip: float = 5.0,
    ) -> None:
        """Initialize DenseLayer parameters.

        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            learning_rate: Learning rate for parameters.
            weight_decay: L2 regularization coefficient.
            grad_clip: Maximum gradient norm for clipping.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

        # Xavier uniform initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features)).astype(np.float32)
        self.b = np.zeros((out_features,), dtype=np.float32)

        self.x: np.ndarray = np.array([])
        self.dW: np.ndarray = np.array([])
        self.db: np.ndarray = np.array([])

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
        """Backward pass with L2 regularization and gradient clipping.

        Args:
            dL_dout: Gradient wrt output (N, out_features).

        Returns:
            Gradient wrt input (N, in_features).
        """
        N = self.x.shape[0]
        self.dW = np.dot(self.x.T, dL_dout)
        self.db = np.sum(dL_dout, axis=0)

        dL_dW = (self.dW / N) + self.weight_decay * self.W
        dL_db = self.db / N

        if self.grad_clip > 0.0:
            dL_dW = np.clip(dL_dW, -self.grad_clip, self.grad_clip)
            dL_db = np.clip(dL_db, -self.grad_clip, self.grad_clip)

        dL_dx = np.dot(dL_dout, self.W.T)

        self.W -= self.learning_rate * dL_dW
        self.b -= self.learning_rate * dL_db

        return dL_dx
