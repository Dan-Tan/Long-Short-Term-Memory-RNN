"""Sequential LSTM Model Container for time-series forecasting with residual skip connection."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from src.lstm.layers import LSTMLayer
from src.lstm.dense import DenseLayer
from src.lstm.utils import mean_squared_error


class SequentialLSTM:
    """Sequential LSTM Model container with optional residual skip connection and autocorrelation penalty."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        output_dim: int = 1,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        dropout: float = 0.0,
        grad_clip: float = 5.0,
        use_residual: bool = False,
        autocorr_penalty: float = 0.0,
    ) -> None:
        """Initialize SequentialLSTM container.

        Args:
            input_dim: Feature size per sequence step.
            hidden_dim: LSTM hidden units.
            output_dim: Forecast output dimension.
            learning_rate: Learning rate for parameters.
            weight_decay: L2 regularization coefficient.
            dropout: Recurrent dropout rate (0.0 to 1.0).
            grad_clip: Maximum gradient clipping norm.
            use_residual: Whether to use residual skip connection (adds last step input x_last to forecast).
            autocorr_penalty: Loss penalty weight penalizing residual autocorrelation across mini-batch samples.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual
        self.autocorr_penalty = autocorr_penalty

        self.lstm = LSTMLayer(
            input_dim,
            hidden_dim,
            return_sequences=False,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
        )
        self.dense = DenseLayer(
            hidden_dim,
            output_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
        )

    def set_learning_rate(self, lr: float) -> None:
        """Set learning rate across all sub-layers."""
        self.lstm.set_learning_rate(lr)
        self.dense.set_learning_rate(lr)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through LSTM layer followed by Dense projection and optional Residual Skip.

        Args:
            X: Input tensor of shape (N, T, input_dim).
            training: Whether in training mode.

        Returns:
            Output predictions of shape (N, output_dim).
        """
        h_out = self.lstm.forward(X, training=training)
        residual = self.dense.forward(h_out, training=training)

        if self.use_residual:
            x_last = X[:, -1, : min(self.input_dim, self.output_dim)]
            if x_last.shape[1] < self.output_dim:
                x_last = np.tile(x_last, (1, int(np.ceil(self.output_dim / x_last.shape[1]))))[:, : self.output_dim]
            preds = x_last + residual
            return preds

        return residual

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for input sequences."""
        return self.forward(X, training=False)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate Mean Squared Error (MSE) loss on test dataset."""
        preds = self.predict(X)
        return mean_squared_error(preds, y)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train model using mini-batch stochastic gradient descent.

        Args:
            X_train: Input sequence tensor (N, T, input_dim).
            y_train: Target values (N, output_dim).
            epochs: Training epochs.
            batch_size: Mini-batch size.
            X_val: Optional validation input sequences.
            y_val: Optional validation targets.
            verbose: Print epoch progress metrics.

        Returns:
            Dictionary containing loss records.
        """
        history: Dict[str, List[float]] = {"loss": [], "val_loss": []}
        num_samples = X_train.shape[0]

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0.0
            num_batches = int(np.ceil(num_samples / batch_size))

            for b in range(num_batches):
                start_idx = b * batch_size
                end_idx = min(start_idx + batch_size, num_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                batch_len = end_idx - start_idx

                # Forward pass
                preds = self.forward(X_batch, training=True)

                # Mean Squared Error Loss
                batch_loss = mean_squared_error(preds, y_batch)

                # Loss & Gradient computation with optional Autocorrelation Penalty
                E = preds - y_batch
                if self.autocorr_penalty > 0.0 and batch_len > 1:
                    diff_E = E[1:] - E[:-1]
                    ac_loss = float(np.mean(diff_E ** 2))
                    batch_loss += self.autocorr_penalty * ac_loss

                    dL_dpreds = 2.0 * E / batch_len
                    d_ac = np.zeros_like(E)
                    d_ac[0] = 2.0 * (E[0] - E[1]) / (batch_len - 1)
                    if batch_len > 2:
                        d_ac[1:-1] = 2.0 * (2.0 * E[1:-1] - E[:-2] - E[2:]) / (batch_len - 1)
                    d_ac[-1] = 2.0 * (E[-1] - E[-2]) / (batch_len - 1)
                    dL_dpreds += self.autocorr_penalty * d_ac
                else:
                    dL_dpreds = 2.0 * E / batch_len

                epoch_loss += batch_loss * batch_len

                # Backward pass
                dh_out = self.dense.backward(dL_dpreds)
                self.lstm.backward(dh_out)

            avg_loss = epoch_loss / num_samples
            history["loss"].append(avg_loss)

            val_str = ""
            if X_val is not None and y_val is not None:
                val_loss = self.evaluate(X_val, y_val)
                history["val_loss"].append(val_loss)
                val_str = f" - val_loss: {val_loss:.6f}"

            if verbose:
                print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.6f}{val_str}")

        return history

