"""Recurrent LSTM sequence layer with dropout support."""

import numpy as np
from typing import Tuple, List, Dict, Union
from src.lstm.cell import LSTMCell


class LSTMLayer:
    """LSTM Layer for sequence processing across multiple time steps."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        return_sequences: bool = False,
        dropout: float = 0.0,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        grad_clip: float = 5.0,
    ) -> None:
        """Initialize LSTMLayer.

        Args:
            input_dim: Input dimension per step.
            hidden_dim: Number of hidden units.
            return_sequences: Whether to return full output sequence or final hidden state.
            dropout: Dropout probability (0.0 to 1.0) applied to hidden states.
            learning_rate: Learning rate for parameters.
            weight_decay: L2 regularization coefficient.
            grad_clip: Maximum gradient norm for clipping.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences
        self.dropout = dropout
        self.cell = LSTMCell(
            input_dim,
            hidden_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
        )

        # Cache for backpropagation through time
        self.caches: List[Dict[str, np.ndarray]] = []
        self.h_states: List[np.ndarray] = []
        self.c_states: List[np.ndarray] = []
        self.dropout_mask: np.ndarray = np.array([])

    def set_learning_rate(self, lr: float) -> None:
        """Set learning rate for the underlying cell."""
        self.cell.learning_rate = lr

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass across sequence steps t=0...T-1.

        Args:
            X: Input tensor of shape (N, T, input_dim).
            training: Whether layer is in training mode.

        Returns:
            Output tensor of shape (N, T, hidden_dim) if return_sequences else (N, hidden_dim).
        """
        N, T, D = X.shape
        h_curr = np.zeros((N, self.hidden_dim), dtype=np.float32)
        c_curr = np.zeros((N, self.hidden_dim), dtype=np.float32)

        caches = []
        h_states = [h_curr]
        c_states = [c_curr]

        outputs = []
        for t in range(T):
            x_t = X[:, t, :]
            h_curr, c_curr, cache = self.cell.forward(x_t, h_curr, c_curr)
            caches.append(cache)
            h_states.append(h_curr)
            c_states.append(c_curr)
            outputs.append(h_curr)

        out_arr = np.stack(outputs, axis=1)  # (N, T, hidden_dim)

        # Inverted dropout regularization during training
        mask = np.ones_like(out_arr)
        if training and self.dropout > 0.0:
            prob = 1.0 - self.dropout
            mask = (np.random.rand(*out_arr.shape) < prob) / prob
            out_arr = out_arr * mask

        if training:
            self.caches = caches
            self.h_states = h_states
            self.c_states = c_states
            self.dropout_mask = mask

        if self.return_sequences:
            return out_arr
        return out_arr[:, -1, :]  # (N, hidden_dim)

    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """Backpropagation Through Time (BPTT).

        Args:
            dL_dout: Gradient wrt layer output. Shape (N, T, hidden_dim) or (N, hidden_dim).

        Returns:
            Gradient wrt input sequence (N, T, input_dim).
        """
        T = len(self.caches)
        N = dL_dout.shape[0]

        if not self.return_sequences:
            dL_dout_seq = np.zeros((N, T, self.hidden_dim), dtype=np.float32)
            dL_dout_seq[:, -1, :] = dL_dout
        else:
            dL_dout_seq = dL_dout

        if self.dropout > 0.0 and self.dropout_mask.size > 0:
            dL_dout_seq = dL_dout_seq * self.dropout_mask

        dX = np.zeros((N, T, self.input_dim), dtype=np.float32)
        dh_next = np.zeros((N, self.hidden_dim), dtype=np.float32)
        dc_next = np.zeros((N, self.hidden_dim), dtype=np.float32)

        for t in reversed(range(T)):
            dh_t = dL_dout_seq[:, t, :] + dh_next
            dx_t, dh_next, dc_next = self.cell.backward(dh_t, dc_next, self.caches[t])
            dX[:, t, :] = dx_t

        self.cell.update_params(batch_size=N)
        return dX
