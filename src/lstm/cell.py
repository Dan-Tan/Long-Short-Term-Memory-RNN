"""Modular, vectorized LSTM Cell implementation."""

import numpy as np
from typing import Tuple, Dict
from src.lstm.activations import Sigmoid, Tanh


class LSTMCell:
    """Single-step Long Short-Term Memory (LSTM) Cell."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        learning_rate: float = 0.01,
    ) -> None:
        """Initialize LSTM cell parameters.

        Args:
            input_dim: Input dimension size.
            hidden_dim: Hidden state / cell state dimension size.
            learning_rate: Learning rate for parameters.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # Xavier / Glorot uniform initialization for combined weights [W_f; W_i; W_c; W_o]
        concat_dim = input_dim + hidden_dim
        limit = np.sqrt(6.0 / (concat_dim + hidden_dim))
        self.W = np.random.uniform(-limit, limit, (4 * hidden_dim, concat_dim)).astype(np.float32)
        self.b = np.zeros((4 * hidden_dim,), dtype=np.float32)

        # Gradient accumulators
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Forward pass for a single time step t.

        Args:
            x: Input array of shape (N, input_dim).
            h_prev: Previous hidden state of shape (N, hidden_dim).
            c_prev: Previous cell state of shape (N, hidden_dim).

        Returns:
            Tuple of (h_next, c_next, cache_dict).
        """
        # Concatenate input and previous hidden state: (N, input_dim + hidden_dim)
        z = np.hstack((x, h_prev))

        # Gate linear projection: (N, 4 * hidden_dim)
        gates = np.dot(z, self.W.T) + self.b

        H = self.hidden_dim
        f_gate = Sigmoid.forward(gates[:, :H])
        i_gate = Sigmoid.forward(gates[:, H:2*H])
        c_tilde = Tanh.forward(gates[:, 2*H:3*H])
        o_gate = Sigmoid.forward(gates[:, 3*H:])

        # Cell state update
        c_next = f_gate * c_prev + i_gate * c_tilde
        tanh_c_next = Tanh.forward(c_next)

        # Hidden state update
        h_next = o_gate * tanh_c_next

        cache = {
            "x": x,
            "h_prev": h_prev,
            "c_prev": c_prev,
            "z": z,
            "f_gate": f_gate,
            "i_gate": i_gate,
            "c_tilde": c_tilde,
            "o_gate": o_gate,
            "c_next": c_next,
            "tanh_c_next": tanh_c_next,
        }

        return h_next, c_next, cache

    def backward(
        self,
        dh_next: np.ndarray,
        dc_next: np.ndarray,
        cache: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass for a single time step t (Backpropagation Through Time).

        Args:
            dh_next: Gradient wrt next hidden state (N, hidden_dim).
            dc_next: Gradient wrt next cell state (N, hidden_dim).
            cache: Cache dictionary saved from forward pass.

        Returns:
            Tuple of (dx, dh_prev, dc_prev).
        """
        z = cache["z"]
        f_gate = cache["f_gate"]
        i_gate = cache["i_gate"]
        c_tilde = cache["c_tilde"]
        o_gate = cache["o_gate"]
        c_prev = cache["c_prev"]
        c_next = cache["c_next"]
        tanh_c_next = cache["tanh_c_next"]

        # Gradient wrt output gate and tanh(c_next)
        do_gate = dh_next * tanh_c_next
        dc_target = dh_next * o_gate * Tanh.derivative(tanh_c_next) + dc_next

        # Gradients wrt individual gates
        df_gate = dc_target * c_prev
        di_gate = dc_target * c_tilde
        dc_tilde = dc_target * i_gate
        dc_prev = dc_target * f_gate

        # Derivatives wrt pre-activation gates
        d_gates_f = df_gate * Sigmoid.derivative(f_gate)
        d_gates_i = di_gate * Sigmoid.derivative(i_gate)
        d_gates_c = dc_tilde * Tanh.derivative(c_tilde)
        d_gates_o = do_gate * Sigmoid.derivative(o_gate)

        # Concatenate gate derivatives: (N, 4 * hidden_dim)
        d_gates = np.hstack((d_gates_f, d_gates_i, d_gates_c, d_gates_o))

        # Weight and bias gradients
        self.dW += np.dot(d_gates.T, z)
        self.db += np.sum(d_gates, axis=0)

        # Gradient wrt concatenated input z = [x, h_prev]
        dz = np.dot(d_gates, self.W)

        dx = dz[:, :self.input_dim]
        dh_prev = dz[:, self.input_dim:]

        return dx, dh_prev, dc_prev

    def update_params(self) -> None:
        """Update weights and zero out gradient accumulators."""
        self.W -= self.learning_rate * self.dW
        self.b -= self.learning_rate * self.db
        self.dW.fill(0.0)
        self.db.fill(0.0)
