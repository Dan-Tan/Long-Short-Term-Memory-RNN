"""Unit tests for numerical gradient checking across LSTM cells and layers."""

import pytest
import numpy as np
from src.lstm.cell import LSTMCell
from src.lstm.layers import LSTMLayer
from src.lstm.dense import DenseLayer
from src.lstm.grad_check import eval_numerical_gradient, compute_relative_error


def test_lstm_cell_gradient_check():
    """Verify analytical vs numerical gradient for LSTM cell weight matrix W."""
    np.random.seed(42)
    N, input_dim, hidden_dim = 2, 3, 4

    cell = LSTMCell(input_dim, hidden_dim)
    cell.W = cell.W.astype(np.float64)
    cell.b = cell.b.astype(np.float64)

    x = np.random.randn(N, input_dim).astype(np.float64)
    h_prev = np.random.randn(N, hidden_dim).astype(np.float64)
    c_prev = np.random.randn(N, hidden_dim).astype(np.float64)

    def loss_func(W_param):
        orig_W = cell.W
        cell.W = W_param
        h_next, c_next, _ = cell.forward(x, h_prev, c_prev)
        loss = float(np.sum(h_next ** 2) + np.sum(c_next ** 2))
        cell.W = orig_W
        return loss

    W_init = cell.W.copy()

    h_next, c_next, cache = cell.forward(x, h_prev, c_prev)
    dh_next = 2.0 * h_next
    dc_next = 2.0 * c_next

    cell.backward(dh_next, dc_next, cache)
    analytical_dW = cell.dW.copy()

    cell.W = W_init.copy()
    numerical_dW = eval_numerical_gradient(loss_func, W_init.copy(), h=1e-6)

    error = compute_relative_error(analytical_dW, numerical_dW)
    assert error < 1e-4, f"LSTMCell weight relative error too high: {error}"


def test_dense_layer_gradient_check():
    """Verify analytical vs numerical gradient for Dense layer weights."""
    np.random.seed(42)
    N, in_f, out_f = 3, 5, 2

    dense = DenseLayer(in_f, out_f)
    dense.W = dense.W.astype(np.float64)
    dense.b = dense.b.astype(np.float64)

    x = np.random.randn(N, in_f).astype(np.float64)

    def loss_func(W_param):
        orig_W = dense.W
        dense.W = W_param
        out = dense.forward(x, training=False)
        loss = float(np.sum(out ** 2))
        dense.W = orig_W
        return loss

    W_init = dense.W.copy()

    out = dense.forward(x, training=True)
    dL_dout = 2.0 * out

    dense.backward(dL_dout)
    analytical_dW = np.dot(x.T, dL_dout)

    dense.W = W_init.copy()
    numerical_dW = eval_numerical_gradient(loss_func, W_init.copy(), h=1e-6)

    error = compute_relative_error(analytical_dW, numerical_dW)
    assert error < 1e-2, f"DenseLayer weight relative error too high: {error}"
