"""Unit tests for LSTM cell, layer, and model."""

import pytest
import numpy as np
from src.lstm.cell import LSTMCell
from src.lstm.layers import LSTMLayer
from src.lstm.model import SequentialLSTM
from src.lstm.utils import generate_sine_wave_data, mean_squared_error


def test_lstm_cell_forward_backward():
    batch_size = 4
    input_dim = 2
    hidden_dim = 8

    cell = LSTMCell(input_dim, hidden_dim)
    x = np.random.randn(batch_size, input_dim).astype(np.float32)
    h_prev = np.random.randn(batch_size, hidden_dim).astype(np.float32)
    c_prev = np.random.randn(batch_size, hidden_dim).astype(np.float32)

    h_next, c_next, cache = cell.forward(x, h_prev, c_prev)
    assert h_next.shape == (batch_size, hidden_dim)
    assert c_next.shape == (batch_size, hidden_dim)

    dh_next = np.random.randn(batch_size, hidden_dim).astype(np.float32)
    dc_next = np.random.randn(batch_size, hidden_dim).astype(np.float32)

    dx, dh_prev, dc_prev = cell.backward(dh_next, dc_next, cache)
    assert dx.shape == x.shape
    assert dh_prev.shape == h_prev.shape
    assert dc_prev.shape == c_prev.shape


def test_lstm_layer_sequence():
    batch_size = 4
    seq_len = 10
    input_dim = 1
    hidden_dim = 16

    layer_seq = LSTMLayer(input_dim, hidden_dim, return_sequences=True)
    layer_last = LSTMLayer(input_dim, hidden_dim, return_sequences=False)

    X = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)

    out_seq = layer_seq.forward(X, training=True)
    assert out_seq.shape == (batch_size, seq_len, hidden_dim)

    out_last = layer_last.forward(X, training=True)
    assert out_last.shape == (batch_size, hidden_dim)


def test_sequential_lstm_end_to_end():
    X_train, y_train = generate_sine_wave_data(num_samples=100, seq_length=15)
    model = SequentialLSTM(input_dim=1, hidden_dim=16, output_dim=1, learning_rate=0.01)

    history = model.fit(X_train, y_train, epochs=2, batch_size=16, verbose=False)
    assert len(history["loss"]) == 2

    preds = model.predict(X_train[:5])
    assert preds.shape == (5, 1)

    mse = model.evaluate(X_train, y_train)
    assert isinstance(mse, float)
