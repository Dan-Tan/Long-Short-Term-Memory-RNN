# Long Short-Term Memory (LSTM) RNN from Scratch in NumPy

A 2D Long Short-Term Memory (LSTM) Recurrent Neural Network framework implemented from first principles in NumPy, without relying on PyTorch or TensorFlow for model operations or automatic differentiation.

This project was originally created upon finishing high school as an early programming project to learn Python and understand recurrent neural network mathematics and Backpropagation Through Time (BPTT). The original unedited state is tagged as [`v0.1.0-legacy`](https://github.com/Dan-Tan/Long-Short-Term-Memory-RNN/tree/v0.1.0-legacy) (commit [`771d128`](https://github.com/Dan-Tan/Long-Short-Term-Memory-RNN/commit/771d1287c8052fe2ae764ad802c67fe9bb670b3e)) and preserved in [legacy/original_lstm.py](legacy/original_lstm.py). The repository contains both that legacy script and a modernized refactor with modular layer abstractions, type annotations, unit tests, and fused matrix gate acceleration.

---

## Visualizations

### Sequence Forecast Predictions
Comparing ground truth signals with LSTM sequence predictions:

![Sequence Forecast Predictions](assets/sequence_predictions.png?v=1.1)

Achieves MSE loss < 0.01 on sequence prediction.

---

## Quickstart & Usage

This project uses `uv` for dependency management.

1. Install dependencies:
```bash
uv sync
```

2. Run the train script:
```bash
uv run train.py
```
