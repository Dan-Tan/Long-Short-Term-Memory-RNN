# Legacy Implementation (High School Project)

This directory preserves the unedited Python script (`original_lstm.py`) written upon finishing high school as one of my first hands-on programming projects exploring deep learning mathematics from first principles.

The exact original commit prior to codebase refactoring is tagged as [`v0.1.0-legacy`](https://github.com/Dan-Tan/Long-Short-Term-Memory-RNN/tree/v0.1.0-legacy) (commit [`771d128`](https://github.com/Dan-Tan/Long-Short-Term-Memory-RNN/commit/771d1287c8052fe2ae764ad802c67fe9bb670b3e)).

## Background & Provenance
* **Original File**: Exported from a Google Colaboratory notebook (`LSTM.ipynb`).
* **Framework Constraints**: Written entirely from scratch using raw **NumPy** matrix operations, without relying on PyTorch or TensorFlow for model execution or automatic differentiation.
* **Early Milestones**:
  * Formulated manual 4-gate LSTM cell equations (Forget gate $f_t$, Input gate $i_t$, Candidate cell state $\tilde{C}_t$, Output gate $o_t$).
  * Derived manual Backpropagation Through Time (BPTT) equations across temporal steps.
  * Implemented custom numerical gradient checking verification scripts.

---
*The original script is preserved unchanged in `original_lstm.py` as well as in the git tag `v0.1.0-legacy`.*
