"""Numerical Gradient Checking module for verifying analytical backpropagation gradients."""

import numpy as np
from typing import Callable


def eval_numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    """Compute numerical gradient of scalar function f wrt array x using centered finite differences.

    Args:
        f: Function taking array x and returning a scalar loss.
        x: Input parameter array.
        h: Finite difference perturbation step size.

    Returns:
        Numerical gradient array of the same shape as x.
    """
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]

        x[idx] = old_val + h
        pos_loss = f(x)

        x[idx] = old_val - h
        neg_loss = f(x)

        x[idx] = old_val

        grad[idx] = (pos_loss - neg_loss) / (2.0 * h)
        it.iternext()

    return grad.astype(x.dtype)


def compute_relative_error(
    grad_analytical: np.ndarray,
    grad_numerical: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Compute relative error norm between analytical and numerical gradients.

    Args:
        grad_analytical: Gradient computed via backpropagation.
        grad_numerical: Gradient computed via finite differences.
        eps: Small epsilon to prevent division by zero.

    Returns:
        Relative error scalar.
    """
    diff_norm = np.linalg.norm(grad_analytical - grad_numerical)
    denom = np.linalg.norm(grad_analytical) + np.linalg.norm(grad_numerical) + eps
    return float(diff_norm / denom)
