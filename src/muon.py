from typing import Tuple

from jax import numpy as jnp


def zeropower_via_newtonschulz5(G, steps: int = 5, eps: float = 1e-7):
    """
    Official Newton-Schulz implementation from Keller Jordan
    Based on: https://github.com/KellerJordan/Muon
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(jnp.bfloat16)

    # Transpose if wide (work with tall matrices)
    if G.shape[-2] > G.shape[-1]:
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (jnp.linalg.norm(X) + eps)

    # Quintic Newton-Schulz iteration adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
    for i in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.shape[-2] > G.shape[-1]:
        X = X.T

    return X.astype(G.dtype)


def muon_update(
    grad: jnp.ndarray,
    momentum: jnp.ndarray,
    beta: float = 0.95,
    steps: int = 5,
    nesterov: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Muon update with momentum and orthogonalization.

    Translation of:
    https://github.com/KellerJordan/Muon/blob/master/muon.py#muon_update

    PyTorch original:
    ```python
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update
    ```

    Args:
        grad: Gradient [any shape, but typically 2D or 4D]
        momentum: Momentum buffer [same shape as grad]
        beta: Momentum coefficient (default: 0.95)
        steps: Newton-Schulz iterations (default: 5)
        nesterov: Use Nesterov momentum (default: True)

    Returns:
        (update, new_momentum): Update to apply and new momentum state
    """
    # Update momentum buffer
    new_momentum = beta * momentum + (1 - beta) * grad

    if nesterov:
        update = (1 - beta) * grad + beta * new_momentum
    else:
        update = new_momentum

    # Handle 4D convolution filters
    original_shape = None
    if update.ndim == 4:
        original_shape = update.shape
        update = update.reshape(update.shape[0], -1)

    # Apply orthogonalization (only for 2D matrices)
    if update.ndim == 2:
        update = zeropower_via_newtonschulz5(update, steps=steps)
        n, m = update.shape[-2], update.shape[-1]
        scale = jnp.sqrt(max(1.0, n / m))
        update = update * scale

    # Restore original shape if we reshaped
    if original_shape is not None:
        update = update.reshape(original_shape)

    return update, new_momentum
