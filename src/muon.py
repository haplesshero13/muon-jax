from typing import Any, NamedTuple, Tuple

import jax
import optax
from jax import numpy as jnp
from optax.transforms import MaskedNode


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


class MuonState(NamedTuple):
    """State for Muon optimizer."""

    momentum: Any  # PyTree of momentum buffers
    count: jnp.ndarray  # Step counter


def muon(
    learning_rate: float,
    momentum: float = 0.95,
    nesterov: bool = True,
    steps: int = 5,
) -> optax.GradientTransformation:
    """
    Muon optimizer as an Optax GradientTransformation.

    Applies orthogonalized momentum updates to matrix parameters (ndim >= 2)
    and standard momentum to vector/scalar parameters.

    Args:
        learning_rate: Learning rate (typically 0.01-0.02 for Muon)
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        steps: Newton-Schulz iteration steps (default: 5)

    Returns:
        An (init_fn, update_fn) tuple for Optax

    Example:
        >>> optimizer = muon(learning_rate=0.02, momentum=0.95)
        >>> opt_state = optimizer.init(params)
        >>> updates, opt_state = optimizer.update(grads, opt_state, params)
        >>> params = optax.apply_updates(params, updates)
    """

    def init_fn(params):
        """Initialize optimizer state."""

        def is_masked(x):
            return isinstance(x, MaskedNode)

        # Create momentum buffers, preserving MaskedNode for skipped params
        def init_momentum(p):
            if is_masked(p):
                return p  # Keep MaskedNode as-is
            return jnp.zeros_like(p)

        momentum_buffers = jax.tree_util.tree_map(
            init_momentum,
            params,
            is_leaf=is_masked,  # Treat MaskedNode as leaf
        )

        return MuonState(
            momentum=momentum_buffers, count=jnp.zeros([], dtype=jnp.int32)
        )

    def update_fn(updates, state, params=None):
        """
        Apply Muon updates.

        Args:
            updates: Gradients (PyTree)
            state: Optimizer state
            params: Optional parameters (unused but part of Optax API)

        Returns:
            (new_updates, new_state) tuple
        """
        del params  # Unused

        def is_masked(x):
            return isinstance(x, MaskedNode)

        # Apply muon_update to get new updates (skip masked nodes)
        def apply_update(g, m):
            if is_masked(g):
                return g  # Return masked node as-is
            return muon_update(g, m, beta=momentum, steps=steps, nesterov=nesterov)[0]

        new_updates = jax.tree.map(
            apply_update, updates, state.momentum, is_leaf=is_masked
        )

        def apply_momentum(g, m):
            if is_masked(g):
                return m  # Keep old momentum for masked params
            return muon_update(g, m, beta=momentum, steps=steps, nesterov=nesterov)[1]

        new_momentum = jax.tree.map(
            apply_momentum, updates, state.momentum, is_leaf=is_masked
        )

        new_state = MuonState(momentum=new_momentum, count=state.count + 1)

        return new_updates, new_state

    # Chain with learning rate scaling
    return optax.chain(
        optax.GradientTransformation(init_fn, update_fn), optax.scale(-learning_rate)
    )
