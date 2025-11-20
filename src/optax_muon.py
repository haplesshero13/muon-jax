"""
Muon Optax Optimizer

Based on KellerJordan's Reference Implementation:
https://github.com/KellerJordan/Muon/blob/master/muon.py

"""

from typing import Any, NamedTuple

import jax
import optax
from jax import numpy as jnp
from optax.transforms import MaskedNode

from .muon import muon_update


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


def muon_with_adam(
    muon_params: set[str],
    muon_lr: float = 0.05,
    muon_momentum: float = 0.95,
    adam_lr: float = 3e-4,
    adam_betas: tuple = (0.9, 0.95),
    adam_eps: float = 1e-10,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """
    Specify which params get Muon, the rest get Adam.

    Args:
        muon_params: Set of parameter names/paths that should use Muon
        muon_lr: Learning rate for Muon parameters
        muon_momentum: Momentum for Muon
        adam_lr: Learning rate for Adam parameters
        adam_betas: Beta coefficients for Adam
        adam_eps: Epsilon for Adam
        weight_decay: Weight decay for all parameters

    Example:
    ```python
            # Identify matrix parameters for Muon in a Set
            muon_param_names = {
                'layer1.weight', 'layer2.weight', 'layer3.weight'
            }

            optimizer = muon_with_adam(
                muon_params=muon_param_names,
                muon_lr=0.05,
                adam_lr=0.0003,
                weight_decay=0.01,
            )
    ```
    """

    def label_fn(path: set[str]):
        """Determine if parameter uses Muon or Adam."""
        # Convert pytree path to string
        path_str = ".".join(path)
        return "muon" if path_str in muon_params else "adam"

    muon_transform = optax.chain(
        muon(learning_rate=1.0, momentum=muon_momentum),
        optax.add_decayed_weights(weight_decay)
        if weight_decay > 0
        else optax.identity(),
        optax.scale(-muon_lr),
    )

    adam_transform = optax.chain(
        optax.scale_by_adam(b1=adam_betas[0], b2=adam_betas[1], eps=adam_eps),
        optax.add_decayed_weights(weight_decay)
        if weight_decay > 0
        else optax.identity(),
        optax.scale(-adam_lr),
    )

    return optax.multi_transform(
        {"muon": muon_transform, "adam": adam_transform}, label_fn
    )
