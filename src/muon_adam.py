"""
Muon Hybrid Optimizer - Muon for matrices, AdamW for everything else

Based on KellerJordan's MuonWithAuxAdam:
https://github.com/KellerJordan/Muon/blob/master/muon.py

Installation:
    pip install pytest jax jaxlib optax

Usage:
    pytest test_muon_hybrid.py -v
"""

from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import optax

from .muon import muon


def create_param_labels(
    params: Any, muon_params_fn: Optional[Callable[[str, Any], bool]] = None
) -> Dict[str, str]:
    """
    Label each parameter as 'muon' or 'adam'.

    Default strategy (matches KellerJordan's approach):
    - Muon: Hidden layer matrices (ndim >= 2, not embed, not lm_head)
    - Adam: Everything else (embeddings, lm_head, scalars, biases)

    Args:
        params: Parameter pytree
        muon_params_fn: Optional custom function(path_str, param) -> bool
                       If True, use Muon; if False, use Adam

    Returns:
        Dict mapping parameter paths to 'muon' or 'adam'
    """

    def default_labeling(path_str: str, param: jnp.ndarray) -> str:
        """
        Default labeling strategy:
        - Use Muon for hidden layer matrices (2D+, not special layers)
        - Use Adam for embeddings, lm_head, and scalars/biases
        """
        # Embeddings → Adam
        if "embed" in path_str.lower():
            return "adam"

        # LM head / output layer → Adam
        if "lm_head" in path_str.lower() or "output" in path_str.lower():
            return "adam"

        # Scalars and vectors (biases, layernorms) → Adam
        if param.ndim < 2:
            return "adam"

        # Hidden layer matrices → Muon
        return "muon"

    # Flatten params to get paths
    flat_params = jax.tree_util.tree_flatten_with_path(params)[0]

    labels = {}
    for path, param in flat_params:
        # Convert path to string
        path_str = "/".join(str(p.key) for p in path)

        # Apply custom or default labeling
        if muon_params_fn is not None:
            use_muon = muon_params_fn(path_str, param)
            label = "muon" if use_muon else "adam"
        else:
            label = default_labeling(path_str, param)

        labels[path_str] = label

    return labels


def label_fn_from_dict(labels: Dict[str, str]) -> Callable:
    """
    Convert label dictionary to a function for optax.multi_transform.

    Args:
        labels: Dict mapping parameter paths to 'muon' or 'adam'

    Returns:
        Function that labels a pytree
    """

    def label_pytree(params):
        """Label each parameter in the pytree"""
        flat_params = jax.tree_util.tree_flatten_with_path(params)[0]

        result = {}
        for path, param in flat_params:
            path_str = "/".join(str(p.key) for p in path)
            result[path_str] = labels.get(path_str, "adam")  # Default to adam

        # Unflatten back to match param structure
        return jax.tree.map(
            lambda p: labels.get(
                "/".join(
                    str(k.key)
                    for k in jax.tree_util.tree_flatten_with_path({"x": p})[0][0][0]
                ),
                "adam",
            ),
            params,
        )

    return label_pytree


def muon_hybrid(
    muon_lr: float = 0.02,
    adam_lr: float = 3e-4,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.01,
    adam_betas: tuple = (0.9, 0.95),
    adam_eps: float = 1e-10,
    adam_weight_decay: float = 0.01,
    steps: int = 5,
    muon_params_fn: Optional[Callable[[str, Any], bool]] = None,
) -> Callable[[Any], optax.GradientTransformation]:
    """
    Hybrid Muon + AdamW optimizer.

    Automatically routes parameters:
    - Hidden layer matrices (2D, not embed/lm_head) → Muon
    - Everything else (embeddings, lm_head, biases) → AdamW

    Args:
        muon_lr: Learning rate for Muon parameters
        adam_lr: Learning rate for Adam parameters
        muon_momentum: Momentum for Muon
        muon_weight_decay: Weight decay for Muon parameters
        adam_betas: Beta coefficients for Adam
        adam_eps: Epsilon for Adam
        adam_weight_decay: Weight decay for Adam parameters
        steps: Newton-Schulz steps for Muon
        muon_params_fn: Optional custom function to identify Muon params

    Returns:
        Function that takes params and returns GradientTransformation

    Example:
        >>> optimizer = muon_hybrid(muon_lr=0.02, adam_lr=3e-4)
        >>> opt_factory = optimizer(params)  # Returns transformation
        >>> opt_state = opt_factory.init(params)
        >>> updates, opt_state = opt_factory.update(grads, opt_state, params)
    """

    def create_optimizer(params):
        """Create optimizer with parameter routing"""

        # Label parameters
        labels = create_param_labels(params, muon_params_fn)

        # Print routing info
        muon_count = sum(1 for v in labels.values() if v == "muon")
        adam_count = sum(1 for v in labels.values() if v == "adam")
        print("Muon Hybrid Optimizer:")
        print(f"  Muon params: {muon_count}")
        print(f"  Adam params: {adam_count}")

        # Create Muon optimizer
        muon_optimizer = optax.chain(
            muon(
                learning_rate=muon_lr,
                momentum=muon_momentum,
                nesterov=True,
                steps=steps,
            ),
            optax.add_decayed_weights(muon_weight_decay),
        )

        # Create AdamW optimizer
        adam_optimizer = optax.adamw(
            learning_rate=adam_lr,
            b1=adam_betas[0],
            b2=adam_betas[1],
            eps=adam_eps,
            weight_decay=adam_weight_decay,
        )

        # Create label function for multi_transform
        def param_labels(params):
            """Return label for each parameter"""
            flat_params_with_path = jax.tree_util.tree_flatten_with_path(params)[0]

            labeled = {}
            for path, param in flat_params_with_path:
                path_str = "/".join(str(p.key) for p in path)
                labeled[path_str] = labels.get(path_str, "adam")

            # Reconstruct tree with labels
            return jax.tree.map(
                lambda p: labels.get(_get_param_path(params, p), "adam"), params
            )

        # Combine with multi_transform
        return optax.multi_transform(
            {"muon": muon_optimizer, "adam": adam_optimizer}, param_labels
        )

    return create_optimizer


def _get_param_path(params, target_param):
    """Helper to get path string for a parameter"""
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    for path, param in flat:
        if param is target_param:
            return "/".join(str(p.key) for p in path)
    return ""


def create_muon_hybrid_optimizer(
    params: Any,
    muon_lr: float = 0.02,
    adam_lr: float = 3e-4,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.01,
    adam_weight_decay: float = 0.01,
) -> optax.GradientTransformation:
    """
    Simplified API: Create hybrid optimizer directly from params.

    This is the easiest way to use the hybrid optimizer.

    Args:
        params: Model parameters (for parameter inspection)
        muon_lr: Learning rate for Muon
        adam_lr: Learning rate for Adam
        muon_momentum: Momentum for Muon
        muon_weight_decay: Weight decay for Muon
        adam_weight_decay: Weight decay for Adam

    Returns:
        Optax GradientTransformation ready to use

    Example:
        >>> optimizer = create_muon_hybrid_optimizer(
        ...     params,
        ...     muon_lr=0.02,
        ...     adam_lr=3e-4
        ... )
        >>> opt_state = optimizer.init(params)
        >>> updates, opt_state = optimizer.update(grads, opt_state, params)
    """
    # Create labels
    labels = create_param_labels(params)

    # Print routing for debug only
    muon_params = [k for k, v in labels.items() if v == "muon"]
    adam_params = [k for k, v in labels.items() if v == "adam"]

    print("Muon Hybrid Optimizer:")
    print(f"  Muon: {len(muon_params)} params")
    for p in muon_params[:5]:  # Show first 5
        print(f"    - {p}")
    if len(muon_params) > 5:
        print(f"    ... and {len(muon_params) - 5} more")

    print(f"  Adam: {len(adam_params)} params")
    for p in adam_params[:5]:
        print(f"    - {p}")
    if len(adam_params) > 5:
        print(f"    ... and {len(adam_params) - 5} more")

    # Create optimizers
    muon_opt = optax.chain(
        muon(learning_rate=muon_lr, momentum=muon_momentum, steps=5),
        optax.add_decayed_weights(muon_weight_decay),
    )

    adam_opt = optax.adamw(
        learning_rate=adam_lr,
        b1=0.9,
        b2=0.95,
        eps=1e-10,
        weight_decay=adam_weight_decay,
    )

    # Create label function
    def param_labels_fn(tree):
        return jax.tree.map(
            lambda _: "muon",  # Placeholder, will use labels dict
            tree,
        )

    # Use multi_transform with explicit labels
    return optax.multi_transform(
        {"muon": muon_opt, "adam": adam_opt},
        lambda tree: jax.tree.map(
            lambda x: labels.get(_get_param_path(params, x), "adam"), tree
        ),
    )
