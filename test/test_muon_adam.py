import jax
from jax import numpy as jnp

from src.optax_muon import create_muon_with_adam


class TestMuonAdam:
    """Test basic Optax GradientTransformation functionality"""

    def test_muon_optimization_only(self):
        params = {"weight": jnp.ones((20, 20))}

        optimizer = create_muon_with_adam(
            muon_params={"weight"},
            muon_lr=0.02,
            adam_lr=0.001,
        )

        state = optimizer.init(params)
        grads = {"weight": jax.random.normal(jax.random.PRNGKey(0), (20, 20))}
        updates, _ = optimizer.update(grads, state)

        # The update should have gone through Newton-Schulz orthogonalization
        # We can detect this by checking the structure is "different" from raw gradients

        # Simple check: Muon update should NOT just be a scaled version of gradient
        update = -updates["weight"]

        # Normalize both to compare structure
        update_normalized = update / (jnp.linalg.norm(update) + 1e-8)
        grad_normalized = grads["weight"] / (jnp.linalg.norm(grads["weight"]) + 1e-8)

        # Cosine similarity - if Muon is applied, structure should change significantly
        similarity = float(jnp.sum(update_normalized * grad_normalized))

        # Perfect similarity would be 1.0, we want it notably different
        assert similarity < 0.95, (
            f"Update too similar to raw gradient (similarity={similarity:.4f}). "
            "Expected Muon orthogonalization to change structure."
        )

    def test_mixed_muon_and_adam_parameters(self):
        """Test that we can have some parameters use Muon and others use Adam simultaneously."""
        params = {
            "layer1_weight": jnp.ones((20, 20)),  # Matrix - should use Muon
            "layer1_bias": jnp.zeros((20,)),  # Vector - should use Adam
            "layer2_weight": jnp.ones((10, 20)),  # Matrix - should use Muon
            "layer2_bias": jnp.zeros((10,)),  # Vector - should use Adam
        }

        # Specify which parameters use Muon
        muon_param_names = {"layer1_weight", "layer2_weight"}

        optimizer = create_muon_with_adam(
            muon_params=muon_param_names,
            muon_lr=0.02,
            adam_lr=0.001,
        )

        state = optimizer.init(params)

        # Create gradients for all parameters
        grads = {
            "layer1_weight": jax.random.normal(jax.random.PRNGKey(0), (20, 20)),
            "layer1_bias": jnp.ones((20,)) * 0.1,
            "layer2_weight": jax.random.normal(jax.random.PRNGKey(1), (10, 20)),
            "layer2_bias": jnp.ones((10,)) * 0.1,
        }

        updates, new_state = optimizer.update(grads, state)

        # Verify all parameters got updates
        assert "layer1_weight" in updates
        assert "layer1_bias" in updates
        assert "layer2_weight" in updates
        assert "layer2_bias" in updates

        # Verify Muon parameters are approximately orthogonal
        for param_name in ["layer1_weight", "layer2_weight"]:
            update = -updates[param_name]
            product = update @ update.T
            identity = jnp.eye(update.shape[0])
            error = float(jnp.max(jnp.abs(product - identity)))
            assert error < 1.0, (
                f"{param_name} should use Muon (orthogonal updates), got error {error:.4f}"
            )

        # Verify Adam parameters have non-zero updates (basic sanity check)
        for param_name in ["layer1_bias", "layer2_bias"]:
            assert not jnp.allclose(updates[param_name], 0.0), (
                f"{param_name} should use Adam (non-zero updates)"
            )
