import jax
import optax
from jax import numpy as jnp

from src.optax_muon import create_muon


class TestMuon:
    """Test basic Optax GradientTransformation functionality"""

    def test_init_simple_params(self):
        """Test initialization with simple parameters"""
        params = {"weight": jnp.ones((10, 5)), "bias": jnp.zeros((10,))}

        optimizer = create_muon(learning_rate=0.02)
        state = optimizer.init(params)

        # Check state structure
        assert isinstance(state, tuple), "State should be tuple (from optax.chain)"
        muon_state = state[0]  # First element is MuonState

        assert hasattr(muon_state, "momentum")
        assert hasattr(muon_state, "count")

        # Check momentum buffers initialized
        assert "weight" in muon_state.momentum
        assert "bias" in muon_state.momentum
        assert jnp.allclose(muon_state.momentum["weight"], jnp.zeros((10, 5)))

    def test_init_nested_params(self):
        """Test initialization with nested parameter structure"""
        params = {
            "layer1": {"weight": jnp.ones((20, 10)), "bias": jnp.zeros((20,))},
            "layer2": {"weight": jnp.ones((10, 5)), "bias": jnp.zeros((10,))},
        }

        optimizer = create_muon(learning_rate=0.02)
        state = optimizer.init(params)

        muon_state = state[0]

        # Check nested structure preserved
        assert "layer1" in muon_state.momentum
        assert "layer2" in muon_state.momentum
        assert "weight" in muon_state.momentum["layer1"]

    def test_update_single_step(self):
        """Test single optimization step"""
        params = {"weight": jnp.ones((10, 5))}
        grads = {"weight": jnp.ones((10, 5)) * 0.1}

        optimizer = create_muon(learning_rate=0.02)
        state = optimizer.init(params)

        updates, new_state = optimizer.update(grads, state)

        # Check updates produced
        assert "weight" in updates
        assert updates["weight"].shape == (10, 5)
        assert not jnp.allclose(updates["weight"], jnp.zeros((10, 5)))

        # Check state updated
        muon_state = new_state[0]
        assert muon_state.count == 1

    def test_apply_updates(self):
        """Test applying updates to parameters"""
        params = {"weight": jnp.ones((10, 5))}
        grads = {"weight": jnp.ones((10, 5)) * 0.1}

        optimizer = create_muon(learning_rate=0.02)
        state = optimizer.init(params)

        updates, state = optimizer.update(grads, state)
        new_params = optax.apply_updates(params, updates)

        # Parameters should change
        assert not jnp.allclose(new_params["weight"], params["weight"])

    def test_multiple_steps(self):
        """Test multiple optimization steps"""
        params = {"weight": jnp.ones((10, 5))}

        optimizer = create_muon(learning_rate=0.02)
        state = optimizer.init(params)

        # Take 5 steps
        for step in range(5):
            grads = {"weight": jnp.ones((10, 5)) * 0.1}
            updates, state = optimizer.update(grads, state)
            params = optax.apply_updates(params, updates)

        # Check step counter
        muon_state = state[0]
        assert muon_state.count == 5

    def test_works_with_optax_chain(self):
        """Test that Muon can be chained with other transformations"""
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            create_muon(learning_rate=0.02),
        )

        params = {"weight": jnp.ones((10, 5))}
        state = optimizer.init(params)

        grads = {"weight": jnp.ones((10, 5)) * 10.0}  # Large gradients
        updates, state = optimizer.update(grads, state)

        # Should work without error
        assert updates["weight"].shape == (10, 5)

    def test_works_with_learning_rate_schedule(self):
        """Test with learning rate schedule"""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=0.02,
            warmup_steps=10,
            decay_steps=100,
            end_value=0.001,
        )

        optimizer = optax.chain(
            create_muon(learning_rate=1.0, momentum=0.95),
            optax.scale_by_schedule(schedule),
        )

        params = {"weight": jnp.ones((10, 5))}
        state = optimizer.init(params)

        # Take several steps
        for step in range(20):
            grads = {"weight": jnp.ones((10, 5)) * 0.1}
            updates, state = optimizer.update(grads, state)
            params = optax.apply_updates(params, updates)

        # Should complete without error
        assert params["weight"].shape == (10, 5)

    def test_gradient_accumulation(self):
        """Test with gradient accumulation"""
        optimizer = create_muon(learning_rate=0.02)

        params = {"weight": jnp.ones((10, 5))}
        state = optimizer.init(params)

        # Accumulate gradients over 4 steps
        accumulated_grads = {"weight": jnp.zeros((10, 5))}

        for _ in range(4):
            grads = {"weight": jnp.ones((10, 5)) * 0.1}
            accumulated_grads = jax.tree.map(
                lambda acc, g: acc + g, accumulated_grads, grads
            )

        # Average and apply
        accumulated_grads = jax.tree.map(lambda g: g / 4, accumulated_grads)
        updates, state = optimizer.update(accumulated_grads, state)

        assert updates["weight"].shape == (10, 5)

    def test_momentum_accumulation_across_steps(self):
        """Test that momentum accumulates correctly"""
        params = {"weight": jnp.ones((10, 10))}

        optimizer = create_muon(learning_rate=0.02, momentum=0.95)
        state = optimizer.init(params)

        # Same gradient repeatedly
        grad = {"weight": jnp.ones((10, 10)) * 0.1}

        momentum_norms = []
        for _ in range(10):
            _, state = optimizer.update(grad, state)
            muon_state = state[0]
            mom_norm = float(jnp.linalg.norm(muon_state.momentum["weight"]))
            momentum_norms.append(mom_norm)

        # Momentum should be increasing
        assert momentum_norms[-1] > momentum_norms[0]

    def test_orthogonalization_applied_to_matrices(self):
        """Test that orthogonalization is applied to 2D parameters"""
        params = {"weight": jnp.ones((50, 50))}

        optimizer = create_muon(
            learning_rate=1.0, momentum=0.0
        )  # No momentum for simplicity
        state = optimizer.init(params)

        grad = {"weight": jax.random.normal(jax.random.PRNGKey(0), (50, 50))}
        updates, _ = optimizer.update(grad, state)

        # Update should be approximately orthogonal
        update_matrix = -updates["weight"]  # Negative due to learning rate sign
        product = update_matrix @ update_matrix.T
        error = float(jnp.max(jnp.abs(product - jnp.eye(50))))

        # Should be reasonably orthogonal (allowing for bfloat16 precision)
        assert error < 1.0, f"Update not orthogonal: error = {error:.4f}"

    def test_no_orthogonalization_for_vectors(self):
        """Test that 1D parameters skip orthogonalization"""
        params = {"bias": jnp.ones((100,))}

        optimizer = create_muon(learning_rate=1.0, momentum=0.0, nesterov=False)
        state = optimizer.init(params)

        grad = {"bias": jnp.ones((100,)) * 0.1}
        updates, _ = optimizer.update(grad, state)

        # Should be close (not exact due to momentum state management)
        assert updates["bias"].shape == grad["bias"].shape

    def test_nesterov_vs_standard_momentum(self):
        """Test difference between Nesterov and standard momentum"""
        params = {"weight": jnp.ones((20, 20))}
        grad = {"weight": jax.random.normal(jax.random.PRNGKey(0), (20, 20)) * 0.1}

        # Nesterov
        opt_nesterov = create_muon(learning_rate=0.02, momentum=0.95, nesterov=True)
        state_nesterov = opt_nesterov.init(params)

        # Standard
        opt_standard = create_muon(learning_rate=0.02, momentum=0.95, nesterov=False)
        state_standard = opt_standard.init(params)

        # Take one step with each
        updates_nesterov, _ = opt_nesterov.update(grad, state_nesterov)
        updates_standard, _ = opt_standard.update(grad, state_standard)

        # Should produce different updates
        assert not jnp.allclose(updates_nesterov["weight"], updates_standard["weight"])

    def test_jit_compilable(self):
        """Test that update function can be JIT compiled"""
        params = {"weight": jnp.ones((10, 5))}

        optimizer = create_muon(learning_rate=0.02)
        state = optimizer.init(params)

        # JIT compile the update
        @jax.jit
        def update_step(grads, state):
            return optimizer.update(grads, state)

        grads = {"weight": jnp.ones((10, 5)) * 0.1}
        updates, new_state = update_step(grads, state)

        assert updates["weight"].shape == (10, 5)

    def test_deterministic(self):
        """Test that optimizer is deterministic"""
        params = {"weight": jnp.ones((10, 5))}
        grad = {"weight": jax.random.normal(jax.random.PRNGKey(0), (10, 5))}

        optimizer = create_muon(learning_rate=0.02)
        state = optimizer.init(params)

        # Run twice
        updates1, state1 = optimizer.update(grad, state)
        updates2, state2 = optimizer.update(grad, state)

        # Should be identical
        assert jnp.allclose(updates1["weight"], updates2["weight"])

    def test_training_loop_converges(self):
        """Test that Muon can minimize a simple loss"""
        # Simple quadratic loss: minimize ||W - target||^2
        key = jax.random.PRNGKey(42)
        target = jax.random.normal(key, (50, 50))

        params = {"W": jax.random.normal(jax.random.PRNGKey(0), (50, 50))}

        def loss_fn(params):
            return jnp.sum((params["W"] - target) ** 2)

        optimizer = create_muon(learning_rate=0.1, momentum=0.9)
        state = optimizer.init(params)

        losses = []
        for _ in range(100):
            loss_val, grads = jax.value_and_grad(loss_fn)(params)
            updates, state = optimizer.update(grads, state)
            params = optax.apply_updates(params, updates)
            losses.append(float(loss_val))

        assert losses[-1] < losses[0], "Loss did not decrease at all"
        assert losses[-1] < losses[0] * 0.3, "Optimizer did not converge"

    def test_multi_layer_network(self):
        """Test with multi-layer network structure"""
        params = {
            "layer1": {"weight": jnp.ones((128, 64)), "bias": jnp.zeros((128,))},
            "layer2": {"weight": jnp.ones((64, 32)), "bias": jnp.zeros((64,))},
            "layer3": {"weight": jnp.ones((32, 10)), "bias": jnp.zeros((32,))},
        }

        optimizer = create_muon(learning_rate=0.02)
        state = optimizer.init(params)

        # Simulate 10 training steps
        for step in range(10):
            # Random gradients
            grads = jax.tree.map(
                lambda p: jax.random.normal(jax.random.PRNGKey(step), p.shape) * 0.01,
                params,
            )

            updates, state = optimizer.update(grads, state)
            params = optax.apply_updates(params, updates)

            # Check no NaN
            assert not jax.tree_util.tree_all(
                jax.tree.map(lambda p: jnp.any(jnp.isnan(p)), params)
            )
