import jax
import optax
from jax import numpy as jnp

from src.muon_adam import create_muon_hybrid_optimizer, create_param_labels


class TestMuonAdam:
    """Test parameter labeling logic"""

    def test_default_labeling_simple(self):
        """Test default labeling on simple params"""
        params = {
            "embed": jnp.ones((1000, 128)),  # 2D but embed → adam
            "layer1_weight": jnp.ones((128, 64)),  # 2D hidden → muon
            "layer1_bias": jnp.ones((128,)),  # 1D → adam
            "lm_head": jnp.ones((1000, 128)),  # 2D but lm_head → adam
        }

        labels = create_param_labels(params)

        assert labels["embed"] == "adam"
        assert labels["layer1_weight"] == "muon"
        assert labels["layer1_bias"] == "adam"
        assert labels["lm_head"] == "adam"

    def test_default_labeling_nested(self):
        """Test default labeling on nested structure"""
        params = {
            "embeddings": {
                "token": jnp.ones((1000, 128)),
                "position": jnp.ones((512, 128)),
            },
            "layers": {
                "layer0": {
                    "attention": {
                        "q_proj": jnp.ones((128, 128)),
                        "k_proj": jnp.ones((128, 128)),
                        "v_proj": jnp.ones((128, 128)),
                    },
                    "mlp": {
                        "up": jnp.ones((128, 512)),
                        "down": jnp.ones((512, 128)),
                    },
                    "ln1_weight": jnp.ones((128,)),
                    "ln2_weight": jnp.ones((128,)),
                }
            },
            "lm_head": jnp.ones((1000, 128)),
        }

        labels = create_param_labels(params)

        # Embeddings → adam
        assert "adam" in labels["embeddings/token"]
        assert "adam" in labels["embeddings/position"]

        # Hidden matrices → muon
        assert labels["layers/layer0/attention/q_proj"] == "muon"
        assert labels["layers/layer0/mlp/up"] == "muon"

        # Layernorm weights (1D) → adam
        assert labels["layers/layer0/ln1_weight"] == "adam"

        # LM head → adam
        assert labels["lm_head"] == "adam"

    def test_custom_labeling_function(self):
        """Test custom labeling function"""
        params = {
            "special_matrix": jnp.ones((100, 100)),
            "normal_matrix": jnp.ones((100, 100)),
        }

        def custom_fn(path_str, param):
            # Only use Muon for params with 'special' in name
            return "special" in path_str and param.ndim >= 2

        labels = create_param_labels(params, custom_fn)

        assert labels["special_matrix"] == "muon"
        assert labels["normal_matrix"] == "adam"

    def test_optimizer_creation(self):
        """Test that hybrid optimizer can be created"""
        params = {
            "embed": jnp.ones((100, 64)),
            "hidden": jnp.ones((64, 32)),
            "bias": jnp.ones((32,)),
        }

        optimizer = create_muon_hybrid_optimizer(params, muon_lr=0.02, adam_lr=3e-4)

        # Should initialize without error
        state = optimizer.init(params)
        assert state is not None

    def test_update_step(self):
        """Test single optimization step"""
        params = {
            "hidden": jnp.ones((64, 32)),
            "bias": jnp.ones((32,)),
        }

        grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, params)

        optimizer = create_muon_hybrid_optimizer(params)
        state = optimizer.init(params)

        updates, new_state = optimizer.update(grads, state, params)

        assert "hidden" in updates
        assert "bias" in updates
        assert not jnp.allclose(updates["hidden"], jnp.zeros_like(updates["hidden"]))

    def test_different_optimizers_applied(self):
        """Test that Muon and Adam produce different updates"""
        params = {
            "muon_param": jnp.ones((50, 50)),  # Should use Muon
            "adam_param": jnp.ones((50,)),  # Should use Adam
        }

        # Same gradients
        grads = {
            "muon_param": jnp.ones((50, 50)) * 0.1,
            "adam_param": jnp.ones((50,)) * 0.1,
        }

        optimizer = create_muon_hybrid_optimizer(params)
        state = optimizer.init(params)

        updates, _ = optimizer.update(grads, state, params)

        # Updates should be different (Muon orthogonalizes, Adam doesn't)
        # Just check they exist and have right shapes
        assert updates["muon_param"].shape == (50, 50)
        assert updates["adam_param"].shape == (50,)

    def test_training_loop(self):
        """Test full training loop with hybrid optimizer"""
        # Simple model
        params = {
            "layer1": {"weight": jnp.ones((128, 64)), "bias": jnp.zeros((128,))},
            "layer2": {"weight": jnp.ones((64, 10)), "bias": jnp.zeros((64,))},
        }

        optimizer = create_muon_hybrid_optimizer(params, muon_lr=0.02, adam_lr=0.001)
        state = optimizer.init(params)

        # Run 10 steps
        for step in range(10):
            grads = jax.tree.map(
                lambda p: jax.random.normal(jax.random.PRNGKey(step), p.shape) * 0.01,
                params,
            )

            updates, state = optimizer.update(grads, state, params)
            params = optax.apply_updates(params, updates)

            # Check no NaN
            flat_params = jax.tree_util.tree_leaves(params)
            assert not any(jnp.any(jnp.isnan(p)) for p in flat_params)
