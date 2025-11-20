import jax
import jax.nn as jnn
import optax
from jax import numpy as jnp

from src.optax_muon import create_muon_with_adam


def test_training_run_learning_rates():
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k_data = jax.random.split(key, 4)

    def run_training(muon_lr, adam_lr):
        """Run training with specific learning rates."""
        params = {
            "layer1": {
                "weight": jax.random.normal(k1, (10, 8)) * 0.1,
                "bias": jnp.zeros((10,)),
            },
            "layer2": {
                "weight": jax.random.normal(k2, (5, 10)) * 0.1,
                "bias": jnp.zeros((5,)),
            },
        }

        optimizer = create_muon_with_adam(
            muon_params={"layer1.weight", "layer2.weight"},
            muon_lr=muon_lr,
            adam_lr=adam_lr,
        )

        opt_state = optimizer.init(params)

        X = jax.random.normal(k_data, (32, 8))
        y = jax.random.normal(k_data, (32, 5))

        def forward(params, x):
            h = x @ params["layer1"]["weight"].T + params["layer1"]["bias"]
            h = jnn.relu(h)
            out = h @ params["layer2"]["weight"].T + params["layer2"]["bias"]
            return out

        def loss_fn(params):
            pred = forward(params, X)
            return jnp.mean((pred - y) ** 2)

        losses = []
        update_norms = []

        for step in range(10):
            loss_val, grads = jax.value_and_grad(loss_fn)(params)
            losses.append(float(loss_val))

            updates, opt_state = optimizer.update(grads, opt_state)

            # Track update magnitude
            total_update_norm = sum(
                float(jnp.linalg.norm(u)) for u in jax.tree_util.tree_leaves(updates)
            )
            update_norms.append(total_update_norm)

            params = optax.apply_updates(params, updates)

        return losses, update_norms

    # Test with different learning rates
    print("\n=== Testing different learning rates ===")

    losses_small, norms_small = run_training(muon_lr=0.001, adam_lr=0.001)
    print(f"\nSmall LR (0.001):")
    print(f"  Loss: {losses_small[0]:.4f} → {losses_small[-1]:.4f}")
    print(f"  Avg update norm: {sum(norms_small) / len(norms_small):.6f}")

    losses_medium, norms_medium = run_training(muon_lr=0.02, adam_lr=0.01)
    print(f"\nMedium LR (0.02/0.01):")
    print(f"  Loss: {losses_medium[0]:.4f} → {losses_medium[-1]:.4f}")
    print(f"  Avg update norm: {sum(norms_medium) / len(norms_medium):.6f}")

    losses_large, norms_large = run_training(muon_lr=0.1, adam_lr=0.1)
    print(f"\nLarge LR (0.1):")
    print(f"  Loss: {losses_large[0]:.4f} → {losses_large[-1]:.4f}")
    print(f"  Avg update norm: {sum(norms_large) / len(norms_large):.6f}")

    # Verify that update norms scale with learning rate
    assert norms_large[0] > norms_medium[0] > norms_small[0], (
        "Update magnitudes should increase with learning rate"
    )

    # Verify that different LRs produce different final losses
    assert not (losses_small[-1] == losses_medium[-1] == losses_large[-1]), (
        "Different learning rates should produce different training trajectories"
    )

    print("\n✓ Learning rates have expected effect on training")


if __name__ == "__main__":
    test_training_run_learning_rates()
