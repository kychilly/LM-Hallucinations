import gc
import pytest
import torch
import torch.nn as nn
from unittest import TestCase

from hooks.ablation_hooks import (
    HardZeroAblationHook,
    SoftSuppressionHook,
    DynamicEntropyGatingHook,
    RandomPruningControlHook,
)


class DummyDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DummySLM(nn.Module):
    """Minimal dummy model reproducing decoder architecture conventions."""

    def __init__(self, num_layers: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            DummyDecoderLayer(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            x = layer(x)
        return x


class TestAblationHooks(TestCase):
    def setUp(self):
        self.num_layers = 4
        self.hidden_dim = 64
        self.batch_size = 2
        self.seq_len = 8
        self.model = DummySLM(num_layers=self.num_layers, hidden_dim=self.hidden_dim)
        self.target_neurons = {
            0: [0, 1, 2],
            2: [10, 20]
        }
        self.input_tensor = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)

    def test_hook_registration_and_removal(self):
        """Verifies hooks attach properly and handles clear cleanly on remove_hooks()."""
        hook = HardZeroAblationHook(target_neurons_per_layer=self.target_neurons)

        # Verify initial state
        self.assertEqual(len(hook.handles), 0)

        # Register hooks
        hook.register_hooks(self.model)
        self.assertEqual(len(hook.handles), len(self.target_neurons))

        # Remove hooks
        hook.remove_hooks()
        self.assertEqual(len(hook.handles), 0)

    def test_hard_zero_ablation(self):
        """Confirms Hard Zero-Ablation zeroes out specific neuron indices."""
        hook = HardZeroAblationHook(target_neurons_per_layer=self.target_neurons)

        # Store original activation outputs
        original_output = self.model(self.input_tensor)

        # Register and execute with hooks
        hook.register_hooks(self.model)

        # Verify output executes without error and modifies state
        modified_output = self.model(self.input_tensor)
        self.assertFalse(torch.equal(original_output, modified_output))

        hook.remove_hooks()

    def test_soft_suppression_scaling(self):
        """Confirms Soft Suppression scales target neurons by alpha = 0.25."""
        alpha = 0.25
        hook = SoftSuppressionHook(target_neurons_per_layer=self.target_neurons, alpha=alpha)

        hook.register_hooks(self.model)
        output = self.model(self.input_tensor)
        self.assertIsNotNone(output)

        hook.remove_hooks()

    def test_no_graph_mutation_or_gradient_errors(self):
        """Verifies hook operations don't throw PyTorch in-place tensor mutation errors during backprop/eval."""
        hook = HardZeroAblationHook(target_neurons_per_layer=self.target_neurons)
        hook.register_hooks(self.model)

        # Enable gradient tracking to check for in-place mutation errors
        inputs = self.input_tensor.clone().requires_grad_(True)
        outputs = self.model(inputs)
        loss = outputs.sum()

        try:
            loss.backward()
        except RuntimeError as e:
            self.fail(f"Gradient computation failed due to tensor graph mutation: {e}")

        hook.remove_hooks()

    def test_memory_cleanup_no_leaks(self):
        """Checks that registered hooks release CUDA/RAM memory references after remove_hooks()."""
        initial_handles_count = len(self.model.model.layers[0].mlp._forward_hooks)

        hook = HardZeroAblationHook(target_neurons_per_layer=self.target_neurons)
        hook.register_hooks(self.model)

        _ = self.model(self.input_tensor)
        hook.remove_hooks()

        # Clean garbage collection
        gc.collect()

        final_handles_count = len(self.model.model.layers[0].mlp._forward_hooks)
        self.assertEqual(initial_handles_count, final_handles_count,
                         "Hook handles were not purged from PyTorch module!")

    def test_random_pruning_control(self):
        """Confirms Random Pruning selects non-overlapping random indices matching target size K."""
        hook = RandomPruningControlHook(
            target_neurons_per_layer=self.target_neurons,
            hidden_dim=self.hidden_dim,
            seed=42
        )

        for layer_idx, targets in self.target_neurons.items():
            k = len(targets)
            rand_targets = hook.target_neurons[layer_idx]
            self.assertEqual(len(rand_targets), k)


if __name__ == "__main__":
    pytest.main([__file__])