import math
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union, Tuple


class ActivationInterventionHook:
    """
    Base class for intervention hooks targeting specific neuron indices across model layers.
    """
    def __init__(
        self,
        target_neurons_per_layer: Dict[int, List[int]],
        alpha: float = 0.0,
        entropy_threshold: Optional[float] = None
    ):
        """
        Args:
            target_neurons_per_layer: Mapping of {layer_index: list_of_neuron_indices}
            alpha: Scaling multiplier applied to targeted activations (0.0 = Hard Zero, 0.25 = Soft)
            entropy_threshold: Cutoff tau for Dynamic Entropy Gating (Condition 3)
        """
        self.target_neurons = target_neurons_per_layer
        self.alpha = alpha
        self.entropy_threshold = entropy_threshold
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _calculate_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculates step-wise categorical token entropy H(P_t) over vocabulary logits."""
        # logits shape: (batch_size, seq_len, vocab_size) or (batch_size, vocab_size)
        probs = torch.softmax(logits.detach(), dim=-1)
        log_probs = torch.log_softmax(logits.detach(), dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: (batch_size, seq_len) or (batch_size,)
        return entropy

    def register_hooks(self, model: nn.Module):
        """Attaches forward hooks to MLP/intermediate activation layers across the model."""
        self.remove_hooks()

        # Traverse decoder layers
        for layer_idx, layer_module in enumerate(self._get_decoder_layers(model)):
            if layer_idx in self.target_neurons and len(self.target_neurons[layer_idx]) > 0:
                target_submodule = self._get_target_submodule(layer_module)
                neurons = torch.tensor(self.target_neurons[layer_idx], dtype=torch.long)

                handle = target_submodule.register_forward_hook(
                    self._create_hook_fn(layer_idx, neurons)
                )
                self.handles.append(handle)

    def _get_decoder_layers(self, model: nn.Module) -> nn.ModuleList:
        """Locates the decoder layer list across common SLM architectures (Llama, Qwen, DeepSeek, Phi, Gemma)."""
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        elif hasattr(model, "layers"):
            return model.layers
        else:
            raise AttributeError("Could not dynamically resolve decoder layers on model architecture.")

    def _get_target_submodule(self, layer_module: nn.Module) -> nn.Module:
        """Locates the MLP intermediate activation layer (down_proj/gate_up_proj output)."""
        if hasattr(layer_module, "mlp"):
            return layer_module.mlp
        elif hasattr(layer_module, "block_sparse_moe"):
            return layer_module.block_sparse_moe
        return layer_module

    def _create_hook_fn(self, layer_idx: int, neurons: torch.Tensor):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor], output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
            # Resolve hidden states tensor format
            is_tuple = isinstance(output, tuple)
            hidden_states = output[0] if is_tuple else output

            # Determine whether intervention condition triggers
            should_intervene = True
            if self.entropy_threshold is not None and len(inputs) > 0:
                # Approximate entropy gating check using current hidden state projection or pass-through
                logits = inputs[0]
                if logits.dim() >= 2 and logits.size(-1) > 1000: # Check if input is vocabulary logits
                    entropy = self._calculate_entropy(logits)
                    should_intervene = (entropy.mean().item() >= self.entropy_threshold)

            if should_intervene:
                # Shape: (batch_size, seq_len, hidden_dim) or (batch_size, hidden_dim)
                device = hidden_states.device
                target_indices = neurons.to(device)

                # Clone tensor to modify in-place without triggering PyTorch view errors
                modified_states = hidden_states.clone()

                # Apply activation multiplier (alpha) to targeted neuron channels
                if modified_states.dim() == 3:
                    modified_states[:, :, target_indices] = modified_states[:, :, target_indices] * self.alpha
                elif modified_states.dim() == 2:
                    modified_states[:, target_indices] = modified_states[:, target_indices] * self.alpha

                if is_tuple:
                    return (modified_states,) + output[1:]
                return modified_states

            return output

        return hook

    def remove_hooks(self):
        """Detaches all registered forward hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


# =====================================================================
# Condition 1: Hard Zero-Ablation
# =====================================================================
class HardZeroAblationHook(ActivationInterventionHook):
    """Multiplies output activations of top-K H-neurons by 0.0."""
    def __init__(self, target_neurons_per_layer: Dict[int, List[int]]):
        super().__init__(target_neurons_per_layer=target_neurons_per_layer, alpha=0.0)


# =====================================================================
# Condition 2: Soft Suppression
# =====================================================================
class SoftSuppressionHook(ActivationInterventionHook):
    """Scales output activations of top-K H-neurons by alpha = 0.25."""
    def __init__(self, target_neurons_per_layer: Dict[int, List[int]], alpha: float = 0.25):
        super().__init__(target_neurons_per_layer=target_neurons_per_layer, alpha=alpha)


# =====================================================================
# Condition 3: Dynamic Entropy Gating
# =====================================================================
class DynamicEntropyGatingHook(ActivationInterventionHook):
    """Applies soft/hard suppression only when step-wise token entropy H(P_t) crosses tau."""
    def __init__(
        self,
        target_neurons_per_layer: Dict[int, List[int]],
        entropy_threshold: float,
        alpha: float = 0.0
    ):
        super().__init__(
            target_neurons_per_layer=target_neurons_per_layer,
            alpha=alpha,
            entropy_threshold=entropy_threshold
        )


# =====================================================================
# Condition 4: Random Pruning (Negative Control)
# =====================================================================
class RandomPruningControlHook(ActivationInterventionHook):
    """Zero-ablates K randomly selected neurons from the same intermediate layers."""
    def __init__(
        self,
        target_neurons_per_layer: Dict[int, List[int]],
        hidden_dim: int,
        seed: int = 42
    ):
        generator = torch.Generator().manual_seed(seed)
        random_neurons_per_layer = {}

        # Generate non-overlapping random indices of matching size K for each layer
        for layer_idx, target_list in target_neurons_per_layer.items():
            k = len(target_list)
            if k > 0:
                rand_indices = torch.randperm(hidden_dim, generator=generator)[:k].tolist()
                random_neurons_per_layer[layer_idx] = rand_indices
            else:
                random_neurons_per_layer[layer_idx] = []

        super().__init__(target_neurons_per_layer=random_neurons_per_layer, alpha=0.0)