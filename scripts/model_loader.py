import torch
from dotenv import load_dotenv
load_dotenv()
from typing import Dict, Any, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

# Unified dictionary of supported SLMs
SUPPORTED_MODELS = {
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma2-2b": "google/gemma-2-2b-it",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "phi3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "deepseek-r1-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
}


class ActivationCache:
    """Utility class to hook into model layers and cache activations during forward passes."""

    def __init__(self):
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def _get_hook(self, layer_name: str):
        def hook(module, input, output):
            # Handles raw hidden states output standard for transformer layers
            if isinstance(output, tuple):
                hidden_state = output[0]
            else:
                hidden_state = output
            # Store copy on CPU to prevent VRAM allocation spikes
            self.activations[layer_name] = hidden_state.detach().cpu()

        return hook

    def register_layer(self, module: torch.nn.Module, layer_name: str):
        hook = module.register_forward_hook(self._get_hook(layer_name))
        self.hooks.append(hook)

    def clear(self):
        self.activations.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class ModelWrapper:
    def __init__(self, model_key: str, device: str = "auto", torch_dtype: torch.dtype = torch.bfloat16):
        if model_key not in SUPPORTED_MODELS:
            raise ValueError(f"Model key '{model_key}' not supported. Choose from: {list(SUPPORTED_MODELS.keys())}")

        self.model_name_or_path = SUPPORTED_MODELS[model_key]
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[+] Loading Tokenizer: {self.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[+] Loading Model: {self.model_name_or_path} on device: {self.device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
        self.cache = ActivationCache()

    def attach_layer_hooks(self, layer_indices: Optional[List[int]] = None):
        """
        Attaches activation hooks to decoder blocks.
        If layer_indices is None, targets all layer blocks.
        """
        # Resolve target layer sub-module across standard transformer architectures
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h
        else:
            raise AttributeError("Unable to resolve layer structure for target model.")

        target_indices = layer_indices if layer_indices is not None else list(range(len(layers)))

        for idx in target_indices:
            layer_module = layers[idx]
            self.cache.register_layer(layer_module, f"layer_{idx}")

        print(f"[+] Attached forward hooks to {len(target_indices)} layers.")

    def run_with_caching(self, text_prompts: List[str]) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        """
        Runs a forward pass on prompts and captures cached activations.
        """
        self.cache.clear()
        inputs = self.tokenizer(text_prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs, self.cache.activations


if __name__ == "__main__":
    # Sanity check with a lightweight model key
    loader = ModelWrapper("deepseek-r1-1.5b")
    loader.attach_layer_hooks(layer_indices=[-1])  # Target last layer
    _, activations = loader.run_with_caching(["Testing activation extraction pipeline."])
    print("Extracted activations shape (Layer -1):", activations["layer_-1"].shape)