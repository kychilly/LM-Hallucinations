import os
import sys
import json
import torch
import argparse
from pathlib import Path
from typing import Dict, List, Any

# 1. Modify Python path FIRST before importing local submodules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Set environment variables
os.environ["OMP_NUM_THREADS"] = "4"

# 3. Third-party imports
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

# 4. Local module imports (resolvable from project root)
from scripts.model_loader import SUPPORTED_MODELS, ModelWrapper
from hooks.ablation_hooks import (
    HardZeroAblationHook,
    SoftSuppressionHook,
    DynamicEntropyGatingHook,
    RandomPruningControlHook,
)

EVAL_BENCHMARKS = [
    "truthfulqa_mc1",
    "truthfulqa_mc2",
    "mmlu",
    "gsm8k",
    "wikitext",
    "xstest"
]

SEEDS = [1, 10, 100, 1000, 10000]
K_VALUES = [1, 10, 50, 100, 200]


def load_target_neurons(config_path: str, model_key: str, k_val: int) -> Dict[int, List[int]]:
    """Extracts top-K target neuron mapping per layer from config/h_neurons.json."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}. Run train_probes.py first.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if model_key not in config or f"top_{k_val}" not in config[model_key]:
        return {}

    candidates = config[model_key][f"top_{k_val}"]
    layer_mapping: Dict[int, List[int]] = {}

    for cand in candidates:
        layer = int(cand["layer"].replace("layer_", ""))
        idx = int(cand["neuron_idx"])
        if layer not in layer_mapping:
            layer_mapping[layer] = []
        layer_mapping[layer].append(idx)

    return layer_mapping


def build_intervention_hook(
        condition: str,
        target_neurons: Dict[int, List[int]],
        hidden_dim: int,
        seed: int
):
    """Instantiates corresponding ablation hook based on specified condition name."""
    if condition == "M1":
        return HardZeroAblationHook(target_neurons_per_layer=target_neurons)
    elif condition == "M2":
        return SoftSuppressionHook(target_neurons_per_layer=target_neurons, alpha=0.25)
    elif condition == "M3":
        return DynamicEntropyGatingHook(target_neurons_per_layer=target_neurons, entropy_threshold=2.5, alpha=0.0)
    elif condition == "M4":
        return RandomPruningControlHook(target_neurons_per_layer=target_neurons, hidden_dim=hidden_dim, seed=seed)
    return None


def run_evaluation_sweep(
        model_key: str,
        config_path: str = "config/h_neurons.json",
        output_dir: str = "results/eval_outputs",
        batch_size: int = 1
):
    """Executes multi-seed, multi-condition benchmark sweep for a given model."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[+] Initializing Evaluation Suite for Model: {model_key}")

    # Initialize model wrapper and task manager
    wrapper = ModelWrapper(model_key=model_key)
    lm_obj = HFLM(pretrained=wrapper.model, tokenizer=wrapper.tokenizer, batch_size=batch_size)
    hidden_dim = wrapper.model.config.hidden_size

    # Custom task manager to register local tasks like xstest from tasks/
    task_manager = TaskManager(include_path="tasks")

    # Conditions list: M0 = Baseline control
    conditions = ["M0", "M1", "M2", "M3", "M4"]

    for seed in SEEDS:
        torch.manual_seed(seed)

        for condition in conditions:
            # Baseline M0 does not vary with K
            k_list = [0] if condition == "M0" else K_VALUES

            for k_val in k_list:
                run_id = f"{model_key}_{condition}_k{k_val}_seed{seed}"
                out_file = Path(output_dir) / f"{run_id}.json"

                if out_file.exists():
                    print(f"[+] Run {run_id} already completed. Skipping...")
                    continue

                print(f"\n[->] Executing Run: {run_id}")

                hook = None
                if condition != "M0":
                    target_neurons = load_target_neurons(config_path, model_key, k_val)
                    hook = build_intervention_hook(condition, target_neurons, hidden_dim, seed)
                    if hook:
                        hook.register_hooks(wrapper.model)

                # Execute evaluation across standard benchmarks via lm-eval harness
                try:
                    eval_results = lm_eval.simple_evaluate(
                        model=lm_obj,
                        tasks=EVAL_BENCHMARKS,
                        num_fewshot=0,
                        random_seed=seed,
                        task_manager=task_manager,
                    )

                    # Save evaluation outputs
                    save_payload = {
                        "model_key": model_key,
                        "condition": condition,
                        "k_value": k_val,
                        "seed": seed,
                        "results": eval_results["results"]
                    }

                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(save_payload, f, indent=2)

                    print(f"[✓] Saved run results to: {out_file}")

                finally:
                    if hook:
                        hook.remove_hooks()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation sweep across ablation conditions.")
    parser.add_argument("--model_key", type=str, default="deepseek-r1-1.5b", choices=list(SUPPORTED_MODELS.keys()))
    parser.add_argument("--config_path", type=str, default="config/h_neurons.json")
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()
    run_evaluation_sweep(
        model_key=args.model_key,
        config_path=args.config_path,
        batch_size=args.batch_size
    )