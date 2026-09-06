import os
import json
import torch
import argparse
from pathlib import Path
from typing import Dict, List, Any

SUPPORTED_MODELS = [
    "llama3.2-3b",
    "gemma2-2b",
    "qwen2.5-3b",
    "phi3.5-mini",
    "deepseek-r1-1.5b"
]

K_VALUES = [1, 10, 50, 100, 200]


def aggregate_model_probes(model_key: str, probes_dir: str = "data/probes") -> Dict[str, Any]:
    """Loads all split files for a given model key and averages contrastive scores across splits."""
    probes_path = Path(probes_dir)
    split_files = list(probes_path.glob(f"{model_key}_*_hneurons.pt"))

    # Filter out any generic files that aren't specific split files
    split_files = [f for f in split_files if not f.name.endswith(f"{model_key}_hneurons.pt")]

    if not split_files:
        print(f"[!] No split files found for model key: {model_key}. Skipping...")
        return {}

    print(f"[+] Aggregating {len(split_files)} split probe file(s) for Model: {model_key}")

    accumulated_scores: Dict[str, torch.Tensor] = {}
    split_count = 0

    for file_path in split_files:
        data = torch.load(file_path, weights_only=False)

        for layer_key, metrics in data.items():
            # Skip non-layer metadata keys if present
            if not isinstance(metrics, dict) or "contrastive_score" not in metrics:
                continue

            score = metrics["contrastive_score"]
            if not isinstance(score, torch.Tensor):
                score = torch.tensor(score)

            if layer_key not in accumulated_scores:
                accumulated_scores[layer_key] = torch.zeros_like(score, dtype=torch.float32)

            accumulated_scores[layer_key] += score.float()

        split_count += 1

    if split_count == 0:
        return {}

    # Compute cross-split mean contrastive score per layer
    mean_scores = {layer_key: score / split_count for layer_key, score in accumulated_scores.items()}

    # Extract top-K candidate H-neuron mappings
    model_config: Dict[str, Any] = {}

    for k in K_VALUES:
        flat_candidates = []

        for layer_key, score_tensor in mean_scores.items():
            k_eff = min(k, len(score_tensor))
            top_vals, top_indices = torch.topk(score_tensor, k=k_eff)

            for idx, val in zip(top_indices.tolist(), top_vals.tolist()):
                flat_candidates.append({
                    "layer": layer_key,
                    "neuron_idx": idx,
                    "score": float(val)
                })

        # Sort candidate pool globally by contrastive score and select top K
        flat_candidates.sort(key=lambda x: x["score"], reverse=True)
        model_config[f"top_{k}"] = flat_candidates[:k]

    return model_config


def run_probe_training(probes_dir: str = "data/probes", output_config: str = "config/h_neurons.json"):
    """Aggregates all probe results across all target SLMs and outputs unified config JSON."""
    os.makedirs(Path(output_config).parent, exist_ok=True)
    all_configs: Dict[str, Any] = {}

    for model_key in SUPPORTED_MODELS:
        model_results = aggregate_model_probes(model_key=model_key, probes_dir=probes_dir)
        if model_results:
            all_configs[model_key] = model_results

    with open(output_config, "w", encoding="utf-8") as f:
        json.dump(all_configs, f, indent=2)

    print(f"\n[✓] Successfully generated target neuron configuration at: {output_config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate probing results and output top-K candidate H-neurons.")
    parser.add_argument("--probes_dir", type=str, default="data/probes")
    parser.add_argument("--output_config", type=str, default="config/h_neurons.json")

    args = parser.parse_args()
    run_probe_training(probes_dir=args.probes_dir, output_config=args.output_config)