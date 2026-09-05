import os
import json
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Any
from model_loader import SUPPORTED_MODELS


def run_probe_training(
    probes_dir: str = "data/probes",
    output_config: str = "config/h_neurons.json",
    top_k_values: List[int] = [10, 50, 100, 200]
):
    """Aggregates per-split probed metrics and outputs top-K candidate H-neuron configurations."""
    os.makedirs(os.path.dirname(output_config), exist_ok=True)
    h_neuron_mapping: Dict[str, Any] = {}
    probes_path = Path(probes_dir)

    for model_key in SUPPORTED_MODELS.keys():
        # Locate all per-split files generated for this model (e.g. deepseek-r1-1.5b_qa_hneurons.pt)
        split_files = list(probes_path.glob(f"{model_key}_*_hneurons.pt"))

        if not split_files:
            print(f"[!] Warning: No split probe files found for '{model_key}' in {probes_dir}. Skipping...")
            continue

        print(f"\n[+] Aggregating {len(split_files)} split probe file(s) for Model: {model_key}")

        # Accumulate metrics across available splits
        layer_accumulators: Dict[str, Dict[str, torch.Tensor]] = {}

        for sf in split_files:
            split_data = torch.load(sf)
            for layer_key, layer_metrics in split_data.items():
                if layer_key not in layer_accumulators:
                    layer_accumulators[layer_key] = {
                        "contrastive_score": torch.zeros_like(layer_metrics["contrastive_score"]),
                        "magnitude_diff": torch.zeros_like(layer_metrics["magnitude_diff"])
                    }
                layer_accumulators[layer_key]["contrastive_score"] += layer_metrics["contrastive_score"]
                layer_accumulators[layer_key]["magnitude_diff"] += layer_metrics["magnitude_diff"]

        # Average across splits and collect global neuron candidate rankings
        all_layer_scores: List[Dict[str, Any]] = []

        for layer_key, acc_metrics in layer_accumulators.items():
            avg_contrastive = acc_metrics["contrastive_score"] / len(split_files)
            avg_magnitude = acc_metrics["magnitude_diff"] / len(split_files)

            # Combined importance score based on contrastive shift & magnitude diff
            importance_scores = avg_contrastive * avg_magnitude

            for neuron_idx, score_val in enumerate(importance_scores):
                score_float = float(score_val.item())
                if score_float > 0:
                    all_layer_scores.append({
                        "layer": layer_key,
                        "neuron_idx": int(neuron_idx),
                        "score": score_float
                    })

        # Sort candidate neurons globally across all layers by importance score
        all_layer_scores.sort(key=lambda x: x["score"], reverse=True)

        model_k_map = {}
        for K in top_k_values:
            top_k_candidates = all_layer_scores[:K]
            model_k_map[f"top_{K}"] = [
                {
                    "layer": item["layer"],
                    "neuron_idx": item["neuron_idx"],
                    "score": round(item["score"], 6)
                }
                for item in top_k_candidates
            ]

        h_neuron_mapping[model_key] = model_k_map
        print(f"[✓] Mapped top-{top_k_values} candidate H-neurons for {model_key}.")

    # Save mapping to config/h_neurons.json
    with open(output_config, "w", encoding="utf-8") as f:
        json.dump(h_neuron_mapping, f, indent=2)

    print(f"\n[+] Saved candidate H-neuron configuration mapping to: {output_config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract top-K candidate H-neuron configurations from split probes.")
    parser.add_argument("--probes_dir", type=str, default="data/probes", help="Path to cached probing split tensors.")
    parser.add_argument("--output_config", type=str, default="config/h_neurons.json", help="Destination JSON path.")

    args = parser.parse_args()
    run_probe_training(probes_dir=args.probes_dir, output_config=args.output_config)