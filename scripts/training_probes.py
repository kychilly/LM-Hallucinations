import os
import json
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Any
from sklearn.linear_model import LogisticRegression
from model_loader import SUPPORTED_MODELS


def train_l1_probe(
        X: np.ndarray,
        y: np.ndarray,
        C: float = 0.1,
        max_iter: int = 1000
) -> np.ndarray:
    """
    Trains an L1-regularized Logistic Regression classifier to enforce sparsity
    and isolate predictive H-neuron weights.

    Args:
        X: Feature matrix of shape (n_samples, num_neurons)
        y: Labels (0 = factual, 1 = hallucinated)
        C: Inverse regularization strength (smaller C = stronger L1 penalty/sparsity)
    """
    clf = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=C,
        random_state=42,
        max_iter=max_iter
    )
    clf.fit(X, y)

    # Absolute magnitude of learned coefficients reflects neuron importance
    return np.abs(clf.coef_.squeeze())


def run_probe_training(
        probes_dir: str = "data/probes",
        output_config: str = "config/h_neurons.json",
        top_k_values: List[int] = [10, 50, 100, 200]
):
    """Iterates over cached activations for all SLMs and extracts top-K candidate H-neurons."""
    os.makedirs(os.path.dirname(output_config), exist_ok=True)
    h_neuron_mapping: Dict[str, Any] = {}

    for model_key in SUPPORTED_MODELS.keys():
        probe_path = Path(probes_dir) / f"{model_key}_hneurons.pt"
        if not probe_path.exists():
            print(
                f"[!] Warning: Probe file {probe_path} not found. Skipping {model_key}. (Run probe_hneurons.py first)")
            continue

        print(f"\n[+] Training L1 Probes for Model: {model_key}")
        probe_data = torch.load(probe_path)

        all_layer_scores: List[Dict[str, Any]] = []

        for layer_key, layer_metrics in probe_data.items():
            # Extract raw mean activations as proxy representation matrices
            f_means = layer_metrics["raw_factual_mean"].numpy()
            h_means = layer_metrics["raw_hallucinated_mean"].numpy()

            # Construct synthetic balanced probe dataset from contrastive distributions
            # X shape: (n_samples, hidden_dim), y: binary classification labels
            n_samples_per_class = 100
            f_samples = f_means + np.random.normal(0, 0.01, size=(n_samples_per_class, f_means.shape[0]))
            h_samples = h_means + np.random.normal(0, 0.01, size=(n_samples_per_class, h_means.shape[0]))

            X = np.vstack([f_samples, h_samples])
            y = np.hstack([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])

            # Fit L1 Probe
            l1_weights = train_l1_probe(X, y, C=0.1)

            # Combine L1 probe weights with contrastive variance score
            contrastive_score = layer_metrics["contrastive_score"].numpy()
            combined_importance = l1_weights * contrastive_score

            for neuron_idx, score in enumerate(combined_importance):
                if score > 0:
                    all_layer_scores.append({
                        "layer": layer_key,
                        "neuron_idx": int(neuron_idx),
                        "score": float(score)
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
        print(f"[+] Successfully mapped top-{top_k_values} candidate H-neurons for {model_key}.")

    # Save mapping to config/h_neurons.json
    with open(output_config, "w") as f:
        json.dump(h_neuron_mapping, f, indent=2)

    print(f"\n[+] Saved complete candidate H-neuron mapping to: {output_config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train L1 probes and output top-K candidate H-neuron configuration.")
    parser.add_argument("--probes_dir", type=str, default="data/probes", help="Path to cached probing tensors.")
    parser.add_argument("--output_config", type=str, default="config/h_neurons.json", help="Destination JSON path.")

    args = parser.parse_args()
    run_probe_training(probes_dir=args.probes_dir, output_config=args.output_config)