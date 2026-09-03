import os
import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv
from datasets import load_from_disk
from model_loader import ModelWrapper, SUPPORTED_MODELS

# Load environment variables (e.g. HF_TOKEN) from .env file
load_dotenv()


def compute_neuron_scores(
    factual_acts: torch.Tensor,
    hallucinated_acts: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Computes single-neuron contribution scores given paired factual vs hallucinated activations.

    Args:
        factual_acts: Tensor of shape (batch, seq_len, hidden_dim)
        hallucinated_acts: Tensor of shape (batch, seq_len, hidden_dim)
    """
    # Target the last token position for probing decision states
    f_last = factual_acts[:, -1, :]  # Shape: (batch, hidden_dim)
    h_last = hallucinated_acts[:, -1, :]  # Shape: (batch, hidden_dim)

    # 1. Activation Magnitude Differentials: |E[A_hallucinated] - E[A_factual]|
    mag_diff = torch.abs(h_last.mean(dim=0) - f_last.mean(dim=0))

    # 2. Contrastive / Relative Activation Variance Score
    # Measures how strongly a neuron shifts relative to its baseline variance
    f_var = f_last.var(dim=0) + 1e-6
    h_var = h_last.var(dim=0) + 1e-6
    contrastive_score = torch.abs(h_last.mean(dim=0) - f_last.mean(dim=0)) / torch.sqrt(f_var + h_var)

    return {
        "magnitude_diff": mag_diff,
        "contrastive_score": contrastive_score,
        "raw_factual_mean": f_last.mean(dim=0),
        "raw_hallucinated_mean": h_last.mean(dim=0)
    }


def run_probing_pipeline(
    model_key: str,
    data_dir: str = "data/processed/halueval_probing_pairs",
    output_dir: str = "data/probes",
    batch_size: int = 8
):
    """Executes forward passes across probing pairs and saves top candidate H-neurons."""
    print(f"\n[+] Starting Neuron Probing for Model: {model_key}")

    # 1. Load Preprocessed Dataset
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Processed dataset not found at '{data_dir}'. Run preprocess.py first.")

    ds_dict = load_from_disk(data_dir)

    # 2. Initialize Model Wrapper & Attach Hooks across all layers
    model_wrapper = ModelWrapper(model_key=model_key)
    model_wrapper.attach_layer_hooks(layer_indices=None)  # Attach to all decoder layers

    layer_scores = {}

    # Iterate over splits in dataset (e.g., qa, dialogue, summarization, general)
    for split_name in ds_dict.keys():
        print(f"[+] Processing Split: {split_name}")
        split_ds = ds_dict[split_name]

        # Extract full sequence texts output by preprocess.py
        factual_prompts = [item["factual_full_text"] for item in split_ds]
        hallucinated_prompts = [item["hallucinated_full_text"] for item in split_ds]

        num_batches = (len(factual_prompts) + batch_size - 1) // batch_size

        split_layer_acts_f = {}
        split_layer_acts_h = {}

        for b in tqdm(range(num_batches), desc=f"Probing {split_name}"):
            f_batch = factual_prompts[b * batch_size : (b + 1) * batch_size]
            h_batch = hallucinated_prompts[b * batch_size : (b + 1) * batch_size]

            # Forward pass: Factual Prompts
            _, f_acts = model_wrapper.run_with_caching(f_batch)

            # Forward pass: Hallucinated Prompts
            _, h_acts = model_wrapper.run_with_caching(h_batch)

            # Store and append layer outputs
            for layer_key in f_acts.keys():
                if layer_key not in split_layer_acts_f:
                    split_layer_acts_f[layer_key] = []
                    split_layer_acts_h[layer_key] = []

                # Take last token activation per prompt in batch
                split_layer_acts_f[layer_key].append(f_acts[layer_key][:, -1, :])
                split_layer_acts_h[layer_key].append(h_acts[layer_key][:, -1, :])

        # Aggregate across batches and compute scores per layer
        for layer_key in split_layer_acts_f.keys():
            all_f = torch.cat(split_layer_acts_f[layer_key], dim=0)
            all_h = torch.cat(split_layer_acts_h[layer_key], dim=0)

            scores = compute_neuron_scores(
                all_f.unsqueeze(1),
                all_h.unsqueeze(1)
            )

            if layer_key not in layer_scores:
                layer_scores[layer_key] = {
                    "magnitude_diff": torch.zeros_like(scores["magnitude_diff"]),
                    "contrastive_score": torch.zeros_like(scores["contrastive_score"])
                }

            # Accumulate scores across dataset splits
            layer_scores[layer_key]["magnitude_diff"] += scores["magnitude_diff"]
            layer_scores[layer_key]["contrastive_score"] += scores["contrastive_score"]

    # 3. Format and Export Top Candidate Neurons
    os.makedirs(output_dir, exist_ok=True)
    summary_path = Path(output_dir) / f"{model_key}_hneurons.pt"

    export_payload = {}
    for layer_key, metrics in layer_scores.items():
        # Average across dataset splits
        avg_mag = metrics["magnitude_diff"] / len(ds_dict.keys())
        avg_contrastive = metrics["contrastive_score"] / len(ds_dict.keys())

        # Top candidate neurons based on contrastive score
        top_k_indices = torch.topk(avg_contrastive, k=min(200, len(avg_contrastive))).indices.tolist()

        export_payload[layer_key] = {
            "magnitude_diff": avg_mag,
            "contrastive_score": avg_contrastive,
            "top_candidate_indices": top_k_indices
        }

    torch.save(export_payload, summary_path)
    print(f"\n[+] Probing completed successfully!")
    print(f"[+] Probing metrics saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe SLM layers for candidate Hallucination Neurons.")
    parser.add_argument(
        "--model_key",
        type=str,
        default="deepseek-r1-1.5b",
        choices=list(SUPPORTED_MODELS.keys()),
        help="Target SLM short key identifier."
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Probing evaluation batch size.")

    args = parser.parse_args()
    run_probing_pipeline(model_key=args.model_key, batch_size=args.batch_size)