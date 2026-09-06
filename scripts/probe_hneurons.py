import os
import gc
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
    """Executes forward passes across probing pairs and saves top candidate H-neurons per split."""
    print(f"\n[+] Starting Neuron Probing for Model: {model_key}")

    # 1. Load Preprocessed Dataset
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Processed dataset not found at '{data_dir}'. Run preprocess.py first.")

    ds_dict = load_from_disk(data_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Initialize Model Wrapper & Attach Hooks across all layers
    model_wrapper = ModelWrapper(model_key=model_key)
    model_wrapper.attach_layer_hooks(layer_indices=None)  # Attach to all decoder layers

    # Iterate over splits in dataset (e.g., qa, dialogue, summarization, general)
    for split_name in ds_dict.keys():
        split_save_path = Path(output_dir) / f"{model_key}_{split_name}_hneurons.pt"

        # Instant skip guard: if the .pt file exists, do not re-run this split
        if split_save_path.exists():
            print(f"[+] Split '{split_name}' already completed ({split_save_path}). Skipping completely...")
            continue

        print(f"\n[+] Processing Split: {split_name}")
        split_ds = ds_dict[split_name]

        # Extract full sequence texts output by preprocess.py
        factual_prompts = [item["factual_full_text"] for item in split_ds]
        hallucinated_prompts = [item["hallucinated_full_text"] for item in split_ds]

        num_batches = (len(factual_prompts) + batch_size - 1) // batch_size

        split_layer_acts_f = {}
        split_layer_acts_h = {}

        for b in tqdm(range(num_batches), desc=f"Probing {split_name}"):
            f_batch = factual_prompts[b * batch_size: (b + 1) * batch_size]
            h_batch = hallucinated_prompts[b * batch_size: (b + 1) * batch_size]

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
                split_layer_acts_f[layer_key].append(f_acts[layer_key][:, -1, :].cpu())
                split_layer_acts_h[layer_key].append(h_acts[layer_key][:, -1, :].cpu())

            # Clear forward-pass caches to prevent RAM accumulation
            del f_acts, h_acts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Compute scores per layer for this specific split
        split_scores = {}
        for layer_key in split_layer_acts_f.keys():
            all_f = torch.cat(split_layer_acts_f[layer_key], dim=0)
            all_h = torch.cat(split_layer_acts_h[layer_key], dim=0)

            scores = compute_neuron_scores(
                all_f.unsqueeze(1),
                all_h.unsqueeze(1)
            )

            top_k_indices = torch.topk(
                scores["contrastive_score"],
                k=min(200, len(scores["contrastive_score"]))
            ).indices.tolist()

            split_scores[layer_key] = {
                "magnitude_diff": scores["magnitude_diff"],
                "contrastive_score": scores["contrastive_score"],
                "top_candidate_indices": top_k_indices
            }

        # Save split output to disk immediately upon split completion
        torch.save(split_scores, split_save_path)
        print(f"[✓] Saved split result to disk: {split_save_path}")

        # Purge references and release memory before next split
        del split_layer_acts_f, split_layer_acts_h, split_scores, factual_prompts, hallucinated_prompts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[+] Probing completed successfully across all splits!")


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