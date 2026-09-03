import os
import yaml
import random
from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)


def format_halueval_pair(record: dict, template_cfg: dict) -> dict:
    """Formats HaluEval entries into balanced factual vs. hallucinated pairs."""
    system_prompt = template_cfg["system_prompt"]
    template = template_cfg["template"]

    # Extract dynamic fields across HaluEval splits
    knowledge = record.get("knowledge", record.get("document", ""))
    question = record.get("question", record.get("user_query", ""))
    history = record.get("dialogue_history", "")

    factual_text = record.get("right_answer", record.get("right_response", record.get("right_summary", "")))
    hallucinated_text = record.get("hallucinated_answer",
                                   record.get("hallucinated_response", record.get("hallucinated_summary", "")))

    formatted_prompt = template.format(
        system_prompt=system_prompt,
        knowledge=knowledge,
        question=question,
        history=history
    )

    return {
        "prompt": formatted_prompt,
        "factual_completion": factual_text,
        "hallucinated_completion": hallucinated_text,
        # Whole sequences for target boundary extraction
        "factual_full_text": formatted_prompt + factual_text,
        "hallucinated_full_text": formatted_prompt + hallucinated_text,
        "prompt_char_len": len(formatted_prompt)
    }


def process_datasets():
    cfg = load_config()
    set_seed(cfg["experiment"]["seed"])

    raw_dir = Path("./data/raw")
    processed_dir = Path(cfg["experiment"]["output_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("[+] Loading raw datasets from local storage...")

    # 1. Process HaluEval Probing Benchmark
    halu_path = raw_dir / "halueval"
    if halu_path.exists():
        print("[+] Preprocessing HaluEval probing dataset...")
        halu_ds = load_from_disk(str(halu_path))
        processed_splits = {}

        for split in halu_ds.keys():
            template_type = "general" if split == "general" else ("dialogue" if split == "dialogue" else "qa")
            template_cfg = cfg["prompt_templates"][template_type]

            records = [format_halueval_pair(item, template_cfg) for item in halu_ds[split]]

            # Subsample if specified
            subsample = cfg["probing"]["subsample_size"]
            if subsample and len(records) > subsample:
                records = random.sample(records, subsample)

            processed_splits[split] = Dataset.from_list(records)

        processed_dict = DatasetDict(processed_splits)
        save_path = processed_dir / "halueval_probing_pairs"
        processed_dict.save_to_disk(str(save_path))
        print(f"    Saved processed probing dataset to: {save_path}")

    print("\n[+] Preprocessing complete.")


if __name__ == "__main__":
    process_datasets()