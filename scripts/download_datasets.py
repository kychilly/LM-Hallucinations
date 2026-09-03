import os
import json
import urllib.request
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi

# 1. Parse .env file manually if present
env_path = Path(".env")
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip("'\"")

RAW_DATA_DIR = Path("./data/raw")
README_PATH = Path("README.md")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")


def get_repo_hash(repo_id: str, token: str = None) -> str:
    try:
        api = HfApi(token=token)
        info = api.dataset_info(repo_id=repo_id)
        return info.sha
    except Exception as e:
        return f"Unknown ({e})"


def load_json_or_jsonl(file_path: Path) -> list:
    """Safely loads both JSON arrays and JSONL files."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            # Standard JSON array
            records = json.loads(content)
        else:
            # JSON-Lines (JSONL)
            for line in content.splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def main():
    metadata_records = []

    if HF_TOKEN:
        print("[+] Detected HF_TOKEN from environment/.env file.")
    else:
        print("[!] Warning: HF_TOKEN not found in environment or .env file.")

    # Standard Hugging Face Datasets
    DATASETS_TO_LOAD = {
        "truthful_qa": {"repo_id": "truthfulqa/truthful_qa", "config": "generation"},
        "mmlu": {"repo_id": "cais/mmlu", "config": "all"},
        "gsm8k": {"repo_id": "openai/gsm8k", "config": "main"},
        "wikitext_103": {"repo_id": "salesforce/wikitext", "config": "wikitext-103-v1"},
        "xstest": {"repo_id": "walledai/XSTest", "config": None},
    }

    for name, spec in DATASETS_TO_LOAD.items():
        repo_id = spec["repo_id"]
        config = spec["config"]
        local_path = RAW_DATA_DIR / name

        commit_hash = get_repo_hash(repo_id, token=HF_TOKEN)
        metadata_records.append({
            "name": name,
            "repo_id": repo_id,
            "config": config or "default",
            "hash": commit_hash,
            "path": str(local_path)
        })

        if local_path.exists() and any(local_path.iterdir()):
            print(f"[=] Skipping '{name}' (already cached)")
            continue

        print(f"\n[+] Processing '{name}' ({repo_id})...")
        try:
            ds = load_dataset(repo_id, config, token=HF_TOKEN) if config else load_dataset(repo_id, token=HF_TOKEN)
            ds.save_to_disk(str(local_path))
            print(f"    Saved to: {local_path}")
        except Exception as e:
            print(f"    [!] Failed to process {name}: {e}")

    # HaluEval via PyData JSON Normalization
    halu_path = RAW_DATA_DIR / "halueval"
    metadata_records.append({
        "name": "halueval",
        "repo_id": "github.com/RUCAIBox/HaluEval",
        "config": "qa/dialogue/summarization/general",
        "hash": "main",
        "path": str(halu_path)
    })

    if halu_path.exists() and any(halu_path.iterdir()):
        print(f"[=] Skipping 'halueval' (already cached)")
    else:
        print("\n[+] Processing 'halueval' (RUCAIBox/HaluEval)...")
        try:
            sources = {
                "qa": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json",
                "dialogue": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/dialogue_data.json",
                "summarization": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/summarization_data.json",
                "general": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/general_data.json",
            }

            temp_dir = RAW_DATA_DIR / "temp_halueval"
            temp_dir.mkdir(exist_ok=True)

            dataset_splits = {}
            for split_name, url in sources.items():
                dest = temp_dir / f"{split_name}.json"
                if not dest.exists():
                    print(f"    Downloading {split_name} split...")
                    urllib.request.urlretrieve(url, dest)

                records = load_json_or_jsonl(dest)
                dataset_splits[split_name] = Dataset.from_list(records)

            ds_dict = DatasetDict(dataset_splits)
            ds_dict.save_to_disk(str(halu_path))

            # Cleanup temp folder
            for f in temp_dir.glob("*.json"):
                f.unlink()
            temp_dir.rmdir()

            print(f"    Saved to: {halu_path}")
        except Exception as e:
            print(f"    [!] Failed to process halueval: {e}")

    # Write provenance details to data/README.md
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write("# Dataset Registry and Provenance\n\n")
        f.write("| Dataset Name | Source Repository | Config | Version Hash (Commit SHA) | Local Path |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for record in metadata_records:
            f.write(
                f"| `{record['name']}` | `{record['repo_id']}` | `{record['config']}` | `{record['hash']}` | `{record['path']}` |\n")

    print(f"\nSuccessfully updated `{README_PATH}`.")


if __name__ == "__main__":
    main()