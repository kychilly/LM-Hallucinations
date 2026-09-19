import os
import shutil
from pathlib import Path

# Directory
base_dir = Path(r"C:\Users\jyam4\PycharmProjects\LM-Hallucinations\results\eval_outputs")

# Models
models = ["deepseek-r1-1.5b", "gemma2-2b", "llama3.2-3b", "phi3.5-mini", "qwen2.5-3b"]

if not base_dir.exists():
    print(f"[!] Directory not found: {base_dir}")
else:
    moved_count = 0
    for file_path in base_dir.iterdir():
        # Process only top-level JSON files in the directory
        if file_path.is_file() and file_path.suffix == ".json":
            filename = file_path.name

            matched_model = None
            for model in models:
                if filename.startswith(model):
                    matched_model = model
                    break

            if matched_model:
                model_folder = base_dir / matched_model
                model_folder.mkdir(parents=True, exist_ok=True)

                dest_path = model_folder / filename
                shutil.move(str(file_path), str(dest_path))
                print(f"[✓] Moved: {filename} -> {matched_model}/")
                moved_count += 1
            else:
                print(f"[!] Skipped (no model match): {filename}")

    print(
        f"\n[✓] Sorting complete! Successfully organized {moved_count} JSON files into their respective model folders.")