import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

EVAL_BENCHMARKS = [
    "truthfulqa_mc1",
    "truthfulqa_mc2",
    "mmlu",
    "gsm8k",
    "wikitext"
]

SEEDS = [1, 10]
K_VALUES = [1, 100]
CONDITIONS = ["M1", "M2", "M3", "M4"]


def extract_primary_metric(task_results: dict) -> float:
    """Extracts a stable primary metric from lm-eval output blocks."""
    # Common primary metrics across standard lm-eval tasks
    for metric in ["acc,none", "acc_norm,none", "exact_match,none", "perplex,none"]:
        if metric in task_results:
            return float(task_results[metric])
    # Fallback to the first available float metric value if custom
    for k, v in task_results.items():
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def load_results_to_dataframe(results_dir: Path) -> pd.DataFrame:
    """Parses all evaluation JSON payloads into a structured pandas DataFrame."""
    records = []

    # Search recursively in case model subfolders exist
    for json_path in results_dir.glob("**/*.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            model_key = data.get("model_key")
            condition = data.get("condition")
            k_val = data.get("k_value")
            seed = data.get("seed")
            task_results = data.get("results", {})

            for task_name, metrics in task_results.items():
                score = extract_primary_metric(metrics)
                records.append({
                    "model_key": model_key,
                    "condition": condition,
                    "k_value": k_val,
                    "seed": seed,
                    "task": task_name,
                    "score": score
                })
        except Exception as e:
            print(f"[!] Error parsing {json_path.name}: {e}")

    return pd.DataFrame(records)


def analyze_performance_shifts(df: pd.DataFrame, output_dir: Path):
    """Computes mean performance shifts, thresholds, and statistical significance against M0."""
    if df.empty:
        print("[!] No evaluation data found to analyze.")
        return

    summary_rows = []

    models = df["model_key"].unique()
    tasks = df["task"].unique()

    for model in models:
        for task in tasks:
            # Filter baseline (M0, k=0) scores indexed by seed
            baseline_subset = df[(df["model_key"] == model) &
                                 (df["task"] == task) &
                                 (df["condition"] == "M0")]

            if baseline_subset.empty:
                continue

            baseline_dict = dict(zip(baseline_subset["seed"], baseline_subset["score"]))

            for cond in CONDITIONS:
                for k in K_VALUES:
                    # Filter intervention subset
                    intervention_subset = df[(df["model_key"] == model) &
                                             (df["task"] == task) &
                                             (df["condition"] == cond) &
                                             (df["k_value"] == k)]

                    if intervention_subset.empty:
                        continue

                    interv_dict = dict(zip(intervention_subset["seed"], intervention_subset["score"]))

                    # Align scores by common seeds for paired testing
                    common_seeds = sorted(list(set(baseline_dict.keys()).intersection(set(interv_dict.keys()))))
                    if len(common_seeds) < 2:
                        continue

                    baseline_scores = np.array([baseline_dict[s] for s in common_seeds])
                    interv_scores = np.array([interv_dict[s] for s in common_seeds])

                    shifts = interv_scores - baseline_scores
                    mean_shift = float(np.mean(shifts))
                    pct_shift = float(np.mean(shifts / (np.abs(baseline_scores) + 1e-9)) * 100)

                    # Statistical Significance Tests
                    try:
                        # Paired t-test
                        t_stat, p_val_ttest = stats.ttest_rel(interv_scores, baseline_scores)
                    except Exception:
                        t_stat, p_val_ttest = np.nan, np.nan

                    try:
                        # Wilcoxon signed-rank test (requires variation)
                        if np.all(shifts == 0):
                            p_val_wilcoxon = 1.0
                        else:
                            _, p_val_wilcoxon = stats.wilcoxon(interv_scores, baseline_scores, zero_method="wilcox")
                    except Exception:
                        p_val_wilcoxon = np.nan

                    summary_rows.append({
                        "model_key": model,
                        "task": task,
                        "condition": cond,
                        "k_value": k,
                        "mean_score_baseline": float(np.mean(baseline_scores)),
                        "mean_score_intervention": float(np.mean(interv_scores)),
                        "mean_absolute_shift": mean_shift,
                        "mean_percent_shift": pct_shift,
                        "p_value_ttest": p_val_ttest,
                        "p_value_wilcoxon": p_val_wilcoxon
                    })

    summary_df = pd.DataFrame(summary_rows)
    output_path = output_dir / "analysis_summary.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"[✓] Analysis complete! Summary statistics saved to: {output_path}")

    # Print high-level degradation alerts
    degraded = summary_df[summary_df["mean_percent_shift"] < -10.0]
    if not degraded.empty:
        print("\n[!] Significant degradation thresholds exceeded (>10% drop vs M0):")
        print(degraded[["model_key", "task", "condition", "k_value", "mean_percent_shift", "p_value_ttest"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results and perform statistical significance testing.")
    parser.add_argument("--results_dir", type=str, default="results/eval_outputs",
                        help="Directory containing evaluation JSON files")
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    print(f"[+] Loading evaluation records from {results_path}...")

    dataframe = load_results_to_dataframe(results_path)
    analyze_performance_shifts(dataframe, results_path)