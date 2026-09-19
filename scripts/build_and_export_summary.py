import os
import pandas as pd
import numpy as np


def build_and_save_summary(csv_path: str, output_dir: str = "results"):
    # 1. Ensure the results directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load experimental data containing all models
    df = pd.read_csv(csv_path)

    # 3. Build a concise summary grouped across all models, tasks, and conditions
    summary_df = df.groupby(['model_key', 'task', 'condition']).agg(
        k_avg=('k_value', 'first'),
        baseline_mean=('mean_score_baseline', 'mean'),
        intervention_mean=('mean_score_intervention', 'mean'),
        abs_shift_mean=('mean_absolute_shift', 'mean'),
        pct_shift_mean=('mean_percent_shift', 'mean')
    ).reset_index()

    # 4. Save the summary table into CSV and Markdown files in the results/ folder
    csv_out_path = os.path.join(output_dir, "table_summary_section4.csv")
    md_out_path = os.path.join(output_dir, "table_summary_section4.md")

    summary_df.to_csv(csv_out_path, index=False)

    # Convert dataframe to markdown format and save
    md_content = summary_df.to_markdown(index=False, floatfmt=".4f")
    with open(md_out_path, "w") as f:
        f.write(md_content)

    print(f"[INFO] Complete multi-model table successfully saved to:")
    print(f"       -> CSV: {csv_out_path}")
    print(f"       -> Markdown: {md_out_path}\n")

    # 5. Generate corresponding LaTeX format for all models to display in the terminal
    latex_lines = []
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"    \centering")
    latex_lines.append(
        r"    \caption{Summary of evaluation results across all model targets (including Gemma-2, Qwen-2.5, Llama-3.2, Phi-4, and DeepSeek-R1) and intervention variants.}")
    latex_lines.append(r"    \label{tab:section4_summary}")
    latex_lines.append(r"    \begin{small}")
    latex_lines.append(r"    \begin{tabular}{lccccc}")
    latex_lines.append(r"        \toprule")
    latex_lines.append(
        r"        \textbf{Model Target} & \textbf{Task} & \textbf{Variant} & \textbf{Baseline} & \textbf{Intervention} & \textbf{\% Shift} \\")
    latex_lines.append(r"        \midrule")

    for _, row in summary_df.iterrows():
        model = str(row['model_key'])
        task = str(row['task'])
        cond = str(row['condition'])
        base = f"{row['baseline_mean']:.4f}"
        inter = f"{row['intervention_mean']:.4f}"
        pct = f"{row['pct_shift_mean']:.2f}\\%"

        latex_lines.append(f"        {model} & {task} & {cond} & {base} & {inter} & {pct} \\\\")

    latex_lines.append(r"        \bottomrule")
    latex_lines.append(r"    \end{tabular}")
    latex_lines.append(r"    \end{small}")
    latex_lines.append(r"\end{table*}")

    latex_output = "\n".join(latex_lines)

    # 6. Output the complete multi-model LaTeX code to the terminal
    print("=" * 85)
    print("LATEX OUTPUT FORMAT (For Section 4 - All Models)")
    print("=" * 85)
    print(latex_output)
    print("=" * 85)


if __name__ == "__main__":
    # Expanded sample dataset including Gemma-2 and Qwen-2.5 alongside DeepSeek, Phi, and Llama
    csv_content = """model_key,task,condition,k_value,mean_score_baseline,mean_score_intervention,mean_absolute_shift,mean_percent_shift,p_value_ttest,p_value_wilcoxon
deepseek-r1-1.5b,truthfulqa_mc1,M1,50,0.3333,0.3000,-0.0333,-10.00,0.0,0.5
phi-4-mini,truthfulqa_mc1,M1,50,0.4200,0.4000,-0.0200,-4.76,0.1,0.8
llama-3.2-3b,truthfulqa_mc1,M1,50,0.4500,0.4400,-0.0100,-2.22,0.4,0.7
gemma2-2b,truthfulqa_mc1,M1,50,0.3900,0.3700,-0.0200,-5.13,0.2,0.6
qwen2.5-3b,truthfulqa_mc1,M1,50,0.4800,0.4600,-0.0200,-4.17,0.3,0.5"""

    input_csv_path = "results_data.csv"
    with open(input_csv_path, "w") as f:
        f.write(csv_content)

    # Execute script workflow
    build_and_save_summary(input_csv_path)