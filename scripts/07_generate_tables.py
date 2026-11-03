"""
Script 7: Generate Comprehensive Human-Readable Comparison Tables
═════════════════════════════════════════════════════════════════

Reads from:
  - results/classification_results.csv
  - results/classwise_metrics.csv
  - results/summary_metrics.csv
  - results/inference_time.csv
  - results/*_classwise.csv
  - results/*_matrix.csv

Generates:
  Table A  –  Per-Class Performance (Model × Variant × Class)
  Table B  –  New Results: All Models × All Experiments (Cross-Representation Matrices)
  Table C  –  Per-Class: Accuracy, Precision, Recall, F1
  Table D  –  Model × Representation Summary
  Table E  –  Latency & Throughput Per Model

Outputs to:
  results/tables/*.csv      (machine-readable)
  results/tables/*.html     (human-readable with heatmap styling)
  results/tables/*.png      (publication-quality visualizations)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from io import StringIO

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR  = RESULTS_DIR / "tables"
VIS_DIR     = RESULTS_DIR / "visualizations"
CSVS_DIR    = RESULTS_DIR / "csvs"

for d in [TABLES_DIR, VIS_DIR, CSVS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def find_csv(name):
    p1 = CSVS_DIR / name
    if p1.exists():
        return p1
    return RESULTS_DIR / name

CLASSIFICATION_CSV = find_csv("classification_results.csv")
CLASSWISE_CSV      = find_csv("classwise_metrics.csv")
SUMMARY_CSV        = find_csv("summary_metrics.csv")
INFERENCE_CSV      = find_csv("inference_time.csv")

REPRESENTATIONS = [
    "Original", "Outline", "Dotted", "Dashed", "Sketch", "Silhouette",
    "ColorTint_Red", "ColorTint_Green", "ColorTint_Blue"
]

CLASSES = ["car", "cat", "chair", "circle", "cup", "dog",
           "rectangle", "square", "table", "triangle"]

MODELS = ["ResNet-50", "ViT-B/16", "DINOv3",
          "CLIP_ViT-B/32", "EVA-CLIP_EVA02-B/16", "SigLIP-2_Base"]

# Matplotlib style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 13,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.1
})


# ─────────────────────────────────────────────────────────────
# HTML Helpers
# ─────────────────────────────────────────────────────────────

HTML_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
    color: #e0e0e0;
    padding: 40px 20px;
    min-height: 100vh;
  }}

  .container {{
    max-width: 1400px;
    margin: 0 auto;
  }}

  h1 {{
    font-size: 2.2em;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
  }}

  h2 {{
    font-size: 1.5em;
    font-weight: 600;
    color: #a78bfa;
    margin: 40px 0 15px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(167, 139, 250, 0.3);
  }}

  h3 {{
    font-size: 1.15em;
    font-weight: 500;
    color: #c4b5fd;
    margin: 25px 0 10px 0;
  }}

  p.subtitle {{
    color: #9ca3af;
    font-size: 0.95em;
    margin-bottom: 30px;
  }}

  .table-wrapper {{
    overflow-x: auto;
    margin: 15px 0 30px 0;
    border-radius: 12px;
    background: rgba(30, 30, 60, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(167, 139, 250, 0.15);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85em;
  }}

  th {{
    background: rgba(88, 66, 150, 0.4);
    color: #d8b4fe;
    font-weight: 600;
    padding: 12px 14px;
    text-align: center;
    border-bottom: 2px solid rgba(167, 139, 250, 0.3);
    position: sticky;
    top: 0;
    z-index: 1;
    white-space: nowrap;
  }}

  td {{
    padding: 9px 12px;
    text-align: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    font-variant-numeric: tabular-nums;
    transition: background 0.2s;
  }}

  tr:hover td {{
    background: rgba(167, 139, 250, 0.08) !important;
  }}

  td.row-header {{
    text-align: left;
    font-weight: 500;
    color: #c4b5fd;
    white-space: nowrap;
  }}

  .legend {{
    display: flex;
    gap: 20px;
    margin: 10px 0 20px 0;
    flex-wrap: wrap;
  }}

  .legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.85em;
    color: #9ca3af;
  }}

  .legend-color {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
  }}

  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.75em;
    font-weight: 600;
  }}

  .badge-best {{
    background: rgba(16, 185, 129, 0.2);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
  }}

  .badge-worst {{
    background: rgba(239, 68, 68, 0.2);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
  }}

  footer {{
    margin-top: 60px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.1);
    text-align: center;
    color: #6b7280;
    font-size: 0.82em;
  }}
</style>
</head>
<body>
<div class="container">
"""

HTML_FOOTER = """
<footer>
  <p>Generated by <strong>07_generate_tables.py</strong> — Representation-Invariant Object Recognition</p>
</footer>
</div>
</body>
</html>
"""


def accuracy_to_color(val, vmin=0, vmax=100):
    """Map accuracy percentage to RGB color string for heatmap cells."""
    if pd.isna(val):
        return "rgba(50,50,50,0.5)"
    norm = max(0.0, min(1.0, (val - vmin) / max(vmax - vmin, 1)))
    # Red → Yellow → Green gradient
    if norm < 0.5:
        r = 220
        g = int(60 + norm * 2 * 160)
        b = 60
    else:
        r = int(220 - (norm - 0.5) * 2 * 180)
        g = 200
        b = int(60 + (norm - 0.5) * 2 * 80)
    return f"rgba({r},{g},{b},0.25)"


def df_to_heatmap_html(df, title="", value_fmt=".2f", vmin=0, vmax=100, row_header_col=None):
    """Convert a DataFrame into an HTML table with heatmap-styled cells."""
    html = []
    if title:
        html.append(f'<h3>{title}</h3>')
    html.append('<div class="table-wrapper"><table>')

    # Header
    html.append('<thead><tr>')
    for col in df.columns:
        html.append(f'<th>{col}</th>')
    html.append('</tr></thead>')

    # Body
    html.append('<tbody>')
    for _, row in df.iterrows():
        html.append('<tr>')
        for ci, col in enumerate(df.columns):
            val = row[col]
            if ci == 0 and row_header_col is not None:
                html.append(f'<td class="row-header">{val}</td>')
            elif isinstance(val, (int, float)) and not pd.isna(val):
                bg = accuracy_to_color(val, vmin, vmax)
                html.append(f'<td style="background:{bg}">{val:{value_fmt}}%</td>')
            else:
                html.append(f'<td>{val}</td>')
        html.append('</tr>')
    html.append('</tbody></table></div>')
    return '\n'.join(html)


def df_to_metrics_html(df, title=""):
    """Convert a per-class metrics DataFrame to HTML with color coding."""
    html = []
    if title:
        html.append(f'<h3>{title}</h3>')
    html.append('<div class="table-wrapper"><table>')

    html.append('<thead><tr>')
    for col in df.columns:
        html.append(f'<th>{col}</th>')
    html.append('</tr></thead>')

    html.append('<tbody>')
    for _, row in df.iterrows():
        html.append('<tr>')
        for ci, col in enumerate(df.columns):
            val = row[col]
            if col in ("accuracy", "precision", "recall", "f1", "Accuracy", "Precision", "Recall", "F1"):
                if isinstance(val, (int, float)) and not pd.isna(val):
                    display_val = val * 100 if val <= 1.0 else val
                    bg = accuracy_to_color(display_val, 0, 100)
                    html.append(f'<td style="background:{bg}">{display_val:.2f}%</td>')
                else:
                    html.append(f'<td>{val}</td>')
            elif ci == 0 or col in ("class_name", "Class", "model", "Model"):
                html.append(f'<td class="row-header">{val}</td>')
            else:
                html.append(f'<td>{val}</td>')
        html.append('</tr>')
    html.append('</tbody></table></div>')
    return '\n'.join(html)


def filter_primary_benchmark_rows(df):
    """
    Filter to the primary benchmark row for every model:
    - Supervised / fine-tuned / probed models: Train_All_Combined (trained on all variants)
    - Zero-Shot VLMs: Zero-Shot
    """
    if "training_setup" not in df.columns or "model" not in df.columns:
        return df

    dfs = []
    for model_name, m_df in df.groupby("model"):
        setups = m_df["training_setup"].unique()
        all_comb = [s for s in setups if "all_combined" in str(s).lower()]
        if all_comb:
            dfs.append(m_df[m_df["training_setup"] == all_comb[0]])
        elif any("zero" in str(s).lower() for s in setups):
            z_setups = [s for s in setups if "zero" in str(s).lower()]
            dfs.append(m_df[m_df["training_setup"] == z_setups[0]])
        else:
            dfs.append(m_df)
    return pd.concat(dfs, ignore_index=True) if dfs else df


# ─────────────────────────────────────────────────────────────
# Table A: Per-Class Performance (Model × Variant × Class)
# ─────────────────────────────────────────────────────────────

def generate_table_a(classwise_df):
    """For every class, on every variant, how each model performs."""
    print("  ➤ Generating Table A: Per-Class Performance (Model × Variant × Class)")

    html_parts = []
    html_parts.append(f'<h2>Table A — Per-Class Performance: Every Class × Every Variant × Every Model</h2>')
    html_parts.append('<p class="subtitle">Shows accuracy (%) for each class on each test representation, across ALL benchmarked models. '
                      'Primary benchmark setup: Train_All_Combined (supervised) / Zero-Shot (VLMs).</p>')

    # Normalize column names
    df = classwise_df.copy()
    col_map = {}
    for c in df.columns:
        col_map[c] = c.strip().lower().replace(" ", "_")
    df.columns = [col_map.get(c, c) for c in df.columns]

    # Filter to primary benchmark setup per model so no models are omitted
    df = filter_primary_benchmark_rows(df)

    available_models = sorted(df["model"].unique()) if "model" in df.columns else []
    available_reps = [r for r in REPRESENTATIONS if r in df["test_representation"].unique()] if "test_representation" in df.columns else REPRESENTATIONS

    # For each class, create a sub-table
    for cls in CLASSES:
        cls_df = df[df["class_name"] == cls] if "class_name" in df.columns else pd.DataFrame()
        if cls_df.empty:
            continue

        # Pivot: rows = test_representation, columns = model, values = accuracy
        pivot = cls_df.pivot_table(
            index="test_representation",
            columns="model",
            values="accuracy",
            aggfunc="mean"
        )

        # Convert to percentage if needed
        if pivot.max().max() <= 1.0:
            pivot = pivot * 100

        # Reindex
        pivot = pivot.reindex(index=[r for r in available_reps if r in pivot.index])
        pivot = pivot.reset_index()
        pivot.columns = ["Test Representation"] + list(pivot.columns[1:])

        html_parts.append(df_to_heatmap_html(
            pivot,
            title=f'Class: <span style="color:#818cf8">{cls.upper()}</span>',
            row_header_col=0
        ))

    # Save CSV: wide format
    if "class_name" in df.columns and "test_representation" in df.columns and "model" in df.columns:
        wide_df = df.pivot_table(
            index=["class_name", "test_representation"],
            columns="model",
            values="accuracy",
            aggfunc="mean"
        )
        if wide_df.max().max() <= 1.0:
            wide_df = wide_df * 100
        wide_df = wide_df.reset_index()
        wide_df.to_csv(TABLES_DIR / "table_a_perclass_performance.csv", index=False, float_format="%.2f")

    return '\n'.join(html_parts)


# ─────────────────────────────────────────────────────────────
# Table B: All Models × All Experiments (Cross-Rep Matrix)
# ─────────────────────────────────────────────────────────────

def generate_table_b(classification_df):
    """Full cross-representation matrix for every model."""
    print("  ➤ Generating Table B: All Models × All Experiments (Cross-Rep Matrices)")

    html_parts = []
    html_parts.append(f'<h2>Table B — All Models × All Experiments: Cross-Representation Accuracy Matrices</h2>')
    html_parts.append('<p class="subtitle">Each sub-table shows the cross-representation accuracy matrix for one model. '
                      'Rows = Training Setup, Columns = Test Representation.</p>')

    df = classification_df.copy()
    col_map = {}
    for c in df.columns:
        col_map[c] = c.strip().lower().replace(" ", "_")
    df.columns = [col_map.get(c, c) for c in df.columns]

    # Filter to aggregate (class_name == "ALL" or no class_name)
    if "class_name" in df.columns:
        agg_df = df[df["class_name"] == "ALL"]
        if agg_df.empty:
            agg_df = df.groupby(["model", "training_setup", "test_representation"]).agg(
                accuracy=("accuracy", "mean"),
                f1_score=("f1_score", "mean") if "f1_score" in df.columns else ("accuracy", "mean")
            ).reset_index()
    else:
        agg_df = df

    available_models = sorted(agg_df["model"].unique()) if "model" in agg_df.columns else []

    all_matrices = []
    for model_name in available_models:
        model_df = agg_df[agg_df["model"] == model_name]

        pivot = model_df.pivot_table(
            index="training_setup",
            columns="test_representation",
            values="accuracy",
            aggfunc="mean"
        )

        if pivot.max().max() <= 1.0:
            pivot = pivot * 100

        # Reorder columns and rows
        col_order = [r for r in REPRESENTATIONS if r in pivot.columns]
        row_order_pref = [f"Train_{r}" for r in REPRESENTATIONS] + ["Train_All_Combined"]
        row_order = [r for r in row_order_pref if r in pivot.index]
        # Add any rows not in preferred order
        row_order += [r for r in pivot.index if r not in row_order]

        pivot = pivot.reindex(index=row_order, columns=col_order)
        pivot = pivot.reset_index()
        pivot.columns = ["Training Setup"] + list(pivot.columns[1:])

        html_parts.append(df_to_heatmap_html(
            pivot,
            title=f'Model: <span style="color:#818cf8">{model_name}</span>',
            row_header_col=0
        ))

        # Store for CSV
        pivot_csv = pivot.copy()
        pivot_csv.insert(0, "Model", model_name)
        all_matrices.append(pivot_csv)

    # Save combined CSV
    if all_matrices:
        combined = pd.concat(all_matrices, ignore_index=True)
        combined.to_csv(TABLES_DIR / "table_b_cross_representation_matrices.csv", index=False, float_format="%.2f")

    return '\n'.join(html_parts)


# ─────────────────────────────────────────────────────────────
# Table C: Per-Class Metrics (Accuracy, Precision, Recall, F1)
# ─────────────────────────────────────────────────────────────

def generate_table_c(classwise_df):
    """Per-class accuracy, precision, recall, F1 for each model/variant."""
    print("  ➤ Generating Table C: Per-Class Metrics (Accuracy, Precision, Recall, F1)")

    html_parts = []
    html_parts.append(f'<h2>Table C — Per-Class Metrics: Accuracy, Precision, Recall, F1</h2>')
    html_parts.append('<p class="subtitle">Detailed per-class metrics for each model on each test representation. '
                      'Training setup: Train_All_Combined (or best available).</p>')

    df = classwise_df.copy()
    col_map = {}
    for c in df.columns:
        col_map[c] = c.strip().lower().replace(" ", "_")
    # Filter to primary benchmark setup per model so all models are included
    df = filter_primary_benchmark_rows(df)

    # Filter out aggregate rows
    if "class_name" in df.columns:
        df = df[df["class_name"] != "ALL"]

    available_models = sorted(df["model"].unique()) if "model" in df.columns else []

    for model_name in available_models:
        model_df = df[df["model"] == model_name].copy()

        # For each representation, show per-class metrics
        available_reps = sorted(model_df["test_representation"].unique()) if "test_representation" in model_df.columns else []

        for rep in available_reps:
            rep_df = model_df[model_df["test_representation"] == rep].copy()
            if rep_df.empty:
                continue

            # Select display columns
            display_cols = []
            for c in ["class_name", "accuracy", "precision", "recall", "f1"]:
                if c in rep_df.columns:
                    display_cols.append(c)

            if not display_cols:
                continue

            display_df = rep_df[display_cols].copy()
            display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]

            html_parts.append(df_to_metrics_html(
                display_df,
                title=f'{model_name} → Test: <span style="color:#818cf8">{rep}</span>'
            ))

    # Save CSV
    df.to_csv(TABLES_DIR / "table_c_perclass_metrics.csv", index=False, float_format="%.4f")

    return '\n'.join(html_parts)


# ─────────────────────────────────────────────────────────────
# Table D: Model × Representation Summary
# ─────────────────────────────────────────────────────────────

def generate_table_d(classification_df):
    """Aggregated: Mean accuracy per model per test representation."""
    print("  ➤ Generating Table D: Model × Representation Summary")

    html_parts = []
    html_parts.append(f'<h2>Table D — Model × Representation Summary</h2>')
    html_parts.append('<p class="subtitle">Mean accuracy (%) per model per test representation, '
                      'aggregated across all training setups.</p>')

    df = classification_df.copy()
    col_map = {}
    for c in df.columns:
        col_map[c] = c.strip().lower().replace(" ", "_")
    df.columns = [col_map.get(c, c) for c in df.columns]

    # Filter to aggregate rows
    if "class_name" in df.columns:
        agg = df[df["class_name"] == "ALL"]
        if agg.empty:
            agg = df.groupby(["model", "test_representation"]).agg(accuracy=("accuracy", "mean")).reset_index()
    else:
        agg = df.groupby(["model", "test_representation"]).agg(accuracy=("accuracy", "mean")).reset_index()

    pivot = agg.pivot_table(index="model", columns="test_representation", values="accuracy", aggfunc="mean")

    if pivot.max().max() <= 1.0:
        pivot = pivot * 100

    col_order = [r for r in REPRESENTATIONS if r in pivot.columns]
    pivot = pivot.reindex(columns=col_order)

    # Add mean column
    pivot["Mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("Mean", ascending=False)
    pivot = pivot.reset_index()
    pivot.columns = ["Model"] + list(pivot.columns[1:])

    html_parts.append(df_to_heatmap_html(pivot, row_header_col=0))

    # Save CSV
    pivot.to_csv(TABLES_DIR / "table_d_model_representation_summary.csv", index=False, float_format="%.2f")

    # Also generate a heatmap plot
    plot_df = pivot.set_index("Model")
    if "Mean" in plot_df.columns:
        plot_df_no_mean = plot_df.drop(columns=["Mean"])
    else:
        plot_df_no_mean = plot_df

    fig, ax = plt.subplots(figsize=(14, max(4, len(plot_df) * 0.8)))
    sns.heatmap(plot_df_no_mean, annot=True, fmt=".1f", cmap="RdYlGn",
                vmin=0, vmax=100, ax=ax, linewidths=0.5,
                cbar_kws={"label": "Accuracy (%)"})
    ax.set_title("Model × Representation Summary — Mean Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(TABLES_DIR / "table_d_heatmap.png", dpi=200)
    plt.close(fig)

    return '\n'.join(html_parts)


# ─────────────────────────────────────────────────────────────
# Table E: Latency & Throughput Per Model
# ─────────────────────────────────────────────────────────────

def generate_table_e():
    """Latency and throughput comparison across models."""
    print("  ➤ Generating Table E: Latency & Throughput Per Model")

    html_parts = []
    html_parts.append(f'<h2>Table E — Latency & Throughput Per Model</h2>')
    html_parts.append('<p class="subtitle">Inference performance metrics for each model architecture.</p>')

    if not INFERENCE_CSV.exists():
        html_parts.append('<p style="color:#f87171;">⚠ inference_time.csv not found. Skipping Table E.</p>')
        return '\n'.join(html_parts)

    df = pd.read_csv(INFERENCE_CSV)

    # Clean up column names
    df.columns = [c.strip() for c in df.columns]

    # Display table
    html_parts.append('<div class="table-wrapper"><table>')
    html_parts.append('<thead><tr>')
    display_cols = [c for c in df.columns if c]
    for col in display_cols:
        html_parts.append(f'<th>{col}</th>')
    html_parts.append('</tr></thead>')

    html_parts.append('<tbody>')
    for _, row in df.iterrows():
        html_parts.append('<tr>')
        for ci, col in enumerate(display_cols):
            val = row[col]
            if ci == 0:
                html_parts.append(f'<td class="row-header">{val}</td>')
            elif isinstance(val, float):
                html_parts.append(f'<td>{val:.4f}</td>')
            else:
                html_parts.append(f'<td>{val}</td>')
        html_parts.append('</tr>')
    html_parts.append('</tbody></table></div>')

    # Save CSV copy
    df.to_csv(TABLES_DIR / "table_e_latency_throughput.csv", index=False)

    # Generate bar chart
    if "avg_latency_ms" in df.columns and "throughput_fps" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Latency bar chart
        colors_latency = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df)))
        bars1 = axes[0].barh(df["model"], df["avg_latency_ms"], color=colors_latency, edgecolor="white", linewidth=0.5)
        axes[0].set_xlabel("Average Latency (ms)")
        axes[0].set_title("Inference Latency per Model", fontweight="bold")
        for bar, val in zip(bars1, df["avg_latency_ms"]):
            axes[0].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                        f'{val:.2f}ms', va='center', fontsize=9)
        axes[0].invert_yaxis()

        # Throughput bar chart
        colors_tp = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df)))
        bars2 = axes[1].barh(df["model"], df["throughput_fps"], color=colors_tp, edgecolor="white", linewidth=0.5)
        axes[1].set_xlabel("Throughput (images/sec)")
        axes[1].set_title("Throughput per Model", fontweight="bold")
        for bar, val in zip(bars2, df["throughput_fps"]):
            axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}', va='center', fontsize=9)
        axes[1].invert_yaxis()

        plt.tight_layout()
        fig.savefig(TABLES_DIR / "table_e_latency_throughput.png", dpi=200)
        plt.close(fig)

    return '\n'.join(html_parts)


# ─────────────────────────────────────────────────────────────
# Generate Per-Class Heatmap Visualizations
# ─────────────────────────────────────────────────────────────

def generate_perclass_heatmaps(classwise_df):
    """Generate per-class accuracy heatmaps as PNG images."""
    print("  ➤ Generating per-class heatmap visualizations")

    df = classwise_df.copy()
    col_map = {}
    for c in df.columns:
        col_map[c] = c.strip().lower().replace(" ", "_")
    df.columns = [col_map.get(c, c) for c in df.columns]

    df = filter_primary_benchmark_rows(df)

    if "class_name" in df.columns:
        df = df[df["class_name"] != "ALL"]

    available_models = sorted(df["model"].unique()) if "model" in df.columns else []

    for metric in ["accuracy", "precision", "recall", "f1"]:
        if metric not in df.columns:
            continue

        for model_name in available_models:
            model_df = df[df["model"] == model_name]
            if model_df.empty:
                continue

            pivot = model_df.pivot_table(
                index="class_name", columns="test_representation",
                values=metric, aggfunc="mean"
            )
            if pivot.empty:
                continue

            if pivot.max().max() <= 1.0:
                pivot = pivot * 100

            col_order = [r for r in REPRESENTATIONS if r in pivot.columns]
            pivot = pivot.reindex(columns=col_order)
            row_order = [c for c in CLASSES if c in pivot.index]
            pivot = pivot.reindex(index=row_order)

            fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.6)))
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                        vmin=0, vmax=100, ax=ax, linewidths=0.5,
                        cbar_kws={"label": f"{metric.title()} (%)"})
            safe_model = model_name.replace("/", "_").replace(" ", "_")
            ax.set_title(f"{model_name} — Per-Class {metric.title()} (%) across Representations",
                        fontsize=12, fontweight="bold")
            ax.set_ylabel("Class")
            ax.set_xlabel("Test Representation")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fig.savefig(VIS_DIR / f"heatmap_{safe_model}_{metric}.png", dpi=200)
            plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  GENERATING COMPREHENSIVE COMPARISON TABLES")
    print("=" * 70)

    # Load data
    classification_df = None
    classwise_df = None

    if CLASSIFICATION_CSV.exists():
        classification_df = pd.read_csv(CLASSIFICATION_CSV)
        print(f"  ✓ Loaded classification_results.csv ({len(classification_df)} rows)")
    else:
        print(f"  ✗ {CLASSIFICATION_CSV} not found")

    if CLASSWISE_CSV.exists():
        classwise_df = pd.read_csv(CLASSWISE_CSV)
        print(f"  ✓ Loaded classwise_metrics.csv ({len(classwise_df)} rows)")
    else:
        print(f"  ✗ {CLASSWISE_CSV} not found")

    if SUMMARY_CSV.exists():
        summary_df = pd.read_csv(SUMMARY_CSV)
        print(f"  ✓ Loaded summary_metrics.csv ({len(summary_df)} rows)")
    else:
        summary_df = None
        print(f"  ✗ {SUMMARY_CSV} not found")

    # ── Build HTML Report ──
    html_content = HTML_HEADER.format(title="Comprehensive Model Comparison Tables")
    html_content += '<h1>Representation-Invariant Object Recognition</h1>\n'
    html_content += '<p class="subtitle">Comprehensive comparison tables across all models, representations, and classes</p>\n'

    # Table A
    if classwise_df is not None:
        html_content += generate_table_a(classwise_df)
    else:
        html_content += '<h2>Table A</h2><p style="color:#f87171;">classwise_metrics.csv not found</p>'

    # Table B
    if classification_df is not None:
        html_content += generate_table_b(classification_df)
    else:
        html_content += '<h2>Table B</h2><p style="color:#f87171;">classification_results.csv not found</p>'

    # Table C
    if classwise_df is not None:
        html_content += generate_table_c(classwise_df)
    else:
        html_content += '<h2>Table C</h2><p style="color:#f87171;">classwise_metrics.csv not found</p>'

    # Table D
    if classification_df is not None:
        html_content += generate_table_d(classification_df)
    else:
        html_content += '<h2>Table D</h2><p style="color:#f87171;">classification_results.csv not found</p>'

    # Table E
    html_content += generate_table_e()

    html_content += HTML_FOOTER

    # Save HTML report
    report_path = TABLES_DIR / "comprehensive_tables_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n  ✓ HTML report saved to: {report_path}")

    # Generate per-class heatmap PNGs
    if classwise_df is not None:
        generate_perclass_heatmaps(classwise_df)

    print(f"\n  ✓ All tables saved to: {TABLES_DIR}")
    print("=" * 70)
    print("  DONE — Table generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
