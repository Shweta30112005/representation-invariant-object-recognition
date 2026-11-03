"""
Script 12: Multi-Criteria Top-10 Combination Selection
=======================================================
Computes a Composite Robustness Score (CRS) using 7 metrics for each
combination from the 511-experiment, then selects the Top 10 per model.

Runs locally — no GPU required. Uses only existing CSV files.

Outputs:
  results/top10_selection/full_ranking_{model}.csv
  results/top10_selection/top10_{model}.csv
  results/top10_selection/top10_zero_shot_baselines.csv
  results/top10_selection/crs_weights.json
  results/top10_selection/selection_justification_{model}.md
  results/top10_selection/visualizations/*.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
COMBO_DIR = PROJECT_DIR / "results" / "combo_experiment"
OUTPUT_DIR = PROJECT_DIR / "results" / "top10_selection"
VIZ_DIR = OUTPUT_DIR / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

REPRESENTATIONS = [
    "Original", "Outline", "Dotted", "Dashed", "Sketch",
    "Silhouette", "ColorTint_Red", "ColorTint_Green", "ColorTint_Blue"
]

SHORT_NAMES = {
    "Original": "Org", "Outline": "Out", "Dotted": "Dot",
    "Dashed": "Dsh", "Sketch": "Skt", "Silhouette": "Sil",
    "ColorTint_Red": "Red", "ColorTint_Green": "Grn", "ColorTint_Blue": "Blu"
}

# CRS Weights (from approved plan)
CRS_WEIGHTS = {
    "mean_acc": 0.20,
    "worst_case_acc": 0.25,
    "stability": 0.10,
    "mean_f1": 0.15,
    "worst_class_acc": 0.10,
    "pareto_efficiency": 0.10,
    "ood_transfer_gap": 0.10,
}

ZERO_SHOT_MODELS = ["CLIP ViT-B/32", "EVA-CLIP (EVA02-B/16)", "SigLIP-2 Base", "Qwen3-VL-8B"]
TRAINED_MODELS = ["ResNet-50", "DINOv3", "ViT-B/16"]


def load_data():
    """Load combo results and classwise data."""
    combo_df = pd.read_csv(COMBO_DIR / "combo_results.csv")
    classwise_df = pd.read_csv(COMBO_DIR / "combo_classwise.csv")
    return combo_df, classwise_df


def compute_metrics(combo_df, classwise_df, model_name):
    """Compute all 7 metrics for every combination of a given model."""
    mdf = combo_df[combo_df["model"] == model_name].copy()

    if mdf.empty:
        print(f"  [WARN] No data for model '{model_name}'. Skipping.")
        return pd.DataFrame()

    # Pivot: rows = train_combo, columns = test_representation, values = accuracy
    acc_pivot = mdf.pivot_table(
        index=["train_combo", "train_combo_full", "combo_size"],
        columns="test_representation",
        values="accuracy"
    ).reset_index()

    # Also get F1 pivot
    f1_pivot = mdf.pivot_table(
        index="train_combo", columns="test_representation", values="f1"
    ).reset_index()

    # Precision and Recall pivots
    prec_pivot = mdf.pivot_table(
        index="train_combo", columns="test_representation", values="precision"
    ).reset_index()
    rec_pivot = mdf.pivot_table(
        index="train_combo", columns="test_representation", values="recall"
    ).reset_index()

    results = []
    test_reps = [r for r in REPRESENTATIONS if r in acc_pivot.columns]

    for _, row in acc_pivot.iterrows():
        combo = row["train_combo"]
        combo_full = row["train_combo_full"]
        combo_size = int(row["combo_size"])

        accs = [row[r] * 100 for r in test_reps if pd.notna(row[r])]  # as percentages
        if not accs:
            continue

        # Metric 1: Mean Accuracy
        mean_acc = np.mean(accs)

        # Metric 2: Worst-Case Accuracy
        worst_case_acc = np.min(accs)

        # Metric 3: Std Deviation (lower is better)
        std_acc = np.std(accs) if len(accs) > 1 else 0.0

        # Metric 4: Mean F1
        f1_row = f1_pivot[f1_pivot["train_combo"] == combo]
        f1_vals = []
        if not f1_row.empty:
            for r in test_reps:
                if r in f1_row.columns and pd.notna(f1_row.iloc[0][r]):
                    f1_vals.append(f1_row.iloc[0][r])
        mean_f1 = np.mean(f1_vals) if f1_vals else 0.0

        # Mean Precision and Recall
        prec_row = prec_pivot[prec_pivot["train_combo"] == combo]
        prec_vals = []
        if not prec_row.empty:
            for r in test_reps:
                if r in prec_row.columns and pd.notna(prec_row.iloc[0][r]):
                    prec_vals.append(prec_row.iloc[0][r])
        mean_prec = np.mean(prec_vals) if prec_vals else 0.0

        rec_row = rec_pivot[rec_pivot["train_combo"] == combo]
        rec_vals = []
        if not rec_row.empty:
            for r in test_reps:
                if r in rec_row.columns and pd.notna(rec_row.iloc[0][r]):
                    rec_vals.append(rec_row.iloc[0][r])
        mean_rec = np.mean(rec_vals) if rec_vals else 0.0

        # Metric 5: Worst-Case Per-Class Accuracy
        cw = classwise_df[
            (classwise_df["model"] == model_name) &
            (classwise_df["train_combo"] == combo)
        ]
        worst_class_acc = cw["accuracy"].min() * 100 if not cw.empty else 0.0

        # Metric 6: Pareto Efficiency (accuracy per training image)
        n_train_images = max(combo_size * 2400, 1)
        pareto_eff = mean_acc / n_train_images

        # Metric 7: OOD Transfer Gap
        # Compute mean accuracy on variants NOT included in training
        combo_parts_short = combo.split("+")
        short_to_full = {v: k for k, v in SHORT_NAMES.items()}
        in_training = set()
        for p in combo_parts_short:
            if p in short_to_full:
                in_training.add(short_to_full[p])
            elif p == "Zero-Shot":
                pass  # all variants are OOD for zero-shot
            else:
                # Try matching full name
                for r in REPRESENTATIONS:
                    if r == p:
                        in_training.add(r)

        ood_accs = [row[r] * 100 for r in test_reps if r not in in_training and pd.notna(row[r])]
        ood_transfer = np.mean(ood_accs) if ood_accs else mean_acc

        # Per-variant accuracy columns
        per_var = {}
        for r in test_reps:
            per_var[f"acc_{SHORT_NAMES.get(r, r)}"] = row[r] * 100 if pd.notna(row[r]) else 0.0

        results.append({
            "train_combo": combo,
            "train_combo_full": combo_full,
            "combo_size": combo_size,
            "mean_acc": mean_acc,
            "worst_case_acc": worst_case_acc,
            "std_acc": std_acc,
            "mean_f1": mean_f1,
            "mean_precision": mean_prec,
            "mean_recall": mean_rec,
            "worst_class_acc": worst_class_acc,
            "pareto_efficiency": pareto_eff,
            "ood_transfer_gap": ood_transfer,
            **per_var
        })

    return pd.DataFrame(results)


def compute_crs(metrics_df):
    """Compute Composite Robustness Score with min-max normalization."""
    if metrics_df.empty:
        return metrics_df

    df = metrics_df.copy()

    # Min-max normalize each metric
    for col in ["mean_acc", "worst_case_acc", "mean_f1", "worst_class_acc",
                 "pareto_efficiency", "ood_transfer_gap"]:
        vmin, vmax = df[col].min(), df[col].max()
        if vmax > vmin:
            df[f"{col}_norm"] = (df[col] - vmin) / (vmax - vmin)
        else:
            df[f"{col}_norm"] = 1.0

    # Stability: lower std is better, so normalize inversely
    vmin, vmax = df["std_acc"].min(), df["std_acc"].max()
    if vmax > vmin:
        df["stability_norm"] = 1.0 - (df["std_acc"] - vmin) / (vmax - vmin)
    else:
        df["stability_norm"] = 1.0

    # Compute CRS
    df["crs_score"] = (
        CRS_WEIGHTS["mean_acc"] * df["mean_acc_norm"] +
        CRS_WEIGHTS["worst_case_acc"] * df["worst_case_acc_norm"] +
        CRS_WEIGHTS["stability"] * df["stability_norm"] +
        CRS_WEIGHTS["mean_f1"] * df["mean_f1_norm"] +
        CRS_WEIGHTS["worst_class_acc"] * df["worst_class_acc_norm"] +
        CRS_WEIGHTS["pareto_efficiency"] * df["pareto_efficiency_norm"] +
        CRS_WEIGHTS["ood_transfer_gap"] * df["ood_transfer_gap_norm"]
    )

    # Store individual normalized contributions for visualization
    df["crs_mean_acc_contrib"] = CRS_WEIGHTS["mean_acc"] * df["mean_acc_norm"]
    df["crs_worst_case_contrib"] = CRS_WEIGHTS["worst_case_acc"] * df["worst_case_acc_norm"]
    df["crs_stability_contrib"] = CRS_WEIGHTS["stability"] * df["stability_norm"]
    df["crs_f1_contrib"] = CRS_WEIGHTS["mean_f1"] * df["mean_f1_norm"]
    df["crs_worst_class_contrib"] = CRS_WEIGHTS["worst_class_acc"] * df["worst_class_acc_norm"]
    df["crs_pareto_contrib"] = CRS_WEIGHTS["pareto_efficiency"] * df["pareto_efficiency_norm"]
    df["crs_ood_contrib"] = CRS_WEIGHTS["ood_transfer_gap"] * df["ood_transfer_gap_norm"]

    return df.sort_values("crs_score", ascending=False).reset_index(drop=True)


def get_selection_rationale(row, model_name):
    """Generates concise, solid grounds explaining why this combination was selected."""
    reasons = []
    size = int(row["combo_size"])
    worst = row["worst_case_acc"]
    std = row["std_acc"]
    f1 = row["mean_f1"]
    
    if size <= 5:
        pct_saved = round((1 - size / 9.0) * 100)
        reasons.append(f"Pareto optimal: only {size} variants (saves {pct_saved}% training data & compute)")
    elif size >= 8:
        reasons.append(f"Reference ceiling: comprehensive {size}-variant ensemble")
    else:
        reasons.append(f"Balanced {size}-variant structural basis")

    if worst >= 91.5:
        reasons.append(f"high worst-case floor ({worst:.2f}%) prevents representation collapse")
    elif worst >= 87.0:
        reasons.append(f"strong worst-case floor ({worst:.2f}%)")

    if std < 2.6:
        reasons.append(f"exceptional stability across modalities (sigma={std:.2f}%)")
    elif std < 3.5:
        reasons.append(f"stable cross-modal consistency (sigma={std:.2f}%)")

    if f1 > 0.95:
        reasons.append("near-perfect class-level F1 score (>0.95)")
    elif f1 > 0.92:
        reasons.append(f"high class balance (F1={f1:.4f})")

    return "; ".join(reasons)


def generate_justification(top10_df, model_name):
    """Generate a markdown justification for each Top-10 selection."""
    lines = [
        f"# Top 10 Combination Selection Justification — {model_name}",
        f"",
        f"## Composite Robustness Score (CRS) Framework",
        f"Each combination is scored using 7 weighted metrics:",
        f"- Mean Accuracy (20%) — overall performance",
        f"- Worst-Case Accuracy (25%) — robustness under worst variant",
        f"- Stability/1-σ (10%) — consistency across variants",
        f"- Mean F1 (15%) — class-balanced performance",
        f"- Worst-Case Class Accuracy (10%) — no class left behind",
        f"- Pareto Efficiency (10%) — accuracy per training image",
        f"- OOD Transfer Gap (10%) — generalization to unseen variants",
        f"",
        f"## Selected Combinations",
        f""
    ]

    for i, row in top10_df.iterrows():
        rank = i + 1
        lines.append(f"### Rank {rank}: `{row['train_combo']}` (Size {int(row['combo_size'])})")
        lines.append(f"- **CRS Score**: {row['crs_score']:.4f}")
        lines.append(f"- **Mean Accuracy**: {row['mean_acc']:.2f}%")
        lines.append(f"- **Worst-Case Accuracy**: {row['worst_case_acc']:.2f}% (variant with lowest performance)")
        lines.append(f"- **Std Deviation**: {row['std_acc']:.2f}% (lower = more consistent)")
        lines.append(f"- **Mean F1**: {row['mean_f1']:.4f}")
        lines.append(f"- **Worst Class Accuracy**: {row['worst_class_acc']:.2f}%")
        lines.append(f"- **Pareto Efficiency**: {row['pareto_efficiency']:.6f} (acc/training_image)")
        lines.append(f"- **OOD Transfer**: {row['ood_transfer_gap']:.2f}%")

        # Variant-level breakdown
        var_cols = [c for c in top10_df.columns if c.startswith("acc_")]
        lines.append(f"- **Per-Variant Accuracy**: " + ", ".join(
            f"{c.replace('acc_', '')}: {row[c]:.1f}%" for c in var_cols
        ))

        # Why selected
        reasons = []
        if row['worst_case_acc'] >= 85:
            reasons.append("strong worst-case (≥85%)")
        if row['std_acc'] < 4:
            reasons.append("highly consistent (σ<4%)")
        if row['combo_size'] <= 5:
            reasons.append(f"efficient (only {int(row['combo_size'])} variants needed)")
        if row['mean_f1'] > 0.93:
            reasons.append("excellent class balance (F1>0.93)")
        if row['ood_transfer_gap'] > 90:
            reasons.append("strong OOD transfer (>90%)")

        if reasons:
            lines.append(f"- **Selection Rationale**: {'; '.join(reasons)}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────

def viz_crs_breakdown(top10_df, model_name):
    """Stacked horizontal bar chart of CRS metric contributions."""
    fig, ax = plt.subplots(figsize=(14, 8))

    contrib_cols = [
        ("crs_mean_acc_contrib", "Mean Acc", "#2196F3"),
        ("crs_worst_case_contrib", "Worst-Case", "#F44336"),
        ("crs_stability_contrib", "Stability", "#4CAF50"),
        ("crs_f1_contrib", "Mean F1", "#FF9800"),
        ("crs_worst_class_contrib", "Worst Class", "#9C27B0"),
        ("crs_pareto_contrib", "Pareto Eff.", "#00BCD4"),
        ("crs_ood_contrib", "OOD Transfer", "#795548"),
    ]

    labels = [f"#{i+1}: {row['train_combo']}" for i, row in top10_df.iterrows()]
    labels = labels[::-1]  # reverse for bottom-to-top

    y_pos = np.arange(len(labels))
    left = np.zeros(len(labels))

    for col, label, color in contrib_cols:
        vals = top10_df[col].values[::-1]
        ax.barh(y_pos, vals, left=left, label=label, color=color, edgecolor='white', linewidth=0.5)
        left += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("CRS Score Contribution", fontsize=12)
    ax.set_title(f"CRS Breakdown — Top 10 Combinations ({model_name})", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"crs_breakdown_{model_name.replace('/', '_').replace(' ', '_')}.png", dpi=150)
    plt.close()
    print(f"  ✓ CRS breakdown chart saved")


def viz_radar(top10_df, model_name):
    """9-axis radar chart showing accuracy profile for each Top-10 combo."""
    var_cols = [c for c in top10_df.columns if c.startswith("acc_")]
    var_labels = [c.replace("acc_", "") for c in var_cols]
    N = len(var_labels)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, (_, row) in enumerate(top10_df.head(10).iterrows()):
        values = [row[c] for c in var_cols]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=1.5, label=f"#{i+1}: {row['train_combo']}",
                color=colors[i], markersize=3)
        ax.fill(angles, values, alpha=0.05, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(var_labels, fontsize=10)
    ax.set_ylim(50, 100)
    ax.set_title(f"Accuracy Radar — Top 10 ({model_name})", fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"radar_top10_{model_name.replace('/', '_').replace(' ', '_')}.png", dpi=150)
    plt.close()
    print(f"  ✓ Radar chart saved")


def viz_pareto(ranked_df, top10_df, model_name):
    """Pareto frontier: mean accuracy vs combo_size."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # All combos as light scatter
    ax.scatter(ranked_df["combo_size"], ranked_df["mean_acc"],
               alpha=0.15, s=15, c='gray', label='All 511 combos')

    # Top 10 highlighted
    scatter = ax.scatter(top10_df["combo_size"], top10_df["mean_acc"],
                         c=top10_df["crs_score"], cmap='RdYlGn', s=120,
                         edgecolors='black', linewidth=1.5, zorder=5,
                         label='Top 10 (by CRS)')

    for i, row in top10_df.iterrows():
        ax.annotate(f"#{i+1}", (row["combo_size"], row["mean_acc"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8, fontweight='bold')

    # Pareto frontier line (best per size)
    pareto = ranked_df.groupby("combo_size")["mean_acc"].max().reset_index()
    ax.plot(pareto["combo_size"], pareto["mean_acc"], 'r--', linewidth=2, alpha=0.7,
            label='Pareto Frontier')

    plt.colorbar(scatter, label='CRS Score')
    ax.set_xlabel("Training Combination Size (k)", fontsize=12)
    ax.set_ylabel("Mean Accuracy (%)", fontsize=12)
    ax.set_title(f"Pareto Frontier — Accuracy vs Combo Size ({model_name})", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(range(1, 10))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"pareto_frontier_{model_name.replace('/', '_').replace(' ', '_')}.png", dpi=150)
    plt.close()
    print(f"  ✓ Pareto frontier saved")


def viz_heatmap(top10_df, model_name):
    """10 × 9 annotated heatmap of accuracy per variant."""
    var_cols = [c for c in top10_df.columns if c.startswith("acc_")]
    var_labels = [c.replace("acc_", "") for c in var_cols]
    combo_labels = [f"#{i+1}: {row['train_combo']}" for i, row in top10_df.iterrows()]

    data = top10_df[var_cols].values

    fig, ax = plt.subplots(figsize=(14, 8))
    im = sns.heatmap(data, annot=True, fmt=".1f", cmap="YlOrRd",
                     xticklabels=var_labels, yticklabels=combo_labels,
                     ax=ax, vmin=60, vmax=100, linewidths=0.5,
                     cbar_kws={'label': 'Accuracy (%)'})
    ax.set_title(f"Top 10 × 9 Variant Accuracy Heatmap ({model_name})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Test Variant", fontsize=12)
    ax.set_ylabel("Training Combination (by CRS rank)", fontsize=12)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"heatmap_top10_{model_name.replace('/', '_').replace(' ', '_')}.png", dpi=150)
    plt.close()
    print(f"  ✓ Accuracy heatmap saved")


def viz_worst_case(top10_dfs):
    """Horizontal bar chart comparing worst-case accuracy across models."""
    fig, ax = plt.subplots(figsize=(14, 8))

    y_offset = 0
    y_ticks = []
    y_labels = []
    colors = {"ResNet-50": "#2196F3", "DINOv3": "#F44336"}

    for model_name, top10 in top10_dfs.items():
        for i, row in top10.iterrows():
            bar_color = colors.get(model_name, "#999")
            ax.barh(y_offset, row["worst_case_acc"], color=bar_color, edgecolor='white', height=0.7)
            ax.text(row["worst_case_acc"] + 0.5, y_offset, f"{row['worst_case_acc']:.1f}%", va='center', fontsize=8)
            y_ticks.append(y_offset)
            y_labels.append(f"[{model_name}] #{i+1}: {row['train_combo']}")
            y_offset += 1
        y_offset += 0.5

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Worst-Case Accuracy (%)", fontsize=12)
    ax.set_title("Worst-Case Accuracy Across Top 10 Combos (All Models)", fontsize=14, fontweight='bold')
    ax.axvline(x=85, color='red', linestyle='--', alpha=0.5, label='85% threshold')
    ax.legend()
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "worst_case_barplot.png", dpi=150)
    plt.close()
    print(f"  ✓ Worst-case bar plot saved")


def viz_metric_correlation(ranked_df, model_name):
    """Correlation matrix of the 7 CRS metrics."""
    metric_cols = ["mean_acc", "worst_case_acc", "std_acc", "mean_f1",
                   "worst_class_acc", "pareto_efficiency", "ood_transfer_gap"]
    metric_labels = ["Mean Acc", "Worst-Case", "Std (↓)", "Mean F1",
                     "Worst Class", "Pareto Eff", "OOD Transfer"]

    corr = ranked_df[metric_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                xticklabels=metric_labels, yticklabels=metric_labels,
                ax=ax, vmin=-1, vmax=1, linewidths=0.5)
    ax.set_title(f"Metric Correlation Matrix ({model_name})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"metric_correlation_{model_name.replace('/', '_').replace(' ', '_')}.png", dpi=150)
    plt.close()
    print(f"  ✓ Metric correlation matrix saved")


def viz_size_distribution(top10_dfs):
    """Pie chart of combo size distribution across all Top 10 selections."""
    fig, axes = plt.subplots(1, len(top10_dfs), figsize=(7 * len(top10_dfs), 6))
    if len(top10_dfs) == 1:
        axes = [axes]

    for ax, (model_name, top10) in zip(axes, top10_dfs.items()):
        size_counts = top10["combo_size"].value_counts().sort_index()
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(size_counts)))
        wedges, texts, autotexts = ax.pie(
            size_counts.values,
            labels=[f"Size {int(s)}" for s in size_counts.index],
            autopct='%1.0f%%', colors=colors_pie,
            textprops={'fontsize': 10}
        )
        ax.set_title(f"Top 10 Size Distribution\n({model_name})", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "size_distribution_pie.png", dpi=150)
    plt.close()
    print(f"  ✓ Size distribution pie chart saved")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  SCRIPT 12: MULTI-CRITERIA TOP-10 COMBINATION SELECTION")
    print("=" * 70)

    # Save CRS weights
    with open(OUTPUT_DIR / "crs_weights.json", "w") as f:
        json.dump(CRS_WEIGHTS, f, indent=2)
    print(f"\n✓ CRS weights saved to {OUTPUT_DIR / 'crs_weights.json'}")

    # Load data
    print("\nLoading experiment data...")
    combo_df, classwise_df = load_data()
    print(f"  combo_results.csv: {len(combo_df)} rows")
    print(f"  combo_classwise.csv: {len(classwise_df)} rows")

    top10_dfs = {}

    # Automatically identify trained models present in combo_results.csv
    available_models = [m for m in combo_df["model"].unique() if m not in ZERO_SHOT_MODELS]
    target_models = [m for m in TRAINED_MODELS if m in available_models]

    # ── Process Trained Models ──
    for model_name in target_models:
        print(f"\n{'─' * 50}")
        print(f"  Processing: {model_name}")
        print(f"{'─' * 50}")

        # Compute metrics
        metrics_df = compute_metrics(combo_df, classwise_df, model_name)
        if metrics_df.empty:
            continue

        # Compute CRS and rank
        ranked_df = compute_crs(metrics_df)
        safe_name = model_name.replace("/", "_").replace(" ", "_")

        # Save full ranking
        full_rank_path = OUTPUT_DIR / f"full_ranking_{safe_name}.csv"
        ranked_df["rank"] = range(1, len(ranked_df) + 1)
        ranked_df.to_csv(full_rank_path, index=False)
        print(f"  ✓ Full ranking saved: {full_rank_path} ({len(ranked_df)} combos)")

        # Extract Top 10 and attach why_chosen rationale
        top10 = ranked_df.head(10).copy()
        top10["rank"] = range(1, 11)
        top10["why_chosen"] = [get_selection_rationale(r, model_name) for _, r in top10.iterrows()]
        top10_path = OUTPUT_DIR / f"top10_{safe_name}.csv"
        top10.to_csv(top10_path, index=False)
        print(f"  ✓ Top 10 saved: {top10_path}")

        top10_dfs[model_name] = top10

        # Print Top 10 summary
        print(f"\n  {'─' * 60}")
        print(f"  Top 10 Combinations for {model_name} (by CRS)")
        print(f"  {'─' * 60}")
        print(f"  {'Rank':<5} {'Combo':<35} {'Size':>4} {'MeanAcc':>8} {'WorstAcc':>9} {'Std':>6} {'F1':>6} {'CRS':>6}")
        print(f"  {'─' * 80}")
        for _, row in top10.iterrows():
            print(f"  {int(row['rank']):<5} {row['train_combo']:<35} {int(row['combo_size']):>4} "
                  f"{row['mean_acc']:>7.2f}% {row['worst_case_acc']:>8.2f}% {row['std_acc']:>5.2f} "
                  f"{row['mean_f1']:>5.4f} {row['crs_score']:>5.4f}")

        # Generate justification
        justification = generate_justification(top10, model_name)
        just_path = OUTPUT_DIR / f"selection_justification_{safe_name}.md"
        with open(just_path, "w", encoding="utf-8") as f:
            f.write(justification)
        print(f"\n  ✓ Selection justification saved: {just_path}")

        # Generate visualizations
        print(f"\n  Generating visualizations for {model_name}...")
        viz_crs_breakdown(top10, model_name)
        viz_radar(top10, model_name)
        viz_pareto(ranked_df, top10, model_name)
        viz_heatmap(top10, model_name)
        viz_metric_correlation(ranked_df, model_name)

    # ── Process Zero-Shot Models (Baseline) ──
    print(f"\n{'─' * 50}")
    print(f"  Processing Zero-Shot Baselines")
    print(f"{'─' * 50}")

    zs_results = []
    for model_name in ZERO_SHOT_MODELS:
        metrics_df = compute_metrics(combo_df, classwise_df, model_name)
        if not metrics_df.empty:
            metrics_df["model"] = model_name
            zs_results.append(metrics_df)

    if zs_results:
        zs_df = pd.concat(zs_results, ignore_index=True)
        zs_path = OUTPUT_DIR / "top10_zero_shot_baselines.csv"
        zs_df.to_csv(zs_path, index=False)
        print(f"  ✓ Zero-shot baselines saved: {zs_path}")

        for _, row in zs_df.iterrows():
            print(f"  {row.get('model', 'Unknown')}: Mean Acc={row['mean_acc']:.2f}%, "
                  f"Worst={row['worst_case_acc']:.2f}%, F1={row['mean_f1']:.4f}")

    # ── Cross-model visualizations ──
    if top10_dfs:
        print(f"\n  Generating cross-model visualizations...")
        viz_worst_case(top10_dfs)
        viz_size_distribution(top10_dfs)

    print(f"\n{'=' * 70}")
    print(f"  SELECTION COMPLETE — Results in: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

