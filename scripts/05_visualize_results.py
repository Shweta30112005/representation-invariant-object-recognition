"""
Script 5: Generate Publication-Quality Visualizations for BTP Results
Reads from: results/classification_results.csv
Outputs to: results/plots/
  1. 01_cross_representation_heatmaps.png (ResNet-50, ViT-B/16, DINOv3 heatmaps)
  2. 02_zero_shot_clip_vs_evaclip.png (Zero-Shot CLIP vs EVA-CLIP)
  3. 03_train_original_generalization.png (Out-of-domain robustness when trained on Original)
  4. 04_all_combined_comparison.png (Performance when trained on all representations)
  5. 05_summary_dashboard.png (Consolidated 4-panel executive visualization)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "figure.dpi": 300
})

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULTS_DIR / "classification_results.csv"

REPRESENTATIONS = ["Original", "Outline", "Dotted", "Dashed", "Sketch", "Silhouette"]
SETUPS_ORDER = [
    "Train_Original",
    "Train_Outline",
    "Train_Dotted",
    "Train_Dashed",
    "Train_Sketch",
    "Train_Silhouette",
    "Train_All_Combined"
]
SETUP_LABELS = [
    "Original",
    "Outline",
    "Dotted",
    "Dashed",
    "Sketch",
    "Silhouette",
    "All Combined"
]


def load_data():
    df = pd.read_csv(CSV_PATH)
    # Ensure accuracy in percentage format
    if df["accuracy"].max() <= 1.0:
        df["accuracy_pct"] = df["accuracy"] * 100.0
    else:
        df["accuracy_pct"] = df["accuracy"]
    return df


def plot_heatmaps(df):
    """Plot 3-panel cross-representation heatmaps for ResNet-50, ViT-B/16, DINOv3."""
    models = ["ResNet-50", "ViT-B/16", "DINOv3"]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), sharey=True)

    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])

    for i, model_name in enumerate(models):
        df_m = df[df["model"] == model_name]
        pivot = df_m.pivot(index="training_setup", columns="test_representation", values="accuracy_pct")
        pivot = pivot.reindex(index=SETUPS_ORDER, columns=REPRESENTATIONS)
        pivot.index = SETUP_LABELS

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            vmin=10,
            vmax=100,
            cbar=(i == 2),
            cbar_ax=cbar_ax if i == 2 else None,
            cbar_kws={'label': 'Accuracy (%)'} if i == 2 else None,
            ax=axes[i],
            linewidths=1.0,
            linecolor="white"
        )

        axes[i].set_title(f"{model_name}\nCross-Representation Matrix", fontweight="bold", pad=12)
        axes[i].set_xlabel("Test Representation", fontweight="bold", labelpad=8)
        if i == 0:
            axes[i].set_ylabel("Training Setup", fontweight="bold", labelpad=8)
        else:
            axes[i].set_ylabel("")

        axes[i].tick_params(axis="x", rotation=35)
        axes[i].tick_params(axis="y", rotation=0)

    plt.suptitle("Cross-Representation Generalization Matrices Across Deep Architectures", fontsize=16, fontweight="bold", y=1.02)
    out_file = PLOTS_DIR / "01_cross_representation_heatmaps.png"
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [Saved] {out_file.name}")


def plot_zero_shot_comparison(df):
    """Plot bar chart comparing Zero-Shot CLIP ViT-B/32 vs EVA-CLIP (EVA02-B/16)."""
    df_zs = df[df["training_setup"] == "Zero-Shot"].copy()
    
    # Pivot to order representations
    pivot = df_zs.pivot(index="test_representation", columns="model", values="accuracy_pct")
    pivot = pivot.reindex(REPRESENTATIONS)

    x = np.arange(len(REPRESENTATIONS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    color_clip = "#4A90E2"
    color_evaclip = "#50E3C2"

    bars1 = ax.bar(x - width/2, pivot["CLIP ViT-B/32"], width, label="CLIP ViT-B/32", color=color_clip, edgecolor="#2C3E50", alpha=0.9)
    bars2 = ax.bar(x + width/2, pivot["EVA-CLIP (EVA02-B/16)"], width, label="EVA-CLIP (EVA02-B/16)", color=color_evaclip, edgecolor="#2C3E50", alpha=0.9)

    # Add data labels
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight="semibold")

    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight="semibold")

    ax.set_ylabel("Zero-Shot Accuracy (%)", fontweight="bold")
    ax.set_title("Zero-Shot Representation Robustness: CLIP vs EVA-CLIP", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(REPRESENTATIONS, fontweight="semibold")
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    out_file = PLOTS_DIR / "02_zero_shot_clip_vs_evaclip.png"
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [Saved] {out_file.name}")


def plot_train_original_generalization(df):
    """
    Plot out-of-distribution generalization when models are trained ONLY on Original images,
    alongside zero-shot vision-language models.
    """
    fig, ax = plt.subplots(figsize=(12, 6.5))

    # Filter models trained on Original or Zero-Shot
    subset = df[
        ((df["model"].isin(["ResNet-50", "ViT-B/16", "DINOv3"])) & (df["training_setup"] == "Train_Original")) |
        (df["training_setup"] == "Zero-Shot")
    ].copy()

    # Create mapping labels
    label_map = {
        ("ResNet-50", "Train_Original"): "ResNet-50 (Train: Original)",
        ("ViT-B/16", "Train_Original"): "ViT-B/16 (Train: Original)",
        ("DINOv3", "Train_Original"): "DINOv3 (Train: Original)",
        ("CLIP ViT-B/32", "Zero-Shot"): "CLIP ViT-B/32 (Zero-Shot)",
        ("EVA-CLIP (EVA02-B/16)", "Zero-Shot"): "EVA-CLIP (Zero-Shot)"
    }

    subset["model_label"] = subset.apply(lambda r: label_map.get((r["model"], r["training_setup"]), r["model"]), axis=1)

    pivot = subset.pivot(index="test_representation", columns="model_label", values="accuracy_pct")
    pivot = pivot.reindex(REPRESENTATIONS)

    colors = ["#E74C3C", "#E67E22", "#27AE60", "#2980B9", "#8E44AD"]
    markers = ["o", "s", "^", "D", "P"]

    for i, col in enumerate(pivot.columns):
        ax.plot(pivot.index, pivot[col], marker=markers[i], linewidth=2.5, markersize=8, label=col, color=colors[i])
        for x_val, y_val in zip(pivot.index, pivot[col]):
            ax.annotate(f"{y_val:.1f}%", xy=(x_val, y_val), xytext=(0, 7),
                        textcoords="offset points", ha='center', fontsize=9, fontweight="medium")

    ax.set_ylabel("Classification Accuracy (%)", fontweight="bold")
    ax.set_title("Out-of-Distribution Robustness: Generalization from Original Images to Abstract Variants", fontweight="bold", pad=15)
    ax.set_ylim(25, 108)
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.6)

    out_file = PLOTS_DIR / "03_train_original_generalization.png"
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [Saved] {out_file.name}")


def plot_all_combined_comparison(df):
    """Plot performance across all representations when models are trained on All Combined."""
    fig, ax = plt.subplots(figsize=(11, 6))

    df_comb = df[df["training_setup"] == "Train_All_Combined"].copy()
    pivot = df_comb.pivot(index="test_representation", columns="model", values="accuracy_pct")
    pivot = pivot.reindex(REPRESENTATIONS)

    x = np.arange(len(REPRESENTATIONS))
    width = 0.25

    colors = ["#3498DB", "#E67E22", "#2ECC71"]
    models = ["ResNet-50", "ViT-B/16", "DINOv3"]

    for i, m in enumerate(models):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, pivot[m], width, label=m, color=colors[i], edgecolor="#2C3E50", alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 4), textcoords="offset points", ha='center', fontsize=9, fontweight="bold")

    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Multi-Domain Training: Performance when Trained on All Combined Representations", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(REPRESENTATIONS, fontweight="semibold")
    ax.set_ylim(70, 103)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    out_file = PLOTS_DIR / "04_all_combined_comparison.png"
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [Saved] {out_file.name}")


def plot_summary_dashboard(df):
    """Generate a comprehensive 4-panel executive summary dashboard."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # 1. Top-Left: Zero-Shot CLIP vs EVA-CLIP
    ax1 = fig.add_subplot(gs[0, 0])
    df_zs = df[df["training_setup"] == "Zero-Shot"].copy()
    p_zs = df_zs.pivot(index="test_representation", columns="model", values="accuracy_pct").reindex(REPRESENTATIONS)
    x = np.arange(len(REPRESENTATIONS))
    w = 0.35
    ax1.bar(x - w/2, p_zs["CLIP ViT-B/32"], w, label="CLIP ViT-B/32", color="#3498DB")
    ax1.bar(x + w/2, p_zs["EVA-CLIP (EVA02-B/16)"], w, label="EVA-CLIP (EVA02-B/16)", color="#1ABC9C")
    ax1.set_title("(A) Zero-Shot Generalization: CLIP vs EVA-CLIP", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(REPRESENTATIONS, rotation=25)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(50, 100)
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Top-Right: Train on Original Out-of-Domain Generalization
    ax2 = fig.add_subplot(gs[0, 1])
    subset = df[((df["model"].isin(["ResNet-50", "ViT-B/16", "DINOv3"])) & (df["training_setup"] == "Train_Original")) |
                (df["training_setup"] == "Zero-Shot")].copy()
    label_map = {
        ("ResNet-50", "Train_Original"): "ResNet-50",
        ("ViT-B/16", "Train_Original"): "ViT-B/16",
        ("DINOv3", "Train_Original"): "DINOv3",
        ("CLIP ViT-B/32", "Zero-Shot"): "CLIP (Zero-Shot)",
        ("EVA-CLIP (EVA02-B/16)", "Zero-Shot"): "EVA-CLIP (Zero-Shot)"
    }
    subset["model_label"] = subset.apply(lambda r: label_map.get((r["model"], r["training_setup"]), r["model"]), axis=1)
    p_gen = subset.pivot(index="test_representation", columns="model_label", values="accuracy_pct").reindex(REPRESENTATIONS)
    for col in p_gen.columns:
        ax2.plot(p_gen.index, p_gen[col], marker='o', linewidth=2, label=col)
    ax2.set_title("(B) Out-of-Domain Generalization (Train on Original Only)", fontweight="bold")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xticks(range(len(REPRESENTATIONS)))
    ax2.set_xticklabels(REPRESENTATIONS, rotation=25)
    ax2.set_ylim(30, 105)
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Bottom-Left: Average Out-of-Domain Generalization by Architecture
    ax3 = fig.add_subplot(gs[1, 0])
    # Compute in-domain vs out-of-domain averages for each model
    ood_stats = []
    for m in ["ResNet-50", "ViT-B/16", "DINOv3"]:
        m_df = df[df["model"] == m]
        in_domain = []
        out_domain = []
        for rep in ["Original", "Outline", "Dotted", "Dashed", "Sketch", "Silhouette"]:
            s_name = f"Train_{rep}"
            val_in = m_df[(m_df["training_setup"] == s_name) & (m_df["test_representation"] == rep)]["accuracy_pct"].values
            if len(val_in) > 0:
                in_domain.append(val_in[0])
            val_out = m_df[(m_df["training_setup"] == s_name) & (m_df["test_representation"] != rep)]["accuracy_pct"].values
            out_domain.extend(val_out)
        ood_stats.append({
            "model": m,
            "In-Domain Avg": np.mean(in_domain),
            "Out-of-Domain Avg": np.mean(out_domain)
        })

    # Zero-shot as out-of-domain benchmark
    for m in ["CLIP ViT-B/32", "EVA-CLIP (EVA02-B/16)"]:
        m_df = df[df["model"] == m]
        ood_stats.append({
            "model": m.split()[0],
            "In-Domain Avg": m_df[m_df["test_representation"] == "Original"]["accuracy_pct"].values[0],
            "Out-of-Domain Avg": m_df[m_df["test_representation"] != "Original"]["accuracy_pct"].mean()
        })

    df_ood = pd.DataFrame(ood_stats)
    x_idx = np.arange(len(df_ood))
    w = 0.35
    b1 = ax3.bar(x_idx - w/2, df_ood["In-Domain Avg"], w, label="In-Domain (Source Style)", color="#2C3E50")
    b2 = ax3.bar(x_idx + w/2, df_ood["Out-of-Domain Avg"], w, label="Cross-Domain (Abstract Variants)", color="#E74C3C")
    for bar in b1:
        h = bar.get_height()
        ax3.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in b2:
        h = bar.get_height()
        ax3.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    ax3.set_title("(C) In-Domain vs Cross-Domain Generalization Gap", fontweight="bold")
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(df_ood["model"], rotation=15)
    ax3.set_ylabel("Mean Accuracy (%)")
    ax3.set_ylim(0, 110)
    ax3.legend(loc="upper right")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    # 4. Bottom-Right: Macro F1-Score Breakdown
    ax4 = fig.add_subplot(gs[1, 1])
    f1_summary = []
    for m in ["ResNet-50", "ViT-B/16", "DINOv3"]:
        f1_val = df[(df["model"] == m) & (df["training_setup"] == "Train_All_Combined")]["f1_score"].mean()
        f1_summary.append({"Model": m, "Mode": "All-Combined", "F1": f1_val})
    for m in ["CLIP ViT-B/32", "EVA-CLIP (EVA02-B/16)"]:
        f1_val = df[df["model"] == m]["f1_score"].mean()
        f1_summary.append({"Model": m.replace(" (EVA02-B/16)", ""), "Mode": "Zero-Shot", "F1": f1_val})

    df_f1 = pd.DataFrame(f1_summary)
    bar_f1 = ax4.bar(df_f1["Model"], df_f1["F1"], color=["#3498DB", "#E67E22", "#2ECC71", "#9B59B6", "#1ABC9C"], edgecolor="#2C3E50")
    for bar in bar_f1:
        h = bar.get_height()
        ax4.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight="bold")
    ax4.set_title("(D) Overall Mean Macro F1-Score Across All 6 Representations", fontweight="bold")
    ax4.set_ylabel("Macro F1-Score")
    ax4.set_ylim(0.5, 1.0)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("BTP Visual Representation Generalization: Executive Summary Dashboard", fontsize=17, fontweight="bold", y=0.99)
    out_file = PLOTS_DIR / "05_summary_dashboard.png"
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [Saved] {out_file.name}")


def main():
    print("=" * 60)
    print("Generating High-Resolution Result Visualizations")
    print(f"Source: {CSV_PATH}")
    print(f"Output: {PLOTS_DIR}")
    print("=" * 60)

    if not CSV_PATH.exists():
        print(f"[FAIL] {CSV_PATH} not found!")
        return

    df = load_data()
    print(f"Loaded {len(df)} evaluation records.\n")

    print("Generating figures...")
    plot_heatmaps(df)
    plot_zero_shot_comparison(df)
    plot_train_original_generalization(df)
    plot_all_combined_comparison(df)
    plot_summary_dashboard(df)

    print("\n" + "=" * 60)
    print(f"[OK] All 5 visualization figures saved in: {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
