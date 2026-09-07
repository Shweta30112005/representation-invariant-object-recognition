"""
Script 5: Generate Publication-Quality Visualizations for BTP Results
Reads from: results/classification_results.csv, results/classwise_metrics.csv, results/inference_time.csv
Outputs to: results/plots/
  1. 01_cross_representation_heatmaps.png (ResNet-50, ViT-B/16, DINOv3 heatmaps)
  2. 02_zero_shot_clip_vs_evaclip.png (Zero-Shot CLIP vs EVA-CLIP vs SigLIP-2)
  3. 03_train_original_generalization.png (Out-of-domain robustness when trained on Original)
  4. 04_all_combined_comparison.png (Performance when trained on all representations)
  5. 05_summary_dashboard.png (Consolidated 4-panel executive visualization)
  6. 06_classwise_metrics_heatmap.png (Per-class precision/recall heatmaps)
  7. 07_inference_time_comparison.png (Model-wise inference time comparison)
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
CLASSWISE_CSV = RESULTS_DIR / "classwise_metrics.csv"
INFERENCE_CSV = RESULTS_DIR / "inference_time.csv"

REPRESENTATIONS = [
    "Original", "Outline", "Dotted", "Dashed", "Sketch", "Silhouette",
    "ColorTint_Red", "ColorTint_Green", "ColorTint_Blue"
]
SETUPS_ORDER = [
    "Train_Original",
    "Train_Outline",
    "Train_Dotted",
    "Train_Dashed",
    "Train_Sketch",
    "Train_Silhouette",
    "Train_ColorTint_Red",
    "Train_ColorTint_Green",
    "Train_ColorTint_Blue",
    "Train_All_Combined"
]
SETUP_LABELS = [
    "Original",
    "Outline",
    "Dotted",
    "Dashed",
    "Sketch",
    "Silhouette",
    "ColorTint Red",
    "ColorTint Green",
    "ColorTint Blue",
    "All Combined"
]


def load_data():
    df = pd.read_csv(CSV_PATH)
    if "class_name" in df.columns:
        df_all = df[df["class_name"] == "ALL"]
        if not df_all.empty:
            df = df_all.copy()
        else:
            df = df.groupby(["model", "training_setup", "test_representation"]).agg({
                "accuracy": "mean", "f1_score": "mean"
            }).reset_index()
    # Ensure accuracy in percentage format
    if df["accuracy"].max() <= 1.0:
        df["accuracy_pct"] = df["accuracy"] * 100.0
    else:
        df["accuracy_pct"] = df["accuracy"]
    return df


def get_available_reps(df):
    """Get representations that are actually present in the data."""
    available = df["test_representation"].unique().tolist()
    return [r for r in REPRESENTATIONS if r in available]


def get_available_setups(df):
    """Get training setups that are actually present in the data."""
    available = df["training_setup"].unique().tolist()
    setups = [s for s in SETUPS_ORDER if s in available]
    labels = [SETUP_LABELS[SETUPS_ORDER.index(s)] for s in setups]
    return setups, labels


def plot_heatmaps(df):
    """Plot 3-panel cross-representation heatmaps for ResNet-50, ViT-B/16, DINOv3."""
    models = ["ResNet-50", "ViT-B/16", "DINOv3"]
    reps = get_available_reps(df)
    setups, labels = get_available_setups(df)

    fig, axes = plt.subplots(1, 3, figsize=(max(21, 7 * len(reps) / 6), max(6.5, len(setups) * 0.8)), sharey=True)

    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])

    for i, model_name in enumerate(models):
        df_m = df[df["model"] == model_name]
        pivot = df_m.pivot(index="training_setup", columns="test_representation", values="accuracy_pct")
        pivot = pivot.reindex(index=setups, columns=reps)
        pivot.index = labels

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
    """Plot bar chart comparing Zero-Shot CLIP, EVA-CLIP, and SigLIP-2."""
    df_zs = df[df["training_setup"] == "Zero-Shot"].copy()
    reps = get_available_reps(df_zs)

    # Pivot to order representations
    pivot = df_zs.pivot(index="test_representation", columns="model", values="accuracy_pct")
    pivot = pivot.reindex(reps)

    models = [
        "CLIP ViT-B/32",
        "EVA-CLIP (EVA02-B/16)",
        "SigLIP-2 Base"
    ]
    available_models = [m for m in models if m in pivot.columns]

    if not available_models:
        print("  [WARN] No zero-shot model results found. Skipping zero-shot plot.")
        return

    x = np.arange(len(reps))
    width = 0.75 / len(available_models)

    fig, ax = plt.subplots(figsize=(max(12, len(reps) * 1.5), 6.5))

    colors = {
        "CLIP ViT-B/32": "#4A90E2",
        "EVA-CLIP (EVA02-B/16)": "#50E3C2",
        "SigLIP-2 Base": "#9B59B6"
    }

    labels = {
        "CLIP ViT-B/32": "CLIP ViT-B/32",
        "EVA-CLIP (EVA02-B/16)": "EVA-CLIP (EVA02-B/16)",
        "SigLIP-2 Base": "SigLIP-2"
    }

    for i, model_name in enumerate(available_models):
        offset = (i - (len(available_models) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            pivot[model_name],
            width,
            label=labels[model_name],
            color=colors[model_name],
            edgecolor="#2C3E50",
            alpha=0.9
        )

        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.annotate(
                    f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="semibold"
                )

    ax.set_ylabel("Zero-Shot Accuracy (%)", fontweight="bold")
    ax.set_title(
        "Zero-Shot Representation Robustness: CLIP vs EVA-CLIP vs SigLIP-2",
        fontweight="bold",
        pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels(reps, fontweight="semibold", rotation=30, ha="right")
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
    reps = get_available_reps(df)
    fig, ax = plt.subplots(figsize=(max(12, len(reps) * 1.5), 6.5))

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
        ("EVA-CLIP (EVA02-B/16)", "Zero-Shot"): "EVA-CLIP (Zero-Shot)",
        ("SigLIP-2 Base", "Zero-Shot"): "SigLIP-2 (Zero-Shot)"
    }

    subset["model_label"] = subset.apply(lambda r: label_map.get((r["model"], r["training_setup"]), r["model"]), axis=1)

    pivot = subset.pivot(index="test_representation", columns="model_label", values="accuracy_pct")
    pivot = pivot.reindex(reps)

    colors = ["#E74C3C", "#E67E22", "#27AE60", "#2980B9", "#8E44AD", "#9B59B6"]
    markers = ["o", "s", "^", "D", "P", "*"]

    for i, col in enumerate(pivot.columns):
        ax.plot(pivot.index, pivot[col], marker=markers[i % len(markers)], linewidth=2.5, markersize=8, label=col, color=colors[i % len(colors)])
        for x_val, y_val in zip(pivot.index, pivot[col]):
            if not np.isnan(y_val):
                ax.annotate(f"{y_val:.1f}%", xy=(x_val, y_val), xytext=(0, 7),
                            textcoords="offset points", ha='center', fontsize=8, fontweight="medium")

    ax.set_ylabel("Classification Accuracy (%)", fontweight="bold")
    ax.set_title("Out-of-Distribution Robustness: Generalization from Original Images to Abstract Variants", fontweight="bold", pad=15)
    ax.set_ylim(25, 108)
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=30, ha="right")

    out_file = PLOTS_DIR / "03_train_original_generalization.png"
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [Saved] {out_file.name}")


def plot_all_combined_comparison(df):
    """Plot performance across all representations when models are trained on All Combined."""
    reps = get_available_reps(df)
    fig, ax = plt.subplots(figsize=(max(11, len(reps) * 1.3), 6))

    df_comb = df[df["training_setup"] == "Train_All_Combined"].copy()
    pivot = df_comb.pivot(index="test_representation", columns="model", values="accuracy_pct")
    pivot = pivot.reindex(reps)

    x = np.arange(len(reps))
    width = 0.25

    colors = ["#3498DB", "#E67E22", "#2ECC71"]
    models = ["ResNet-50", "ViT-B/16", "DINOv3"]

    for i, m in enumerate(models):
        if m not in pivot.columns:
            continue
        offset = (i - 1) * width
        bars = ax.bar(x + offset, pivot[m], width, label=m, color=colors[i], edgecolor="#2C3E50", alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 4), textcoords="offset points", ha='center', fontsize=8, fontweight="bold")

    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Multi-Domain Training: Performance when Trained on All Combined Representations", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(reps, fontweight="semibold", rotation=30, ha="right")
    ax.set_ylim(70, 103)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    out_file = PLOTS_DIR / "04_all_combined_comparison.png"
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [Saved] {out_file.name}")


def plot_summary_dashboard(df):
    """Generate a comprehensive 4-panel executive summary dashboard."""
    reps = get_available_reps(df)
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # 1. Top-Left: Zero-Shot CLIP vs EVA-CLIP vs SigLIP-2
    ax1 = fig.add_subplot(gs[0, 0])
    df_zs = df[df["training_setup"] == "Zero-Shot"].copy()
    p_zs = df_zs.pivot(index="test_representation", columns="model", values="accuracy_pct").reindex(reps)

    zs_models = [
        "CLIP ViT-B/32",
        "EVA-CLIP (EVA02-B/16)",
        "SigLIP-2 Base"
    ]
    zs_available = [m for m in zs_models if m in p_zs.columns]

    x = np.arange(len(reps))
    w = 0.75 / max(len(zs_available), 1)

    zs_colors = {
        "CLIP ViT-B/32": "#3498DB",
        "EVA-CLIP (EVA02-B/16)": "#1ABC9C",
        "SigLIP-2 Base": "#9B59B6"
    }
    zs_labels = {
        "CLIP ViT-B/32": "CLIP (Zero-Shot)",
        "EVA-CLIP (EVA02-B/16)": "EVA-CLIP (Zero-Shot)",
        "SigLIP-2 Base": "SigLIP-2 (Zero-Shot)"
    }

    for i, m in enumerate(zs_available):
        offset = (i - (len(zs_available) - 1) / 2) * w
        ax1.bar(
            x + offset,
            p_zs[m],
            w,
            label=zs_labels[m],
            color=zs_colors[m]
        )

    ax1.set_title("(A) Zero-Shot Generalization", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(reps, rotation=35, fontsize=8)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(50, 100)
    ax1.legend(loc="upper right", fontsize=7)
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
        ("EVA-CLIP (EVA02-B/16)", "Zero-Shot"): "EVA-CLIP (Zero-Shot)",
        ("SigLIP-2 Base", "Zero-Shot"): "SigLIP-2 (Zero-Shot)"
    }
    subset["model_label"] = subset.apply(lambda r: label_map.get((r["model"], r["training_setup"]), r["model"]), axis=1)
    p_gen = subset.pivot(index="test_representation", columns="model_label", values="accuracy_pct").reindex(reps)
    for col in p_gen.columns:
        ax2.plot(p_gen.index, p_gen[col], marker='o', linewidth=2, label=col)
    ax2.set_title("(B) Out-of-Domain Generalization (Train on Original Only)", fontweight="bold")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xticks(range(len(reps)))
    ax2.set_xticklabels(reps, rotation=35, fontsize=8)
    ax2.set_ylim(30, 105)
    ax2.legend(loc="lower left", fontsize=7)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Bottom-Left: Average Out-of-Domain Generalization by Architecture
    ax3 = fig.add_subplot(gs[1, 0])
    ood_stats = []
    for m in ["ResNet-50", "ViT-B/16", "DINOv3"]:
        m_df = df[df["model"] == m]
        in_domain = []
        out_domain = []
        for rep in reps:
            s_name = f"Train_{rep}"
            val_in = m_df[(m_df["training_setup"] == s_name) & (m_df["test_representation"] == rep)]["accuracy_pct"].values
            if len(val_in) > 0:
                in_domain.append(val_in[0])
            val_out = m_df[(m_df["training_setup"] == s_name) & (m_df["test_representation"] != rep)]["accuracy_pct"].values
            out_domain.extend(val_out)
        if in_domain:
            ood_stats.append({
                "model": m,
                "In-Domain Avg": np.mean(in_domain),
                "Out-of-Domain Avg": np.mean(out_domain) if out_domain else 0
            })

    for m in ["CLIP ViT-B/32", "EVA-CLIP (EVA02-B/16)", "SigLIP-2 Base"]:
        m_df = df[df["model"] == m]
        if m_df.empty:
            continue

        display_name = {
            "CLIP ViT-B/32": "CLIP",
            "EVA-CLIP (EVA02-B/16)": "EVA-CLIP",
            "SigLIP-2 Base": "SigLIP-2"
        }[m]

        original_values = m_df[m_df["test_representation"] == "Original"]["accuracy_pct"].values

        ood_stats.append({
            "model": display_name,
            "In-Domain Avg": original_values[0] if len(original_values) > 0 else np.nan,
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
    for m in ["CLIP ViT-B/32", "EVA-CLIP (EVA02-B/16)", "SigLIP-2 Base"]:
        m_df = df[df["model"] == m]
        if m_df.empty:
            continue

        display_name = {
            "CLIP ViT-B/32": "CLIP ViT-B/32",
            "EVA-CLIP (EVA02-B/16)": "EVA-CLIP",
            "SigLIP-2 Base": "SigLIP-2"
        }[m]

        f1_val = m_df["f1_score"].mean()
        f1_summary.append({"Model": display_name, "Mode": "Zero-Shot", "F1": f1_val})

    df_f1 = pd.DataFrame(f1_summary)
    f1_colors = {
        "ResNet-50": "#3498DB",
        "ViT-B/16": "#E67E22",
        "DINOv3": "#2ECC71",
        "CLIP ViT-B/32": "#9B59B6",
        "EVA-CLIP": "#1ABC9C",
        "SigLIP-2": "#8E44AD"
    }
    bar_colors = [f1_colors.get(m, "#7F8C8D") for m in df_f1["Model"]]

    bar_f1 = ax4.bar(
        df_f1["Model"],
        df_f1["F1"],
        color=bar_colors,
        edgecolor="#2C3E50"
    )
    for bar in bar_f1:
        h = bar.get_height()
        ax4.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight="bold")
    ax4.set_title("(D) Overall Mean Macro F1-Score Across All Representations", fontweight="bold")
    ax4.set_ylabel("Macro F1-Score")
    ax4.set_ylim(0.5, 1.0)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("BTP Visual Representation Generalization: Executive Summary Dashboard", fontsize=17, fontweight="bold", y=0.99)
    out_file = PLOTS_DIR / "05_summary_dashboard.png"
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [Saved] {out_file.name}")


def plot_classwise_metrics(df_cw=None):
    """
    Plot class-wise precision and recall heatmaps.
    Uses results/classwise_metrics.csv if available.
    """
    if df_cw is None:
        if not CLASSWISE_CSV.exists():
            print("  [SKIP] classwise_metrics.csv not found. Skipping class-wise plot.")
            return
        df_cw = pd.read_csv(CLASSWISE_CSV)

    if df_cw.empty:
        print("  [SKIP] classwise_metrics.csv is empty. Skipping class-wise plot.")
        return

    # Focus on Train_All_Combined for supervised models and Zero-Shot for VL models
    supervised_models = ["ResNet-50", "ViT-B/16", "DINOv3"]
    zs_models = ["CLIP ViT-B/32", "EVA-CLIP (EVA02-B/16)", "SigLIP-2 Base"]

    focus_df = df_cw[
        ((df_cw["model"].isin(supervised_models)) & (df_cw["training_setup"] == "Train_All_Combined") & (df_cw["test_representation"] == "Original")) |
        ((df_cw["model"].isin(zs_models)) & (df_cw["test_representation"] == "Original"))
    ].copy()

    if focus_df.empty:
        # Fallback: use any available data
        focus_df = df_cw.groupby(["model", "class_name"]).first().reset_index()

    if focus_df.empty:
        print("  [SKIP] No data for class-wise plot.")
        return

    fig, axes = plt.subplots(1, 4, figsize=(32, 8))

    for idx, metric in enumerate(["accuracy", "precision", "recall", "f1_score"]):
        pivot = focus_df.pivot_table(
            index="class_name", columns="model", values=metric
        ) * 100
        pivot = pivot.reindex(sorted(pivot.index))

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            vmin=50,
            vmax=100,
            ax=axes[idx],
            linewidths=0.8,
            linecolor="white",
            cbar_kws={'label': f"{'F1-Score' if metric == 'f1_score' else metric.title()} (%)"}
        )
        metric_label = "F1-Score" if metric == "f1_score" else metric.title()
        axes[idx].set_title(f"Class-wise {metric_label} (%)\n(Train: All-Combined / Zero-Shot, Test: Original)", fontweight="bold", fontsize=11)
        axes[idx].set_xlabel("Model", fontweight="bold")
        if idx == 0:
            axes[idx].set_ylabel("Class", fontweight="bold")
        else:
            axes[idx].set_ylabel("")
        axes[idx].tick_params(axis="x", rotation=35)

    plt.suptitle("Class-wise Classification Metrics Across Models", fontsize=16, fontweight="bold", y=1.02)
    out_file = PLOTS_DIR / "06_classwise_metrics_heatmap.png"
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [Saved] {out_file.name}")


def plot_inference_time(df_inf=None):
    """
    Plot model-wise inference time comparison bar chart.
    Uses results/inference_time.csv if available.
    """
    if df_inf is None:
        if not INFERENCE_CSV.exists():
            print("  [SKIP] inference_time.csv not found. Skipping inference time plot.")
            return
        df_inf = pd.read_csv(INFERENCE_CSV)

    if df_inf.empty:
        print("  [SKIP] inference_time.csv is empty. Skipping inference time plot.")
        return

    # Take one measurement per model (first available)
    df_unique = df_inf.drop_duplicates(subset=["model"], keep="first").copy()
    df_unique = df_unique.sort_values("avg_time_ms", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Average inference time (ms/image)
    colors = ["#3498DB", "#E67E22", "#2ECC71", "#9B59B6", "#1ABC9C", "#8E44AD", "#E74C3C"]
    bar_colors = colors[:len(df_unique)]

    bars1 = ax1.barh(df_unique["model"], df_unique["avg_time_ms"], color=bar_colors, edgecolor="#2C3E50", alpha=0.9)
    for bar, val in zip(bars1, df_unique["avg_time_ms"]):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f} ms", va='center', ha='left', fontweight="bold", fontsize=10)

    ax1.set_xlabel("Average Inference Time (ms/image)", fontweight="bold")
    ax1.set_title("Model Inference Latency", fontweight="bold", pad=15)
    ax1.grid(axis="x", linestyle="--", alpha=0.6)
    ax1.set_xlim(0, df_unique["avg_time_ms"].max() * 1.3)

    # Right: Throughput (images/sec)
    df_unique_tput = df_unique.sort_values("throughput_img_per_sec", ascending=True)
    bars2 = ax2.barh(df_unique_tput["model"], df_unique_tput["throughput_img_per_sec"], color=bar_colors, edgecolor="#2C3E50", alpha=0.9)
    for bar, val in zip(bars2, df_unique_tput["throughput_img_per_sec"]):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{val:.0f} img/s", va='center', ha='left', fontweight="bold", fontsize=10)

    ax2.set_xlabel("Throughput (images/sec)", fontweight="bold")
    ax2.set_title("Model Throughput", fontweight="bold", pad=15)
    ax2.grid(axis="x", linestyle="--", alpha=0.6)
    ax2.set_xlim(0, df_unique_tput["throughput_img_per_sec"].max() * 1.3)

    plt.suptitle("Model-wise Computational Complexity Comparison", fontsize=16, fontweight="bold", y=1.02)
    out_file = PLOTS_DIR / "07_inference_time_comparison.png"
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
    plot_classwise_metrics()
    plot_inference_time()

    total_plots = 5
    if CLASSWISE_CSV.exists():
        total_plots += 1
    if INFERENCE_CSV.exists():
        total_plots += 1

    print("\n" + "=" * 60)
    print(f"[OK] All {total_plots} visualization figures saved in: {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
