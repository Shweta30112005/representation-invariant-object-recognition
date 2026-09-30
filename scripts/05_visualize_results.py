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

# Qwen3-VL is stored separately because it is evaluated by Script 7.
QWEN_RESULTS_DIR = RESULTS_DIR / "qwen3vl"
QWEN_RESULTS_CSV = QWEN_RESULTS_DIR / "qwen3vl_results.csv"
QWEN_CLASSWISE_CSV = QWEN_RESULTS_DIR / "qwen3vl_classwise.csv"
QWEN_INFERENCE_CSV = QWEN_RESULTS_DIR / "inference_time.csv"

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


def _canonical_model_from_name(name):
    """Infer the project model name from a result filename."""
    s = name.lower().replace("-", "_")
    if "resnet_50" in s:
        return "ResNet-50"
    if "vit_b_16" in s or "vit_b16" in s:
        return "ViT-B/16"
    if "dinov3" in s or "dino_v3" in s:
        return "DINOv3"
    if "clip_vit_b_32" in s or "clip_vit_b_3" in s:
        return "CLIP ViT-B/32"
    if "eva_clip" in s:
        return "EVA-CLIP (EVA02-B/16)"
    if "siglip_2" in s or "siglip2" in s:
        return "SigLIP-2 Base"
    return None


def _rename_result_columns(df):
    """Normalize common column spellings used by the project's CSV files."""
    rename = {}
    for c in df.columns:
        lc = str(c).strip().lower().replace(" ", "_").replace("-", "_")
        if lc in {"model", "model_name"}:
            rename[c] = "model"
        elif lc in {"training_setup", "train_setup", "setup", "training"}:
            rename[c] = "training_setup"
        elif lc in {"test_representation", "representation", "test_rep", "test_style"}:
            rename[c] = "test_representation"
        elif lc in {"accuracy", "acc", "accuracy_pct", "accuracy_percent"}:
            rename[c] = "accuracy"
        elif lc in {"f1_score", "f1", "macro_f1", "macro_f1_score"}:
            rename[c] = "f1_score"
    return df.rename(columns=rename)


def _normalize_one_result_file(path):
    """Read one model-specific CSV and convert it to Script-5 result schema."""
    try:
        raw = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()

    raw = _rename_result_columns(raw.copy())
    inferred_model = _canonical_model_from_name(path.name)

    # If the file already has the canonical long format, use it directly.
    if {"test_representation", "accuracy"}.issubset(raw.columns):
        out = raw.copy()
        if "model" not in out.columns and inferred_model:
            out["model"] = inferred_model
        if "training_setup" not in out.columns:
            # Zero-shot model files do not have a training setup column.
            if inferred_model in {
                "CLIP ViT-B/32", "EVA-CLIP (EVA02-B/16)", "SigLIP-2 Base"
            }:
                out["training_setup"] = "Zero-Shot"
            else:
                out["training_setup"] = "Train_Original"
        return out

    # Matrix CSVs are usually wide: rows are training setups and columns are
    # the nine test representations. Convert them to the same long format.
    rep_cols = [r for r in REPRESENTATIONS if r in raw.columns]
    if rep_cols:
        setup_col = None
        for candidate in ["training_setup", "setup", "train_setup", "Unnamed: 0"]:
            if candidate in raw.columns:
                setup_col = candidate
                break
        if setup_col is None:
            # First non-representation column is normally the setup/index.
            others = [c for c in raw.columns if c not in rep_cols]
            if others:
                setup_col = others[0]
        if setup_col is not None:
            out = raw.melt(id_vars=[setup_col], value_vars=rep_cols,
                           var_name="test_representation", value_name="accuracy")
            out = out.rename(columns={setup_col: "training_setup"})
            out["model"] = inferred_model or "Unknown"
            return out

    return pd.DataFrame()


def load_data():
    """
    Load the original combined classification CSV and, when it has been
    replaced by a model-specific run, recover the older model results from
    the individual CSVs visible in results/.

    This keeps the original Script-5 plots unchanged while making the loader
    compatible with the current results folder structure.
    """
    frames = []

    # Main combined file (whatever models are currently present in it).
    if CSV_PATH.exists():
        f = _normalize_one_result_file(CSV_PATH)
        if not f.empty:
            frames.append(f)

    # Recover model-specific result/matrix files from results/.
    for path in sorted(RESULTS_DIR.glob("*.csv")):
        if path.name in {CSV_PATH.name, CLASSWISE_CSV.name, INFERENCE_CSV.name}:
            continue
        if path.name.endswith("_backup.csv"):
            continue
        if "classwise" in path.name.lower() or "summary" in path.name.lower():
            continue
        f = _normalize_one_result_file(path)
        if not f.empty and "model" in f.columns:
            frames.append(f)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.dropna(subset=["model", "training_setup", "test_representation", "accuracy"])

    # Remove duplicate rows. The main combined CSV wins over recovered files.
    df = df.drop_duplicates(
        subset=["model", "training_setup", "test_representation"], keep="first"
    )

    # Ensure numeric metrics.
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    if "f1_score" in df.columns:
        df["f1_score"] = pd.to_numeric(df["f1_score"], errors="coerce")

    if df["accuracy"].max(skipna=True) <= 1.0:
        df["accuracy_pct"] = df["accuracy"] * 100.0
    else:
        df["accuracy_pct"] = df["accuracy"]

    return df


def load_qwen_results():
    """Load Qwen3-VL results from results/qwen3vl/."""
    if not QWEN_RESULTS_CSV.exists():
        print(f"  [INFO] Qwen results not found: {QWEN_RESULTS_CSV}")
        return pd.DataFrame()
    q = _rename_result_columns(pd.read_csv(QWEN_RESULTS_CSV))
    if q.empty or not {"test_representation", "accuracy"}.issubset(q.columns):
        return pd.DataFrame()
    q["model"] = "Qwen3-VL-8B"
    q["training_setup"] = "Zero-Shot"
    q["accuracy"] = pd.to_numeric(q["accuracy"], errors="coerce")
    q["accuracy_pct"] = q["accuracy"] * 100 if q["accuracy"].max(skipna=True) <= 1 else q["accuracy"]
    if "f1_score" in q.columns:
        q["f1_score"] = pd.to_numeric(q["f1_score"], errors="coerce")
    return q

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
        "SigLIP-2 Base",
        "Qwen3-VL-8B"
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
        "SigLIP-2 Base": "#9B59B6",
        "Qwen3-VL-8B": "#E67E22"
    }

    labels = {
        "CLIP ViT-B/32": "CLIP ViT-B/32",
        "EVA-CLIP (EVA02-B/16)": "EVA-CLIP (EVA02-B/16)",
        "SigLIP-2 Base": "SigLIP-2",
        "Qwen3-VL-8B": "Qwen3-VL-8B"
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
    Plot out-of-distribution generalization when models are trained
    ONLY on Original images, alongside zero-shot vision-language models.

    All models and representations are shown in ONE figure.
    """

    reps = get_available_reps(df)

    # ------------------------------------------------------------
    # Models included in Fig. 3
    # ------------------------------------------------------------

    model_order = [
        "CLIP ViT-B/32",
        "DINOv3",
        "EVA-CLIP (EVA02-B/16)",
        "Qwen3-VL-8B",
        "ResNet-50",
        "SigLIP-2 Base",
        "ViT-B/16"
    ]

    # ------------------------------------------------------------
    # Filter relevant results
    # ------------------------------------------------------------

    subset = df[
        (
            df["model"].isin([
                "ResNet-50",
                "ViT-B/16",
                "DINOv3"
            ])
            & (df["training_setup"] == "Train_Original")
        )
        |
        (
            df["model"].isin([
                "CLIP ViT-B/32",
                "EVA-CLIP (EVA02-B/16)",
                "SigLIP-2 Base",
                "Qwen3-VL-8B"
            ])
            & (df["training_setup"] == "Zero-Shot")
        )
    ].copy()

    # ------------------------------------------------------------
    # Pivot
    # ------------------------------------------------------------

    pivot = subset.pivot(
        index="test_representation",
        columns="model",
        values="accuracy_pct"
    )

    pivot = pivot.reindex(
        index=reps,
        columns=model_order
    )

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(18, 9)
    )

    x = np.arange(len(reps))

    # ------------------------------------------------------------
    # Model styles
    # ------------------------------------------------------------

    styles = {
        "CLIP ViT-B/32": {
            "color": "#E74C3C",
            "marker": "o",
            "linestyle": "-",
            "label": "CLIP ViT-B/32 (Zero-Shot)"
        },

        "DINOv3": {
            "color": "#E67E22",
            "marker": "s",
            "linestyle": "-",
            "label": "DINOv3 (Train: Original)"
        },

        "EVA-CLIP (EVA02-B/16)": {
            "color": "#27AE60",
            "marker": "^",
            "linestyle": "-",
            "label": "EVA-CLIP (Zero-Shot)"
        },

        "Qwen3-VL-8B": {
            "color": "#2980B9",
            "marker": "D",
            "linestyle": "-",
            "label": "Qwen3-VL-8B (Zero-Shot)"
        },

        "ResNet-50": {
            "color": "#8E44AD",
            "marker": "P",
            "linestyle": "-",
            "label": "ResNet-50 (Train: Original)"
        },

        "SigLIP-2 Base": {
            "color": "#9B59B6",
            "marker": "*",
            "linestyle": "-",
            "label": "SigLIP-2 (Zero-Shot)"
        },

        "ViT-B/16": {
            "color": "#C0392B",
            "marker": "X",
            "linestyle": "-",
            "label": "ViT-B/16 (Train: Original)"
        }
    }

    # ------------------------------------------------------------
    # Plot all models
    # ------------------------------------------------------------

    for model in model_order:

        if model not in pivot.columns:
            continue

        values = pivot[model].values
        style = styles[model]

        ax.plot(
            x,
            values,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.8,
            markersize=9,
            markeredgewidth=1.2,
            label=style["label"],
            zorder=3
        )

    # ------------------------------------------------------------
    # Annotate ONLY important low/outlier values
    #
    # This avoids the 63-label overlap in the old figure.
    # ------------------------------------------------------------

    for model in model_order:

        if model not in pivot.columns:
            continue

        values = pivot[model].values
        style = styles[model]

        for i, value in enumerate(values):

            if np.isnan(value):
                continue

            # Only annotate points below 70%.
            # These are visually important drops.
            if value < 70:

                ax.annotate(
                    f"{value:.1f}%",
                    xy=(i, value),
                    xytext=(0, -18),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=9,
                    fontweight="bold",
                    color=style["color"]
                )

    # ------------------------------------------------------------
    # Annotate Original performance
    # ------------------------------------------------------------

    original_idx = 0

    for model in model_order:

        if model not in pivot.columns:
            continue

        value = pivot.iloc[original_idx][model]

        if pd.isna(value):
            continue

        ax.annotate(
            f"{value:.1f}%",
            xy=(original_idx, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="semibold"
        )

    # ------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------

    ax.set_ylabel(
        "Classification Accuracy (%)",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Input Representation",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_title(
        "Out-of-Distribution Robustness: "
        "Generalization from Original Images to Alternative Representations",
        fontsize=18,
        fontweight="bold",
        pad=20
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        reps,
        fontsize=12,
        fontweight="semibold",
        rotation=25,
        ha="right"
    )

    # Full scale so the severe drops are visible
    ax.set_ylim(0, 105)

    ax.set_yticks(
        np.arange(0, 101, 10)
    )

    ax.tick_params(
        axis="y",
        labelsize=11
    )

    # ------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=1,
        alpha=0.45,
        zorder=0
    )

    ax.grid(
        axis="x",
        visible=False
    )

    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=11,
        frameon=True,
        facecolor="white",
        edgecolor="#BDC3C7",
        columnspacing=1.5,
        handlelength=3
    )

    # ------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------

    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.90,
        bottom=0.25
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    out_file = PLOTS_DIR / "03_train_original_generalization.png"

    plt.savefig(
        out_file,
        bbox_inches="tight",
        dpi=300
    )

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
        "SigLIP-2 Base",
        "Qwen3-VL-8B"
    ]
    zs_available = [m for m in zs_models if m in p_zs.columns]

    x = np.arange(len(reps))
    w = 0.75 / max(len(zs_available), 1)

    zs_colors = {
        "CLIP ViT-B/32": "#3498DB",
        "EVA-CLIP (EVA02-B/16)": "#1ABC9C",
        "SigLIP-2 Base": "#9B59B6",
        "Qwen3-VL-8B": "#E67E22"
    }
    zs_labels = {
        "CLIP ViT-B/32": "CLIP (Zero-Shot)",
        "EVA-CLIP (EVA02-B/16)": "EVA-CLIP (Zero-Shot)",
        "SigLIP-2 Base": "SigLIP-2 (Zero-Shot)",
        "Qwen3-VL-8B": "Qwen3-VL-8B (Zero-Shot)"
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
        ("SigLIP-2 Base", "Zero-Shot"): "SigLIP-2 (Zero-Shot)",
        ("Qwen3-VL-8B", "Zero-Shot"): "Qwen3-VL-8B (Zero-Shot)"
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

    for m in ["CLIP ViT-B/32", "EVA-CLIP (EVA02-B/16)", "SigLIP-2 Base", "Qwen3-VL-8B"]:
        m_df = df[df["model"] == m]
        if m_df.empty:
            continue

        display_name = {
            "CLIP ViT-B/32": "CLIP",
            "EVA-CLIP (EVA02-B/16)": "EVA-CLIP",
            "SigLIP-2 Base": "SigLIP-2",
            "Qwen3-VL-8B": "Qwen3-VL-8B"
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
        "SigLIP-2": "#8E44AD",
        "Qwen3-VL-8B": "#E67E22"
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


def load_classwise_data():
    """Load the original classwise CSV plus model-specific classwise CSVs and Qwen."""
    frames=[]
    if CLASSWISE_CSV.exists():
        try:
            frames.append(pd.read_csv(CLASSWISE_CSV))
        except Exception:
            pass
    for path in sorted(RESULTS_DIR.glob("*.csv")):
        if path.name in {CLASSWISE_CSV.name, CSV_PATH.name, INFERENCE_CSV.name}:
            continue
        if "classwise" not in path.name.lower():
            continue
        try:
            q=pd.read_csv(path)
        except Exception:
            continue
        if q.empty:
            continue
        q=_rename_result_columns(q)
        model=_canonical_model_from_name(path.name)
        if "model" not in q.columns and model:
            q["model"]=model
        if "training_setup" not in q.columns:
            q["training_setup"]="Zero-Shot" if model in {"CLIP ViT-B/32","EVA-CLIP (EVA02-B/16)","SigLIP-2 Base"} else "Train_All_Combined"
        if "test_representation" not in q.columns:
            q["test_representation"]="Original"
        frames.append(q)
    if QWEN_CLASSWISE_CSV.exists():
        try:
            q=pd.read_csv(QWEN_CLASSWISE_CSV)
            if not q.empty:
                q=_rename_result_columns(q)
                q["model"]="Qwen3-VL-8B"
                q["training_setup"]="Zero-Shot"
                if "test_representation" not in q.columns:
                    q["test_representation"]="Original"
                frames.append(q)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames,ignore_index=True,sort=False)


def plot_classwise_metrics(df_cw=None):
    """Plot class-wise precision and recall heatmaps."""
    if df_cw is None:
        df_cw=load_classwise_data()
    if df_cw.empty:
        print("  [SKIP] No class-wise results found. Skipping class-wise plot.")
        return

    supervised_models=["ResNet-50","ViT-B/16","DINOv3"]
    zs_models=["CLIP ViT-B/32","EVA-CLIP (EVA02-B/16)","SigLIP-2 Base","Qwen3-VL-8B"]
    focus_df=df_cw[
        ((df_cw["model"].isin(supervised_models)) & (df_cw["training_setup"]=="Train_All_Combined") & (df_cw["test_representation"]=="Original")) |
        ((df_cw["model"].isin(zs_models)) & (df_cw["test_representation"]=="Original"))
    ].copy()
    if focus_df.empty:
        focus_df=df_cw.copy()
    if focus_df.empty:
        print("  [SKIP] No data for class-wise plot.")
        return

    fig,axes=plt.subplots(1,4,figsize=(32,8))
    for idx,metric in enumerate(["accuracy","precision","recall","f1_score"]):
        if metric not in focus_df.columns:
            continue
        pivot=focus_df.pivot_table(index="class_name",columns="model",values=metric)*100
        pivot=pivot.reindex(sorted(pivot.index))
        sns.heatmap(pivot,annot=True,fmt=".1f",cmap="RdYlGn",vmin=50,vmax=100,ax=axes[idx],linewidths=0.8,linecolor="white",cbar_kws={'label':f"{'F1-Score' if metric=='f1_score' else metric.title()} (%)"})
        metric_label="F1-Score" if metric=="f1_score" else metric.title()
        axes[idx].set_title(f"Class-wise {metric_label} (%)\n(Train: All-Combined / Zero-Shot, Test: Original)",fontweight="bold",fontsize=11)
        axes[idx].set_xlabel("Model",fontweight="bold")
        axes[idx].set_ylabel("Class", fontweight="bold" if idx == 0 else "normal")
        axes[idx].tick_params(axis="x",rotation=35)
    plt.suptitle("Class-wise Classification Metrics Across Models",fontsize=16,fontweight="bold",y=1.02)
    out_file=PLOTS_DIR/"06_classwise_metrics_heatmap.png"
    plt.savefig(out_file,bbox_inches="tight",dpi=300); plt.close()
    print(f"  [Saved] {out_file.name}")

def load_inference_data():
    frames=[]
    if INFERENCE_CSV.exists():
        try: frames.append(pd.read_csv(INFERENCE_CSV))
        except Exception: pass
    for path in sorted(RESULTS_DIR.glob("*.csv")):
        if path.name in {INFERENCE_CSV.name, CSV_PATH.name, CLASSWISE_CSV.name}: continue
        if "inference" not in path.name.lower(): continue
        try: q=pd.read_csv(path)
        except Exception: continue
        if q.empty: continue
        q=_rename_result_columns(q)
        model=_canonical_model_from_name(path.name)
        if "model" not in q.columns and model: q["model"]=model
        frames.append(q)
    if QWEN_INFERENCE_CSV.exists():
        try:
            q=pd.read_csv(QWEN_INFERENCE_CSV)
            if not q.empty:
                q=_rename_result_columns(q); q["model"]="Qwen3-VL-8B"; frames.append(q)
        except Exception: pass
    if not frames: return pd.DataFrame()
    df=pd.concat(frames,ignore_index=True,sort=False)
    # normalize timing names
    rename={}
    for c in df.columns:
        lc=str(c).lower().replace(" ","_").replace("-","_")
        if lc in {"avg_inference_time_ms","average_inference_time_ms","avg_time_ms"}: rename[c]="avg_time_ms"
        elif lc in {"throughput","throughput_img_per_sec","throughput_images_per_sec"}: rename[c]="throughput_img_per_sec"
        elif lc in {"avg_inference_time_sec","average_inference_time_sec","avg_time_sec"}: rename[c]="avg_time_sec"
        elif lc in {"total_inference_time_sec","total_time_sec"}: rename[c]="total_time_sec"
        elif lc in {"num_images","n_images"}: rename[c]="num_images"
    df=df.rename(columns=rename)
    if "avg_time_ms" not in df.columns and "avg_time_sec" in df.columns:
        df["avg_time_ms"]=pd.to_numeric(df["avg_time_sec"],errors="coerce")*1000
    if "avg_time_ms" not in df.columns and {"total_time_sec","num_images"}.issubset(df.columns):
        df["avg_time_ms"]=pd.to_numeric(df["total_time_sec"],errors="coerce")/pd.to_numeric(df["num_images"],errors="coerce")*1000
    if "throughput_img_per_sec" not in df.columns and "avg_time_ms" in df.columns:
        df["throughput_img_per_sec"]=1000/pd.to_numeric(df["avg_time_ms"],errors="coerce")
    return df


def plot_inference_time(df_inf=None):
    """Plot model-wise inference time comparison bar chart."""
    if df_inf is None: df_inf=load_inference_data()
    if df_inf.empty or "model" not in df_inf.columns or "avg_time_ms" not in df_inf.columns:
        print("  [SKIP] Inference-time data not found. Skipping inference time plot.")
        return
    df_unique=df_inf.drop_duplicates(subset=["model"],keep="first").copy()
    df_unique=df_unique.sort_values("avg_time_ms",ascending=True)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,6))
    colors=["#3498DB","#E67E22","#2ECC71","#9B59B6","#1ABC9C","#8E44AD","#E74C3C"]
    bar_colors=colors[:len(df_unique)]
    bars1=ax1.barh(df_unique["model"],df_unique["avg_time_ms"],color=bar_colors,edgecolor="#2C3E50",alpha=0.9)
    for bar,val in zip(bars1,df_unique["avg_time_ms"]): ax1.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,f"{val:.2f} ms",va='center',ha='left',fontweight="bold",fontsize=10)
    ax1.set_xlabel("Average Inference Time (ms/image)",fontweight="bold"); ax1.set_title("Model Inference Latency",fontweight="bold",pad=15); ax1.grid(axis="x",linestyle="--",alpha=0.6); ax1.set_xlim(0,df_unique["avg_time_ms"].max()*1.3)
    df_unique_tput=df_unique.sort_values("throughput_img_per_sec",ascending=True)
    bars2=ax2.barh(df_unique_tput["model"],df_unique_tput["throughput_img_per_sec"],color=bar_colors,edgecolor="#2C3E50",alpha=0.9)
    for bar,val in zip(bars2,df_unique_tput["throughput_img_per_sec"]): ax2.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,f"{val:.0f} img/s",va='center',ha='left',fontweight="bold",fontsize=10)
    ax2.set_xlabel("Throughput (images/sec)",fontweight="bold"); ax2.set_title("Model Throughput",fontweight="bold",pad=15); ax2.grid(axis="x",linestyle="--",alpha=0.6); ax2.set_xlim(0,df_unique_tput["throughput_img_per_sec"].max()*1.3)
    plt.suptitle("Model-wise Computational Complexity Comparison",fontsize=16,fontweight="bold",y=1.02)
    out_file=PLOTS_DIR/"07_inference_time_comparison.png"; plt.savefig(out_file,bbox_inches="tight",dpi=300); plt.close(); print(f"  [Saved] {out_file.name}")

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
    qwen_df = load_qwen_results()
    if not qwen_df.empty:
        df = pd.concat([df, qwen_df], ignore_index=True, sort=False)
    print(f"Loaded {len(df)} evaluation records.")
    if not df.empty and "model" in df.columns:
        print("Models loaded:")
        for model_name in df["model"].dropna().unique():
            print(f"  - {model_name}")
    print()

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