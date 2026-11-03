"""
Script 11: Analyze 511-Combination Experiment Results
=====================================================

Reads the results from 10_combo_train_eval.py and generates:
1. Per-model 511×9 accuracy matrices (CSV)
2. Summary statistics by combo size
3. Best/worst combinations analysis
4. Cross-variant transfer heatmaps
5. Comprehensive markdown report

Usage:
    python 11_analyze_combo_results.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

# ─────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "combo_experiment"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

REPRESENTATIONS = [
    "Original", "Outline", "Dotted", "Dashed", "Sketch",
    "Silhouette", "ColorTint_Red", "ColorTint_Green", "ColorTint_Blue"
]

SHORT_LABELS = {
    "Original": "Org", "Outline": "Out", "Dotted": "Dot",
    "Dashed": "Dsh", "Sketch": "Skt", "Silhouette": "Sil",
    "ColorTint_Red": "Red", "ColorTint_Green": "Grn", "ColorTint_Blue": "Blu",
}


def load_results():
    """Load combo experiment results."""
    results_csv = RESULTS_DIR / "combo_results.csv"
    classwise_csv = RESULTS_DIR / "combo_classwise.csv"
    inference_csv = RESULTS_DIR / "combo_inference.csv"

    df_results = pd.read_csv(results_csv) if results_csv.exists() else pd.DataFrame()
    df_classwise = pd.read_csv(classwise_csv) if classwise_csv.exists() else pd.DataFrame()
    df_inference = pd.read_csv(inference_csv) if inference_csv.exists() else pd.DataFrame()

    return df_results, df_classwise, df_inference


def generate_model_matrices(df):
    """Generate and save 511×9 matrices per model."""
    matrices = {}

    for model_name in df["model"].unique():
        df_model = df[df["model"] == model_name].copy()

        # Accuracy matrix
        pivot_acc = df_model.pivot_table(
            index="train_combo",
            columns="test_representation",
            values="accuracy",
            aggfunc="first"
        ) * 100

        # F1 matrix
        pivot_f1 = df_model.pivot_table(
            index="train_combo",
            columns="test_representation",
            values="f1",
            aggfunc="first"
        )

        # Precision matrix
        pivot_prec = df_model.pivot_table(
            index="train_combo",
            columns="test_representation",
            values="precision",
            aggfunc="first"
        )

        # Recall matrix
        pivot_rec = df_model.pivot_table(
            index="train_combo",
            columns="test_representation",
            values="recall",
            aggfunc="first"
        )

        # Order columns
        ordered_cols = [c for c in REPRESENTATIONS if c in pivot_acc.columns]

        for pivot, metric_name in [
            (pivot_acc, "accuracy_pct"),
            (pivot_f1, "f1"),
            (pivot_prec, "precision"),
            (pivot_rec, "recall")
        ]:
            pivot = pivot.reindex(columns=ordered_cols)

            # Add average column
            pivot["Average"] = pivot.mean(axis=1)

            # Sort by average descending
            pivot = pivot.sort_values("Average", ascending=False)

            safe_name = model_name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
            filepath = ANALYSIS_DIR / f"{safe_name}_{metric_name}_matrix.csv"
            pivot.round(4).to_csv(filepath)

            if metric_name == "accuracy_pct":
                main_matrix_path = RESULTS_DIR / f"matrix_511x9_{safe_name}.csv"
                pivot.round(2).to_csv(main_matrix_path)
                print(f"    [511x9 MATRIX] {main_matrix_path.name}")

        matrices[model_name] = pivot_acc.reindex(columns=ordered_cols)

    return matrices


def analyze_by_combo_size(df):
    """Analyze performance by combination size."""
    analysis = []

    for model_name in df["model"].unique():
        df_model = df[df["model"] == model_name]

        if "combo_size" not in df_model.columns:
            continue

        for size in sorted(df_model["combo_size"].unique()):
            df_size = df_model[df_model["combo_size"] == size]

            n_combos = df_size["train_combo"].nunique()

            for test_rep in REPRESENTATIONS:
                df_rep = df_size[df_size["test_representation"] == test_rep]
                if df_rep.empty:
                    continue

                analysis.append({
                    "model": model_name,
                    "combo_size": int(size),
                    "n_combos": n_combos,
                    "test_representation": test_rep,
                    "mean_accuracy": round(df_rep["accuracy"].mean() * 100, 4),
                    "std_accuracy": round(df_rep["accuracy"].std() * 100, 4),
                    "min_accuracy": round(df_rep["accuracy"].min() * 100, 4),
                    "max_accuracy": round(df_rep["accuracy"].max() * 100, 4),
                    "mean_f1": round(df_rep["f1"].mean(), 4),
                    "std_f1": round(df_rep["f1"].std(), 4),
                    "mean_precision": round(df_rep["precision"].mean(), 4),
                    "mean_recall": round(df_rep["recall"].mean(), 4),
                })

    df_analysis = pd.DataFrame(analysis)
    if not df_analysis.empty:
        df_analysis.to_csv(ANALYSIS_DIR / "performance_by_combo_size.csv", index=False)
    return df_analysis


def find_best_worst_combos(df):
    """Find best and worst combinations per model."""
    best_worst = []

    for model_name in df["model"].unique():
        df_model = df[df["model"] == model_name]
        if df_model.empty:
            continue

        # Average accuracy across all test variants for each combo
        combo_avg = df_model.groupby(["train_combo", "train_combo_full"]).agg({
            "accuracy": "mean",
            "f1": "mean",
            "precision": "mean",
            "recall": "mean",
            "combo_size": "first",
            "train_images": "first",
            "train_time_sec": "first",
        }).reset_index()

        combo_avg = combo_avg.sort_values("accuracy", ascending=False)

        # Top 20
        for rank, (_, row) in enumerate(combo_avg.head(20).iterrows(), 1):
            best_worst.append({
                "model": model_name,
                "rank": rank,
                "category": "TOP",
                "train_combo": row["train_combo"],
                "train_combo_full": row["train_combo_full"],
                "combo_size": int(row["combo_size"]),
                "train_images": int(row["train_images"]),
                "avg_accuracy_pct": round(row["accuracy"] * 100, 4),
                "avg_f1": round(row["f1"], 4),
                "avg_precision": round(row["precision"], 4),
                "avg_recall": round(row["recall"], 4),
                "train_time_sec": round(row["train_time_sec"], 2),
            })

        # Bottom 20
        for rank, (_, row) in enumerate(combo_avg.tail(20).iterrows(), 1):
            best_worst.append({
                "model": model_name,
                "rank": rank,
                "category": "BOTTOM",
                "train_combo": row["train_combo"],
                "train_combo_full": row["train_combo_full"],
                "combo_size": int(row["combo_size"]),
                "train_images": int(row["train_images"]),
                "avg_accuracy_pct": round(row["accuracy"] * 100, 4),
                "avg_f1": round(row["f1"], 4),
                "avg_precision": round(row["precision"], 4),
                "avg_recall": round(row["recall"], 4),
                "train_time_sec": round(row["train_time_sec"], 2),
            })

        # Best per combo size
        for size in sorted(df_model["combo_size"].unique()):
            size_combos = combo_avg[combo_avg["combo_size"] == size]
            if size_combos.empty:
                continue
            best_row = size_combos.iloc[0]
            best_worst.append({
                "model": model_name,
                "rank": 1,
                "category": f"BEST_SIZE_{int(size)}",
                "train_combo": best_row["train_combo"],
                "train_combo_full": best_row["train_combo_full"],
                "combo_size": int(best_row["combo_size"]),
                "train_images": int(best_row["train_images"]),
                "avg_accuracy_pct": round(best_row["accuracy"] * 100, 4),
                "avg_f1": round(best_row["f1"], 4),
                "avg_precision": round(best_row["precision"], 4),
                "avg_recall": round(best_row["recall"], 4),
                "train_time_sec": round(best_row["train_time_sec"], 2),
            })

    df_bw = pd.DataFrame(best_worst)
    if not df_bw.empty:
        df_bw.to_csv(ANALYSIS_DIR / "best_worst_combinations.csv", index=False)
    return df_bw


def analyze_variant_contribution(df):
    """
    Analyze how much each variant contributes when added to training.
    For each variant V, compare combos containing V vs not containing V.
    """
    contribution = []

    for model_name in df["model"].unique():
        df_model = df[df["model"] == model_name]
        if "train_combo_full" not in df_model.columns:
            continue

        for variant in REPRESENTATIONS:
            # Combos that include this variant
            df_with = df_model[df_model["train_combo_full"].str.contains(variant, na=False)]
            df_without = df_model[~df_model["train_combo_full"].str.contains(variant, na=False)]

            if df_with.empty or df_without.empty:
                continue

            for test_rep in REPRESENTATIONS:
                acc_with = df_with[df_with["test_representation"] == test_rep]["accuracy"]
                acc_without = df_without[df_without["test_representation"] == test_rep]["accuracy"]

                if acc_with.empty or acc_without.empty:
                    continue

                contribution.append({
                    "model": model_name,
                    "training_variant": variant,
                    "test_representation": test_rep,
                    "mean_acc_with_variant": round(acc_with.mean() * 100, 4),
                    "mean_acc_without_variant": round(acc_without.mean() * 100, 4),
                    "accuracy_delta_pct": round(
                        (acc_with.mean() - acc_without.mean()) * 100, 4
                    ),
                    "n_combos_with": acc_with.count(),
                    "n_combos_without": acc_without.count(),
                })

    df_contrib = pd.DataFrame(contribution)
    if not df_contrib.empty:
        df_contrib.to_csv(ANALYSIS_DIR / "variant_contribution.csv", index=False)
    return df_contrib


def analyze_single_vs_multi(df):
    """Compare single-variant training vs multi-variant training."""
    comparison = []

    for model_name in df["model"].unique():
        df_model = df[df["model"] == model_name]
        if "combo_size" not in df_model.columns:
            continue

        # Single variant results (size=1)
        df_single = df_model[df_model["combo_size"] == 1]

        for test_rep in REPRESENTATIONS:
            # Single variant trained on same variant as test
            df_same = df_single[
                (df_single["test_representation"] == test_rep) &
                (df_single["train_combo_full"] == test_rep)
            ]

            # All-combined (size=9)
            df_all = df_model[
                (df_model["combo_size"] == 9) &
                (df_model["test_representation"] == test_rep)
            ]

            # Best combo for this test_rep (any size)
            df_test = df_model[df_model["test_representation"] == test_rep]
            if df_test.empty:
                continue

            best_idx = df_test["accuracy"].idxmax()
            best_row = df_test.loc[best_idx]

            comparison.append({
                "model": model_name,
                "test_representation": test_rep,
                "single_same_acc": round(df_same["accuracy"].values[0] * 100, 4) if not df_same.empty else None,
                "all_combined_acc": round(df_all["accuracy"].values[0] * 100, 4) if not df_all.empty else None,
                "best_combo_acc": round(best_row["accuracy"] * 100, 4),
                "best_combo": best_row["train_combo"],
                "best_combo_size": int(best_row["combo_size"]) if "combo_size" in best_row.index else None,
            })

    df_comp = pd.DataFrame(comparison)
    if not df_comp.empty:
        df_comp.to_csv(ANALYSIS_DIR / "single_vs_multi_comparison.csv", index=False)
    return df_comp


def generate_report(df_results, df_classwise, df_inference,
                    df_size, df_bw, df_contrib, df_single_multi):
    """Generate comprehensive markdown report."""

    report = []
    report.append("# 511-Combination Training Experiment — Detailed Analysis Report\n")
    report.append("## Representation-Invariant Object Recognition\n")
    report.append(f"**Generated:** Results from exhaustive training experiment\n")
    report.append(f"**Representations (9):** {', '.join(REPRESENTATIONS)}\n")
    report.append(f"**Total Training Combinations:** 2^9 - 1 = 511\n")
    report.append(f"**Result Matrix per Model:** 511 × 9\n\n")

    # ── Overview ──
    report.append("---\n")
    report.append("## 1. Experiment Overview\n\n")

    if not df_results.empty:
        models = df_results["model"].unique()
        report.append(f"**Models evaluated:** {', '.join(models)}\n\n")

        for model_name in models:
            df_m = df_results[df_results["model"] == model_name]
            n_combos = df_m["train_combo"].nunique()
            n_evals = len(df_m)
            report.append(f"- **{model_name}**: {n_combos} training combinations, "
                          f"{n_evals} total evaluations\n")

    # ── Inference Benchmarks ──
    report.append("\n---\n")
    report.append("## 2. Model-Wise Inference Benchmarks\n\n")

    if not df_inference.empty:
        report.append("| Model | Architecture | Device | Avg Latency (ms) | "
                      "Throughput (FPS) | Total Images |\n")
        report.append("|-------|-------------|--------|:-----------------:|"
                      ":----------------:|:------------:|\n")

        for _, row in df_inference.iterrows():
            report.append(f"| {row.get('model', 'N/A')} "
                          f"| {row.get('architecture_type', 'N/A')} "
                          f"| {row.get('device', 'N/A')} "
                          f"| {row.get('avg_latency_ms', 0):.2f} "
                          f"| {row.get('throughput_fps', 0):.1f} "
                          f"| {row.get('total_images', 0)} |\n")
    else:
        report.append("*No inference benchmark data available.*\n")

    # ── Performance by Combo Size ──
    report.append("\n---\n")
    report.append("## 3. Performance by Training Combination Size\n\n")
    report.append("This shows how accuracy changes as more variants are included in training.\n\n")

    if not df_size.empty:
        for model_name in df_size["model"].unique():
            df_m = df_size[df_size["model"] == model_name]
            report.append(f"### {model_name}\n\n")

            # Summary table: one row per combo size, average across all test reps
            size_summary = df_m.groupby("combo_size").agg({
                "mean_accuracy": "mean",
                "std_accuracy": "mean",
                "n_combos": "first",
                "mean_f1": "mean",
                "mean_precision": "mean",
                "mean_recall": "mean",
            }).reset_index()

            report.append("| Combo Size | # Combos | Mean Acc (%) | Std Acc (%) | "
                          "Mean F1 | Mean Precision | Mean Recall |\n")
            report.append("|:----------:|:--------:|:------------:|:-----------:|"
                          ":-------:|:--------------:|:-----------:|\n")

            for _, row in size_summary.iterrows():
                report.append(f"| {int(row['combo_size'])} "
                              f"| {int(row['n_combos'])} "
                              f"| {row['mean_accuracy']:.2f} "
                              f"| {row['std_accuracy']:.2f} "
                              f"| {row['mean_f1']:.4f} "
                              f"| {row['mean_precision']:.4f} "
                              f"| {row['mean_recall']:.4f} |\n")
            report.append("\n")

            # Detailed breakdown: per test representation
            report.append(f"**Detailed breakdown per test representation:**\n\n")
            report.append("| Combo Size | Test Rep | Mean Acc (%) | Std (%) | Min (%) | Max (%) |\n")
            report.append("|:----------:|----------|:------------:|:-------:|:-------:|:-------:|\n")

            for _, row in df_m.iterrows():
                report.append(f"| {int(row['combo_size'])} "
                              f"| {row['test_representation']} "
                              f"| {row['mean_accuracy']:.2f} "
                              f"| {row['std_accuracy']:.2f} "
                              f"| {row['min_accuracy']:.2f} "
                              f"| {row['max_accuracy']:.2f} |\n")
            report.append("\n")

    # ── Best & Worst Combinations ──
    report.append("\n---\n")
    report.append("## 4. Best & Worst Training Combinations\n\n")

    if not df_bw.empty:
        for model_name in df_bw["model"].unique():
            df_m = df_bw[df_bw["model"] == model_name]

            report.append(f"### {model_name}\n\n")

            # Top 20
            df_top = df_m[df_m["category"] == "TOP"].sort_values("rank")
            if not df_top.empty:
                report.append("#### Top 20 Combinations (by average accuracy across all test variants)\n\n")
                report.append("| Rank | Combination | Size | Avg Acc (%) | "
                              "Avg F1 | Avg Prec | Avg Recall | Train Time |\n")
                report.append("|:----:|-------------|:----:|:-----------:|"
                              ":------:|:--------:|:----------:|:----------:|\n")

                for _, row in df_top.iterrows():
                    report.append(f"| {int(row['rank'])} "
                                  f"| {row['train_combo']} "
                                  f"| {int(row['combo_size'])} "
                                  f"| {row['avg_accuracy_pct']:.2f} "
                                  f"| {row['avg_f1']:.4f} "
                                  f"| {row['avg_precision']:.4f} "
                                  f"| {row['avg_recall']:.4f} "
                                  f"| {row['train_time_sec']:.1f}s |\n")
                report.append("\n")

            # Bottom 20
            df_bot = df_m[df_m["category"] == "BOTTOM"].sort_values("rank")
            if not df_bot.empty:
                report.append("#### Bottom 20 Combinations\n\n")
                report.append("| Rank | Combination | Size | Avg Acc (%) | "
                              "Avg F1 | Train Time |\n")
                report.append("|:----:|-------------|:----:|:-----------:|"
                              ":------:|:----------:|\n")

                for _, row in df_bot.iterrows():
                    report.append(f"| {int(row['rank'])} "
                                  f"| {row['train_combo']} "
                                  f"| {int(row['combo_size'])} "
                                  f"| {row['avg_accuracy_pct']:.2f} "
                                  f"| {row['avg_f1']:.4f} "
                                  f"| {row['train_time_sec']:.1f}s |\n")
                report.append("\n")

            # Best per size
            size_cats = [c for c in df_m["category"].unique() if c.startswith("BEST_SIZE_")]
            if size_cats:
                report.append("#### Best Combination per Size\n\n")
                report.append("| Size | Best Combination | Avg Acc (%) | "
                              "Avg F1 | Train Images |\n")
                report.append("|:----:|-----------------|:-----------:|"
                              ":------:|:------------:|\n")

                for cat in sorted(size_cats, key=lambda x: int(x.split("_")[-1])):
                    row = df_m[df_m["category"] == cat].iloc[0]
                    report.append(f"| {int(row['combo_size'])} "
                                  f"| {row['train_combo']} "
                                  f"| {row['avg_accuracy_pct']:.2f} "
                                  f"| {row['avg_f1']:.4f} "
                                  f"| {int(row['train_images'])} |\n")
                report.append("\n")

    # ── Variant Contribution Analysis ──
    report.append("\n---\n")
    report.append("## 5. Variant Contribution Analysis\n\n")
    report.append("This shows how much each training variant contributes to "
                  "performance when included vs excluded.\n\n")

    if not df_contrib.empty:
        for model_name in df_contrib["model"].unique():
            df_m = df_contrib[df_contrib["model"] == model_name]

            report.append(f"### {model_name}\n\n")

            # Average delta per training variant
            variant_avg = df_m.groupby("training_variant").agg({
                "accuracy_delta_pct": "mean",
            }).sort_values("accuracy_delta_pct", ascending=False).reset_index()

            report.append("#### Average Accuracy Delta When Variant is Included in Training\n\n")
            report.append("| Training Variant | Avg Δ Accuracy (%) | Impact |\n")
            report.append("|-----------------|:------------------:|--------|\n")

            for _, row in variant_avg.iterrows():
                delta = row["accuracy_delta_pct"]
                impact = "🟢 Positive" if delta > 0.5 else ("🔴 Negative" if delta < -0.5 else "⚪ Neutral")
                report.append(f"| {row['training_variant']} "
                              f"| {delta:+.2f} "
                              f"| {impact} |\n")
            report.append("\n")

            # Cross-contribution matrix
            report.append("#### Contribution Matrix (Training Variant → Test Variant)\n\n")

            pivot = df_m.pivot_table(
                index="training_variant",
                columns="test_representation",
                values="accuracy_delta_pct",
                aggfunc="first"
            )
            ordered_cols = [c for c in REPRESENTATIONS if c in pivot.columns]
            pivot = pivot.reindex(columns=ordered_cols)

            # Table header
            header = "| Training \\ Test | " + " | ".join(
                SHORT_LABELS.get(c, c) for c in ordered_cols
            ) + " |\n"
            report.append(header)
            report.append("|" + "|".join(["-"*15] + [":---:"]*len(ordered_cols)) + "|\n")

            for variant in REPRESENTATIONS:
                if variant not in pivot.index:
                    continue
                row_vals = []
                for col in ordered_cols:
                    val = pivot.loc[variant, col] if col in pivot.columns else 0
                    row_vals.append(f"{val:+.2f}" if pd.notna(val) else "—")
                report.append(f"| {SHORT_LABELS.get(variant, variant)} | "
                              + " | ".join(row_vals) + " |\n")
            report.append("\n")

    # ── Single vs Multi Comparison ──
    report.append("\n---\n")
    report.append("## 6. Single vs Multi-Variant Training Comparison\n\n")
    report.append("Compares: (a) training on same variant as test, "
                  "(b) training on all 9 variants, (c) best combo found.\n\n")

    if not df_single_multi.empty:
        for model_name in df_single_multi["model"].unique():
            df_m = df_single_multi[df_single_multi["model"] == model_name]

            report.append(f"### {model_name}\n\n")
            report.append("| Test Variant | Same-as-Test (%) | All Combined (%) | "
                          "Best Combo (%) | Best Combo | Best Size |\n")
            report.append("|-------------|:----------------:|:----------------:|"
                          ":--------------:|------------|:---------:|\n")

            for _, row in df_m.iterrows():
                same = f"{row['single_same_acc']:.2f}" if pd.notna(row.get('single_same_acc')) else "—"
                allc = f"{row['all_combined_acc']:.2f}" if pd.notna(row.get('all_combined_acc')) else "—"
                best = f"{row['best_combo_acc']:.2f}"
                size = str(int(row['best_combo_size'])) if pd.notna(row.get('best_combo_size')) else "—"

                report.append(f"| {row['test_representation']} "
                              f"| {same} "
                              f"| {allc} "
                              f"| {best} "
                              f"| {row['best_combo']} "
                              f"| {size} |\n")
            report.append("\n")

    # ── Key Findings ──
    report.append("\n---\n")
    report.append("## 7. Key Findings & Insights\n\n")

    if not df_results.empty:
        for model_name in df_results["model"].unique():
            df_m = df_results[df_results["model"] == model_name]
            if "combo_size" not in df_m.columns:
                continue

            report.append(f"### {model_name}\n\n")

            # Overall best
            combo_avg = df_m.groupby("train_combo")["accuracy"].mean()
            best_combo = combo_avg.idxmax()
            best_acc = combo_avg.max() * 100

            report.append(f"1. **Best overall combination**: `{best_combo}` "
                          f"(avg accuracy: {best_acc:.2f}%)\n")

            # Size that gives best average
            size_avg = df_m.groupby("combo_size")["accuracy"].mean() * 100
            best_size = size_avg.idxmax()
            report.append(f"2. **Optimal combo size**: {int(best_size)} variants "
                          f"(avg accuracy: {size_avg[best_size]:.2f}%)\n")

            # Diminishing returns analysis
            if len(size_avg) > 1:
                improvements = size_avg.diff()
                if improvements.iloc[-1] < 0.5 and len(improvements) > 3:
                    plateau_size = None
                    for s in sorted(size_avg.index):
                        if s > 1 and improvements.get(s, 0) < 0.5:
                            plateau_size = s
                            break
                    if plateau_size:
                        report.append(f"3. **Diminishing returns**: Performance plateaus "
                                      f"around {int(plateau_size)} variants\n")

            # Self-transfer: how well does training on X transfer to testing on X
            df_single = df_m[df_m["combo_size"] == 1]
            self_transfer_scores = []
            for rep in REPRESENTATIONS:
                df_self = df_single[
                    (df_single["train_combo_full"] == rep) &
                    (df_single["test_representation"] == rep)
                ]
                if not df_self.empty:
                    self_transfer_scores.append((rep, df_self["accuracy"].values[0] * 100))

            if self_transfer_scores:
                best_self = max(self_transfer_scores, key=lambda x: x[1])
                worst_self = min(self_transfer_scores, key=lambda x: x[1])
                report.append(f"4. **Best self-transfer**: {best_self[0]} ({best_self[1]:.1f}%)\n")
                report.append(f"5. **Worst self-transfer**: {worst_self[0]} ({worst_self[1]:.1f}%)\n")

            report.append("\n")

    # ── Files Generated ──
    report.append("\n---\n")
    report.append("## 8. Output Files\n\n")
    report.append("| File | Description |\n")
    report.append("|------|------------|\n")
    report.append(f"| `combo_results.csv` | Full 511×9 results for all models |\n")
    report.append(f"| `combo_classwise.csv` | Per-class metrics for all combinations |\n")
    report.append(f"| `combo_inference.csv` | Model-wise inference benchmarks |\n")
    report.append(f"| `analysis/performance_by_combo_size.csv` | Aggregated stats by combination size |\n")
    report.append(f"| `analysis/best_worst_combinations.csv` | Top/Bottom 20 combos per model |\n")
    report.append(f"| `analysis/variant_contribution.csv` | Impact of each variant on accuracy |\n")
    report.append(f"| `analysis/single_vs_multi_comparison.csv` | Single vs multi-variant comparison |\n")
    report.append(f"| `analysis/*_accuracy_pct_matrix.csv` | 511×9 accuracy matrices per model |\n")
    report.append(f"| `analysis/*_f1_matrix.csv` | 511×9 F1 matrices per model |\n")
    report.append(f"| `analysis/*_precision_matrix.csv` | 511×9 precision matrices per model |\n")
    report.append(f"| `analysis/*_recall_matrix.csv` | 511×9 recall matrices per model |\n")

    report_text = "".join(report)

    report_path = ANALYSIS_DIR / "experiment_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"[OK] Markdown report saved to: {report_path}")

    # Plain text human-readable version
    txt_path = RESULTS_DIR / "results_explanation.txt"
    # Convert markdown headers/tables to clean plain text
    clean_txt = report_text.replace("# ", "").replace("## ", "\n--- ").replace("### ", "\n").replace("**", "")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(clean_txt)
    print(f"[OK] Plain-text explanation saved to: {txt_path}")

    return report_text


def main():
    print("=" * 70)
    print("  ANALYZING 511-COMBINATION EXPERIMENT RESULTS")
    print("=" * 70)

    df_results, df_classwise, df_inference = load_results()

    if df_results.empty:
        print("[ERROR] No results found. Run 10_combo_train_eval.py first.")
        return

    print(f"\n  Total result rows: {len(df_results)}")
    print(f"  Models: {df_results['model'].unique().tolist()}")
    print(f"  Unique combinations: {df_results['train_combo'].nunique()}")

    print("\n  Generating per-model matrices...")
    matrices = generate_model_matrices(df_results)

    print("  Analyzing by combination size...")
    df_size = analyze_by_combo_size(df_results)

    print("  Finding best/worst combinations...")
    df_bw = find_best_worst_combos(df_results)

    print("  Analyzing variant contributions...")
    df_contrib = analyze_variant_contribution(df_results)

    print("  Comparing single vs multi-variant training...")
    df_single_multi = analyze_single_vs_multi(df_results)

    print("  Generating comprehensive report...")
    generate_report(
        df_results, df_classwise, df_inference,
        df_size, df_bw, df_contrib, df_single_multi
    )

    print(f"\n{'='*70}")
    print(f"  ANALYSIS COMPLETE")
    print(f"  All results saved to: {ANALYSIS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
