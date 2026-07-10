"""
Create professor-update graphics for the boundary-weighted reconstruction and
full-hybrid decision experiments.

Place this file at:
    src/plot_boundary_weighted_professor_update.py

Purpose
-------
This is a reporting/diagnostic plotting script only. It reads the comparison
CSVs already created by:
    src/compare_original_vs_boundary_weighted_results.py

It does not retrain models and does not modify the core pipeline.

Inputs
------
output/intermediate_csv/boundary_weighted_comparison/
    queue_reconstruction_test_comparison.csv
    decision_test_comparison.csv
    queue_reconstruction_test_best_summary.csv
    decision_test_best_summary.csv

Outputs
-------
output/intermediate_csv/boundary_weighted_professor_update/
    selected_queue_reconstruction_comparison.csv
    selected_decision_comparison.csv
    professor_update_summary.txt

output/final_plots/professor_update_boundary_weighted/
    fig_01_queue_rmse_original_vs_boundary_weighted.png
    fig_02_queue_mae_original_vs_boundary_weighted.png
    fig_03_queue_rmse_improvement_pct.png
    fig_04_decision_f1_original_vs_boundary_weighted.png
    fig_05_decision_rmse_original_vs_boundary_weighted.png
    fig_06_decision_precision_recall_boundary_weighted.png
    fig_07_decision_f1_improvement_pct.png

Notes
-----
These figures are meant for a quick professor update. They are intentionally
simple and may not be the final publication figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import PROJECT_ROOT


# =============================================================================
# Paths
# =============================================================================

COMPARISON_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "boundary_weighted_comparison"
OUT_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "boundary_weighted_professor_update"
FIG_DIR = PROJECT_ROOT / "output" / "final_plots" / "professor_update_boundary_weighted"

QUEUE_TEST_COMPARISON_FILE = COMPARISON_DIR / "queue_reconstruction_test_comparison.csv"
DECISION_TEST_COMPARISON_FILE = COMPARISON_DIR / "decision_test_comparison.csv"
QUEUE_BEST_SUMMARY_FILE = COMPARISON_DIR / "queue_reconstruction_test_best_summary.csv"
DECISION_BEST_SUMMARY_FILE = COMPARISON_DIR / "decision_test_best_summary.csv"


# =============================================================================
# User settings
# =============================================================================

FIGURE_DPI = 300
SHOW_FIGURES = False

# Keep these to the main story only. Add/remove IDs later if needed.
SELECTED_QUEUE_CURVES = [
    "physics_baseline",
    "physics_ml_gru",
    "physics_ml_cv_gru",
    "physics_ml_cv_xgb",
]

QUEUE_DISPLAY_LABELS = {
    "physics_baseline": "Physics\nbaseline",
    "physics_ml_gru": "Physics + ML\nGRU",
    "physics_ml_cv_gru": "Physics + ML + CV\nGRU",
    "physics_ml_cv_xgb": "Physics + ML + CV\nXGBoost",
}

SELECTED_DECISION_MODELS = [
    "queue_derived_threshold_physics_ml_cv_xgb",
    "direct_ml_decision_xgb_regression_threshold",
    "full_hybrid_decision_xgb_from_physics_ml_cv_gru_regression_threshold",
    "full_hybrid_decision_xgb_from_physics_ml_cv_xgb_regression_threshold",
    "full_hybrid_decision_xgb_from_physics_ml_cv_xgb_dual_head",
]

DECISION_DISPLAY_LABELS = {
    "queue_derived_threshold_physics_ml_cv_xgb": "Queue-derived\nPhysics+ML+CV XGB",
    "direct_ml_decision_xgb_regression_threshold": "Direct ML-only\nXGB reg.",
    "full_hybrid_decision_xgb_from_physics_ml_cv_gru_regression_threshold": "Full hybrid\nGRU profile + XGB reg.",
    "full_hybrid_decision_xgb_from_physics_ml_cv_xgb_regression_threshold": "Full hybrid\nXGB profile + XGB reg.",
    "full_hybrid_decision_xgb_from_physics_ml_cv_xgb_dual_head": "Full hybrid\nXGB profile + XGB prob.",
}


# =============================================================================
# Helpers
# =============================================================================

def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file:\n{path}\n\n"
            "Run src/compare_original_vs_boundary_weighted_results.py first."
        )


def read_csv(path: Path) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path)


def safe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_zero_line(ax) -> None:
    ax.axhline(0, linewidth=0.8)


def style_axis(ax) -> None:
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="both", labelsize=9)


def save_figure(fig, filename: str) -> None:
    out_path = FIG_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)
    print(f"[Saved figure] {out_path}")


def save_table(df: pd.DataFrame, filename: str) -> None:
    out_path = OUT_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"[Saved table] {out_path}")


def paired_bar_plot(
    df: pd.DataFrame,
    label_col: str,
    original_col: str,
    boundary_col: str,
    title: str,
    ylabel: str,
    filename: str,
    value_format: str = "{:.1f}",
) -> None:
    plot_df = df.copy()
    plot_df[original_col] = pd.to_numeric(plot_df[original_col], errors="coerce")
    plot_df[boundary_col] = pd.to_numeric(plot_df[boundary_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[original_col, boundary_col])

    x = np.arange(len(plot_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    b1 = ax.bar(x - width / 2, plot_df[original_col].to_numpy(dtype=float), width, label="Original")
    b2 = ax.bar(x + width / 2, plot_df[boundary_col].to_numpy(dtype=float), width, label="Boundary-weighted")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df[label_col].astype(str), rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True)
    style_axis(ax)

    for bars in [b1, b2]:
        for bar in bars:
            y = bar.get_height()
            if np.isfinite(y):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    value_format.format(y),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    save_figure(fig, filename)


def single_bar_plot(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    filename: str,
    value_format: str = "{:.1f}%",
) -> None:
    plot_df = df.copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_col])

    x = np.arange(len(plot_df))
    y = plot_df[value_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    bars = ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df[label_col].astype(str), rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    add_zero_line(ax)
    style_axis(ax)

    for bar, yi in zip(bars, y):
        if np.isfinite(yi):
            va = "bottom" if yi >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yi,
                value_format.format(yi),
                ha="center",
                va=va,
                fontsize=8,
            )

    save_figure(fig, filename)


# =============================================================================
# Data preparation
# =============================================================================

def prepare_queue_comparison() -> pd.DataFrame:
    queue = read_csv(QUEUE_TEST_COMPARISON_FILE)

    needed = [
        "curve_id",
        "rmse_ft_original",
        "rmse_ft_boundary_weighted",
        "rmse_ft_improvement_pct",
        "mae_ft_original",
        "mae_ft_boundary_weighted",
        "mae_ft_improvement_pct",
    ]
    missing = [c for c in needed if c not in queue.columns]
    if missing:
        raise ValueError(f"Queue comparison file missing columns: {missing}")

    queue = queue[queue["curve_id"].isin(SELECTED_QUEUE_CURVES)].copy()
    order = {curve_id: i for i, curve_id in enumerate(SELECTED_QUEUE_CURVES)}
    queue["plot_order"] = queue["curve_id"].map(order)
    queue = queue.sort_values("plot_order").reset_index(drop=True)
    queue["method_label"] = queue["curve_id"].map(QUEUE_DISPLAY_LABELS).fillna(queue["curve_id"])

    numeric_cols = [c for c in queue.columns if c.endswith("_original") or c.endswith("_boundary_weighted") or c.endswith("_improvement_pct")]
    queue = safe_numeric(queue, numeric_cols)
    return queue


def prepare_decision_comparison() -> pd.DataFrame:
    decision = read_csv(DECISION_TEST_COMPARISON_FILE)

    needed = [
        "model_id",
        "residual_rmse_ft_original",
        "residual_rmse_ft_boundary_weighted",
        "residual_rmse_ft_improvement_pct",
        "failure_f1_original",
        "failure_f1_boundary_weighted",
        "failure_f1_improvement_pct",
        "failure_precision_boundary_weighted",
        "failure_recall_boundary_weighted",
    ]
    missing = [c for c in needed if c not in decision.columns]
    if missing:
        raise ValueError(f"Decision comparison file missing columns: {missing}")

    decision = decision[decision["model_id"].isin(SELECTED_DECISION_MODELS)].copy()
    order = {model_id: i for i, model_id in enumerate(SELECTED_DECISION_MODELS)}
    decision["plot_order"] = decision["model_id"].map(order)
    decision = decision.sort_values("plot_order").reset_index(drop=True)
    decision["method_label"] = decision["model_id"].map(DECISION_DISPLAY_LABELS).fillna(decision["model_id"])

    numeric_cols = [
        "residual_rmse_ft_original",
        "residual_rmse_ft_boundary_weighted",
        "residual_rmse_ft_improvement_pct",
        "failure_f1_original",
        "failure_f1_boundary_weighted",
        "failure_f1_improvement_pct",
        "failure_precision_boundary_weighted",
        "failure_recall_boundary_weighted",
    ]
    decision = safe_numeric(decision, numeric_cols)

    # Convert classification scores to percentage for plotting.
    for col in [
        "failure_f1_original",
        "failure_f1_boundary_weighted",
        "failure_precision_boundary_weighted",
        "failure_recall_boundary_weighted",
    ]:
        decision[f"{col}_pct"] = 100.0 * decision[col]

    return decision


# =============================================================================
# Plot creation
# =============================================================================

def make_queue_plots(queue: pd.DataFrame) -> None:
    paired_bar_plot(
        queue,
        label_col="method_label",
        original_col="rmse_ft_original",
        boundary_col="rmse_ft_boundary_weighted",
        title="Queue reconstruction RMSE: original vs boundary-weighted",
        ylabel="RMSE (ft)",
        filename="fig_01_queue_rmse_original_vs_boundary_weighted.png",
    )

    paired_bar_plot(
        queue,
        label_col="method_label",
        original_col="mae_ft_original",
        boundary_col="mae_ft_boundary_weighted",
        title="Queue reconstruction MAE: original vs boundary-weighted",
        ylabel="MAE (ft)",
        filename="fig_02_queue_mae_original_vs_boundary_weighted.png",
    )

    single_bar_plot(
        queue,
        label_col="method_label",
        value_col="rmse_ft_improvement_pct",
        title="Queue reconstruction RMSE improvement from boundary weighting",
        ylabel="Improvement in RMSE (%)",
        filename="fig_03_queue_rmse_improvement_pct.png",
    )


def make_decision_plots(decision: pd.DataFrame) -> None:
    paired_bar_plot(
        decision,
        label_col="method_label",
        original_col="failure_f1_original_pct",
        boundary_col="failure_f1_boundary_weighted_pct",
        title="Cycle-failure F1: original vs boundary-weighted",
        ylabel="F1-score (%)",
        filename="fig_04_decision_f1_original_vs_boundary_weighted.png",
        value_format="{:.1f}",
    )

    paired_bar_plot(
        decision,
        label_col="method_label",
        original_col="residual_rmse_ft_original",
        boundary_col="residual_rmse_ft_boundary_weighted",
        title="Residual queue RMSE: original vs boundary-weighted",
        ylabel="Residual queue RMSE (ft)",
        filename="fig_05_decision_rmse_original_vs_boundary_weighted.png",
    )

    # Precision/recall for the boundary-weighted run only.
    pr = decision.copy()
    x = np.arange(len(pr))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    b1 = ax.bar(
        x - width / 2,
        pr["failure_precision_boundary_weighted_pct"].to_numpy(dtype=float),
        width,
        label="Precision",
    )
    b2 = ax.bar(
        x + width / 2,
        pr["failure_recall_boundary_weighted_pct"].to_numpy(dtype=float),
        width,
        label="Recall",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(pr["method_label"].astype(str), rotation=20, ha="right")
    ax.set_title("Boundary-weighted decision models: precision and recall")
    ax.set_ylabel("Score (%)")
    ax.legend(frameon=True)
    ax.set_ylim(0, 110)
    style_axis(ax)

    for bars in [b1, b2]:
        for bar in bars:
            y = bar.get_height()
            if np.isfinite(y):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    f"{y:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    save_figure(fig, "fig_06_decision_precision_recall_boundary_weighted.png")

    single_bar_plot(
        decision,
        label_col="method_label",
        value_col="failure_f1_improvement_pct",
        title="Cycle-failure F1 improvement from boundary weighting",
        ylabel="Improvement in F1 (%)",
        filename="fig_07_decision_f1_improvement_pct.png",
    )


# =============================================================================
# Text summary
# =============================================================================

def write_summary(queue: pd.DataFrame, decision: pd.DataFrame) -> None:
    summary_path = OUT_DIR / "professor_update_summary.txt"

    q_gru = queue[queue["curve_id"].eq("physics_ml_cv_gru")]
    d_best = decision[decision["model_id"].eq("full_hybrid_decision_xgb_from_physics_ml_cv_xgb_regression_threshold")]

    lines = []
    lines.append("Boundary-weighted reconstruction and full hybrid decision update")
    lines.append("=" * 72)
    lines.append("")

    if not q_gru.empty:
        r = q_gru.iloc[0]
        lines.append("Queue reconstruction result:")
        lines.append(
            f"- Physics + ML + CV GRU RMSE changed from "
            f"{r['rmse_ft_original']:.3f} ft to {r['rmse_ft_boundary_weighted']:.3f} ft "
            f"({r['rmse_ft_improvement_pct']:.2f}% improvement)."
        )
        lines.append(
            f"- MAE changed from {r['mae_ft_original']:.3f} ft to "
            f"{r['mae_ft_boundary_weighted']:.3f} ft "
            f"({r['mae_ft_improvement_pct']:.2f}% improvement)."
        )
        lines.append("")

    if not d_best.empty:
        r = d_best.iloc[0]
        lines.append("Best decision-support result:")
        lines.append("- Model: Full hybrid Physics + ML + CV XGBoost residual-threshold decision model.")
        lines.append(
            f"- Residual queue RMSE changed from {r['residual_rmse_ft_original']:.3f} ft "
            f"to {r['residual_rmse_ft_boundary_weighted']:.3f} ft "
            f"({r['residual_rmse_ft_improvement_pct']:.2f}% improvement)."
        )
        lines.append(
            f"- Cycle-failure F1 changed from {r['failure_f1_original']:.3f} "
            f"to {r['failure_f1_boundary_weighted']:.3f} "
            f"({r['failure_f1_improvement_pct']:.2f}% improvement)."
        )
        lines.append(
            f"- Boundary-weighted precision = {r['failure_precision_boundary_weighted']:.3f}, "
            f"recall = {r['failure_recall_boundary_weighted']:.3f}."
        )
        lines.append("")

    lines.append("Interpretation:")
    lines.append(
        "- Boundary weighting improves the main GRU queue-profile reconstruction and "
        "the strongest full hybrid decision result."
    )
    lines.append(
        "- GRU remains the preferred model for full queue-profile reconstruction, "
        "while the cycle-level XGBoost decision head gives the strongest residual-queue "
        "and cycle-failure prediction."
    )

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Saved summary] {summary_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_dirs()

    print("=" * 88)
    print("Creating professor-update graphics for boundary-weighted experiment")
    print("=" * 88)
    print(f"Project root  : {PROJECT_ROOT}")
    print(f"Comparison dir: {COMPARISON_DIR}")
    print(f"Output dir    : {OUT_DIR}")
    print(f"Figure dir    : {FIG_DIR}")
    print("=" * 88)

    queue = prepare_queue_comparison()
    decision = prepare_decision_comparison()

    save_table(queue, "selected_queue_reconstruction_comparison.csv")
    save_table(decision, "selected_decision_comparison.csv")

    make_queue_plots(queue)
    make_decision_plots(decision)
    write_summary(queue, decision)

    print("\nDone.")


if __name__ == "__main__":
    main()
