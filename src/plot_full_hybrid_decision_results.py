"""
Plot and summarize full hybrid decision-model results.

Place this file at:
    src/plot_full_hybrid_decision_results.py

Purpose
-------
This is the active decision-oriented plotting/table script after consolidating
older decision analyses.

It reads only the consolidated full-hybrid decision-model outputs:
    output/intermediate_csv/full_hybrid_decision_models

and creates final comparison tables/figures for:
    1) queue-derived decision baseline
    2) direct ML-only decision baseline
    3) full hybrid decision model

This script replaces the older plot_decision_oriented_results.py in the active
workflow. The older script can be moved to src/legacy_decision/.
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

FULL_HYBRID_DIR = (
    PROJECT_ROOT
    / "output"
    / "intermediate_csv"
    / "full_hybrid_decision_models"
)

METRICS_TEST_FILE = FULL_HYBRID_DIR / "full_hybrid_decision_metrics_test.csv"
METRICS_BY_RATE_FILE = FULL_HYBRID_DIR / "full_hybrid_decision_metrics_by_rate.csv"
THRESHOLDS_FILE = FULL_HYBRID_DIR / "selected_thresholds_validation.csv"
TRAINING_SUMMARY_FILE = FULL_HYBRID_DIR / "full_hybrid_decision_training_summary.csv"

OUT_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "full_hybrid_decision_summary"
TABLE_DIR = PROJECT_ROOT / "output" / "tables" / "publication_outputs"
FIG_DIR = PROJECT_ROOT / "output" / "final_plots" / "publication_outputs"


# =============================================================================
# Model selection for compact publication plots
# =============================================================================

# These are the main rows to show in the compact comparison figures.
# The metrics CSVs still keep every model; this list only controls the plot/table
# subset used for the main publication-style decision comparison.
SELECTED_OVERALL_MODELS = [
    "queue_derived_threshold_physics_ml_cv_xgb",
    "direct_ml_decision_xgb_regression_threshold",
    "full_hybrid_decision_xgb_from_physics_ml_cv_xgb_regression_threshold",
    "full_hybrid_decision_xgb_from_physics_ml_cv_gru_regression_threshold",
]

SELECTED_BY_RATE_MODELS = [
    "queue_derived_threshold_physics_ml_cv_xgb",
    "direct_ml_decision_xgb_regression_threshold",
    "full_hybrid_decision_xgb_from_physics_ml_cv_xgb_regression_threshold",
    "full_hybrid_decision_xgb_from_physics_ml_cv_gru_regression_threshold",
]

MODEL_LABELS = {
    "queue_derived_threshold_physics_baseline": "Queue-derived\nphysics baseline",
    "queue_derived_threshold_physics_ml_gru": "Queue-derived\nPhysics + ML GRU",
    "queue_derived_threshold_physics_ml_cv_gru": "Queue-derived\nPhysics + ML + CV GRU",
    "queue_derived_threshold_physics_ml_cv_xgb": "Queue-derived\nPhysics + ML + CV XGBoost",
    "direct_ml_decision_xgb_regression_threshold": "Direct ML-only\nXGBoost reg.-threshold",
    "direct_ml_decision_xgb_dual_head": "Direct ML-only\nXGBoost dual-head",
    "full_hybrid_decision_xgb_from_physics_ml_cv_xgb_regression_threshold": "Full hybrid\nfrom XGBoost recon.",
    "full_hybrid_decision_xgb_from_physics_ml_cv_xgb_dual_head": "Full hybrid\nXGBoost prob. head",
    "full_hybrid_decision_xgb_from_physics_ml_cv_gru_regression_threshold": "Full hybrid\nfrom GRU recon.",
    "full_hybrid_decision_xgb_from_physics_ml_cv_gru_dual_head": "Full hybrid\nGRU prob. head",
}

MODEL_ORDER = {model_id: i + 1 for i, model_id in enumerate(SELECTED_OVERALL_MODELS)}

METRIC_COLS = [
    "residual_mae_ft",
    "residual_rmse_ft",
    "residual_bias_ft",
    "failure_accuracy",
    "failure_precision",
    "failure_recall",
    "failure_f1",
    "tp",
    "fp",
    "fn",
    "tn",
]


# =============================================================================
# Helpers
# =============================================================================

def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file:\n{path}")


def read_csv(path: Path) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path)


def clean_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in METRIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "cv_rate_pct" in out.columns:
        out["cv_rate_pct"] = pd.to_numeric(out["cv_rate_pct"], errors="coerce")
    return out


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["method_label"] = out["model_id"].map(MODEL_LABELS).fillna(out["model_id"].astype(str))
    out["plot_order"] = out["model_id"].map(MODEL_ORDER).fillna(99).astype(int)
    return out


def add_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["failure_accuracy", "failure_precision", "failure_recall", "failure_f1"]:
        if col in out.columns:
            out[f"{col}_pct"] = 100.0 * pd.to_numeric(out[col], errors="coerce")
    return out


def save_table(df: pd.DataFrame, filename: str) -> None:
    out_intermediate = OUT_DIR / filename
    out_publication = TABLE_DIR / filename
    df.to_csv(out_intermediate, index=False)
    df.to_csv(out_publication, index=False)
    print(f"[Saved] {out_intermediate}")
    print(f"[Saved] {out_publication}")


def selected_or_available(df: pd.DataFrame, model_ids: list[str]) -> pd.DataFrame:
    selected = df[df["model_id"].isin(model_ids)].copy()
    if not selected.empty:
        return selected

    # Fallback for environments using the HistGradientBoosting fallback names.
    # Keep the best row from each broad method family if exact XGBoost IDs are absent.
    work = df.copy()
    work["sort_f1"] = work["failure_f1"].fillna(-1.0)
    work["sort_rmse"] = work["residual_rmse_ft"].fillna(1e12)
    work = work.sort_values(["model_family_type", "sort_f1", "sort_rmse"], ascending=[True, False, True])
    return work.groupby("model_family_type", as_index=False, sort=False).head(1).drop(columns=["sort_f1", "sort_rmse"])


def save_bar_plot(
    df: pd.DataFrame,
    y_col: str,
    title: str,
    ylabel: str,
    filename: str,
    value_format: str,
) -> None:
    plot_df = df.copy()
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df[np.isfinite(plot_df[y_col])].copy()
    plot_df = plot_df.sort_values("plot_order").reset_index(drop=True)

    if plot_df.empty:
        print(f"[WARN] Empty plot skipped: {filename}")
        return

    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    x = np.arange(len(plot_df))
    y = plot_df[y_col].to_numpy(dtype=float)

    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["method_label"].astype(str), rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)

    for xi, yi in zip(x, y):
        ax.text(xi, yi, value_format.format(yi), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = FIG_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Saved] {out_path}")


def save_line_plot_by_rate(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    plot_df = df.copy()
    plot_df["cv_rate_pct"] = pd.to_numeric(plot_df["cv_rate_pct"], errors="coerce")
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    plot_df = plot_df.dropna(subset=["cv_rate_pct", metric_col]).copy()
    plot_df = plot_df.sort_values(["plot_order", "cv_rate_pct"])

    if plot_df.empty:
        print(f"[WARN] Empty plot skipped: {filename}")
        return

    fig, ax = plt.subplots(figsize=(10.2, 5.8))

    for label, g in plot_df.groupby("method_label", sort=False):
        g = g.sort_values("cv_rate_pct")
        ax.plot(
            g["cv_rate_pct"].to_numpy(dtype=float),
            g[metric_col].to_numpy(dtype=float),
            marker="o",
            linewidth=1.8,
            label=str(label).replace("\n", " "),
        )

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("CV penetration rate (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True)

    fig.tight_layout()
    out_path = FIG_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Saved] {out_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_dirs()

    print("=" * 96)
    print("Full hybrid decision-model final plots and tables")
    print("=" * 96)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Input dir    : {FULL_HYBRID_DIR}")
    print(f"Output dir   : {OUT_DIR}")
    print(f"Table dir    : {TABLE_DIR}")
    print(f"Figure dir   : {FIG_DIR}")
    print("=" * 96)

    test = add_display_columns(clean_metric_columns(read_csv(METRICS_TEST_FILE)))
    by_rate = add_display_columns(clean_metric_columns(read_csv(METRICS_BY_RATE_FILE)))

    overall_all = add_percent_columns(test.sort_values(["failure_f1", "residual_rmse_ft"], ascending=[False, True]).reset_index(drop=True))
    save_table(overall_all, "full_hybrid_decision_all_test_metrics.csv")

    overall_selected = selected_or_available(test, SELECTED_OVERALL_MODELS)
    overall_selected = add_percent_columns(add_display_columns(overall_selected))
    overall_selected = overall_selected.sort_values("plot_order").reset_index(drop=True)
    save_table(overall_selected, "full_hybrid_decision_overall_test_comparison.csv")

    by_rate_selected = by_rate[by_rate["model_id"].isin(SELECTED_BY_RATE_MODELS)].copy()
    if by_rate_selected.empty:
        by_rate_selected = by_rate.copy()
    by_rate_selected = add_percent_columns(add_display_columns(by_rate_selected))
    by_rate_selected = by_rate_selected.sort_values(["plot_order", "cv_rate_pct"]).reset_index(drop=True)
    save_table(by_rate_selected, "full_hybrid_decision_by_cv_rate_test_comparison.csv")

    best_failure = overall_all.sort_values(["failure_f1", "residual_rmse_ft"], ascending=[False, True]).head(5).copy()
    best_residual = overall_all.sort_values(["residual_rmse_ft", "failure_f1"], ascending=[True, False]).head(5).copy()
    save_table(best_failure, "full_hybrid_decision_best_failure_f1_test.csv")
    save_table(best_residual, "full_hybrid_decision_best_residual_rmse_test.csv")

    if THRESHOLDS_FILE.exists():
        thresholds = pd.read_csv(THRESHOLDS_FILE)
        save_table(thresholds, "full_hybrid_decision_selected_thresholds_validation.csv")

    if TRAINING_SUMMARY_FILE.exists():
        training = pd.read_csv(TRAINING_SUMMARY_FILE)
        save_table(training, "full_hybrid_decision_training_summary.csv")

    print("\nSelected overall comparison:")
    display_cols = [
        "method_label",
        "model_family_type",
        "decision_mode",
        "profile_curve_id",
        "residual_mae_ft",
        "residual_rmse_ft",
        "residual_bias_ft",
        "failure_accuracy",
        "failure_precision",
        "failure_recall",
        "failure_f1",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    display_cols = [c for c in display_cols if c in overall_selected.columns]
    print(overall_selected[display_cols].round(3).to_string(index=False))

    save_bar_plot(
        overall_selected,
        y_col="residual_rmse_ft",
        title="Residual queue prediction error at green end: test set",
        ylabel="Residual queue RMSE (ft)",
        filename="full_hybrid_decision_residual_rmse_test_comparison.png",
        value_format="{:.1f}",
    )

    save_bar_plot(
        overall_selected,
        y_col="failure_f1",
        title="Cycle-failure classification performance: test set",
        ylabel="F1-score",
        filename="full_hybrid_decision_failure_f1_test_comparison.png",
        value_format="{:.2f}",
    )

    save_line_plot_by_rate(
        by_rate_selected,
        metric_col="residual_rmse_ft",
        title="Residual queue RMSE by CV penetration rate: test set",
        ylabel="Residual queue RMSE (ft)",
        filename="full_hybrid_decision_residual_rmse_by_cv_rate_test.png",
    )

    save_line_plot_by_rate(
        by_rate_selected,
        metric_col="failure_f1",
        title="Cycle-failure F1-score by CV penetration rate: test set",
        ylabel="F1-score",
        filename="full_hybrid_decision_failure_f1_by_cv_rate_test.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
