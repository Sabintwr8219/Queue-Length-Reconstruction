"""
Plot and summarize decision-oriented queue reconstruction results.

Purpose
-------
Create final decision-oriented comparison tables and publication-ready plots for:

1) Decisions derived from reconstructed queue profiles.
2) Direct cycle-level decision baselines.
3) CV penetration sensitivity.
4) Congested-cycle sensitivity.

Inputs
------
output/intermediate_csv/decision_from_profiles/
    decision_from_profiles_metrics_test.csv
    decision_from_profiles_metrics_by_split_rate_curve.csv
    decision_from_profiles_metrics_test_congested.csv

output/intermediate_csv/direct_decision_baselines/
    direct_decision_metrics_test.csv
    direct_decision_metrics_by_split_rate_model.csv

Outputs
-------
output/intermediate_csv/decision_oriented_summary/
output/tables/publication_outputs/
output/final_plots/publication_outputs/

Notes
-----
The GT row from profile-derived decision evaluation is kept only as a sanity check
in the raw input and is excluded from final comparison tables/plots.
"""

from __future__ import annotations

from config import PROJECT_ROOT

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================

PROFILE_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "decision_from_profiles"
DIRECT_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "direct_decision_baselines"

PROFILE_TEST_FILE = PROFILE_DIR / "decision_from_profiles_metrics_test.csv"
PROFILE_BY_RATE_FILE = PROFILE_DIR / "decision_from_profiles_metrics_by_split_rate_curve.csv"
PROFILE_CONGESTED_FILE = PROFILE_DIR / "decision_from_profiles_metrics_test_congested.csv"

DIRECT_TEST_FILE = DIRECT_DIR / "direct_decision_metrics_test.csv"
DIRECT_BY_RATE_FILE = DIRECT_DIR / "direct_decision_metrics_by_split_rate_model.csv"

OUT_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "decision_oriented_summary"
TABLE_DIR = PROJECT_ROOT / "output" / "tables" / "publication_outputs"
FIG_DIR = PROJECT_ROOT / "output" / "final_plots" / "publication_outputs"


# =============================================================================
# Settings
# =============================================================================

SELECTED_PROFILE_CURVES = [
    "physics_baseline",
    "physics_ml_gru",
    "physics_ml_cv_gru",
    "physics_ml_cv_xgb",
]

SELECTED_DIRECT_MODELS = [
    "direct_cycle_xgb_regression_threshold",
    "direct_cycle_xgb_classifier_probability",
]

PROFILE_LABELS = {
    "physics_baseline": "Physics baseline",
    "physics_ml_gru": "Physics + ML GRU",
    "physics_ml_cv_gru": "Physics + ML + CV GRU",
    "physics_ml_cv_xgb": "Physics + ML + CV XGBoost",
}

DIRECT_LABELS = {
    "direct_cycle_xgb_regression_threshold": "Direct cycle XGBoost\nregression-threshold",
    "direct_cycle_xgb_classifier_probability": "Direct cycle XGBoost\nclassifier",
}

PLOT_LABELS = {
    "physics_baseline": "Physics\nbaseline",
    "physics_ml_gru": "Physics + ML\nGRU",
    "physics_ml_cv_gru": "Physics + ML + CV\nGRU",
    "physics_ml_cv_xgb": "Physics + ML + CV\nXGBoost",
    "direct_cycle_xgb_regression_threshold": "Direct cycle\nXGBoost reg.",
    "direct_cycle_xgb_classifier_probability": "Direct cycle\nXGBoost cls.",
}

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
# Utilities
# =============================================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file:\n{path}")


def read_csv(path: Path) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path)


def add_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "failure_accuracy",
        "failure_precision",
        "failure_recall",
        "failure_f1",
    ]:
        if col in out.columns:
            out[f"{col}_pct"] = 100.0 * pd.to_numeric(out[col], errors="coerce")
    return out


def clean_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in METRIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def save_table(df: pd.DataFrame, filename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    out_intermediate = OUT_DIR / filename
    out_publication = TABLE_DIR / filename

    df.to_csv(out_intermediate, index=False)
    df.to_csv(out_publication, index=False)

    print(f"[Saved] {out_intermediate}")
    print(f"[Saved] {out_publication}")


def save_bar_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    filename: str,
    value_format: str = "{:.1f}",
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = df.copy()
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df[np.isfinite(plot_df[y_col])].copy()

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(plot_df))
    y = plot_df[y_col].to_numpy(dtype=float)

    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df[x_col].astype(str), rotation=25, ha="right")
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
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = df.copy()
    plot_df["cv_rate_pct"] = pd.to_numeric(plot_df["cv_rate_pct"], errors="coerce")
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    plot_df = plot_df.dropna(subset=["cv_rate_pct", metric_col]).copy()

    fig, ax = plt.subplots(figsize=(9.5, 5.3))

    for label, g in plot_df.groupby("method_label", sort=False):
        g = g.sort_values("cv_rate_pct")
        ax.plot(
            g["cv_rate_pct"].to_numpy(dtype=float),
            g[metric_col].to_numpy(dtype=float),
            marker="o",
            label=str(label),
        )

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("CV penetration rate (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    out_path = FIG_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Saved] {out_path}")


# =============================================================================
# Build comparison tables
# =============================================================================

def build_overall_test_comparison() -> pd.DataFrame:
    profile = read_csv(PROFILE_TEST_FILE)
    direct = read_csv(DIRECT_TEST_FILE)

    profile = profile[profile["curve_id"].isin(SELECTED_PROFILE_CURVES)].copy()
    profile = clean_metric_columns(profile)
    profile["source_type"] = "profile_derived"
    profile["method_id"] = profile["curve_id"].astype(str)
    profile["method_label"] = profile["curve_id"].map(PROFILE_LABELS).fillna(profile["curve_label"])
    profile["decision_mode"] = "queue_profile_threshold"

    profile_keep = [
        "source_type",
        "method_id",
        "method_label",
        "decision_mode",
    ] + METRIC_COLS
    profile_keep = [c for c in profile_keep if c in profile.columns]
    profile_out = profile[profile_keep].copy()

    direct = direct[direct["model_id"].isin(SELECTED_DIRECT_MODELS)].copy()
    direct = clean_metric_columns(direct)
    direct["source_type"] = "direct_cycle_baseline"
    direct["method_id"] = direct["model_id"].astype(str)
    direct["method_label"] = direct["model_id"].map(DIRECT_LABELS).fillna(direct["model_id"])

    direct_keep = [
        "source_type",
        "method_id",
        "method_label",
        "decision_mode",
    ] + METRIC_COLS
    direct_keep = [c for c in direct_keep if c in direct.columns]
    direct_out = direct[direct_keep].copy()

    combined = pd.concat([profile_out, direct_out], ignore_index=True)
    combined = add_percent_columns(combined)

    order = {
        "physics_baseline": 1,
        "physics_ml_gru": 2,
        "physics_ml_cv_gru": 3,
        "physics_ml_cv_xgb": 4,
        "direct_cycle_xgb_regression_threshold": 5,
        "direct_cycle_xgb_classifier_probability": 6,
    }
    combined["plot_order"] = combined["method_id"].map(order).fillna(99).astype(int)
    combined = combined.sort_values("plot_order").reset_index(drop=True)

    return combined


def build_by_rate_comparison() -> pd.DataFrame:
    profile = read_csv(PROFILE_BY_RATE_FILE)
    direct = read_csv(DIRECT_BY_RATE_FILE)

    profile = profile[
        (profile["ml_split"].astype(str).eq("test"))
        & (profile["curve_id"].isin(["physics_ml_cv_gru", "physics_ml_cv_xgb"]))
    ].copy()
    profile = clean_metric_columns(profile)
    profile["source_type"] = "profile_derived"
    profile["method_id"] = profile["curve_id"].astype(str)
    profile["method_label"] = profile["curve_id"].map(PROFILE_LABELS)
    profile["decision_mode"] = "queue_profile_threshold"

    profile_keep = [
        "source_type",
        "method_id",
        "method_label",
        "decision_mode",
        "cv_rate_pct",
        "cv_rate_group",
    ] + METRIC_COLS
    profile_keep = [c for c in profile_keep if c in profile.columns]
    profile_out = profile[profile_keep].copy()

    direct = direct[
        (direct["ml_split"].astype(str).eq("test"))
        & (direct["model_id"].astype(str).eq("direct_cycle_xgb_regression_threshold"))
    ].copy()
    direct = clean_metric_columns(direct)
    direct["source_type"] = "direct_cycle_baseline"
    direct["method_id"] = direct["model_id"].astype(str)
    direct["method_label"] = direct["model_id"].map(DIRECT_LABELS)
    direct_keep = [
        "source_type",
        "method_id",
        "method_label",
        "decision_mode",
        "cv_rate_pct",
        "cv_rate_group",
    ] + METRIC_COLS
    direct_keep = [c for c in direct_keep if c in direct.columns]
    direct_out = direct[direct_keep].copy()

    combined = pd.concat([profile_out, direct_out], ignore_index=True)
    combined = add_percent_columns(combined)
    combined["cv_rate_pct"] = pd.to_numeric(combined["cv_rate_pct"], errors="coerce")
    combined = combined.sort_values(["cv_rate_pct", "source_type", "method_id"]).reset_index(drop=True)

    return combined


def build_congested_profile_comparison() -> pd.DataFrame:
    congested = read_csv(PROFILE_CONGESTED_FILE)

    congested = congested[
        congested["curve_id"].isin(
            [
                "physics_baseline",
                "physics_ml_gru",
                "physics_ml_cv_gru",
                "physics_ml_cv_xgb",
            ]
        )
    ].copy()

    congested = clean_metric_columns(congested)
    congested["source_type"] = "profile_derived_congested"
    congested["method_id"] = congested["curve_id"].astype(str)
    congested["method_label"] = congested["curve_id"].map(PROFILE_LABELS)
    congested["decision_mode"] = "queue_profile_threshold"
    congested = add_percent_columns(congested)

    keep_cols = [
        "source_type",
        "method_id",
        "method_label",
        "decision_mode",
        "cv_rate_pct",
        "cv_rate_group",
        "traffic_condition",
    ] + METRIC_COLS + [
        "failure_accuracy_pct",
        "failure_precision_pct",
        "failure_recall_pct",
        "failure_f1_pct",
    ]
    keep_cols = [c for c in keep_cols if c in congested.columns]

    congested = congested[keep_cols].copy()
    congested["cv_rate_pct"] = pd.to_numeric(congested["cv_rate_pct"], errors="coerce")
    congested = congested.sort_values(["cv_rate_pct", "method_id"]).reset_index(drop=True)

    return congested


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("Decision-oriented final plots and tables")
    print("=" * 96)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output dir   : {OUT_DIR}")
    print(f"Table dir    : {TABLE_DIR}")
    print(f"Figure dir   : {FIG_DIR}")
    print("=" * 96)

    overall = build_overall_test_comparison()
    by_rate = build_by_rate_comparison()
    congested = build_congested_profile_comparison()

    save_table(overall, "decision_overall_test_comparison.csv")
    save_table(by_rate, "decision_by_cv_rate_test_comparison.csv")
    save_table(congested, "decision_congested_test_profile_comparison.csv")

    print("\nOverall test comparison:")
    display_cols = [
        "method_label",
        "decision_mode",
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
    display_cols = [c for c in display_cols if c in overall.columns]
    print(overall[display_cols].round(3).to_string(index=False))

    plot_df = overall.copy()
    plot_df["plot_label"] = plot_df["method_id"].map(PLOT_LABELS).fillna(plot_df["method_label"])

    save_bar_plot(
        df=plot_df,
        x_col="plot_label",
        y_col="residual_rmse_ft",
        title="Residual queue prediction error at green end: test set",
        ylabel="Residual queue RMSE (ft)",
        filename="decision_residual_rmse_test_comparison.png",
    )

    save_bar_plot(
        df=plot_df,
        x_col="plot_label",
        y_col="failure_f1",
        title="Cycle-failure classification performance: test set",
        ylabel="F1-score",
        filename="decision_failure_f1_test_comparison.png",
        value_format="{:.2f}",
    )

    rate_plot = by_rate.copy()
    rate_plot["method_label"] = rate_plot["method_id"].map(
        {
            "physics_ml_cv_gru": "Profile: Physics + ML + CV GRU",
            "physics_ml_cv_xgb": "Profile: Physics + ML + CV XGBoost",
            "direct_cycle_xgb_regression_threshold": "Direct cycle XGBoost",
        }
    ).fillna(rate_plot["method_label"])

    save_line_plot_by_rate(
        df=rate_plot,
        metric_col="residual_rmse_ft",
        title="Residual queue RMSE by CV penetration rate: test set",
        ylabel="Residual queue RMSE (ft)",
        filename="decision_residual_rmse_by_cv_rate_test.png",
    )

    save_line_plot_by_rate(
        df=rate_plot,
        metric_col="failure_f1",
        title="Cycle-failure F1-score by CV penetration rate: test set",
        ylabel="F1-score",
        filename="decision_failure_f1_by_cv_rate_test.png",
    )

    congested_selected = congested[
        congested["method_id"].isin(["physics_baseline", "physics_ml_gru", "physics_ml_cv_gru", "physics_ml_cv_xgb"])
    ].copy()

    # One compact congested-cycle line plot for F1. It uses only profile-derived methods.
    congested_plot = congested_selected.copy()
    congested_plot["method_label"] = congested_plot["method_id"].map(
        {
            "physics_baseline": "Physics baseline",
            "physics_ml_gru": "Physics + ML GRU",
            "physics_ml_cv_gru": "Physics + ML + CV GRU",
            "physics_ml_cv_xgb": "Physics + ML + CV XGBoost",
        }
    )

    save_line_plot_by_rate(
        df=congested_plot,
        metric_col="failure_f1",
        title="Cycle-failure F1-score on congested cycles: test set",
        ylabel="F1-score",
        filename="decision_congested_failure_f1_by_cv_rate_test.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
