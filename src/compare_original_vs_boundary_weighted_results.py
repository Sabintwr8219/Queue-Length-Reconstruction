"""
Compare original vs boundary-weighted Queue Length Reconstruction results.

Place this file at:
    src/compare_original_vs_boundary_weighted_results.py

Purpose
-------
This script compares the stable/original pipeline outputs against the
experimental boundary-weighted reconstruction outputs.

It reads:
    output/intermediate_csv/method_family_queue_length_evaluation
    output/intermediate_csv/method_family_queue_length_evaluation_boundary_weighted
    output/intermediate_csv/full_hybrid_decision_models
    output/intermediate_csv/full_hybrid_decision_models_boundary_weighted

It writes:
    output/intermediate_csv/boundary_weighted_comparison/
        queue_reconstruction_validation_comparison.csv
        queue_reconstruction_test_comparison.csv
        queue_reconstruction_test_best_summary.csv
        queue_reconstruction_test_by_rate_comparison.csv
        decision_validation_comparison.csv
        decision_test_comparison.csv
        decision_test_best_summary.csv
        decision_test_by_rate_comparison.csv
        boundary_weighted_comparison_readme.txt

Notes
-----
For queue reconstruction metrics, lower values are better.
For decision metrics, residual errors are lower-better and classification
metrics such as accuracy, precision, recall, and F1 are higher-better.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import math
import pandas as pd
import numpy as np

from config import PROJECT_ROOT


# =============================================================================
# Paths
# =============================================================================

INTERMEDIATE_DIR = PROJECT_ROOT / "output" / "intermediate_csv"

ORIG_METHOD_DIR = INTERMEDIATE_DIR / "method_family_queue_length_evaluation"
BW_METHOD_DIR = INTERMEDIATE_DIR / "method_family_queue_length_evaluation_boundary_weighted"

ORIG_DECISION_DIR = INTERMEDIATE_DIR / "full_hybrid_decision_models"
BW_DECISION_DIR = INTERMEDIATE_DIR / "full_hybrid_decision_models_boundary_weighted"

OUT_DIR = INTERMEDIATE_DIR / "boundary_weighted_comparison"


# =============================================================================
# Settings
# =============================================================================

QUEUE_ERROR_METRICS = [
    "mae_ft",
    "rmse_ft",
    "abc_ft_s",
    "mean_cycle_peak_error_ft",
    "median_cycle_peak_error_ft",
    "rmse_cycle_peak_error_ft",
]

DECISION_ERROR_METRICS = [
    "residual_mae_ft",
    "residual_rmse_ft",
    "residual_bias_ft",
    "residual_max_abs_error_ft",
]

DECISION_SCORE_METRICS = [
    "failure_accuracy",
    "failure_precision",
    "failure_recall",
    "failure_specificity",
    "failure_f1",
]

DECISION_COUNT_METRICS = ["tp", "fp", "fn", "tn"]

SELECTED_QUEUE_CURVES = [
    "physics_baseline",
    "physics_ml_gru",
    "physics_ml_cv_gru",
    "physics_ml_cv_xgb",
]

SELECTED_DECISION_MODELS = [
    "queue_derived_threshold_physics_baseline",
    "queue_derived_threshold_physics_ml_gru",
    "queue_derived_threshold_physics_ml_cv_gru",
    "queue_derived_threshold_physics_ml_cv_xgb",
    "direct_ml_decision_xgb_regression_threshold",
    "direct_ml_decision_xgb_dual_head",
    "full_hybrid_decision_xgb_from_physics_ml_cv_gru_regression_threshold",
    "full_hybrid_decision_xgb_from_physics_ml_cv_gru_dual_head",
    "full_hybrid_decision_xgb_from_physics_ml_cv_xgb_regression_threshold",
    "full_hybrid_decision_xgb_from_physics_ml_cv_xgb_dual_head",
]


# =============================================================================
# Helpers
# =============================================================================

def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}:\n{path}")


def read_csv_required(path: Path, label: str) -> pd.DataFrame:
    require_file(path, label)
    return pd.read_csv(path)


def safe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def first_existing_cols(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def improvement_pct_lower_better(original: pd.Series, boundary: pd.Series) -> pd.Series:
    o = pd.to_numeric(original, errors="coerce")
    b = pd.to_numeric(boundary, errors="coerce")
    out = 100.0 * (o - b) / o.replace(0, np.nan)
    return out


def improvement_pct_higher_better(original: pd.Series, boundary: pd.Series) -> pd.Series:
    o = pd.to_numeric(original, errors="coerce")
    b = pd.to_numeric(boundary, errors="coerce")
    out = 100.0 * (b - o) / o.replace(0, np.nan)
    return out


def add_delta_columns(
    comp: pd.DataFrame,
    metric_cols: list[str],
    lower_better: bool,
) -> pd.DataFrame:
    out = comp.copy()
    for m in metric_cols:
        o_col = f"{m}_original"
        b_col = f"{m}_boundary_weighted"
        if o_col not in out.columns or b_col not in out.columns:
            continue
        out[f"{m}_delta_bw_minus_original"] = pd.to_numeric(out[b_col], errors="coerce") - pd.to_numeric(out[o_col], errors="coerce")
        if lower_better:
            out[f"{m}_improvement_pct"] = improvement_pct_lower_better(out[o_col], out[b_col])
        else:
            out[f"{m}_improvement_pct"] = improvement_pct_higher_better(out[o_col], out[b_col])
    return out


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    path = OUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"[Saved] {path}")
    return path


def print_small_table(title: str, df: pd.DataFrame, cols: list[str], max_rows: int = 12) -> None:
    print("\n" + title)
    print("-" * len(title))
    if df.empty:
        print("[empty]")
        return
    existing = [c for c in cols if c in df.columns]
    if not existing:
        print(df.head(max_rows).to_string(index=False))
    else:
        print(df[existing].head(max_rows).to_string(index=False))


# =============================================================================
# Queue reconstruction comparison
# =============================================================================

def load_queue_summary(directory: Path, split: str) -> pd.DataFrame:
    path = directory / f"method_family_metrics_summary_{split}.csv"
    df = read_csv_required(path, f"queue reconstruction {split} summary")
    if "curve_id" in df.columns:
        df["curve_id"] = df["curve_id"].astype(str)
    if "method_family" in df.columns:
        df["method_family"] = df["method_family"].astype(str)
    if "model_name" in df.columns:
        df["model_name"] = df["model_name"].astype(str)
    numeric_cols = first_existing_cols(df, QUEUE_ERROR_METRICS)
    df = safe_numeric(df, numeric_cols)
    return df


def compare_queue_summary(split: str) -> pd.DataFrame:
    orig = load_queue_summary(ORIG_METHOD_DIR, split)
    bw = load_queue_summary(BW_METHOD_DIR, split)

    key = "curve_id" if "curve_id" in orig.columns and "curve_id" in bw.columns else "curve_label"
    keep_meta = [key]
    for c in ["method_family", "model_name", "curve_label"]:
        if c in orig.columns and c in bw.columns and c != key:
            keep_meta.append(c)

    metric_cols = sorted(set(first_existing_cols(orig, QUEUE_ERROR_METRICS)) & set(first_existing_cols(bw, QUEUE_ERROR_METRICS)))

    left = orig[keep_meta + metric_cols].copy()
    right = bw[keep_meta + metric_cols].copy()

    comp = left.merge(
        right,
        on=keep_meta,
        how="inner",
        suffixes=("_original", "_boundary_weighted"),
    )

    comp = add_delta_columns(comp, metric_cols, lower_better=True)

    if key == "curve_id":
        selected_order = {cid: i for i, cid in enumerate(SELECTED_QUEUE_CURVES)}
        comp["selected_order"] = comp[key].map(selected_order)
        comp = comp.sort_values(["selected_order", key], na_position="last").drop(columns=["selected_order"])
    else:
        comp = comp.sort_values(key)

    return comp.reset_index(drop=True)


def compare_queue_by_rate_test() -> pd.DataFrame:
    orig_path = ORIG_METHOD_DIR / "method_family_metrics_summary_by_split_rate_curve.csv"
    bw_path = BW_METHOD_DIR / "method_family_metrics_summary_by_split_rate_curve.csv"
    orig = read_csv_required(orig_path, "original queue by-rate metrics")
    bw = read_csv_required(bw_path, "boundary-weighted queue by-rate metrics")

    if "ml_split" in orig.columns:
        orig = orig[orig["ml_split"].astype(str).eq("test")].copy()
    if "ml_split" in bw.columns:
        bw = bw[bw["ml_split"].astype(str).eq("test")].copy()

    keys = [c for c in ["ml_split", "cv_rate_pct", "curve_id", "method_family", "model_name", "curve_label"] if c in orig.columns and c in bw.columns]
    metric_cols = sorted(set(first_existing_cols(orig, QUEUE_ERROR_METRICS)) & set(first_existing_cols(bw, QUEUE_ERROR_METRICS)))

    orig = safe_numeric(orig, ["cv_rate_pct"] + metric_cols)
    bw = safe_numeric(bw, ["cv_rate_pct"] + metric_cols)

    comp = orig[keys + metric_cols].merge(
        bw[keys + metric_cols],
        on=keys,
        how="inner",
        suffixes=("_original", "_boundary_weighted"),
    )
    comp = add_delta_columns(comp, metric_cols, lower_better=True)
    sort_cols = [c for c in ["curve_id", "cv_rate_pct"] if c in comp.columns]
    if sort_cols:
        comp = comp.sort_values(sort_cols)
    return comp.reset_index(drop=True)


def make_queue_best_summary(test_comp: pd.DataFrame) -> pd.DataFrame:
    if test_comp.empty:
        return pd.DataFrame()

    out = test_comp.copy()
    if "curve_id" in out.columns:
        out = out[out["curve_id"].isin(SELECTED_QUEUE_CURVES)].copy()

    cols = [
        "curve_id",
        "curve_label",
        "method_family",
        "model_name",
        "mae_ft_original",
        "mae_ft_boundary_weighted",
        "mae_ft_improvement_pct",
        "rmse_ft_original",
        "rmse_ft_boundary_weighted",
        "rmse_ft_improvement_pct",
        "mean_cycle_peak_error_ft_original",
        "mean_cycle_peak_error_ft_boundary_weighted",
        "mean_cycle_peak_error_ft_improvement_pct",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index(drop=True)


# =============================================================================
# Decision comparison
# =============================================================================

def load_decision_summary(directory: Path, split: str) -> pd.DataFrame:
    path = directory / f"full_hybrid_decision_metrics_{split}.csv"
    df = read_csv_required(path, f"decision {split} summary")
    if "model_id" in df.columns:
        df["model_id"] = df["model_id"].astype(str)
    if "model_family_type" in df.columns:
        df["model_family_type"] = df["model_family_type"].astype(str)
    if "decision_mode" in df.columns:
        df["decision_mode"] = df["decision_mode"].astype(str)
    if "profile_curve_id" in df.columns:
        df["profile_curve_id"] = df["profile_curve_id"].astype(str)

    numeric_cols = first_existing_cols(df, DECISION_ERROR_METRICS + DECISION_SCORE_METRICS + DECISION_COUNT_METRICS)
    df = safe_numeric(df, numeric_cols)
    return df


def compare_decision_summary(split: str) -> pd.DataFrame:
    orig = load_decision_summary(ORIG_DECISION_DIR, split)
    bw = load_decision_summary(BW_DECISION_DIR, split)

    keys = [c for c in ["model_id", "model_family_type", "decision_mode", "profile_curve_id"] if c in orig.columns and c in bw.columns]
    metric_cols = sorted(
        set(first_existing_cols(orig, DECISION_ERROR_METRICS + DECISION_SCORE_METRICS + DECISION_COUNT_METRICS))
        & set(first_existing_cols(bw, DECISION_ERROR_METRICS + DECISION_SCORE_METRICS + DECISION_COUNT_METRICS))
    )

    comp = orig[keys + metric_cols].merge(
        bw[keys + metric_cols],
        on=keys,
        how="inner",
        suffixes=("_original", "_boundary_weighted"),
    )

    error_metrics = [m for m in DECISION_ERROR_METRICS if m in metric_cols]
    score_metrics = [m for m in DECISION_SCORE_METRICS if m in metric_cols]

    # Residual bias is signed, so percent improvement is not always meaningful.
    # Still compute lower-better deltas only for absolute error metrics.
    lower_error_metrics = [m for m in error_metrics if m != "residual_bias_ft"]
    comp = add_delta_columns(comp, lower_error_metrics, lower_better=True)

    # Bias delta only.
    if "residual_bias_ft" in metric_cols:
        comp["residual_bias_ft_delta_bw_minus_original"] = (
            pd.to_numeric(comp["residual_bias_ft_boundary_weighted"], errors="coerce")
            - pd.to_numeric(comp["residual_bias_ft_original"], errors="coerce")
        )

    comp = add_delta_columns(comp, score_metrics, lower_better=False)

    if "model_id" in comp.columns:
        selected_order = {mid: i for i, mid in enumerate(SELECTED_DECISION_MODELS)}
        comp["selected_order"] = comp["model_id"].map(selected_order)
        comp = comp.sort_values(["selected_order", "model_id"], na_position="last").drop(columns=["selected_order"])

    return comp.reset_index(drop=True)


def compare_decision_by_rate_test() -> pd.DataFrame:
    orig_path = ORIG_DECISION_DIR / "full_hybrid_decision_metrics_by_split_rate_model.csv"
    bw_path = BW_DECISION_DIR / "full_hybrid_decision_metrics_by_split_rate_model.csv"
    orig = read_csv_required(orig_path, "original decision by-rate metrics")
    bw = read_csv_required(bw_path, "boundary-weighted decision by-rate metrics")

    if "ml_split" in orig.columns:
        orig = orig[orig["ml_split"].astype(str).eq("test")].copy()
    if "ml_split" in bw.columns:
        bw = bw[bw["ml_split"].astype(str).eq("test")].copy()

    keys = [c for c in ["ml_split", "cv_rate_pct", "model_id", "model_family_type", "decision_mode", "profile_curve_id"] if c in orig.columns and c in bw.columns]
    metric_cols = sorted(
        set(first_existing_cols(orig, DECISION_ERROR_METRICS + DECISION_SCORE_METRICS + DECISION_COUNT_METRICS))
        & set(first_existing_cols(bw, DECISION_ERROR_METRICS + DECISION_SCORE_METRICS + DECISION_COUNT_METRICS))
    )

    orig = safe_numeric(orig, ["cv_rate_pct"] + metric_cols)
    bw = safe_numeric(bw, ["cv_rate_pct"] + metric_cols)

    comp = orig[keys + metric_cols].merge(
        bw[keys + metric_cols],
        on=keys,
        how="inner",
        suffixes=("_original", "_boundary_weighted"),
    )

    lower_error_metrics = [m for m in DECISION_ERROR_METRICS if m in metric_cols and m != "residual_bias_ft"]
    score_metrics = [m for m in DECISION_SCORE_METRICS if m in metric_cols]
    comp = add_delta_columns(comp, lower_error_metrics, lower_better=True)
    comp = add_delta_columns(comp, score_metrics, lower_better=False)

    sort_cols = [c for c in ["model_id", "cv_rate_pct"] if c in comp.columns]
    if sort_cols:
        comp = comp.sort_values(sort_cols)
    return comp.reset_index(drop=True)


def make_decision_best_summary(test_comp: pd.DataFrame) -> pd.DataFrame:
    if test_comp.empty:
        return pd.DataFrame()

    out = test_comp.copy()
    if "model_id" in out.columns:
        out = out[out["model_id"].isin(SELECTED_DECISION_MODELS)].copy()

    cols = [
        "model_id",
        "model_family_type",
        "decision_mode",
        "profile_curve_id",
        "residual_mae_ft_original",
        "residual_mae_ft_boundary_weighted",
        "residual_mae_ft_improvement_pct",
        "residual_rmse_ft_original",
        "residual_rmse_ft_boundary_weighted",
        "residual_rmse_ft_improvement_pct",
        "failure_f1_original",
        "failure_f1_boundary_weighted",
        "failure_f1_improvement_pct",
        "failure_accuracy_original",
        "failure_accuracy_boundary_weighted",
        "failure_precision_original",
        "failure_precision_boundary_weighted",
        "failure_recall_original",
        "failure_recall_boundary_weighted",
        "tp_original",
        "tp_boundary_weighted",
        "fp_original",
        "fp_boundary_weighted",
        "fn_original",
        "fn_boundary_weighted",
        "tn_original",
        "tn_boundary_weighted",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index(drop=True)


# =============================================================================
# README summary
# =============================================================================

def write_readme(
    queue_test_best: pd.DataFrame,
    decision_test_best: pd.DataFrame,
) -> None:
    lines = []
    lines.append("Boundary-weighted comparison summary")
    lines.append("=" * 44)
    lines.append("")
    lines.append("This folder compares the original/stable pipeline against the experimental boundary-weighted reconstruction pipeline.")
    lines.append("")
    lines.append("Lower is better for queue reconstruction errors and residual-queue errors.")
    lines.append("Higher is better for decision classification metrics such as accuracy, precision, recall, and F1.")
    lines.append("")

    if not queue_test_best.empty and "rmse_ft_boundary_weighted" in queue_test_best.columns:
        q = queue_test_best.sort_values("rmse_ft_boundary_weighted").iloc[0]
        lines.append("Best boundary-weighted queue reconstruction row by test RMSE:")
        lines.append(f"  curve_id: {q.get('curve_id', '')}")
        lines.append(f"  curve_label: {q.get('curve_label', '')}")
        lines.append(f"  original RMSE: {q.get('rmse_ft_original', np.nan):.3f} ft")
        lines.append(f"  boundary-weighted RMSE: {q.get('rmse_ft_boundary_weighted', np.nan):.3f} ft")
        if pd.notna(q.get("rmse_ft_improvement_pct", np.nan)):
            lines.append(f"  RMSE improvement: {q.get('rmse_ft_improvement_pct'):.2f}%")
        lines.append("")

    if not decision_test_best.empty and "failure_f1_boundary_weighted" in decision_test_best.columns:
        d = decision_test_best.sort_values("failure_f1_boundary_weighted", ascending=False).iloc[0]
        lines.append("Best boundary-weighted decision row by test F1:")
        lines.append(f"  model_id: {d.get('model_id', '')}")
        lines.append(f"  original F1: {d.get('failure_f1_original', np.nan):.3f}")
        lines.append(f"  boundary-weighted F1: {d.get('failure_f1_boundary_weighted', np.nan):.3f}")
        if pd.notna(d.get("failure_f1_improvement_pct", np.nan)):
            lines.append(f"  F1 improvement: {d.get('failure_f1_improvement_pct'):.2f}%")
        lines.append(f"  boundary-weighted residual RMSE: {d.get('residual_rmse_ft_boundary_weighted', np.nan):.3f} ft")
        lines.append(f"  boundary-weighted precision: {d.get('failure_precision_boundary_weighted', np.nan):.3f}")
        lines.append(f"  boundary-weighted recall: {d.get('failure_recall_boundary_weighted', np.nan):.3f}")
        lines.append("")

    out_path = OUT_DIR / "boundary_weighted_comparison_readme.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Saved] {out_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_dirs()

    print("=" * 88)
    print("Comparing original vs boundary-weighted results")
    print("=" * 88)
    print(f"Project root     : {PROJECT_ROOT}")
    print(f"Original method  : {ORIG_METHOD_DIR}")
    print(f"BW method        : {BW_METHOD_DIR}")
    print(f"Original decision: {ORIG_DECISION_DIR}")
    print(f"BW decision      : {BW_DECISION_DIR}")
    print(f"Output dir       : {OUT_DIR}")
    print("=" * 88)

    q_val = compare_queue_summary("validation")
    q_test = compare_queue_summary("test")
    q_rate = compare_queue_by_rate_test()
    q_best = make_queue_best_summary(q_test)

    save_csv(q_val, "queue_reconstruction_validation_comparison.csv")
    save_csv(q_test, "queue_reconstruction_test_comparison.csv")
    save_csv(q_rate, "queue_reconstruction_test_by_rate_comparison.csv")
    save_csv(q_best, "queue_reconstruction_test_best_summary.csv")

    d_val = compare_decision_summary("validation")
    d_test = compare_decision_summary("test")
    d_rate = compare_decision_by_rate_test()
    d_best = make_decision_best_summary(d_test)

    save_csv(d_val, "decision_validation_comparison.csv")
    save_csv(d_test, "decision_test_comparison.csv")
    save_csv(d_rate, "decision_test_by_rate_comparison.csv")
    save_csv(d_best, "decision_test_best_summary.csv")

    write_readme(q_best, d_best)

    print_small_table(
        "Queue reconstruction test comparison, selected rows",
        q_best,
        [
            "curve_id",
            "rmse_ft_original",
            "rmse_ft_boundary_weighted",
            "rmse_ft_improvement_pct",
            "mae_ft_original",
            "mae_ft_boundary_weighted",
            "mae_ft_improvement_pct",
        ],
    )

    print_small_table(
        "Decision test comparison, selected rows",
        d_best,
        [
            "model_id",
            "residual_rmse_ft_original",
            "residual_rmse_ft_boundary_weighted",
            "residual_rmse_ft_improvement_pct",
            "failure_f1_original",
            "failure_f1_boundary_weighted",
            "failure_f1_improvement_pct",
            "failure_precision_boundary_weighted",
            "failure_recall_boundary_weighted",
        ],
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
