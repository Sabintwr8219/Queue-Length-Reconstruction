"""
Evaluate decision-oriented outcomes derived from reconstructed queue profiles.

Purpose
-------
This script implements decision baseline/proposed-method evaluation where the
operational decision is derived directly from a reconstructed queue profile.

For each reconstructed queue profile:
    predicted_residual_queue_ft = predicted queue at GT green-end time
    predicted_cycle_failure = 1 if predicted_residual_queue_ft >= threshold

Inputs
------
1) output/intermediate_csv/decision_labels/decision_labels_cycle_level.csv
2) output/intermediate_csv/method_family_queue_length_evaluation/
       method_family_predictions_allruns_allrates.csv
3) output/intermediate_csv/method_family_queue_length_evaluation/
       method_family_curve_catalog.csv

Outputs
-------
output/intermediate_csv/decision_from_profiles/
    decision_from_profiles_cycle_level.csv
    decision_from_profiles_metrics_by_split_rate_curve.csv
    decision_from_profiles_metrics_by_split_curve.csv
    decision_from_profiles_metrics_by_split_rate_condition_curve.csv
    decision_from_profiles_metrics_validation.csv
    decision_from_profiles_metrics_test.csv
    decision_from_profiles_metrics_test_low_cv.csv
    decision_from_profiles_metrics_test_congested.csv
"""

from __future__ import annotations

from config import PROJECT_ROOT, CV_RATES_PCT

from pathlib import Path
from typing import Iterable

import math
import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

DECISION_LABEL_FILE = (
    PROJECT_ROOT
    / "output"
    / "intermediate_csv"
    / "decision_labels"
    / "decision_labels_cycle_level.csv"
)

METHOD_FAMILY_DIR = (
    PROJECT_ROOT
    / "output"
    / "intermediate_csv"
    / "method_family_queue_length_evaluation"
)

METHOD_FAMILY_PRED_FILE = METHOD_FAMILY_DIR / "method_family_predictions_allruns_allrates.csv"
CURVE_CATALOG_FILE = METHOD_FAMILY_DIR / "method_family_curve_catalog.csv"

OUT_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "decision_from_profiles"

DEFAULT_FAILURE_THRESHOLD_FT = 25.0

# Used to match green-end timestamps robustly.
TIME_KEY_SCALE = 1000.0


# =============================================================================
# Utility functions
# =============================================================================

def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def make_time_key(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.rint(values * TIME_KEY_SCALE).astype("Int64")


def clean_metadata_value(value):
    if pd.isna(value):
        return ""
    return value


def cv_rate_group(rate) -> str:
    if pd.isna(rate):
        return "unknown"

    r = int(rate)

    if r <= 5:
        return "low_cv_1_5pct"
    if r <= 20:
        return "medium_cv_10_20pct"
    return "high_cv_50_100pct"


def safe_divide(num: float, den: float) -> float:
    if den == 0:
        return np.nan
    return float(num / den)


# =============================================================================
# Load inputs
# =============================================================================

def load_decision_labels() -> tuple[pd.DataFrame, float]:
    if not DECISION_LABEL_FILE.exists():
        raise FileNotFoundError(
            f"Could not find decision-label file:\n{DECISION_LABEL_FILE}\n"
            "Run src/generate_decision_labels.py first."
        )

    labels = pd.read_csv(DECISION_LABEL_FILE)

    required = [
        "cycle_uid",
        "run_id",
        "ml_split",
        "cycle_number",
        "green_end_time_sec",
        "residual_queue_ft",
        "cycle_failure",
        "traffic_condition",
        "gt_peak_queue_ft",
    ]
    require_columns(labels, required, "decision labels")

    numeric_cols = [
        "run_id",
        "cycle_number",
        "green_end_time_sec",
        "residual_queue_ft",
        "cycle_failure",
        "gt_peak_queue_ft",
    ]

    for col in numeric_cols:
        labels[col] = pd.to_numeric(labels[col], errors="coerce")

    labels = labels.dropna(
        subset=[
            "run_id",
            "cycle_number",
            "green_end_time_sec",
            "residual_queue_ft",
            "cycle_failure",
        ]
    ).copy()

    labels["run_id"] = labels["run_id"].astype(int)
    labels["cycle_number"] = labels["cycle_number"].astype(int)
    labels["cycle_failure"] = labels["cycle_failure"].astype(int)
    labels["ml_split"] = labels["ml_split"].astype(str)
    labels["traffic_condition"] = labels["traffic_condition"].astype(str)

    if "failure_threshold_ft" in labels.columns:
        threshold_values = pd.to_numeric(labels["failure_threshold_ft"], errors="coerce").dropna()
        if not threshold_values.empty:
            failure_threshold_ft = float(threshold_values.iloc[0])
        else:
            failure_threshold_ft = float(DEFAULT_FAILURE_THRESHOLD_FT)
    else:
        failure_threshold_ft = float(DEFAULT_FAILURE_THRESHOLD_FT)

    labels["time_key_ms"] = make_time_key(labels["green_end_time_sec"])

    return labels, failure_threshold_ft


def load_curve_catalog() -> pd.DataFrame:
    if not CURVE_CATALOG_FILE.exists():
        raise FileNotFoundError(
            f"Could not find curve catalog:\n{CURVE_CATALOG_FILE}\n"
            "Run src/evaluate_method_family_queue_length.py first."
        )

    catalog = pd.read_csv(CURVE_CATALOG_FILE)

    required = [
        "curve_id",
        "method_family",
        "model_name",
        "curve_label",
        "prediction_col",
    ]
    require_columns(catalog, required, "curve catalog")

    catalog = catalog.dropna(subset=["curve_id", "prediction_col"]).copy()
    catalog["curve_id"] = catalog["curve_id"].astype(str)
    catalog["prediction_col"] = catalog["prediction_col"].astype(str)

    return catalog


def load_method_family_predictions(prediction_cols: list[str]) -> pd.DataFrame:
    if not METHOD_FAMILY_PRED_FILE.exists():
        raise FileNotFoundError(
            f"Could not find method-family predictions:\n{METHOD_FAMILY_PRED_FILE}\n"
            "Run src/evaluate_method_family_queue_length.py first."
        )

    header = pd.read_csv(METHOD_FAMILY_PRED_FILE, nrows=0)
    available_cols = set(header.columns)

    base_cols = ["run_id", "cv_rate_pct", "time_sec"]
    missing_base = sorted(set(base_cols) - available_cols)
    if missing_base:
        raise ValueError(f"method-family prediction file missing base columns: {missing_base}")

    existing_prediction_cols = [c for c in prediction_cols if c in available_cols]

    if not existing_prediction_cols:
        raise ValueError(
            "None of the prediction columns in the curve catalog were found in the "
            "method-family prediction file."
        )

    usecols = base_cols + existing_prediction_cols

    pred = pd.read_csv(METHOD_FAMILY_PRED_FILE, usecols=usecols)

    for col in ["run_id", "cv_rate_pct", "time_sec"]:
        pred[col] = pd.to_numeric(pred[col], errors="coerce")

    pred = pred.dropna(subset=["run_id", "cv_rate_pct", "time_sec"]).copy()
    pred["run_id"] = pred["run_id"].astype(int)
    pred["cv_rate_pct"] = pred["cv_rate_pct"].astype(int)

    pred = pred[pred["cv_rate_pct"].isin(CV_RATES_PCT)].copy()

    for col in existing_prediction_cols:
        pred[col] = pd.to_numeric(pred[col], errors="coerce")

    pred["time_key_ms"] = make_time_key(pred["time_sec"])

    return pred


# =============================================================================
# Build cycle-level profile-derived decision table
# =============================================================================

def extract_profiles_at_green_end(
    labels: pd.DataFrame,
    pred: pd.DataFrame,
) -> pd.DataFrame:
    label_keys = labels[["run_id", "time_key_ms"]].drop_duplicates().copy()

    pred_at_green = pred.merge(
        label_keys,
        on=["run_id", "time_key_ms"],
        how="inner",
    ).copy()

    pred_at_green = pred_at_green.rename(columns={"time_sec": "matched_profile_time_sec"})

    label_cols = [
        "cycle_uid",
        "run_id",
        "ml_split",
        "cycle_number",
        "cycle_start_time_sec",
        "cycle_end_time_sec",
        "green_end_time_sec",
        "residual_queue_ft",
        "cycle_failure",
        "gt_peak_queue_ft",
        "traffic_condition",
        "time_key_ms",
    ]
    label_cols = [c for c in label_cols if c in labels.columns]

    wide = labels[label_cols].merge(
        pred_at_green,
        on=["run_id", "time_key_ms"],
        how="left",
    )

    if "matched_profile_time_sec" in wide.columns:
        wide["time_match_error_sec"] = (
            pd.to_numeric(wide["matched_profile_time_sec"], errors="coerce")
            - pd.to_numeric(wide["green_end_time_sec"], errors="coerce")
        )
    else:
        wide["time_match_error_sec"] = np.nan

    return wide


def build_long_decision_table(
    wide: pd.DataFrame,
    catalog: pd.DataFrame,
    failure_threshold_ft: float,
) -> pd.DataFrame:
    prediction_cols_existing = set(wide.columns)

    id_cols = [
        "cycle_uid",
        "run_id",
        "ml_split",
        "cycle_number",
        "cycle_start_time_sec",
        "cycle_end_time_sec",
        "green_end_time_sec",
        "matched_profile_time_sec",
        "time_match_error_sec",
        "cv_rate_pct",
        "residual_queue_ft",
        "cycle_failure",
        "gt_peak_queue_ft",
        "traffic_condition",
    ]
    id_cols = [c for c in id_cols if c in wide.columns]

    parts = []

    for _, row in catalog.iterrows():
        pred_col = str(row["prediction_col"])

        if pred_col not in prediction_cols_existing:
            continue

        tmp = wide[id_cols + [pred_col]].copy()
        tmp = tmp.rename(columns={pred_col: "pred_residual_queue_ft"})

        tmp["curve_id"] = clean_metadata_value(row.get("curve_id", ""))
        tmp["method_family"] = clean_metadata_value(row.get("method_family", ""))
        tmp["model_name"] = clean_metadata_value(row.get("model_name", ""))
        tmp["curve_label"] = clean_metadata_value(row.get("curve_label", ""))
        tmp["prediction_col"] = pred_col
        tmp["source"] = clean_metadata_value(row.get("source", ""))
        tmp["uses_physics"] = clean_metadata_value(row.get("uses_physics", ""))
        tmp["uses_ml"] = clean_metadata_value(row.get("uses_ml", ""))
        tmp["uses_cv"] = clean_metadata_value(row.get("uses_cv", ""))
        tmp["interpolation_method"] = clean_metadata_value(row.get("interpolation_method", ""))

        parts.append(tmp)

    if not parts:
        raise RuntimeError("No valid curve prediction columns were found for decision evaluation.")

    long = pd.concat(parts, ignore_index=True)

    long = long.rename(
        columns={
            "residual_queue_ft": "gt_residual_queue_ft",
            "cycle_failure": "gt_cycle_failure",
        }
    )

    long["gt_residual_queue_ft"] = pd.to_numeric(long["gt_residual_queue_ft"], errors="coerce")
    long["pred_residual_queue_ft"] = pd.to_numeric(long["pred_residual_queue_ft"], errors="coerce")
    long["gt_cycle_failure"] = pd.to_numeric(long["gt_cycle_failure"], errors="coerce")

    long["profile_valid"] = (
        np.isfinite(long["gt_residual_queue_ft"])
        & np.isfinite(long["pred_residual_queue_ft"])
        & np.isfinite(long["gt_cycle_failure"])
    )

    long["pred_cycle_failure"] = np.where(
        long["profile_valid"],
        (long["pred_residual_queue_ft"] >= float(failure_threshold_ft)).astype(int),
        np.nan,
    )

    long["residual_error_ft"] = (
        long["pred_residual_queue_ft"] - long["gt_residual_queue_ft"]
    )
    long["abs_residual_error_ft"] = np.abs(long["residual_error_ft"])
    long["failure_threshold_ft"] = float(failure_threshold_ft)

    long["cv_rate_group"] = long["cv_rate_pct"].apply(cv_rate_group)

    order_cols = [
        "cycle_uid",
        "run_id",
        "ml_split",
        "cycle_number",
        "cv_rate_pct",
        "cv_rate_group",
        "traffic_condition",
        "curve_id",
        "method_family",
        "model_name",
        "curve_label",
        "prediction_col",
        "source",
        "uses_physics",
        "uses_ml",
        "uses_cv",
        "interpolation_method",
        "cycle_start_time_sec",
        "cycle_end_time_sec",
        "green_end_time_sec",
        "matched_profile_time_sec",
        "time_match_error_sec",
        "gt_peak_queue_ft",
        "gt_residual_queue_ft",
        "pred_residual_queue_ft",
        "residual_error_ft",
        "abs_residual_error_ft",
        "failure_threshold_ft",
        "gt_cycle_failure",
        "pred_cycle_failure",
        "profile_valid",
    ]
    order_cols = [c for c in order_cols if c in long.columns]

    return long[order_cols].copy()


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics_for_group(g: pd.DataFrame) -> dict:
    n_rows = int(len(g))
    valid = g[g["profile_valid"].astype(bool)].copy()
    n_valid = int(len(valid))

    if n_valid == 0:
        return {
            "n_rows": n_rows,
            "n_valid": 0,
            "valid_pct": 0.0,
            "residual_mae_ft": np.nan,
            "residual_rmse_ft": np.nan,
            "residual_bias_ft": np.nan,
            "residual_max_abs_error_ft": np.nan,
            "gt_failure_rate_pct": np.nan,
            "pred_failure_rate_pct": np.nan,
            "failure_accuracy": np.nan,
            "failure_precision": np.nan,
            "failure_recall": np.nan,
            "failure_specificity": np.nan,
            "failure_f1": np.nan,
            "failure_false_alarm_rate": np.nan,
            "failure_miss_rate": np.nan,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

    gt_q = valid["gt_residual_queue_ft"].to_numpy(dtype=float)
    pred_q = valid["pred_residual_queue_ft"].to_numpy(dtype=float)
    err = pred_q - gt_q

    gt_fail = valid["gt_cycle_failure"].astype(int).to_numpy()
    pred_fail = valid["pred_cycle_failure"].astype(int).to_numpy()

    tp = int(np.sum((gt_fail == 1) & (pred_fail == 1)))
    fp = int(np.sum((gt_fail == 0) & (pred_fail == 1)))
    fn = int(np.sum((gt_fail == 1) & (pred_fail == 0)))
    tn = int(np.sum((gt_fail == 0) & (pred_fail == 0)))

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    accuracy = safe_divide(tp + tn, tp + fp + fn + tn)
    false_alarm_rate = safe_divide(fp, fp + tn)
    miss_rate = safe_divide(fn, fn + tp)

    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = np.nan

    return {
        "n_rows": n_rows,
        "n_valid": n_valid,
        "valid_pct": float(100.0 * n_valid / n_rows) if n_rows else np.nan,
        "residual_mae_ft": float(np.mean(np.abs(err))),
        "residual_rmse_ft": float(math.sqrt(np.mean(err ** 2))),
        "residual_bias_ft": float(np.mean(err)),
        "residual_max_abs_error_ft": float(np.max(np.abs(err))),
        "gt_failure_rate_pct": float(100.0 * np.mean(gt_fail)),
        "pred_failure_rate_pct": float(100.0 * np.mean(pred_fail)),
        "failure_accuracy": accuracy,
        "failure_precision": precision,
        "failure_recall": recall,
        "failure_specificity": specificity,
        "failure_f1": f1,
        "failure_false_alarm_rate": false_alarm_rate,
        "failure_miss_rate": miss_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def group_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []

    for key, g in df.groupby(group_cols, dropna=False, sort=True):
        if len(group_cols) == 1:
            key = (key,)

        row = {col: key[i] for i, col in enumerate(group_cols)}
        row.update(compute_metrics_for_group(g))
        rows.append(row)

    out = pd.DataFrame(rows)

    sort_cols = [c for c in group_cols if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    return out


def save_selected_summaries(metrics_split_curve: pd.DataFrame) -> None:
    validation = metrics_split_curve[
        metrics_split_curve["ml_split"].astype(str).eq("validation")
    ].copy()

    test = metrics_split_curve[
        metrics_split_curve["ml_split"].astype(str).eq("test")
    ].copy()

    validation = validation.sort_values(
        ["residual_rmse_ft", "residual_mae_ft"],
        ascending=[True, True],
    ).reset_index(drop=True)

    test = test.sort_values(
        ["residual_rmse_ft", "residual_mae_ft"],
        ascending=[True, True],
    ).reset_index(drop=True)

    validation.to_csv(OUT_DIR / "decision_from_profiles_metrics_validation.csv", index=False)
    test.to_csv(OUT_DIR / "decision_from_profiles_metrics_test.csv", index=False)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("Evaluating decisions derived from reconstructed queue profiles")
    print("=" * 96)
    print(f"Project root        : {PROJECT_ROOT}")
    print(f"Decision labels     : {DECISION_LABEL_FILE}")
    print(f"Method predictions  : {METHOD_FAMILY_PRED_FILE}")
    print(f"Curve catalog       : {CURVE_CATALOG_FILE}")
    print(f"Output dir          : {OUT_DIR}")
    print("=" * 96)

    labels, failure_threshold_ft = load_decision_labels()
    catalog = load_curve_catalog()

    prediction_cols = sorted(set(catalog["prediction_col"].astype(str)))
    pred = load_method_family_predictions(prediction_cols)

    existing_prediction_cols = [c for c in prediction_cols if c in pred.columns]
    catalog = catalog[catalog["prediction_col"].isin(existing_prediction_cols)].copy()

    print(f"Loaded decision labels      : {len(labels):,} cycles")
    print(f"Loaded profile rows         : {len(pred):,} rows")
    print(f"Curves evaluated            : {len(catalog):,}")
    print(f"Failure threshold           : {failure_threshold_ft:.1f} ft")

    print("\nDecision labels by split:")
    print(
        labels.groupby("ml_split")
        .size()
        .reset_index(name="n_cycles")
        .to_string(index=False)
    )

    wide = extract_profiles_at_green_end(labels, pred)

    expected_rows = len(labels) * len(CV_RATES_PCT)
    print(f"\nRows after green-end merge  : {len(wide):,}")
    print(f"Expected approx rows        : {expected_rows:,}")

    missing_rate_rows = int(wide["cv_rate_pct"].isna().sum()) if "cv_rate_pct" in wide.columns else len(wide)
    if missing_rate_rows:
        print(f"[WARN] Rows without matched profile time/rate: {missing_rate_rows:,}")

    long = build_long_decision_table(
        wide=wide,
        catalog=catalog,
        failure_threshold_ft=failure_threshold_ft,
    )

    cycle_level_path = OUT_DIR / "decision_from_profiles_cycle_level.csv"
    long.to_csv(cycle_level_path, index=False)

    metrics_by_split_rate_curve = group_metrics(
        long,
        [
            "ml_split",
            "cv_rate_pct",
            "cv_rate_group",
            "curve_id",
            "method_family",
            "model_name",
            "curve_label",
            "prediction_col",
        ],
    )

    metrics_by_split_curve = group_metrics(
        long,
        [
            "ml_split",
            "curve_id",
            "method_family",
            "model_name",
            "curve_label",
            "prediction_col",
        ],
    )

    metrics_by_split_rate_condition_curve = group_metrics(
        long,
        [
            "ml_split",
            "cv_rate_pct",
            "cv_rate_group",
            "traffic_condition",
            "curve_id",
            "method_family",
            "model_name",
            "curve_label",
            "prediction_col",
        ],
    )

    metrics_by_split_rate_curve.to_csv(
        OUT_DIR / "decision_from_profiles_metrics_by_split_rate_curve.csv",
        index=False,
    )

    metrics_by_split_curve.to_csv(
        OUT_DIR / "decision_from_profiles_metrics_by_split_curve.csv",
        index=False,
    )

    metrics_by_split_rate_condition_curve.to_csv(
        OUT_DIR / "decision_from_profiles_metrics_by_split_rate_condition_curve.csv",
        index=False,
    )

    save_selected_summaries(metrics_by_split_curve)

    test_low_cv = metrics_by_split_rate_curve[
        metrics_by_split_rate_curve["ml_split"].astype(str).eq("test")
        & metrics_by_split_rate_curve["cv_rate_group"].astype(str).eq("low_cv_1_5pct")
    ].copy()

    test_congested = metrics_by_split_rate_condition_curve[
        metrics_by_split_rate_condition_curve["ml_split"].astype(str).eq("test")
        & metrics_by_split_rate_condition_curve["traffic_condition"].astype(str).eq("congested")
    ].copy()

    test_low_cv.to_csv(
        OUT_DIR / "decision_from_profiles_metrics_test_low_cv.csv",
        index=False,
    )

    test_congested.to_csv(
        OUT_DIR / "decision_from_profiles_metrics_test_congested.csv",
        index=False,
    )

    print("\nSaved:")
    print(f"  {cycle_level_path}")
    print(f"  {OUT_DIR / 'decision_from_profiles_metrics_by_split_rate_curve.csv'}")
    print(f"  {OUT_DIR / 'decision_from_profiles_metrics_by_split_curve.csv'}")
    print(f"  {OUT_DIR / 'decision_from_profiles_metrics_by_split_rate_condition_curve.csv'}")
    print(f"  {OUT_DIR / 'decision_from_profiles_metrics_validation.csv'}")
    print(f"  {OUT_DIR / 'decision_from_profiles_metrics_test.csv'}")
    print(f"  {OUT_DIR / 'decision_from_profiles_metrics_test_low_cv.csv'}")
    print(f"  {OUT_DIR / 'decision_from_profiles_metrics_test_congested.csv'}")

    print("\nValidation profile-derived decision metrics:")
    validation = metrics_by_split_curve[
        metrics_by_split_curve["ml_split"].astype(str).eq("validation")
    ].copy()
    validation = validation.sort_values(["residual_rmse_ft", "residual_mae_ft"])
    cols = [
        "curve_id",
        "method_family",
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
    cols = [c for c in cols if c in validation.columns]
    print(validation[cols].round(3).to_string(index=False))

    print("\nTest profile-derived decision metrics:")
    test = metrics_by_split_curve[
        metrics_by_split_curve["ml_split"].astype(str).eq("test")
    ].copy()
    test = test.sort_values(["residual_rmse_ft", "residual_mae_ft"])
    cols = [c for c in cols if c in test.columns]
    print(test[cols].round(3).to_string(index=False))

    proposed = metrics_by_split_rate_curve[
        metrics_by_split_rate_curve["ml_split"].astype(str).eq("test")
        & metrics_by_split_rate_curve["curve_id"].astype(str).eq("physics_ml_cv_gru")
    ].copy()

    if not proposed.empty:
        print("\nProposed profile-derived decisions by CV rate:")
        proposed_cols = [
            "cv_rate_pct",
            "cv_rate_group",
            "n_valid",
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
        proposed_cols = [c for c in proposed_cols if c in proposed.columns]
        print(proposed[proposed_cols].round(3).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()