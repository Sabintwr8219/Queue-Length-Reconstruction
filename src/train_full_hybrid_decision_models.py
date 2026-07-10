"""
Train full hybrid cycle-level decision models for residual queue and cycle failure.

Purpose
-------
This script implements the full hybrid decision model requested after supervisor
feedback. It treats the earlier decision-oriented work as baseline analysis and
adds a direct decision head that uses hybrid information instead of only
thresholding the reconstructed queue at the end of green.

The script compares three decision formulations:

1) Queue-derived baseline
   reconstructed profile -> q_hat at green end -> threshold -> cycle failure

2) Direct ML-only decision baseline
   cycle-level signal/count/CV/physics-context features -> residual/failure
   prediction, without reconstructed queue-profile features

3) Full hybrid decision model
   cycle-level features + reconstructed profile summaries + physics baseline
   summaries -> direct residual queue prediction and cycle-failure probability

Inputs
------
output/intermediate_csv/decision_labels/decision_labels_cycle_level.csv
output/intermediate_csv/cv_features/timegrid_features_allruns_allrates.csv
output/intermediate_csv/method_family_queue_length_evaluation/
    method_family_predictions_allruns_allrates.csv
output/intermediate_csv/method_family_queue_length_evaluation/
    method_family_curve_catalog.csv

Outputs
-------
output/intermediate_csv/full_hybrid_decision_models/
    full_hybrid_decision_features_allrates.csv
    direct_ml_decision_features_allrates.csv
    queue_derived_decision_predictions_allrates.csv
    full_hybrid_decision_predictions_allrates.csv
    full_hybrid_decision_metrics_by_split_rate_model.csv
    full_hybrid_decision_metrics_by_split_model.csv
    full_hybrid_decision_metrics_by_split_rate_condition_model.csv
    full_hybrid_decision_metrics_validation.csv
    full_hybrid_decision_metrics_test.csv
    full_hybrid_decision_metrics_by_rate.csv
    selected_thresholds_validation.csv
    full_hybrid_decision_feature_columns_used.csv
    full_hybrid_decision_training_summary.csv
    skipped_cycle_rate_rows.csv
    trained_models/
        *.joblib
"""

from __future__ import annotations

from config import (
    PROJECT_ROOT,
    TRAIN_RUN_IDS,
    VALIDATION_RUN_IDS,
    TEST_RUN_IDS,
    CV_RATES_PCT,
    XGB_RANDOM_SEED,
    XGB_N_ESTIMATORS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE,
    XGB_SUBSAMPLE,
    XGB_COLSAMPLE_BYTREE,
    XGB_REG_LAMBDA,
)

from pathlib import Path
from typing import Iterable
import math
import re

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBRegressor = None
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


# =============================================================================
# Paths and configuration
# =============================================================================

DECISION_LABEL_FILE = (
    PROJECT_ROOT
    / "output"
    / "intermediate_csv"
    / "decision_labels"
    / "decision_labels_cycle_level.csv"
)

TIMEGRID_FEATURE_FILE = (
    PROJECT_ROOT
    / "output"
    / "intermediate_csv"
    / "cv_features"
    / "timegrid_features_allruns_allrates.csv"
)

METHOD_FAMILY_DIR = (
    PROJECT_ROOT
    / "output"
    / "intermediate_csv"
    / "method_family_queue_length_evaluation"
)

METHOD_FAMILY_PRED_FILE = METHOD_FAMILY_DIR / "method_family_predictions_allruns_allrates.csv"
CURVE_CATALOG_FILE = METHOD_FAMILY_DIR / "method_family_curve_catalog.csv"

OUT_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "full_hybrid_decision_models"
MODEL_DIR = OUT_DIR / "trained_models"

DEFAULT_FAILURE_THRESHOLD_FT = 25.0
MIN_ROWS_PER_CYCLE_WINDOW = 2
CLASSIFIER_THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)

# Queue curves to use for queue-derived baselines and hybrid profile features.
# These are intentionally aligned with the current publication comparison script.
QUEUE_DERIVED_CURVES = [
    "physics_baseline",
    "physics_ml_gru",
    "physics_ml_cv_gru",
    "physics_ml_cv_xgb",
]

# Full hybrid decision heads are trained separately using these reconstructed
# profile families as the reconstruction-summary input.
FULL_HYBRID_PROFILE_CURVES = [
    "physics_ml_cv_gru",
    "physics_ml_cv_xgb",
]

PHYSICS_BASELINE_CURVE = "physics_baseline"


# =============================================================================
# Column configuration
# =============================================================================

LABEL_REQUIRED_COLS = [
    "cycle_uid",
    "run_id",
    "ml_split",
    "cycle_number",
    "cycle_start_time_sec",
    "cycle_end_time_sec",
    "green_start_time_sec",
    "green_end_time_sec",
    "residual_queue_ft",
    "cycle_failure",
    "traffic_condition",
]

TIMEGRID_REQUIRED_COLS = [
    "run_id",
    "cv_rate_pct",
    "time_sec",
    "phase_state",
    "phase_elapsed_sec",
    "A_count",
    "D_count",
]

TIMEGRID_OPTIONAL_COLS = [
    "V_count",
    "B_count",
    "n_queue_cumulative",
    "l_eff_fixed_ft",
    "inside_cv_segment",
    "prev_cv_anchor_q_ft",
    "next_cv_anchor_q_ft",
    "time_since_prev_cv_sec",
    "time_to_next_cv_sec",
    "cv_segment_duration_sec",
    "cv_segment_frac",
]

TARGET_RESIDUAL_COL = "residual_queue_ft"
TARGET_FAILURE_COL = "cycle_failure"

NON_FEATURE_COLS = {
    "cycle_uid",
    "run_id",
    "ml_split",
    "cycle_number",
    "cv_rate_group",
    "traffic_condition",
    "residual_queue_ft",
    "cycle_failure",
    "failure_threshold_ft",
    "profile_curve_id",
    "profile_curve_label",
    "profile_prediction_col",
}

LEAKAGE_TERMS = [
    "gt_",
    "q_gt",
    "ground_truth",
    "residual_queue_ft",
    "cycle_failure",
]

METRIC_COLS = [
    "residual_mae_ft",
    "residual_rmse_ft",
    "residual_bias_ft",
    "residual_max_abs_error_ft",
    "failure_accuracy",
    "failure_precision",
    "failure_recall",
    "failure_specificity",
    "failure_f1",
    "failure_false_alarm_rate",
    "failure_miss_rate",
    "tp",
    "fp",
    "fn",
    "tn",
]


# =============================================================================
# Utilities
# =============================================================================

def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def assign_ml_split(run_id: int) -> str:
    if int(run_id) in TRAIN_RUN_IDS:
        return "train"
    if int(run_id) in VALIDATION_RUN_IDS:
        return "validation"
    if int(run_id) in TEST_RUN_IDS:
        return "test"
    return "other"


def cv_rate_group(rate) -> str:
    if pd.isna(rate):
        return "unknown"
    r = int(rate)
    if r <= 5:
        return "low_cv_1_5pct"
    if r <= 20:
        return "medium_cv_10_20pct"
    return "high_cv_50_100pct"


def classify_phase_state(value) -> str:
    p = str(value).strip().lower()
    if p in {"red", "r"} or p.startswith("red"):
        return "red"
    if p in {"yellow", "y", "amber"} or p.startswith("yellow") or "amber" in p:
        return "yellow"
    if p in {"green", "g"} or p.startswith("green") or "green" in p:
        return "green"
    return "other"


def safe_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def safe_divide(num: float, den: float) -> float:
    if den == 0:
        return np.nan
    return float(num / den)


def finite_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yp)
    return y[mask], yp[mask]


def slugify(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip())
    out = re.sub(r"_+", "_", out).strip("_")
    return out.lower() or "model"


def col_at(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return np.nan
    return safe_float(row[col])


# =============================================================================
# Loading inputs
# =============================================================================

def load_decision_labels() -> tuple[pd.DataFrame, float]:
    if not DECISION_LABEL_FILE.exists():
        raise FileNotFoundError(
            f"Could not find decision labels:\n{DECISION_LABEL_FILE}\n"
            "Run src/generate_decision_labels.py first."
        )

    labels = pd.read_csv(DECISION_LABEL_FILE)
    require_columns(labels, LABEL_REQUIRED_COLS, "decision labels")

    numeric_cols = [
        "run_id",
        "cycle_number",
        "cycle_start_time_sec",
        "cycle_end_time_sec",
        "green_start_time_sec",
        "green_end_time_sec",
        "residual_queue_ft",
        "cycle_failure",
        "failure_threshold_ft",
    ]
    for col in numeric_cols:
        if col in labels.columns:
            labels[col] = pd.to_numeric(labels[col], errors="coerce")

    labels = labels.dropna(
        subset=[
            "run_id",
            "cycle_number",
            "cycle_start_time_sec",
            "green_end_time_sec",
            "residual_queue_ft",
            "cycle_failure",
        ]
    ).copy()

    labels["run_id"] = labels["run_id"].astype(int)
    labels["cycle_number"] = labels["cycle_number"].astype(int)
    labels["cycle_failure"] = labels["cycle_failure"].astype(int)
    labels["ml_split"] = labels["run_id"].apply(assign_ml_split)
    labels["traffic_condition"] = labels["traffic_condition"].astype(str)

    if "failure_threshold_ft" in labels.columns:
        threshold_values = pd.to_numeric(labels["failure_threshold_ft"], errors="coerce").dropna()
        failure_threshold_ft = float(threshold_values.iloc[0]) if not threshold_values.empty else DEFAULT_FAILURE_THRESHOLD_FT
    else:
        labels["failure_threshold_ft"] = DEFAULT_FAILURE_THRESHOLD_FT
        failure_threshold_ft = DEFAULT_FAILURE_THRESHOLD_FT

    model_runs = sorted(set(TRAIN_RUN_IDS + VALIDATION_RUN_IDS + TEST_RUN_IDS))
    labels = labels[labels["run_id"].isin(model_runs)].copy()
    labels = labels.sort_values(["run_id", "cycle_number"]).reset_index(drop=True)

    return labels, float(failure_threshold_ft)


def load_timegrid_features() -> pd.DataFrame:
    if not TIMEGRID_FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find time-grid feature table:\n{TIMEGRID_FEATURE_FILE}\n"
            "Run src/build_cv_features.py first."
        )

    header = pd.read_csv(TIMEGRID_FEATURE_FILE, nrows=0)
    available = set(header.columns)
    require_columns(header, TIMEGRID_REQUIRED_COLS, "timegrid feature table")

    usecols = TIMEGRID_REQUIRED_COLS + [c for c in TIMEGRID_OPTIONAL_COLS if c in available]
    usecols = list(dict.fromkeys(usecols))
    df = pd.read_csv(TIMEGRID_FEATURE_FILE, usecols=usecols)

    for col in usecols:
        if col == "phase_state":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["run_id", "cv_rate_pct", "time_sec"]).copy()
    df["run_id"] = df["run_id"].astype(int)
    df["cv_rate_pct"] = df["cv_rate_pct"].astype(int)
    df["phase_state"] = df["phase_state"].astype(str).str.strip().str.lower()
    df["phase_class"] = df["phase_state"].apply(classify_phase_state)

    model_runs = sorted(set(TRAIN_RUN_IDS + VALIDATION_RUN_IDS + TEST_RUN_IDS))
    df = df[df["run_id"].isin(model_runs)].copy()
    df = df[df["cv_rate_pct"].isin(CV_RATES_PCT)].copy()
    df = df.sort_values(["run_id", "cv_rate_pct", "time_sec"]).reset_index(drop=True)
    return df


def load_curve_catalog() -> pd.DataFrame:
    if not CURVE_CATALOG_FILE.exists():
        raise FileNotFoundError(
            f"Could not find curve catalog:\n{CURVE_CATALOG_FILE}\n"
            "Run src/evaluate_method_family_queue_length.py first."
        )

    catalog = pd.read_csv(CURVE_CATALOG_FILE)
    require_columns(
        catalog,
        ["curve_id", "method_family", "model_name", "curve_label", "prediction_col"],
        "curve catalog",
    )

    catalog = catalog.dropna(subset=["curve_id", "prediction_col"]).copy()
    catalog["curve_id"] = catalog["curve_id"].astype(str)
    catalog["prediction_col"] = catalog["prediction_col"].astype(str)
    return catalog


def load_method_family_predictions(catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not METHOD_FAMILY_PRED_FILE.exists():
        raise FileNotFoundError(
            f"Could not find method-family predictions:\n{METHOD_FAMILY_PRED_FILE}\n"
            "Run src/evaluate_method_family_queue_length.py first."
        )

    selected_curve_ids = sorted(set(QUEUE_DERIVED_CURVES + FULL_HYBRID_PROFILE_CURVES + [PHYSICS_BASELINE_CURVE]))
    selected_catalog = catalog[catalog["curve_id"].isin(selected_curve_ids)].copy()
    if selected_catalog.empty:
        raise ValueError(f"None of the requested curve IDs were found in the curve catalog: {selected_curve_ids}")

    header = pd.read_csv(METHOD_FAMILY_PRED_FILE, nrows=0)
    available = set(header.columns)
    base_cols = ["run_id", "cv_rate_pct", "time_sec"]
    require_columns(header, base_cols, "method-family prediction table")

    selected_catalog = selected_catalog[selected_catalog["prediction_col"].isin(available)].copy()
    if selected_catalog.empty:
        raise ValueError("Selected curves were found in the catalog, but their prediction columns were not in the prediction file.")

    prediction_cols = sorted(set(selected_catalog["prediction_col"].astype(str)))
    pred = pd.read_csv(METHOD_FAMILY_PRED_FILE, usecols=base_cols + prediction_cols)

    for col in base_cols + prediction_cols:
        pred[col] = pd.to_numeric(pred[col], errors="coerce")

    pred = pred.dropna(subset=["run_id", "cv_rate_pct", "time_sec"]).copy()
    pred["run_id"] = pred["run_id"].astype(int)
    pred["cv_rate_pct"] = pred["cv_rate_pct"].astype(int)

    model_runs = sorted(set(TRAIN_RUN_IDS + VALIDATION_RUN_IDS + TEST_RUN_IDS))
    pred = pred[pred["run_id"].isin(model_runs)].copy()
    pred = pred[pred["cv_rate_pct"].isin(CV_RATES_PCT)].copy()
    pred = pred.sort_values(["run_id", "cv_rate_pct", "time_sec"]).reset_index(drop=True)

    return pred, selected_catalog


# =============================================================================
# Feature construction
# =============================================================================

def add_change_rate_features(out: dict, start_row: pd.Series, end_row: pd.Series, col: str, prefix: str, elapsed_sec: float) -> None:
    start_val = col_at(start_row, col)
    end_val = col_at(end_row, col)
    delta = end_val - start_val if np.isfinite(start_val) and np.isfinite(end_val) else np.nan

    out[f"{prefix}_start"] = start_val
    out[f"{prefix}_green_end"] = end_val
    out[f"{prefix}_change_to_green_end"] = delta
    out[f"{prefix}_rate_to_green_end_per_sec"] = delta / elapsed_sec if np.isfinite(delta) and elapsed_sec > 0 else np.nan


def add_window_summary_features(out: dict, win: pd.DataFrame, col: str, prefix: str) -> None:
    if col not in win.columns:
        out[f"{prefix}_mean_to_green_end"] = np.nan
        out[f"{prefix}_max_to_green_end"] = np.nan
        out[f"{prefix}_min_to_green_end"] = np.nan
        out[f"{prefix}_green_end"] = np.nan
        return

    values = pd.to_numeric(win[col], errors="coerce")
    finite_values = values[np.isfinite(values)]

    if finite_values.empty:
        out[f"{prefix}_mean_to_green_end"] = np.nan
        out[f"{prefix}_max_to_green_end"] = np.nan
        out[f"{prefix}_min_to_green_end"] = np.nan
    else:
        out[f"{prefix}_mean_to_green_end"] = float(finite_values.mean())
        out[f"{prefix}_max_to_green_end"] = float(finite_values.max())
        out[f"{prefix}_min_to_green_end"] = float(finite_values.min())

    out[f"{prefix}_green_end"] = col_at(win.iloc[-1], col)


def add_profile_summary_features(out: dict, prof_win: pd.DataFrame, pred_col: str, prefix: str, green_end_time: float) -> None:
    fields = [
        "q_green_end",
        "q_mean_to_green_end",
        "q_max_to_green_end",
        "q_min_to_green_end",
        "q_std_to_green_end",
        "q_range_to_green_end",
        "q_change_to_green_end",
        "q_slope_to_green_end_per_sec",
        "q_10sec_before_green_end",
        "q_change_last_10sec",
        "q_mean_last_10sec",
        "q_max_last_10sec",
    ]

    if prof_win.empty or pred_col not in prof_win.columns:
        for f in fields:
            out[f"{prefix}_{f}"] = np.nan
        return

    work = prof_win[["time_sec", pred_col]].copy()
    work["time_sec"] = pd.to_numeric(work["time_sec"], errors="coerce")
    work[pred_col] = pd.to_numeric(work[pred_col], errors="coerce")
    work = work.dropna(subset=["time_sec"]).sort_values("time_sec")

    q = work[pred_col].to_numpy(dtype=float)
    t = work["time_sec"].to_numpy(dtype=float)
    finite = np.isfinite(q) & np.isfinite(t)

    if not np.any(finite):
        for f in fields:
            out[f"{prefix}_{f}"] = np.nan
        return

    qf = q[finite]
    tf = t[finite]
    q_start = float(qf[0])
    q_end = float(qf[-1])
    elapsed = max(float(tf[-1] - tf[0]), 1.0)

    out[f"{prefix}_q_green_end"] = q_end
    out[f"{prefix}_q_mean_to_green_end"] = float(np.mean(qf))
    out[f"{prefix}_q_max_to_green_end"] = float(np.max(qf))
    out[f"{prefix}_q_min_to_green_end"] = float(np.min(qf))
    out[f"{prefix}_q_std_to_green_end"] = float(np.std(qf)) if len(qf) > 1 else 0.0
    out[f"{prefix}_q_range_to_green_end"] = float(np.max(qf) - np.min(qf))
    out[f"{prefix}_q_change_to_green_end"] = float(q_end - q_start)
    out[f"{prefix}_q_slope_to_green_end_per_sec"] = float((q_end - q_start) / elapsed)

    target_time = float(green_end_time) - 10.0
    before_mask = tf <= target_time + 1e-9
    if np.any(before_mask):
        q_10 = float(qf[np.where(before_mask)[0][-1]])
    else:
        q_10 = float(q_start)

    last10_mask = tf >= target_time - 1e-9
    q_last10 = qf[last10_mask] if np.any(last10_mask) else qf

    out[f"{prefix}_q_10sec_before_green_end"] = q_10
    out[f"{prefix}_q_change_last_10sec"] = float(q_end - q_10)
    out[f"{prefix}_q_mean_last_10sec"] = float(np.mean(q_last10))
    out[f"{prefix}_q_max_last_10sec"] = float(np.max(q_last10))


def build_direct_base_features_for_cycle_rate(label_row: pd.Series, rate_df: pd.DataFrame, cv_rate_pct: int) -> tuple[dict | None, dict | None]:
    cycle_start = safe_float(label_row["cycle_start_time_sec"])
    green_end = safe_float(label_row["green_end_time_sec"])

    if not np.isfinite(cycle_start) or not np.isfinite(green_end) or green_end < cycle_start:
        return None, {
            "cycle_uid": label_row.get("cycle_uid", ""),
            "run_id": int(label_row["run_id"]),
            "cv_rate_pct": int(cv_rate_pct),
            "reason": "invalid_cycle_or_green_end_time",
        }

    t = pd.to_numeric(rate_df["time_sec"], errors="coerce").to_numpy(dtype=float)
    mask = (t >= cycle_start - 1e-9) & (t <= green_end + 1e-9)
    win = rate_df.loc[mask].sort_values("time_sec").copy()

    if len(win) < MIN_ROWS_PER_CYCLE_WINDOW:
        return None, {
            "cycle_uid": label_row.get("cycle_uid", ""),
            "run_id": int(label_row["run_id"]),
            "cv_rate_pct": int(cv_rate_pct),
            "reason": "not_enough_timegrid_rows_before_green_end",
            "n_rows": int(len(win)),
        }

    start_row = win.iloc[0]
    end_row = win.iloc[-1]
    elapsed_sec = max(float(end_row["time_sec"] - start_row["time_sec"]), 1.0)
    phase_class = win["phase_class"].astype(str)

    green_start = safe_float(label_row.get("green_start_time_sec", np.nan))
    cycle_end = safe_float(label_row.get("cycle_end_time_sec", np.nan))
    cycle_duration = safe_float(label_row.get("cycle_duration_sec", np.nan))
    if not np.isfinite(cycle_duration) and np.isfinite(cycle_end):
        cycle_duration = cycle_end - cycle_start

    out = {
        "cycle_uid": str(label_row["cycle_uid"]),
        "run_id": int(label_row["run_id"]),
        "ml_split": str(label_row["ml_split"]),
        "cycle_number": int(label_row["cycle_number"]),
        "cv_rate_pct": int(cv_rate_pct),
        "cv_rate_group": cv_rate_group(cv_rate_pct),
        "traffic_condition": str(label_row["traffic_condition"]),
        "residual_queue_ft": safe_float(label_row["residual_queue_ft"]),
        "cycle_failure": int(label_row["cycle_failure"]),
        "failure_threshold_ft": safe_float(label_row.get("failure_threshold_ft", DEFAULT_FAILURE_THRESHOLD_FT)),
        "cycle_duration_sec": cycle_duration,
        "time_from_cycle_start_to_green_end_sec": green_end - cycle_start,
        "green_start_after_cycle_start_sec": green_start - cycle_start if np.isfinite(green_start) else np.nan,
        "green_duration_until_green_end_sec": green_end - green_start if np.isfinite(green_start) else np.nan,
        "green_end_to_cycle_end_sec": cycle_end - green_end if np.isfinite(cycle_end) else np.nan,
        "n_samples_to_green_end": int(len(win)),
        "red_sample_frac_to_green_end": float(phase_class.eq("red").mean()),
        "green_sample_frac_to_green_end": float(phase_class.eq("green").mean()),
        "yellow_sample_frac_to_green_end": float(phase_class.eq("yellow").mean()),
        "other_phase_sample_frac_to_green_end": float(phase_class.eq("other").mean()),
        "phase_elapsed_green_end_sec": col_at(end_row, "phase_elapsed_sec"),
        "cv_rate_pct_feature": float(cv_rate_pct),
    }

    for col, prefix in [
        ("A_count", "A_count"),
        ("D_count", "D_count"),
        ("V_count", "V_count"),
        ("B_count", "B_count"),
        ("n_queue_cumulative", "n_queue_cumulative"),
    ]:
        if col in win.columns:
            add_change_rate_features(out, start_row, end_row, col, prefix, elapsed_sec)

    if "A_count" in win.columns and "D_count" in win.columns:
        a_start = col_at(start_row, "A_count")
        a_end = col_at(end_row, "A_count")
        d_start = col_at(start_row, "D_count")
        d_end = col_at(end_row, "D_count")
        if all(np.isfinite(x) for x in [a_start, a_end, d_start, d_end]):
            arrivals = a_end - a_start
            departures = d_end - d_start
            net = arrivals - departures
        else:
            arrivals = departures = net = np.nan

        out["arrivals_to_green_end"] = arrivals
        out["departures_to_green_end"] = departures
        out["net_count_to_green_end"] = net
        out["arrival_departure_ratio_to_green_end"] = (
            arrivals / max(departures, 1.0) if np.isfinite(arrivals) and np.isfinite(departures) else np.nan
        )

    if "l_eff_fixed_ft" in win.columns:
        out["l_eff_fixed_ft_green_end"] = col_at(end_row, "l_eff_fixed_ft")

    for col, prefix in [
        ("inside_cv_segment", "inside_cv_segment"),
        ("prev_cv_anchor_q_ft", "prev_cv_anchor_q_ft"),
        ("next_cv_anchor_q_ft", "next_cv_anchor_q_ft"),
        ("time_since_prev_cv_sec", "time_since_prev_cv_sec"),
        ("time_to_next_cv_sec", "time_to_next_cv_sec"),
        ("cv_segment_duration_sec", "cv_segment_duration_sec"),
        ("cv_segment_frac", "cv_segment_frac"),
    ]:
        if col in win.columns:
            add_window_summary_features(out, win, col, prefix)

    cv_available = np.zeros(len(win), dtype=bool)
    for col in ["prev_cv_anchor_q_ft", "next_cv_anchor_q_ft"]:
        if col in win.columns:
            cv_available = cv_available | np.isfinite(pd.to_numeric(win[col], errors="coerce").to_numpy(dtype=float))

    out["cv_anchor_context_available_frac_to_green_end"] = float(np.mean(cv_available)) if len(cv_available) else np.nan
    out["cv_anchor_context_available_green_end"] = float(cv_available[-1]) if len(cv_available) else np.nan

    return out, None


def build_feature_tables(
    labels: pd.DataFrame,
    timegrid: pd.DataFrame,
    profile_pred: pd.DataFrame,
    catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    direct_rows = []
    hybrid_rows = []
    queue_baseline_rows = []
    skipped = []

    catalog_map = catalog.set_index("curve_id").to_dict(orient="index")
    curve_ids_needed = sorted(set(QUEUE_DERIVED_CURVES + FULL_HYBRID_PROFILE_CURVES + [PHYSICS_BASELINE_CURVE]))

    labels_by_run = {int(r): g.copy() for r, g in labels.groupby("run_id", sort=True)}
    profile_groups = {
        (int(r), int(rate)): g.sort_values("time_sec").reset_index(drop=True)
        for (r, rate), g in profile_pred.groupby(["run_id", "cv_rate_pct"], sort=True)
    }

    for (run_id, rate), g in timegrid.groupby(["run_id", "cv_rate_pct"], sort=True):
        run_id = int(run_id)
        rate = int(rate)
        if run_id not in labels_by_run:
            continue

        rate_df = g.sort_values("time_sec").reset_index(drop=True)
        prof_df = profile_groups.get((run_id, rate), pd.DataFrame())
        lab_run = labels_by_run[run_id]

        for _, label_row in lab_run.iterrows():
            base, skipped_row = build_direct_base_features_for_cycle_rate(label_row, rate_df, rate)
            if skipped_row is not None:
                skipped.append(skipped_row)
            if base is None:
                continue

            direct_rows.append(dict(base))

            cycle_start = safe_float(label_row["cycle_start_time_sec"])
            green_end = safe_float(label_row["green_end_time_sec"])
            if prof_df.empty:
                prof_win = pd.DataFrame()
            else:
                t = pd.to_numeric(prof_df["time_sec"], errors="coerce").to_numpy(dtype=float)
                mask = (t >= cycle_start - 1e-9) & (t <= green_end + 1e-9)
                prof_win = prof_df.loc[mask].sort_values("time_sec").copy()

            profile_summaries = {}
            for curve_id in curve_ids_needed:
                if curve_id not in catalog_map:
                    continue
                pred_col = str(catalog_map[curve_id].get("prediction_col", ""))
                prefix = f"curve_{curve_id}"
                add_profile_summary_features(profile_summaries, prof_win, pred_col, prefix, green_end)

            for curve_id in QUEUE_DERIVED_CURVES:
                if curve_id not in catalog_map:
                    continue
                pred_col = str(catalog_map[curve_id].get("prediction_col", ""))
                q_key = f"curve_{curve_id}_q_green_end"
                q_green = profile_summaries.get(q_key, np.nan)
                row = dict(base)
                row["model_id"] = f"queue_derived_threshold_{curve_id}"
                row["model_family_type"] = "queue_derived_baseline"
                row["decision_mode"] = "queue_profile_threshold"
                row["profile_curve_id"] = curve_id
                row["profile_curve_label"] = catalog_map[curve_id].get("curve_label", curve_id)
                row["profile_prediction_col"] = pred_col
                row["pred_residual_queue_ft"] = q_green
                row["pred_cycle_failure"] = int(q_green >= float(base["failure_threshold_ft"])) if np.isfinite(q_green) else np.nan
                row["pred_cycle_failure_prob"] = np.nan
                row["classifier_threshold"] = np.nan
                queue_baseline_rows.append(row)

            for curve_id in FULL_HYBRID_PROFILE_CURVES:
                if curve_id not in catalog_map:
                    continue
                row = dict(base)
                row["profile_curve_id"] = curve_id
                row["profile_curve_label"] = catalog_map[curve_id].get("curve_label", curve_id)
                row["profile_prediction_col"] = catalog_map[curve_id].get("prediction_col", "")

                # General reconstructed profile features for the selected curve.
                selected_prefix = f"curve_{curve_id}"
                for key, val in profile_summaries.items():
                    if key.startswith(selected_prefix + "_"):
                        clean_key = key.replace(selected_prefix + "_", "reconstructed_profile_", 1)
                        row[clean_key] = val

                # Physics baseline summaries are added separately so the hybrid
                # decision head can use explicit physics context even when the
                # selected reconstructed curve is GRU/XGBoost.
                physics_prefix = f"curve_{PHYSICS_BASELINE_CURVE}"
                for key, val in profile_summaries.items():
                    if key.startswith(physics_prefix + "_"):
                        clean_key = key.replace(physics_prefix + "_", "physics_baseline_profile_", 1)
                        row[clean_key] = val

                hybrid_rows.append(row)

    direct_features = pd.DataFrame(direct_rows)
    hybrid_features = pd.DataFrame(hybrid_rows)
    queue_baselines = pd.DataFrame(queue_baseline_rows)
    skipped_df = pd.DataFrame(skipped)

    if direct_features.empty:
        raise RuntimeError("No direct ML feature rows were generated.")
    if hybrid_features.empty:
        raise RuntimeError("No full hybrid feature rows were generated.")
    if queue_baselines.empty:
        raise RuntimeError("No queue-derived baseline rows were generated.")

    direct_features = direct_features.sort_values(["run_id", "cycle_number", "cv_rate_pct"]).reset_index(drop=True)
    hybrid_features = hybrid_features.sort_values(["profile_curve_id", "run_id", "cycle_number", "cv_rate_pct"]).reset_index(drop=True)
    queue_baselines = queue_baselines.sort_values(["model_id", "run_id", "cycle_number", "cv_rate_pct"]).reset_index(drop=True)

    return direct_features, hybrid_features, queue_baselines, skipped_df


# =============================================================================
# Feature selection and models
# =============================================================================

def select_feature_columns(features: pd.DataFrame, label: str) -> list[str]:
    candidate_cols = []
    for col in features.columns:
        if col in NON_FEATURE_COLS:
            continue
        if col.endswith("_uid"):
            continue
        if features[col].dtype.kind in "biufc":
            candidate_cols.append(col)

    clean_cols = []
    for col in candidate_cols:
        lower = col.lower()
        if any(term in lower for term in LEAKAGE_TERMS):
            continue
        clean_cols.append(col)

    if not clean_cols:
        raise ValueError(f"No usable feature columns were found for {label}.")

    return clean_cols


def make_regressor():
    if XGBOOST_AVAILABLE:
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE_BYTREE,
            reg_lambda=XGB_REG_LAMBDA,
            random_state=XGB_RANDOM_SEED,
            n_jobs=-1,
            tree_method="hist",
        )
        backend = "xgboost"
    else:
        model = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=XGB_RANDOM_SEED,
        )
        backend = "hist_gradient_boosting_fallback"

    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)]), backend


def make_classifier(y_train: np.ndarray):
    y_train = np.asarray(y_train, dtype=int)
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))

    if XGBOOST_AVAILABLE:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE_BYTREE,
            reg_lambda=XGB_REG_LAMBDA,
            random_state=XGB_RANDOM_SEED,
            n_jobs=-1,
            tree_method="hist",
            scale_pos_weight=float(n_neg / max(n_pos, 1)),
        )
        backend = "xgboost"
    else:
        model = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=XGB_RANDOM_SEED,
        )
        backend = "hist_gradient_boosting_fallback"

    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)]), backend


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def classification_scores_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
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
        "failure_accuracy": accuracy,
        "failure_precision": precision,
        "failure_recall": recall,
        "failure_specificity": specificity,
        "failure_f1": f1,
        "failure_false_alarm_rate": false_alarm_rate,
        "failure_miss_rate": miss_rate,
    }


def choose_classifier_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(prob)
    y_true = y_true[mask].astype(int)
    prob = prob[mask]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.50

    best_threshold = 0.50
    best_tuple = (-np.inf, -np.inf, -np.inf)

    for threshold in CLASSIFIER_THRESHOLD_GRID:
        pred = (prob >= float(threshold)).astype(int)
        tp, fp, fn, tn = confusion_counts(y_true, pred)
        scores = classification_scores_from_counts(tp, fp, fn, tn)
        f1 = scores["failure_f1"]
        recall = scores["failure_recall"]
        accuracy = scores["failure_accuracy"]
        ranking_tuple = (
            f1 if np.isfinite(f1) else -np.inf,
            recall if np.isfinite(recall) else -np.inf,
            accuracy if np.isfinite(accuracy) else -np.inf,
        )
        if ranking_tuple > best_tuple:
            best_tuple = ranking_tuple
            best_threshold = float(threshold)

    return best_threshold


def fit_dual_head_models(
    features: pd.DataFrame,
    feature_cols: list[str],
    model_prefix: str,
    model_family_type: str,
    decision_mode_label: str,
    profile_curve_id: str | None,
    profile_curve_label: str | None,
    failure_threshold_ft: float,
) -> tuple[pd.DataFrame, list[dict], list[dict], dict]:
    train = features[features["ml_split"].astype(str).eq("train")].copy()
    validation = features[features["ml_split"].astype(str).eq("validation")].copy()

    if train.empty:
        raise ValueError(f"No training rows found for {model_prefix}.")

    X_train = train[feature_cols].copy()
    y_reg_train = train[TARGET_RESIDUAL_COL].to_numpy(dtype=float)
    y_cls_train = train[TARGET_FAILURE_COL].to_numpy(dtype=int)

    regressor, reg_backend = make_regressor()
    regressor.fit(X_train, y_reg_train)

    classifier = None
    cls_backend = "skipped_single_class_train"
    classifier_threshold = np.nan

    if len(np.unique(y_cls_train)) >= 2:
        classifier, cls_backend = make_classifier(y_cls_train)
        classifier.fit(X_train, y_cls_train)
        if not validation.empty:
            val_prob = classifier.predict_proba(validation[feature_cols].copy())[:, 1]
            classifier_threshold = choose_classifier_threshold(validation[TARGET_FAILURE_COL].to_numpy(dtype=int), val_prob)
        else:
            classifier_threshold = 0.50

    pred_source = features.copy()
    pred_reg = regressor.predict(pred_source[feature_cols].copy()).astype(float)
    pred_reg = np.maximum(pred_reg, 0.0)

    if classifier is not None:
        prob = classifier.predict_proba(pred_source[feature_cols].copy())[:, 1].astype(float)
        pred_cls = (prob >= float(classifier_threshold)).astype(int)
    else:
        prob = np.full(len(pred_source), np.nan)
        pred_cls = np.full(len(pred_source), np.nan)

    common_cols = [
        "cycle_uid",
        "run_id",
        "ml_split",
        "cycle_number",
        "cv_rate_pct",
        "cv_rate_group",
        "traffic_condition",
        "residual_queue_ft",
        "cycle_failure",
        "failure_threshold_ft",
    ]
    common_cols = [c for c in common_cols if c in pred_source.columns]

    # Main full/direct model row: residual queue from the regression head and
    # cycle-failure label from the probability head. This is the row that best
    # matches the Overleaf formulation: direct residual + direct probability.
    dual = pred_source[common_cols].copy()
    dual["model_id"] = f"{model_prefix}_dual_head"
    dual["model_family_type"] = model_family_type
    dual["decision_mode"] = decision_mode_label
    dual["profile_curve_id"] = profile_curve_id or ""
    dual["profile_curve_label"] = profile_curve_label or ""
    dual["pred_residual_queue_ft"] = pred_reg
    dual["pred_cycle_failure"] = pred_cls
    dual["pred_cycle_failure_prob"] = prob
    dual["classifier_threshold"] = classifier_threshold

    # Auxiliary row: residual regression plus fixed residual-queue threshold.
    # This allows direct comparison against the old queue-derived thresholding
    # logic while still using a learned residual-queue head.
    reg_thresh = pred_source[common_cols].copy()
    reg_thresh["model_id"] = f"{model_prefix}_regression_threshold"
    reg_thresh["model_family_type"] = model_family_type
    reg_thresh["decision_mode"] = "direct_residual_regression_threshold"
    reg_thresh["profile_curve_id"] = profile_curve_id or ""
    reg_thresh["profile_curve_label"] = profile_curve_label or ""
    reg_thresh["pred_residual_queue_ft"] = pred_reg
    reg_thresh["pred_cycle_failure"] = (pred_reg >= float(failure_threshold_ft)).astype(int)
    reg_thresh["pred_cycle_failure_prob"] = np.nan
    reg_thresh["classifier_threshold"] = np.nan

    long = pd.concat([dual, reg_thresh], ignore_index=True)
    long = long.rename(columns={"residual_queue_ft": "gt_residual_queue_ft", "cycle_failure": "gt_cycle_failure"})
    long["failure_threshold_ft"] = float(failure_threshold_ft)

    training_rows = [
        {
            "model_id": f"{model_prefix}_dual_head",
            "model_family_type": model_family_type,
            "decision_mode": decision_mode_label,
            "profile_curve_id": profile_curve_id or "",
            "backend_regressor": reg_backend,
            "backend_classifier": cls_backend,
            "n_train_rows": int(len(train)),
            "n_validation_rows": int(len(validation)),
            "n_feature_cols": int(len(feature_cols)),
            "failure_threshold_ft": float(failure_threshold_ft),
            "classifier_threshold": classifier_threshold,
            "status": "trained" if classifier is not None else "classifier_skipped",
        }
    ]

    threshold_rows = [
        {
            "model_id": f"{model_prefix}_dual_head",
            "model_family_type": model_family_type,
            "decision_mode": decision_mode_label,
            "profile_curve_id": profile_curve_id or "",
            "classifier_threshold": classifier_threshold,
            "selected_on_split": "validation",
            "threshold_grid": ",".join([f"{x:.2f}" for x in CLASSIFIER_THRESHOLD_GRID]),
        }
    ]

    model_objects = {
        "regressor": regressor,
        "classifier": classifier,
        "reg_backend": reg_backend,
        "cls_backend": cls_backend,
        "feature_cols": feature_cols,
        "classifier_threshold": classifier_threshold,
    }

    return long, training_rows, threshold_rows, model_objects


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics_for_group(g: pd.DataFrame) -> dict:
    n_rows = int(len(g))

    y_resid, yp_resid = finite_arrays(g["gt_residual_queue_ft"], g["pred_residual_queue_ft"])
    n_valid_residual = int(len(y_resid))

    if n_valid_residual:
        err = yp_resid - y_resid
        residual_mae_ft = float(mean_absolute_error(y_resid, yp_resid))
        residual_rmse_ft = float(math.sqrt(mean_squared_error(y_resid, yp_resid)))
        residual_bias_ft = float(np.mean(err))
        residual_max_abs_error_ft = float(np.max(np.abs(err)))
    else:
        residual_mae_ft = np.nan
        residual_rmse_ft = np.nan
        residual_bias_ft = np.nan
        residual_max_abs_error_ft = np.nan

    gt_fail = pd.to_numeric(g["gt_cycle_failure"], errors="coerce")
    pred_fail = pd.to_numeric(g["pred_cycle_failure"], errors="coerce")
    mask = np.isfinite(gt_fail) & np.isfinite(pred_fail)
    gt_arr = gt_fail[mask].astype(int).to_numpy()
    pred_arr = pred_fail[mask].astype(int).to_numpy()
    n_valid_decision = int(len(gt_arr))

    if n_valid_decision:
        tp, fp, fn, tn = confusion_counts(gt_arr, pred_arr)
        scores = classification_scores_from_counts(tp, fp, fn, tn)
        gt_failure_rate_pct = float(100.0 * np.mean(gt_arr))
        pred_failure_rate_pct = float(100.0 * np.mean(pred_arr))
    else:
        tp = fp = fn = tn = 0
        scores = {
            "failure_accuracy": np.nan,
            "failure_precision": np.nan,
            "failure_recall": np.nan,
            "failure_specificity": np.nan,
            "failure_f1": np.nan,
            "failure_false_alarm_rate": np.nan,
            "failure_miss_rate": np.nan,
        }
        gt_failure_rate_pct = np.nan
        pred_failure_rate_pct = np.nan

    return {
        "n_rows": n_rows,
        "n_valid_residual": n_valid_residual,
        "n_valid_decision": n_valid_decision,
        "valid_decision_pct": float(100.0 * n_valid_decision / n_rows) if n_rows else np.nan,
        "residual_mae_ft": residual_mae_ft,
        "residual_rmse_ft": residual_rmse_ft,
        "residual_bias_ft": residual_bias_ft,
        "residual_max_abs_error_ft": residual_max_abs_error_ft,
        "gt_failure_rate_pct": gt_failure_rate_pct,
        "pred_failure_rate_pct": pred_failure_rate_pct,
        **scores,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def group_metrics(long: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in long.groupby(group_cols, dropna=False, sort=True):
        if len(group_cols) == 1:
            key = (key,)
        row = {col: key[i] for i, col in enumerate(group_cols)}
        row.update(compute_metrics_for_group(g))
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(group_cols).reset_index(drop=True)
    return out


def save_selected_summaries(metrics_split_model: pd.DataFrame) -> None:
    validation = metrics_split_model[metrics_split_model["ml_split"].astype(str).eq("validation")].copy()
    test = metrics_split_model[metrics_split_model["ml_split"].astype(str).eq("test")].copy()

    for df in [validation, test]:
        df["sort_rmse"] = df["residual_rmse_ft"].fillna(1e12)
        df["sort_f1"] = df["failure_f1"].fillna(-1.0)

    validation = validation.sort_values(["sort_rmse", "sort_f1"], ascending=[True, False]).drop(columns=["sort_rmse", "sort_f1"])
    test = test.sort_values(["sort_rmse", "sort_f1"], ascending=[True, False]).drop(columns=["sort_rmse", "sort_f1"])

    validation.to_csv(OUT_DIR / "full_hybrid_decision_metrics_validation.csv", index=False)
    test.to_csv(OUT_DIR / "full_hybrid_decision_metrics_test.csv", index=False)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("Training full hybrid decision models")
    print("=" * 96)
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Decision labels    : {DECISION_LABEL_FILE}")
    print(f"Time-grid features : {TIMEGRID_FEATURE_FILE}")
    print(f"Method predictions : {METHOD_FAMILY_PRED_FILE}")
    print(f"Curve catalog      : {CURVE_CATALOG_FILE}")
    print(f"Output dir         : {OUT_DIR}")
    print(f"CV rates           : {CV_RATES_PCT}")
    print(f"XGBoost available  : {XGBOOST_AVAILABLE}")
    print("=" * 96)

    labels, failure_threshold_ft = load_decision_labels()
    timegrid = load_timegrid_features()
    catalog = load_curve_catalog()
    profile_pred, selected_catalog = load_method_family_predictions(catalog)

    print(f"Loaded labels          : {len(labels):,} cycles")
    print(f"Loaded time-grid rows  : {len(timegrid):,}")
    print(f"Loaded profile rows    : {len(profile_pred):,}")
    print(f"Selected curves        : {len(selected_catalog):,}")
    print(f"Failure threshold      : {failure_threshold_ft:.1f} ft")

    print("\nLabels by split:")
    print(labels.groupby("ml_split").size().reset_index(name="n_cycles").to_string(index=False))

    print("\nBuilding decision feature tables...")
    direct_features, hybrid_features, queue_baselines, skipped = build_feature_tables(
        labels=labels,
        timegrid=timegrid,
        profile_pred=profile_pred,
        catalog=selected_catalog,
    )

    direct_features.to_csv(OUT_DIR / "direct_ml_decision_features_allrates.csv", index=False)
    hybrid_features.to_csv(OUT_DIR / "full_hybrid_decision_features_allrates.csv", index=False)
    skipped.to_csv(OUT_DIR / "skipped_cycle_rate_rows.csv", index=False)

    print(f"Direct feature rows     : {len(direct_features):,}")
    print(f"Full hybrid feature rows: {len(hybrid_features):,}")
    print(f"Queue baseline rows     : {len(queue_baselines):,}")
    print(f"Skipped rows            : {len(skipped):,}")

    long_parts = []
    training_rows = []
    threshold_rows = []
    feature_rows = []

    # Queue-derived baseline rows are already predictions.
    queue_long = queue_baselines.rename(
        columns={
            "residual_queue_ft": "gt_residual_queue_ft",
            "cycle_failure": "gt_cycle_failure",
        }
    )
    long_parts.append(queue_long)
    queue_long.to_csv(OUT_DIR / "queue_derived_decision_predictions_allrates.csv", index=False)

    # Direct ML-only baseline trained in this same script for side-by-side comparison.
    direct_feature_cols = select_feature_columns(direct_features, "direct ML-only decision baseline")
    direct_long, direct_train_rows, direct_threshold_rows, direct_models = fit_dual_head_models(
        features=direct_features,
        feature_cols=direct_feature_cols,
        model_prefix="direct_ml_decision_xgb" if XGBOOST_AVAILABLE else "direct_ml_decision_hgb",
        model_family_type="direct_ml_only_decision_baseline",
        decision_mode_label="direct_ml_dual_head",
        profile_curve_id=None,
        profile_curve_label=None,
        failure_threshold_ft=failure_threshold_ft,
    )
    long_parts.append(direct_long)
    training_rows.extend(direct_train_rows)
    threshold_rows.extend(direct_threshold_rows)
    for c in direct_feature_cols:
        feature_rows.append({"model_scope": "direct_ml_only", "profile_curve_id": "", "feature_name": c, "used_as_model_input": True})

    joblib.dump(direct_models["regressor"], MODEL_DIR / "direct_ml_decision_regressor.joblib")
    if direct_models["classifier"] is not None:
        joblib.dump(direct_models["classifier"], MODEL_DIR / "direct_ml_decision_classifier.joblib")

    # Full hybrid decision heads. Train one pair of heads for each selected
    # reconstructed-profile source.
    for curve_id in FULL_HYBRID_PROFILE_CURVES:
        subset = hybrid_features[hybrid_features["profile_curve_id"].astype(str).eq(curve_id)].copy()
        if subset.empty:
            print(f"[WARN] Skipping {curve_id}: no hybrid feature rows found.")
            continue

        profile_label = str(subset["profile_curve_label"].iloc[0]) if "profile_curve_label" in subset.columns else curve_id
        hybrid_feature_cols = select_feature_columns(subset, f"full hybrid decision model from {curve_id}")
        model_prefix = f"full_hybrid_decision_xgb_from_{curve_id}" if XGBOOST_AVAILABLE else f"full_hybrid_decision_hgb_from_{curve_id}"

        hybrid_long, hybrid_train_rows, hybrid_threshold_rows, hybrid_models = fit_dual_head_models(
            features=subset,
            feature_cols=hybrid_feature_cols,
            model_prefix=model_prefix,
            model_family_type="full_hybrid_decision_model",
            decision_mode_label="direct_residual_and_failure_probability",
            profile_curve_id=curve_id,
            profile_curve_label=profile_label,
            failure_threshold_ft=failure_threshold_ft,
        )
        long_parts.append(hybrid_long)
        training_rows.extend(hybrid_train_rows)
        threshold_rows.extend(hybrid_threshold_rows)

        for c in hybrid_feature_cols:
            feature_rows.append({"model_scope": "full_hybrid", "profile_curve_id": curve_id, "feature_name": c, "used_as_model_input": True})

        safe_curve = slugify(curve_id)
        joblib.dump(hybrid_models["regressor"], MODEL_DIR / f"full_hybrid_decision_regressor_from_{safe_curve}.joblib")
        if hybrid_models["classifier"] is not None:
            joblib.dump(hybrid_models["classifier"], MODEL_DIR / f"full_hybrid_decision_classifier_from_{safe_curve}.joblib")

    training_summary = pd.DataFrame(training_rows)
    threshold_summary = pd.DataFrame(threshold_rows)
    feature_summary = pd.DataFrame(feature_rows)

    training_summary.to_csv(OUT_DIR / "full_hybrid_decision_training_summary.csv", index=False)
    threshold_summary.to_csv(OUT_DIR / "selected_thresholds_validation.csv", index=False)
    feature_summary.to_csv(OUT_DIR / "full_hybrid_decision_feature_columns_used.csv", index=False)

    all_predictions = pd.concat(long_parts, ignore_index=True)
    all_predictions.to_csv(OUT_DIR / "full_hybrid_decision_predictions_allrates.csv", index=False)

    metrics_by_split_rate_model = group_metrics(
        all_predictions,
        [
            "ml_split",
            "cv_rate_pct",
            "cv_rate_group",
            "model_id",
            "model_family_type",
            "decision_mode",
            "profile_curve_id",
            "profile_curve_label",
        ],
    )

    metrics_by_split_model = group_metrics(
        all_predictions,
        [
            "ml_split",
            "model_id",
            "model_family_type",
            "decision_mode",
            "profile_curve_id",
            "profile_curve_label",
        ],
    )

    metrics_by_split_rate_condition_model = group_metrics(
        all_predictions,
        [
            "ml_split",
            "cv_rate_pct",
            "cv_rate_group",
            "traffic_condition",
            "model_id",
            "model_family_type",
            "decision_mode",
            "profile_curve_id",
            "profile_curve_label",
        ],
    )

    metrics_by_split_rate_model.to_csv(OUT_DIR / "full_hybrid_decision_metrics_by_split_rate_model.csv", index=False)
    metrics_by_split_model.to_csv(OUT_DIR / "full_hybrid_decision_metrics_by_split_model.csv", index=False)
    metrics_by_split_rate_condition_model.to_csv(OUT_DIR / "full_hybrid_decision_metrics_by_split_rate_condition_model.csv", index=False)

    # Requested by-rate output: test split only.
    metrics_by_rate = metrics_by_split_rate_model[metrics_by_split_rate_model["ml_split"].astype(str).eq("test")].copy()
    metrics_by_rate.to_csv(OUT_DIR / "full_hybrid_decision_metrics_by_rate.csv", index=False)

    save_selected_summaries(metrics_by_split_model)

    print("\nSaved:")
    for filename in [
        "direct_ml_decision_features_allrates.csv",
        "full_hybrid_decision_features_allrates.csv",
        "queue_derived_decision_predictions_allrates.csv",
        "full_hybrid_decision_predictions_allrates.csv",
        "full_hybrid_decision_metrics_by_split_rate_model.csv",
        "full_hybrid_decision_metrics_by_split_model.csv",
        "full_hybrid_decision_metrics_validation.csv",
        "full_hybrid_decision_metrics_test.csv",
        "full_hybrid_decision_metrics_by_rate.csv",
        "selected_thresholds_validation.csv",
        "full_hybrid_decision_feature_columns_used.csv",
        "full_hybrid_decision_training_summary.csv",
    ]:
        print(f"  {OUT_DIR / filename}")

    print("\nTraining summary:")
    if not training_summary.empty:
        print(training_summary.round(3).to_string(index=False))

    display_cols = [
        "model_id",
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

    validation = metrics_by_split_model[metrics_by_split_model["ml_split"].astype(str).eq("validation")].copy()
    validation["sort_rmse"] = validation["residual_rmse_ft"].fillna(1e12)
    validation["sort_f1"] = validation["failure_f1"].fillna(-1.0)
    validation = validation.sort_values(["sort_rmse", "sort_f1"], ascending=[True, False]).drop(columns=["sort_rmse", "sort_f1"])

    test = metrics_by_split_model[metrics_by_split_model["ml_split"].astype(str).eq("test")].copy()
    test["sort_rmse"] = test["residual_rmse_ft"].fillna(1e12)
    test["sort_f1"] = test["failure_f1"].fillna(-1.0)
    test = test.sort_values(["sort_rmse", "sort_f1"], ascending=[True, False]).drop(columns=["sort_rmse", "sort_f1"])

    display_cols = [c for c in display_cols if c in test.columns]

    print("\nValidation metrics:")
    print(validation[display_cols].round(3).to_string(index=False))

    print("\nTest metrics:")
    print(test[display_cols].round(3).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
