"""
Train direct cycle-level decision baselines without reconstructing queue profiles.

Purpose
-------
This script implements the second decision-oriented baseline requested for the
queue reconstruction paper:

    cycle-level features -> direct residual queue / cycle-failure prediction

Unlike evaluate_decision_from_profiles.py, this script does not sample a full
reconstructed queue profile at the end of green. It builds one feature vector per
cycle and CV penetration rate using signal/count/CV/physics context available up
to the green-end timestamp, then trains:

1) Direct residual-queue regression
2) Direct cycle-failure classification

Inputs
------
output/intermediate_csv/decision_labels/decision_labels_cycle_level.csv
output/intermediate_csv/cv_features/timegrid_features_allruns_allrates.csv

Outputs
-------
output/intermediate_csv/direct_decision_baselines/
    direct_decision_cycle_features_allrates.csv
    direct_decision_predictions_allruns_allrates.csv
    direct_decision_metrics_by_split_rate_model.csv
    direct_decision_metrics_by_split_model.csv
    direct_decision_metrics_by_split_rate_condition_model.csv
    direct_decision_metrics_validation.csv
    direct_decision_metrics_test.csv
    direct_decision_feature_columns_used.csv
    direct_decision_training_summary.csv
    direct_decision_skipped_cycles.csv
    trained_models/
        direct_residual_queue_regressor.joblib
        direct_cycle_failure_classifier.joblib
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

import math
from pathlib import Path
from typing import Iterable

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

FEATURE_FILE = (
    PROJECT_ROOT
    / "output"
    / "intermediate_csv"
    / "cv_features"
    / "timegrid_features_allruns_allrates.csv"
)

DECISION_LABEL_FILE = (
    PROJECT_ROOT
    / "output"
    / "intermediate_csv"
    / "decision_labels"
    / "decision_labels_cycle_level.csv"
)

OUT_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "direct_decision_baselines"
MODEL_DIR = OUT_DIR / "trained_models"

DEFAULT_FAILURE_THRESHOLD_FT = 25.0
MIN_ROWS_PER_CYCLE_WINDOW = 2

# Candidate probability thresholds for direct classifier. The best value is
# selected on the validation split only.
CLASSIFIER_THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


# =============================================================================
# Column configuration
# =============================================================================

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
    # Physics/cumulative-count context. These are cycle-level predictors only;
    # the script does not reconstruct a queue profile.
    "V_count",
    "B_count",
    "n_queue_cumulative",
    "l_eff_fixed_ft",

    # CV context features from build_cv_features.py.
    "inside_cv_segment",
    "prev_cv_anchor_q_ft",
    "next_cv_anchor_q_ft",
    "time_since_prev_cv_sec",
    "time_to_next_cv_sec",
    "cv_segment_duration_sec",
    "cv_segment_frac",
]

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
    "failure_threshold_ft",
    "traffic_condition",
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
}


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


def classify_phase_state(value) -> str:
    p = str(value).strip().lower()
    if p in {"red", "r"} or p.startswith("red"):
        return "red"
    if p in {"yellow", "y", "amber"} or p.startswith("yellow") or "amber" in p:
        return "yellow"
    if p in {"green", "g"} or p.startswith("green") or "green" in p:
        return "green"
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


# =============================================================================
# Load inputs
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

    threshold_values = pd.to_numeric(labels["failure_threshold_ft"], errors="coerce").dropna()
    if threshold_values.empty:
        failure_threshold_ft = float(DEFAULT_FAILURE_THRESHOLD_FT)
    else:
        failure_threshold_ft = float(threshold_values.iloc[0])

    model_runs = sorted(set(TRAIN_RUN_IDS + VALIDATION_RUN_IDS + TEST_RUN_IDS))
    labels = labels[labels["run_id"].isin(model_runs)].copy()
    labels = labels.sort_values(["run_id", "cycle_number"]).reset_index(drop=True)

    return labels, failure_threshold_ft


def load_timegrid_features() -> pd.DataFrame:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find feature table:\n{FEATURE_FILE}\n"
            "Run src/build_cv_features.py first."
        )

    header = pd.read_csv(FEATURE_FILE, nrows=0)
    available = set(header.columns)

    require_columns(header, TIMEGRID_REQUIRED_COLS, "timegrid feature table")

    usecols = TIMEGRID_REQUIRED_COLS + [c for c in TIMEGRID_OPTIONAL_COLS if c in available]
    usecols = list(dict.fromkeys(usecols))

    df = pd.read_csv(FEATURE_FILE, usecols=usecols)

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


# =============================================================================
# Cycle-level feature construction
# =============================================================================

def col_at(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return np.nan
    return safe_float(row[col])


def add_change_rate_features(
    out: dict,
    start_row: pd.Series,
    end_row: pd.Series,
    col: str,
    prefix: str,
    elapsed_sec: float,
) -> None:
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


def build_features_for_cycle_rate(
    label_row: pd.Series,
    rate_df: pd.DataFrame,
    cv_rate_pct: int,
) -> tuple[dict | None, dict | None]:
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
    n_samples = int(len(win))

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

        # Known signal/cycle timing context.
        "cycle_duration_sec": cycle_duration,
        "time_from_cycle_start_to_green_end_sec": green_end - cycle_start,
        "green_start_after_cycle_start_sec": green_start - cycle_start if np.isfinite(green_start) else np.nan,
        "green_duration_until_green_end_sec": green_end - green_start if np.isfinite(green_start) else np.nan,
        "green_end_to_cycle_end_sec": cycle_end - green_end if np.isfinite(cycle_end) else np.nan,

        # Time-grid coverage and phase composition up to green end.
        "n_samples_to_green_end": n_samples,
        "red_sample_frac_to_green_end": float(phase_class.eq("red").mean()),
        "green_sample_frac_to_green_end": float(phase_class.eq("green").mean()),
        "yellow_sample_frac_to_green_end": float(phase_class.eq("yellow").mean()),
        "other_phase_sample_frac_to_green_end": float(phase_class.eq("other").mean()),
        "phase_elapsed_green_end_sec": col_at(end_row, "phase_elapsed_sec"),

        # Let the direct model know CV availability level, but do not expose run ID.
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

    # CV context at and before the green-end time. These features do not include
    # the GT residual label or any reconstructed profile prediction.
    cv_cols = [
        ("inside_cv_segment", "inside_cv_segment"),
        ("prev_cv_anchor_q_ft", "prev_cv_anchor_q_ft"),
        ("next_cv_anchor_q_ft", "next_cv_anchor_q_ft"),
        ("time_since_prev_cv_sec", "time_since_prev_cv_sec"),
        ("time_to_next_cv_sec", "time_to_next_cv_sec"),
        ("cv_segment_duration_sec", "cv_segment_duration_sec"),
        ("cv_segment_frac", "cv_segment_frac"),
    ]

    for col, prefix in cv_cols:
        if col in win.columns:
            add_window_summary_features(out, win, col, prefix)

    cv_available = np.zeros(len(win), dtype=bool)
    for col in ["prev_cv_anchor_q_ft", "next_cv_anchor_q_ft"]:
        if col in win.columns:
            cv_available = cv_available | np.isfinite(pd.to_numeric(win[col], errors="coerce").to_numpy(dtype=float))

    out["cv_anchor_context_available_frac_to_green_end"] = float(np.mean(cv_available)) if len(cv_available) else np.nan
    out["cv_anchor_context_available_green_end"] = float(cv_available[-1]) if len(cv_available) else np.nan

    return out, None


def build_cycle_feature_table(labels: pd.DataFrame, timegrid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    skipped = []

    labels_by_run = {int(r): g.copy() for r, g in labels.groupby("run_id", sort=True)}

    for (run_id, rate), g in timegrid.groupby(["run_id", "cv_rate_pct"], sort=True):
        run_id = int(run_id)
        rate = int(rate)
        if run_id not in labels_by_run:
            continue

        rate_df = g.sort_values("time_sec").reset_index(drop=True)
        lab_run = labels_by_run[run_id]

        for _, label_row in lab_run.iterrows():
            feature_row, skipped_row = build_features_for_cycle_rate(label_row, rate_df, rate)
            if feature_row is not None:
                rows.append(feature_row)
            if skipped_row is not None:
                skipped.append(skipped_row)

    features = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)

    if features.empty:
        raise RuntimeError("No direct decision cycle features were generated.")

    features = features.sort_values(["run_id", "cycle_number", "cv_rate_pct"]).reset_index(drop=True)
    return features, skipped_df


# =============================================================================
# Model training
# =============================================================================

def select_feature_columns(features: pd.DataFrame) -> list[str]:
    candidate_cols = []
    for col in features.columns:
        if col in NON_FEATURE_COLS:
            continue
        if col.endswith("_uid"):
            continue
        if features[col].dtype.kind in "biufc":
            candidate_cols.append(col)

    # Remove any accidental leakage columns by name pattern.
    leakage_terms = ["gt_", "q_gt", "peak_queue", "residual_queue", "cycle_failure"]
    clean_cols = []
    for col in candidate_cols:
        lower = col.lower()
        if any(term in lower for term in leakage_terms):
            continue
        clean_cols.append(col)

    if not clean_cols:
        raise ValueError("No usable direct decision feature columns were found.")

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

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )
    return pipe, backend


def make_classifier(y_train: np.ndarray):
    y_train = np.asarray(y_train, dtype=int)
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))

    if XGBOOST_AVAILABLE:
        scale_pos_weight = float(n_neg / max(n_pos, 1))
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
            scale_pos_weight=scale_pos_weight,
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

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )
    return pipe, backend


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


def train_direct_models(features: pd.DataFrame, feature_cols: list[str], failure_threshold_ft: float):
    train = features[features["ml_split"].astype(str).eq("train")].copy()
    validation = features[features["ml_split"].astype(str).eq("validation")].copy()

    if train.empty:
        raise ValueError("No training rows found for direct decision baseline.")

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
            classifier_threshold = choose_classifier_threshold(
                validation[TARGET_FAILURE_COL].to_numpy(dtype=int),
                val_prob,
            )
        else:
            classifier_threshold = 0.50

    pred = features.copy()

    pred_reg = regressor.predict(pred[feature_cols].copy()).astype(float)
    pred_reg = np.maximum(pred_reg, 0.0)
    pred["pred_residual_queue_ft_direct_reg"] = pred_reg
    pred["pred_cycle_failure_from_direct_reg"] = (
        pred["pred_residual_queue_ft_direct_reg"].to_numpy(dtype=float) >= float(failure_threshold_ft)
    ).astype(int)

    if classifier is not None:
        prob = classifier.predict_proba(pred[feature_cols].copy())[:, 1].astype(float)
        pred["pred_cycle_failure_prob_direct_cls"] = prob
        pred["classifier_threshold"] = float(classifier_threshold)
        pred["pred_cycle_failure_direct_cls"] = (prob >= float(classifier_threshold)).astype(int)
    else:
        pred["pred_cycle_failure_prob_direct_cls"] = np.nan
        pred["classifier_threshold"] = np.nan
        pred["pred_cycle_failure_direct_cls"] = np.nan

    training_summary = pd.DataFrame(
        [
            {
                "model_id": "direct_cycle_xgb_regression_threshold" if reg_backend == "xgboost" else "direct_cycle_hgb_regression_threshold",
                "model_type": "residual_queue_regression_plus_threshold",
                "backend": reg_backend,
                "target": TARGET_RESIDUAL_COL,
                "n_train_rows": int(len(train)),
                "n_validation_rows": int(len(validation)),
                "failure_threshold_ft": float(failure_threshold_ft),
                "classifier_threshold": np.nan,
                "status": "trained",
                "model_file": str(MODEL_DIR / "direct_residual_queue_regressor.joblib"),
            },
            {
                "model_id": "direct_cycle_xgb_classifier_probability" if cls_backend == "xgboost" else "direct_cycle_hgb_classifier_probability",
                "model_type": "cycle_failure_classifier_probability",
                "backend": cls_backend,
                "target": TARGET_FAILURE_COL,
                "n_train_rows": int(len(train)),
                "n_validation_rows": int(len(validation)),
                "failure_threshold_ft": float(failure_threshold_ft),
                "classifier_threshold": classifier_threshold,
                "status": "trained" if classifier is not None else "skipped",
                "model_file": str(MODEL_DIR / "direct_cycle_failure_classifier.joblib") if classifier is not None else "",
            },
        ]
    )

    return pred, regressor, classifier, training_summary


# =============================================================================
# Metrics
# =============================================================================

def make_long_prediction_table(pred: pd.DataFrame, failure_threshold_ft: float) -> pd.DataFrame:
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
    common_cols = [c for c in common_cols if c in pred.columns]

    reg = pred[common_cols].copy()
    reg["model_id"] = "direct_cycle_xgb_regression_threshold" if XGBOOST_AVAILABLE else "direct_cycle_hgb_regression_threshold"
    reg["decision_baseline"] = "Direct cycle-level ML"
    reg["decision_mode"] = "regression_threshold"
    reg["pred_residual_queue_ft"] = pred["pred_residual_queue_ft_direct_reg"].to_numpy(dtype=float)
    reg["pred_cycle_failure"] = pred["pred_cycle_failure_from_direct_reg"].to_numpy(dtype=int)
    reg["pred_cycle_failure_prob"] = np.nan
    reg["classifier_threshold"] = np.nan

    cls = pred[common_cols].copy()
    cls["model_id"] = "direct_cycle_xgb_classifier_probability" if XGBOOST_AVAILABLE else "direct_cycle_hgb_classifier_probability"
    cls["decision_baseline"] = "Direct cycle-level ML"
    cls["decision_mode"] = "classifier_probability"
    cls["pred_residual_queue_ft"] = np.nan
    cls["pred_cycle_failure"] = pred["pred_cycle_failure_direct_cls"]
    cls["pred_cycle_failure_prob"] = pred["pred_cycle_failure_prob_direct_cls"]
    cls["classifier_threshold"] = pred["classifier_threshold"]

    long = pd.concat([reg, cls], ignore_index=True)
    long = long.rename(
        columns={
            "residual_queue_ft": "gt_residual_queue_ft",
            "cycle_failure": "gt_cycle_failure",
        }
    )
    long["failure_threshold_ft"] = float(failure_threshold_ft)
    return long


def compute_metrics_for_group(g: pd.DataFrame) -> dict:
    n_rows = int(len(g))

    # Residual-queue metrics where a residual prediction exists.
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

    # Classification metrics where a decision prediction exists.
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

    # Put residual-regression rows first by RMSE, then classifier-only rows by F1.
    validation["sort_rmse"] = validation["residual_rmse_ft"].fillna(1e12)
    validation["sort_f1"] = validation["failure_f1"].fillna(-1.0)
    validation = validation.sort_values(["sort_rmse", "sort_f1"], ascending=[True, False]).drop(columns=["sort_rmse", "sort_f1"])

    test["sort_rmse"] = test["residual_rmse_ft"].fillna(1e12)
    test["sort_f1"] = test["failure_f1"].fillna(-1.0)
    test = test.sort_values(["sort_rmse", "sort_f1"], ascending=[True, False]).drop(columns=["sort_rmse", "sort_f1"])

    validation.to_csv(OUT_DIR / "direct_decision_metrics_validation.csv", index=False)
    test.to_csv(OUT_DIR / "direct_decision_metrics_test.csv", index=False)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("Training direct cycle-level decision baselines")
    print("=" * 96)
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Decision labels    : {DECISION_LABEL_FILE}")
    print(f"Feature table      : {FEATURE_FILE}")
    print(f"Output dir         : {OUT_DIR}")
    print(f"CV rates           : {CV_RATES_PCT}")
    print(f"XGBoost available  : {XGBOOST_AVAILABLE}")
    print("=" * 96)

    labels, failure_threshold_ft = load_decision_labels()
    timegrid = load_timegrid_features()

    print(f"Loaded labels          : {len(labels):,} cycles")
    print(f"Loaded time-grid rows  : {len(timegrid):,}")
    print(f"Failure threshold      : {failure_threshold_ft:.1f} ft")

    print("\nLabels by split:")
    print(labels.groupby("ml_split").size().reset_index(name="n_cycles").to_string(index=False))

    print("\nBuilding direct cycle-level feature table...")
    features, skipped = build_cycle_feature_table(labels, timegrid)

    feature_cols = select_feature_columns(features)

    print(f"Generated feature rows : {len(features):,}")
    print(f"Feature columns        : {len(feature_cols):,}")
    print(f"Skipped cycle/rate rows: {len(skipped):,}")

    features.to_csv(OUT_DIR / "direct_decision_cycle_features_allrates.csv", index=False)
    skipped.to_csv(OUT_DIR / "direct_decision_skipped_cycles.csv", index=False)

    pd.DataFrame(
        {
            "feature_name": feature_cols,
            "feature_type": "numeric",
            "used_as_model_input": True,
        }
    ).to_csv(OUT_DIR / "direct_decision_feature_columns_used.csv", index=False)

    pred, regressor, classifier, training_summary = train_direct_models(
        features=features,
        feature_cols=feature_cols,
        failure_threshold_ft=failure_threshold_ft,
    )

    joblib.dump(regressor, MODEL_DIR / "direct_residual_queue_regressor.joblib")
    if classifier is not None:
        joblib.dump(classifier, MODEL_DIR / "direct_cycle_failure_classifier.joblib")

    training_summary.to_csv(OUT_DIR / "direct_decision_training_summary.csv", index=False)
    pred.to_csv(OUT_DIR / "direct_decision_predictions_allruns_allrates.csv", index=False)

    long = make_long_prediction_table(pred, failure_threshold_ft)
    long.to_csv(OUT_DIR / "direct_decision_predictions_long.csv", index=False)

    metrics_by_split_rate_model = group_metrics(
        long,
        [
            "ml_split",
            "cv_rate_pct",
            "cv_rate_group",
            "model_id",
            "decision_baseline",
            "decision_mode",
        ],
    )

    metrics_by_split_model = group_metrics(
        long,
        [
            "ml_split",
            "model_id",
            "decision_baseline",
            "decision_mode",
        ],
    )

    metrics_by_split_rate_condition_model = group_metrics(
        long,
        [
            "ml_split",
            "cv_rate_pct",
            "cv_rate_group",
            "traffic_condition",
            "model_id",
            "decision_baseline",
            "decision_mode",
        ],
    )

    metrics_by_split_rate_model.to_csv(
        OUT_DIR / "direct_decision_metrics_by_split_rate_model.csv",
        index=False,
    )
    metrics_by_split_model.to_csv(
        OUT_DIR / "direct_decision_metrics_by_split_model.csv",
        index=False,
    )
    metrics_by_split_rate_condition_model.to_csv(
        OUT_DIR / "direct_decision_metrics_by_split_rate_condition_model.csv",
        index=False,
    )

    save_selected_summaries(metrics_by_split_model)

    test_low_cv = metrics_by_split_rate_model[
        metrics_by_split_rate_model["ml_split"].astype(str).eq("test")
        & metrics_by_split_rate_model["cv_rate_group"].astype(str).eq("low_cv_1_5pct")
    ].copy()
    test_congested = metrics_by_split_rate_condition_model[
        metrics_by_split_rate_condition_model["ml_split"].astype(str).eq("test")
        & metrics_by_split_rate_condition_model["traffic_condition"].astype(str).eq("congested")
    ].copy()

    test_low_cv.to_csv(OUT_DIR / "direct_decision_metrics_test_low_cv.csv", index=False)
    test_congested.to_csv(OUT_DIR / "direct_decision_metrics_test_congested.csv", index=False)

    print("\nSaved:")
    print(f"  {OUT_DIR / 'direct_decision_cycle_features_allrates.csv'}")
    print(f"  {OUT_DIR / 'direct_decision_predictions_allruns_allrates.csv'}")
    print(f"  {OUT_DIR / 'direct_decision_predictions_long.csv'}")
    print(f"  {OUT_DIR / 'direct_decision_metrics_by_split_rate_model.csv'}")
    print(f"  {OUT_DIR / 'direct_decision_metrics_by_split_model.csv'}")
    print(f"  {OUT_DIR / 'direct_decision_metrics_validation.csv'}")
    print(f"  {OUT_DIR / 'direct_decision_metrics_test.csv'}")

    print("\nTraining summary:")
    print(training_summary.round(3).to_string(index=False))

    display_cols = [
        "model_id",
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

    validation = metrics_by_split_model[metrics_by_split_model["ml_split"].astype(str).eq("validation")].copy()
    print("\nValidation direct-decision metrics:")
    print(validation[display_cols].round(3).to_string(index=False))

    test = metrics_by_split_model[metrics_by_split_model["ml_split"].astype(str).eq("test")].copy()
    print("\nTest direct-decision metrics:")
    print(test[display_cols].round(3).to_string(index=False))

    print("\nDirect regression-threshold test metrics by CV rate:")
    by_rate = metrics_by_split_rate_model[
        metrics_by_split_rate_model["ml_split"].astype(str).eq("test")
        & metrics_by_split_rate_model["decision_mode"].astype(str).eq("regression_threshold")
    ].copy()
    by_rate_cols = [
        "cv_rate_pct",
        "cv_rate_group",
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
    print(by_rate[by_rate_cols].round(3).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
