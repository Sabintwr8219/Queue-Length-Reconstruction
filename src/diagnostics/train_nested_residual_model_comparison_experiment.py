"""
Comprehensive nested residual-model experiment for residual-queue estimation.

This script is intentionally separate from the production workflow. It answers
one focused question: can a model that starts from the queue-derived residual
estimate and learns only the remaining error outperform the existing baselines
without degrading cycle-failure performance?

What is tested
--------------
A. Plain nested residual correction with several regression families.
B. Training-weighted correction models that emphasize:
   - large queue-derived errors,
   - cycles near the failure threshold,
   - both objectives together.
C. Gated nested correction:
   - a classifier learns whether the base estimate needs correction,
   - a regressor learns the correction magnitude for cases needing correction.
D. Direction-and-magnitude nested correction:
   - a classifier learns underestimation / approximately correct / overestimation,
   - a regressor learns the absolute correction magnitude.

The script performs model selection using validation data only. Test data are
used only after a candidate has been selected. No production file is changed.

Required upstream files
-----------------------
output/intermediate_csv/full_hybrid_decision_models_boundary_weighted/
    full_hybrid_decision_features_allrates.csv
    full_hybrid_decision_predictions_allrates.csv
    queue_derived_decision_predictions_allrates.csv

Primary outputs
---------------
output/intermediate_csv/nested_residual_model_comparison_experiment/
    model_availability.csv
    feature_columns_used.csv
    screening_validation_metrics.csv
    tuned_validation_metrics.csv
    gated_validation_metrics.csv
    direction_magnitude_validation_metrics.csv
    candidate_validation_metrics_all.csv
    selected_models_validation.csv
    selected_predictions_allrates.csv
    selected_test_metrics.csv
    selected_metrics_by_rate.csv
    comparison_with_existing_test.csv
    top_test_error_rows.csv
    experiment_manifest.csv
    trained_models/*.joblib
"""

from __future__ import annotations

from config import (
    PROJECT_ROOT,
    TRAIN_RUN_IDS,
    VALIDATION_RUN_IDS,
    TEST_RUN_IDS,
    CV_RATES_PCT,
    XGB_RANDOM_SEED,
)

from pathlib import Path
from typing import Any, Iterable
import json
import math
import re
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CatBoostClassifier = None
    CatBoostRegressor = None
    CATBOOST_AVAILABLE = False


# =============================================================================
# Paths and experiment configuration
# =============================================================================

SOURCE_DIR = (
    PROJECT_ROOT
    / "output"
    / "intermediate_csv"
    / "full_hybrid_decision_models_boundary_weighted"
)
FEATURE_FILE = SOURCE_DIR / "full_hybrid_decision_features_allrates.csv"
CURRENT_PREDICTION_FILE = SOURCE_DIR / "full_hybrid_decision_predictions_allrates.csv"
QUEUE_BASELINE_PREDICTION_FILE = SOURCE_DIR / "queue_derived_decision_predictions_allrates.csv"

OUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "intermediate_csv"
    / "nested_residual_model_comparison_experiment"
)
MODEL_DIR = OUT_DIR / "trained_models"

# The current decision-stage profile source. Add "physics_ml_cv_gru" only when a
# second full comparison is deliberately needed. Earlier experiments showed the
# XGBoost profile source to be substantially stronger for cycle-level decisions.
PROFILE_CURVE_IDS = ["physics_ml_cv_xgb"]

BASE_RESIDUAL_SOURCE_COL = "reconstructed_profile_q_green_end"
GT_RESIDUAL_COL = "residual_queue_ft"
GT_FAILURE_COL = "cycle_failure"
DEFAULT_FAILURE_THRESHOLD_FT = 25.0

CURRENT_FULL_HYBRID_MODEL_ID = (
    "full_hybrid_decision_xgb_from_physics_ml_cv_xgb_regression_threshold"
)
QUEUE_DERIVED_MODEL_ID = "queue_derived_threshold_physics_ml_cv_xgb"

KEY_COLS = ["cycle_uid", "run_id", "cycle_number", "cv_rate_pct"]
RANDOM_SEED = int(XGB_RANDOM_SEED)

# Search effort. Increase N_PARAMETER_SAMPLES for an even larger search.
N_PARAMETER_SAMPLES = 18
TOP_FAMILIES_FOR_TUNING = 4
TOP_FAMILIES_FOR_NESTED = 3

# Gated and direction/magnitude architecture settings.
GATE_ERROR_THRESHOLDS_FT = [5.0, 10.0, 15.0, 25.0, 40.0]
DIRECTION_DEADBANDS_FT = [5.0, 10.0, 15.0, 25.0]
GATE_PROBABILITY_THRESHOLD = 0.50

# Candidate must not fall below the existing full-hybrid validation result.
GUARDRAIL_TOLERANCE = 1e-12

REQUIRED_FEATURE_COLS = [
    "cycle_uid",
    "run_id",
    "ml_split",
    "cycle_number",
    "cv_rate_pct",
    "traffic_condition",
    GT_RESIDUAL_COL,
    GT_FAILURE_COL,
    "failure_threshold_ft",
    "profile_curve_id",
    "profile_curve_label",
    BASE_RESIDUAL_SOURCE_COL,
]

NON_FEATURE_COLS = {
    "cycle_uid",
    "run_id",
    "ml_split",
    "cycle_number",
    "cv_rate_group",
    "traffic_condition",
    GT_RESIDUAL_COL,
    GT_FAILURE_COL,
    "failure_threshold_ft",
    "profile_curve_id",
    "profile_curve_label",
    "profile_prediction_col",
    "queue_derived_residual_ft",
    "residual_correction_target_ft",
    "current_full_hybrid_residual_ft",
}

LEAKAGE_TERMS = [
    "gt_",
    "q_gt",
    "ground_truth",
    "residual_queue_ft",
    "cycle_failure",
    "residual_correction_target_ft",
]


# =============================================================================
# Utility functions
# =============================================================================

def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def assign_ml_split(run_id: int) -> str:
    run_id = int(run_id)
    if run_id in TRAIN_RUN_IDS:
        return "train"
    if run_id in VALIDATION_RUN_IDS:
        return "validation"
    if run_id in TEST_RUN_IDS:
        return "test"
    return "other"


def cv_rate_group(rate: Any) -> str:
    if pd.isna(rate):
        return "unknown"
    rate = int(rate)
    if rate <= 5:
        return "low_cv_1_5pct"
    if rate <= 20:
        return "medium_cv_10_20pct"
    return "high_cv_50_100pct"


def safe_divide(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def slugify(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip())
    out = re.sub(r"_+", "_", out).strip("_")
    return out.lower() or "model"


def json_dumps_safe(value: Any) -> str:
    def convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    return json.dumps(value, sort_keys=True, default=convert)


def finite_arrays(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yp)
    return y[mask], yp[mask]


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def classification_scores(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    accuracy = safe_divide(tp + tn, tp + fp + fn + tn)
    false_alarm_rate = safe_divide(fp, fp + tn)
    miss_rate = safe_divide(fn, fn + tp)
    if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0:
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


def compute_metrics(y_true: Any, y_pred: Any, y_failure: Any, threshold_ft: float) -> dict[str, Any]:
    y, yp = finite_arrays(y_true, y_pred)
    if len(y):
        err = yp - y
        mae = float(mean_absolute_error(y, yp))
        rmse = float(math.sqrt(mean_squared_error(y, yp)))
        bias = float(np.mean(err))
        max_abs = float(np.max(np.abs(err)))
    else:
        mae = rmse = bias = max_abs = np.nan

    gt_fail = np.asarray(y_failure, dtype=float)
    pred_fail = (np.asarray(y_pred, dtype=float) >= float(threshold_ft)).astype(float)
    mask = np.isfinite(gt_fail) & np.isfinite(pred_fail)
    gt = gt_fail[mask].astype(int)
    pred = pred_fail[mask].astype(int)
    if len(gt):
        tp, fp, fn, tn = confusion_counts(gt, pred)
        scores = classification_scores(tp, fp, fn, tn)
    else:
        tp = fp = fn = tn = 0
        scores = {k: np.nan for k in [
            "failure_accuracy", "failure_precision", "failure_recall",
            "failure_specificity", "failure_f1",
            "failure_false_alarm_rate", "failure_miss_rate",
        ]}

    return {
        "n_rows": int(len(np.asarray(y_pred))),
        "residual_mae_ft": mae,
        "residual_rmse_ft": rmse,
        "residual_bias_ft": bias,
        "residual_max_abs_error_ft": max_abs,
        **scores,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def guardrail_pass(metrics: dict[str, Any], reference: dict[str, Any]) -> bool:
    f1 = metrics.get("failure_f1", np.nan)
    acc = metrics.get("failure_accuracy", np.nan)
    ref_f1 = reference.get("failure_f1", np.nan)
    ref_acc = reference.get("failure_accuracy", np.nan)
    return bool(
        np.isfinite(f1)
        and np.isfinite(acc)
        and np.isfinite(ref_f1)
        and np.isfinite(ref_acc)
        and f1 + GUARDRAIL_TOLERANCE >= ref_f1
        and acc + GUARDRAIL_TOLERANCE >= ref_acc
    )


def rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["guardrail_rank"] = (~out["classification_guardrail_pass"].astype(bool)).astype(int)
    out["sort_rmse"] = out["residual_rmse_ft"].fillna(1e12)
    out["sort_mae"] = out["residual_mae_ft"].fillna(1e12)
    out["sort_f1"] = -out["failure_f1"].fillna(-1.0)
    out["sort_accuracy"] = -out["failure_accuracy"].fillna(-1.0)
    out = out.sort_values(
        ["guardrail_rank", "sort_rmse", "sort_mae", "sort_f1", "sort_accuracy"],
        ascending=True,
    ).drop(columns=["guardrail_rank", "sort_rmse", "sort_mae", "sort_f1", "sort_accuracy"])
    return out.reset_index(drop=True)


# =============================================================================
# Loading and data preparation
# =============================================================================

def load_feature_data() -> pd.DataFrame:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find feature table:\n{FEATURE_FILE}\n"
            "Run src/train_full_hybrid_decision_models_boundary_weighted.py first."
        )

    df = pd.read_csv(FEATURE_FILE)
    require_columns(df, REQUIRED_FEATURE_COLS, "full-hybrid decision feature table")

    numeric_cols = [
        "run_id", "cycle_number", "cv_rate_pct", GT_RESIDUAL_COL,
        GT_FAILURE_COL, "failure_threshold_ft", BASE_RESIDUAL_SOURCE_COL,
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["profile_curve_id"].astype(str).isin(PROFILE_CURVE_IDS)].copy()
    df = df.dropna(subset=[
        "run_id", "cycle_number", "cv_rate_pct", GT_RESIDUAL_COL,
        GT_FAILURE_COL, BASE_RESIDUAL_SOURCE_COL,
    ]).copy()

    df["run_id"] = df["run_id"].astype(int)
    df["cycle_number"] = df["cycle_number"].astype(int)
    df["cv_rate_pct"] = df["cv_rate_pct"].astype(int)
    df[GT_FAILURE_COL] = df[GT_FAILURE_COL].astype(int)
    df["ml_split"] = df["run_id"].apply(assign_ml_split)
    df["cv_rate_group"] = df["cv_rate_pct"].apply(cv_rate_group)
    df["queue_derived_residual_ft"] = np.maximum(
        pd.to_numeric(df[BASE_RESIDUAL_SOURCE_COL], errors="coerce"), 0.0
    )
    df["residual_correction_target_ft"] = (
        df[GT_RESIDUAL_COL] - df["queue_derived_residual_ft"]
    )

    model_runs = set(TRAIN_RUN_IDS + VALIDATION_RUN_IDS + TEST_RUN_IDS)
    df = df[df["run_id"].isin(model_runs)].copy()
    df = df[df["cv_rate_pct"].isin(CV_RATES_PCT)].copy()
    return df.sort_values([
        "profile_curve_id", "run_id", "cycle_number", "cv_rate_pct"
    ]).reset_index(drop=True)


def load_existing_predictions() -> pd.DataFrame:
    parts = []

    if CURRENT_PREDICTION_FILE.exists():
        cur = pd.read_csv(CURRENT_PREDICTION_FILE)
        needed = KEY_COLS + [
            "ml_split", "traffic_condition", "model_id",
            "gt_residual_queue_ft", "gt_cycle_failure",
            "pred_residual_queue_ft", "pred_cycle_failure",
            "failure_threshold_ft",
        ]
        if set(needed).issubset(cur.columns):
            cur = cur[cur["model_id"].astype(str).eq(CURRENT_FULL_HYBRID_MODEL_ID)].copy()
            cur["comparison_source"] = "current_full_hybrid"
            parts.append(cur[needed + ["comparison_source"]])

    if QUEUE_BASELINE_PREDICTION_FILE.exists():
        base = pd.read_csv(QUEUE_BASELINE_PREDICTION_FILE)
        rename = {}
        if "residual_queue_ft" in base.columns:
            rename["residual_queue_ft"] = "gt_residual_queue_ft"
        if "cycle_failure" in base.columns:
            rename["cycle_failure"] = "gt_cycle_failure"
        base = base.rename(columns=rename)
        needed = KEY_COLS + [
            "ml_split", "traffic_condition", "model_id",
            "gt_residual_queue_ft", "gt_cycle_failure",
            "pred_residual_queue_ft", "pred_cycle_failure",
            "failure_threshold_ft",
        ]
        if set(needed).issubset(base.columns):
            base = base[base["model_id"].astype(str).eq(QUEUE_DERIVED_MODEL_ID)].copy()
            base["comparison_source"] = "queue_derived_baseline"
            parts.append(base[needed + ["comparison_source"]])

    if not parts:
        raise FileNotFoundError(
            "Could not load the current full-hybrid and queue-derived comparison predictions."
        )

    out = pd.concat(parts, ignore_index=True)
    for col in ["run_id", "cycle_number", "cv_rate_pct", "failure_threshold_ft"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["run_id"] = out["run_id"].astype(int)
    out["cycle_number"] = out["cycle_number"].astype(int)
    out["cv_rate_pct"] = out["cv_rate_pct"].astype(int)
    return out


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in NON_FEATURE_COLS or col.endswith("_uid"):
            continue
        if df[col].dtype.kind not in "biufc":
            continue
        lower = col.lower()
        if any(term in lower for term in LEAKAGE_TERMS):
            continue
        cols.append(col)
    if not cols:
        raise ValueError("No usable model features were found.")
    return cols


def existing_reference_metrics(existing: pd.DataFrame, split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for source, g in existing[existing["ml_split"].astype(str).eq(split)].groupby("comparison_source"):
        threshold = float(pd.to_numeric(g["failure_threshold_ft"], errors="coerce").dropna().iloc[0])
        m = compute_metrics(
            g["gt_residual_queue_ft"],
            g["pred_residual_queue_ft"],
            g["gt_cycle_failure"],
            threshold,
        )
        m.update({
            "candidate_id": str(g["model_id"].iloc[0]),
            "strategy_group": source,
            "model_family": source,
            "weight_scheme": "existing",
            "split": split,
            "classification_guardrail_pass": True,
        })
        rows.append(m)
    metrics_df = pd.DataFrame(rows)

    ref_rows = metrics_df[metrics_df["strategy_group"].eq("current_full_hybrid")]
    if ref_rows.empty:
        raise ValueError("Current full-hybrid validation reference could not be calculated.")
    ref = ref_rows.iloc[0].to_dict()
    return metrics_df, ref


# =============================================================================
# Weight schemes
# =============================================================================

def training_weights(df: pd.DataFrame, scheme: str, threshold_ft: float) -> np.ndarray:
    correction = np.abs(df["residual_correction_target_ft"].to_numpy(dtype=float))
    gt = df[GT_RESIDUAL_COL].to_numpy(dtype=float)

    large_scaled = np.clip(correction / 50.0, 0.0, 3.0)
    near_threshold = np.exp(-np.abs(gt - float(threshold_ft)) / 10.0)

    if scheme == "plain":
        w = np.ones(len(df), dtype=float)
    elif scheme == "large_error_moderate":
        w = 1.0 + 2.0 * large_scaled
    elif scheme == "large_error_strong":
        w = 1.0 + 4.0 * large_scaled
    elif scheme == "near_threshold_moderate":
        w = 1.0 + 2.0 * near_threshold
    elif scheme == "near_threshold_strong":
        w = 1.0 + 4.0 * near_threshold
    elif scheme == "combined_moderate":
        w = 1.0 + 2.0 * large_scaled + 2.0 * near_threshold
    elif scheme == "combined_strong":
        w = 1.0 + 4.0 * large_scaled + 3.0 * near_threshold
    else:
        raise ValueError(f"Unknown weight scheme: {scheme}")

    w = np.clip(w, 0.25, 12.0)
    return w / np.mean(w)


WEIGHT_SCHEMES = [
    "plain",
    "large_error_moderate",
    "large_error_strong",
    "near_threshold_moderate",
    "near_threshold_strong",
    "combined_moderate",
    "combined_strong",
]


# =============================================================================
# Model factories and search spaces
# =============================================================================

def available_regression_families() -> list[str]:
    families = [
        "ridge",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "hist_gradient_boosting",
        "mlp",
    ]
    if XGBOOST_AVAILABLE:
        families.append("xgboost")
    if LIGHTGBM_AVAILABLE:
        families.append("lightgbm")
    if CATBOOST_AVAILABLE:
        families.append("catboost")
    return families


def supports_weighted_training(family: str) -> bool:
    return family != "mlp"


def base_regression_params(family: str) -> dict[str, Any]:
    configs = {
        "ridge": {"alpha": 10.0},
        "random_forest": {
            "n_estimators": 600,
            "max_depth": 14,
            "min_samples_leaf": 2,
            "max_features": 0.8,
        },
        "extra_trees": {
            "n_estimators": 600,
            "max_depth": None,
            "min_samples_leaf": 2,
            "max_features": 0.9,
        },
        "gradient_boosting": {
            "n_estimators": 400,
            "learning_rate": 0.03,
            "max_depth": 2,
            "min_samples_leaf": 3,
            "loss": "huber",
        },
        "hist_gradient_boosting": {
            "max_iter": 500,
            "learning_rate": 0.04,
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0,
        },
        "mlp": {
            "hidden_layer_sizes": (128, 64, 32),
            "alpha": 1e-3,
            "learning_rate_init": 5e-4,
            "max_iter": 800,
        },
        "xgboost": {
            "n_estimators": 600,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 3.0,
            "min_child_weight": 2.0,
        },
        "lightgbm": {
            "n_estimators": 600,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "reg_lambda": 2.0,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },
        "catboost": {
            "iterations": 700,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
        },
    }
    return dict(configs[family])


def regression_search_space(family: str) -> dict[str, list[Any]]:
    spaces = {
        "ridge": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0],
        },
        "random_forest": {
            "n_estimators": [400, 700, 1000],
            "max_depth": [None, 8, 12, 18],
            "min_samples_leaf": [1, 2, 4, 7],
            "max_features": ["sqrt", 0.5, 0.8, 1.0],
        },
        "extra_trees": {
            "n_estimators": [400, 700, 1000],
            "max_depth": [None, 8, 12, 18],
            "min_samples_leaf": [1, 2, 4, 7],
            "max_features": ["sqrt", 0.5, 0.8, 1.0],
        },
        "gradient_boosting": {
            "n_estimators": [200, 400, 700],
            "learning_rate": [0.015, 0.03, 0.06],
            "max_depth": [1, 2, 3],
            "min_samples_leaf": [2, 4, 8],
            "loss": ["squared_error", "huber", "absolute_error"],
        },
        "hist_gradient_boosting": {
            "max_iter": [300, 500, 800],
            "learning_rate": [0.02, 0.04, 0.08],
            "max_leaf_nodes": [15, 31, 63],
            "l2_regularization": [0.0, 1.0, 5.0, 15.0],
            "min_samples_leaf": [10, 20, 35],
        },
        "mlp": {
            "hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
            "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "learning_rate_init": [1e-4, 5e-4, 1e-3],
            "max_iter": [600, 1000],
        },
        "xgboost": {
            "n_estimators": [300, 500, 800, 1100],
            "max_depth": [2, 3, 4, 6],
            "learning_rate": [0.015, 0.03, 0.06],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_lambda": [1.0, 3.0, 8.0, 20.0],
            "min_child_weight": [1.0, 3.0, 8.0],
        },
        "lightgbm": {
            "n_estimators": [300, 600, 1000],
            "learning_rate": [0.015, 0.03, 0.06],
            "num_leaves": [15, 31, 63],
            "max_depth": [-1, 6, 10],
            "min_child_samples": [10, 20, 40],
            "reg_lambda": [0.0, 2.0, 8.0],
            "subsample": [0.75, 0.9, 1.0],
            "colsample_bytree": [0.75, 0.9, 1.0],
        },
        "catboost": {
            "iterations": [400, 700, 1000],
            "depth": [4, 6, 8],
            "learning_rate": [0.015, 0.03, 0.06],
            "l2_leaf_reg": [1.0, 3.0, 8.0, 20.0],
            "loss_function": ["RMSE", "MAE"],
        },
    }
    return spaces[family]


def make_regression_pipeline(family: str, params: dict[str, Any]) -> Pipeline:
    if family == "ridge":
        model = Ridge(**params)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])

    if family == "random_forest":
        model = RandomForestRegressor(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            **params,
        )
    elif family == "extra_trees":
        model = ExtraTreesRegressor(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            **params,
        )
    elif family == "gradient_boosting":
        model = GradientBoostingRegressor(
            random_state=RANDOM_SEED,
            **params,
        )
    elif family == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            random_state=RANDOM_SEED,
            **params,
        )
    elif family == "mlp":
        model = MLPRegressor(
            random_state=RANDOM_SEED,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=40,
            **params,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])
    elif family == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost is not available.")
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            tree_method="hist",
            **params,
        )
    elif family == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise RuntimeError("LightGBM is not available.")
        model = LGBMRegressor(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )
    elif family == "catboost":
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost is not available.")
        model = CatBoostRegressor(
            random_seed=RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,
            **params,
        )
    else:
        raise ValueError(f"Unknown regression family: {family}")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def classifier_family_for(regression_family: str) -> str:
    if regression_family == "ridge":
        return "logistic"
    if regression_family == "mlp":
        return "hist_gradient_boosting"
    return regression_family


def make_classifier_pipeline(
    family: str,
    n_classes: int,
    class_ratio: float | None = None,
) -> Pipeline:
    if family == "logistic":
        model = LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_SEED,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])

    if family == "random_forest":
        model = RandomForestClassifier(
            n_estimators=700,
            max_depth=14,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    elif family == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=700,
            max_depth=None,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    elif family == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
        )
    elif family == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_SEED,
        )
    elif family == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost is not available.")
        kwargs: dict[str, Any] = {
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "tree_method": "hist",
            "n_estimators": 600,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 3.0,
        }
        if n_classes == 2:
            kwargs.update({"objective": "binary:logistic", "eval_metric": "logloss"})
        else:
            kwargs.update({
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "num_class": int(n_classes),
            })
        model = XGBClassifier(**kwargs)
    elif family == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise RuntimeError("LightGBM is not available.")
        objective = "binary" if n_classes == 2 else "multiclass"
        model = LGBMClassifier(
            objective=objective,
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1,
        )
    elif family == "catboost":
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost is not available.")
        loss = "Logloss" if n_classes == 2 else "MultiClass"
        model = CatBoostClassifier(
            iterations=700,
            depth=6,
            learning_rate=0.03,
            loss_function=loss,
            random_seed=RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,
        )
    else:
        raise ValueError(f"Unknown classifier family: {family}")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def fit_with_optional_weights(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
) -> Pipeline:
    if sample_weight is None:
        pipeline.fit(X, y)
        return pipeline
    try:
        pipeline.fit(X, y, model__sample_weight=sample_weight)
    except TypeError:
        pipeline.fit(X, y)
    return pipeline


# =============================================================================
# Plain and weighted correction candidates
# =============================================================================

def evaluate_correction_candidate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: list[str],
    family: str,
    params: dict[str, Any],
    weight_scheme: str,
    threshold_ft: float,
    reference_metrics: dict[str, Any],
    stage: str,
    candidate_index: int,
) -> dict[str, Any]:
    X_train = train[feature_cols]
    y_train = train["residual_correction_target_ft"].to_numpy(dtype=float)
    X_val = validation[feature_cols]

    weights = None
    if weight_scheme != "plain" and supports_weighted_training(family):
        weights = training_weights(train, weight_scheme, threshold_ft)
    elif weight_scheme != "plain" and not supports_weighted_training(family):
        raise ValueError(f"{family} does not support the requested weighted variant.")

    started = time.perf_counter()
    model = make_regression_pipeline(family, params)
    fit_with_optional_weights(model, X_train, y_train, weights)
    pred_correction = model.predict(X_val).astype(float)
    pred_residual = np.maximum(
        validation["queue_derived_residual_ft"].to_numpy(dtype=float) + pred_correction,
        0.0,
    )
    elapsed = time.perf_counter() - started

    metrics = compute_metrics(
        validation[GT_RESIDUAL_COL],
        pred_residual,
        validation[GT_FAILURE_COL],
        threshold_ft,
    )
    metrics.update({
        "candidate_id": f"{stage}_{candidate_index:05d}_{family}_{weight_scheme}",
        "strategy_group": "plain_or_weighted_correction",
        "search_stage": stage,
        "model_family": family,
        "gate_model_family": "",
        "correction_model_family": family,
        "weight_scheme": weight_scheme,
        "gate_error_threshold_ft": np.nan,
        "direction_deadband_ft": np.nan,
        "model_params_json": json_dumps_safe(params),
        "classifier_params_json": "{}",
        "fit_seconds": float(elapsed),
    })
    metrics["classification_guardrail_pass"] = guardrail_pass(metrics, reference_metrics)
    return metrics


def screening_search(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: list[str],
    threshold_ft: float,
    reference_metrics: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    idx = 0
    for family in available_regression_families():
        schemes = WEIGHT_SCHEMES if supports_weighted_training(family) else ["plain"]
        for scheme in schemes:
            idx += 1
            try:
                row = evaluate_correction_candidate(
                    train, validation, feature_cols, family,
                    base_regression_params(family), scheme,
                    threshold_ft, reference_metrics,
                    stage="screening", candidate_index=idx,
                )
                rows.append(row)
                print(
                    f"[SCREEN] {family:24s} {scheme:26s} "
                    f"RMSE={row['residual_rmse_ft']:.3f} F1={row['failure_f1']:.3f} "
                    f"guardrail={row['classification_guardrail_pass']}"
                )
            except Exception as exc:
                print(f"[WARN] Screening failed for {family}/{scheme}: {exc}")
    return pd.DataFrame(rows)


def select_top_families(screening: pd.DataFrame) -> list[str]:
    ranked_family_rows = []
    for family, g in screening.groupby("model_family", sort=True):
        ranked = rank_candidates(g)
        if not ranked.empty:
            ranked_family_rows.append(ranked.iloc[0].to_dict())
    ranked_families = rank_candidates(pd.DataFrame(ranked_family_rows))
    return ranked_families["model_family"].astype(str).head(TOP_FAMILIES_FOR_TUNING).tolist()


def sampled_parameter_configs(family: str) -> list[dict[str, Any]]:
    space = regression_search_space(family)
    total_grid = int(np.prod([len(v) for v in space.values()]))
    if total_grid <= N_PARAMETER_SAMPLES:
        return list(ParameterGrid(space))
    return list(ParameterSampler(space, n_iter=N_PARAMETER_SAMPLES, random_state=RANDOM_SEED))


def tuning_search(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: list[str],
    threshold_ft: float,
    reference_metrics: dict[str, Any],
    top_families: list[str],
) -> pd.DataFrame:
    rows = []
    idx = 0
    for family in top_families:
        configs = sampled_parameter_configs(family)
        schemes = WEIGHT_SCHEMES if supports_weighted_training(family) else ["plain"]
        print(f"\n[TUNE] {family}: {len(configs)} parameter sets x {len(schemes)} weight schemes")
        for params in configs:
            for scheme in schemes:
                idx += 1
                try:
                    row = evaluate_correction_candidate(
                        train, validation, feature_cols, family, params, scheme,
                        threshold_ft, reference_metrics,
                        stage="tuned", candidate_index=idx,
                    )
                    rows.append(row)
                except Exception as exc:
                    print(f"[WARN] Tuning failed for {family}/{scheme}: {exc}")
        family_rows = pd.DataFrame([r for r in rows if r["model_family"] == family])
        if not family_rows.empty:
            best = rank_candidates(family_rows).iloc[0]
            print(
                f"[TUNE BEST] {family}: RMSE={best['residual_rmse_ft']:.3f}, "
                f"F1={best['failure_f1']:.3f}, guardrail={best['classification_guardrail_pass']}"
            )
    return pd.DataFrame(rows)


# =============================================================================
# Gated nested correction
# =============================================================================

def best_regression_spec_by_family(candidates: pd.DataFrame) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for family, g in candidates.groupby("model_family", sort=True):
        best = rank_candidates(g).iloc[0]
        specs[str(family)] = {
            "params": json.loads(best["model_params_json"]),
            "weight_scheme": str(best["weight_scheme"]),
        }
    return specs


def classifier_sample_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = max(len(classes), 1)
    weight_map = {int(c): total / (n_classes * max(int(n), 1)) for c, n in zip(classes, counts)}
    return np.asarray([weight_map[int(v)] for v in y], dtype=float)


def gated_search(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: list[str],
    threshold_ft: float,
    reference_metrics: dict[str, Any],
    regression_specs: dict[str, dict[str, Any]],
    nested_families: list[str],
) -> pd.DataFrame:
    rows = []
    idx = 0

    pairs = [(f, f) for f in nested_families]
    if "xgboost" in nested_families and "extra_trees" in nested_families:
        pairs.extend([
            ("xgboost", "extra_trees"),
            ("extra_trees", "xgboost"),
        ])
    pairs = list(dict.fromkeys(pairs))

    X_train = train[feature_cols]
    X_val = validation[feature_cols]
    correction_train = train["residual_correction_target_ft"].to_numpy(dtype=float)

    for gate_family, correction_family in pairs:
        if correction_family not in regression_specs:
            continue
        spec = regression_specs[correction_family]
        for error_threshold in GATE_ERROR_THRESHOLDS_FT:
            idx += 1
            gate_target = (np.abs(correction_train) >= float(error_threshold)).astype(int)
            if len(np.unique(gate_target)) < 2 or int(gate_target.sum()) < 10:
                continue

            try:
                n_pos = int(np.sum(gate_target == 1))
                n_neg = int(np.sum(gate_target == 0))
                class_ratio = n_neg / max(n_pos, 1)
                gate = make_classifier_pipeline(
                    classifier_family_for(gate_family),
                    n_classes=2,
                    class_ratio=class_ratio,
                )
                fit_with_optional_weights(
                    gate,
                    X_train,
                    gate_target,
                    classifier_sample_weights(gate_target),
                )

                correction_mask = gate_target == 1
                correction_model = make_regression_pipeline(
                    correction_family,
                    spec["params"],
                )
                correction_weights = None
                if supports_weighted_training(correction_family):
                    all_weights = training_weights(train, spec["weight_scheme"], threshold_ft)
                    correction_weights = all_weights[correction_mask]
                fit_with_optional_weights(
                    correction_model,
                    X_train.loc[correction_mask],
                    correction_train[correction_mask],
                    correction_weights,
                )

                gate_prob = gate.predict_proba(X_val)[:, 1]
                gate_pred = gate_prob >= GATE_PROBABILITY_THRESHOLD
                correction_pred = correction_model.predict(X_val).astype(float)
                correction_pred = np.where(gate_pred, correction_pred, 0.0)
                residual_pred = np.maximum(
                    validation["queue_derived_residual_ft"].to_numpy(dtype=float)
                    + correction_pred,
                    0.0,
                )

                metrics = compute_metrics(
                    validation[GT_RESIDUAL_COL], residual_pred,
                    validation[GT_FAILURE_COL], threshold_ft,
                )
                metrics.update({
                    "candidate_id": f"gated_{idx:05d}_{gate_family}_{correction_family}_{int(error_threshold)}ft",
                    "strategy_group": "gated_nested_correction",
                    "search_stage": "gated",
                    "model_family": f"{gate_family}_gate__{correction_family}_correction",
                    "gate_model_family": gate_family,
                    "correction_model_family": correction_family,
                    "weight_scheme": spec["weight_scheme"],
                    "gate_error_threshold_ft": float(error_threshold),
                    "direction_deadband_ft": np.nan,
                    "model_params_json": json_dumps_safe(spec["params"]),
                    "classifier_params_json": json_dumps_safe({
                        "gate_probability_threshold": GATE_PROBABILITY_THRESHOLD,
                    }),
                    "fit_seconds": np.nan,
                    "gate_positive_rate_train": float(np.mean(gate_target)),
                    "gate_positive_rate_validation": float(np.mean(gate_pred)),
                })
                metrics["classification_guardrail_pass"] = guardrail_pass(metrics, reference_metrics)
                rows.append(metrics)
                print(
                    f"[GATE] {gate_family}->{correction_family} err>={error_threshold:4.0f} "
                    f"RMSE={metrics['residual_rmse_ft']:.3f} F1={metrics['failure_f1']:.3f} "
                    f"guardrail={metrics['classification_guardrail_pass']}"
                )
            except Exception as exc:
                print(f"[WARN] Gated model failed for {gate_family}->{correction_family}: {exc}")

    return pd.DataFrame(rows)


# =============================================================================
# Direction-and-magnitude nested correction
# =============================================================================

def direction_magnitude_search(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: list[str],
    threshold_ft: float,
    reference_metrics: dict[str, Any],
    regression_specs: dict[str, dict[str, Any]],
    nested_families: list[str],
) -> pd.DataFrame:
    rows = []
    idx = 0

    pairs = [(f, f) for f in nested_families]
    if "xgboost" in nested_families and "extra_trees" in nested_families:
        pairs.extend([
            ("xgboost", "extra_trees"),
            ("extra_trees", "xgboost"),
        ])
    pairs = list(dict.fromkeys(pairs))

    X_train = train[feature_cols]
    X_val = validation[feature_cols]
    correction_train = train["residual_correction_target_ft"].to_numpy(dtype=float)

    for direction_family, magnitude_family in pairs:
        if magnitude_family not in regression_specs:
            continue
        spec = regression_specs[magnitude_family]

        for deadband in DIRECTION_DEADBANDS_FT:
            idx += 1
            # 0 = base overestimates (negative correction)
            # 1 = approximately correct
            # 2 = base underestimates (positive correction)
            direction_target = np.full(len(train), 1, dtype=int)
            direction_target[correction_train < -float(deadband)] = 0
            direction_target[correction_train > float(deadband)] = 2

            if len(np.unique(direction_target)) < 3:
                continue

            nonneutral = direction_target != 1
            if int(np.sum(nonneutral)) < 20:
                continue

            try:
                direction_model = make_classifier_pipeline(
                    classifier_family_for(direction_family),
                    n_classes=3,
                )
                fit_with_optional_weights(
                    direction_model,
                    X_train,
                    direction_target,
                    classifier_sample_weights(direction_target),
                )

                magnitude_model = make_regression_pipeline(
                    magnitude_family,
                    spec["params"],
                )
                magnitude_weights = None
                if supports_weighted_training(magnitude_family):
                    all_weights = training_weights(train, spec["weight_scheme"], threshold_ft)
                    magnitude_weights = all_weights[nonneutral]
                fit_with_optional_weights(
                    magnitude_model,
                    X_train.loc[nonneutral],
                    np.abs(correction_train[nonneutral]),
                    magnitude_weights,
                )

                direction_pred = direction_model.predict(X_val).astype(int)
                magnitude_pred = np.maximum(magnitude_model.predict(X_val).astype(float), 0.0)
                sign = np.zeros(len(validation), dtype=float)
                sign[direction_pred == 0] = -1.0
                sign[direction_pred == 2] = 1.0
                correction_pred = sign * magnitude_pred
                residual_pred = np.maximum(
                    validation["queue_derived_residual_ft"].to_numpy(dtype=float)
                    + correction_pred,
                    0.0,
                )

                metrics = compute_metrics(
                    validation[GT_RESIDUAL_COL], residual_pred,
                    validation[GT_FAILURE_COL], threshold_ft,
                )
                metrics.update({
                    "candidate_id": f"direction_{idx:05d}_{direction_family}_{magnitude_family}_{int(deadband)}ft",
                    "strategy_group": "direction_magnitude_correction",
                    "search_stage": "direction_magnitude",
                    "model_family": f"{direction_family}_direction__{magnitude_family}_magnitude",
                    "gate_model_family": direction_family,
                    "correction_model_family": magnitude_family,
                    "weight_scheme": spec["weight_scheme"],
                    "gate_error_threshold_ft": np.nan,
                    "direction_deadband_ft": float(deadband),
                    "model_params_json": json_dumps_safe(spec["params"]),
                    "classifier_params_json": "{}",
                    "fit_seconds": np.nan,
                    "direction_negative_rate_train": float(np.mean(direction_target == 0)),
                    "direction_neutral_rate_train": float(np.mean(direction_target == 1)),
                    "direction_positive_rate_train": float(np.mean(direction_target == 2)),
                })
                metrics["classification_guardrail_pass"] = guardrail_pass(metrics, reference_metrics)
                rows.append(metrics)
                print(
                    f"[DIR] {direction_family}->{magnitude_family} deadband={deadband:4.0f} "
                    f"RMSE={metrics['residual_rmse_ft']:.3f} F1={metrics['failure_f1']:.3f} "
                    f"guardrail={metrics['classification_guardrail_pass']}"
                )
            except Exception as exc:
                print(
                    f"[WARN] Direction/magnitude failed for "
                    f"{direction_family}->{magnitude_family}: {exc}"
                )

    return pd.DataFrame(rows)


# =============================================================================
# Selected-candidate reconstruction and test evaluation
# =============================================================================

def parse_candidate_spec(row: pd.Series) -> dict[str, Any]:
    return {
        "candidate_id": str(row["candidate_id"]),
        "strategy_group": str(row["strategy_group"]),
        "model_family": str(row["model_family"]),
        "gate_model_family": str(row.get("gate_model_family", "")),
        "correction_model_family": str(row.get("correction_model_family", "")),
        "weight_scheme": str(row.get("weight_scheme", "plain")),
        "gate_error_threshold_ft": float(row.get("gate_error_threshold_ft", np.nan)),
        "direction_deadband_ft": float(row.get("direction_deadband_ft", np.nan)),
        "params": json.loads(str(row.get("model_params_json", "{}"))),
    }


def fit_selected_candidate(
    spec: dict[str, Any],
    train: pd.DataFrame,
    all_data: pd.DataFrame,
    feature_cols: list[str],
    threshold_ft: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    X_train = train[feature_cols]
    X_all = all_data[feature_cols]
    correction_train = train["residual_correction_target_ft"].to_numpy(dtype=float)
    base_all = all_data["queue_derived_residual_ft"].to_numpy(dtype=float)

    strategy = spec["strategy_group"]
    bundle: dict[str, Any] = {"spec": spec, "feature_cols": feature_cols}

    if strategy == "plain_or_weighted_correction":
        family = spec["correction_model_family"]
        model = make_regression_pipeline(family, spec["params"])
        weights = None
        if spec["weight_scheme"] != "plain" and supports_weighted_training(family):
            weights = training_weights(train, spec["weight_scheme"], threshold_ft)
        fit_with_optional_weights(model, X_train, correction_train, weights)
        correction_pred = model.predict(X_all).astype(float)
        bundle["correction_model"] = model

    elif strategy == "gated_nested_correction":
        gate_family = spec["gate_model_family"]
        correction_family = spec["correction_model_family"]
        gate_threshold = float(spec["gate_error_threshold_ft"])
        gate_target = (np.abs(correction_train) >= gate_threshold).astype(int)

        n_pos = int(np.sum(gate_target == 1))
        n_neg = int(np.sum(gate_target == 0))
        gate = make_classifier_pipeline(
            classifier_family_for(gate_family),
            n_classes=2,
            class_ratio=n_neg / max(n_pos, 1),
        )
        fit_with_optional_weights(
            gate, X_train, gate_target, classifier_sample_weights(gate_target)
        )

        correction_mask = gate_target == 1
        correction_model = make_regression_pipeline(correction_family, spec["params"])
        weights = None
        if supports_weighted_training(correction_family):
            all_weights = training_weights(train, spec["weight_scheme"], threshold_ft)
            weights = all_weights[correction_mask]
        fit_with_optional_weights(
            correction_model,
            X_train.loc[correction_mask],
            correction_train[correction_mask],
            weights,
        )

        gate_pred = gate.predict_proba(X_all)[:, 1] >= GATE_PROBABILITY_THRESHOLD
        raw_corr = correction_model.predict(X_all).astype(float)
        correction_pred = np.where(gate_pred, raw_corr, 0.0)
        bundle.update({"gate_model": gate, "correction_model": correction_model})

    elif strategy == "direction_magnitude_correction":
        direction_family = spec["gate_model_family"]
        magnitude_family = spec["correction_model_family"]
        deadband = float(spec["direction_deadband_ft"])

        direction_target = np.full(len(train), 1, dtype=int)
        direction_target[correction_train < -deadband] = 0
        direction_target[correction_train > deadband] = 2
        nonneutral = direction_target != 1

        direction_model = make_classifier_pipeline(
            classifier_family_for(direction_family),
            n_classes=3,
        )
        fit_with_optional_weights(
            direction_model,
            X_train,
            direction_target,
            classifier_sample_weights(direction_target),
        )

        magnitude_model = make_regression_pipeline(magnitude_family, spec["params"])
        weights = None
        if supports_weighted_training(magnitude_family):
            all_weights = training_weights(train, spec["weight_scheme"], threshold_ft)
            weights = all_weights[nonneutral]
        fit_with_optional_weights(
            magnitude_model,
            X_train.loc[nonneutral],
            np.abs(correction_train[nonneutral]),
            weights,
        )

        direction_pred = direction_model.predict(X_all).astype(int)
        magnitude_pred = np.maximum(magnitude_model.predict(X_all).astype(float), 0.0)
        sign = np.zeros(len(all_data), dtype=float)
        sign[direction_pred == 0] = -1.0
        sign[direction_pred == 2] = 1.0
        correction_pred = sign * magnitude_pred
        bundle.update({
            "direction_model": direction_model,
            "magnitude_model": magnitude_model,
        })
    else:
        raise ValueError(f"Unsupported strategy group: {strategy}")

    residual_pred = np.maximum(base_all + correction_pred, 0.0)
    bundle["failure_threshold_ft"] = threshold_ft
    return residual_pred, bundle


def select_candidates_for_test(all_candidates: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []

    # Best overall candidate.
    ranked_all = rank_candidates(all_candidates)
    if not ranked_all.empty:
        overall = ranked_all.iloc[0].copy()
        overall["selection_scope"] = "overall_best"
        selected_rows.append(overall)

    # Best candidate from each architecture group.
    for strategy, g in all_candidates.groupby("strategy_group", sort=True):
        ranked = rank_candidates(g)
        if not ranked.empty:
            row = ranked.iloc[0].copy()
            row["selection_scope"] = f"best_{strategy}"
            selected_rows.append(row)

    # Best candidate from each correction learner family for diagnostic clarity.
    for family, g in all_candidates.groupby("correction_model_family", sort=True):
        ranked = rank_candidates(g)
        if not ranked.empty:
            row = ranked.iloc[0].copy()
            row["selection_scope"] = f"best_family_{family}"
            selected_rows.append(row)

    out = pd.DataFrame(selected_rows)
    if out.empty:
        raise RuntimeError("No candidates were available for test evaluation.")
    out = out.drop_duplicates(subset=["candidate_id"]).reset_index(drop=True)
    return out


def predictions_to_long(
    data: pd.DataFrame,
    candidate_id: str,
    selection_scope: str,
    strategy_group: str,
    model_family: str,
    residual_pred: np.ndarray,
    threshold_ft: float,
) -> pd.DataFrame:
    cols = [
        "cycle_uid", "run_id", "ml_split", "cycle_number", "cv_rate_pct",
        "cv_rate_group", "traffic_condition", "profile_curve_id",
        "profile_curve_label", GT_RESIDUAL_COL, GT_FAILURE_COL,
        "queue_derived_residual_ft", "residual_correction_target_ft",
    ]
    cols = [c for c in cols if c in data.columns]
    out = data[cols].copy()
    out = out.rename(columns={
        GT_RESIDUAL_COL: "gt_residual_queue_ft",
        GT_FAILURE_COL: "gt_cycle_failure",
    })
    out["candidate_id"] = candidate_id
    out["selection_scope"] = selection_scope
    out["strategy_group"] = strategy_group
    out["model_family"] = model_family
    out["pred_residual_queue_ft"] = residual_pred
    out["pred_residual_correction_ft"] = (
        residual_pred - out["queue_derived_residual_ft"].to_numpy(dtype=float)
    )
    out["pred_cycle_failure"] = (residual_pred >= threshold_ft).astype(int)
    out["failure_threshold_ft"] = threshold_ft
    out["residual_error_ft"] = (
        out["pred_residual_queue_ft"] - out["gt_residual_queue_ft"]
    )
    out["residual_abs_error_ft"] = np.abs(out["residual_error_ft"])
    return out


def metrics_from_prediction_long(predictions: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in predictions.groupby(group_cols, dropna=False, sort=True):
        if len(group_cols) == 1:
            key = (key,)
        row = {c: key[i] for i, c in enumerate(group_cols)}
        threshold = float(pd.to_numeric(g["failure_threshold_ft"], errors="coerce").dropna().iloc[0])
        row.update(compute_metrics(
            g["gt_residual_queue_ft"],
            g["pred_residual_queue_ft"],
            g["gt_cycle_failure"],
            threshold,
        ))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 104)
    print("Comprehensive nested residual-model comparison experiment")
    print("=" * 104)
    print(f"Feature file       : {FEATURE_FILE}")
    print(f"Current predictions: {CURRENT_PREDICTION_FILE}")
    print(f"Queue baseline     : {QUEUE_BASELINE_PREDICTION_FILE}")
    print(f"Output folder      : {OUT_DIR}")
    print(f"Profile source(s)  : {PROFILE_CURVE_IDS}")
    print("=" * 104)

    availability = pd.DataFrame([
        {"model_family": "xgboost", "available": XGBOOST_AVAILABLE},
        {"model_family": "lightgbm", "available": LIGHTGBM_AVAILABLE},
        {"model_family": "catboost", "available": CATBOOST_AVAILABLE},
        {"model_family": "scikit_learn_models", "available": True},
    ])
    availability.to_csv(OUT_DIR / "model_availability.csv", index=False)
    print("\nModel availability:")
    print(availability.to_string(index=False))

    data = load_feature_data()
    existing = load_existing_predictions()
    feature_cols = select_feature_columns(data)

    feature_df = pd.DataFrame({"feature_name": feature_cols})
    feature_df["used_as_model_input"] = True
    feature_df.to_csv(OUT_DIR / "feature_columns_used.csv", index=False)

    threshold_values = pd.to_numeric(data["failure_threshold_ft"], errors="coerce").dropna()
    threshold_ft = float(threshold_values.iloc[0]) if not threshold_values.empty else DEFAULT_FAILURE_THRESHOLD_FT

    train = data[data["ml_split"].eq("train")].copy()
    validation = data[data["ml_split"].eq("validation")].copy()
    test = data[data["ml_split"].eq("test")].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("Train, validation, and test rows are all required.")

    existing_validation, validation_reference = existing_reference_metrics(existing, "validation")
    existing_test, _ = existing_reference_metrics(existing, "test")

    print(f"\nRows: train={len(train):,}, validation={len(validation):,}, test={len(test):,}")
    print(f"Feature columns: {len(feature_cols):,}")
    print(f"Failure threshold: {threshold_ft:.1f} ft")
    print("\nCurrent validation reference:")
    print(existing_validation.round(4).to_string(index=False))

    # -------------------------------------------------------------------------
    # A. Broad family and weighting screening
    # -------------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("A. Screening regression families and training-weight schemes")
    print("=" * 104)
    screening = screening_search(
        train, validation, feature_cols, threshold_ft, validation_reference
    )
    screening = rank_candidates(screening)
    screening.to_csv(OUT_DIR / "screening_validation_metrics.csv", index=False)

    top_families = select_top_families(screening)
    print(f"\nTop families selected for detailed tuning: {top_families}")

    # -------------------------------------------------------------------------
    # B. Hyperparameter tuning of top families
    # -------------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("B. Detailed validation tuning of top correction families")
    print("=" * 104)
    tuned = tuning_search(
        train, validation, feature_cols, threshold_ft,
        validation_reference, top_families,
    )
    tuned = rank_candidates(tuned)
    tuned.to_csv(OUT_DIR / "tuned_validation_metrics.csv", index=False)

    correction_candidates = pd.concat([screening, tuned], ignore_index=True)
    correction_candidates = rank_candidates(correction_candidates)
    regression_specs = best_regression_spec_by_family(correction_candidates)

    nested_ranked = rank_candidates(correction_candidates)
    nested_families = []
    for family in nested_ranked["model_family"].astype(str):
        if family not in nested_families:
            nested_families.append(family)
        if len(nested_families) >= TOP_FAMILIES_FOR_NESTED:
            break
    print(f"Top families selected for nested architectures: {nested_families}")

    # -------------------------------------------------------------------------
    # C. Gated correction models
    # -------------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("C. Gated nested correction models")
    print("=" * 104)
    gated = gated_search(
        train, validation, feature_cols, threshold_ft,
        validation_reference, regression_specs, nested_families,
    )
    gated = rank_candidates(gated)
    gated.to_csv(OUT_DIR / "gated_validation_metrics.csv", index=False)

    # -------------------------------------------------------------------------
    # D. Direction and magnitude models
    # -------------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("D. Direction-and-magnitude nested correction models")
    print("=" * 104)
    direction = direction_magnitude_search(
        train, validation, feature_cols, threshold_ft,
        validation_reference, regression_specs, nested_families,
    )
    direction = rank_candidates(direction)
    direction.to_csv(OUT_DIR / "direction_magnitude_validation_metrics.csv", index=False)

    all_candidates = pd.concat(
        [screening, tuned, gated, direction],
        ignore_index=True,
        sort=False,
    )
    all_candidates = rank_candidates(all_candidates)
    all_candidates.to_csv(OUT_DIR / "candidate_validation_metrics_all.csv", index=False)

    selected = select_candidates_for_test(all_candidates)
    selected.to_csv(OUT_DIR / "selected_models_validation.csv", index=False)

    print("\nSelected validation candidates:")
    display = [
        "selection_scope", "candidate_id", "strategy_group", "model_family",
        "weight_scheme", "residual_mae_ft", "residual_rmse_ft",
        "failure_accuracy", "failure_precision", "failure_recall", "failure_f1",
        "classification_guardrail_pass",
    ]
    display = [c for c in display if c in selected.columns]
    print(selected[display].round(4).to_string(index=False))

    # -------------------------------------------------------------------------
    # Fit only validation-selected candidates and evaluate test data.
    # -------------------------------------------------------------------------
    prediction_parts = []
    for _, row in selected.iterrows():
        spec = parse_candidate_spec(row)
        pred_all, model_bundle = fit_selected_candidate(
            spec, train, data, feature_cols, threshold_ft
        )
        selection_scope = str(row["selection_scope"])
        pred_long = predictions_to_long(
            data=data,
            candidate_id=spec["candidate_id"],
            selection_scope=selection_scope,
            strategy_group=spec["strategy_group"],
            model_family=spec["model_family"],
            residual_pred=pred_all,
            threshold_ft=threshold_ft,
        )
        prediction_parts.append(pred_long)
        joblib.dump(
            model_bundle,
            MODEL_DIR / f"{slugify(selection_scope)}__{slugify(spec['candidate_id'])}.joblib",
        )

    selected_predictions = pd.concat(prediction_parts, ignore_index=True)
    selected_predictions.to_csv(OUT_DIR / "selected_predictions_allrates.csv", index=False)

    selected_test_predictions = selected_predictions[
        selected_predictions["ml_split"].eq("test")
    ].copy()

    selected_test_metrics = metrics_from_prediction_long(
        selected_test_predictions,
        [
            "selection_scope", "candidate_id", "strategy_group", "model_family",
        ],
    )
    selected_test_metrics.to_csv(OUT_DIR / "selected_test_metrics.csv", index=False)

    selected_metrics_by_rate = metrics_from_prediction_long(
        selected_test_predictions,
        [
            "selection_scope", "candidate_id", "strategy_group", "model_family",
            "cv_rate_pct", "cv_rate_group",
        ],
    )
    selected_metrics_by_rate.to_csv(OUT_DIR / "selected_metrics_by_rate.csv", index=False)

    comparison = pd.concat([
        existing_test.assign(selection_scope="existing", split="test"),
        selected_test_metrics.assign(split="test"),
    ], ignore_index=True, sort=False)
    comparison = comparison.sort_values(
        ["residual_rmse_ft", "failure_f1"], ascending=[True, False]
    ).reset_index(drop=True)
    comparison.to_csv(OUT_DIR / "comparison_with_existing_test.csv", index=False)

    overall_id = str(selected[selected["selection_scope"].eq("overall_best")]["candidate_id"].iloc[0])
    top_errors = selected_test_predictions[
        selected_test_predictions["candidate_id"].eq(overall_id)
    ].copy()
    top_errors = top_errors.sort_values("residual_abs_error_ft", ascending=False).head(100)
    top_errors.to_csv(OUT_DIR / "top_test_error_rows.csv", index=False)

    manifest = pd.DataFrame([
        {"item": "profile_curve_ids", "value": ",".join(PROFILE_CURVE_IDS)},
        {"item": "failure_threshold_ft", "value": threshold_ft},
        {"item": "n_train_rows", "value": len(train)},
        {"item": "n_validation_rows", "value": len(validation)},
        {"item": "n_test_rows", "value": len(test)},
        {"item": "n_feature_columns", "value": len(feature_cols)},
        {"item": "n_parameter_samples", "value": N_PARAMETER_SAMPLES},
        {"item": "top_families_for_tuning", "value": ",".join(top_families)},
        {"item": "top_families_for_nested", "value": ",".join(nested_families)},
        {"item": "xgboost_available", "value": XGBOOST_AVAILABLE},
        {"item": "lightgbm_available", "value": LIGHTGBM_AVAILABLE},
        {"item": "catboost_available", "value": CATBOOST_AVAILABLE},
        {"item": "selection_rule", "value": "validation guardrail first, then RMSE, MAE, F1, accuracy"},
    ])
    manifest.to_csv(OUT_DIR / "experiment_manifest.csv", index=False)

    print("\n" + "=" * 104)
    print("TEST COMPARISON")
    print("=" * 104)
    test_display = [
        "selection_scope", "candidate_id", "strategy_group", "model_family",
        "residual_mae_ft", "residual_rmse_ft", "residual_bias_ft",
        "failure_accuracy", "failure_precision", "failure_recall", "failure_f1",
        "tp", "fp", "fn", "tn",
    ]
    test_display = [c for c in test_display if c in comparison.columns]
    print(comparison[test_display].round(4).to_string(index=False))

    print("\nSaved outputs:")
    for filename in [
        "model_availability.csv",
        "feature_columns_used.csv",
        "screening_validation_metrics.csv",
        "tuned_validation_metrics.csv",
        "gated_validation_metrics.csv",
        "direction_magnitude_validation_metrics.csv",
        "candidate_validation_metrics_all.csv",
        "selected_models_validation.csv",
        "selected_predictions_allrates.csv",
        "selected_test_metrics.csv",
        "selected_metrics_by_rate.csv",
        "comparison_with_existing_test.csv",
        "top_test_error_rows.csv",
        "experiment_manifest.csv",
    ]:
        print(f"  {OUT_DIR / filename}")

    print("\nDone.")


if __name__ == "__main__":
    main()
