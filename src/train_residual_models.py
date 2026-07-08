"""
Raw queue-length residual model training for the revised CV queue reconstruction workflow.

Place this file at:
    src/train_residual_models.py

Purpose
-------
Train three global residual-learning models using the saved revised feature file:
    1) XGBoost residual model
    2) GRU residual model
    3) LSTM residual model

Target:
    residual_ft = q_gt_ft - q_baseline_fixed_ft

Important workflow rule
-----------------------
This script trains RAW residual models only. CV anchor correction is NOT applied here.
CV anchor features are excluded from the model input. The predictions saved here will be
used later by a separate CV-anchor adjustment script.

Outputs
-------
    output/intermediate_csv/ml_raw_predictions/
        ml_raw_predictions_allruns_allrates.csv
        ml_raw_predictions_runXXX_rateYYY.csv
        ml_raw_metrics_by_model_run_rate.csv
        ml_raw_metrics_by_model_run.csv
        ml_raw_training_summary.csv
        feature_columns_used.csv
        trained_models/
            xgb_raw_residual_model.joblib
            gru_raw_residual_model.pt
            lstm_raw_residual_model.pt
            nn_feature_scaler.joblib
            nn_target_scaler.joblib

Notes
-----
Because raw residual models do not use CV-anchor features, the model prediction is the
same for a given run/time across all CV penetration rates. The script still saves
predictions for every run/rate row so later anchor-correction and comparison scripts can
read one self-contained table.
"""

from __future__ import annotations

from config import (
    PROJECT_ROOT,
    TRAIN_RUN_IDS,
    TEST_RUN_IDS,
    VALIDATION_RUN_IDS,
    CV_RATES_PCT,
    XGB_RANDOM_SEED,
    XGB_N_ESTIMATORS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE,
    XGB_SUBSAMPLE,
    XGB_COLSAMPLE_BYTREE,
    XGB_REG_LAMBDA,
    NN_RANDOM_SEED,
    NN_SEQUENCE_LEN,
    NN_SEQUENCE_STRIDE,
    NN_HIDDEN_SIZE,
    NN_NUM_LAYERS,
    NN_DROPOUT,
    NN_BATCH_SIZE,
    NN_EPOCHS,
    NN_LEARNING_RATE,
    NN_WEIGHT_DECAY,
    NN_GRAD_CLIP_NORM
)

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    Dataset = object
    DataLoader = None
    TORCH_AVAILABLE = False


# =============================================================================
# Stage-specific constants
# =============================================================================

# Professional repo layout: <project_root>/src/<script>.py

CV_FEATURE_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "cv_features"
FEATURE_FILE = CV_FEATURE_DIR / "timegrid_features_allruns_allrates.csv"

OUT_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "ml_raw_predictions"
MODEL_DIR = OUT_DIR / "trained_models"
FIG_DIR = OUT_DIR / "figures"

# Train/test split by simulation run. Keep this strict to avoid leakage.

# CV rates are included only for saving repeated prediction rows. Raw models do not use CV features.

# Model toggles.
TRAIN_XGBOOST = True
TRAIN_GRU = True
TRAIN_LSTM = True

# XGBoost settings.

# Neural-network settings.

# Use GPU if available.
DEVICE = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

# Save per-run/rate prediction CSVs in addition to the combined file.
SAVE_PER_RUN_RATE_FILES = True

# If True, all raw model predictions are clipped to nonnegative queue length after adding residual to baseline.
CLIP_QUEUE_PREDICTIONS_TO_NONNEGATIVE = True

# Constraint-aware GRU/LSTM training. The supervised target remains the residual
# error, but the loss also evaluates the implied queue length in feet:
#     q_pred = q_baseline_fixed_ft + residual_pred
#
# These values are adopted from the residual sequence diagnostic lambda sweep.
# Selection was based on validation performance and visual behavior:
#     LAMBDA_ZERO_QUEUE_MATCH = 0.40
#     LAMBDA_RESIDUAL_DQ      = 0.12
#     LAMBDA_RESIDUAL_D2Q     = 0.35
USE_PHYSICAL_CONSTRAINT_LOSS = True
USE_SUPERVISED_DYNAMICS_LOSS = True
QUEUE_CONSTRAINT_SCALE_FT = 100.0
LAMBDA_NONNEGATIVE_QUEUE = 0.10
LAMBDA_SUDDEN_DROP = 0.03
LAMBDA_CURVATURE = 0.01
LAMBDA_DQ_MATCH = 0.55
LAMBDA_D2Q_MATCH = 0.18
LAMBDA_WINDOW_PEAK_MATCH = 0.30
LAMBDA_WINDOW_MEAN_MATCH = 0.12
LAMBDA_ZERO_QUEUE_MATCH = 0.40
LAMBDA_RESIDUAL_DQ = 0.12
LAMBDA_RESIDUAL_D2Q = 0.35
MAX_QUEUE_DROP_PER_STEP_FT = 25.0
ZERO_QUEUE_TOL_FT = 15.0
RESIDUAL_CONSTRAINT_SCALE_FT = 75.0

# =============================================================================
# Feature selection
# =============================================================================

# These are observable / baseline features only. CV-anchor context is intentionally excluded.
# =============================================================================
# Feature selection
# =============================================================================

# Physics + ML residual model.
#
# Target:
#     q_gt_ft - q_baseline_fixed_ft
#
# Allowed inputs:
#     - signal phase state
#     - elapsed time within current phase
#     - direct detector counts A and D
#     - physics-derived cumulative-count states V, B, and n_queue
#
# Excluded:
#     - absolute simulation time
#     - normalized simulation time
#     - q_baseline_fixed_ft as direct predictor
#     - slope/delta/normalized redundant features
#     - CV-anchor features
NUMERIC_FEATURE_CANDIDATES = [
    "phase_elapsed_sec",
    "A_count",
    "D_count",
    "V_count",
    "B_count",
    "n_queue_cumulative",
]

CATEGORICAL_FEATURE_CANDIDATES = [
    "phase_state",
]

TARGET_COL = "target_residual_from_baseline_ft"
GT_COL = "q_gt_ft"
BASELINE_Q_COL = "q_baseline_fixed_ft"

ID_COLS_TO_KEEP = [
    "run_id",
    "run_split",
    "ml_split",
    "cv_rate_pct",
    "time_sec",

    # Useful context retained in prediction files.
    "phase_state",
    "phase_elapsed_sec",

    # Targets and baseline.
    "q_gt_ft",
    "q_baseline_fixed_ft",
    "target_residual_from_baseline_ft",

    # Count-state columns retained for diagnostics and downstream transforms.
    "A_count",
    "D_count",
    "V_count",
    "B_count",
    "n_queue_cumulative",
    "l_eff_fixed_ft",
]


# =============================================================================
# Utilities
# =============================================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def finite_metric_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_sec: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Return finite y_true/y_pred arrays, optionally with finite time values.

    This prevents metric computation from failing when a prediction column has
    NaNs, such as CV-only interpolation outside the first/last CV anchor range.
    """
    y = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    if time_sec is None:
        mask = np.isfinite(y) & np.isfinite(yp)
        return y[mask], yp[mask], None

    t = np.asarray(time_sec, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(yp)
    return y[mask], yp[mask], t[mask]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE after removing non-finite values."""
    y, yp, _ = finite_metric_arrays(y_true, y_pred)

    if len(y) == 0:
        return np.nan

    return float(math.sqrt(mean_squared_error(y, yp)))


def safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE after removing non-finite values."""
    y, yp, _ = finite_metric_arrays(y_true, y_pred)

    if len(y) == 0:
        return np.nan

    return float(mean_absolute_error(y, yp))


def max_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute maximum absolute error after removing non-finite values."""
    y, yp, _ = finite_metric_arrays(y_true, y_pred)

    if len(y) == 0:
        return np.nan

    return float(np.max(np.abs(y - yp)))


def area_abs_error(time_sec: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute area between curves after removing non-finite values."""
    y, yp, t = finite_metric_arrays(y_true, y_pred, time_sec=time_sec)

    if t is None or len(t) < 2:
        return np.nan

    abs_err = np.abs(y - yp)
    order = np.argsort(t)

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(abs_err[order], t[order]))

    return float(np.trapz(abs_err[order], t[order]))

def format_rate(rate: int) -> str:
    return f"{int(rate):03d}"


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def load_feature_table() -> pd.DataFrame:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find feature file:\n{FEATURE_FILE}\n"
            "Run src/build_cv_features.py first."
        )

    df = pd.read_csv(FEATURE_FILE)
    require_columns(df, ["run_id", "cv_rate_pct", "time_sec", GT_COL, BASELINE_Q_COL, TARGET_COL], "feature table")

    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce").astype("Int64")
    df["cv_rate_pct"] = pd.to_numeric(df["cv_rate_pct"], errors="coerce").astype("Int64")
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    df[GT_COL] = pd.to_numeric(df[GT_COL], errors="coerce")
    df[BASELINE_Q_COL] = pd.to_numeric(df[BASELINE_Q_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    df = df.dropna(subset=["run_id", "cv_rate_pct", "time_sec", GT_COL, BASELINE_Q_COL, TARGET_COL]).copy()
    df["run_id"] = df["run_id"].astype(int)
    df["cv_rate_pct"] = df["cv_rate_pct"].astype(int)

    # Keep only requested runs/rates if present.
    # Keep only requested train/validation/test runs and requested CV rates.
    model_run_ids = TRAIN_RUN_IDS + VALIDATION_RUN_IDS + TEST_RUN_IDS

    df = df[df["run_id"].isin(model_run_ids)].copy()
    df = df[df["cv_rate_pct"].isin(CV_RATES_PCT)].copy()

# Add explicit ML split label.
    df["ml_split"] = "other"
    df.loc[df["run_id"].isin(TRAIN_RUN_IDS), "ml_split"] = "train"
    df.loc[df["run_id"].isin(VALIDATION_RUN_IDS), "ml_split"] = "validation"
    df.loc[df["run_id"].isin(TEST_RUN_IDS), "ml_split"] = "test"

    df = df.sort_values(["run_id", "cv_rate_pct", "time_sec"]).reset_index(drop=True)
    return df


def build_raw_unique_table(all_features: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate run/time rows because raw models exclude CV features."""
    sort_cols = ["run_id", "time_sec", "cv_rate_pct"]
    raw = all_features.sort_values(sort_cols).copy()
    raw = raw.drop_duplicates(subset=["run_id", "time_sec"], keep="first").copy()
    raw = raw.sort_values(["run_id", "time_sec"]).reset_index(drop=True)
    return raw


def select_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = [c for c in NUMERIC_FEATURE_CANDIDATES if c in df.columns]
    categorical_cols = [c for c in CATEGORICAL_FEATURE_CANDIDATES if c in df.columns]
    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable feature columns were found. Check the feature-engineering output.")
    return numeric_cols, categorical_cols


def make_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_cols,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


# =============================================================================
# XGBoost / tree model
# =============================================================================

def train_xgb_model(raw: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]):
    train = raw[raw["run_id"].isin(TRAIN_RUN_IDS)].copy()
    X_train = train[numeric_cols + categorical_cols].copy()
    y_train = train[TARGET_COL].to_numpy(dtype=float)

    preprocessor = make_preprocessor(numeric_cols, categorical_cols)

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
        model_name = "xgboost"
    else:
        model = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=XGB_RANDOM_SEED,
        )
        model_name = "hist_gradient_boosting_fallback"

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    return pipe, model_name


def predict_xgb(pipe: Pipeline, df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> np.ndarray:
    X = df[numeric_cols + categorical_cols].copy()
    return pipe.predict(X).astype(float)


# =============================================================================
# GRU / LSTM models
# =============================================================================

if TORCH_AVAILABLE:
    class SequenceWindowDataset(Dataset):
        def __init__(self, windows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
            self.windows = windows

        def __len__(self):
            return len(self.windows)

        def __getitem__(self, idx):
            x, y, baseline, q_true = self.windows[idx]
            return (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(baseline, dtype=torch.float32),
                torch.tensor(q_true, dtype=torch.float32),
            )


    class RNNResidualModel(nn.Module):
        def __init__(self, input_dim: int, cell_type: str = "GRU"):
            super().__init__()
            self.cell_type = cell_type.upper()
            if self.cell_type == "GRU":
                self.rnn = nn.GRU(
                    input_size=input_dim,
                    hidden_size=NN_HIDDEN_SIZE,
                    num_layers=NN_NUM_LAYERS,
                    batch_first=True,
                    dropout=NN_DROPOUT if NN_NUM_LAYERS > 1 else 0.0,
                )
            elif self.cell_type == "LSTM":
                self.rnn = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=NN_HIDDEN_SIZE,
                    num_layers=NN_NUM_LAYERS,
                    batch_first=True,
                    dropout=NN_DROPOUT if NN_NUM_LAYERS > 1 else 0.0,
                )
            else:
                raise ValueError("cell_type must be GRU or LSTM")

            self.head = nn.Sequential(
                nn.Linear(NN_HIDDEN_SIZE, NN_HIDDEN_SIZE // 2),
                nn.ReLU(),
                nn.Linear(NN_HIDDEN_SIZE // 2, 1),
            )

        def forward(self, x):
            out, _ = self.rnn(x)
            pred = self.head(out).squeeze(-1)
            return pred


def build_windows_for_runs(
    raw: pd.DataFrame,
    feature_matrix: np.ndarray,
    y_scaled: np.ndarray,
    baseline_ft: np.ndarray,
    q_true_ft: np.ndarray,
    run_ids: list[int],
    sequence_len: int,
    stride: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    windows = []
    raw_index = raw.reset_index(drop=True)

    for run_id in run_ids:
        idx = raw_index.index[raw_index["run_id"] == int(run_id)].to_numpy()
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(raw_index.loc[idx, "time_sec"].to_numpy(dtype=float))]
        X_run = feature_matrix[idx]
        y_run = y_scaled[idx]
        base_run = baseline_ft[idx]
        q_true_run = q_true_ft[idx]

        if len(idx) <= sequence_len:
            windows.append((X_run, y_run, base_run, q_true_run))
            continue

        starts = list(range(0, len(idx) - sequence_len + 1, stride))
        if starts[-1] != len(idx) - sequence_len:
            starts.append(len(idx) - sequence_len)
        for s in starts:
            e = s + sequence_len
            windows.append((X_run[s:e], y_run[s:e], base_run[s:e], q_true_run[s:e]))

    return windows


def physical_queue_constraint_loss(
    pred_scaled,
    baseline_ft,
    target_mean,
    target_scale,
):
    if not USE_PHYSICAL_CONSTRAINT_LOSS:
        return pred_scaled.new_tensor(0.0)

    residual_pred_ft = pred_scaled * target_scale + target_mean
    q_pred_ft = baseline_ft + residual_pred_ft
    denom = float(QUEUE_CONSTRAINT_SCALE_FT)

    nonnegative_penalty = torch.mean((torch.relu(-q_pred_ft) / denom) ** 2)

    if q_pred_ft.shape[1] < 2:
        drop_penalty = q_pred_ft.new_tensor(0.0)
    else:
        dq = q_pred_ft[:, 1:] - q_pred_ft[:, :-1]
        excess_drop = torch.relu((-dq) - float(MAX_QUEUE_DROP_PER_STEP_FT))
        drop_penalty = torch.mean((excess_drop / denom) ** 2)

    if q_pred_ft.shape[1] < 3:
        curvature_penalty = q_pred_ft.new_tensor(0.0)
    else:
        second_diff = q_pred_ft[:, 2:] - 2.0 * q_pred_ft[:, 1:-1] + q_pred_ft[:, :-2]
        curvature_penalty = torch.mean((second_diff / denom) ** 2)

    return (
        LAMBDA_NONNEGATIVE_QUEUE * nonnegative_penalty
        + LAMBDA_SUDDEN_DROP * drop_penalty
        + LAMBDA_CURVATURE * curvature_penalty
    )


def supervised_queue_dynamics_loss(
    pred_scaled,
    baseline_ft,
    q_true_ft,
    target_mean,
    target_scale,
):
    """Match implied queue dynamics so true discharge is preserved."""
    if not USE_SUPERVISED_DYNAMICS_LOSS:
        return pred_scaled.new_tensor(0.0)

    residual_pred_ft = pred_scaled * target_scale + target_mean
    q_pred_ft = baseline_ft + residual_pred_ft
    denom = float(QUEUE_CONSTRAINT_SCALE_FT)

    if q_pred_ft.shape[1] < 2:
        dq_loss = q_pred_ft.new_tensor(0.0)
    else:
        dq_pred = q_pred_ft[:, 1:] - q_pred_ft[:, :-1]
        dq_true = q_true_ft[:, 1:] - q_true_ft[:, :-1]
        dq_loss = torch.mean(((dq_pred - dq_true) / denom) ** 2)

    if q_pred_ft.shape[1] < 3:
        d2q_loss = q_pred_ft.new_tensor(0.0)
    else:
        d2q_pred = q_pred_ft[:, 2:] - 2.0 * q_pred_ft[:, 1:-1] + q_pred_ft[:, :-2]
        d2q_true = q_true_ft[:, 2:] - 2.0 * q_true_ft[:, 1:-1] + q_true_ft[:, :-2]
        d2q_loss = torch.mean(((d2q_pred - d2q_true) / denom) ** 2)

    return LAMBDA_DQ_MATCH * dq_loss + LAMBDA_D2Q_MATCH * d2q_loss


def supervised_queue_shape_loss(
    pred_scaled,
    baseline_ft,
    q_true_ft,
    target_mean,
    target_scale,
):
    """Match coarse implied queue shape without post-hoc smoothing."""
    if not USE_SUPERVISED_DYNAMICS_LOSS:
        return pred_scaled.new_tensor(0.0)

    residual_pred_ft = pred_scaled * target_scale + target_mean
    q_pred_ft = baseline_ft + residual_pred_ft
    denom = float(QUEUE_CONSTRAINT_SCALE_FT)

    pred_peak = torch.amax(q_pred_ft, dim=1)
    true_peak = torch.amax(q_true_ft, dim=1)
    peak_loss = torch.mean(((pred_peak - true_peak) / denom) ** 2)

    pred_mean = torch.mean(q_pred_ft, dim=1)
    true_mean = torch.mean(q_true_ft, dim=1)
    mean_loss = torch.mean(((pred_mean - true_mean) / denom) ** 2)

    near_zero_mask = q_true_ft <= float(ZERO_QUEUE_TOL_FT)
    if torch.any(near_zero_mask):
        false_queue = torch.relu(q_pred_ft[near_zero_mask] - float(ZERO_QUEUE_TOL_FT))
        zero_loss = torch.mean((false_queue / denom) ** 2)
    else:
        zero_loss = q_pred_ft.new_tensor(0.0)

    return (
        LAMBDA_WINDOW_PEAK_MATCH * peak_loss
        + LAMBDA_WINDOW_MEAN_MATCH * mean_loss
        + LAMBDA_ZERO_QUEUE_MATCH * zero_loss
    )


def residual_correction_smoothness_loss(
    pred_scaled,
    target_mean,
    target_scale,
):
    """
    Encourage the residual correction to be low-frequency.

    The cumulative-count representation already carries the main queue
    rise/dissipation pattern. The learned residual should correct systematic
    bias, not create rapid local reversals in the reconstructed queue.
    """
    residual_pred_ft = pred_scaled * target_scale + target_mean
    denom = float(RESIDUAL_CONSTRAINT_SCALE_FT)

    if residual_pred_ft.shape[1] < 2:
        dq_loss = residual_pred_ft.new_tensor(0.0)
    else:
        dq = residual_pred_ft[:, 1:] - residual_pred_ft[:, :-1]
        dq_loss = torch.mean((dq / denom) ** 2)

    if residual_pred_ft.shape[1] < 3:
        d2q_loss = residual_pred_ft.new_tensor(0.0)
    else:
        d2q = residual_pred_ft[:, 2:] - 2.0 * residual_pred_ft[:, 1:-1] + residual_pred_ft[:, :-2]
        d2q_loss = torch.mean((d2q / denom) ** 2)

    return LAMBDA_RESIDUAL_DQ * dq_loss + LAMBDA_RESIDUAL_D2Q * d2q_loss


def train_rnn_model(
    raw: pd.DataFrame,
    feature_matrix_unscaled: np.ndarray,
    y: np.ndarray,
    cell_type: str,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available; cannot train GRU/LSTM.")

    train_mask = raw["run_id"].isin(TRAIN_RUN_IDS).to_numpy()
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(feature_matrix_unscaled[train_mask])
    # Transform all rows using scaler fit on train only.
    X_all_scaled = scaler_X.transform(feature_matrix_unscaled)
    y_train_scaled = scaler_y.fit_transform(y[train_mask].reshape(-1, 1)).ravel()
    y_all_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()
    baseline_all_ft = raw[BASELINE_Q_COL].to_numpy(dtype=float)
    q_true_all_ft = raw[GT_COL].to_numpy(dtype=float)

    # Training and validation are now strictly separated by run.
    train_runs_for_windows = list(TRAIN_RUN_IDS)
    val_runs_for_windows = list(VALIDATION_RUN_IDS)

    train_windows = build_windows_for_runs(raw, X_all_scaled, y_all_scaled, baseline_all_ft, q_true_all_ft, train_runs_for_windows, NN_SEQUENCE_LEN, NN_SEQUENCE_STRIDE)
    val_windows = build_windows_for_runs(raw, X_all_scaled, y_all_scaled, baseline_all_ft, q_true_all_ft, val_runs_for_windows, NN_SEQUENCE_LEN, NN_SEQUENCE_STRIDE)

    train_loader = DataLoader(SequenceWindowDataset(train_windows), batch_size=NN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SequenceWindowDataset(val_windows), batch_size=NN_BATCH_SIZE, shuffle=False) if val_windows else None

    model = RNNResidualModel(input_dim=X_all_scaled.shape[1], cell_type=cell_type).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=NN_LEARNING_RATE, weight_decay=NN_WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    target_mean = torch.tensor(float(scaler_y.mean_[0]), dtype=torch.float32, device=DEVICE)
    target_scale = torch.tensor(float(scaler_y.scale_[0]), dtype=torch.float32, device=DEVICE)

    history = []
    best_state = None
    best_val = float("inf")

    for epoch in range(1, NN_EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb, baseb, qtrueb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            baseb = baseb.to(DEVICE)
            qtrueb = qtrueb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            mse_loss = loss_fn(pred, yb)
            constraint_loss = physical_queue_constraint_loss(pred, baseb, target_mean, target_scale)
            dynamics_loss = supervised_queue_dynamics_loss(pred, baseb, qtrueb, target_mean, target_scale)
            shape_loss = supervised_queue_shape_loss(pred, baseb, qtrueb, target_mean, target_scale)
            residual_smoothness_loss = residual_correction_smoothness_loss(pred, target_mean, target_scale)
            loss = mse_loss + constraint_loss + dynamics_loss + shape_loss + residual_smoothness_loss
            loss.backward()
            if NN_GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), NN_GRAD_CLIP_NORM)
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_loss = np.nan
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb, baseb, qtrueb in val_loader:
                    xb = xb.to(DEVICE)
                    yb = yb.to(DEVICE)
                    baseb = baseb.to(DEVICE)
                    qtrueb = qtrueb.to(DEVICE)
                    pred = model(xb)
                    mse_loss = loss_fn(pred, yb)
                    constraint_loss = physical_queue_constraint_loss(pred, baseb, target_mean, target_scale)
                    dynamics_loss = supervised_queue_dynamics_loss(pred, baseb, qtrueb, target_mean, target_scale)
                    shape_loss = supervised_queue_shape_loss(pred, baseb, qtrueb, target_mean, target_scale)
                    residual_smoothness_loss = residual_correction_smoothness_loss(pred, target_mean, target_scale)
                    loss = mse_loss + constraint_loss + dynamics_loss + shape_loss + residual_smoothness_loss
                    val_losses.append(float(loss.item()))
            val_loss = float(np.mean(val_losses)) if val_losses else np.nan
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append({"epoch": epoch, "cell_type": cell_type, "train_loss_scaled": train_loss, "val_loss_scaled": val_loss})
        if epoch == 1 or epoch % 5 == 0 or epoch == NN_EPOCHS:
            print(f"    {cell_type} epoch {epoch:03d}/{NN_EPOCHS} | train={train_loss:.5f} | val={val_loss:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler_X, scaler_y, pd.DataFrame(history)


def predict_rnn_by_run(
    model,
    raw: pd.DataFrame,
    feature_matrix_unscaled: np.ndarray,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
) -> np.ndarray:
    X_scaled = scaler_X.transform(feature_matrix_unscaled)
    pred_scaled_all = np.full(len(raw), np.nan, dtype=float)
    raw_index = raw.reset_index(drop=True)

    model.eval()
    with torch.no_grad():
        for run_id in sorted(raw_index["run_id"].unique()):
            idx = raw_index.index[raw_index["run_id"] == int(run_id)].to_numpy()
            idx = idx[np.argsort(raw_index.loc[idx, "time_sec"].to_numpy(dtype=float))]
            X_run = torch.tensor(X_scaled[idx], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            pred_scaled = model(X_run).squeeze(0).detach().cpu().numpy()
            pred_scaled_all[idx] = pred_scaled

    pred = scaler_y.inverse_transform(pred_scaled_all.reshape(-1, 1)).ravel()
    return pred.astype(float)


# =============================================================================
# Predictions and metrics
# =============================================================================

def add_queue_prediction_from_residual(df: pd.DataFrame, residual_col: str, out_q_col: str) -> None:
    q = df[BASELINE_Q_COL].to_numpy(dtype=float) + df[residual_col].to_numpy(dtype=float)
    if CLIP_QUEUE_PREDICTIONS_TO_NONNEGATIVE:
        q = np.maximum(q, 0.0)
    df[out_q_col] = q


def compute_metrics_for_table(pred: pd.DataFrame, q_cols: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_rate = []
    rows_run = []

    for (run_id, rate), g in pred.groupby(["run_id", "cv_rate_pct"], sort=True):
        y = g[GT_COL].to_numpy(dtype=float)
        t = g["time_sec"].to_numpy(dtype=float)
        for model_name, q_col in q_cols.items():
            if q_col not in g.columns:
                continue
            yp = g[q_col].to_numpy(dtype=float)
            rows_rate.append({
                "run_id": int(run_id),
                "cv_rate_pct": int(rate),
                "model": model_name,
                "n_samples": int(len(g)),
                "mae_ft": safe_mae(y, yp),
                "rmse_ft": rmse(y, yp),
                "max_abs_error_ft": max_abs_error(y, yp),
                "area_abs_error_ft_s": area_abs_error(t, y, yp),
                "mean_q_gt_ft": float(np.nanmean(y)) if len(y) else np.nan,
                "mean_q_pred_ft": float(np.nanmean(yp)) if len(yp) else np.nan,
            })

    for run_id, g in pred.groupby("run_id", sort=True):
        y = g.drop_duplicates(subset=["time_sec"])[GT_COL].to_numpy(dtype=float)
        unique_g = g.drop_duplicates(subset=["time_sec"]).sort_values("time_sec")
        t = unique_g["time_sec"].to_numpy(dtype=float)
        for model_name, q_col in q_cols.items():
            if q_col not in unique_g.columns:
                continue
            yp = unique_g[q_col].to_numpy(dtype=float)
            rows_run.append({
                "run_id": int(run_id),
                "model": model_name,
                "n_samples": int(len(unique_g)),
                "mae_ft": safe_mae(y, yp),
                "rmse_ft": rmse(y, yp),
                "max_abs_error_ft": max_abs_error(y, yp),
                "area_abs_error_ft_s": area_abs_error(t, y, yp),
                "mean_q_gt_ft": float(np.mean(y)) if len(y) else np.nan,
                "mean_q_pred_ft": float(np.mean(yp)) if len(yp) else np.nan,
            })

    return pd.DataFrame(rows_rate), pd.DataFrame(rows_run)


def make_prediction_table(all_features: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    # Keep all run/rate rows for later anchor correction. Attach raw predictions by run/time.
    pred_cols = [
        "run_id",
        "time_sec",
        "residual_pred_xgb_raw_ft",
        "q_pred_xgb_raw_ft",
        "residual_pred_gru_raw_ft",
        "q_pred_gru_raw_ft",
        "residual_pred_lstm_raw_ft",
        "q_pred_lstm_raw_ft",
    ]
    available_pred_cols = [c for c in pred_cols if c in raw.columns]

    pred = all_features[[c for c in ID_COLS_TO_KEEP if c in all_features.columns]].copy()
    pred = pred.merge(raw[available_pred_cols], on=["run_id", "time_sec"], how="left")

    # Add split label in case feature file did not have it.
    pred["ml_split"] = "other"
    pred.loc[pred["run_id"].isin(TRAIN_RUN_IDS), "ml_split"] = "train"
    pred.loc[pred["run_id"].isin(VALIDATION_RUN_IDS), "ml_split"] = "validation"
    pred.loc[pred["run_id"].isin(TEST_RUN_IDS), "ml_split"] = "test"
    return pred.sort_values(["run_id", "cv_rate_pct", "time_sec"]).reset_index(drop=True)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_all_seeds(NN_RANDOM_SEED)

    print("=" * 96)
    print("Raw residual ML training for revised queue-length reconstruction")
    print("=" * 96)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Feature file : {FEATURE_FILE}")
    print(f"Output dir   : {OUT_DIR}")
    print(f"Train runs      : {TRAIN_RUN_IDS}")
    print(f"Validation runs : {VALIDATION_RUN_IDS}")
    print(f"Test runs       : {TEST_RUN_IDS}")
    print(f"CV rates     : {CV_RATES_PCT}")
    print(f"XGBoost available: {XGBOOST_AVAILABLE}")
    print(f"PyTorch available: {TORCH_AVAILABLE} | device={DEVICE}")
    print("=" * 96)

    all_features = load_feature_table()
    raw = build_raw_unique_table(all_features)
    numeric_cols, categorical_cols = select_feature_columns(raw)

    print(f"Loaded feature rows all rates : {len(all_features):,}")
    print(f"Unique raw time-grid rows     : {len(raw):,}")
    print(f"Numeric features              : {len(numeric_cols)}")
    print(f"Categorical features          : {len(categorical_cols)}")

    # Save feature column list.
    feature_col_df = pd.DataFrame({
        "feature_name": numeric_cols + categorical_cols,
        "feature_type": ["numeric"] * len(numeric_cols) + ["categorical"] * len(categorical_cols),
    })
    feature_col_df.to_csv(OUT_DIR / "feature_columns_used.csv", index=False)

    # XGBoost / fallback tree model.
    training_summary = []
    histories = []

    if TRAIN_XGBOOST:
        print("\n[Training XGBoost/raw tree residual model]")
        xgb_pipe, xgb_model_name = train_xgb_model(raw, numeric_cols, categorical_cols)
        raw["residual_pred_xgb_raw_ft"] = predict_xgb(xgb_pipe, raw, numeric_cols, categorical_cols)
        add_queue_prediction_from_residual(raw, "residual_pred_xgb_raw_ft", "q_pred_xgb_raw_ft")
        joblib.dump(xgb_pipe, MODEL_DIR / "xgb_raw_residual_model.joblib")
        training_summary.append({"model": "xgb_raw", "backend": xgb_model_name, "status": "trained"})
        print(f"  [Saved] {MODEL_DIR / 'xgb_raw_residual_model.joblib'}")

    # Prepare transformed feature matrix for RNN using same preprocessing but one-hot encoded.
    # Fit preprocessing on training rows only to avoid leakage.
    if TORCH_AVAILABLE and (TRAIN_GRU or TRAIN_LSTM):
        print("\n[Preparing NN feature matrix]")
        nn_preprocessor = make_preprocessor(numeric_cols, categorical_cols)
        train_mask = raw["run_id"].isin(TRAIN_RUN_IDS).to_numpy()
        nn_preprocessor.fit(raw.loc[train_mask, numeric_cols + categorical_cols])
        X_all = nn_preprocessor.transform(raw[numeric_cols + categorical_cols]).astype(float)
        y_all = raw[TARGET_COL].to_numpy(dtype=float)
        joblib.dump(nn_preprocessor, MODEL_DIR / "nn_raw_feature_preprocessor.joblib")

        nn_feature_names = get_feature_names(nn_preprocessor)
        if nn_feature_names:
            pd.DataFrame({"nn_feature_name": nn_feature_names}).to_csv(OUT_DIR / "nn_encoded_feature_columns.csv", index=False)
    else:
        X_all = None
        y_all = None

    if TRAIN_GRU:
        if not TORCH_AVAILABLE:
            print("\n[Skipping GRU] PyTorch not available.")
            training_summary.append({"model": "gru_raw", "backend": "pytorch", "status": "skipped_no_torch"})
        else:
            print("\n[Training GRU residual model]")
            gru_model, gru_scaler_X, gru_scaler_y, gru_hist = train_rnn_model(raw, X_all, y_all, "GRU")
            raw["residual_pred_gru_raw_ft"] = predict_rnn_by_run(gru_model, raw, X_all, gru_scaler_X, gru_scaler_y)
            add_queue_prediction_from_residual(raw, "residual_pred_gru_raw_ft", "q_pred_gru_raw_ft")
            torch.save({
                "model_state_dict": gru_model.state_dict(),
                "input_dim": int(X_all.shape[1]),
                "cell_type": "GRU",
                "hidden_size": NN_HIDDEN_SIZE,
                "num_layers": NN_NUM_LAYERS,
                "dropout": NN_DROPOUT,
            }, MODEL_DIR / "gru_raw_residual_model.pt")
            joblib.dump(gru_scaler_X, MODEL_DIR / "gru_feature_scaler.joblib")
            joblib.dump(gru_scaler_y, MODEL_DIR / "gru_target_scaler.joblib")
            histories.append(gru_hist)
            training_summary.append({"model": "gru_raw", "backend": "pytorch", "status": "trained"})
            print(f"  [Saved] {MODEL_DIR / 'gru_raw_residual_model.pt'}")

    if TRAIN_LSTM:
        if not TORCH_AVAILABLE:
            print("\n[Skipping LSTM] PyTorch not available.")
            training_summary.append({"model": "lstm_raw", "backend": "pytorch", "status": "skipped_no_torch"})
        else:
            print("\n[Training LSTM residual model]")
            lstm_model, lstm_scaler_X, lstm_scaler_y, lstm_hist = train_rnn_model(raw, X_all, y_all, "LSTM")
            raw["residual_pred_lstm_raw_ft"] = predict_rnn_by_run(lstm_model, raw, X_all, lstm_scaler_X, lstm_scaler_y)
            add_queue_prediction_from_residual(raw, "residual_pred_lstm_raw_ft", "q_pred_lstm_raw_ft")
            torch.save({
                "model_state_dict": lstm_model.state_dict(),
                "input_dim": int(X_all.shape[1]),
                "cell_type": "LSTM",
                "hidden_size": NN_HIDDEN_SIZE,
                "num_layers": NN_NUM_LAYERS,
                "dropout": NN_DROPOUT,
            }, MODEL_DIR / "lstm_raw_residual_model.pt")
            joblib.dump(lstm_scaler_X, MODEL_DIR / "lstm_feature_scaler.joblib")
            joblib.dump(lstm_scaler_y, MODEL_DIR / "lstm_target_scaler.joblib")
            histories.append(lstm_hist)
            training_summary.append({"model": "lstm_raw", "backend": "pytorch", "status": "trained"})
            print(f"  [Saved] {MODEL_DIR / 'lstm_raw_residual_model.pt'}")

    # Save training history and summaries.
    if histories:
        hist_all = pd.concat(histories, ignore_index=True)
        hist_all.to_csv(OUT_DIR / "nn_training_history.csv", index=False)
        print(f"[Saved] {OUT_DIR / 'nn_training_history.csv'}")

    summary_df = pd.DataFrame(training_summary)
    summary_df["train_runs"] = ",".join([f"{r:03d}" for r in TRAIN_RUN_IDS])
    summary_df["validation_runs"] = ",".join([f"{r:03d}" for r in VALIDATION_RUN_IDS])
    summary_df["test_runs"] = ",".join([f"{r:03d}" for r in TEST_RUN_IDS])
    summary_df["target"] = TARGET_COL
    summary_df["raw_model_uses_cv_anchor_features"] = False
    summary_df.to_csv(OUT_DIR / "ml_raw_training_summary.csv", index=False)

    # Build prediction table repeated for all CV rates.
    pred = make_prediction_table(all_features, raw)

    # Save combined prediction file.
    pred_all_path = OUT_DIR / "ml_raw_predictions_allruns_allrates.csv"
    pred.to_csv(pred_all_path, index=False)
    print(f"\n[Saved combined predictions] {pred_all_path}")

    # Save per-run/rate prediction files.
    if SAVE_PER_RUN_RATE_FILES:
        for (run_id, rate), g in pred.groupby(["run_id", "cv_rate_pct"], sort=True):
            out_path = OUT_DIR / f"ml_raw_predictions_run{int(run_id):03d}_rate{format_rate(int(rate))}.csv"
            g.to_csv(out_path, index=False)
        print(f"[Saved per-run/rate prediction files] {OUT_DIR}")

    # Metrics.
    q_cols = {"baseline_fixed": BASELINE_Q_COL}
    if "q_pred_xgb_raw_ft" in pred.columns:
        q_cols["xgb_raw"] = "q_pred_xgb_raw_ft"
    if "q_pred_gru_raw_ft" in pred.columns:
        q_cols["gru_raw"] = "q_pred_gru_raw_ft"
    if "q_pred_lstm_raw_ft" in pred.columns:
        q_cols["lstm_raw"] = "q_pred_lstm_raw_ft"
    

    metrics_rate, metrics_run = compute_metrics_for_table(pred, q_cols)
    metrics_rate_path = OUT_DIR / "ml_raw_metrics_by_model_run_rate.csv"
    metrics_run_path = OUT_DIR / "ml_raw_metrics_by_model_run.csv"
    metrics_rate.to_csv(metrics_rate_path, index=False)
    metrics_run.to_csv(metrics_run_path, index=False)
    print(f"[Saved metrics] {metrics_rate_path}")
    print(f"[Saved metrics] {metrics_run_path}")

    # Quick test-run summary printed to terminal.
    test_metrics = metrics_run[metrics_run["run_id"].isin(TEST_RUN_IDS)].copy()
    if not test_metrics.empty:
        print("\nTest-run summary by model:")
        print(
            test_metrics.groupby("model")[["mae_ft", "rmse_ft", "max_abs_error_ft"]]
            .mean()
            .sort_values("rmse_ft")
            .round(3)
            .to_string()
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
