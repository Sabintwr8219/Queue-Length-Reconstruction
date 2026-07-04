"""
Direct ML queue-length model training.

Place this file at:
    src/train_ml_direct_models.py

Purpose
-------
Train direct queue-length prediction models in queue-length space:

1) ML-only
   Target:
       q_gt_ft
   Features:
       phase_elapsed_sec
       phase_state
       A_count
       D_count

2) ML + CV
   Target:
       q_gt_ft
   Features:
       phase_elapsed_sec
       phase_state
       A_count
       D_count
       practical CV anchor/segment features

Important modeling rule
-----------------------
This script does NOT train residuals.

All targets here are queue length in feet:
    q_gt_ft

Physics-derived variables V, B, n_queue are excluded from ML-only and ML+CV.
They will be used only in Physics + ML residual models.

Outputs
-------
    output/intermediate_csv/ml_direct_predictions/
        ml_direct_predictions_allruns_allrates.csv
        ml_direct_predictions_runXXX_rateYYY.csv
        ml_direct_metrics_by_model_run_rate.csv
        ml_direct_metrics_by_model_run.csv
        ml_direct_training_summary.csv
        feature_columns_used_ml_direct.csv
        nn_training_history_ml_direct.csv
        trained_models/
            xgb_ml_only_model.joblib
            gru_ml_only_model.pt
            lstm_ml_only_model.pt
            xgb_ml_cv_model.joblib
            gru_ml_cv_model.pt
            lstm_ml_cv_model.pt
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
    NN_GRAD_CLIP_NORM,
)

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# Paths
# =============================================================================

CV_FEATURE_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "cv_features"
FEATURE_FILE = CV_FEATURE_DIR / "timegrid_features_allruns_allrates.csv"

OUT_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "ml_only_sequence_diagnostics"
MODEL_DIR = OUT_DIR / "trained_models"
FIG_DIR = OUT_DIR / "figures"


# =============================================================================
# Run settings
# =============================================================================

# Protect against accidental leakage if VALIDATION_RUN_IDS is also present in TRAIN_RUN_IDS.
MODEL_TRAIN_RUN_IDS = [r for r in TRAIN_RUN_IDS if r not in VALIDATION_RUN_IDS]
if not MODEL_TRAIN_RUN_IDS:
    MODEL_TRAIN_RUN_IDS = list(TRAIN_RUN_IDS)

MODEL_VALIDATION_RUN_IDS = list(VALIDATION_RUN_IDS)
MODEL_TEST_RUN_IDS = list(TEST_RUN_IDS)

TRAIN_XGBOOST = False
TRAIN_GRU = True
TRAIN_LSTM = True

DEVICE = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

SAVE_PER_RUN_RATE_FILES = True
CLIP_QUEUE_PREDICTIONS_TO_NONNEGATIVE = True

# Constraint-aware GRU/LSTM training. Direct models predict queue length in feet,
# so the physical penalties are applied to the inverse-scaled prediction itself.
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
LAMBDA_ZERO_QUEUE_MATCH = 0.45
MAX_QUEUE_DROP_PER_STEP_FT = 25.0
ZERO_QUEUE_TOL_FT = 15.0

DIAGNOSTIC_RUN_IDS = list(TEST_RUN_IDS[:1]) if TEST_RUN_IDS else []
DIAGNOSTIC_CV_RATE_PCT = 10
DIAGNOSTIC_MAX_CYCLES = 12


# =============================================================================
# Feature sets
# =============================================================================

TARGET_COL = "q_gt_ft"

# ML-only means data-driven queue prediction without physics-derived states
# and without CV anchor information.
ML_ONLY_NUMERIC_FEATURES = [
    "phase_elapsed_sec",
    "cycle_elapsed_sec",
    "cycle_frac",
    "cycle_duration_sec",
    "A_since_cycle_start",
    "D_since_cycle_start",
    "net_count_since_cycle_start",
    "arrival_count_last_10s",
    "departure_count_last_10s",
    "net_count_last_10s",
    "arrival_count_last_30s",
    "departure_count_last_30s",
    "net_count_last_30s",
]

ML_ONLY_CATEGORICAL_FEATURES = [
    "phase_state",
]

# ML + CV keeps the same directly observable detector/signal inputs and adds
# practical CV anchor/segment context.
#
# We intentionally avoid:
#   - absolute simulation time
#   - prev/next absolute CV anchor times
#   - order IDs and segment IDs
#   - baseline residual features
#   - V, B, n_queue physics states
#   - direct interpolated CV queue-value features
ML_CV_NUMERIC_FEATURES = [
    "phase_elapsed_sec",
    "A_count",
    "D_count",
    "inside_cv_segment",
    "prev_cv_anchor_q_ft",
    "next_cv_anchor_q_ft",
    "time_since_prev_cv_sec",
    "time_to_next_cv_sec",
    "cv_segment_duration_sec",
    "cv_segment_frac",
]

ML_CV_CATEGORICAL_FEATURES = [
    "phase_state",
]

ID_COLS_TO_KEEP = [
    "run_id",
    "run_split",
    "ml_split",
    "cv_rate_pct",
    "time_sec",

    # Signal and direct detector context.
    "phase_state",
    "phase_elapsed_sec",
    "A_count",
    "D_count",

    # GT target.
    "q_gt_ft",

    # Baseline and physics states retained for downstream comparison only.
    "q_baseline_fixed_ft",
    "target_residual_from_baseline_ft",
    "V_count",
    "B_count",
    "n_queue_cumulative",
    "l_eff_fixed_ft",

    # CV context retained for downstream comparison/correction only.
    "inside_cv_segment",
    "prev_cv_anchor_q_ft",
    "next_cv_anchor_q_ft",
    "time_since_prev_cv_sec",
    "time_to_next_cv_sec",
    "cv_segment_duration_sec",
    "cv_segment_frac",
]


# =============================================================================
# Dataclass
# =============================================================================

@dataclass
class ModelFamily:
    family_name: str
    prediction_prefix: str
    numeric_features: list[str]
    categorical_features: list[str]
    uses_cv_features: bool
    deduplicate_run_time: bool
    sequence_group_cols: list[str]


MODEL_FAMILIES = [
    ModelFamily(
        family_name="ml_only",
        prediction_prefix="ml_only",
        numeric_features=ML_ONLY_NUMERIC_FEATURES,
        categorical_features=ML_ONLY_CATEGORICAL_FEATURES,
        uses_cv_features=False,
        deduplicate_run_time=True,
        sequence_group_cols=["run_id"],
    ),
]


# =============================================================================
# Utility functions
# =============================================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def format_rate(rate: int) -> str:
    return f"{int(rate):03d}"


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def finite_metric_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_sec: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    y = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    if time_sec is None:
        mask = np.isfinite(y) & np.isfinite(yp)
        return y[mask], yp[mask], None

    t = np.asarray(time_sec, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(yp)
    return y[mask], yp[mask], t[mask]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y, yp, _ = finite_metric_arrays(y_true, y_pred)
    if len(y) == 0:
        return np.nan
    return float(math.sqrt(mean_squared_error(y, yp)))


def safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y, yp, _ = finite_metric_arrays(y_true, y_pred)
    if len(y) == 0:
        return np.nan
    return float(mean_absolute_error(y, yp))


def max_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y, yp, _ = finite_metric_arrays(y_true, y_pred)
    if len(y) == 0:
        return np.nan
    return float(np.max(np.abs(y - yp)))


def area_abs_error(time_sec: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y, yp, t = finite_metric_arrays(y_true, y_pred, time_sec=time_sec)
    if t is None or len(t) < 2:
        return np.nan

    order = np.argsort(t)
    abs_err = np.abs(y - yp)

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(abs_err[order], t[order]))

    return float(np.trapz(abs_err[order], t[order]))


# =============================================================================
# Loading and preprocessing
# =============================================================================

def load_feature_table() -> pd.DataFrame:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find feature file:\n{FEATURE_FILE}\n"
            "Run src/build_cv_features.py first."
        )

    df = pd.read_csv(FEATURE_FILE)

    required = [
        "run_id",
        "cv_rate_pct",
        "time_sec",
        TARGET_COL,
        "phase_state",
        "phase_elapsed_sec",
        "A_count",
        "D_count",
    ]
    require_columns(df, required, "feature table")

    numeric_basic = [
        "run_id",
        "cv_rate_pct",
        "time_sec",
        TARGET_COL,
        "phase_elapsed_sec",
        "A_count",
        "D_count",
    ]

    for col in numeric_basic:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["run_id", "cv_rate_pct", "time_sec", TARGET_COL]).copy()
    df["run_id"] = df["run_id"].astype(int)
    df["cv_rate_pct"] = df["cv_rate_pct"].astype(int)

    all_model_runs = sorted(set(MODEL_TRAIN_RUN_IDS + MODEL_VALIDATION_RUN_IDS + MODEL_TEST_RUN_IDS))

    df = df[df["run_id"].isin(all_model_runs)].copy()
    df = df[df["cv_rate_pct"].isin(CV_RATES_PCT)].copy()

    df["phase_state"] = df["phase_state"].astype(str).str.strip().str.lower()

    df["ml_split"] = "other"
    df.loc[df["run_id"].isin(MODEL_TRAIN_RUN_IDS), "ml_split"] = "train"
    df.loc[df["run_id"].isin(MODEL_VALIDATION_RUN_IDS), "ml_split"] = "validation"
    df.loc[df["run_id"].isin(MODEL_TEST_RUN_IDS), "ml_split"] = "test"

    df = df.sort_values(["run_id", "cv_rate_pct", "time_sec"]).reset_index(drop=True)
    return df


def add_signal_cycle_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add count-context features that are still ML-only inputs.

    These are derived from signal phase and observable cumulative counts. They
    avoid physics-derived queue states, but give the sequence model local cycle
    context so it does not have to infer every queue pulse from absolute A/D
    count levels.
    """
    work = df.sort_values(["run_id", "cv_rate_pct", "time_sec"]).copy()
    parts = []

    for _, g in work.groupby(["run_id", "cv_rate_pct"], sort=True):
        g = g.sort_values("time_sec").copy()
        phase = g["phase_state"].astype(str).str.lower()
        time = pd.to_numeric(g["time_sec"], errors="coerce").astype(float)
        a_count = pd.to_numeric(g["A_count"], errors="coerce").astype(float)
        d_count = pd.to_numeric(g["D_count"], errors="coerce").astype(float)

        red_start = (phase.eq("red") & ~phase.shift(1, fill_value="").eq("red")).astype(int)
        cycle_id = red_start.cumsum()
        if len(cycle_id) and int(cycle_id.iloc[0]) == 0:
            cycle_id = cycle_id + 1

        g["cycle_id_signal"] = cycle_id.astype(int)
        cycle_start_time = time.groupby(cycle_id).transform("min")
        cycle_end_time = time.groupby(cycle_id).transform("max")
        cycle_duration = (cycle_end_time - cycle_start_time).replace(0.0, np.nan)

        g["cycle_elapsed_sec"] = (time - cycle_start_time).clip(lower=0.0)
        g["cycle_duration_sec"] = cycle_duration.fillna(cycle_duration.median()).fillna(1.0)
        g["cycle_frac"] = (g["cycle_elapsed_sec"] / g["cycle_duration_sec"]).clip(0.0, 1.0)

        a_start = a_count.groupby(cycle_id).transform("first")
        d_start = d_count.groupby(cycle_id).transform("first")
        g["A_since_cycle_start"] = (a_count - a_start).clip(lower=0.0)
        g["D_since_cycle_start"] = (d_count - d_start).clip(lower=0.0)
        g["net_count_since_cycle_start"] = g["A_since_cycle_start"] - g["D_since_cycle_start"]

        for lag in [10, 30]:
            a_lag = a_count.shift(lag).fillna(a_count.iloc[0])
            d_lag = d_count.shift(lag).fillna(d_count.iloc[0])
            g[f"arrival_count_last_{lag}s"] = (a_count - a_lag).clip(lower=0.0)
            g[f"departure_count_last_{lag}s"] = (d_count - d_lag).clip(lower=0.0)
            g[f"net_count_last_{lag}s"] = g[f"arrival_count_last_{lag}s"] - g[f"departure_count_last_{lag}s"]

        parts.append(g)

    return pd.concat(parts, ignore_index=True).sort_values(["run_id", "cv_rate_pct", "time_sec"]).reset_index(drop=True)


def prepare_family_table(all_features: pd.DataFrame, family: ModelFamily) -> pd.DataFrame:
    """
    Prepare training table for one model family.

    ML-only removes duplicate CV-rate rows because its features do not depend on CV rate.
    ML+CV keeps every run/rate row because CV features vary by rate.
    """
    df = add_signal_cycle_context_features(all_features)

    if family.deduplicate_run_time:
        df = (
            df.sort_values(["run_id", "time_sec", "cv_rate_pct"])
            .drop_duplicates(subset=["run_id", "time_sec"], keep="first")
            .sort_values(["run_id", "time_sec"])
            .reset_index(drop=True)
        )
    else:
        df = df.sort_values(["run_id", "cv_rate_pct", "time_sec"]).reset_index(drop=True)

    return df


def select_feature_columns(df: pd.DataFrame, family: ModelFamily) -> tuple[list[str], list[str]]:
    numeric_cols = [c for c in family.numeric_features if c in df.columns]
    categorical_cols = [c for c in family.categorical_features if c in df.columns]

    required = family.numeric_features + family.categorical_features
    missing_required = [c for c in required if c not in df.columns]

    if missing_required:
        raise ValueError(
            f"{family.family_name} missing required feature columns: {missing_required}\n"
            "If phase_elapsed_sec is missing, rerun src/build_cv_features.py."
        )

    if not numeric_cols and not categorical_cols:
        raise ValueError(f"No usable feature columns found for {family.family_name}.")

    return numeric_cols, categorical_cols


def make_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_onehot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


# =============================================================================
# XGBoost / tree model
# =============================================================================

def train_xgb_model(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
):
    train = df[df["run_id"].isin(MODEL_TRAIN_RUN_IDS)].copy()

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
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)
    return pipe, backend


def predict_xgb(
    pipe: Pipeline,
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> np.ndarray:
    X = df[numeric_cols + categorical_cols].copy()
    pred = pipe.predict(X).astype(float)

    if CLIP_QUEUE_PREDICTIONS_TO_NONNEGATIVE:
        pred = np.maximum(pred, 0.0)

    return pred


# =============================================================================
# GRU / LSTM
# =============================================================================

if TORCH_AVAILABLE:

    class SequenceWindowDataset(Dataset):
        def __init__(self, windows: list[tuple[np.ndarray, np.ndarray]]):
            self.windows = windows

        def __len__(self):
            return len(self.windows)

        def __getitem__(self, idx):
            x, y = self.windows[idx]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


    class RNNDirectQueueModel(nn.Module):
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


def build_windows_for_groups(
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    y_scaled: np.ndarray,
    run_ids: list[int],
    group_cols: list[str],
    sequence_len: int,
    stride: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    windows = []
    work = df.reset_index(drop=True)
    work = work[work["run_id"].isin(run_ids)].copy()

    if work.empty:
        return windows

    for _, g in work.groupby(group_cols, sort=True):
        idx = g.index.to_numpy()
        idx = idx[np.argsort(work.loc[idx, "time_sec"].to_numpy(dtype=float))]

        X_group = feature_matrix[idx]
        y_group = y_scaled[idx]

        if len(idx) <= sequence_len:
            windows.append((X_group, y_group))
            continue

        starts = list(range(0, len(idx) - sequence_len + 1, stride))
        if starts[-1] != len(idx) - sequence_len:
            starts.append(len(idx) - sequence_len)

        for s in starts:
            e = s + sequence_len
            windows.append((X_group[s:e], y_group[s:e]))

    return windows


def physical_queue_constraint_loss(
    pred_scaled,
    target_mean,
    target_scale,
):
    if not USE_PHYSICAL_CONSTRAINT_LOSS:
        return pred_scaled.new_tensor(0.0)

    q_pred_ft = pred_scaled * target_scale + target_mean
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
    true_scaled,
    target_mean,
    target_scale,
):
    """Match queue dynamics so true discharge is preserved but artificial wiggles are penalized."""
    if not USE_SUPERVISED_DYNAMICS_LOSS:
        return pred_scaled.new_tensor(0.0)

    q_pred_ft = pred_scaled * target_scale + target_mean
    q_true_ft = true_scaled * target_scale + target_mean
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
    true_scaled,
    target_mean,
    target_scale,
):
    """Match coarse queue shape so the network does not invent cycle pulses."""
    if not USE_SUPERVISED_DYNAMICS_LOSS:
        return pred_scaled.new_tensor(0.0)

    q_pred_ft = pred_scaled * target_scale + target_mean
    q_true_ft = true_scaled * target_scale + target_mean
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


def train_rnn_model(
    df: pd.DataFrame,
    feature_matrix_unscaled: np.ndarray,
    y: np.ndarray,
    family: ModelFamily,
    cell_type: str,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available; cannot train GRU/LSTM.")

    train_mask = df["run_id"].isin(MODEL_TRAIN_RUN_IDS).to_numpy()

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    scaler_X.fit(feature_matrix_unscaled[train_mask])
    X_all_scaled = scaler_X.transform(feature_matrix_unscaled)

    scaler_y.fit(y[train_mask].reshape(-1, 1))
    y_all_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()

    train_windows = build_windows_for_groups(
        df=df,
        feature_matrix=X_all_scaled,
        y_scaled=y_all_scaled,
        run_ids=MODEL_TRAIN_RUN_IDS,
        group_cols=family.sequence_group_cols,
        sequence_len=NN_SEQUENCE_LEN,
        stride=NN_SEQUENCE_STRIDE,
    )

    val_windows = build_windows_for_groups(
        df=df,
        feature_matrix=X_all_scaled,
        y_scaled=y_all_scaled,
        run_ids=MODEL_VALIDATION_RUN_IDS,
        group_cols=family.sequence_group_cols,
        sequence_len=NN_SEQUENCE_LEN,
        stride=NN_SEQUENCE_STRIDE,
    )

    if not train_windows:
        raise ValueError(f"No training windows created for {family.family_name} {cell_type}.")

    train_loader = DataLoader(
        SequenceWindowDataset(train_windows),
        batch_size=NN_BATCH_SIZE,
        shuffle=True,
    )

    val_loader = (
        DataLoader(
            SequenceWindowDataset(val_windows),
            batch_size=NN_BATCH_SIZE,
            shuffle=False,
        )
        if val_windows
        else None
    )

    model = RNNDirectQueueModel(
        input_dim=X_all_scaled.shape[1],
        cell_type=cell_type,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=NN_LEARNING_RATE,
        weight_decay=NN_WEIGHT_DECAY,
    )

    loss_fn = nn.MSELoss()
    target_mean = torch.tensor(float(scaler_y.mean_[0]), dtype=torch.float32, device=DEVICE)
    target_scale = torch.tensor(float(scaler_y.scale_[0]), dtype=torch.float32, device=DEVICE)

    history = []
    best_state = None
    best_val = float("inf")

    for epoch in range(1, NN_EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            mse_loss = loss_fn(pred, yb)
            constraint_loss = physical_queue_constraint_loss(pred, target_mean, target_scale)
            dynamics_loss = supervised_queue_dynamics_loss(pred, yb, target_mean, target_scale)
            shape_loss = supervised_queue_shape_loss(pred, yb, target_mean, target_scale)
            loss = mse_loss + constraint_loss + dynamics_loss + shape_loss
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
                for xb, yb in val_loader:
                    xb = xb.to(DEVICE)
                    yb = yb.to(DEVICE)
                    pred = model(xb)
                    mse_loss = loss_fn(pred, yb)
                    constraint_loss = physical_queue_constraint_loss(pred, target_mean, target_scale)
                    dynamics_loss = supervised_queue_dynamics_loss(pred, yb, target_mean, target_scale)
                    shape_loss = supervised_queue_shape_loss(pred, yb, target_mean, target_scale)
                    loss = mse_loss + constraint_loss + dynamics_loss + shape_loss
                    val_losses.append(float(loss.item()))

            val_loss = float(np.mean(val_losses)) if val_losses else np.nan

            if np.isfinite(val_loss) and val_loss < best_val:
                best_val = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
        else:
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        history.append(
            {
                "epoch": epoch,
                "family": family.family_name,
                "cell_type": cell_type,
                "train_loss_scaled": train_loss,
                "val_loss_scaled": val_loss,
            }
        )

        if epoch == 1 or epoch % 5 == 0 or epoch == NN_EPOCHS:
            print(
                f"    {family.family_name} {cell_type} "
                f"epoch {epoch:03d}/{NN_EPOCHS} | "
                f"train={train_loss:.5f} | val={val_loss:.5f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler_X, scaler_y, pd.DataFrame(history)


def predict_rnn_by_group(
    model,
    df: pd.DataFrame,
    feature_matrix_unscaled: np.ndarray,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    group_cols: list[str],
) -> np.ndarray:
    X_scaled = scaler_X.transform(feature_matrix_unscaled)

    pred_scaled_all = np.full(len(df), np.nan, dtype=float)
    work = df.reset_index(drop=True)

    model.eval()

    with torch.no_grad():
        for _, g in work.groupby(group_cols, sort=True):
            idx = g.index.to_numpy()
            idx = idx[np.argsort(work.loc[idx, "time_sec"].to_numpy(dtype=float))]

            X_group = torch.tensor(
                X_scaled[idx],
                dtype=torch.float32,
                device=DEVICE,
            ).unsqueeze(0)

            pred_scaled = model(X_group).squeeze(0).detach().cpu().numpy()
            pred_scaled_all[idx] = pred_scaled

    pred = scaler_y.inverse_transform(pred_scaled_all.reshape(-1, 1)).ravel()

    if CLIP_QUEUE_PREDICTIONS_TO_NONNEGATIVE:
        pred = np.maximum(pred, 0.0)

    return pred.astype(float)


# =============================================================================
# Family training
# =============================================================================

def train_one_family(
    all_features: pd.DataFrame,
    family: ModelFamily,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    print("\n" + "=" * 96)
    print(f"Training family: {family.family_name}")
    print("=" * 96)

    df = prepare_family_table(all_features, family)
    numeric_cols, categorical_cols = select_feature_columns(df, family)

    print(f"Rows used for family      : {len(df):,}")
    print(f"Uses CV features          : {family.uses_cv_features}")
    print(f"Deduplicated run/time     : {family.deduplicate_run_time}")
    print(f"Numeric features          : {numeric_cols}")
    print(f"Categorical features      : {categorical_cols}")

    pred_cols_created = []
    summary_rows = []
    histories = []

    # Save feature list for this family.
    feature_rows = []
    for c in numeric_cols:
        feature_rows.append(
            {
                "family": family.family_name,
                "feature_name": c,
                "feature_type": "numeric",
                "used_as_model_input": True,
            }
        )
    for c in categorical_cols:
        feature_rows.append(
            {
                "family": family.family_name,
                "feature_name": c,
                "feature_type": "categorical",
                "used_as_model_input": True,
            }
        )

    feature_df = pd.DataFrame(feature_rows)

    # -------------------------------------------------------------------------
    # XGBoost / tree model
    # -------------------------------------------------------------------------
    if TRAIN_XGBOOST:
        print(f"\n[Training XGBoost direct model: {family.family_name}]")
        xgb_pipe, backend = train_xgb_model(df, numeric_cols, categorical_cols)

        q_col = f"q_pred_xgb_{family.prediction_prefix}_ft"
        df[q_col] = predict_xgb(xgb_pipe, df, numeric_cols, categorical_cols)
        pred_cols_created.append(q_col)

        model_path = MODEL_DIR / f"xgb_{family.prediction_prefix}_model.joblib"
        joblib.dump(xgb_pipe, model_path)

        summary_rows.append(
            {
                "family": family.family_name,
                "model": f"xgb_{family.prediction_prefix}",
                "backend": backend,
                "status": "trained",
                "target": TARGET_COL,
                "uses_cv_features": family.uses_cv_features,
                "model_file": str(model_path),
            }
        )

        print(f"  [Saved] {model_path}")

    # -------------------------------------------------------------------------
    # Prepare NN matrix
    # -------------------------------------------------------------------------
    if TORCH_AVAILABLE and (TRAIN_GRU or TRAIN_LSTM):
        print(f"\n[Preparing NN matrix: {family.family_name}]")

        nn_preprocessor = make_preprocessor(numeric_cols, categorical_cols)

        train_mask = df["run_id"].isin(MODEL_TRAIN_RUN_IDS).to_numpy()
        nn_preprocessor.fit(df.loc[train_mask, numeric_cols + categorical_cols])

        X_all = nn_preprocessor.transform(df[numeric_cols + categorical_cols]).astype(float)
        y_all = df[TARGET_COL].to_numpy(dtype=float)

        preprocessor_path = MODEL_DIR / f"nn_{family.prediction_prefix}_feature_preprocessor.joblib"
        joblib.dump(nn_preprocessor, preprocessor_path)

        nn_feature_names = get_feature_names(nn_preprocessor)
        if nn_feature_names:
            pd.DataFrame(
                {
                    "family": family.family_name,
                    "nn_encoded_feature_name": nn_feature_names,
                }
            ).to_csv(
                OUT_DIR / f"nn_encoded_feature_columns_{family.prediction_prefix}.csv",
                index=False,
            )
    else:
        X_all = None
        y_all = None

    # -------------------------------------------------------------------------
    # GRU
    # -------------------------------------------------------------------------
    if TRAIN_GRU:
        if not TORCH_AVAILABLE:
            print(f"\n[Skipping GRU: {family.family_name}] PyTorch not available.")
            summary_rows.append(
                {
                    "family": family.family_name,
                    "model": f"gru_{family.prediction_prefix}",
                    "backend": "pytorch",
                    "status": "skipped_no_torch",
                    "target": TARGET_COL,
                    "uses_cv_features": family.uses_cv_features,
                    "model_file": "",
                }
            )
        else:
            print(f"\n[Training GRU direct model: {family.family_name}]")
            gru_model, gru_scaler_X, gru_scaler_y, gru_hist = train_rnn_model(
                df=df,
                feature_matrix_unscaled=X_all,
                y=y_all,
                family=family,
                cell_type="GRU",
            )

            q_col = f"q_pred_gru_{family.prediction_prefix}_ft"
            df[q_col] = predict_rnn_by_group(
                model=gru_model,
                df=df,
                feature_matrix_unscaled=X_all,
                scaler_X=gru_scaler_X,
                scaler_y=gru_scaler_y,
                group_cols=family.sequence_group_cols,
            )
            pred_cols_created.append(q_col)

            model_path = MODEL_DIR / f"gru_{family.prediction_prefix}_model.pt"

            torch.save(
                {
                    "model_state_dict": gru_model.state_dict(),
                    "input_dim": int(X_all.shape[1]),
                    "cell_type": "GRU",
                    "hidden_size": NN_HIDDEN_SIZE,
                    "num_layers": NN_NUM_LAYERS,
                    "dropout": NN_DROPOUT,
                    "family": family.family_name,
                    "target": TARGET_COL,
                },
                model_path,
            )

            joblib.dump(gru_scaler_X, MODEL_DIR / f"gru_{family.prediction_prefix}_feature_scaler.joblib")
            joblib.dump(gru_scaler_y, MODEL_DIR / f"gru_{family.prediction_prefix}_target_scaler.joblib")

            histories.append(gru_hist)

            summary_rows.append(
                {
                    "family": family.family_name,
                    "model": f"gru_{family.prediction_prefix}",
                    "backend": "pytorch",
                    "status": "trained",
                    "target": TARGET_COL,
                    "uses_cv_features": family.uses_cv_features,
                    "model_file": str(model_path),
                }
            )

            print(f"  [Saved] {model_path}")

    # -------------------------------------------------------------------------
    # LSTM
    # -------------------------------------------------------------------------
    if TRAIN_LSTM:
        if not TORCH_AVAILABLE:
            print(f"\n[Skipping LSTM: {family.family_name}] PyTorch not available.")
            summary_rows.append(
                {
                    "family": family.family_name,
                    "model": f"lstm_{family.prediction_prefix}",
                    "backend": "pytorch",
                    "status": "skipped_no_torch",
                    "target": TARGET_COL,
                    "uses_cv_features": family.uses_cv_features,
                    "model_file": "",
                }
            )
        else:
            print(f"\n[Training LSTM direct model: {family.family_name}]")
            lstm_model, lstm_scaler_X, lstm_scaler_y, lstm_hist = train_rnn_model(
                df=df,
                feature_matrix_unscaled=X_all,
                y=y_all,
                family=family,
                cell_type="LSTM",
            )

            q_col = f"q_pred_lstm_{family.prediction_prefix}_ft"
            df[q_col] = predict_rnn_by_group(
                model=lstm_model,
                df=df,
                feature_matrix_unscaled=X_all,
                scaler_X=lstm_scaler_X,
                scaler_y=lstm_scaler_y,
                group_cols=family.sequence_group_cols,
            )
            pred_cols_created.append(q_col)

            model_path = MODEL_DIR / f"lstm_{family.prediction_prefix}_model.pt"

            torch.save(
                {
                    "model_state_dict": lstm_model.state_dict(),
                    "input_dim": int(X_all.shape[1]),
                    "cell_type": "LSTM",
                    "hidden_size": NN_HIDDEN_SIZE,
                    "num_layers": NN_NUM_LAYERS,
                    "dropout": NN_DROPOUT,
                    "family": family.family_name,
                    "target": TARGET_COL,
                },
                model_path,
            )

            joblib.dump(lstm_scaler_X, MODEL_DIR / f"lstm_{family.prediction_prefix}_feature_scaler.joblib")
            joblib.dump(lstm_scaler_y, MODEL_DIR / f"lstm_{family.prediction_prefix}_target_scaler.joblib")

            histories.append(lstm_hist)

            summary_rows.append(
                {
                    "family": family.family_name,
                    "model": f"lstm_{family.prediction_prefix}",
                    "backend": "pytorch",
                    "status": "trained",
                    "target": TARGET_COL,
                    "uses_cv_features": family.uses_cv_features,
                    "model_file": str(model_path),
                }
            )

            print(f"  [Saved] {model_path}")

    # Keep only prediction columns needed for merging.
    merge_keys = ["run_id", "time_sec"]
    if not family.deduplicate_run_time:
        merge_keys = ["run_id", "cv_rate_pct", "time_sec"]

    prediction_table = df[merge_keys + pred_cols_created].copy()

    summary_df = pd.DataFrame(summary_rows)

    return prediction_table, summary_df, histories + [feature_df]


# =============================================================================
# Prediction table and metrics
# =============================================================================

def build_combined_prediction_table(
    all_features: pd.DataFrame,
    family_prediction_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    keep_cols = [c for c in ID_COLS_TO_KEEP if c in all_features.columns]
    pred = all_features[keep_cols].copy()

    if "ml_split" not in pred.columns:
        pred["ml_split"] = "other"
        pred.loc[pred["run_id"].isin(MODEL_TRAIN_RUN_IDS), "ml_split"] = "train"
        pred.loc[pred["run_id"].isin(MODEL_VALIDATION_RUN_IDS), "ml_split"] = "validation"
        pred.loc[pred["run_id"].isin(MODEL_TEST_RUN_IDS), "ml_split"] = "test"

    for family in MODEL_FAMILIES:
        fam_pred = family_prediction_tables[family.family_name]

        if family.deduplicate_run_time:
            pred = pred.merge(fam_pred, on=["run_id", "time_sec"], how="left")
        else:
            pred = pred.merge(fam_pred, on=["run_id", "cv_rate_pct", "time_sec"], how="left")

    pred = pred.sort_values(["run_id", "cv_rate_pct", "time_sec"]).reset_index(drop=True)
    return pred


def compute_metrics_by_run_rate(pred: pd.DataFrame) -> pd.DataFrame:
    q_cols = {}

    for col in pred.columns:
        if col.startswith("q_pred_") and col.endswith("_ft"):
            model_name = col.replace("q_pred_", "").replace("_ft", "")
            q_cols[model_name] = col

    rows = []

    for (run_id, rate), g in pred.groupby(["run_id", "cv_rate_pct"], sort=True):
        y = g[TARGET_COL].to_numpy(dtype=float)
        t = g["time_sec"].to_numpy(dtype=float)

        split_values = sorted(set(g["ml_split"].dropna().astype(str)))
        split_label = ",".join(split_values) if split_values else "unknown"

        for model_name, q_col in q_cols.items():
            yp = g[q_col].to_numpy(dtype=float)

            if "_ml_only" in model_name:
                family = "ml_only"
            elif "_ml_cv" in model_name:
                family = "ml_cv"
            else:
                family = "unknown"

            rows.append(
                {
                    "run_id": int(run_id),
                    "cv_rate_pct": int(rate),
                    "ml_split": split_label,
                    "family": family,
                    "model": model_name,
                    "q_pred_col": q_col,
                    "n_samples": int(len(g)),
                    "mae_ft": safe_mae(y, yp),
                    "rmse_ft": rmse(y, yp),
                    "max_abs_error_ft": max_abs_error(y, yp),
                    "area_abs_error_ft_s": area_abs_error(t, y, yp),
                    "mean_q_gt_ft": float(np.nanmean(y)) if len(y) else np.nan,
                    "mean_q_pred_ft": float(np.nanmean(yp)) if len(yp) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def summarize_metrics_by_run(metrics_rate: pd.DataFrame) -> pd.DataFrame:
    if metrics_rate.empty:
        return pd.DataFrame()

    group_cols = ["run_id", "ml_split", "family", "model", "q_pred_col"]

    metric_cols = [
        "mae_ft",
        "rmse_ft",
        "max_abs_error_ft",
        "area_abs_error_ft_s",
        "mean_q_gt_ft",
        "mean_q_pred_ft",
    ]

    out = (
        metrics_rate.groupby(group_cols, as_index=False)[metric_cols]
        .mean()
        .sort_values(["run_id", "family", "model"])
        .reset_index(drop=True)
    )

    return out


# =============================================================================
# Diagnostic plots
# =============================================================================

def get_diagnostic_prediction_columns(pred: pd.DataFrame) -> list[str]:
    cols = []
    for col in [
        "q_pred_gru_ml_only_ft",
        "q_pred_lstm_ml_only_ft",
    ]:
        if col in pred.columns:
            cols.append(col)
    return cols


def diagnostic_label(col: str) -> str:
    mapping = {
        "q_gt_ft": "Ground Truth",
        "q_pred_gru_ml_only_ft": "ML-only GRU",
        "q_pred_lstm_ml_only_ft": "ML-only LSTM",
    }
    return mapping.get(col, col)


def save_ml_only_full_run_plot(pred: pd.DataFrame, run_id: int, rate: int) -> None:
    df = pred[
        (pred["run_id"].astype(int) == int(run_id))
        & (pred["cv_rate_pct"].astype(int) == int(rate))
    ].copy()
    if df.empty:
        print(f"[WARN] No diagnostic rows for run {run_id:03d}, rate {rate}%.")
        return

    df = df.sort_values("time_sec")
    pred_cols = get_diagnostic_prediction_columns(df)
    if not pred_cols:
        print("[WARN] No GRU/LSTM ML-only prediction columns for full-run diagnostic plot.")
        return

    fig, ax = plt.subplots(figsize=(16, 5.8))
    ax.plot(
        df["time_sec"],
        pd.to_numeric(df["q_gt_ft"], errors="coerce"),
        color="black",
        linewidth=1.8,
        label="Ground Truth",
    )

    colors = {
        "q_pred_gru_ml_only_ft": "tab:purple",
        "q_pred_lstm_ml_only_ft": "tab:blue",
    }
    for col in pred_cols:
        ax.plot(
            df["time_sec"],
            pd.to_numeric(df[col], errors="coerce"),
            linewidth=1.5,
            alpha=0.9,
            label=diagnostic_label(col),
            color=colors.get(col),
        )

    ax.set_title(f"ML-only sequence diagnostic | Run {run_id:03d} | CV {rate}%")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Queue length (ft)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    out_path = FIG_DIR / f"ml_only_sequence_full_run{run_id:03d}_rate{rate:03d}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Saved diagnostic figure] {out_path}")


def infer_red_to_red_cycles(df: pd.DataFrame) -> list[tuple[float, float]]:
    if "phase_state" not in df.columns:
        return []

    work = df.sort_values("time_sec").copy()
    phase = work["phase_state"].astype(str).str.lower().to_numpy()
    time = pd.to_numeric(work["time_sec"], errors="coerce").to_numpy(dtype=float)

    red_start_times = []
    prev = ""
    for p, t in zip(phase, time):
        if p == "red" and prev != "red" and np.isfinite(t):
            red_start_times.append(float(t))
        prev = p

    cycles = []
    for s, e in zip(red_start_times[:-1], red_start_times[1:]):
        if e > s:
            cycles.append((s, e))
    return cycles


def save_ml_only_cyclewise_plot(pred: pd.DataFrame, run_id: int, rate: int) -> None:
    df = pred[
        (pred["run_id"].astype(int) == int(run_id))
        & (pred["cv_rate_pct"].astype(int) == int(rate))
    ].copy()
    if df.empty:
        return

    df = df.sort_values("time_sec")
    pred_cols = get_diagnostic_prediction_columns(df)
    if not pred_cols:
        return

    cycles = infer_red_to_red_cycles(df)
    if not cycles:
        t_min = float(df["time_sec"].min())
        t_max = float(df["time_sec"].max())
        edges = np.linspace(t_min, t_max, DIAGNOSTIC_MAX_CYCLES + 1)
        cycles = [(float(edges[i]), float(edges[i + 1])) for i in range(DIAGNOSTIC_MAX_CYCLES)]

    cycles = cycles[: int(DIAGNOSTIC_MAX_CYCLES)]
    if not cycles:
        return

    ncols = 3
    nrows = int(math.ceil(len(cycles) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.0 * nrows), squeeze=False)

    colors = {
        "q_pred_gru_ml_only_ft": "tab:purple",
        "q_pred_lstm_ml_only_ft": "tab:blue",
    }

    for i, (start, end) in enumerate(cycles):
        ax = axes[i // ncols][i % ncols]
        g = df[(df["time_sec"] >= start) & (df["time_sec"] <= end)].copy()

        ax.plot(
            g["time_sec"],
            pd.to_numeric(g["q_gt_ft"], errors="coerce"),
            color="black",
            linewidth=1.7,
            label="Ground Truth",
        )
        for col in pred_cols:
            ax.plot(
                g["time_sec"],
                pd.to_numeric(g[col], errors="coerce"),
                linewidth=1.3,
                alpha=0.9,
                label=diagnostic_label(col),
                color=colors.get(col),
            )

        ax.set_title(f"Cycle {i + 1}: {start:.1f}-{end:.1f}s", fontsize=9)
        ax.grid(True, alpha=0.22)
        ax.set_ylim(bottom=0)

    for j in range(len(cycles), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), frameon=True)
    fig.suptitle(f"ML-only sequence diagnostic cycles | Run {run_id:03d} | CV {rate}%", y=0.995)
    fig.supxlabel("Time (s)")
    fig.supylabel("Queue length (ft)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = FIG_DIR / f"ml_only_sequence_cyclewise_run{run_id:03d}_rate{rate:03d}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Saved diagnostic figure] {out_path}")


def save_ml_only_diagnostic_plots(pred: pd.DataFrame) -> None:
    run_ids = DIAGNOSTIC_RUN_IDS or sorted(pred.loc[pred["ml_split"].astype(str).str.contains("test", na=False), "run_id"].dropna().astype(int).unique())[:1]
    for run_id in run_ids:
        save_ml_only_full_run_plot(pred, int(run_id), int(DIAGNOSTIC_CV_RATE_PCT))
        save_ml_only_cyclewise_plot(pred, int(run_id), int(DIAGNOSTIC_CV_RATE_PCT))


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    set_all_seeds(NN_RANDOM_SEED)

    print("=" * 96)
    print("Direct ML queue-length model training")
    print("=" * 96)
    print(f"Project root   : {PROJECT_ROOT}")
    print(f"Feature file   : {FEATURE_FILE}")
    print(f"Output dir     : {OUT_DIR}")
    print(f"Train runs     : {MODEL_TRAIN_RUN_IDS}")
    print(f"Validation     : {MODEL_VALIDATION_RUN_IDS}")
    print(f"Test runs      : {MODEL_TEST_RUN_IDS}")
    print(f"CV rates       : {CV_RATES_PCT}")
    print(f"Target         : {TARGET_COL}")
    print(f"XGBoost avail. : {XGBOOST_AVAILABLE}")
    print(f"PyTorch avail. : {TORCH_AVAILABLE} | device={DEVICE}")
    print("=" * 96)

    all_features = load_feature_table()

    print(f"Loaded rows: {len(all_features):,}")
    print(
        all_features.groupby(["ml_split", "run_id"], sort=True)
        .size()
        .reset_index(name="rows")
        .to_string(index=False)
    )

    family_prediction_tables = {}
    summary_parts = []
    history_parts = []
    feature_parts = []

    for family in MODEL_FAMILIES:
        fam_pred, fam_summary, fam_outputs = train_one_family(all_features, family)
        family_prediction_tables[family.family_name] = fam_pred
        summary_parts.append(fam_summary)

        for item in fam_outputs:
            if "epoch" in item.columns:
                history_parts.append(item)
            elif "feature_name" in item.columns:
                feature_parts.append(item)

    pred = build_combined_prediction_table(all_features, family_prediction_tables)

    pred_all_path = OUT_DIR / "ml_direct_predictions_allruns_allrates.csv"
    pred.to_csv(pred_all_path, index=False)
    print(f"\n[Saved combined predictions] {pred_all_path}")

    if SAVE_PER_RUN_RATE_FILES:
        for (run_id, rate), g in pred.groupby(["run_id", "cv_rate_pct"], sort=True):
            out_path = OUT_DIR / f"ml_direct_predictions_run{int(run_id):03d}_rate{format_rate(int(rate))}.csv"
            g.to_csv(out_path, index=False)

        print(f"[Saved per-run/rate prediction files] {OUT_DIR}")

    if summary_parts:
        summary = pd.concat(summary_parts, ignore_index=True)
    else:
        summary = pd.DataFrame()

    if not summary.empty:
        summary["train_runs"] = ",".join([f"{r:03d}" for r in MODEL_TRAIN_RUN_IDS])
        summary["validation_runs"] = ",".join([f"{r:03d}" for r in MODEL_VALIDATION_RUN_IDS])
        summary["test_runs"] = ",".join([f"{r:03d}" for r in MODEL_TEST_RUN_IDS])
        summary.to_csv(OUT_DIR / "ml_direct_training_summary.csv", index=False)
        print(f"[Saved] {OUT_DIR / 'ml_direct_training_summary.csv'}")

    if history_parts:
        history = pd.concat(history_parts, ignore_index=True)
        history.to_csv(OUT_DIR / "nn_training_history_ml_direct.csv", index=False)
        print(f"[Saved] {OUT_DIR / 'nn_training_history_ml_direct.csv'}")

    if feature_parts:
        features_used = pd.concat(feature_parts, ignore_index=True)
        features_used.to_csv(OUT_DIR / "feature_columns_used_ml_direct.csv", index=False)
        print(f"[Saved] {OUT_DIR / 'feature_columns_used_ml_direct.csv'}")

    metrics_rate = compute_metrics_by_run_rate(pred)
    metrics_run = summarize_metrics_by_run(metrics_rate)

    metrics_rate_path = OUT_DIR / "ml_direct_metrics_by_model_run_rate.csv"
    metrics_run_path = OUT_DIR / "ml_direct_metrics_by_model_run.csv"

    metrics_rate.to_csv(metrics_rate_path, index=False)
    metrics_run.to_csv(metrics_run_path, index=False)

    print(f"[Saved metrics] {metrics_rate_path}")
    print(f"[Saved metrics] {metrics_run_path}")

    save_ml_only_diagnostic_plots(pred)

    # Terminal summaries.
    val_metrics = metrics_rate[metrics_rate["ml_split"].str.contains("validation", na=False)].copy()
    test_metrics = metrics_rate[metrics_rate["ml_split"].str.contains("test", na=False)].copy()

    if not val_metrics.empty:
        print("\nValidation summary by model:")
        print(
            val_metrics.groupby(["family", "model"])[["mae_ft", "rmse_ft", "max_abs_error_ft"]]
            .mean()
            .sort_values("rmse_ft")
            .round(3)
            .to_string()
        )

    if not test_metrics.empty:
        print("\nTest summary by model:")
        print(
            test_metrics.groupby(["family", "model"])[["mae_ft", "rmse_ft", "max_abs_error_ft"]]
            .mean()
            .sort_values("rmse_ft")
            .round(3)
            .to_string()
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
