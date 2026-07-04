"""
Supplemental robustness and ablation studies for Queue Length Reconstruction.

Place this file at:
    src/run_supplemental_robustness_ablation.py

Purpose
-------
This is a standalone supplemental-analysis script.

It does NOT modify the core pipeline.
It does NOT modify the main publication plotting script.
It creates its own CSV outputs, plots, and PNG tables.

Main studies
------------
1. Ablation study
   Retrains the final selected GRU Physics + ML + CV residual model after
   removing feature groups.

2. Robustness study
   Trains the full GRU Physics + ML + CV residual model once, then perturbs
   selected test features at inference time.

Final model family tested
-------------------------
Physics + ML + CV residual learning:
    target residual = q_gt_ft - q_physics_baseline_ft
    final queue     = q_physics_baseline_ft + predicted residual

Outputs
-------
output/intermediate_csv/supplemental_robustness_ablation/
    ablation_metrics_by_case_run_rate.csv
    ablation_summary_by_case_rate.csv
    robustness_metrics_by_case_seed_run_rate.csv
    robustness_summary_by_case_rate.csv
    robustness_degradation_summary.csv
    figures/
    tables_png/
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Iterable
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import config as cfg


# =============================================================================
# 0. CONFIGURATION WINDOW
# =============================================================================

PROJECT_ROOT = cfg.PROJECT_ROOT
INTERMEDIATE_DIR = PROJECT_ROOT / "output" / "intermediate_csv"

FEATURE_FILE = INTERMEDIATE_DIR / "cv_features" / "timegrid_features_allruns_allrates.csv"

OUT_DIR = INTERMEDIATE_DIR / "supplemental_robustness_ablation"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables_png"

TRAIN_RUN_IDS = list(getattr(cfg, "TRAIN_RUN_IDS", [5, 6, 7, 8, 9, 10]))
VALIDATION_RUN_IDS = list(getattr(cfg, "VALIDATION_RUN_IDS", [11]))
TEST_RUN_IDS = list(getattr(cfg, "TEST_RUN_IDS", [12, 13, 14]))
CV_RATES_PCT = list(getattr(cfg, "CV_RATES_PCT", [1, 2, 5, 10, 20, 50, 100]))

# Main selected model.
MODEL_NAME = "GRU"
TARGET_MODE = "physics_residual"

# Set to None to use all available windows. For smoke testing, use smaller values.
MAX_TRAIN_WINDOWS_PER_CASE = 250_000
MAX_VALID_WINDOWS_PER_CASE = 80_000

# GRU/window settings.
SEQUENCE_LENGTH = 12
SEQUENCE_STRIDE_TRAIN = 3
SEQUENCE_STRIDE_VALID = 5
BATCH_SIZE = 512
MAX_EPOCHS = 40
PATIENCE = 6
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.0
GRAD_CLIP_NORM = 5.0

# Reproducibility.
GLOBAL_SEED = 42
ROBUSTNESS_SEEDS = [101, 202, 303]

# Output/control.
MAKE_ABLATION_STUDY = False
MAKE_ROBUSTNESS_STUDY = True
MAKE_PLOTS_AND_TABLES = True
SELECTED_TABLE_RATE = 10
FIGURE_DPI = 300
TABLE_DPI = 300
SHOW_FIGURES = False

# Metrics to show in PNG tables.
TABLE_METRICS = [
    "mae_ft",
    "rmse_ft",
    "abc_ft_s",
    "mean_cycle_peak_abs_error_ft",
]

# -------------------------------------------------------------------------
# Feature groups
# -------------------------------------------------------------------------

SIGNAL_NUMERIC_FEATURES = [
    "phase_elapsed_sec",
]

COUNT_PHYSICS_FEATURES = [
    "A_count",
    "D_count",
    "V_count",
    "B_count",
    "n_queue_cumulative",
]

CV_ANCHOR_FEATURES = [
    "inside_cv_segment",
    "prev_cv_anchor_q_ft",
    "next_cv_anchor_q_ft",
    "time_since_prev_cv_sec",
    "time_to_next_cv_sec",
    "cv_segment_duration_sec",
    "cv_segment_frac",
]

CV_QUEUE_VALUE_FEATURES = [
    "prev_cv_anchor_q_ft",
    "next_cv_anchor_q_ft",
]

CV_TIMING_CONTEXT_FEATURES = [
    "inside_cv_segment",
    "time_since_prev_cv_sec",
    "time_to_next_cv_sec",
    "cv_segment_duration_sec",
    "cv_segment_frac",
]

PHASE_CATEGORICAL_COL = "phase_state"
PHASE_LEVELS = ["red", "green", "amber", "yellow", "unknown"]

FULL_NUMERIC_FEATURES = (
    SIGNAL_NUMERIC_FEATURES
    + COUNT_PHYSICS_FEATURES
    + CV_ANCHOR_FEATURES
)

# Ablation cases. Each case retrains the model.
ABLATION_CASES = {
    "full_model": {
        "description": "All selected Physics + ML + CV features",
        "remove_numeric": [],
        "remove_phase": False,
    },
    "no_signal_features": {
        "description": "Remove signal phase state and phase elapsed time",
        "remove_numeric": SIGNAL_NUMERIC_FEATURES,
        "remove_phase": True,
    },
    "no_count_physics_features": {
        "description": "Remove A/D/V/B/n_queue cumulative-count features",
        "remove_numeric": COUNT_PHYSICS_FEATURES,
        "remove_phase": False,
    },
    "no_cv_anchor_features": {
        "description": "Remove all CV-anchor features",
        "remove_numeric": CV_ANCHOR_FEATURES,
        "remove_phase": False,
    },
    "no_cv_queue_value_features": {
        "description": "Remove previous/next/interpolated CV queue values",
        "remove_numeric": CV_QUEUE_VALUE_FEATURES,
        "remove_phase": False,
    },
    "no_cv_timing_context_features": {
        "description": "Remove CV timing/segment context features",
        "remove_numeric": CV_TIMING_CONTEXT_FEATURES,
        "remove_phase": False,
    },
}

# Robustness cases. These perturb test features only after clean training.
ROBUSTNESS_CASES = {
    "clean": {
        "description": "No perturbation",
        "type": "none",
    },
    "cv_queue_noise_25ft": {
        "description": "Gaussian noise with sigma=25 ft added to CV queue-value features",
        "type": "cv_queue_noise",
        "sigma_ft": 25.0,
    },
    "cv_queue_noise_50ft": {
        "description": "Gaussian noise with sigma=50 ft added to CV queue-value features",
        "type": "cv_queue_noise",
        "sigma_ft": 50.0,
    },
    "cv_timing_jitter_2s": {
        "description": "Gaussian timing jitter with sigma=2 s added to CV timing features",
        "type": "cv_timing_jitter",
        "sigma_sec": 2.0,
    },
    "count_noise_1veh": {
        "description": "Gaussian noise with sigma=1 vehicle added to cumulative-count features",
        "type": "count_noise",
        "sigma_count": 1.0,
    },
    "phase_elapsed_noise_2s": {
        "description": "Gaussian noise with sigma=2 s added to phase_elapsed_sec",
        "type": "phase_elapsed_noise",
        "sigma_sec": 2.0,
    },
    "cv_feature_dropout_20pct": {
        "description": "Randomly mask CV-anchor feature rows with probability 20%",
        "type": "cv_feature_dropout",
        "dropout_prob": 0.20,
    },
}


# =============================================================================
# 1. GENERAL HELPERS
# =============================================================================

def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_filename(text: str) -> str:
    out = str(text).strip().lower()
    for ch in [" ", "+", "/", "\\", "(", ")", "%", ":", ";", "|", ".", "·"]:
        out = out.replace(ch, "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def save_table_png(table_df: pd.DataFrame, filename: str, title: str) -> None:
    if table_df.empty:
        print(f"[WARN] Empty table skipped: {filename}")
        return

    fig_width = max(9.0, 1.45 * len(table_df.columns))
    fig_height = max(2.8, 0.42 * len(table_df) + 1.4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.25)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
        if col == 0 and row > 0:
            cell.set_text_props(ha="left")

    fig.tight_layout()
    out_path = TABLE_DIR / filename
    fig.savefig(out_path, dpi=TABLE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved table PNG] {out_path}")


def save_line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    if df.empty:
        print(f"[WARN] Empty plot skipped: {filename}")
        return

    fig, ax = plt.subplots(figsize=(10.8, 6.0))

    for label, g in df.groupby(group_col, sort=False):
        g = g.sort_values(x_col)
        ax.plot(
            pd.to_numeric(g[x_col], errors="coerce"),
            pd.to_numeric(g[y_col], errors="coerce"),
            marker="o",
            linewidth=2.0,
            label=str(label),
        )

    ax.set_title(title)
    ax.set_xlabel("CV penetration rate (%)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()

    out_path = FIG_DIR / filename
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

    print(f"[Saved figure] {out_path}")


# =============================================================================
# 2. DATA LOADING AND FEATURE PREPARATION
# =============================================================================

def load_feature_data() -> pd.DataFrame:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Missing feature file:\n{FEATURE_FILE}")

    df = pd.read_csv(FEATURE_FILE, low_memory=False)

    required = [
        "run_id",
        "cv_rate_pct",
        "time_sec",
        "q_gt_ft",
    ]
    require_columns(df, required, "feature table")

    df["run_id"] = safe_numeric(df["run_id"])
    df["cv_rate_pct"] = safe_numeric(df["cv_rate_pct"])
    df["time_sec"] = safe_numeric(df["time_sec"])

    df = df.dropna(subset=["run_id", "cv_rate_pct", "time_sec", "q_gt_ft"]).copy()
    df["run_id"] = df["run_id"].astype(int)
    df["cv_rate_pct"] = df["cv_rate_pct"].astype(int)

    # Standardize baseline column name.
    if "q_physics_baseline_ft" in df.columns:
        df["q_physics_baseline_ft"] = safe_numeric(df["q_physics_baseline_ft"])
    elif "q_baseline_fixed_ft" in df.columns:
        df["q_physics_baseline_ft"] = safe_numeric(df["q_baseline_fixed_ft"])
    else:
        raise ValueError("Feature table needs q_physics_baseline_ft or q_baseline_fixed_ft.")

    df["q_gt_ft"] = safe_numeric(df["q_gt_ft"])
    df["q_physics_baseline_ft"] = safe_numeric(df["q_physics_baseline_ft"])

    # Assign split if not already available.
    if "ml_split" not in df.columns:
        df["ml_split"] = "unused"
        df.loc[df["run_id"].isin(TRAIN_RUN_IDS), "ml_split"] = "train"
        df.loc[df["run_id"].isin(VALIDATION_RUN_IDS), "ml_split"] = "validation"
        df.loc[df["run_id"].isin(TEST_RUN_IDS), "ml_split"] = "test"
    else:
        df["ml_split"] = df["ml_split"].astype(str).str.lower().str.strip()

    if PHASE_CATEGORICAL_COL not in df.columns:
        df[PHASE_CATEGORICAL_COL] = "unknown"

    df[PHASE_CATEGORICAL_COL] = (
        df[PHASE_CATEGORICAL_COL]
        .astype(str)
        .str.lower()
        .str.strip()
        .replace({"nan": "unknown", "": "unknown"})
    )

    # Add missing numeric feature columns as NaN so feature cases remain stable.
    for col in FULL_NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = safe_numeric(df[col])

    df["target_residual_ft"] = df["q_gt_ft"] - df["q_physics_baseline_ft"]

    df = df.sort_values(["run_id", "cv_rate_pct", "time_sec"]).reset_index(drop=True)

    print(f"[Loaded feature table] rows={len(df):,}, cols={len(df.columns):,}")
    print(f"Runs available: {sorted(df['run_id'].unique().tolist())}")
    print(f"CV rates available: {sorted(df['cv_rate_pct'].unique().tolist())}")
    print(df["ml_split"].value_counts(dropna=False).to_string())

    return df


def phase_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    phase = df[PHASE_CATEGORICAL_COL].astype(str).str.lower().str.strip()
    for level in PHASE_LEVELS:
        out[f"phase_{level}"] = (phase == level).astype(float)
    return out


@dataclass
class FeatureSpec:
    case_name: str
    description: str
    numeric_cols: list[str]
    use_phase: bool


@dataclass
class Preprocessor:
    numeric_cols: list[str]
    phase_cols: list[str]
    mean: pd.Series
    std: pd.Series
    fill: pd.Series

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parts = []

        if self.numeric_cols:
            num = df[self.numeric_cols].copy()
            for col in self.numeric_cols:
                num[col] = safe_numeric(num[col])
            num = num.fillna(self.fill)
            num = (num - self.mean) / self.std
            parts.append(num.astype(np.float32))

        if self.phase_cols:
            ph = phase_one_hot(df)
            for col in self.phase_cols:
                if col not in ph.columns:
                    ph[col] = 0.0
            parts.append(ph[self.phase_cols].astype(np.float32))

        if not parts:
            raise ValueError("No model features selected.")

        X = pd.concat(parts, axis=1).to_numpy(dtype=np.float32)
        y = df["target_residual_ft"].to_numpy(dtype=np.float32)
        base = df["q_physics_baseline_ft"].to_numpy(dtype=np.float32)
        return X, y, base


def make_feature_spec(case_name: str, case_info: dict) -> FeatureSpec:
    remove_numeric = set(case_info.get("remove_numeric", []))
    numeric_cols = [c for c in FULL_NUMERIC_FEATURES if c not in remove_numeric]

    use_phase = not bool(case_info.get("remove_phase", False))
    phase_cols = [f"phase_{p}" for p in PHASE_LEVELS] if use_phase else []

    return FeatureSpec(
        case_name=case_name,
        description=str(case_info.get("description", case_name)),
        numeric_cols=numeric_cols,
        use_phase=use_phase,
    )


def fit_preprocessor(train_df: pd.DataFrame, spec: FeatureSpec) -> Preprocessor:
    if spec.numeric_cols:
        num = train_df[spec.numeric_cols].copy()
        for col in spec.numeric_cols:
            num[col] = safe_numeric(num[col])
        fill = num.median(numeric_only=True).fillna(0.0)
        num = num.fillna(fill)
        mean = num.mean()
        std = num.std().replace(0, 1.0).fillna(1.0)
    else:
        fill = pd.Series(dtype=float)
        mean = pd.Series(dtype=float)
        std = pd.Series(dtype=float)

    phase_cols = [f"phase_{p}" for p in PHASE_LEVELS] if spec.use_phase else []

    return Preprocessor(
        numeric_cols=spec.numeric_cols,
        phase_cols=phase_cols,
        mean=mean,
        std=std,
        fill=fill,
    )


# =============================================================================
# 3. GRU DATASET AND MODEL
# =============================================================================

class SequenceDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: pd.DataFrame,
        seq_len: int,
        stride: int,
        max_windows: int | None,
        seed: int,
    ):
        self.X = X
        self.y = y
        self.seq_len = int(seq_len)

        windows = []

        for _, g in groups.groupby(["run_id", "cv_rate_pct"], sort=False):
            idx = g.index.to_numpy(dtype=int)
            if len(idx) == 0:
                continue

            # Label at end index; sequence uses previous seq_len rows within same group.
            for pos in range(0, len(idx), max(1, int(stride))):
                end_idx = idx[pos]
                windows.append(end_idx)

        if max_windows is not None and len(windows) > int(max_windows):
            rng = np.random.default_rng(seed)
            windows = rng.choice(np.array(windows), size=int(max_windows), replace=False).tolist()

        self.windows = np.array(windows, dtype=int)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        end_idx = int(self.windows[i])

        # This assumes original X is sorted by run/rate/time and that sampled windows
        # do not cross group boundaries because group indexing was used for end_idx.
        # For first rows of a group, padding is handled approximately by repeating
        # the earliest available row within the previous seq_len slice.
        start_idx = max(0, end_idx - self.seq_len + 1)
        seq = self.X[start_idx:end_idx + 1]

        if len(seq) < self.seq_len:
            pad = np.repeat(seq[[0]], self.seq_len - len(seq), axis=0)
            seq = np.vstack([pad, seq])

        return torch.from_numpy(seq.astype(np.float32)), torch.tensor(self.y[end_idx], dtype=torch.float32)


class GRURegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        pred = self.head(last).squeeze(-1)
        return pred


def train_gru_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    spec: FeatureSpec,
    seed: int,
) -> tuple[GRURegressor, Preprocessor, dict]:
    set_seed(seed)

    pre = fit_preprocessor(train_df, spec)

    X_train, y_train, _ = pre.transform(train_df)
    X_valid, y_valid, _ = pre.transform(valid_df)

    train_groups = train_df[["run_id", "cv_rate_pct"]].copy()
    valid_groups = valid_df[["run_id", "cv_rate_pct"]].copy()

    train_ds = SequenceDataset(
        X=X_train,
        y=y_train,
        groups=train_groups,
        seq_len=SEQUENCE_LENGTH,
        stride=SEQUENCE_STRIDE_TRAIN,
        max_windows=MAX_TRAIN_WINDOWS_PER_CASE,
        seed=seed,
    )

    valid_ds = SequenceDataset(
        X=X_valid,
        y=y_valid,
        groups=valid_groups,
        seq_len=SEQUENCE_LENGTH,
        stride=SEQUENCE_STRIDE_VALID,
        max_windows=MAX_VALID_WINDOWS_PER_CASE,
        seed=seed + 1,
    )

    if len(train_ds) == 0:
        raise RuntimeError(f"No training windows for case {spec.case_name}.")
    if len(valid_ds) == 0:
        raise RuntimeError(f"No validation windows for case {spec.case_name}.")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRURegressor(input_dim=X_train.shape[1]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_state = None
    best_valid = float("inf")
    bad_epochs = 0
    history = []

    print(f"\n[Training] {spec.case_name}")
    print(f"  features numeric={len(spec.numeric_cols)}, phase={spec.use_phase}, input_dim={X_train.shape[1]}")
    print(f"  train_windows={len(train_ds):,}, valid_windows={len(valid_ds):,}, device={device}")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()

            if GRAD_CLIP_NORM is not None:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            opt.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                valid_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(train_losses))
        valid_loss = float(np.mean(valid_losses))
        history.append({"epoch": epoch, "train_mse": train_loss, "valid_mse": valid_loss})

        print(f"  epoch {epoch:02d} | train_mse={train_loss:.4f} | valid_mse={valid_loss:.4f}")

        if valid_loss < best_valid - 1e-6:
            best_valid = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print(f"  early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model = model.to("cpu")
    info = {
        "best_valid_mse": best_valid,
        "n_train_windows": len(train_ds),
        "n_valid_windows": len(valid_ds),
        "history": history,
    }

    return model, pre, info


def predict_residuals_groupwise(model: GRURegressor, pre: Preprocessor, df: pd.DataFrame) -> np.ndarray:
    X, _, _ = pre.transform(df)
    model.eval()

    preds = np.full(len(df), np.nan, dtype=float)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for _, g in df.groupby(["run_id", "cv_rate_pct"], sort=False):
            idx = g.index.to_numpy(dtype=int)
            Xg = X[idx]

            rows = []
            for pos in range(len(idx)):
                start = max(0, pos - SEQUENCE_LENGTH + 1)
                seq = Xg[start:pos + 1]
                if len(seq) < SEQUENCE_LENGTH:
                    pad = np.repeat(seq[[0]], SEQUENCE_LENGTH - len(seq), axis=0)
                    seq = np.vstack([pad, seq])
                rows.append(seq)

                if len(rows) >= 4096 or pos == len(idx) - 1:
                    xb = torch.tensor(np.asarray(rows, dtype=np.float32), dtype=torch.float32).to(device)
                    yhat = model(xb).detach().cpu().numpy().astype(float)
                    write_start = pos - len(rows) + 1
                    preds[idx[write_start:pos + 1]] = yhat
                    rows = []

    model = model.to("cpu")
    return preds


# =============================================================================
# 4. METRICS
# =============================================================================

def phase_intervals_from_timegrid(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or PHASE_CATEGORICAL_COL not in df.columns:
        return pd.DataFrame(columns=["start", "end", "state"])

    d = df[["time_sec", PHASE_CATEGORICAL_COL]].copy()
    d["time_sec"] = safe_numeric(d["time_sec"])
    d[PHASE_CATEGORICAL_COL] = d[PHASE_CATEGORICAL_COL].astype(str).str.lower().str.strip()
    d = d.dropna(subset=["time_sec"]).sort_values("time_sec").drop_duplicates("time_sec", keep="last")

    if len(d) < 2:
        return pd.DataFrame(columns=["start", "end", "state"])

    d["changed"] = d[PHASE_CATEGORICAL_COL].ne(d[PHASE_CATEGORICAL_COL].shift()).astype(int)
    d["segment_id"] = d["changed"].cumsum()

    intervals = (
        d.groupby("segment_id", as_index=False)
        .agg(start=("time_sec", "first"), state=(PHASE_CATEGORICAL_COL, "first"))
        .sort_values("start")
        .reset_index(drop=True)
    )
    intervals["end"] = intervals["start"].shift(-1)

    dt = np.nanmedian(np.diff(d["time_sec"].to_numpy(dtype=float))) if len(d) > 2 else 0.1
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.1

    intervals["end"] = intervals["end"].fillna(float(d["time_sec"].max()) + dt)
    intervals = intervals[intervals["end"] > intervals["start"]].copy()
    return intervals[["start", "end", "state"]].reset_index(drop=True)


def red_to_red_cycle_windows(df: pd.DataFrame) -> pd.DataFrame:
    intervals = phase_intervals_from_timegrid(df)
    if intervals.empty:
        return pd.DataFrame(columns=["cycle_id", "cycle_start_sec", "cycle_end_sec"])

    intervals["prev_state"] = intervals["state"].shift()
    red_starts = intervals[
        (intervals["state"] == "red")
        & (intervals["prev_state"] != "red")
    ]["start"].to_numpy(dtype=float)

    red_starts = np.sort(red_starts[np.isfinite(red_starts)])

    if len(red_starts) < 2:
        return pd.DataFrame(columns=["cycle_id", "cycle_start_sec", "cycle_end_sec"])

    raw = [(float(red_starts[i]), float(red_starts[i + 1])) for i in range(len(red_starts) - 1)]

    # Drop first and last partial/boundary cycles.
    if len(raw) <= 2:
        return pd.DataFrame(columns=["cycle_id", "cycle_start_sec", "cycle_end_sec"])

    kept = raw[1:-1]

    return pd.DataFrame(
        [
            {"cycle_id": i, "cycle_start_sec": s, "cycle_end_sec": e}
            for i, (s, e) in enumerate(kept, start=1)
        ]
    )


def cycle_peak_error(df: pd.DataFrame, pred_col: str) -> dict:
    cycles = red_to_red_cycle_windows(df)
    if cycles.empty:
        return {
            "n_cycles": 0,
            "mean_cycle_peak_abs_error_ft": np.nan,
            "rmse_cycle_peak_error_ft": np.nan,
        }

    errs = []

    for _, cyc in cycles.iterrows():
        t0 = float(cyc["cycle_start_sec"])
        t1 = float(cyc["cycle_end_sec"])
        g = df[(df["time_sec"] >= t0) & (df["time_sec"] < t1)].copy()

        if g.empty:
            continue

        gt_peak = float(np.nanmax(pd.to_numeric(g["q_gt_ft"], errors="coerce")))
        pred_peak = float(np.nanmax(pd.to_numeric(g[pred_col], errors="coerce")))

        if np.isfinite(gt_peak) and np.isfinite(pred_peak):
            errs.append(pred_peak - gt_peak)

    if not errs:
        return {
            "n_cycles": 0,
            "mean_cycle_peak_abs_error_ft": np.nan,
            "rmse_cycle_peak_error_ft": np.nan,
        }

    e = np.asarray(errs, dtype=float)

    return {
        "n_cycles": int(len(e)),
        "mean_cycle_peak_abs_error_ft": float(np.mean(np.abs(e))),
        "rmse_cycle_peak_error_ft": float(np.sqrt(np.mean(e ** 2))),
    }


def evaluate_predictions(df: pd.DataFrame, pred_col: str, case_name: str, study_type: str, seed: int | None) -> pd.DataFrame:
    rows = []

    for (run_id, rate), g in df.groupby(["run_id", "cv_rate_pct"], sort=True):
        g = g.sort_values("time_sec").copy()

        gt = pd.to_numeric(g["q_gt_ft"], errors="coerce").to_numpy(dtype=float)
        pred = pd.to_numeric(g[pred_col], errors="coerce").to_numpy(dtype=float)
        time = pd.to_numeric(g["time_sec"], errors="coerce").to_numpy(dtype=float)

        mask = np.isfinite(gt) & np.isfinite(pred) & np.isfinite(time)
        gt = gt[mask]
        pred = pred[mask]
        time = time[mask]

        if len(gt) == 0:
            continue

        err = pred - gt
        abs_err = np.abs(err)

        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        abc = trapezoid(abs_err, time) if len(time) >= 2 else np.nan

        cpe = cycle_peak_error(g, pred_col)

        rows.append(
            {
                "study_type": study_type,
                "case_name": case_name,
                "seed": seed if seed is not None else -1,
                "run_id": int(run_id),
                "cv_rate_pct": int(rate),
                "n_rows": int(len(gt)),
                "mae_ft": mae,
                "rmse_ft": rmse,
                "abc_ft_s": abc,
                **cpe,
            }
        )

    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["study_type", "case_name", "cv_rate_pct"]

    summary = (
        metrics.groupby(group_cols, as_index=False)
        .agg(
            n_runs=("run_id", "nunique"),
            n_rows=("n_rows", "sum"),
            mae_ft=("mae_ft", "mean"),
            rmse_ft=("rmse_ft", "mean"),
            abc_ft_s=("abc_ft_s", "mean"),
            mean_cycle_peak_abs_error_ft=("mean_cycle_peak_abs_error_ft", "mean"),
            rmse_cycle_peak_error_ft=("rmse_cycle_peak_error_ft", "mean"),
        )
    )

    return summary.sort_values(["study_type", "case_name", "cv_rate_pct"]).reset_index(drop=True)


# =============================================================================
# 5. ABLATION STUDY
# =============================================================================

def run_ablation_study(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["run_id"].isin(TRAIN_RUN_IDS)].copy().reset_index(drop=True)
    valid_df = df[df["run_id"].isin(VALIDATION_RUN_IDS)].copy().reset_index(drop=True)
    test_df = df[df["run_id"].isin(TEST_RUN_IDS)].copy().reset_index(drop=True)

    all_metrics = []
    training_rows = []

    for case_name, case_info in ABLATION_CASES.items():
        spec = make_feature_spec(case_name, case_info)

        model, pre, info = train_gru_model(
            train_df=train_df,
            valid_df=valid_df,
            spec=spec,
            seed=GLOBAL_SEED,
        )

        pred_df = test_df.copy()
        residual_pred = predict_residuals_groupwise(model, pre, pred_df)

        pred_df["q_pred_supplemental_ft"] = np.maximum(
            pred_df["q_physics_baseline_ft"].to_numpy(dtype=float) + residual_pred,
            0.0,
        )

        metrics = evaluate_predictions(
            df=pred_df,
            pred_col="q_pred_supplemental_ft",
            case_name=case_name,
            study_type="ablation",
            seed=None,
        )
        all_metrics.append(metrics)

        training_rows.append(
            {
                "study_type": "ablation",
                "case_name": case_name,
                "description": spec.description,
                "n_numeric_features": len(spec.numeric_cols),
                "use_phase_features": spec.use_phase,
                "best_valid_mse": info["best_valid_mse"],
                "n_train_windows": info["n_train_windows"],
                "n_valid_windows": info["n_valid_windows"],
            }
        )

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    summary = summarize_metrics(metrics_all)

    training_info = pd.DataFrame(training_rows)
    training_info.to_csv(OUT_DIR / "ablation_training_info.csv", index=False)

    return metrics_all, summary


# =============================================================================
# 6. ROBUSTNESS STUDY
# =============================================================================

def perturb_test_features(df: pd.DataFrame, case_name: str, case_info: dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    kind = str(case_info.get("type", "none"))

    if kind == "none":
        return out

    if kind == "cv_queue_noise":
        sigma = float(case_info["sigma_ft"])
        for col in CV_QUEUE_VALUE_FEATURES:
            if col in out.columns:
                noise = rng.normal(0.0, sigma, size=len(out))
                out[col] = np.maximum(pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float) + noise, 0.0)
        return out

    if kind == "cv_timing_jitter":
        sigma = float(case_info["sigma_sec"])
        for col in ["time_since_prev_cv_sec", "time_to_next_cv_sec", "cv_segment_duration_sec"]:
            if col in out.columns:
                noise = rng.normal(0.0, sigma, size=len(out))
                out[col] = np.maximum(pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float) + noise, 0.0)
        if "cv_segment_frac" in out.columns:
            noise = rng.normal(0.0, 0.05, size=len(out))
            out["cv_segment_frac"] = np.clip(pd.to_numeric(out["cv_segment_frac"], errors="coerce").to_numpy(dtype=float) + noise, 0.0, 1.0)
        return out

    if kind == "count_noise":
        sigma = float(case_info["sigma_count"])
        for col in COUNT_PHYSICS_FEATURES:
            if col in out.columns:
                noise = rng.normal(0.0, sigma, size=len(out))
                out[col] = np.maximum(pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float) + noise, 0.0)
        return out

    if kind == "phase_elapsed_noise":
        sigma = float(case_info["sigma_sec"])
        if "phase_elapsed_sec" in out.columns:
            noise = rng.normal(0.0, sigma, size=len(out))
            out["phase_elapsed_sec"] = np.maximum(pd.to_numeric(out["phase_elapsed_sec"], errors="coerce").to_numpy(dtype=float) + noise, 0.0)
        return out

    if kind == "cv_feature_dropout":
        p = float(case_info["dropout_prob"])
        mask = rng.random(len(out)) < p

        for col in CV_ANCHOR_FEATURES:
            if col not in out.columns:
                continue

            arr = (
                pd.to_numeric(out[col], errors="coerce")
                .to_numpy(dtype=float)
                .copy()
            )
            arr[mask] = 0.0
            out[col] = arr

        return out

    raise ValueError(f"Unknown robustness perturbation type: {kind}")


def run_robustness_study(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["run_id"].isin(TRAIN_RUN_IDS)].copy().reset_index(drop=True)
    valid_df = df[df["run_id"].isin(VALIDATION_RUN_IDS)].copy().reset_index(drop=True)
    test_df_clean = df[df["run_id"].isin(TEST_RUN_IDS)].copy().reset_index(drop=True)

    full_spec = make_feature_spec("full_model", ABLATION_CASES["full_model"])

    model, pre, info = train_gru_model(
        train_df=train_df,
        valid_df=valid_df,
        spec=full_spec,
        seed=GLOBAL_SEED,
    )

    all_metrics = []

    for case_name, case_info in ROBUSTNESS_CASES.items():
        seeds = [GLOBAL_SEED] if case_name == "clean" else ROBUSTNESS_SEEDS

        for seed in seeds:
            print(f"\n[Robustness] case={case_name}, seed={seed}")

            perturbed = perturb_test_features(
                df=test_df_clean,
                case_name=case_name,
                case_info=case_info,
                seed=seed,
            )

            residual_pred = predict_residuals_groupwise(model, pre, perturbed)

            pred_df = perturbed.copy()
            pred_df["q_pred_supplemental_ft"] = np.maximum(
                pred_df["q_physics_baseline_ft"].to_numpy(dtype=float) + residual_pred,
                0.0,
            )

            metrics = evaluate_predictions(
                df=pred_df,
                pred_col="q_pred_supplemental_ft",
                case_name=case_name,
                study_type="robustness",
                seed=seed,
            )
            all_metrics.append(metrics)

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    summary = summarize_metrics(metrics_all)

    clean = summary[summary["case_name"] == "clean"].copy()
    clean = clean.rename(
        columns={
            "mae_ft": "clean_mae_ft",
            "rmse_ft": "clean_rmse_ft",
            "abc_ft_s": "clean_abc_ft_s",
            "mean_cycle_peak_abs_error_ft": "clean_mean_cycle_peak_abs_error_ft",
        }
    )

    deg = summary.merge(
        clean[
            [
                "cv_rate_pct",
                "clean_mae_ft",
                "clean_rmse_ft",
                "clean_abc_ft_s",
                "clean_mean_cycle_peak_abs_error_ft",
            ]
        ],
        on="cv_rate_pct",
        how="left",
    )

    deg["rmse_pct_change_vs_clean"] = 100.0 * (deg["rmse_ft"] - deg["clean_rmse_ft"]) / deg["clean_rmse_ft"]
    deg["mae_pct_change_vs_clean"] = 100.0 * (deg["mae_ft"] - deg["clean_mae_ft"]) / deg["clean_mae_ft"]
    deg["cycle_peak_pct_change_vs_clean"] = (
        100.0
        * (deg["mean_cycle_peak_abs_error_ft"] - deg["clean_mean_cycle_peak_abs_error_ft"])
        / deg["clean_mean_cycle_peak_abs_error_ft"]
    )

    training_info = pd.DataFrame(
        [
            {
                "study_type": "robustness",
                "case_name": "full_model_clean_training",
                "description": "Full Physics + ML + CV GRU model trained once on clean training data",
                "best_valid_mse": info["best_valid_mse"],
                "n_train_windows": info["n_train_windows"],
                "n_valid_windows": info["n_valid_windows"],
            }
        ]
    )
    training_info.to_csv(OUT_DIR / "robustness_training_info.csv", index=False)

    return metrics_all, summary, deg


# =============================================================================
# 7. SUPPLEMENTAL TABLES AND PLOTS
# =============================================================================

def format_table(summary: pd.DataFrame, selected_rate: int, case_order: list[str] | None = None) -> pd.DataFrame:
    d = summary[pd.to_numeric(summary["cv_rate_pct"], errors="coerce") == int(selected_rate)].copy()

    if case_order is not None:
        order = {c: i for i, c in enumerate(case_order)}
        d["_order"] = d["case_name"].map(order).fillna(999).astype(int)
        d = d.sort_values("_order").drop(columns="_order")
    else:
        d = d.sort_values("case_name")

    out = pd.DataFrame()
    out["Case"] = d["case_name"].astype(str)

    for metric in TABLE_METRICS:
        if metric not in d.columns:
            continue
        label = {
            "mae_ft": "MAE (ft)",
            "rmse_ft": "RMSE (ft)",
            "abc_ft_s": "ABC (ft·s)",
            "mean_cycle_peak_abs_error_ft": "Mean cycle peak error (ft)",
        }.get(metric, metric)

        if metric == "abc_ft_s":
            out[label] = d[metric].map(lambda x: "" if pd.isna(x) else f"{float(x):,.0f}")
        else:
            out[label] = d[metric].map(lambda x: "" if pd.isna(x) else f"{float(x):.2f}")

    return out


def make_supplemental_outputs(
    ablation_summary: pd.DataFrame | None,
    robustness_summary: pd.DataFrame | None,
    robustness_degradation: pd.DataFrame | None,
) -> None:
    if ablation_summary is not None and not ablation_summary.empty:
        ablation_table = format_table(
            ablation_summary,
            selected_rate=SELECTED_TABLE_RATE,
            case_order=list(ABLATION_CASES.keys()),
        )
        ablation_table.to_csv(
            OUT_DIR / f"ablation_summary_table_rate{SELECTED_TABLE_RATE:03d}.csv",
            index=False,
        )
        save_table_png(
            ablation_table,
            f"ablation_summary_table_rate{SELECTED_TABLE_RATE:03d}.png",
            f"Supplemental ablation study | CV {SELECTED_TABLE_RATE}%",
        )

        save_line_plot(
            df=ablation_summary,
            x_col="cv_rate_pct",
            y_col="rmse_ft",
            group_col="case_name",
            title="Supplemental ablation study | RMSE by CV penetration rate",
            ylabel="RMSE (ft)",
            filename="ablation_rmse_by_cv_rate.png",
        )

        save_line_plot(
            df=ablation_summary,
            x_col="cv_rate_pct",
            y_col="mean_cycle_peak_abs_error_ft",
            group_col="case_name",
            title="Supplemental ablation study | Cycle peak error by CV penetration rate",
            ylabel="Mean cycle peak absolute error (ft)",
            filename="ablation_cycle_peak_error_by_cv_rate.png",
        )

    if robustness_summary is not None and not robustness_summary.empty:
        robustness_table = format_table(
            robustness_summary,
            selected_rate=SELECTED_TABLE_RATE,
            case_order=list(ROBUSTNESS_CASES.keys()),
        )
        robustness_table.to_csv(
            OUT_DIR / f"robustness_summary_table_rate{SELECTED_TABLE_RATE:03d}.csv",
            index=False,
        )
        save_table_png(
            robustness_table,
            f"robustness_summary_table_rate{SELECTED_TABLE_RATE:03d}.png",
            f"Supplemental robustness study | CV {SELECTED_TABLE_RATE}%",
        )

        # Main robustness plot: moderate perturbation cases only.
    robustness_moderate = robustness_summary[
        robustness_summary["case_name"] != "cv_feature_dropout_20pct"
    ].copy()

    save_line_plot(
        df=robustness_moderate,
        x_col="cv_rate_pct",
        y_col="rmse_ft",
        group_col="case_name",
        title="Supplemental robustness study | Moderate perturbations",
        ylabel="RMSE (ft)",
        filename="robustness_rmse_by_cv_rate_moderate_perturbations.png",
    )

    # Separate stress-test plot: CV-anchor feature dropout.
    robustness_dropout = robustness_summary[
        robustness_summary["case_name"].isin(["clean", "cv_feature_dropout_20pct"])
    ].copy()

    save_line_plot(
        df=robustness_dropout,
        x_col="cv_rate_pct",
        y_col="rmse_ft",
        group_col="case_name",
        title="CV-anchor feature dropout stress test",
        ylabel="RMSE (ft)",
        filename="robustness_cv_feature_dropout_stress_test.png",
    )

    if robustness_degradation is not None and not robustness_degradation.empty:
        d = robustness_degradation[
            pd.to_numeric(robustness_degradation["cv_rate_pct"], errors="coerce") == int(SELECTED_TABLE_RATE)
        ].copy()

        d = d[d["case_name"] != "clean"].copy()

        d_moderate = d[d["case_name"] != "cv_feature_dropout_20pct"].copy()
        d_dropout = d[d["case_name"] == "cv_feature_dropout_20pct"].copy()
        d = d.sort_values("rmse_pct_change_vs_clean")

        out = pd.DataFrame()
        out["Case"] = d["case_name"].astype(str)
        out["RMSE change vs clean (%)"] = d["rmse_pct_change_vs_clean"].map(
            lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
        )
        out["MAE change vs clean (%)"] = d["mae_pct_change_vs_clean"].map(
            lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
        )
        out["Cycle peak change vs clean (%)"] = d["cycle_peak_pct_change_vs_clean"].map(
            lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
        )

        out.to_csv(
            OUT_DIR / f"robustness_degradation_table_rate{SELECTED_TABLE_RATE:03d}.csv",
            index=False,
        )
        save_table_png(
            out,
            f"robustness_degradation_table_rate{SELECTED_TABLE_RATE:03d}.png",
            f"Robustness degradation relative to clean | CV {SELECTED_TABLE_RATE}%",
        )

        fig, ax = plt.subplots(figsize=(10.8, 6.0))
        ax.bar(
            d_moderate["case_name"].astype(str),
            pd.to_numeric(d_moderate["rmse_pct_change_vs_clean"], errors="coerce"),
        )
        ax.set_title(f"Moderate perturbation degradation | CV {SELECTED_TABLE_RATE}%")
        ax.set_ylabel("RMSE change relative to clean (%)")
        ax.set_xlabel("Perturbation case")
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()

        out_path = FIG_DIR / f"robustness_rmse_degradation_moderate_rate{SELECTED_TABLE_RATE:03d}.png"
        fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close(fig)
        print(f"[Saved figure] {out_path}")

        if not d_dropout.empty:
            fig, ax = plt.subplots(figsize=(7.5, 5.2))
            ax.bar(
                d_dropout["case_name"].astype(str),
                pd.to_numeric(d_dropout["rmse_pct_change_vs_clean"], errors="coerce"),
            )
            ax.set_title(f"CV-anchor feature dropout stress test | CV {SELECTED_TABLE_RATE}%")
            ax.set_ylabel("RMSE change relative to clean (%)")
            ax.set_xlabel("Stress-test case")
            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()

            out_path = FIG_DIR / f"robustness_rmse_degradation_dropout_stress_rate{SELECTED_TABLE_RATE:03d}.png"
            fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
            if SHOW_FIGURES:
                plt.show()
            else:
                plt.close(fig)
            print(f"[Saved figure] {out_path}")
        fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close(fig)
        print(f"[Saved figure] {out_path}")


# =============================================================================
# 8. MAIN
# =============================================================================

def main() -> None:
    ensure_dirs()
    set_seed(GLOBAL_SEED)

    print("=" * 100)
    print("Supplemental robustness and ablation studies")
    print("=" * 100)
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Feature file       : {FEATURE_FILE}")
    print(f"Output folder      : {OUT_DIR}")
    print(f"Train runs         : {TRAIN_RUN_IDS}")
    print(f"Validation runs    : {VALIDATION_RUN_IDS}")
    print(f"Test runs          : {TEST_RUN_IDS}")
    print(f"CV rates           : {CV_RATES_PCT}")
    print(f"Model              : {MODEL_NAME}")
    print(f"Sequence length    : {SEQUENCE_LENGTH}")
    print(f"Max train windows  : {MAX_TRAIN_WINDOWS_PER_CASE}")
    print(f"Max valid windows  : {MAX_VALID_WINDOWS_PER_CASE}")
    print("=" * 100)

    df = load_feature_data()

    ablation_metrics = None
    ablation_summary = None
    robustness_metrics = None
    robustness_summary = None
    robustness_degradation = None

    if MAKE_ABLATION_STUDY:
        print("\n" + "=" * 100)
        print("Running ablation study")
        print("=" * 100)

        ablation_metrics, ablation_summary = run_ablation_study(df)

        path1 = OUT_DIR / "ablation_metrics_by_case_run_rate.csv"
        path2 = OUT_DIR / "ablation_summary_by_case_rate.csv"

        ablation_metrics.to_csv(path1, index=False)
        ablation_summary.to_csv(path2, index=False)

        print(f"[Saved] {path1}")
        print(f"[Saved] {path2}")

    if MAKE_ROBUSTNESS_STUDY:
        print("\n" + "=" * 100)
        print("Running robustness study")
        print("=" * 100)

        robustness_metrics, robustness_summary, robustness_degradation = run_robustness_study(df)

        path1 = OUT_DIR / "robustness_metrics_by_case_seed_run_rate.csv"
        path2 = OUT_DIR / "robustness_summary_by_case_rate.csv"
        path3 = OUT_DIR / "robustness_degradation_summary.csv"

        robustness_metrics.to_csv(path1, index=False)
        robustness_summary.to_csv(path2, index=False)
        robustness_degradation.to_csv(path3, index=False)

        print(f"[Saved] {path1}")
        print(f"[Saved] {path2}")
        print(f"[Saved] {path3}")

    if MAKE_PLOTS_AND_TABLES:
        print("\n" + "=" * 100)
        print("Creating supplemental plots and PNG tables")
        print("=" * 100)

        # Allow plot/table generation from already-saved CSVs when training is switched off.
        if ablation_summary is None:
            p = OUT_DIR / "ablation_summary_by_case_rate.csv"
            if p.exists():
                ablation_summary = pd.read_csv(p)
                print(f"[Loaded existing ablation summary] {p}")

        if robustness_summary is None:
            p = OUT_DIR / "robustness_summary_by_case_rate.csv"
            if p.exists():
                robustness_summary = pd.read_csv(p)
                print(f"[Loaded existing robustness summary] {p}")

        if robustness_degradation is None:
            p = OUT_DIR / "robustness_degradation_summary.csv"
            if p.exists():
                robustness_degradation = pd.read_csv(p)
                print(f"[Loaded existing robustness degradation summary] {p}")

        make_supplemental_outputs(
            ablation_summary=ablation_summary,
            robustness_summary=robustness_summary,
            robustness_degradation=robustness_degradation,
        )

    print("\nDone.")
    print(f"Supplemental outputs saved under:\n{OUT_DIR}")


if __name__ == "__main__":
    main()
