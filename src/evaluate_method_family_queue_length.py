"""
Evaluate method-family queue-length predictions.

Place this file at:
    src/evaluate_method_family_queue_length.py

Purpose
-------
Build one unified queue-length evaluation table across all major method families:

    1. Physics baseline
    2. CV-only
    3. Physics + CV
    4. ML-only
    5. ML + CV
    6. Physics + ML
    7. Physics + ML + CV
    8. Optional legacy anchor-adjusted Physics + ML + CV curves are excluded
       from the current workflow.

Queue-length metrics:
    - MAE
    - RMSE
    - ABC
    - cycle-based peak queue error

Cycle definition:
    - red start to next red start
    - first and last cycle windows are dropped as partial/boundary cycles

This script saves all detailed CSVs first.
Plotting scripts should read these outputs and should not recompute metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import math
import numpy as np
import pandas as pd

from config import (
    PROJECT_ROOT,
    RUN_IDS,
    TRAIN_RUN_IDS,
    VALIDATION_RUN_IDS,
    TEST_RUN_IDS,
    CV_RATES_PCT,
)

# =============================================================================
# User settings
# =============================================================================

CV_INTERPOLATION_METHODS = ["linear"]

BEST_SELECTION_METRIC = "rmse_ft"
BEST_SELECTION_SPLIT = "validation"

CLIP_NONNEGATIVE = True

SAVE_PLOT_READY_SELECTED_WIDE = True
SAVE_PLOT_READY_CYCLE_PEAKS = True


# =============================================================================
# Paths
# =============================================================================

INTERMEDIATE_DIR = PROJECT_ROOT / "output" / "intermediate_csv"

CV_FEATURE_DIR = INTERMEDIATE_DIR / "cv_features"
ML_RAW_DIR = INTERMEDIATE_DIR / "ml_raw_predictions"
ML_DIRECT_DIR = INTERMEDIATE_DIR / "ml_direct_predictions"
ML_RESIDUAL_CV_DIR = INTERMEDIATE_DIR / "ml_residual_cv_predictions"

FEATURE_FILE = CV_FEATURE_DIR / "timegrid_features_allruns_allrates.csv"

ML_RAW_FILE = ML_RAW_DIR / "ml_raw_predictions_allruns_allrates.csv"
ML_DIRECT_FILE = ML_DIRECT_DIR / "ml_direct_predictions_allruns_allrates.csv"
ML_RESIDUAL_CV_FILE = ML_RESIDUAL_CV_DIR / "ml_residual_cv_predictions_allruns_allrates.csv"

OUT_DIR = INTERMEDIATE_DIR / "method_family_queue_length_evaluation"

PREDICTIONS_OUT = OUT_DIR / "method_family_predictions_allruns_allrates.csv"
CATALOG_OUT = OUT_DIR / "method_family_curve_catalog.csv"

METRICS_RUN_RATE_OUT = OUT_DIR / "method_family_metrics_by_run_rate_curve.csv"
METRICS_SUMMARY_SPLIT_RATE_OUT = OUT_DIR / "method_family_metrics_summary_by_split_rate_curve.csv"
METRICS_SUMMARY_VALIDATION_OUT = OUT_DIR / "method_family_metrics_summary_validation.csv"
METRICS_SUMMARY_TEST_OUT = OUT_DIR / "method_family_metrics_summary_test.csv"

CYCLE_PEAK_ERRORS_OUT = OUT_DIR / "method_family_cycle_peak_errors.csv"
CYCLE_PEAK_SUMMARY_RUN_RATE_OUT = OUT_DIR / "method_family_cycle_peak_summary_by_run_rate_curve.csv"
CYCLE_PEAK_SUMMARY_SPLIT_RATE_OUT = OUT_DIR / "method_family_cycle_peak_summary_by_split_rate_curve.csv"

BEST_MODELS_VALIDATION_OUT = OUT_DIR / "method_family_best_models_validation.csv"
PLOT_READY_SELECTED_CURVES_OUT = OUT_DIR / "method_family_plot_ready_selected_curves.csv"
PLOT_READY_CYCLE_PEAKS_OUT = OUT_DIR / "method_family_plot_ready_cycle_peaks.csv"


# =============================================================================
# Constants
# =============================================================================

KEY_COLS = ["run_id", "cv_rate_pct", "time_sec"]

GT_COL = "q_gt_ft"
BASELINE_COL = "q_baseline_fixed_ft"
PHASE_COL = "phase_state"

BASE_KEEP_COLS = [
    "run_id",
    "run_split",
    "ml_split",
    "cv_rate_pct",
    "time_sec",
    "phase_state",
    "phase_elapsed_sec",
    "q_gt_ft",
    "q_baseline_fixed_ft",
    "A_count",
    "D_count",
    "V_count",
    "B_count",
    "n_queue_cumulative",
    "l_eff_fixed_ft",
]


# =============================================================================
# Helpers
# =============================================================================

def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def format_rate(rate: int) -> str:
    return f"{int(rate):03d}"


def get_ml_split(run_id: int) -> str:
    if int(run_id) in TRAIN_RUN_IDS:
        return "train"
    if int(run_id) in VALIDATION_RUN_IDS:
        return "validation"
    if int(run_id) in TEST_RUN_IDS:
        return "test"
    return "other"


def safe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def clip_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if CLIP_NONNEGATIVE:
        q = np.maximum(q, 0.0)
    return q


def finite_arrays(
    t: np.ndarray,
    y: np.ndarray,
    yp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)

    mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(yp)
    return t[mask], y[mask], yp[mask]


def mae(y: np.ndarray, yp: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yp)

    if mask.sum() == 0:
        return np.nan

    return float(np.mean(np.abs(y[mask] - yp[mask])))


def rmse(y: np.ndarray, yp: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yp = np.asarray(yp, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yp)

    if mask.sum() == 0:
        return np.nan

    return float(math.sqrt(np.mean((y[mask] - yp[mask]) ** 2)))


def abc(t: np.ndarray, y: np.ndarray, yp: np.ndarray) -> float:
    t, y, yp = finite_arrays(t, y, yp)

    if len(t) < 2:
        return np.nan

    order = np.argsort(t)
    abs_err = np.abs(y - yp)

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(abs_err[order], t[order]))

    return float(np.trapz(abs_err[order], t[order]))


def rmse_from_values(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return np.nan

    return float(math.sqrt(np.mean(arr ** 2)))


# =============================================================================
# Loading
# =============================================================================

def load_base_feature_table() -> pd.DataFrame:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Missing feature file:\n{FEATURE_FILE}\n"
            "Run src/build_cv_features.py first."
        )

    df = pd.read_csv(FEATURE_FILE)

    require_columns(
        df,
        ["run_id", "cv_rate_pct", "time_sec", GT_COL, BASELINE_COL, PHASE_COL],
        "feature table",
    )

    keep = [c for c in BASE_KEEP_COLS if c in df.columns]
    df = df[keep].copy()

    numeric_cols = [
        "run_id",
        "cv_rate_pct",
        "time_sec",
        "phase_elapsed_sec",
        GT_COL,
        BASELINE_COL,
        "A_count",
        "D_count",
        "V_count",
        "B_count",
        "n_queue_cumulative",
        "l_eff_fixed_ft",
    ]
    df = safe_numeric(df, numeric_cols)

    df = df.dropna(subset=["run_id", "cv_rate_pct", "time_sec", GT_COL, BASELINE_COL]).copy()

    df["run_id"] = df["run_id"].astype(int)
    df["cv_rate_pct"] = df["cv_rate_pct"].astype(int)

    df = df[df["run_id"].isin(RUN_IDS)].copy()
    df = df[df["cv_rate_pct"].isin(CV_RATES_PCT)].copy()

    df[PHASE_COL] = df[PHASE_COL].astype(str).str.strip().str.lower()

    df["ml_split"] = df["run_id"].apply(get_ml_split)

    df["q_physics_baseline_ft"] = pd.to_numeric(df[BASELINE_COL], errors="coerce")

    df = df.sort_values(KEY_COLS).reset_index(drop=True)

    return df


def read_prediction_file(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[WARN] Missing {label}: {path}")
        return None

    df = pd.read_csv(path)
    require_columns(df, KEY_COLS, label)

    df = safe_numeric(df, KEY_COLS)
    df = df.dropna(subset=KEY_COLS).copy()
    df["run_id"] = df["run_id"].astype(int)
    df["cv_rate_pct"] = df["cv_rate_pct"].astype(int)
    df = df.sort_values(KEY_COLS).drop_duplicates(KEY_COLS, keep="last").reset_index(drop=True)

    return df


def merge_selected_prediction_columns(
    base: pd.DataFrame,
    source: pd.DataFrame | None,
    rename_map: dict[str, str],
    source_label: str,
) -> pd.DataFrame:
    out = base.copy()

    if source is None:
        return out

    available = [c for c in rename_map if c in source.columns]

    if not available:
        print(f"[WARN] No selected columns found in {source_label}.")
        return out

    keep = KEY_COLS + available
    temp = source[keep].copy()
    temp = temp.rename(columns={c: rename_map[c] for c in available})

    for c in rename_map.values():
        if c in temp.columns:
            temp[c] = pd.to_numeric(temp[c], errors="coerce")

    out = out.merge(temp, on=KEY_COLS, how="left")

    print(f"[Merged] {source_label}: {len(available)} columns")
    return out


# =============================================================================
# CV interpolation and physics + CV correction
# =============================================================================

def anchor_file_path(run_id: int, rate: int) -> Path:
    return CV_FEATURE_DIR / f"cv_anchors_run{int(run_id):03d}_rate{int(rate):03d}.csv"


def load_cv_anchors(run_id: int, rate: int) -> pd.DataFrame:
    path = anchor_file_path(run_id, rate)

    if not path.exists():
        return pd.DataFrame(columns=["cv_anchor_time_sec", "cv_anchor_q_ft"])

    a = pd.read_csv(path)

    if "cv_anchor_time_sec" not in a.columns or "cv_anchor_q_ft" not in a.columns:
        return pd.DataFrame(columns=["cv_anchor_time_sec", "cv_anchor_q_ft"])

    a = a[["cv_anchor_time_sec", "cv_anchor_q_ft"]].copy()
    a["cv_anchor_time_sec"] = pd.to_numeric(a["cv_anchor_time_sec"], errors="coerce")
    a["cv_anchor_q_ft"] = pd.to_numeric(a["cv_anchor_q_ft"], errors="coerce")
    a = a.dropna(subset=["cv_anchor_time_sec", "cv_anchor_q_ft"]).copy()

    if a.empty:
        return a

    a = (
        a.sort_values("cv_anchor_time_sec")
        .drop_duplicates("cv_anchor_time_sec", keep="last")
        .reset_index(drop=True)
    )

    return a


def interpolate_series(
    anchor_t: np.ndarray,
    anchor_y: np.ndarray,
    grid_t: np.ndarray,
    method: str,
) -> np.ndarray:
    anchor_t = np.asarray(anchor_t, dtype=float)
    anchor_y = np.asarray(anchor_y, dtype=float)
    grid_t = np.asarray(grid_t, dtype=float)

    mask = np.isfinite(anchor_t) & np.isfinite(anchor_y)
    anchor_t = anchor_t[mask]
    anchor_y = anchor_y[mask]

    if len(anchor_t) == 0:
        return np.full(len(grid_t), np.nan, dtype=float)

    order = np.argsort(anchor_t)
    anchor_t = anchor_t[order]
    anchor_y = anchor_y[order]

    # Remove duplicate times after sorting.
    temp = pd.DataFrame({"t": anchor_t, "y": anchor_y})
    temp = temp.drop_duplicates("t", keep="last")
    anchor_t = temp["t"].to_numpy(dtype=float)
    anchor_y = temp["y"].to_numpy(dtype=float)

    if len(anchor_t) == 1:
        return np.full(len(grid_t), float(anchor_y[0]), dtype=float)

    method = str(method).lower().strip()

    if method == "linear":
        return np.interp(grid_t, anchor_t, anchor_y).astype(float)

    raise ValueError(f"Unknown interpolation method: {method}")


def add_cv_only_and_physics_cv_curves(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()

    for method in CV_INTERPOLATION_METHODS:
        out[f"q_cv_only_{method}_ft"] = np.nan
        out[f"q_physics_cv_{method}_ft"] = np.nan

    for (run_id, rate), idx in out.groupby(["run_id", "cv_rate_pct"], sort=True).groups.items():
        idx = np.asarray(list(idx), dtype=int)

        g = out.loc[idx].sort_values("time_sec").copy()
        grid_t = g["time_sec"].to_numpy(dtype=float)
        baseline = g["q_physics_baseline_ft"].to_numpy(dtype=float)

        anchors = load_cv_anchors(int(run_id), int(rate))

        if anchors.empty:
            print(f"[WARN] No CV anchors for run {int(run_id):03d}, rate {int(rate):03d}")
            continue

        anchor_t = anchors["cv_anchor_time_sec"].to_numpy(dtype=float)
        anchor_q = anchors["cv_anchor_q_ft"].to_numpy(dtype=float)

        baseline_at_anchor = np.interp(anchor_t, grid_t, baseline)
        anchor_residual = anchor_q - baseline_at_anchor

        for method in CV_INTERPOLATION_METHODS:
            q_cv = interpolate_series(anchor_t, anchor_q, grid_t, method=method)
            residual_interp = interpolate_series(anchor_t, anchor_residual, grid_t, method=method)

            q_physics_cv = baseline + residual_interp

            q_cv = clip_q(q_cv)
            q_physics_cv = clip_q(q_physics_cv)

            out.loc[g.index, f"q_cv_only_{method}_ft"] = q_cv
            out.loc[g.index, f"q_physics_cv_{method}_ft"] = q_physics_cv

    return out


# =============================================================================
# Curve catalog
# =============================================================================

def make_catalog(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add_curve(
        curve_id: str,
        method_family: str,
        model_name: str,
        curve_label: str,
        prediction_col: str,
        source: str,
        uses_physics: bool,
        uses_ml: bool,
        uses_cv: bool,
        interpolation_method: str = "",
        include_in_best_selection: bool = True,
        is_reference: bool = False,
    ) -> None:
        if prediction_col not in pred.columns:
            return

        rows.append(
            {
                "curve_id": curve_id,
                "method_family": method_family,
                "model_name": model_name,
                "curve_label": curve_label,
                "prediction_col": prediction_col,
                "source": source,
                "uses_physics": int(uses_physics),
                "uses_ml": int(uses_ml),
                "uses_cv": int(uses_cv),
                "interpolation_method": interpolation_method,
                "include_in_best_selection": int(include_in_best_selection),
                "is_reference": int(is_reference),
            }
        )

    add_curve(
        "gt",
        "GT",
        "GT",
        "GT",
        GT_COL,
        "feature_table",
        uses_physics=False,
        uses_ml=False,
        uses_cv=False,
        include_in_best_selection=False,
        is_reference=True,
    )

    add_curve(
        "physics_baseline",
        "Physics baseline",
        "baseline",
        "Physics baseline",
        "q_physics_baseline_ft",
        "feature_table",
        uses_physics=True,
        uses_ml=False,
        uses_cv=False,
    )

    for method in CV_INTERPOLATION_METHODS:
        add_curve(
            f"cv_only_{method}",
            "CV-only",
            method,
            f"CV-only ({method})",
            f"q_cv_only_{method}_ft",
            "computed_from_cv_anchors",
            uses_physics=False,
            uses_ml=False,
            uses_cv=True,
            interpolation_method=method,
        )

        add_curve(
            f"physics_cv_{method}",
            "Physics + CV",
            method,
            f"Physics + CV ({method})",
            f"q_physics_cv_{method}_ft",
            "computed_from_cv_anchors",
            uses_physics=True,
            uses_ml=False,
            uses_cv=True,
            interpolation_method=method,
        )

    for model in ["xgb", "gru", "lstm"]:
        add_curve(
            f"ml_only_{model}",
            "ML-only",
            model.upper() if model == "gru" or model == "lstm" else "XGBoost",
            f"ML-only ({model.upper() if model != 'xgb' else 'XGBoost'})",
            f"q_pred_{model}_ml_only_ft",
            "ml_direct_predictions",
            uses_physics=False,
            uses_ml=True,
            uses_cv=False,
        )

        add_curve(
            f"ml_cv_{model}",
            "ML + CV",
            model.upper() if model == "gru" or model == "lstm" else "XGBoost",
            f"ML + CV ({model.upper() if model != 'xgb' else 'XGBoost'})",
            f"q_pred_{model}_ml_cv_ft",
            "ml_direct_predictions",
            uses_physics=False,
            uses_ml=True,
            uses_cv=True,
        )

        add_curve(
            f"physics_ml_{model}",
            "Physics + ML",
            model.upper() if model == "gru" or model == "lstm" else "XGBoost",
            f"Physics + ML ({model.upper() if model != 'xgb' else 'XGBoost'})",
            f"q_pred_{model}_physics_ml_ft",
            "ml_raw_predictions",
            uses_physics=True,
            uses_ml=True,
            uses_cv=False,
        )

        add_curve(
            f"physics_ml_cv_{model}",
            "Physics + ML + CV",
            model.upper() if model == "gru" or model == "lstm" else "XGBoost",
            f"Physics + ML + CV ({model.upper() if model != 'xgb' else 'XGBoost'})",
            f"q_pred_{model}_physics_ml_cv_ft",
            "ml_residual_cv_predictions",
            uses_physics=True,
            uses_ml=True,
            uses_cv=True,
        )

    catalog = pd.DataFrame(rows)
    return catalog


# =============================================================================
# Merge all predictions
# =============================================================================

def build_unified_prediction_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = load_base_feature_table()

    print(f"[Loaded base feature table] {len(pred):,} rows")

    pred = add_cv_only_and_physics_cv_curves(pred)

    # Direct ML: ML-only and ML + CV.
    direct = read_prediction_file(ML_DIRECT_FILE, "ML direct predictions")
    direct_map = {
        "q_pred_xgb_ml_only_ft": "q_pred_xgb_ml_only_ft",
        "q_pred_gru_ml_only_ft": "q_pred_gru_ml_only_ft",
        "q_pred_lstm_ml_only_ft": "q_pred_lstm_ml_only_ft",
        "q_pred_xgb_ml_cv_ft": "q_pred_xgb_ml_cv_ft",
        "q_pred_gru_ml_cv_ft": "q_pred_gru_ml_cv_ft",
        "q_pred_lstm_ml_cv_ft": "q_pred_lstm_ml_cv_ft",
    }
    pred = merge_selected_prediction_columns(pred, direct, direct_map, "ML direct predictions")

    # Physics + ML residual raw predictions.
    raw = read_prediction_file(ML_RAW_FILE, "Physics + ML raw predictions")
    raw_map = {
        "q_pred_xgb_raw_ft": "q_pred_xgb_physics_ml_ft",
        "q_pred_gru_raw_ft": "q_pred_gru_physics_ml_ft",
        "q_pred_lstm_raw_ft": "q_pred_lstm_physics_ml_ft",
    }
    pred = merge_selected_prediction_columns(pred, raw, raw_map, "Physics + ML raw predictions")

    # Physics + ML + CV residual predictions.
    residual_cv = read_prediction_file(ML_RESIDUAL_CV_FILE, "Physics + ML + CV predictions")
    residual_cv_map = {
        "q_pred_xgb_physics_ml_cv_ft": "q_pred_xgb_physics_ml_cv_ft",
        "q_pred_gru_physics_ml_cv_ft": "q_pred_gru_physics_ml_cv_ft",
        "q_pred_lstm_physics_ml_cv_ft": "q_pred_lstm_physics_ml_cv_ft",
    }
    pred = merge_selected_prediction_columns(
        pred,
        residual_cv,
        residual_cv_map,
        "Physics + ML + CV predictions",
    )

    catalog = make_catalog(pred)

    return pred, catalog


# =============================================================================
# Pointwise metrics
# =============================================================================

def compute_pointwise_metrics(pred: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    rows = []

    curves = catalog[catalog["is_reference"] == 0].copy()

    for _, curve in curves.iterrows():
        curve_id = curve["curve_id"]
        q_col = curve["prediction_col"]

        if q_col not in pred.columns:
            continue

        for (run_id, rate), g in pred.groupby(["run_id", "cv_rate_pct"], sort=True):
            t = g["time_sec"].to_numpy(dtype=float)
            y = g[GT_COL].to_numpy(dtype=float)
            yp = g[q_col].to_numpy(dtype=float)

            valid_mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(yp)

            split_values = sorted(set(g["ml_split"].astype(str)))
            split_label = ",".join(split_values) if split_values else get_ml_split(int(run_id))

            rows.append(
                {
                    "run_id": int(run_id),
                    "cv_rate_pct": int(rate),
                    "ml_split": split_label,
                    "curve_id": curve_id,
                    "method_family": curve["method_family"],
                    "model_name": curve["model_name"],
                    "curve_label": curve["curve_label"],
                    "prediction_col": q_col,
                    "source": curve["source"],
                    "uses_physics": int(curve["uses_physics"]),
                    "uses_ml": int(curve["uses_ml"]),
                    "uses_cv": int(curve["uses_cv"]),
                    "interpolation_method": curve["interpolation_method"],
                    "n_samples": int(len(g)),
                    "n_valid": int(valid_mask.sum()),
                    "valid_pct": float(100.0 * valid_mask.sum() / len(g)) if len(g) else np.nan,
                    "mae_ft": mae(y, yp),
                    "rmse_ft": rmse(y, yp),
                    "abc_ft_s": abc(t, y, yp),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Cycle detection and cycle peak metrics
# =============================================================================

def phase_intervals_from_timegrid(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["time_sec", PHASE_COL]].copy()
    d["time_sec"] = pd.to_numeric(d["time_sec"], errors="coerce")
    d[PHASE_COL] = d[PHASE_COL].astype(str).str.strip().str.lower()
    d = d.dropna(subset=["time_sec"]).sort_values("time_sec")
    d = d.drop_duplicates("time_sec", keep="last").reset_index(drop=True)

    if len(d) < 2:
        return pd.DataFrame(columns=["start_sec", "end_sec", "phase_state"])

    change = d[PHASE_COL].ne(d[PHASE_COL].shift()).astype(int)
    d["_phase_segment_id"] = change.cumsum()

    intervals = (
        d.groupby("_phase_segment_id", as_index=False)
        .agg(
            start_sec=("time_sec", "first"),
            phase_state=(PHASE_COL, "first"),
        )
        .sort_values("start_sec")
        .reset_index(drop=True)
    )

    intervals["end_sec"] = intervals["start_sec"].shift(-1)

    fallback_step = float(np.nanmedian(np.diff(d["time_sec"].to_numpy(dtype=float)))) if len(d) > 1 else 1.0
    if not np.isfinite(fallback_step) or fallback_step <= 0:
        fallback_step = 1.0

    intervals["end_sec"] = intervals["end_sec"].fillna(float(d["time_sec"].max()) + fallback_step)
    intervals = intervals[intervals["end_sec"] > intervals["start_sec"]].copy()

    return intervals[["start_sec", "end_sec", "phase_state"]]


def red_to_red_cycle_windows_for_run(run_df: pd.DataFrame) -> pd.DataFrame:
    intervals = phase_intervals_from_timegrid(run_df)

    if intervals.empty:
        return pd.DataFrame(columns=["cycle_id", "cycle_start_sec", "cycle_end_sec"])

    intervals["prev_phase"] = intervals["phase_state"].shift()

    red_starts = intervals[
        (intervals["phase_state"] == "red")
        & (intervals["prev_phase"] != "red")
    ]["start_sec"].to_numpy(dtype=float)

    red_starts = np.sort(red_starts[np.isfinite(red_starts)])

    if len(red_starts) < 2:
        return pd.DataFrame(columns=["cycle_id", "cycle_start_sec", "cycle_end_sec"])

    raw_windows = []
    for i in range(len(red_starts) - 1):
        raw_windows.append((float(red_starts[i]), float(red_starts[i + 1])))

    # Drop first and last boundary/partial cycles.
    if len(raw_windows) <= 2:
        return pd.DataFrame(columns=["cycle_id", "cycle_start_sec", "cycle_end_sec"])

    kept_windows = raw_windows[1:-1]

    rows = []
    for i, (start, end) in enumerate(kept_windows, start=1):
        rows.append(
            {
                "cycle_id": int(i),
                "cycle_start_sec": float(start),
                "cycle_end_sec": float(end),
            }
        )

    return pd.DataFrame(rows)


def build_cycle_windows(pred: pd.DataFrame) -> dict[int, pd.DataFrame]:
    windows_by_run = {}

    for run_id, g in pred.groupby("run_id", sort=True):
        # Phase is repeated by CV rate, so use one rate only.
        one_rate = g[g["cv_rate_pct"] == g["cv_rate_pct"].min()].copy()
        windows_by_run[int(run_id)] = red_to_red_cycle_windows_for_run(one_rate)

    return windows_by_run


def compute_cycle_peak_errors(pred: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    windows_by_run = build_cycle_windows(pred)
    rows = []

    curves = catalog[catalog["is_reference"] == 0].copy()

    for _, curve in curves.iterrows():
        curve_id = curve["curve_id"]
        q_col = curve["prediction_col"]

        if q_col not in pred.columns:
            continue

        for (run_id, rate), g in pred.groupby(["run_id", "cv_rate_pct"], sort=True):
            windows = windows_by_run.get(int(run_id), pd.DataFrame())

            if windows.empty:
                continue

            g = g.sort_values("time_sec").copy()

            for _, cyc in windows.iterrows():
                start = float(cyc["cycle_start_sec"])
                end = float(cyc["cycle_end_sec"])

                sub = g[
                    (g["time_sec"] >= start)
                    & (g["time_sec"] < end)
                ].copy()

                if sub.empty:
                    continue

                qgt = pd.to_numeric(sub[GT_COL], errors="coerce")
                qpred = pd.to_numeric(sub[q_col], errors="coerce")

                if qgt.notna().sum() == 0 or qpred.notna().sum() == 0:
                    continue

                idx_gt = qgt.idxmax()
                idx_pred = qpred.idxmax()

                qgt_peak = float(qgt.loc[idx_gt])
                qpred_peak = float(qpred.loc[idx_pred])

                split_values = sorted(set(sub["ml_split"].astype(str)))
                split_label = ",".join(split_values) if split_values else get_ml_split(int(run_id))

                rows.append(
                    {
                        "run_id": int(run_id),
                        "cv_rate_pct": int(rate),
                        "ml_split": split_label,
                        "cycle_id": int(cyc["cycle_id"]),
                        "cycle_start_sec": start,
                        "cycle_end_sec": end,
                        "curve_id": curve_id,
                        "method_family": curve["method_family"],
                        "model_name": curve["model_name"],
                        "curve_label": curve["curve_label"],
                        "prediction_col": q_col,
                        "source": curve["source"],
                        "interpolation_method": curve["interpolation_method"],
                        "q_gt_peak_ft": qgt_peak,
                        "q_pred_peak_ft": qpred_peak,
                        "peak_queue_error_ft": abs(qpred_peak - qgt_peak),
                        "time_gt_peak_sec": float(sub.loc[idx_gt, "time_sec"]),
                        "time_pred_peak_sec": float(sub.loc[idx_pred, "time_sec"]),
                    }
                )

    return pd.DataFrame(rows)


def summarize_cycle_peaks(cycle_errors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cycle_errors.empty:
        empty_cols = [
            "run_id",
            "cv_rate_pct",
            "ml_split",
            "curve_id",
            "mean_cycle_peak_error_ft",
            "median_cycle_peak_error_ft",
            "rmse_cycle_peak_error_ft",
            "n_cycles",
        ]
        return pd.DataFrame(columns=empty_cols), pd.DataFrame()

    group_cols_run_rate = [
        "run_id",
        "cv_rate_pct",
        "ml_split",
        "curve_id",
        "method_family",
        "model_name",
        "curve_label",
        "prediction_col",
        "source",
        "interpolation_method",
    ]

    run_rate = (
        cycle_errors.groupby(group_cols_run_rate, as_index=False)
        .agg(
            mean_cycle_peak_error_ft=("peak_queue_error_ft", "mean"),
            median_cycle_peak_error_ft=("peak_queue_error_ft", "median"),
            rmse_cycle_peak_error_ft=("peak_queue_error_ft", rmse_from_values),
            n_cycles=("cycle_id", "nunique"),
        )
        .reset_index(drop=True)
    )

    group_cols_split_rate = [
        "ml_split",
        "cv_rate_pct",
        "curve_id",
        "method_family",
        "model_name",
        "curve_label",
        "prediction_col",
        "source",
        "interpolation_method",
    ]

    split_rate = (
        run_rate.groupby(group_cols_split_rate, as_index=False)
        .agg(
            mean_cycle_peak_error_ft=("mean_cycle_peak_error_ft", "mean"),
            median_cycle_peak_error_ft=("median_cycle_peak_error_ft", "mean"),
            rmse_cycle_peak_error_ft=("rmse_cycle_peak_error_ft", "mean"),
            n_cycles=("n_cycles", "sum"),
        )
        .reset_index(drop=True)
    )

    return run_rate, split_rate


# =============================================================================
# Metric summaries and best selection
# =============================================================================

def merge_point_and_cycle_metrics(
    point_metrics: pd.DataFrame,
    cycle_summary_run_rate: pd.DataFrame,
) -> pd.DataFrame:
    if point_metrics.empty:
        return point_metrics

    if cycle_summary_run_rate.empty:
        point_metrics["mean_cycle_peak_error_ft"] = np.nan
        point_metrics["median_cycle_peak_error_ft"] = np.nan
        point_metrics["rmse_cycle_peak_error_ft"] = np.nan
        point_metrics["n_cycles"] = 0
        return point_metrics

    merge_cols = ["run_id", "cv_rate_pct", "ml_split", "curve_id"]

    add_cols = merge_cols + [
        "mean_cycle_peak_error_ft",
        "median_cycle_peak_error_ft",
        "rmse_cycle_peak_error_ft",
        "n_cycles",
    ]

    out = point_metrics.merge(
        cycle_summary_run_rate[add_cols],
        on=merge_cols,
        how="left",
    )

    return out


def summarize_by_split_rate(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()

    group_cols = [
        "ml_split",
        "cv_rate_pct",
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
    ]

    metric_cols = [
        "mae_ft",
        "rmse_ft",
        "abc_ft_s",
        "valid_pct",
        "mean_cycle_peak_error_ft",
        "median_cycle_peak_error_ft",
        "rmse_cycle_peak_error_ft",
        "n_cycles",
    ]

    out = (
        metrics.groupby(group_cols, as_index=False)[metric_cols]
        .mean()
        .sort_values(["ml_split", "cv_rate_pct", "method_family", "curve_id"])
        .reset_index(drop=True)
    )

    return out


def summarize_by_split_curve(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()

    group_cols = [
        "ml_split",
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
    ]

    metric_cols = [
        "mae_ft",
        "rmse_ft",
        "abc_ft_s",
        "valid_pct",
        "mean_cycle_peak_error_ft",
        "median_cycle_peak_error_ft",
        "rmse_cycle_peak_error_ft",
        "n_cycles",
    ]

    out = (
        metrics.groupby(group_cols, as_index=False)[metric_cols]
        .mean()
        .sort_values(["ml_split", "method_family", "curve_id"])
        .reset_index(drop=True)
    )

    return out


def select_best_models_validation(
    metrics: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()

    eligible = catalog[
        (catalog["is_reference"] == 0)
        & (catalog["include_in_best_selection"] == 1)
    ][["curve_id"]].copy()

    val = metrics[
        metrics["ml_split"].astype(str).str.contains(BEST_SELECTION_SPLIT, na=False)
    ].copy()

    val = val.merge(eligible, on="curve_id", how="inner")

    if val.empty:
        return pd.DataFrame()

    curve_summary = summarize_by_split_curve(val)

    if BEST_SELECTION_METRIC not in curve_summary.columns:
        raise ValueError(f"BEST_SELECTION_METRIC not found: {BEST_SELECTION_METRIC}")

    rows = []

    for family, g in curve_summary.groupby("method_family", sort=True):
        g = g.dropna(subset=[BEST_SELECTION_METRIC]).copy()

        if g.empty:
            continue

        best = g.sort_values(BEST_SELECTION_METRIC, ascending=True).iloc[0].to_dict()
        best["selection_metric"] = BEST_SELECTION_METRIC
        best["selection_split"] = BEST_SELECTION_SPLIT
        best["selection_rank_within_family"] = 1
        rows.append(best)

    return pd.DataFrame(rows)


# =============================================================================
# Plot-ready outputs
# =============================================================================

def save_plot_ready_selected_curves(
    pred: pd.DataFrame,
    catalog: pd.DataFrame,
    best: pd.DataFrame,
) -> None:
    if best.empty:
        print("[WARN] No validation best models found; plot-ready selected curves not saved.")
        return

    selected_cols = []
    selected_curve_ids = []

    for _, row in best.iterrows():
        q_col = row["prediction_col"]
        if q_col in pred.columns:
            selected_cols.append(q_col)
            selected_curve_ids.append(row["curve_id"])

    selected_cols = sorted(set(selected_cols))

    keep_cols = [
        "run_id",
        "ml_split",
        "cv_rate_pct",
        "time_sec",
        "phase_state",
        "phase_elapsed_sec",
        GT_COL,
    ]

    keep_cols = [c for c in keep_cols if c in pred.columns]

    out = pred[keep_cols + selected_cols].copy()
    out.to_csv(PLOT_READY_SELECTED_CURVES_OUT, index=False)

    selected_catalog = catalog[catalog["curve_id"].isin(selected_curve_ids)].copy()
    selected_catalog.to_csv(OUT_DIR / "method_family_plot_ready_selected_curve_catalog.csv", index=False)

    print(f"[Saved plot-ready selected curves] {PLOT_READY_SELECTED_CURVES_OUT}")


def save_plot_ready_cycle_peaks(
    cycle_errors: pd.DataFrame,
    best: pd.DataFrame,
) -> None:
    if cycle_errors.empty or best.empty:
        print("[WARN] Cycle peak plot-ready file not saved.")
        return

    selected_curve_ids = set(best["curve_id"].astype(str))

    selected = cycle_errors[cycle_errors["curve_id"].astype(str).isin(selected_curve_ids)].copy()

    gt_rows = (
        cycle_errors[
            [
                "run_id",
                "cv_rate_pct",
                "ml_split",
                "cycle_id",
                "cycle_start_sec",
                "cycle_end_sec",
                "q_gt_peak_ft",
            ]
        ]
        .drop_duplicates()
        .rename(columns={"q_gt_peak_ft": "q_peak_ft"})
    )

    gt_rows["curve_id"] = "gt"
    gt_rows["method_family"] = "GT"
    gt_rows["model_name"] = "GT"
    gt_rows["curve_label"] = "GT"
    gt_rows["peak_queue_error_ft"] = 0.0
    gt_rows["time_peak_sec"] = np.nan

    method_rows = selected[
        [
            "run_id",
            "cv_rate_pct",
            "ml_split",
            "cycle_id",
            "cycle_start_sec",
            "cycle_end_sec",
            "curve_id",
            "method_family",
            "model_name",
            "curve_label",
            "q_pred_peak_ft",
            "peak_queue_error_ft",
            "time_pred_peak_sec",
        ]
    ].copy()

    method_rows = method_rows.rename(
        columns={
            "q_pred_peak_ft": "q_peak_ft",
            "time_pred_peak_sec": "time_peak_sec",
        }
    )

    out = pd.concat([gt_rows, method_rows], ignore_index=True)
    out = out.sort_values(["run_id", "cv_rate_pct", "cycle_id", "method_family", "curve_id"]).reset_index(drop=True)

    out.to_csv(PLOT_READY_CYCLE_PEAKS_OUT, index=False)
    print(f"[Saved plot-ready cycle peaks] {PLOT_READY_CYCLE_PEAKS_OUT}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_dirs()

    print("=" * 96)
    print("Method-family queue-length evaluation")
    print("=" * 96)
    print(f"Project root         : {PROJECT_ROOT}")
    print(f"Feature file         : {FEATURE_FILE}")
    print(f"ML raw file          : {ML_RAW_FILE}")
    print(f"ML direct file       : {ML_DIRECT_FILE}")
    print(f"ML residual CV file  : {ML_RESIDUAL_CV_FILE}")
    print(f"Output dir           : {OUT_DIR}")
    print(f"Runs                 : {RUN_IDS}")
    print(f"Train runs           : {TRAIN_RUN_IDS}")
    print(f"Validation runs      : {VALIDATION_RUN_IDS}")
    print(f"Test runs            : {TEST_RUN_IDS}")
    print(f"CV rates             : {CV_RATES_PCT}")
    print("=" * 96)

    pred, catalog = build_unified_prediction_table()

    pred.to_csv(PREDICTIONS_OUT, index=False)
    catalog.to_csv(CATALOG_OUT, index=False)

    print(f"[Saved unified predictions] {PREDICTIONS_OUT}")
    print(f"[Saved curve catalog] {CATALOG_OUT}")

    print("\nComputing pointwise queue-length metrics...")
    point_metrics = compute_pointwise_metrics(pred, catalog)

    print("Computing red-to-red cycle peak errors...")
    cycle_errors = compute_cycle_peak_errors(pred, catalog)
    cycle_summary_run_rate, cycle_summary_split_rate = summarize_cycle_peaks(cycle_errors)

    metrics_run_rate = merge_point_and_cycle_metrics(point_metrics, cycle_summary_run_rate)

    summary_split_rate = summarize_by_split_rate(metrics_run_rate)
    summary_split_curve = summarize_by_split_curve(metrics_run_rate)

    validation_summary = summary_split_curve[
        summary_split_curve["ml_split"].astype(str).str.contains("validation", na=False)
    ].copy()

    test_summary = summary_split_curve[
        summary_split_curve["ml_split"].astype(str).str.contains("test", na=False)
    ].copy()

    best_validation = select_best_models_validation(metrics_run_rate, catalog)

    metrics_run_rate.to_csv(METRICS_RUN_RATE_OUT, index=False)
    summary_split_rate.to_csv(METRICS_SUMMARY_SPLIT_RATE_OUT, index=False)
    validation_summary.to_csv(METRICS_SUMMARY_VALIDATION_OUT, index=False)
    test_summary.to_csv(METRICS_SUMMARY_TEST_OUT, index=False)

    cycle_errors.to_csv(CYCLE_PEAK_ERRORS_OUT, index=False)
    cycle_summary_run_rate.to_csv(CYCLE_PEAK_SUMMARY_RUN_RATE_OUT, index=False)
    cycle_summary_split_rate.to_csv(CYCLE_PEAK_SUMMARY_SPLIT_RATE_OUT, index=False)

    best_validation.to_csv(BEST_MODELS_VALIDATION_OUT, index=False)

    print(f"[Saved] {METRICS_RUN_RATE_OUT}")
    print(f"[Saved] {METRICS_SUMMARY_SPLIT_RATE_OUT}")
    print(f"[Saved] {METRICS_SUMMARY_VALIDATION_OUT}")
    print(f"[Saved] {METRICS_SUMMARY_TEST_OUT}")
    print(f"[Saved] {CYCLE_PEAK_ERRORS_OUT}")
    print(f"[Saved] {CYCLE_PEAK_SUMMARY_RUN_RATE_OUT}")
    print(f"[Saved] {CYCLE_PEAK_SUMMARY_SPLIT_RATE_OUT}")
    print(f"[Saved] {BEST_MODELS_VALIDATION_OUT}")

    if SAVE_PLOT_READY_SELECTED_WIDE:
        save_plot_ready_selected_curves(pred, catalog, best_validation)

    if SAVE_PLOT_READY_CYCLE_PEAKS:
        save_plot_ready_cycle_peaks(cycle_errors, best_validation)

    print("\nValidation best models by family:")
    if best_validation.empty:
        print("No validation best-model table created.")
    else:
        show_cols = [
            "method_family",
            "curve_id",
            "curve_label",
            "mae_ft",
            "rmse_ft",
            "abc_ft_s",
            "mean_cycle_peak_error_ft",
        ]
        show_cols = [c for c in show_cols if c in best_validation.columns]
        print(best_validation[show_cols].round(3).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
