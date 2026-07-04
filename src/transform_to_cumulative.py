"""
Transform selected queue-length reconstruction curve into cumulative-count space.

Place this file at:
    src/transform_to_cumulative.py

Purpose
-------
This revised script transforms only the final selected queue-length curve into
cumulative-count B space.

Final selected curve:
    Physics + ML + CV
    Model: GRU
    Final plotting label: CCT + GRU + CV

This script reads the new method-family queue-length prediction output:
    output/intermediate_csv/method_family_queue_length_evaluation/
        method_family_predictions_allruns_allrates.csv

and writes:
    output/intermediate_csv/cumulative_transformed/
        cumulative_B_transformed_runXXX_rateYYY.csv
        cumulative_B_transformed_allruns_allrates.csv

The transformed cumulative-count comparison will include only:
    1. GT cumulative event curve
    2. Physics baseline B curve
    3. Proposed corrected B curve

The A/D/V/B cumulative-count theory plot remains separate and is not mixed
with this final comparison file.

Core relationship
-----------------
    n_queue_hat(t) = Q_hat(t) / l_eff_fixed
    B_hat(t)       = D(t) + n_queue_hat(t)

Because B_hat(t) is a cumulative count curve, it is bounded and forced to be
nondecreasing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import config as cfg


# =============================================================================
# Project configuration
# =============================================================================

PROJECT_ROOT = cfg.PROJECT_ROOT
RUN_IDS = list(cfg.RUN_IDS)
CV_RATES_PCT = list(cfg.CV_RATES_PCT)

# Use config values if available; otherwise use safe defaults.
L_EFF_FALLBACK_FT = float(getattr(cfg, "L_EFF_FALLBACK_FT", 25.0))
CLIP_Q_NONNEGATIVE = bool(getattr(cfg, "CLIP_Q_NONNEGATIVE", True))
BOUND_B_BETWEEN_D_AND_A = bool(getattr(cfg, "BOUND_B_BETWEEN_D_AND_A", True))
FORCE_B_MONOTONE = bool(getattr(cfg, "FORCE_B_MONOTONE", True))
ROUND_TIME_DECIMALS = int(getattr(cfg, "ROUND_TIME_DECIMALS", 6))


# =============================================================================
# Stage-specific settings
# =============================================================================

INTERMEDIATE_DIR = PROJECT_ROOT / "output" / "intermediate_csv"

METHOD_EVAL_DIR = INTERMEDIATE_DIR / "method_family_queue_length_evaluation"
BASELINE_DIR = INTERMEDIATE_DIR / "baseline"
GT_DIR = INTERMEDIATE_DIR / "gt"
OUT_DIR = INTERMEDIATE_DIR / "cumulative_transformed"

METHOD_PREDICTIONS_FILE = METHOD_EVAL_DIR / "method_family_predictions_allruns_allrates.csv"
METHOD_CATALOG_FILE = METHOD_EVAL_DIR / "method_family_curve_catalog.csv"

BASELINE_TIMEGRID_PATTERN = BASELINE_DIR / "baseline_queue_count_timegrid_run{run_id:03d}.csv"
GT_CUMULATIVE_PATTERN = GT_DIR / "gt_cumulative_events_run{run_id:03d}.csv"

PER_RUN_RATE_OUT_PATTERN = OUT_DIR / "cumulative_B_transformed_run{run_id:03d}_rate{rate:03d}.csv"
ALLRUNS_ALLRATES_OUT = OUT_DIR / "cumulative_B_transformed_allruns_allrates.csv"

# -------------------------------------------------------------------------
# Final selected queue-length curve
# -------------------------------------------------------------------------
# Leave SELECTED_PREDICTION_COL as None to auto-resolve from the curve catalog.
# If auto-resolution fails, set it manually to:
#     "q_pred_gru_physics_ml_cv_ft"
SELECTED_PREDICTION_COL: str | None = None

SELECTED_METHOD_FAMILY = "Physics + ML + CV"
SELECTED_MODEL_NAME = "GRU"

# This is the final publication-facing label. It does not need to match the
# internal method_family label exactly.
PROPOSED_LABEL = "CCT + GRU + CV"

# Output transformed B curve columns.
PROPOSED_Q_COL_OUT = "q_pred_gru_physics_ml_cv_ft"
PROPOSED_N_QUEUE_COL_OUT = "n_queue_physics_ml_cv_gru"
PROPOSED_B_RAW_COL_OUT = "B_physics_ml_cv_gru_count_raw"
PROPOSED_B_COL_OUT = "B_physics_ml_cv_gru_count"

# Process all configured runs/rates.
PROCESS_RUN_IDS = RUN_IDS
PROCESS_CV_RATES_PCT = CV_RATES_PCT


# =============================================================================
# Basic helpers
# =============================================================================

def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def fmt(path: Path, **kwargs) -> Path:
    return Path(str(path).format(**kwargs))


def read_csv_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}:\n{path}")
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_time_col(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    require_columns(d, ["time_sec"], "time-grid table")
    d["time_sec"] = to_numeric(d["time_sec"])
    d = d.dropna(subset=["time_sec"]).copy()
    d["time_key"] = d["time_sec"].round(ROUND_TIME_DECIMALS)
    return d


def choose_first_existing_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find {label}. Tried: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def finite_median_or_fallback(values: pd.Series, fallback: float) -> float:
    vals = to_numeric(values).to_numpy(dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) == 0:
        return float(fallback)
    return float(np.nanmedian(vals))


def clean_numeric_array(values, fill_value: float = 0.0) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if np.isfinite(arr).any():
        arr = pd.Series(arr).interpolate(limit_direction="both").to_numpy(dtype=float)
    else:
        arr = np.full(len(arr), fill_value, dtype=float)
    return arr


def make_cumulative_nonnegative(values) -> np.ndarray:
    arr = clean_numeric_array(values, fill_value=0.0)
    arr = np.maximum(arr, 0.0)
    return np.maximum.accumulate(arr)


# =============================================================================
# Input loaders
# =============================================================================

def load_method_predictions() -> pd.DataFrame:
    pred = read_csv_required(METHOD_PREDICTIONS_FILE, "method-family predictions")
    require_columns(
        pred,
        ["run_id", "cv_rate_pct", "time_sec"],
        "method-family predictions",
    )

    pred["run_id"] = to_numeric(pred["run_id"])
    pred["cv_rate_pct"] = to_numeric(pred["cv_rate_pct"])
    pred["time_sec"] = to_numeric(pred["time_sec"])

    pred = pred.dropna(subset=["run_id", "cv_rate_pct", "time_sec"]).copy()
    pred["run_id"] = pred["run_id"].astype(int)
    pred["cv_rate_pct"] = pred["cv_rate_pct"].astype(int)

    pred = normalize_time_col(pred)
    pred = pred.sort_values(["run_id", "cv_rate_pct", "time_sec"]).reset_index(drop=True)

    return pred


def load_method_catalog() -> pd.DataFrame:
    if not METHOD_CATALOG_FILE.exists():
        return pd.DataFrame()

    catalog = pd.read_csv(METHOD_CATALOG_FILE)
    required = ["curve_id", "method_family", "curve_label", "prediction_col"]
    if not set(required).issubset(catalog.columns):
        print(f"[WARN] Curve catalog exists but is missing required columns: {METHOD_CATALOG_FILE}")
        return pd.DataFrame()

    return catalog.copy()


def resolve_selected_prediction_col(pred: pd.DataFrame, catalog: pd.DataFrame) -> str:
    """
    Resolve the selected final queue-length prediction column.

    Priority:
    1. User-specified SELECTED_PREDICTION_COL.
    2. Catalog row matching method family and GRU.
    3. Common fallback column name.
    """
    if SELECTED_PREDICTION_COL is not None:
        if SELECTED_PREDICTION_COL not in pred.columns:
            raise ValueError(
                f"SELECTED_PREDICTION_COL='{SELECTED_PREDICTION_COL}' not found in predictions."
            )
        return SELECTED_PREDICTION_COL

    if not catalog.empty:
        c = catalog.copy()
        c["method_family_l"] = c["method_family"].astype(str).str.lower().str.strip()
        c["curve_id_l"] = c["curve_id"].astype(str).str.lower().str.strip()
        c["curve_label_l"] = c["curve_label"].astype(str).str.lower().str.strip()

        if "model_name" in c.columns:
            c["model_name_l"] = c["model_name"].astype(str).str.lower().str.strip()
        else:
            c["model_name_l"] = ""

        search_text = (
            c["curve_id_l"].fillna("")
            + " "
            + c["curve_label_l"].fillna("")
            + " "
            + c["model_name_l"].fillna("")
        )

        family_mask = c["method_family_l"] == SELECTED_METHOD_FAMILY.lower()
        model_mask = search_text.str.contains(SELECTED_MODEL_NAME.lower(), na=False)

        candidates = c[family_mask & model_mask].copy()

        if not candidates.empty:
            for _, row in candidates.iterrows():
                col = str(row["prediction_col"])
                if col in pred.columns:
                    print("[Selected curve from catalog]")
                    print(f"  curve_id       : {row.get('curve_id', '')}")
                    print(f"  method_family  : {row.get('method_family', '')}")
                    print(f"  curve_label    : {row.get('curve_label', '')}")
                    print(f"  prediction_col : {col}")
                    return col

    fallback_candidates = [
        "q_pred_gru_physics_ml_cv_ft",
        "q_physics_ml_cv_gru_ft",
    ]

    for col in fallback_candidates:
        if col in pred.columns:
            print(f"[Selected curve from fallback column] {col}")
            return col

    raise ValueError(
        "Could not resolve selected GRU Physics + ML + CV prediction column.\n"
        f"Tried fallback columns: {fallback_candidates}\n"
        f"Available prediction columns include:\n"
        f"{[c for c in pred.columns if c.startswith('q_')]}"
    )


def load_baseline_timegrid(run_id: int) -> pd.DataFrame:
    path = fmt(BASELINE_TIMEGRID_PATTERN, run_id=run_id)
    base = read_csv_required(path, f"baseline time-grid run {run_id:03d}")

    base = normalize_time_col(base)

    count_candidates = {
        "A_count": ["A_count", "A", "A_cum", "arrival_count"],
        "D_count": ["D_count", "D", "D_cum", "departure_count"],
        "V_count": ["V_count", "V", "V_cum"],
        "B_baseline_count": ["B_count", "B", "B_cum", "B_baseline_count"],
    }

    a_col = choose_first_existing_column(base, count_candidates["A_count"], "A_count")
    d_col = choose_first_existing_column(base, count_candidates["D_count"], "D_count")
    v_col = choose_first_existing_column(base, count_candidates["V_count"], "V_count")
    b_col = choose_first_existing_column(base, count_candidates["B_baseline_count"], "B_baseline_count")

    keep = ["time_key", "time_sec", a_col, d_col, v_col, b_col]

    optional_cols = [
        "l_eff_fixed_ft",
        "q_baseline_fixed_ft",
        "q_physics_baseline_ft",
        "phase_state",
    ]
    keep += [c for c in optional_cols if c in base.columns]

    out = base[keep].copy()
    out = out.rename(
        columns={
            a_col: "A_count",
            d_col: "D_count",
            v_col: "V_count",
            b_col: "B_baseline_count_raw",
            "time_sec": "time_sec_baseline",
        }
    )

    return out


def load_gt_cumulative_events(run_id: int) -> pd.DataFrame:
    path = fmt(GT_CUMULATIVE_PATTERN, run_id=run_id)
    gt = read_csv_required(path, f"GT cumulative events run {run_id:03d}")

    require_columns(gt, ["t_event_sec", "N_gt"], f"GT cumulative events run {run_id:03d}")

    gt["t_event_sec"] = to_numeric(gt["t_event_sec"])
    gt["N_gt"] = to_numeric(gt["N_gt"])

    gt = gt.dropna(subset=["t_event_sec", "N_gt"]).copy()
    gt["N_gt"] = gt["N_gt"].astype(int)
    gt = gt.sort_values(["N_gt", "t_event_sec"]).reset_index(drop=True)

    return gt[["t_event_sec", "N_gt"]]


def gt_step_on_timegrid(time_sec: np.ndarray, gt_events: pd.DataFrame) -> np.ndarray:
    t = np.asarray(time_sec, dtype=float)
    event_times = gt_events["t_event_sec"].to_numpy(dtype=float)
    event_times = np.sort(event_times[np.isfinite(event_times)])
    return np.searchsorted(event_times, t, side="right").astype(float)


# =============================================================================
# Transformation logic
# =============================================================================

def clean_baseline_b_curve(
    b_baseline_raw,
    d_count,
    a_count,
) -> np.ndarray:
    """
    Clean the baseline cumulative B curve using the same physical safeguards:
    - interpolate missing values
    - bound between D and A
    - force nondecreasing
    """
    b = clean_numeric_array(b_baseline_raw, fill_value=0.0)
    d = make_cumulative_nonnegative(d_count)
    a = make_cumulative_nonnegative(a_count)

    if BOUND_B_BETWEEN_D_AND_A:
        b = np.maximum(b, d)
        b = np.minimum(b, a)

    if FORCE_B_MONOTONE:
        b = np.maximum.accumulate(b)
        if BOUND_B_BETWEEN_D_AND_A:
            b = np.minimum(b, a)
            b = np.maximum.accumulate(b)

    return b


def cumulative_transform_one_queue_curve(
    q_ft,
    d_count,
    l_eff: float,
    a_count=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform one queue-length curve Q_hat(t) to cumulative B_hat(t).

    Returns
    -------
    n_queue_hat:
        Predicted queue count from Q_hat/l_eff.
    b_raw:
        Raw B_hat = D + n_queue_hat before physical bounding/monotonic correction.
    b_final:
        Final bounded and monotonic cumulative B_hat.
    """
    q = pd.to_numeric(pd.Series(q_ft), errors="coerce").to_numpy(dtype=float)

    if CLIP_Q_NONNEGATIVE:
        q = np.where(np.isfinite(q), q, np.nan)
        q = np.maximum(q, 0.0)

    # Preserve old behavior: fill missing queue values by interpolation so
    # isolated NaNs do not break cumulative B.
    if np.isfinite(q).any():
        q = pd.Series(q).interpolate(limit_direction="both").to_numpy(dtype=float)
    else:
        q = np.zeros(len(q), dtype=float)

    d = make_cumulative_nonnegative(d_count)
    a = make_cumulative_nonnegative(a_count) if a_count is not None else None

    n_queue_hat = q / max(float(l_eff), 1e-9)
    b_raw = d + n_queue_hat

    b = b_raw.copy()

    if BOUND_B_BETWEEN_D_AND_A:
        b = np.maximum(b, d)
        if a is not None:
            b = np.minimum(b, a)

    if FORCE_B_MONOTONE:
        b = np.maximum.accumulate(b)
        if BOUND_B_BETWEEN_D_AND_A and a is not None:
            # A is cumulative and should be nondecreasing. This second pass
            # mirrors the old transform script behavior.
            b = np.minimum(b, a)
            b = np.maximum.accumulate(b)

    return n_queue_hat, b_raw, b


def transform_one_run_rate(
    method_pred_all: pd.DataFrame,
    selected_q_col: str,
    run_id: int,
    rate: int,
) -> pd.DataFrame:
    pred = method_pred_all[
        (method_pred_all["run_id"] == int(run_id))
        & (method_pred_all["cv_rate_pct"] == int(rate))
    ].copy()

    if pred.empty:
        raise ValueError(f"No method-family prediction rows for run {run_id:03d}, rate {rate:03d}.")

    base = load_baseline_timegrid(run_id)

    keep_pred = [
        "run_id",
        "cv_rate_pct",
        "time_key",
        "time_sec",
        selected_q_col,
    ]

    optional_pred_cols = [
        "ml_split",
        "phase_state",
        "q_gt_ft",
        "q_baseline_fixed_ft",
        "q_physics_baseline_ft",
    ]
    keep_pred += [c for c in optional_pred_cols if c in pred.columns]

    pred_small = pred[keep_pred].copy()

    merged = pred_small.merge(base, on="time_key", how="left")

    if merged.empty:
        raise ValueError(f"Merged table is empty for run {run_id:03d}, rate {rate:03d}.")

    # Prefer prediction time, but keep baseline time for diagnostics if needed.
    merged["time_sec"] = to_numeric(merged["time_sec"])

    # If phase_state was not in method predictions, use baseline phase_state.
    if "phase_state" not in merged.columns and "phase_state_baseline" in merged.columns:
        merged["phase_state"] = merged["phase_state_baseline"]

    if "phase_state" not in merged.columns:
        merged["phase_state"] = "unknown"

    # Standardize baseline queue-length column.
    if "q_physics_baseline_ft" in merged.columns:
        q_base_col = "q_physics_baseline_ft"
    elif "q_baseline_fixed_ft" in merged.columns:
        q_base_col = "q_baseline_fixed_ft"
    elif "q_baseline_fixed_ft_y" in merged.columns:
        q_base_col = "q_baseline_fixed_ft_y"
    else:
        q_base_col = None

    if q_base_col is not None:
        merged["q_physics_baseline_ft"] = to_numeric(merged[q_base_col])
    else:
        merged["q_physics_baseline_ft"] = np.nan

    # Clean cumulative-count columns.
    merged["A_count"] = make_cumulative_nonnegative(merged["A_count"])
    merged["D_count"] = make_cumulative_nonnegative(merged["D_count"])
    merged["V_count"] = make_cumulative_nonnegative(merged["V_count"])

    merged["B_baseline_count"] = clean_baseline_b_curve(
        merged["B_baseline_count_raw"],
        merged["D_count"],
        merged["A_count"],
    )

    # Effective spacing. Use the median fixed effective spacing for this run.
    if "l_eff_fixed_ft" in merged.columns:
        l_eff = finite_median_or_fallback(merged["l_eff_fixed_ft"], L_EFF_FALLBACK_FT)
    else:
        l_eff = L_EFF_FALLBACK_FT

    merged["l_eff_fixed_ft"] = float(l_eff)

    # Proposed selected queue-length curve.
    merged[PROPOSED_Q_COL_OUT] = to_numeric(merged[selected_q_col])

    n_queue, b_raw, b_final = cumulative_transform_one_queue_curve(
        q_ft=merged[PROPOSED_Q_COL_OUT],
        d_count=merged["D_count"],
        l_eff=l_eff,
        a_count=merged["A_count"],
    )

    merged[PROPOSED_N_QUEUE_COL_OUT] = n_queue
    merged[PROPOSED_B_RAW_COL_OUT] = b_raw
    merged[PROPOSED_B_COL_OUT] = b_final

    # GT cumulative step curve on the same time grid.
    gt_events = load_gt_cumulative_events(run_id)
    merged["B_gt_count"] = gt_step_on_timegrid(
        merged["time_sec"].to_numpy(dtype=float),
        gt_events,
    )

    # Clean and order output columns.
    preferred_cols = [
        "run_id",
        "cv_rate_pct",
        "time_sec",
        "ml_split",
        "phase_state",
        "A_count",
        "D_count",
        "V_count",
        "B_gt_count",
        "B_baseline_count",
        "l_eff_fixed_ft",
        "q_gt_ft",
        "q_physics_baseline_ft",
        PROPOSED_Q_COL_OUT,
        PROPOSED_N_QUEUE_COL_OUT,
        PROPOSED_B_RAW_COL_OUT,
        PROPOSED_B_COL_OUT,
    ]

    cols = [c for c in preferred_cols if c in merged.columns]
    out = merged[cols].copy()
    out = out.sort_values("time_sec").drop_duplicates("time_sec", keep="last").reset_index(drop=True)

    return out


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_dirs()

    print("=" * 96)
    print("Transform selected queue-length curve to cumulative-count space")
    print("=" * 96)
    print(f"Project root        : {PROJECT_ROOT}")
    print(f"Input predictions   : {METHOD_PREDICTIONS_FILE}")
    print(f"Output folder       : {OUT_DIR}")
    print(f"Selected family     : {SELECTED_METHOD_FAMILY}")
    print(f"Selected model      : {SELECTED_MODEL_NAME}")
    print(f"Publication label   : {PROPOSED_LABEL}")
    print("=" * 96)

    method_pred_all = load_method_predictions()
    catalog = load_method_catalog()

    selected_q_col = resolve_selected_prediction_col(method_pred_all, catalog)
    print(f"\n[Using queue-length column] {selected_q_col}")

    all_parts = []

    for run_id in PROCESS_RUN_IDS:
        for rate in PROCESS_CV_RATES_PCT:
            print(f"\n--- Transforming run {run_id:03d}, CV {rate:03d}% ---")

            out = transform_one_run_rate(
                method_pred_all=method_pred_all,
                selected_q_col=selected_q_col,
                run_id=int(run_id),
                rate=int(rate),
            )

            out_path = fmt(PER_RUN_RATE_OUT_PATTERN, run_id=int(run_id), rate=int(rate))
            out.to_csv(out_path, index=False)
            print(f"[Saved] {out_path} | rows={len(out):,}")

            all_parts.append(out)

    if all_parts:
        combined = pd.concat(all_parts, ignore_index=True)
        combined = combined.sort_values(["run_id", "cv_rate_pct", "time_sec"]).reset_index(drop=True)
        combined.to_csv(ALLRUNS_ALLRATES_OUT, index=False)
        print(f"\n[Saved combined] {ALLRUNS_ALLRATES_OUT} | rows={len(combined):,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
