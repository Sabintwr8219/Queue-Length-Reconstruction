"""
Cumulative-count-space event-timing evaluation.

Place this file at:
    src/evaluate_event_timing.py

Purpose
-------
Evaluate transformed cumulative B curves using vehicle-event horizontal timing error.

For each event number N:
    t_GT(N)      = GT event time from gt_cumulative_events_runXXX.csv
    t_hat(N)     = first time where reconstructed B_hat(t) >= N
    error_sec    = t_hat(N) - t_GT(N)

Metrics:
    MAE_sec
    RMSE_sec
    Bias_sec

Main comparison curves:
    Baseline B
    CCT + GRU + CV transformed B

Inputs
------
    output/intermediate_csv/cumulative_transformed/cumulative_B_transformed_runXXX_rateYYY.csv
    output/intermediate_csv/gt/gt_cumulative_events_runXXX.csv

Outputs
-------
    output/intermediate_csv/evaluation_cumulative_event_timing/
        cumulative_event_timing_metrics_by_run_rate_curve.csv
        cumulative_event_timing_metrics_summary_by_rate_model.csv
        event_timing_errors_all.csv
        tables_png/
            cumulative_event_timing_summary_table.png
            cumulative_event_timing_summary_table_rateXXX.png
"""

from __future__ import annotations

from config import (
    PROJECT_ROOT,
    TEST_RUN_IDS,
    CV_RATES_PCT,
    USE_LINEAR_CROSSING_INTERPOLATION,
    DROP_UNREACHED_EVENTS
)

from pathlib import Path
import math

import numpy as np
import pandas as pd


# =============================================================================
# Stage-specific constants
# =============================================================================

# Professional repo layout: <project_root>/src/<script>.py

REVISED_DIR = PROJECT_ROOT / "output" / "intermediate_csv"

CUM_TRANSFORMED_DIR = REVISED_DIR / "cumulative_transformed"
GT_DIR = REVISED_DIR / "gt"

OUT_DIR = REVISED_DIR / "evaluation_cumulative_event_timing"
TABLE_DIR = OUT_DIR / "tables_png"

# Final test runs requested.


# Main final cumulative comparison set.
CURVES_TO_EVALUATE = {
    "Baseline B": {
        "column": "B_baseline_count",
        "curve_family": "Baseline",
        "ml_model": "None",
        "method": "Cumulative-count baseline",
    },
    "CCT + GRU + CV": {
        "column": "B_physics_ml_cv_gru_count",
        "curve_family": "Physics + ML + CV",
        "ml_model": "GRU",
        "method": "Selected final cumulative-count curve",
    },
}

# If True, crossing time is linearly interpolated between the two nearest
# time-grid rows around the event count N.
# If False, crossing time is simply the first grid time where B_hat >= N.

# Only evaluate event numbers that are reachable by the reconstructed curve.
# This avoids fake errors when a curve never reaches the final GT count.

# PNG table formatting.
MAKE_TABLE_PNG = False
TABLE_DPI = 250
TABLE_FONT_SIZE = 8
TABLE_MAX_ROWS_PER_IMAGE = 120


# =============================================================================
# Helpers
# =============================================================================

def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def transformed_file_path(run_id: int, rate: int) -> Path:
    return CUM_TRANSFORMED_DIR / f"cumulative_B_transformed_run{run_id:03d}_rate{rate:03d}.csv"


def gt_cumulative_file_path(run_id: int) -> Path:
    return GT_DIR / f"gt_cumulative_events_run{run_id:03d}.csv"


def require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def load_gt_event_times(run_id: int) -> pd.DataFrame:
    path = gt_cumulative_file_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing GT cumulative file: {path}")

    gt = pd.read_csv(path)

    # Expected from gt_revised_extraction.py:
    # t_event_sec, N_gt
    require_columns(gt, ["t_event_sec", "N_gt"], f"GT cumulative run {run_id:03d}")

    gt["t_event_sec"] = pd.to_numeric(gt["t_event_sec"], errors="coerce")
    gt["N_gt"] = pd.to_numeric(gt["N_gt"], errors="coerce")
    gt = gt.dropna(subset=["t_event_sec", "N_gt"]).copy()

    gt["N_gt"] = gt["N_gt"].astype(int)
    gt = gt.sort_values(["N_gt", "t_event_sec"]).drop_duplicates("N_gt", keep="first")

    return gt[["N_gt", "t_event_sec"]].rename(columns={"t_event_sec": "t_gt_sec"})


def first_crossing_time(
    time_sec: np.ndarray,
    b_curve: np.ndarray,
    n_event: int,
    use_interp: bool = True,
) -> float:
    """
    Return first time where B_hat(t) >= n_event.

    If interpolation is enabled, estimate the crossing time between the previous
    and current time-grid row. This is useful because reconstructed B curves are
    continuous/fractional after queue-length transformation.
    """
    t = np.asarray(time_sec, dtype=float)
    b = np.asarray(b_curve, dtype=float)

    mask = np.isfinite(t) & np.isfinite(b)
    if mask.sum() == 0:
        return np.nan

    t = t[mask]
    b = b[mask]

    order = np.argsort(t)
    t = t[order]
    b = b[order]

    hit = np.where(b >= float(n_event))[0]
    if len(hit) == 0:
        return np.nan

    idx = int(hit[0])

    if not use_interp or idx == 0:
        return float(t[idx])

    t0, t1 = float(t[idx - 1]), float(t[idx])
    b0, b1 = float(b[idx - 1]), float(b[idx])

    if not np.isfinite(b0) or not np.isfinite(b1):
        return float(t[idx])

    if b1 <= b0:
        return float(t[idx])

    # Linear interpolation for B(t) crossing N.
    frac = (float(n_event) - b0) / (b1 - b0)
    frac = min(max(frac, 0.0), 1.0)

    return float(t0 + frac * (t1 - t0))


def compute_event_errors_for_curve(
    transformed: pd.DataFrame,
    gt_events: pd.DataFrame,
    curve_label: str,
    curve_info: dict,
    run_id: int,
    rate: int,
) -> pd.DataFrame:
    col = curve_info["column"]
    if col not in transformed.columns:
        print(f"[WARN] Run {run_id:03d}, rate {rate:03d}: missing curve column {col}")
        return pd.DataFrame()

    require_columns(transformed, ["time_sec", col], f"transformed run {run_id:03d} rate {rate:03d}")

    d = transformed[["time_sec", col]].copy()
    d["time_sec"] = pd.to_numeric(d["time_sec"], errors="coerce")
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["time_sec", col]).sort_values("time_sec")

    if d.empty:
        return pd.DataFrame()

    time_arr = d["time_sec"].to_numpy(dtype=float)
    b_arr = d[col].to_numpy(dtype=float)

    rows = []
    for _, r in gt_events.iterrows():
        n = int(r["N_gt"])
        t_gt = float(r["t_gt_sec"])

        t_hat = first_crossing_time(
            time_sec=time_arr,
            b_curve=b_arr,
            n_event=n,
            use_interp=USE_LINEAR_CROSSING_INTERPOLATION,
        )

        if DROP_UNREACHED_EVENTS and not np.isfinite(t_hat):
            continue

        err = t_hat - t_gt if np.isfinite(t_hat) else np.nan

        rows.append(
            {
                "run_id": int(run_id),
                "cv_rate_pct": int(rate),
                "curve_label": curve_label,
                "curve_family": curve_info["curve_family"],
                "ml_model": curve_info["ml_model"],
                "method": curve_info["method"],
                "curve_column": col,
                "N_gt": n,
                "t_gt_sec": t_gt,
                "t_hat_sec": t_hat,
                "timing_error_sec": err,
                "abs_timing_error_sec": abs(err) if np.isfinite(err) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def compute_metrics(errors: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = [
        "run_id",
        "cv_rate_pct",
        "curve_label",
        "curve_family",
        "ml_model",
        "method",
        "curve_column",
    ]

    for keys, g in errors.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = {group_cols[i]: keys[i] for i in range(len(group_cols))}

        e = pd.to_numeric(g["timing_error_sec"], errors="coerce").to_numpy(dtype=float)
        e = e[np.isfinite(e)]

        n_valid = int(len(e))
        n_total_gt = int(g["N_gt"].max()) if len(g) else 0

        if n_valid == 0:
            mae = rmse = bias = maxae = np.nan
        else:
            mae = float(np.mean(np.abs(e)))
            rmse = float(np.sqrt(np.mean(e ** 2)))
            bias = float(np.mean(e))
            maxae = float(np.max(np.abs(e)))

        rows.append(
            {
                **base,
                "n_events_valid": n_valid,
                "max_N_evaluated": n_total_gt,
                "mae_sec": mae,
                "rmse_sec": rmse,
                "bias_sec": bias,
                "max_abs_error_sec": maxae,
            }
        )

    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "cv_rate_pct",
        "curve_label",
        "curve_family",
        "ml_model",
        "method",
        "curve_column",
    ]

    summary = (
        metrics.groupby(group_cols, as_index=False)
        .agg(
            n_runs=("run_id", "nunique"),
            n_events_valid_avg=("n_events_valid", "mean"),
            mae_sec_avg=("mae_sec", "mean"),
            mae_sec_std=("mae_sec", "std"),
            rmse_sec_avg=("rmse_sec", "mean"),
            rmse_sec_std=("rmse_sec", "std"),
            bias_sec_avg=("bias_sec", "mean"),
            bias_sec_std=("bias_sec", "std"),
            max_abs_error_sec_avg=("max_abs_error_sec", "mean"),
        )
    )

    family_order = {
    "Baseline": 0,
    "Physics + ML + CV": 1,
    }

    model_order = {
    "None": 0,
    "GRU": 1,
    }

    summary["_family_order"] = summary["curve_family"].map(family_order).fillna(99)
    summary["_model_order"] = summary["ml_model"].map(model_order).fillna(99)

    summary = summary.sort_values(
        ["cv_rate_pct", "_family_order", "_model_order", "curve_label"]
    ).drop(columns=["_family_order", "_model_order"])

    return summary


def format_summary_for_png(summary: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "cv_rate_pct",
        "curve_family",
        "ml_model",
        "method",
        "n_runs",
        "n_events_valid_avg",
        "mae_sec_avg",
        "rmse_sec_avg",
        "bias_sec_avg",
        "max_abs_error_sec_avg",
    ]

    tbl = summary[[c for c in keep_cols if c in summary.columns]].copy()

    tbl = tbl.rename(
        columns={
            "cv_rate_pct": "CV %",
            "curve_family": "Curve family",
            "ml_model": "ML model",
            "method": "Method",
            "n_runs": "Runs",
            "n_events_valid_avg": "Avg events",
            "mae_sec_avg": "Avg MAE (s)",
            "rmse_sec_avg": "Avg RMSE (s)",
            "bias_sec_avg": "Avg Bias (s)",
            "max_abs_error_sec_avg": "Avg MaxAE (s)",
        }
    )

    for c in ["Avg events"]:
        if c in tbl.columns:
            tbl[c] = tbl[c].map(lambda x: "" if pd.isna(x) else f"{x:.0f}")

    for c in ["Avg MAE (s)", "Avg RMSE (s)", "Avg Bias (s)", "Avg MaxAE (s)"]:
        if c in tbl.columns:
            tbl[c] = tbl[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    return tbl


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_dirs()

    print("=" * 88)
    print("Cumulative-count event-timing evaluation")
    print("=" * 88)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Input dir    : {CUM_TRANSFORMED_DIR}")
    print(f"GT dir       : {GT_DIR}")
    print(f"Output dir   : {OUT_DIR}")
    print(f"Test runs    : {TEST_RUN_IDS}")
    print(f"CV rates     : {CV_RATES_PCT}")
    print("=" * 88)

    all_error_tables = []

    for run_id in TEST_RUN_IDS:
        gt_events = load_gt_event_times(run_id)

        for rate in CV_RATES_PCT:
            path = transformed_file_path(run_id, rate)

            if not path.exists():
                print(f"[WARN] Missing transformed file, skipping: {path}")
                continue

            transformed = pd.read_csv(path)

            for curve_label, curve_info in CURVES_TO_EVALUATE.items():
                err_df = compute_event_errors_for_curve(
                    transformed=transformed,
                    gt_events=gt_events,
                    curve_label=curve_label,
                    curve_info=curve_info,
                    run_id=run_id,
                    rate=rate,
                )

                if not err_df.empty:
                    all_error_tables.append(err_df)

    if not all_error_tables:
        raise RuntimeError("No event-timing errors were computed. Check inputs and curve columns.")

    errors = pd.concat(all_error_tables, ignore_index=True)

    errors_path = OUT_DIR / "event_timing_errors_all.csv"
    errors.to_csv(errors_path, index=False)
    print(f"[Saved] {errors_path}")

    metrics = compute_metrics(errors)

    metrics_path = OUT_DIR / "cumulative_event_timing_metrics_by_run_rate_curve.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"[Saved] {metrics_path}")

    summary = summarize_metrics(metrics)

    summary_path = OUT_DIR / "cumulative_event_timing_metrics_summary_by_rate_model.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[Saved] {summary_path}")

    print("\nDone.")
    print("Main outputs:")
    print(f"  {metrics_path}")
    print(f"  {summary_path}")
    print(f"  {TABLE_DIR}")


if __name__ == "__main__":
    main()
