"""
Revised CV sampling and queue-length feature engineering.

Place this file at:
    src/build_cv_features.py

Purpose
-------
This script prepares the CV-based inputs for the next ML step.
It does NOT train XGBoost or GRU models.

It reads the revised GT and cumulative-count baseline outputs:
    output/intermediate_csv/gt/gt_queue_join_events_runXXX.csv
    output/intermediate_csv/gt/gt_queue_length_timegrid_runXXX.csv
    output/intermediate_csv/baseline/baseline_queue_count_timegrid_runXXX.csv

It randomly selects CV vehicles for each penetration rate and saves:
    output/intermediate_csv/cv_features/cv_allocation_runXXX_rateYYY.csv
    output/intermediate_csv/cv_features/cv_anchors_runXXX_rateYYY.csv
    output/intermediate_csv/cv_features/timegrid_features_runXXX_rateYYY.csv
    output/intermediate_csv/cv_features/timegrid_features_allruns_allrates.csv

Important definitions
---------------------
1. CV allocation is done at the vehicle level.
2. CV anchors are queued CV vehicles plus forced boundary CV vehicles.
   Normal queued CV anchor:
       anchor time = t_queue_join_sec
       anchor value = q_gt_ft from the saved GT time-grid at that time
   Forced first/last boundary CV anchor:
       anchor = (t_exit_sec, 0 ft)
   The direct join-location value |s_join| is kept only as a diagnostic column.
3. The continuous GT curve used later is NOT recomputed here.
   It is read directly from gt_queue_length_timegrid_runXXX.csv using q_gt_ft.
4. Linear interpolation between CV anchors is prepared as a baseline feature.
5. Rows outside the first/last CV anchor are marked as outside CV segments.

This keeps the pipeline honest: after GT creation, future scripts read saved GT
for training/evaluation only; the model inputs come from baseline curves,
phase/cumulative-count variables, and sampled CV anchors.
"""

from __future__ import annotations

from config import (
    PROJECT_ROOT,
    RUN_IDS,
    TRAIN_RUN_IDS,
    VALIDATION_RUN_IDS,
    TEST_RUN_IDS,
    CV_RATES_PCT,
    RANDOM_SEED,
    ASOF_TOL_SEC,
)

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# Stage-specific constants
# =============================================================================

# Professional repo layout: <project_root>/src/<script>.py

# Run split used later by ML training. This script only labels rows by split.

# CV penetration rates.
NESTED_CV_SAMPLING = True
ADD_SYNTHETIC_TIMEGRID_BOUNDARY_ANCHORS = True
# Time-grid merge behavior. The script first tries exact merge on time_sec.
# This tolerance is used only as a fallback if exact floating-point times do not align.

# Input folders.
GT_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "gt"
BASELINE_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "baseline"

# Output folder.
OUT_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "cv_features"

# File patterns.
GT_EVENTS_PATTERN = GT_DIR / "gt_queue_join_events_run{run_id:03d}.csv"
GT_TIMEGRID_PATTERN = GT_DIR / "gt_queue_length_timegrid_run{run_id:03d}.csv"
BASELINE_TIMEGRID_PATTERN = BASELINE_DIR / "baseline_queue_count_timegrid_run{run_id:03d}.csv"

CV_ALLOC_PATTERN = OUT_DIR / "cv_allocation_run{run_id:03d}_rate{rate:03d}.csv"
CV_ANCHORS_PATTERN = OUT_DIR / "cv_anchors_run{run_id:03d}_rate{rate:03d}.csv"
FEATURES_PATTERN = OUT_DIR / "timegrid_features_run{run_id:03d}_rate{rate:03d}.csv"

CV_ALLOC_ALLRUNS_PATTERN = OUT_DIR / "cv_allocation_allruns_rate{rate:03d}.csv"
CV_ANCHORS_ALLRUNS_PATTERN = OUT_DIR / "cv_anchors_allruns_rate{rate:03d}.csv"
FEATURES_ALLRUNS_ALLRATES = OUT_DIR / "timegrid_features_allruns_allrates.csv"
FEATURES_ALLRUNS_PATTERN = OUT_DIR / "timegrid_features_allruns_rate{rate:03d}.csv"

# Columns to keep from baseline if present.
BASELINE_OPTIONAL_COLS = [
    "A_count",
    "D_count",
    "V_count",
    "B_count",
    "n_queue_cumulative",
    "q_baseline_fixed_ft",
    "l_eff_fixed_ft",
    "phase_state",
]

# If 1, the script saves larger per-run/rate feature files.
SAVE_PER_RUN_RATE_FEATURES = True


# =============================================================================
# Helpers
# =============================================================================

def fmt(path: Path, **kwargs) -> Path:
    return Path(str(path).format(**kwargs))


def require_columns(df: pd.DataFrame, required: set[str] | list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def as_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def bool_to_int(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    text = series.astype(str).str.strip().str.lower()
    return text.isin(["1", "true", "yes", "y"]).astype(int)


def load_gt_events(run_id: int) -> pd.DataFrame:
    path = fmt(GT_EVENTS_PATTERN, run_id=run_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing GT events file: {path}")

    df = pd.read_csv(path)
    require_columns(
        df,
        ["run_id", "veh_uid", "joined_queue", "t_queue_join_sec", "queue_length_at_join_ft", "t_exit_sec"],
        f"GT events run {run_id:03d}",
    )

    df["veh_uid"] = df["veh_uid"].astype(str).str.strip()
    df["joined_queue"] = bool_to_int(df["joined_queue"])
    df = as_numeric(
        df,
        [
            "run_id",
            "t_queue_join_sec",
            "queue_length_at_join_ft",
            "t_exit_sec",
            "t_event_sec",
            "queue_length_for_event_ft",
        ],
    )
    df = df.dropna(subset=["veh_uid"]).copy()
    df["run_id"] = int(run_id)
    return df


def load_gt_timegrid(run_id: int) -> pd.DataFrame:
    path = fmt(GT_TIMEGRID_PATTERN, run_id=run_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing GT timegrid file: {path}")

    df = pd.read_csv(path)
    require_columns(df, ["run_id", "time_sec", "q_gt_ft"], f"GT timegrid run {run_id:03d}")
    df = as_numeric(df, ["run_id", "time_sec", "q_gt_ft"])
    keep = [c for c in ["run_id", "time_sec", "q_gt_ft"] if c in df.columns]
    return df[keep].dropna(subset=["time_sec"]).sort_values("time_sec").copy()


def lookup_q_gt_at_times(gt_grid: pd.DataFrame, event_times: pd.Series | np.ndarray) -> np.ndarray:
    """Return q_gt_ft from the saved GT time-grid at requested event times.

    This is the key correction: CV anchor queue length must be the exact same
    GT curve value used in later plots/evaluation, not the vehicle-level direct
    join distance |s_join|. Linear interpolation on the saved time grid is used
    only to handle tiny floating-point offsets between event times and grid times.
    """
    if gt_grid.empty:
        return np.full(len(event_times), np.nan)

    g = gt_grid[["time_sec", "q_gt_ft"]].dropna().sort_values("time_sec").copy()
    if g.empty:
        return np.full(len(event_times), np.nan)

    t_grid = g["time_sec"].to_numpy(dtype=float)
    q_grid = g["q_gt_ft"].to_numpy(dtype=float)
    t_event = pd.to_numeric(pd.Series(event_times), errors="coerce").to_numpy(dtype=float)

    out = np.full(len(t_event), np.nan, dtype=float)
    ok = np.isfinite(t_event)
    if np.any(ok):
        out[ok] = np.interp(t_event[ok], t_grid, q_grid, left=q_grid[0], right=q_grid[-1])
    return out


def load_baseline_timegrid(run_id: int) -> pd.DataFrame:
    path = fmt(BASELINE_TIMEGRID_PATTERN, run_id=run_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline timegrid file: {path}")

    df = pd.read_csv(path)
    require_columns(df, ["run_id", "time_sec", "q_baseline_fixed_ft"], f"baseline timegrid run {run_id:03d}")

    keep = ["run_id", "time_sec"] + [c for c in BASELINE_OPTIONAL_COLS if c in df.columns]
    df = df[keep].copy()
    numeric_cols = [c for c in keep if c not in ["phase_state"]]
    df = as_numeric(df, numeric_cols)
    df = df.dropna(subset=["time_sec"]).sort_values("time_sec").copy()

    if "phase_state" not in df.columns:
        df["phase_state"] = "unknown"

    df["phase_state"] = df["phase_state"].astype(str).str.strip().str.lower()
    return df


def assign_vehicle_cv_scores(gt_events: pd.DataFrame, run_id: int) -> pd.DataFrame:
    """
    Create reproducible per-vehicle random scores for nested CV sampling.

    This table intentionally keeps the vehicle-level GT event information,
    because later scripts should be able to read the CV allocation file directly
    without re-opening the GT event file.
    """
    keep_cols = [
        "run_id",
        "veh_uid",
        "joined_queue",
        "validation_category",
        "official_join_rule",
        "t_queue_join_sec",
        "s_queue_join_ft",
        "queue_length_at_join_ft",
        "speed_at_join_fps",
        "t_exit_sec",
        "t_event_sec",
        "queue_length_for_event_ft",
    ]
    keep_cols = [c for c in keep_cols if c in gt_events.columns]

    veh = gt_events[keep_cols].drop_duplicates("veh_uid").copy()
    veh["run_id"] = int(run_id)

    # Event-order index, equivalent to the old N/N_gt idea but based on the
    # revised event time: queued vehicles use t_queue_join_sec; nonqueued use exit.
    if "t_event_sec" in veh.columns:
        veh = veh.sort_values(["t_event_sec", "veh_uid"]).reset_index(drop=True)
    else:
        veh = veh.sort_values("veh_uid").reset_index(drop=True)
    veh["N_event"] = np.arange(1, len(veh) + 1, dtype=int)
    max_n = float(veh["N_event"].max()) if len(veh) else 0.0
    veh["N_event_norm"] = veh["N_event"] / max_n if max_n > 0 else 0.0

    if NESTED_CV_SAMPLING:
        rng = np.random.default_rng(RANDOM_SEED + int(run_id) * 1009)
        veh["cv_score"] = rng.random(len(veh))
    else:
        # Stored score still exists for traceability, but each rate will resample separately.
        veh["cv_score"] = np.nan

    return veh


def build_cv_allocation(gt_events: pd.DataFrame, run_id: int, rate: int, base_scores: pd.DataFrame) -> pd.DataFrame:
    veh = base_scores.copy()

    if NESTED_CV_SAMPLING:
        veh["is_cv"] = (veh["cv_score"] <= float(rate) / 100.0).astype(int)
    else:
        rng = np.random.default_rng(RANDOM_SEED + int(run_id) * 1009 + int(rate) * 17)
        veh["cv_score"] = rng.random(len(veh))
        veh["is_cv"] = (veh["cv_score"] <= float(rate) / 100.0).astype(int)

    # Force first and last event-order vehicles to be CVs for boundary anchoring.
    # These boundary anchors use exit time and zero queue length, even if the
    # vehicle also has a normal queue-join record. This gives a fixed zero-queue
    # start/end boundary for later CV-segment correction.
    veh["is_forced_boundary_cv"] = 0
    veh["boundary_role"] = ""
    if len(veh) >= 1:
        first_idx = veh["N_event"].idxmin()
        last_idx = veh["N_event"].idxmax()
        veh.loc[first_idx, "is_cv"] = 1
        veh.loc[first_idx, "is_forced_boundary_cv"] = 1
        veh.loc[first_idx, "boundary_role"] = "first"
        veh.loc[last_idx, "is_cv"] = 1
        veh.loc[last_idx, "is_forced_boundary_cv"] = 1
        if first_idx == last_idx:
            veh.loc[last_idx, "boundary_role"] = "first+last"
        else:
            veh.loc[last_idx, "boundary_role"] = "last"

    veh["cv_rate_pct"] = int(rate)
    veh["run_id"] = int(run_id)

    preferred = [
        "run_id",
        "cv_rate_pct",
        "veh_uid",
        "N_event",
        "N_event_norm",
        "cv_score",
        "is_cv",
        "is_forced_boundary_cv",
        "boundary_role",
        "joined_queue",
        "validation_category",
        "official_join_rule",
        "t_queue_join_sec",
        "s_queue_join_ft",
        "queue_length_at_join_ft",
        "speed_at_join_fps",
        "t_exit_sec",
        "t_event_sec",
        "queue_length_for_event_ft",
    ]
    cols = [c for c in preferred if c in veh.columns] + [c for c in veh.columns if c not in preferred]
    return veh[cols].copy()


def build_cv_anchors(gt_events: pd.DataFrame, cv_alloc: pd.DataFrame, gt_grid: pd.DataFrame, run_id: int, rate: int) -> pd.DataFrame:
    """
    Build CV queue-length anchors.

    Normal queued CV vehicles provide anchors in the SAME queue-length space
    as the final saved/plotted GT curve:
        time = t_queue_join_sec
        q    = q_gt_ft from gt_queue_length_timegrid_runXXX.csv at that time

    Forced boundary CV vehicles provide zero-queue boundary anchors:
        time = t_exit_sec, q = 0

    The direct vehicle-level join distance |s_join| is kept as
    cv_anchor_q_direct_join_ft only for diagnostics. It is not used for anchor
    correction because the final continuous GT curve is q_gt_ft.
    """
    cv = cv_alloc[cv_alloc["is_cv"] == 1].copy()
    if cv.empty:
        return pd.DataFrame()

    # Start from the allocation table so forced-boundary metadata is retained.
    anchors = cv.copy()

    required = ["t_exit_sec", "joined_queue"]
    require_columns(anchors, required, f"CV allocation run {run_id:03d} rate {rate:03d}")

    anchors["joined_queue"] = bool_to_int(anchors["joined_queue"])
    anchors["is_forced_boundary_cv"] = anchors.get("is_forced_boundary_cv", 0)
    anchors["is_forced_boundary_cv"] = pd.to_numeric(anchors["is_forced_boundary_cv"], errors="coerce").fillna(0).astype(int)

    # Normal anchors: queued CVs that are not forced boundaries.
    normal_mask = (
        (anchors["is_forced_boundary_cv"] == 0)
        & (anchors["joined_queue"].astype(int) == 1)
        & anchors["t_queue_join_sec"].notna()
        & anchors["queue_length_at_join_ft"].notna()
    )

    # Boundary anchors: forced first/last vehicles with zero queue length at exit time.
    boundary_mask = (anchors["is_forced_boundary_cv"] == 1) & anchors["t_exit_sec"].notna()

    anchors = anchors[normal_mask | boundary_mask].copy()
    if anchors.empty:
        return pd.DataFrame()

    anchors["is_boundary_anchor"] = (anchors["is_forced_boundary_cv"] == 1).astype(int)
    anchors["is_cv_anchor"] = 1

    anchors["cv_anchor_time_sec"] = np.where(
        anchors["is_boundary_anchor"] == 1,
        pd.to_numeric(anchors["t_exit_sec"], errors="coerce"),
        pd.to_numeric(anchors["t_queue_join_sec"], errors="coerce"),
    )
    # Keep the old direct join-distance queue length only for diagnostics.
    anchors["cv_anchor_q_direct_join_ft"] = np.where(
        anchors["is_boundary_anchor"] == 1,
        0.0,
        pd.to_numeric(anchors["queue_length_at_join_ft"], errors="coerce"),
    )

    # Correct anchor queue value: use the saved final GT time-grid curve value
    # at the anchor time. This makes CV dots lie exactly on the plotted GT curve.
    q_from_timegrid = lookup_q_gt_at_times(gt_grid, anchors["cv_anchor_time_sec"])
    anchors["cv_anchor_q_ft"] = np.where(
        anchors["is_boundary_anchor"] == 1,
        0.0,
        q_from_timegrid,
    )
    anchors["cv_anchor_s_join_ft"] = np.where(
        anchors["is_boundary_anchor"] == 1,
        0.0,
        pd.to_numeric(anchors.get("s_queue_join_ft", np.nan), errors="coerce"),
    )
    anchors["cv_anchor_speed_fps"] = np.where(
        anchors["is_boundary_anchor"] == 1,
        np.nan,
        pd.to_numeric(anchors.get("speed_at_join_fps", np.nan), errors="coerce"),
    )
    anchors["cv_anchor_source"] = np.where(
        anchors["is_boundary_anchor"] == 1,
        "forced_boundary_exit_zero",
        "saved_gt_timegrid_q_at_join_time",
    )

    anchors = anchors.dropna(subset=["cv_anchor_time_sec", "cv_anchor_q_ft"]).copy()
    anchors = anchors[np.isfinite(anchors["cv_anchor_time_sec"]) & np.isfinite(anchors["cv_anchor_q_ft"])].copy()
    anchors = anchors[anchors["cv_anchor_q_ft"] >= 0].copy()

    anchors["run_id"] = int(run_id)
    anchors["cv_rate_pct"] = int(rate)
    
    anchors["is_synthetic_timegrid_boundary"] = 0

    if ADD_SYNTHETIC_TIMEGRID_BOUNDARY_ANCHORS:
        t_grid = pd.to_numeric(gt_grid["time_sec"], errors="coerce").dropna()

        if not t_grid.empty:
            t_start = float(t_grid.min())
            t_end = float(t_grid.max())

            synthetic_rows = []

            for role, t_anchor in [
                ("timegrid_start", t_start),
                ("timegrid_end", t_end),
            ]:
                synthetic_rows.append(
                    {
                        "run_id": int(run_id),
                        "cv_rate_pct": int(rate),
                        "veh_uid": f"synthetic_{role}",
                        "N_event": np.nan,
                        "N_event_norm": np.nan,
                        "cv_score": np.nan,
                        "is_cv": 1,
                        "is_forced_boundary_cv": 1,
                        "boundary_role": role,
                        "is_boundary_anchor": 1,
                        "is_synthetic_timegrid_boundary": 1,
                        "is_cv_anchor": 1,
                        "cv_anchor_source": f"synthetic_{role}_zero",
                        "joined_queue": 0,
                        "cv_anchor_time_sec": t_anchor,
                        "cv_anchor_q_ft": 0.0,
                        "cv_anchor_q_direct_join_ft": 0.0,
                        "cv_anchor_s_join_ft": 0.0,
                        "cv_anchor_speed_fps": np.nan,
                        "t_queue_join_sec": np.nan,
                        "queue_length_at_join_ft": np.nan,
                        "t_exit_sec": np.nan,
                        "t_event_sec": np.nan,
                    }
                )

        anchors = pd.concat(
            [anchors, pd.DataFrame(synthetic_rows)],
            ignore_index=True,
            sort=False,
        )
    
    
    anchors = anchors.sort_values(["cv_anchor_time_sec", "veh_uid", "is_boundary_anchor"]).reset_index(drop=True)
    anchors["cv_anchor_order"] = np.arange(1, len(anchors) + 1, dtype=int)

    preferred = [
        "run_id",
        "cv_rate_pct",
        "veh_uid",
        "N_event",
        "N_event_norm",
        "cv_anchor_order",
        "cv_score",
        "is_cv",
        "is_forced_boundary_cv",
        "boundary_role",
        "is_boundary_anchor",
        "is_synthetic_timegrid_boundary",
        "is_cv_anchor",
        "cv_anchor_source",
        "joined_queue",
        "validation_category",
        "official_join_rule",
        "cv_anchor_time_sec",
        "cv_anchor_q_ft",
        "cv_anchor_q_direct_join_ft",
        "cv_anchor_s_join_ft",
        "cv_anchor_speed_fps",
        "t_queue_join_sec",
        "queue_length_at_join_ft",
        "t_exit_sec",
        "t_event_sec",
    ]
    cols = [c for c in preferred if c in anchors.columns] + [c for c in anchors.columns if c not in preferred]
    return anchors[cols].copy()

def merge_gt_and_baseline_timegrids(gt_grid: pd.DataFrame, baseline: pd.DataFrame, run_id: int) -> pd.DataFrame:
    """
    Merge baseline and GT on time; baseline is the main time grid.

    First use an exact merge on rounded time stamps. The as-of tolerance is only
    a fallback for tiny floating-point grid differences between saved CSVs.
    """
    b = baseline.sort_values("time_sec").copy()
    g = gt_grid.sort_values("time_sec").copy()

    b["time_key"] = b["time_sec"].round(6)
    g["time_key"] = g["time_sec"].round(6)

    merged = b.merge(g[["time_key", "q_gt_ft"]], on="time_key", how="left")
    missing = int(merged["q_gt_ft"].isna().sum()) if "q_gt_ft" in merged.columns else len(merged)

    if missing > 0:
        b2 = b.drop(columns=["time_key"]).sort_values("time_sec").copy()
        g2 = g.drop(columns=["time_key"]).sort_values("time_sec").copy()
        fallback = pd.merge_asof(
            b2,
            g2[["time_sec", "q_gt_ft"]],
            on="time_sec",
            direction="nearest",
            tolerance=ASOF_TOL_SEC,
        )
        # Fill only missing rows from fallback.
        merged_no_key = merged.drop(columns=["time_key"]).copy()
        fill_mask = merged_no_key["q_gt_ft"].isna()
        merged_no_key.loc[fill_mask, "q_gt_ft"] = fallback.loc[fill_mask, "q_gt_ft"].to_numpy()
        merged = merged_no_key
    else:
        merged = merged.drop(columns=["time_key"])

    merged["run_id"] = int(run_id)
    return merged


def add_phase_and_count_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add phase, phase-elapsed, and count-based diagnostic features.

    Important modeling decision:
    ----------------------------
    Absolute simulation time variables are retained for sorting/merging/windowing,
    but they should NOT be used as ML predictors.

    Main ML-relevant phase features:
        phase_state
        phase_elapsed_sec

    phase_elapsed_sec is computed as:
        current time - start time of the current continuous phase-state interval
    """
    out = df.copy()

    if "phase_state" not in out.columns:
        out["phase_state"] = "unknown"

    out["phase_state"] = out["phase_state"].astype(str).str.strip().str.lower()

    # Keep these indicator columns for diagnostics/backward compatibility.
    # Future ML scripts can choose not to use them.
    out["phase_green"] = (out["phase_state"] == "green").astype(int)
    out["phase_amber"] = (out["phase_state"] == "amber").astype(int)
    out["phase_red"] = (out["phase_state"] == "red").astype(int)

    # Sort before all time-dependent feature calculations.
    out = out.sort_values(["run_id", "time_sec"]).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Time columns retained for indexing/sorting/sequence construction only.
    # These should not be used as direct ML input features.
    # -------------------------------------------------------------------------
    out["time_step_sec"] = out.groupby("run_id")["time_sec"].diff().fillna(0.0)
    out["time_in_run_sec"] = out["time_sec"] - out.groupby("run_id")["time_sec"].transform("min")
    run_duration = out.groupby("run_id")["time_in_run_sec"].transform("max").replace(0, np.nan)
    out["time_norm_run"] = (out["time_in_run_sec"] / run_duration).fillna(0.0)

    # -------------------------------------------------------------------------
    # New phase elapsed feature.
    # This is the usable phase-timing feature for ML.
    # -------------------------------------------------------------------------
    phase_change = (
        out["phase_state"]
        .ne(out.groupby("run_id")["phase_state"].shift())
        .astype(int)
    )
    out["_phase_segment_id"] = phase_change.groupby(out["run_id"]).cumsum()

    phase_segment_start_time = out.groupby(
        ["run_id", "_phase_segment_id"]
    )["time_sec"].transform("first")

    out["phase_elapsed_sec"] = out["time_sec"] - phase_segment_start_time
    out["phase_elapsed_sec"] = (
        pd.to_numeric(out["phase_elapsed_sec"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    out = out.drop(columns=["_phase_segment_id"])

    # -------------------------------------------------------------------------
    # Count/baseline-derived diagnostic features.
    # These are retained for backward compatibility and later method variants.
    # Future training scripts will select only the allowed feature subset.
    # -------------------------------------------------------------------------
    if "n_queue_cumulative" in out.columns:
        out["n_queue_cumulative"] = pd.to_numeric(
            out["n_queue_cumulative"], errors="coerce"
        ).fillna(0)
        out["delta_n_queue"] = out.groupby("run_id")["n_queue_cumulative"].diff().fillna(0)
    else:
        out["n_queue_cumulative"] = np.nan
        out["delta_n_queue"] = np.nan

    if "q_baseline_fixed_ft" in out.columns:
        out["q_baseline_fixed_ft"] = pd.to_numeric(
            out["q_baseline_fixed_ft"], errors="coerce"
        ).fillna(0)
        out["delta_q_baseline_ft"] = out.groupby("run_id")["q_baseline_fixed_ft"].diff().fillna(0)
    else:
        out["q_baseline_fixed_ft"] = np.nan
        out["delta_q_baseline_ft"] = np.nan

    dt = out["time_step_sec"].replace(0, np.nan)

    out["slope_q_baseline_ftps"] = (
        out["delta_q_baseline_ft"] / dt
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["slope_n_queue_per_s"] = (
        out["delta_n_queue"] / dt
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["delta_q_baseline_next_ft"] = (
        out.groupby("run_id")["q_baseline_fixed_ft"].shift(-1)
        - out["q_baseline_fixed_ft"]
    )
    out["delta_n_queue_next"] = (
        out.groupby("run_id")["n_queue_cumulative"].shift(-1)
        - out["n_queue_cumulative"]
    )

    out["delta_q_baseline_next_ft"] = out["delta_q_baseline_next_ft"].fillna(0.0)
    out["delta_n_queue_next"] = out["delta_n_queue_next"].fillna(0.0)

    max_q = out.groupby("run_id")["q_baseline_fixed_ft"].transform("max").replace(0, np.nan)
    max_nq = out.groupby("run_id")["n_queue_cumulative"].transform("max").replace(0, np.nan)

    out["q_baseline_norm_run"] = (out["q_baseline_fixed_ft"] / max_q).fillna(0.0)
    out["n_queue_norm_run"] = (out["n_queue_cumulative"] / max_nq).fillna(0.0)

    # Retained as a diagnostic/backward-compatible column.
    # We will not use this in the new minimal feature sets.
    cond_noq = out["n_queue_cumulative"].fillna(0) <= 0
    cond_depart = (out["phase_green"] == 1) & (out["delta_n_queue"] < 0)
    cond_form = (
        (out["phase_red"].eq(1) | out["phase_amber"].eq(1))
        & (out["delta_n_queue"] > 0)
    )
    cond_stand = (
        (out["phase_red"].eq(1) | out["phase_amber"].eq(1))
        & (out["n_queue_cumulative"].fillna(0) > 0)
    )

    out["observable_queue_state"] = "uncertain"
    out.loc[cond_noq, "observable_queue_state"] = "no_queue"
    out.loc[cond_form, "observable_queue_state"] = "formation"
    out.loc[cond_stand & ~cond_form, "observable_queue_state"] = "standing"
    out.loc[cond_depart, "observable_queue_state"] = "departure"

    return out


def add_cv_segment_features(df: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """Add previous/next CV anchor features and linear interpolation values."""
    out = df.sort_values("time_sec").copy().reset_index(drop=True)

    # Initialize defaults.
    defaults_float = [
        "prev_cv_anchor_time_sec",
        "prev_cv_anchor_q_ft",
        "prev_cv_anchor_order",
        "prev_cv_anchor_baseline_q_ft",
        "prev_cv_anchor_residual_ft",
        "next_cv_anchor_time_sec",
        "next_cv_anchor_q_ft",
        "next_cv_anchor_order",
        "next_cv_anchor_baseline_q_ft",
        "next_cv_anchor_residual_ft",
        "time_since_prev_cv_sec",
        "time_to_next_cv_sec",
        "cv_segment_duration_sec",
        "cv_segment_q_delta_ft",
        "cv_segment_baseline_delta_ft",
        "cv_segment_residual_delta_ft",
        "cv_segment_frac",
        "q_cv_linear_interp_ft",
        "residual_from_cv_interp_ft",
        "q_gap_from_prev_cv_anchor_ft",
        "q_gap_to_next_cv_anchor_ft",
        "baseline_q_gap_from_prev_cv_anchor_ft",
        "baseline_q_gap_to_next_cv_anchor_ft",
    ]
    for col in defaults_float:
        out[col] = np.nan
    out["inside_cv_segment"] = 0
    out["cv_segment_id"] = -1
    out["cv_anchor_order_gap"] = np.nan
    out["cv_seen_so_far"] = 0
    out["is_cv_anchor_time"] = 0
    out["n_cv_anchors_run_rate"] = int(len(anchors))

    if "q_gt_ft" in out.columns and "q_baseline_fixed_ft" in out.columns:
        out["target_residual_from_baseline_ft"] = out["q_gt_ft"] - out["q_baseline_fixed_ft"]

    if anchors.empty or len(anchors) < 2:
        return out

    a = anchors.sort_values("cv_anchor_time_sec").drop_duplicates("cv_anchor_time_sec").reset_index(drop=True)
    a_times = a["cv_anchor_time_sec"].to_numpy(dtype=float)
    a_q = a["cv_anchor_q_ft"].to_numpy(dtype=float)
    if "cv_anchor_order" in a.columns:
        a_order = a["cv_anchor_order"].to_numpy(dtype=float)
    else:
        a_order = np.arange(1, len(a) + 1, dtype=float)

    t = out["time_sec"].to_numpy(dtype=float)
    q_base = out["q_baseline_fixed_ft"].to_numpy(dtype=float) if "q_baseline_fixed_ft" in out.columns else np.full(len(out), np.nan)

    # Baseline value at anchor times. This is observable because it comes from the baseline curve.
    valid_base = np.isfinite(t) & np.isfinite(q_base)
    if np.sum(valid_base) >= 2:
        a_base_q = np.interp(a_times, t[valid_base], q_base[valid_base])
    else:
        a_base_q = np.full(len(a_times), np.nan)
    a_resid = a_q - a_base_q

    right_idx = np.searchsorted(a_times, t, side="right")
    left_idx = right_idx - 1

    # Latest CV-so-far features, analogous to the old prev_cv_join_sec / latest_cv_N.
    latest_valid = left_idx >= 0
    out.loc[latest_valid, "cv_seen_so_far"] = 1
    out.loc[latest_valid, "prev_cv_anchor_time_sec"] = a_times[left_idx[latest_valid]]
    out.loc[latest_valid, "prev_cv_anchor_q_ft"] = a_q[left_idx[latest_valid]]
    out.loc[latest_valid, "prev_cv_anchor_order"] = a_order[left_idx[latest_valid]]
    out.loc[latest_valid, "prev_cv_anchor_baseline_q_ft"] = a_base_q[left_idx[latest_valid]]
    out.loc[latest_valid, "prev_cv_anchor_residual_ft"] = a_resid[left_idx[latest_valid]]
    out.loc[latest_valid, "time_since_prev_cv_sec"] = t[latest_valid] - a_times[left_idx[latest_valid]]

    # Next CV anchor features.
    next_valid = right_idx < len(a_times)
    out.loc[next_valid, "next_cv_anchor_time_sec"] = a_times[right_idx[next_valid]]
    out.loc[next_valid, "next_cv_anchor_q_ft"] = a_q[right_idx[next_valid]]
    out.loc[next_valid, "next_cv_anchor_order"] = a_order[right_idx[next_valid]]
    out.loc[next_valid, "next_cv_anchor_baseline_q_ft"] = a_base_q[right_idx[next_valid]]
    out.loc[next_valid, "next_cv_anchor_residual_ft"] = a_resid[right_idx[next_valid]]
    out.loc[next_valid, "time_to_next_cv_sec"] = a_times[right_idx[next_valid]] - t[next_valid]

    # Segment rows are valid only when both previous and next anchor exist.
    # Important: do not index a_times[right_idx] before masking because
    # rows after the last anchor have right_idx == len(a_times).
    valid = (left_idx >= 0) & (right_idx < len(a_times))
    same = np.zeros(len(t), dtype=bool)
    same[valid] = a_times[right_idx[valid]] > a_times[left_idx[valid]]

    out.loc[same, "inside_cv_segment"] = 1
    out.loc[same, "cv_segment_duration_sec"] = a_times[right_idx[same]] - a_times[left_idx[same]]
    out.loc[same, "cv_segment_q_delta_ft"] = a_q[right_idx[same]] - a_q[left_idx[same]]
    out.loc[same, "cv_segment_baseline_delta_ft"] = a_base_q[right_idx[same]] - a_base_q[left_idx[same]]
    out.loc[same, "cv_segment_residual_delta_ft"] = a_resid[right_idx[same]] - a_resid[left_idx[same]]
    out.loc[same, "cv_anchor_order_gap"] = a_order[right_idx[same]] - a_order[left_idx[same]]

    frac = (t[same] - a_times[left_idx[same]]) / (a_times[right_idx[same]] - a_times[left_idx[same]])
    out.loc[same, "cv_segment_frac"] = frac
    out.loc[same, "q_cv_linear_interp_ft"] = a_q[left_idx[same]] + frac * (a_q[right_idx[same]] - a_q[left_idx[same]])
    out.loc[same, "cv_segment_id"] = left_idx[same] + 1

    # Relative-to-anchor queue-length features, similar in spirit to old count/time-gap features.
    out.loc[latest_valid, "q_gap_from_prev_cv_anchor_ft"] = out.loc[latest_valid, "q_baseline_fixed_ft"] - a_q[left_idx[latest_valid]]
    out.loc[next_valid, "q_gap_to_next_cv_anchor_ft"] = a_q[right_idx[next_valid]] - out.loc[next_valid, "q_baseline_fixed_ft"]
    out.loc[latest_valid, "baseline_q_gap_from_prev_cv_anchor_ft"] = out.loc[latest_valid, "q_baseline_fixed_ft"] - a_base_q[left_idx[latest_valid]]
    out.loc[next_valid, "baseline_q_gap_to_next_cv_anchor_ft"] = a_base_q[right_idx[next_valid]] - out.loc[next_valid, "q_baseline_fixed_ft"]

    # Flag grid rows that coincide with a CV anchor time.
    if len(a_times):
        nearest_idx = np.searchsorted(t, a_times, side="left")
        nearest_idx = nearest_idx[(nearest_idx >= 0) & (nearest_idx < len(t))]
        out.loc[nearest_idx, "is_cv_anchor_time"] = 1

    if "q_gt_ft" in out.columns:
        out["residual_from_cv_interp_ft"] = out["q_gt_ft"] - out["q_cv_linear_interp_ft"]

    return out

def add_split_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["run_split"] = "other"

    out.loc[out["run_id"].isin(TRAIN_RUN_IDS), "run_split"] = "train"
    out.loc[out["run_id"].isin(VALIDATION_RUN_IDS), "run_split"] = "validation"
    out.loc[out["run_id"].isin(TEST_RUN_IDS), "run_split"] = "test"

    return out


# =============================================================================
# Main processing
# =============================================================================

def process_one_run_rate(run_id: int, rate: int, base_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gt_events = load_gt_events(run_id)
    gt_grid = load_gt_timegrid(run_id)
    baseline = load_baseline_timegrid(run_id)

    cv_alloc = build_cv_allocation(gt_events, run_id, rate, base_scores)
    cv_anchors = build_cv_anchors(gt_events, cv_alloc, gt_grid, run_id, rate)

    timegrid = merge_gt_and_baseline_timegrids(gt_grid, baseline, run_id)
    timegrid = add_phase_and_count_features(timegrid)
    timegrid = add_cv_segment_features(timegrid, cv_anchors)
    timegrid["cv_rate_pct"] = int(rate)
    timegrid = add_split_columns(timegrid)

    # Reorder important columns first.
    preferred = [
        "run_id",
        "run_split",
        "cv_rate_pct",
        "time_sec",
        "time_in_run_sec",
        "time_norm_run",
        "time_step_sec",
        "phase_state",
        "phase_elapsed_sec",
        "phase_green",
        "phase_amber",
        "phase_red",
        "A_count",
        "D_count",
        "V_count",
        "B_count",
        "n_queue_cumulative",
        "n_queue_norm_run",
        "delta_n_queue",
        "delta_n_queue_next",
        "slope_n_queue_per_s",
        "q_baseline_fixed_ft",
        "q_baseline_norm_run",
        "delta_q_baseline_ft",
        "delta_q_baseline_next_ft",
        "slope_q_baseline_ftps",
        "q_gt_ft",
        "target_residual_from_baseline_ft",
        "observable_queue_state",
        "n_cv_anchors_run_rate",
        "inside_cv_segment",
        "cv_segment_id",
        "cv_seen_so_far",
        "is_cv_anchor_time",
        "prev_cv_anchor_time_sec",
        "prev_cv_anchor_q_ft",
        "prev_cv_anchor_order",
        "prev_cv_anchor_baseline_q_ft",
        "prev_cv_anchor_residual_ft",
        "next_cv_anchor_time_sec",
        "next_cv_anchor_q_ft",
        "next_cv_anchor_order",
        "next_cv_anchor_baseline_q_ft",
        "next_cv_anchor_residual_ft",
        "time_since_prev_cv_sec",
        "time_to_next_cv_sec",
        "cv_segment_duration_sec",
        "cv_anchor_order_gap",
        "cv_segment_q_delta_ft",
        "cv_segment_baseline_delta_ft",
        "cv_segment_residual_delta_ft",
        "cv_segment_frac",
        "q_gap_from_prev_cv_anchor_ft",
        "q_gap_to_next_cv_anchor_ft",
        "baseline_q_gap_from_prev_cv_anchor_ft",
        "baseline_q_gap_to_next_cv_anchor_ft",
        "q_cv_linear_interp_ft",
        "residual_from_cv_interp_ft",
        "l_eff_fixed_ft",
    ]
    cols = [c for c in preferred if c in timegrid.columns] + [c for c in timegrid.columns if c not in preferred]
    timegrid = timegrid[cols].copy()

    return cv_alloc, cv_anchors, timegrid


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("Revised CV sampling and queue-length feature engineering")
    print("=" * 88)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"GT dir       : {GT_DIR}")
    print(f"Baseline dir : {BASELINE_DIR}")
    print(f"Output dir   : {OUT_DIR}")
    print(f"Runs         : {RUN_IDS}")
    print(f"CV rates     : {CV_RATES_PCT}")
    print("=" * 88)

    # Load GT event files once per run to create stable vehicle scores.
    base_scores_by_run: dict[int, pd.DataFrame] = {}
    for run_id in RUN_IDS:
        gt_events = load_gt_events(run_id)
        base_scores_by_run[int(run_id)] = assign_vehicle_cv_scores(gt_events, int(run_id))

    all_features = []

    for rate in CV_RATES_PCT:
        rate_allocs = []
        rate_anchors = []
        rate_features = []

        for run_id in RUN_IDS:
            print(f"\n[Run {run_id:03d} | CV {rate:03d}%]")
            cv_alloc, cv_anchors, features = process_one_run_rate(
                int(run_id), int(rate), base_scores_by_run[int(run_id)]
            )

            alloc_path = fmt(CV_ALLOC_PATTERN, run_id=int(run_id), rate=int(rate))
            anchors_path = fmt(CV_ANCHORS_PATTERN, run_id=int(run_id), rate=int(rate))
            features_path = fmt(FEATURES_PATTERN, run_id=int(run_id), rate=int(rate))

            cv_alloc.to_csv(alloc_path, index=False)
            cv_anchors.to_csv(anchors_path, index=False)
            if SAVE_PER_RUN_RATE_FEATURES:
                features.to_csv(features_path, index=False)

            n_cv = int(cv_alloc["is_cv"].sum())
            n_veh = int(len(cv_alloc))
            n_anchor = int(len(cv_anchors))
            n_inside = int(features["inside_cv_segment"].sum()) if "inside_cv_segment" in features.columns else 0

            print(f"  vehicles={n_veh:,}, selected CV={n_cv:,}, queued CV anchors={n_anchor:,}, feature rows in CV segments={n_inside:,}")
            print(f"  [Saved] {alloc_path}")
            print(f"  [Saved] {anchors_path}")
            if SAVE_PER_RUN_RATE_FEATURES:
                print(f"  [Saved] {features_path}")

            rate_allocs.append(cv_alloc)
            rate_anchors.append(cv_anchors)
            rate_features.append(features)

        alloc_all = pd.concat(rate_allocs, ignore_index=True) if rate_allocs else pd.DataFrame()
        anchors_all = pd.concat(rate_anchors, ignore_index=True) if rate_anchors else pd.DataFrame()
        features_all_rate = pd.concat(rate_features, ignore_index=True) if rate_features else pd.DataFrame()

        alloc_all_path = fmt(CV_ALLOC_ALLRUNS_PATTERN, rate=int(rate))
        anchors_all_path = fmt(CV_ANCHORS_ALLRUNS_PATTERN, rate=int(rate))
        features_all_rate_path = fmt(FEATURES_ALLRUNS_PATTERN, rate=int(rate))

        alloc_all.to_csv(alloc_all_path, index=False)
        anchors_all.to_csv(anchors_all_path, index=False)
        features_all_rate.to_csv(features_all_rate_path, index=False)

        print(f"\n[Saved all-runs allocation] {alloc_all_path}")
        print(f"[Saved all-runs anchors]    {anchors_all_path}")
        print(f"[Saved all-runs features]   {features_all_rate_path}")

        all_features.append(features_all_rate)

    final_features = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()
    final_features.to_csv(FEATURES_ALLRUNS_ALLRATES, index=False)
    print(f"\n[Saved combined feature file] {FEATURES_ALLRUNS_ALLRATES}")
    print("Done.")


if __name__ == "__main__":
    main()
