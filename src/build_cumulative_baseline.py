"""
Revised cumulative-count baseline queue-length construction.

Place this file at:
    src/build_cumulative_baseline.py

Purpose
-------
This script uses the existing cumulative-count theory outputs to produce:

1) A/D cumulative-count plot.
2) A/D/V/B cumulative-count plot.
3) Baseline queue length vs. time plot with phase ribbon.

The queue-length baseline is derived only from observable cumulative-count
information:

    n_queue(t) = max(0, V_count(t) - D_count(t))
    q_baseline_ft(t) = l_fixed * n_queue(t)

where l_fixed is read from the revised GT empirical-spacing file. We use the
same fixed l_eff for all baseline states, as decided, because the creep spacing
sample is sparse and much different from the other values.

This script does NOT use CV anchors and does NOT use GT queue-join information
for method construction. GT q_gt_ft is only merged if available for later visual
comparison/diagnostics.
"""

from __future__ import annotations

from config import (
    PROJECT_ROOT,
    RUN_IDS,
    TIMEGRID_DT_SEC
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

# Runs to compute and runs to plot.
PLOT_RUN_IDS = [5]

# Time grid. If GT time-grid exists for a run, the script uses that same axis.
# Otherwise it creates a grid using TIMEGRID_DT_SEC.

# Phase ribbon settings.
PLOT_CONTROLLER = 1
PLOT_SIGNAL_GROUP = 2

# Use fixed l_eff for all baseline queue-length conversion.
LEFF_FALLBACK_FT = 25.0
LEFF_COLUMN = "l_fixed_all_ft"

# Optional comparison overlay in queue-length plot.
SHOW_GT_ON_QUEUE_LENGTH_PLOT = False

# Plot settings.
FIGURE_DPI = 250
SAVE_FIGURES = False
SHOW_FIGURES = False

# Input paths from original/revised pipeline.
PROCESSED_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "preprocessing"
REVISED_GT_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "gt"
OUT_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "baseline"
FIG_DIR = OUT_DIR / "figures"

AD_TIMES_CSV = PROCESSED_DIR / "ad_times_all_runs.csv"
VB_CURVES_CSV = PROCESSED_DIR / "vb_curves_all_runs.csv"
CUMULATIVE_PLOT_READY_CSV = PROCESSED_DIR / "cumulative_curves_plot_ready_allruns.csv"

GT_SPACING_BY_RUN_CSV = REVISED_GT_DIR / "gt_empirical_spacing_by_run.csv"
GT_TIMEGRID_PATTERN = REVISED_GT_DIR / "gt_queue_length_timegrid_run{run_id:03d}.csv"

PHASE_CANDIDATES = [
    PROCESSED_DIR / "master_phase_time.csv",
    PROCESSED_DIR / "master_phase_time_10.csv",
]


# =============================================================================
# Basic helpers
# =============================================================================

def require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def find_existing_path(candidates: list[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def cumulative_count_at_times(event_times: np.ndarray, grid_times: np.ndarray) -> np.ndarray:
    """Return cumulative event count at each grid time."""
    ev = np.asarray(event_times, dtype=float)
    ev = ev[np.isfinite(ev)]
    ev.sort()
    if len(ev) == 0:
        return np.zeros(len(grid_times), dtype=int)
    return np.searchsorted(ev, grid_times, side="right").astype(int)


def event_count_on_grid(event_times: np.ndarray, grid_times: np.ndarray) -> np.ndarray:
    """Map event times to nearest grid time and count events per grid row."""
    counts = np.zeros(len(grid_times), dtype=int)
    ev = np.asarray(event_times, dtype=float)
    ev = ev[np.isfinite(ev)]
    if len(ev) == 0 or len(grid_times) == 0:
        return counts

    idx = np.searchsorted(grid_times, ev, side="left")
    idx[idx >= len(grid_times)] = len(grid_times) - 1
    mask = (idx > 0) & (np.abs(grid_times[idx] - ev) > np.abs(grid_times[idx - 1] - ev))
    idx[mask] -= 1
    for i in idx:
        counts[int(i)] += 1
    return counts


# =============================================================================
# Phase helpers
# =============================================================================

def load_phase_file() -> pd.DataFrame:
    phase_path = find_existing_path(PHASE_CANDIDATES)
    if phase_path is None:
        print("[WARN] No phase file found. Queue-length plot will not include phase ribbon.")
        return pd.DataFrame(columns=["run_id", "simsec", "controller", "signal_group", "state"])

    phase = pd.read_csv(phase_path)
    require_columns(phase, {"run_id", "simsec", "controller", "signal_group", "state"}, phase_path.name)

    phase["run_id"] = safe_numeric(phase["run_id"])
    phase["simsec"] = safe_numeric(phase["simsec"])
    phase["controller"] = safe_numeric(phase["controller"])
    phase["signal_group"] = safe_numeric(phase["signal_group"])
    phase["state"] = phase["state"].astype(str).str.strip().str.lower()
    phase = phase.dropna(subset=["run_id", "simsec", "controller", "signal_group", "state"]).copy()
    phase["run_id"] = phase["run_id"].astype(int)
    phase["controller"] = phase["controller"].astype(int)
    phase["signal_group"] = phase["signal_group"].astype(int)

    print(f"[Loaded] phase file: {phase_path}")
    return phase


def build_signal_intervals(phase_run: pd.DataFrame) -> pd.DataFrame:
    if phase_run.empty:
        return pd.DataFrame(columns=["t_start", "t_end", "state"])

    d = phase_run.sort_values("simsec").copy()
    d = d.drop_duplicates(subset=["simsec"], keep="last").reset_index(drop=True)
    if len(d) < 2:
        return pd.DataFrame(columns=["t_start", "t_end", "state"])

    return pd.DataFrame(
        {
            "t_start": d["simsec"].to_numpy(dtype=float)[:-1],
            "t_end": d["simsec"].to_numpy(dtype=float)[1:],
            "state": d["state"].astype(str).str.lower().to_numpy()[:-1],
        }
    )


def phase_state_at_grid(intervals: pd.DataFrame, grid_times: np.ndarray) -> np.ndarray:
    states = np.array(["unknown"] * len(grid_times), dtype=object)
    if intervals.empty or len(grid_times) == 0:
        return states

    starts = intervals["t_start"].to_numpy(dtype=float)
    ends = intervals["t_end"].to_numpy(dtype=float)
    st = intervals["state"].astype(str).to_numpy()

    idx = np.searchsorted(starts, grid_times, side="right") - 1
    valid = (idx >= 0) & (idx < len(starts)) & (grid_times < ends[np.clip(idx, 0, len(ends)-1)])
    states[valid] = st[idx[valid]]
    return states


# =============================================================================
# Loading helpers
# =============================================================================

def load_ad_vb() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not AD_TIMES_CSV.exists():
        raise FileNotFoundError(f"Missing A/D event file: {AD_TIMES_CSV}\nRun the original cumulative_count_theory.py first.")
    if not VB_CURVES_CSV.exists():
        raise FileNotFoundError(f"Missing V/B curve file: {VB_CURVES_CSV}\nRun the original cumulative_count_theory.py first.")

    ad = pd.read_csv(AD_TIMES_CSV)
    vb = pd.read_csv(VB_CURVES_CSV)

    require_columns(ad, {"run_id", "veh_uid", "t_arr_updet_sec", "t_dep_stopbar_sec"}, AD_TIMES_CSV.name)
    require_columns(vb, {"run_id", "N", "tV_sec", "tD_sec", "tB_sec"}, VB_CURVES_CSV.name)

    ad["run_id"] = safe_numeric(ad["run_id"])
    ad["t_arr_updet_sec"] = safe_numeric(ad["t_arr_updet_sec"])
    ad["t_dep_stopbar_sec"] = safe_numeric(ad["t_dep_stopbar_sec"])
    ad = ad.dropna(subset=["run_id"]).copy()
    ad["run_id"] = ad["run_id"].astype(int)

    vb["run_id"] = safe_numeric(vb["run_id"])
    vb["N"] = safe_numeric(vb["N"])
    for col in ["tV_sec", "tD_sec", "tB_sec"]:
        vb[col] = safe_numeric(vb[col])
    vb = vb.dropna(subset=["run_id"]).copy()
    vb["run_id"] = vb["run_id"].astype(int)

    print(f"[Loaded] {AD_TIMES_CSV}")
    print(f"[Loaded] {VB_CURVES_CSV}")
    return ad, vb


def load_leff_by_run() -> dict[int, float]:
    leff = {}
    if not GT_SPACING_BY_RUN_CSV.exists():
        print(f"[WARN] Missing {GT_SPACING_BY_RUN_CSV}; using fallback {LEFF_FALLBACK_FT:.2f} ft.")
        return leff

    s = pd.read_csv(GT_SPACING_BY_RUN_CSV)
    require_columns(s, {"run_id", LEFF_COLUMN}, GT_SPACING_BY_RUN_CSV.name)
    s["run_id"] = safe_numeric(s["run_id"])
    s[LEFF_COLUMN] = safe_numeric(s[LEFF_COLUMN])
    s = s.dropna(subset=["run_id"]).copy()
    s["run_id"] = s["run_id"].astype(int)
    for _, r in s.iterrows():
        val = float(r[LEFF_COLUMN]) if pd.notna(r[LEFF_COLUMN]) else np.nan
        if np.isfinite(val) and val > 0:
            leff[int(r["run_id"])] = val
    print(f"[Loaded] fixed l_eff values from {GT_SPACING_BY_RUN_CSV}")
    return leff


def load_gt_timegrid_if_available(run_id: int) -> pd.DataFrame:
    p = Path(str(GT_TIMEGRID_PATTERN).format(run_id=int(run_id)))
    if not p.exists():
        return pd.DataFrame()
    gt = pd.read_csv(p)
    if "time_sec" not in gt.columns:
        return pd.DataFrame()
    gt["time_sec"] = safe_numeric(gt["time_sec"])
    if "q_gt_ft" in gt.columns:
        gt["q_gt_ft"] = safe_numeric(gt["q_gt_ft"])
    return gt.dropna(subset=["time_sec"]).copy()


# =============================================================================
# Baseline construction
# =============================================================================

def build_time_grid(run_id: int, run_ad: pd.DataFrame, run_vb: pd.DataFrame, gt_timegrid: pd.DataFrame) -> np.ndarray:
    if not gt_timegrid.empty:
        return np.sort(gt_timegrid["time_sec"].to_numpy(dtype=float))

    all_times = []
    for col in ["t_arr_updet_sec", "t_dep_stopbar_sec"]:
        all_times.extend(run_ad[col].dropna().to_numpy(dtype=float).tolist())
    for col in ["tV_sec", "tD_sec", "tB_sec"]:
        all_times.extend(run_vb[col].dropna().to_numpy(dtype=float).tolist())

    arr = np.asarray(all_times, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        raise ValueError(f"Run {run_id:03d}: no event times available to create time grid.")
    t0 = math.floor(float(np.nanmin(arr)) / TIMEGRID_DT_SEC) * TIMEGRID_DT_SEC
    t1 = math.ceil(float(np.nanmax(arr)) / TIMEGRID_DT_SEC) * TIMEGRID_DT_SEC
    return np.round(np.arange(t0, t1 + 0.5 * TIMEGRID_DT_SEC, TIMEGRID_DT_SEC), 6)


def build_baseline_for_run(
    run_id: int,
    ad: pd.DataFrame,
    vb: pd.DataFrame,
    phase: pd.DataFrame,
    leff_by_run: dict[int, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_ad = ad[ad["run_id"] == int(run_id)].copy()
    run_vb = vb[vb["run_id"] == int(run_id)].copy()
    if run_ad.empty or run_vb.empty:
        print(f"[WARN] Run {run_id:03d}: missing A/D or V/B rows; skipping.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    gt_timegrid = load_gt_timegrid_if_available(run_id)
    grid_times = build_time_grid(run_id, run_ad, run_vb, gt_timegrid)

    tA = run_ad["t_arr_updet_sec"].dropna().to_numpy(dtype=float)
    tD = run_ad["t_dep_stopbar_sec"].dropna().to_numpy(dtype=float)
    tV = run_vb["tV_sec"].dropna().to_numpy(dtype=float)
    tB = run_vb["tB_sec"].dropna().to_numpy(dtype=float)

    A_count = cumulative_count_at_times(tA, grid_times)
    D_count = cumulative_count_at_times(tD, grid_times)
    V_count = cumulative_count_at_times(tV, grid_times)
    B_count = cumulative_count_at_times(tB, grid_times)

    n_queue = np.maximum(0, V_count - D_count).astype(int)
    delta_n_queue = np.diff(n_queue, prepend=n_queue[0]).astype(int)

    # event counts are useful for diagnostics.
    A_event_count = event_count_on_grid(tA, grid_times)
    D_event_count = event_count_on_grid(tD, grid_times)
    V_event_count = event_count_on_grid(tV, grid_times)
    B_event_count = event_count_on_grid(tB, grid_times)

    l_eff = float(leff_by_run.get(int(run_id), LEFF_FALLBACK_FT))
    q_baseline = n_queue.astype(float) * l_eff

    phase_run = phase[
        (phase["run_id"] == int(run_id))
        & (phase["controller"] == int(PLOT_CONTROLLER))
        & (phase["signal_group"] == int(PLOT_SIGNAL_GROUP))
    ].copy()
    intervals = build_signal_intervals(phase_run)
    phase_state = phase_state_at_grid(intervals, grid_times)

    out = pd.DataFrame(
        {
            "run_id": int(run_id),
            "time_sec": grid_times,
            "phase_state": phase_state,
            "A_count": A_count,
            "D_count": D_count,
            "V_count": V_count,
            "B_count": B_count,
            "A_event_count": A_event_count,
            "D_event_count": D_event_count,
            "V_event_count": V_event_count,
            "B_event_count": B_event_count,
            "n_queue_cumulative": n_queue,
            "delta_n_queue": delta_n_queue,
            "l_eff_fixed_ft": l_eff,
            "q_baseline_fixed_ft": q_baseline,
        }
    )

    if not gt_timegrid.empty and "q_gt_ft" in gt_timegrid.columns:
        gt_small = gt_timegrid[["time_sec", "q_gt_ft"]].copy()
        out = out.merge(gt_small, on="time_sec", how="left")

    # Long-format cumulative curves for this run.
    curve_parts = []
    for curve_type, ev in [("A", tA), ("D", tD), ("V", tV), ("B", tB)]:
        ev = np.asarray(ev, dtype=float)
        ev = ev[np.isfinite(ev)]
        ev.sort()
        if len(ev) == 0:
            continue
        curve_parts.append(
            pd.DataFrame(
                {
                    "run_id": int(run_id),
                    "curve_type": curve_type,
                    "N": np.arange(1, len(ev) + 1, dtype=int),
                    "time_sec": ev,
                }
            )
        )
    curves = pd.concat(curve_parts, ignore_index=True) if curve_parts else pd.DataFrame()

    return out, curves, intervals


# =============================================================================
# Plotting
# =============================================================================

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("Revised cumulative-count baseline queue-length construction")
    print("=" * 88)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Run IDs      : {RUN_IDS}")
    print(f"Plot Run IDs : {PLOT_RUN_IDS}")
    print(f"Output dir   : {OUT_DIR}")
    print("=" * 88)

    ad, vb = load_ad_vb()
    phase = load_phase_file()
    leff_by_run = load_leff_by_run()

    all_baseline = []
    all_curves = []

    for run_id in RUN_IDS:
        baseline, curves, intervals = build_baseline_for_run(run_id, ad, vb, phase, leff_by_run)
        if baseline.empty:
            continue

        out_run = OUT_DIR / f"baseline_queue_count_timegrid_run{run_id:03d}.csv"
        baseline.to_csv(out_run, index=False)
        print(f"[Saved] {out_run}")

        curves_run = OUT_DIR / f"baseline_cumulative_curves_run{run_id:03d}.csv"
        curves.to_csv(curves_run, index=False)
        print(f"[Saved] {curves_run}")

        all_baseline.append(baseline)
        all_curves.append(curves)

        print(
            f"[Run {run_id:03d}] l_eff={baseline['l_eff_fixed_ft'].iloc[0]:.3f} ft | "
            f"max n_queue={baseline['n_queue_cumulative'].max():,} | "
            f"max Q_base={baseline['q_baseline_fixed_ft'].max():.1f} ft"
        )


    if all_baseline:
        combined = pd.concat(all_baseline, ignore_index=True)
        combined_path = OUT_DIR / "baseline_queue_count_timegrid_allruns.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n[Saved] {combined_path}")

    if all_curves:
        combined_curves = pd.concat(all_curves, ignore_index=True)
        combined_curves_path = OUT_DIR / "baseline_cumulative_curves_allruns.csv"
        combined_curves.to_csv(combined_curves_path, index=False)
        print(f"[Saved] {combined_curves_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
