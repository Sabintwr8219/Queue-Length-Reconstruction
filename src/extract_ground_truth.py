"""
Revised GT queue-join, GT queue-length, and GT cumulative-event extraction.

Place this file at:
    src/extract_ground_truth.py

Purpose
-------
This script creates the publication-version ground-truth (GT) foundation files.
GT queue-join locations are computed directly from trajectories. The GT queue-length
time profile is then built as a dynamic event-balance curve using accepted
queue-join counts for formation and queued-vehicle exit counts for dissipation.
No active-tail max rule and no cumulative-count baseline transformation are used
in this script.

Inputs
------
Required:
    data/processed_data/traj_nb_filtered.csv

Optional, only for phase-ribbon plots:
    data/processed_data/master_phase_time.csv or master_phase_time_10.csv

Outputs
-------
    output/intermediate_csv/gt/gt_queue_join_events_runXXX.csv
    output/intermediate_csv/gt/gt_queue_join_events_allruns.csv
    output/intermediate_csv/gt/gt_queue_length_timegrid_runXXX.csv
    output/intermediate_csv/gt/gt_queue_length_timegrid_allruns.csv
    output/intermediate_csv/gt/gt_cumulative_events_runXXX.csv
    output/intermediate_csv/gt/gt_cumulative_events_allruns.csv
    output/intermediate_csv/gt/gt_queue_join_summary_by_run.csv
    output/intermediate_csv/gt/gt_empirical_spacing_by_run.csv

Figures, for PLOT_RUN_IDS only:
    output/intermediate_csv/gt/figures/gt_queue_length_full_runXXX.png
    output/intermediate_csv/gt/figures/gt_queue_length_cycles_runXXX.png
    output/intermediate_csv/gt/figures/gt_cumulative_count_runXXX.png
"""

from __future__ import annotations

from config import (
    PROJECT_ROOT,
    RUN_IDS,
    TIMEGRID_DT_SEC,
    CORRIDOR_MIN_FT,
    CORRIDOR_MAX_FT,
    STOPBAR_FT,
    V_STOP_FPS,
    STOP_PERSIST_SEC,
    CREEP_DROP_FPS,
    CREEP_LOOKBACK_SEC,
    CREEP_PERSIST_SEC,
    USE_NEIGHBOR_SUPPORT_FOR_CREEP,
    NEIGHBOR_TIME_TOL_SEC,
    NEIGHBOR_MAX_GAP_FT,
    NEIGHBOR_LOW_SPEED_FPS,
    NEIGHBOR_DROP_FPS,
    JOIN_PERSIST_SEC,
    BACKTRACK_MAX_SEC,
    ONSET_DROP_EPS_FPS,
    M_TO_FT,
    SPACING_MIN_FT,
    SPACING_MAX_FT,
    SPACING_FALLBACK_FT,
    DISCHARGE_ONLY_DURING_GREEN
)

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# Stage-specific constants
# =============================================================================

# Put this script in: <repo_root>/src/
# Professional repo layout: <project_root>/src/<script>.py

# Input files from original pipeline.
INPUT_TRAJ_CSV = PROJECT_ROOT / "output" / "intermediate_csv" / "preprocessing" / "traj_nb_filtered.csv"
PHASE_FILE_CANDIDATES = [
    PROJECT_ROOT / "output" / "intermediate_csv" / "preprocessing" / "master_phase_time.csv",
    PROJECT_ROOT / "output" / "intermediate_csv" / "preprocessing" / "master_phase_time_10.csv",
]

# Revised output folders.
OUT_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "gt"
FIG_DIR = OUT_DIR / "figures"

# Compute GT for these runs. Test with [5], then switch to list(range(5, 15)).
PLOT_RUN_IDS = [5]

# Time grid for GT queue-length curve.

# Empirical discharge spacing settings for dynamic GT queue-length curve.
# These are estimated from accepted GT queue-join events, not hardcoded.


# Basic plots at the end of this same script.
MAKE_PLOTS = False
MAKE_FULL_TIMELINE_PLOT = False
MAKE_CYCLE_PLOTS = False
MAKE_CUMULATIVE_PLOT = False

# Cycle plot settings.
# These control the phase group used for phase ribbons and cycle splitting.
# Cycle boundaries are green starts, i.e., red ends for the selected signal group.
PLOT_CONTROLLER = 1
PLOT_SIGNAL_GROUP = 2
MAX_CYCLES_TO_PLOT = 12
CYCLE_START_STATE = "green"
FALLBACK_CYCLE_LENGTH_SEC = 150.0

# Plot appearance. Aesthetics can be improved later.
FIGURE_DPI = 250
FULL_FIGSIZE = (14, 5)
CYCLE_FIGSIZE = (16, 10)
CUM_FIGSIZE = (12, 5)
SHOW_FIGURES = False
SAVE_FIGURES = False

# Queue-join detection corridor: stopbar-relative position in feet.

# Standstill rule.

# Creep rule.

# Neighbor support for creep.

# Accepted join persistence.

# Backtracking from confirmation point to slowdown onset.

# Units.


# =============================================================================
# Helper functions
# =============================================================================

def require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def robust_dt(times: np.ndarray, default: float = 0.1) -> float:
    t = np.asarray(times, dtype=float)
    t = t[np.isfinite(t)]
    if len(t) < 3:
        return default
    diffs = np.diff(np.sort(np.unique(t)))
    diffs = diffs[(diffs > 1e-9) & (diffs < 10.0)]
    if len(diffs) == 0:
        return default
    return float(np.median(diffs))


def nearest_index_at_time(times: np.ndarray, target_time: float) -> Optional[int]:
    if len(times) == 0 or not np.isfinite(target_time):
        return None
    if target_time < times[0] or target_time > times[-1]:
        return None
    idx = int(np.searchsorted(times, target_time, side="left"))
    if idx == 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    left = idx - 1
    right = idx
    return right if abs(times[right] - target_time) < abs(times[left] - target_time) else left


def nearest_time_value(time_values: np.ndarray, target_time: float) -> Optional[float]:
    idx = nearest_index_at_time(time_values, target_time)
    if idx is None:
        return None
    return float(time_values[idx])


def first_exit_time(group: pd.DataFrame) -> float:
    """First stopbar crossing time; if not found, use final observation time."""
    g = group.sort_values("Total_Sim_Time_Sec")
    t = g["Total_Sim_Time_Sec"].to_numpy(dtype=float)
    s = g["s_rel_stop_ft"].to_numpy(dtype=float)
    idx = np.where(s >= STOPBAR_FT)[0]
    if len(idx):
        return float(t[idx[0]])
    return float(t[-1]) if len(t) else np.nan


def first_persistent_flag_index(flag: np.ndarray, dt: float) -> Optional[int]:
    min_samples = max(1, int(math.ceil(JOIN_PERSIST_SEC / max(dt, 1e-6))))
    hit = np.where(flag)[0]
    if len(hit) == 0:
        return None

    run_len = 0
    for j in range(len(hit)):
        if j == 0 or hit[j] == hit[j - 1] + 1:
            run_len += 1
        else:
            run_len = 1
        if run_len >= min_samples:
            return int(hit[j - run_len + 1])
    return None


def backtrack_to_slowdown_onset(
    times: np.ndarray,
    speeds: np.ndarray,
    in_corridor: np.ndarray,
    confirm_idx: Optional[int],
) -> Optional[int]:
    if confirm_idx is None:
        return None

    t_confirm = float(times[confirm_idx])
    left_time = t_confirm - BACKTRACK_MAX_SEC
    cand = np.where((times >= left_time) & (times <= t_confirm) & in_corridor)[0]
    if len(cand) == 0:
        return confirm_idx

    local_speeds = speeds[cand]
    if np.all(~np.isfinite(local_speeds)):
        return confirm_idx

    peak_speed = np.nanmax(local_speeds)
    peak_candidates = cand[np.where(np.isclose(local_speeds, peak_speed, equal_nan=False))[0]]
    if len(peak_candidates) == 0:
        return confirm_idx

    peak_idx = int(peak_candidates[-1])
    for j in range(peak_idx, confirm_idx + 1):
        if not in_corridor[j]:
            continue
        if speeds[peak_idx] - speeds[j] >= ONSET_DROP_EPS_FPS:
            return int(j)
    return int(peak_idx)


# =============================================================================
# Queue-join detection logic
# =============================================================================

def build_vehicle_debug_flags(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("Total_Sim_Time_Sec").copy().reset_index(drop=True)

    t = g["Total_Sim_Time_Sec"].to_numpy(dtype=float)
    v = g["Speed_fps"].to_numpy(dtype=float)
    s = g["s_rel_stop_ft"].to_numpy(dtype=float)

    n = len(g)
    dt = robust_dt(t, default=TIMEGRID_DT_SEC)

    in_corridor = (s >= CORRIDOR_MIN_FT - 1e-9) & (s <= CORRIDOR_MAX_FT + 1e-9)
    standstill_confirm = np.zeros(n, dtype=bool)
    creep_candidate = np.zeros(n, dtype=bool)
    baseline_speed_1s_ago = np.full(n, np.nan)
    speed_drop_1s = np.full(n, np.nan)

    for i in range(n):
        if not in_corridor[i]:
            continue

        t0 = float(t[i])

        # Standstill: all samples within persistence window remain under stop threshold.
        t_stop_end = t0 + STOP_PERSIST_SEC
        j_stop = int(np.searchsorted(t, t_stop_end, side="right") - 1)
        if j_stop >= i and t[j_stop] >= t_stop_end - 0.5 * dt:
            if np.all(in_corridor[i:j_stop + 1]) and np.nanmax(v[i:j_stop + 1]) <= V_STOP_FPS + 1e-9:
                standstill_confirm[i] = True

        # Creep candidate: speed drop relative to about 1 s earlier, then sustained reduced speed.
        prev_idx = nearest_index_at_time(t, t0 - CREEP_LOOKBACK_SEC)
        if prev_idx is None or not in_corridor[prev_idx]:
            continue

        baseline_speed = float(v[prev_idx])
        baseline_speed_1s_ago[i] = baseline_speed
        speed_drop = baseline_speed - float(v[i])
        speed_drop_1s[i] = speed_drop

        if speed_drop < CREEP_DROP_FPS:
            continue

        t_creep_end = t0 + CREEP_PERSIST_SEC
        j_creep = int(np.searchsorted(t, t_creep_end, side="right") - 1)
        if j_creep >= i and t[j_creep] >= t_creep_end - 0.5 * dt:
            if np.all(in_corridor[i:j_creep + 1]):
                future_v = v[i:j_creep + 1]
                reduced_state_ok = (
                    np.nanmin(future_v) <= baseline_speed - CREEP_DROP_FPS + 1.0
                    and np.nanmedian(future_v) <= baseline_speed - 4.0
                )
                if reduced_state_ok:
                    creep_candidate[i] = True

    g["in_corridor_flag"] = in_corridor
    g["standstill_row_flag"] = standstill_confirm
    g["creep_candidate_row_flag"] = creep_candidate
    g["baseline_speed_1s_ago"] = baseline_speed_1s_ago
    g["speed_drop_1s_fps"] = speed_drop_1s
    return g


def vehicle_has_similar_constrained_condition(vehicle_debug: pd.DataFrame, target_time: float) -> bool:
    t = vehicle_debug["Total_Sim_Time_Sec"].to_numpy(dtype=float)
    v = vehicle_debug["Speed_fps"].to_numpy(dtype=float)
    in_corridor = vehicle_debug["in_corridor_flag"].to_numpy(dtype=bool)

    idx = np.where(np.abs(t - target_time) <= NEIGHBOR_TIME_TOL_SEC)[0]
    if len(idx) == 0:
        return False

    if np.any(vehicle_debug["standstill_row_flag"].to_numpy(dtype=bool)[idx]):
        return True

    for i in idx:
        if not in_corridor[i]:
            continue
        v_now = float(v[i])
        if v_now <= NEIGHBOR_LOW_SPEED_FPS:
            return True
        prev_idx = nearest_index_at_time(t, float(t[i]) - 1.0)
        if prev_idx is None or not in_corridor[prev_idx]:
            continue
        if float(v[prev_idx]) - v_now >= NEIGHBOR_DROP_FPS:
            return True
    return False


def get_neighbor_rows_at_time(frame: pd.DataFrame, veh_uid: str, s_current: float):
    others = frame[frame["veh_uid"] != veh_uid].copy()
    if others.empty:
        return None, None

    ahead = others[others["s_rel_stop_ft"] > s_current].copy()
    behind = others[others["s_rel_stop_ft"] < s_current].copy()

    ahead_row = None
    behind_row = None

    if not ahead.empty:
        ahead["gap_ft"] = ahead["s_rel_stop_ft"] - s_current
        ahead = ahead[ahead["gap_ft"] <= NEIGHBOR_MAX_GAP_FT + 1e-9]
        if not ahead.empty:
            ahead_row = ahead.sort_values("gap_ft", ascending=True).iloc[0]

    if not behind.empty:
        behind["gap_ft"] = s_current - behind["s_rel_stop_ft"]
        behind = behind[behind["gap_ft"] <= NEIGHBOR_MAX_GAP_FT + 1e-9]
        if not behind.empty:
            behind_row = behind.sort_values("gap_ft", ascending=True).iloc[0]

    return ahead_row, behind_row


def identify_events_for_run(run_df: pd.DataFrame, run_id: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    debug_store: dict[str, pd.DataFrame] = {}
    for veh_uid, group in run_df.groupby("veh_uid", sort=False):
        debug_store[str(veh_uid)] = build_vehicle_debug_flags(group)

    time_groups = {float(tv): group.copy() for tv, group in run_df.groupby("Total_Sim_Time_Sec", sort=True)}
    time_keys = np.array(sorted(time_groups.keys()), dtype=float)

    records = []

    for veh_uid, g_debug in debug_store.items():
        g_debug = g_debug.sort_values("Total_Sim_Time_Sec").copy().reset_index(drop=True)

        t = g_debug["Total_Sim_Time_Sec"].to_numpy(dtype=float)
        v = g_debug["Speed_fps"].to_numpy(dtype=float)
        s = g_debug["s_rel_stop_ft"].to_numpy(dtype=float)
        in_corridor = g_debug["in_corridor_flag"].to_numpy(dtype=bool)
        standstill_row = g_debug["standstill_row_flag"].to_numpy(dtype=bool)
        creep_candidate = g_debug["creep_candidate_row_flag"].to_numpy(dtype=bool)

        creep_supported_row = np.zeros(len(g_debug), dtype=bool)
        support_side = np.array([None] * len(g_debug), dtype=object)

        if USE_NEIGHBOR_SUPPORT_FOR_CREEP:
            for i in np.where(creep_candidate)[0]:
                ti = float(t[i])
                frame_time = nearest_time_value(time_keys, ti)
                if frame_time is None:
                    continue
                frame = time_groups[frame_time]
                row_now = frame[frame["veh_uid"] == veh_uid]
                if row_now.empty:
                    continue
                s_now = float(row_now["s_rel_stop_ft"].iloc[0])
                ahead_row, behind_row = get_neighbor_rows_at_time(frame, veh_uid, s_now)

                ahead_ok = False
                behind_ok = False
                if ahead_row is not None:
                    ahead_uid = str(ahead_row["veh_uid"])
                    if ahead_uid in debug_store:
                        ahead_ok = vehicle_has_similar_constrained_condition(debug_store[ahead_uid], ti)
                if behind_row is not None:
                    behind_uid = str(behind_row["veh_uid"])
                    if behind_uid in debug_store:
                        behind_ok = vehicle_has_similar_constrained_condition(debug_store[behind_uid], ti)

                if ahead_ok or behind_ok:
                    creep_supported_row[i] = True
                    if ahead_ok and behind_ok:
                        support_side[i] = "ahead+behind"
                    elif ahead_ok:
                        support_side[i] = "ahead"
                    else:
                        support_side[i] = "behind"
        else:
            creep_supported_row = creep_candidate.copy()

        g_debug["creep_supported_row_flag"] = creep_supported_row
        g_debug["creep_support_side"] = support_side
        debug_store[veh_uid] = g_debug

        dt = robust_dt(t, default=TIMEGRID_DT_SEC)
        first_stand_idx = first_persistent_flag_index(standstill_row, dt)
        first_creep_idx = first_persistent_flag_index(creep_supported_row, dt)

        ever_standstill_accepted = first_stand_idx is not None
        ever_creep_accepted = first_creep_idx is not None
        has_both = bool(ever_standstill_accepted and ever_creep_accepted)

        first_confirm_idx = None
        first_rule = None
        if first_stand_idx is not None and first_creep_idx is not None:
            if t[first_creep_idx] <= t[first_stand_idx]:
                first_confirm_idx = first_creep_idx
                first_rule = "creep"
            else:
                first_confirm_idx = first_stand_idx
                first_rule = "standstill"
        elif first_creep_idx is not None:
            first_confirm_idx = first_creep_idx
            first_rule = "creep"
        elif first_stand_idx is not None:
            first_confirm_idx = first_stand_idx
            first_rule = "standstill"

        joined_queue = first_confirm_idx is not None
        if joined_queue:
            onset_idx = backtrack_to_slowdown_onset(t, v, in_corridor, first_confirm_idx)
            t_join = float(t[onset_idx]) if onset_idx is not None else float(t[first_confirm_idx])
            s_join = float(s[onset_idx]) if onset_idx is not None else float(s[first_confirm_idx])
            v_join = float(v[onset_idx]) if onset_idx is not None else float(v[first_confirm_idx])
            t_confirm = float(t[first_confirm_idx])
            q_join = abs(s_join) if np.isfinite(s_join) and s_join <= 0 else np.nan
        else:
            t_join = np.nan
            s_join = np.nan
            v_join = np.nan
            t_confirm = np.nan
            q_join = 0.0

        t_creep_start = float(t[first_creep_idx]) if first_creep_idx is not None else np.nan
        s_creep_start = float(s[first_creep_idx]) if first_creep_idx is not None else np.nan
        t_stand_start = float(t[first_stand_idx]) if first_stand_idx is not None else np.nan
        s_stand_start = float(s[first_stand_idx]) if first_stand_idx is not None else np.nan
        t_exit = first_exit_time(g_debug)

        category = "nonqueued"
        if joined_queue:
            if has_both:
                category = "both_creep_and_standstill"
            elif first_rule == "creep":
                category = "creep_only"
            elif first_rule == "standstill":
                category = "standstill_only"

        t_event = t_join if joined_queue else t_exit
        q_event = q_join if joined_queue else 0.0

        records.append(
            {
                "run_id": int(run_id),
                "veh_uid": str(veh_uid),
                "joined_queue": int(joined_queue),
                "validation_category": category,
                "official_join_rule": first_rule,
                "ever_creep_accepted": int(ever_creep_accepted),
                "ever_standstill_accepted": int(ever_standstill_accepted),
                "has_both_standstill_and_creep": int(has_both),
                "t_queue_join_sec": t_join,
                "s_queue_join_ft": s_join,
                "queue_length_at_join_ft": q_join,
                "speed_at_join_fps": v_join,
                "t_creep_start_sec": t_creep_start,
                "s_creep_start_ft": s_creep_start,
                "t_standstill_start_sec": t_stand_start,
                "s_standstill_start_ft": s_stand_start,
                "t_join_confirm_sec": t_confirm,
                "t_exit_sec": t_exit,
                "t_event_sec": t_event,
                "queue_length_for_event_ft": q_event,
                "t_first_obs_sec": float(t[0]) if len(t) else np.nan,
                "t_last_obs_sec": float(t[-1]) if len(t) else np.nan,
                "min_s_rel_stop_ft": float(np.nanmin(s)) if len(s) else np.nan,
                "max_s_rel_stop_ft": float(np.nanmax(s)) if len(s) else np.nan,
            }
        )

    return pd.DataFrame(records), debug_store


# =============================================================================
# GT queue-length and cumulative-event files
# =============================================================================

def phase_state_at_time(intervals: pd.DataFrame, t: float) -> str:
    if intervals is None or intervals.empty:
        return "unknown"
    rows = intervals[(intervals["start"] <= float(t)) & (float(t) < intervals["end"])]
    if rows.empty:
        return "unknown"
    return str(rows.iloc[0]["state"]).lower().strip()


def robust_median(values: list[float] | np.ndarray, fallback: float = np.nan) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float(fallback)
    return float(np.nanmedian(arr))


def positive_successive_spacings(queued: pd.DataFrame, rule_filter: Optional[str] = None) -> list[float]:
    """
    Estimate physical spacing samples from successive observed queue-join distances.

    The sample is the positive increase in queue length between consecutive accepted
    queue-join events. This keeps the estimate trajectory-based and avoids a fixed
    25-ft assumption. Very small and very large jumps are filtered out.
    """
    q = queued.copy()
    if rule_filter is not None:
        if rule_filter == "standstill":
            q = q[q["official_join_rule"].astype(str).str.lower() == "standstill"].copy()
        elif rule_filter == "creep":
            q = q[q["official_join_rule"].astype(str).str.lower() == "creep"].copy()
        elif rule_filter == "departure":
            # Departure spacing should represent queued vehicles that actually pass the stopbar.
            q = q[q["t_exit_sec"].notna()].copy()

    if q.empty or len(q) < 2:
        return []

    q = q.sort_values(["t_queue_join_sec", "veh_uid"]).copy()
    vals = q["queue_length_at_join_ft"].to_numpy(dtype=float)
    diffs = np.diff(vals)
    diffs = diffs[(diffs >= SPACING_MIN_FT) & (diffs <= SPACING_MAX_FT)]
    return [float(x) for x in diffs]


def compute_empirical_spacing_values(events: pd.DataFrame, run_id: int) -> dict:
    """
    Compute four empirical spacing values from GT events.

    These are used only for GT curve construction/diagnostics in this script.
    Later baseline calibration can recompute state-specific l_eff using the
    cumulative-count samples and train/test split.
    """
    queued = events[
        (events["joined_queue"] == 1)
        & events["t_queue_join_sec"].notna()
        & events["queue_length_at_join_ft"].notna()
    ].copy()

    all_samples = positive_successive_spacings(queued)
    stand_samples = positive_successive_spacings(queued, "standstill")
    creep_samples = positive_successive_spacings(queued, "creep")
    dep_samples = positive_successive_spacings(queued, "departure")

    l_fixed = robust_median(all_samples, fallback=SPACING_FALLBACK_FT)
    l_stand = robust_median(stand_samples, fallback=l_fixed)
    l_creep = robust_median(creep_samples, fallback=l_fixed)
    l_dep = robust_median(dep_samples, fallback=l_fixed)

    return {
        "run_id": int(run_id),
        "l_fixed_all_ft": float(l_fixed),
        "l_standstill_ft": float(l_stand),
        "l_creep_form_ft": float(l_creep),
        "l_departure_ft": float(l_dep),
        "n_fixed_samples": int(len(all_samples)),
        "n_standstill_samples": int(len(stand_samples)),
        "n_creep_form_samples": int(len(creep_samples)),
        "n_departure_samples": int(len(dep_samples)),
        "spacing_fallback_ft": float(SPACING_FALLBACK_FT),
    }


def map_event_times_to_grid_indices(times: np.ndarray, event_times: np.ndarray) -> np.ndarray:
    """Map event times to the nearest time-grid index."""
    if len(event_times) == 0:
        return np.array([], dtype=int)
    idx = np.searchsorted(times, event_times, side="left")
    idx[idx >= len(times)] = len(times) - 1
    mask = (idx > 0) & (np.abs(times[idx] - event_times) > np.abs(times[idx - 1] - event_times))
    idx[mask] = idx[mask] - 1
    return idx.astype(int)


def build_gt_queue_length_timegrid(
    events: pd.DataFrame,
    run_df: pd.DataFrame,
    run_id: int,
    phase_intervals: Optional[pd.DataFrame] = None,
    spacing_values: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Dynamic GT queue-length curve using the same event-balance values that are plotted.

    This function intentionally does NOT use an active-tail max rule such as
    max(|s_join|). The direct stopbar-relative join distance remains in the
    vehicle-level GT file as queue_length_at_join_ft. The continuous curve is
    generated from the queue formation and dissipation process:

        Q(t) = Q(t-dt) + L_form * join_count(t) - L_departure * departure_count(t)

    Phase rule:
        - During green: growth and discharge are both allowed.
        - During amber/red: growth is allowed, but discharge is not subtracted.
        - If phase is unknown/missing: discharge is allowed so the script still works.

    The saved q_gt_ft is exactly the value used for plots and for later comparison.
    """
    t_min = float(np.floor(run_df["Total_Sim_Time_Sec"].min() / TIMEGRID_DT_SEC) * TIMEGRID_DT_SEC)
    t_max = float(np.ceil(run_df["Total_Sim_Time_Sec"].max() / TIMEGRID_DT_SEC) * TIMEGRID_DT_SEC)
    times = np.round(np.arange(t_min, t_max + 0.5 * TIMEGRID_DT_SEC, TIMEGRID_DT_SEC), 6)

    queued = events[
        (events["joined_queue"] == 1)
        & events["t_queue_join_sec"].notna()
        & events["t_exit_sec"].notna()
    ].copy()

    l_form = SPACING_FALLBACK_FT
    l_departure = SPACING_FALLBACK_FT
    if spacing_values is not None:
        # Formation uses the overall queued spacing for now. Departure uses the
        # departure-specific spacing. Both are empirical and saved in the grid.
        if np.isfinite(spacing_values.get("l_fixed_all_ft", np.nan)):
            l_form = float(spacing_values["l_fixed_all_ft"])
        if np.isfinite(spacing_values.get("l_departure_ft", np.nan)):
            l_departure = float(spacing_values["l_departure_ft"])

    join_count = np.zeros(len(times), dtype=int)
    depart_count_raw = np.zeros(len(times), dtype=int)
    depart_count_applied = np.zeros(len(times), dtype=int)
    join_uid_at_idx = [[] for _ in range(len(times))]
    join_rule_at_idx = [[] for _ in range(len(times))]

    if not queued.empty:
        join_times = queued["t_queue_join_sec"].to_numpy(dtype=float)
        exit_times = queued["t_exit_sec"].to_numpy(dtype=float)
        uids = queued["veh_uid"].astype(str).to_numpy()
        rules = queued["official_join_rule"].astype(str).to_numpy()

        join_idx = map_event_times_to_grid_indices(times, join_times)
        dep_idx = map_event_times_to_grid_indices(times, exit_times)

        for k, idx in enumerate(join_idx):
            join_count[int(idx)] += 1
            join_uid_at_idx[int(idx)].append(str(uids[k]))
            join_rule_at_idx[int(idx)].append(str(rules[k]))
        for idx in dep_idx:
            depart_count_raw[int(idx)] += 1

    if not queued.empty:
        jt = queued["t_queue_join_sec"].to_numpy(dtype=float)
        et = queued["t_exit_sec"].to_numpy(dtype=float)
    else:
        jt = np.array([], dtype=float)
        et = np.array([], dtype=float)

    q_vals = np.zeros(len(times), dtype=float)
    phase_states: list[str] = []
    queue_regime: list[str] = []
    active_counts: list[int] = []
    back_uid: list[Optional[str]] = []
    back_rule: list[Optional[str]] = []
    back_event_time: list[float] = []

    current_back_uid: Optional[str] = None
    current_back_rule: Optional[str] = None
    current_back_event = np.nan

    for i, tt in enumerate(times):
        prev_q = q_vals[i - 1] if i > 0 else 0.0

        state = phase_state_at_time(phase_intervals, float(tt)) if phase_intervals is not None else "unknown"
        state_norm = "amber" if state == "yellow" else state
        if state_norm not in {"green", "amber", "red"}:
            state_norm = "unknown"

        growth_ft = l_form * float(join_count[i])
        raw_departures = int(depart_count_raw[i])

        can_discharge = True
        if DISCHARGE_ONLY_DURING_GREEN and state_norm in {"red", "amber"}:
            can_discharge = False

        applied_departures = raw_departures if can_discharge else 0
        depart_count_applied[i] = applied_departures
        discharge_ft = l_departure * float(applied_departures)

        q_now = prev_q + growth_ft - discharge_ft
        if q_now < 0.0:
            q_now = 0.0

        active_now = int(np.sum((jt <= tt + 1e-9) & (tt < et - 1e-9))) if len(jt) else 0

        # If no queued vehicles are active and no new join occurred, the physical queue is zero.
        # This prevents residual carryover caused by empirical spacing mismatch.
        if active_now == 0 and join_count[i] == 0:
            q_now = 0.0

        # Diagnostics for the latest joining vehicle(s), not used to drive Q(t).
        if join_count[i] > 0:
            current_back_uid = join_uid_at_idx[i][-1] if join_uid_at_idx[i] else None
            current_back_rule = join_rule_at_idx[i][-1] if join_rule_at_idx[i] else None
            current_back_event = float(tt)

        if q_now <= 1e-9:
            q_now = 0.0
            current_back_uid = None
            current_back_rule = None
            current_back_event = np.nan

        if growth_ft > 0 and discharge_ft > 0:
            regime = "growth_and_discharge"
        elif growth_ft > 0:
            regime = "growth"
        elif discharge_ft > 0:
            regime = "discharge"
        elif q_now > 0:
            regime = "holding"
        else:
            regime = "no_queue"

        q_vals[i] = q_now
        phase_states.append(state_norm)
        queue_regime.append(regime)
        active_counts.append(active_now)
        back_uid.append(current_back_uid)
        back_rule.append(current_back_rule)
        back_event_time.append(current_back_event)

    return pd.DataFrame({
        "run_id": int(run_id),
        "time_sec": times.astype(float),
        "q_gt_ft": q_vals.astype(float),
        "q_gt_raw_ft": q_vals.astype(float),
        "join_count": join_count.astype(int),
        "departure_count_raw": depart_count_raw.astype(int),
        "departure_count_applied": depart_count_applied.astype(int),
        "active_queued_vehicle_count": np.asarray(active_counts, dtype=int),
        "phase_state": phase_states,
        "queue_regime": queue_regime,
        "l_form_used_ft": float(l_form),
        "l_departure_used_ft": float(l_departure),
        "back_of_queue_veh_uid": back_uid,
        "back_of_queue_rule": back_rule,
        "back_of_queue_event_time_sec": back_event_time,
    })

def build_gt_cumulative_events(events: pd.DataFrame, run_id: int) -> pd.DataFrame:
    cum = events[
        ["run_id", "veh_uid", "t_event_sec", "joined_queue", "queue_length_for_event_ft"]
    ].copy()
    cum = cum.dropna(subset=["t_event_sec"]).sort_values(["t_event_sec", "veh_uid"]).reset_index(drop=True)
    cum["N_gt"] = np.arange(1, len(cum) + 1, dtype=int)
    return cum[["run_id", "veh_uid", "t_event_sec", "N_gt", "joined_queue", "queue_length_for_event_ft"]]


def summarize_events(events: pd.DataFrame, run_id: int) -> dict:
    counts = events["validation_category"].value_counts(dropna=False).to_dict()
    return {
        "run_id": int(run_id),
        "total_vehicles": int(len(events)),
        "joined_queue": int(events["joined_queue"].sum()),
        "nonqueued": int(counts.get("nonqueued", 0)),
        "standstill_only": int(counts.get("standstill_only", 0)),
        "creep_only": int(counts.get("creep_only", 0)),
        "both_creep_and_standstill": int(counts.get("both_creep_and_standstill", 0)),
    }


# =============================================================================
# Loading
# =============================================================================

def load_filtered_trajectory_for_run(run_id: int) -> pd.DataFrame:
    """
    Load filtered trajectory rows for one simulation run only.

    This avoids loading the full 26M-row traj_nb_filtered.csv into memory.
    """
    if not INPUT_TRAJ_CSV.exists():
        raise FileNotFoundError(
            f"Could not find INPUT_TRAJ_CSV:\n{INPUT_TRAJ_CSV}\n"
            "Check INPUT_TRAJ_CSV in this script or the preprocessing path in config.py."
        )

    header = pd.read_csv(INPUT_TRAJ_CSV, nrows=0)
    cols = set(header.columns)

    required = {"run_id", "veh_uid", "Total_Sim_Time_Sec", "Speed_fps"}
    require_columns(header, required, "filtered trajectory file")

    if "s_rel_stop_ft" in cols:
        usecols = ["run_id", "veh_uid", "Total_Sim_Time_Sec", "Speed_fps", "s_rel_stop_ft"]
    elif "s_rel_stop_m" in cols:
        usecols = ["run_id", "veh_uid", "Total_Sim_Time_Sec", "Speed_fps", "s_rel_stop_m"]
    else:
        raise ValueError("filtered trajectory file must contain s_rel_stop_ft or s_rel_stop_m")

    chunks = []
    chunksize = 1_000_000

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            INPUT_TRAJ_CSV,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
        ),
        start=1,
    ):
        chunk["run_id"] = pd.to_numeric(chunk["run_id"], errors="coerce")
        chunk = chunk[chunk["run_id"] == int(run_id)]

        if chunk.empty:
            continue

        chunk["veh_uid"] = chunk["veh_uid"].astype(str).str.strip()
        chunk["Total_Sim_Time_Sec"] = pd.to_numeric(chunk["Total_Sim_Time_Sec"], errors="coerce")
        chunk["Speed_fps"] = pd.to_numeric(chunk["Speed_fps"], errors="coerce")

        if "s_rel_stop_ft" not in chunk.columns:
            chunk["s_rel_stop_ft"] = pd.to_numeric(chunk["s_rel_stop_m"], errors="coerce") * M_TO_FT
            chunk = chunk.drop(columns=["s_rel_stop_m"])
        else:
            chunk["s_rel_stop_ft"] = pd.to_numeric(chunk["s_rel_stop_ft"], errors="coerce")

        chunk = chunk.dropna(
            subset=[
                "run_id",
                "veh_uid",
                "Total_Sim_Time_Sec",
                "Speed_fps",
                "s_rel_stop_ft",
            ]
        )

        if not chunk.empty:
            chunk["run_id"] = chunk["run_id"].astype(int)
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(
            columns=[
                "run_id",
                "veh_uid",
                "Total_Sim_Time_Sec",
                "Speed_fps",
                "s_rel_stop_ft",
            ]
        )

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values(["veh_uid", "Total_Sim_Time_Sec"]).reset_index(drop=True)

    return df

def resolve_phase_file() -> Optional[Path]:
    for path in PHASE_FILE_CANDIDATES:
        if path.exists():
            return path
    return None


def load_phase_data() -> Optional[pd.DataFrame]:
    phase_path = resolve_phase_file()
    if phase_path is None:
        print("[WARN] No phase file found. Tried:")
        for p in PHASE_FILE_CANDIDATES:
            print(f"       {p}")
        print("       Phase ribbons and cycle plots will use fallback behavior.")
        return None

    print(f"Input phase      : {phase_path}")
    phase = pd.read_csv(phase_path)
    needed = {"run_id", "simsec", "signal_group", "state"}
    if not needed.issubset(set(phase.columns)):
        print(f"[WARN] Phase file missing columns {sorted(needed - set(phase.columns))}. Phase ribbons skipped.")
        return None

    phase = phase.copy()
    phase["run_id"] = pd.to_numeric(phase["run_id"], errors="coerce")
    phase["simsec"] = pd.to_numeric(phase["simsec"], errors="coerce")
    phase["signal_group"] = pd.to_numeric(phase["signal_group"], errors="coerce")
    if "controller" in phase.columns:
        phase["controller"] = pd.to_numeric(phase["controller"], errors="coerce")
    else:
        phase["controller"] = np.nan
    phase["state"] = phase["state"].astype(str).str.lower().str.strip()
    phase["state"] = phase["state"].replace({"yellow": "amber"})

    phase = phase.dropna(subset=["run_id", "simsec", "signal_group", "state"]).copy()
    phase["run_id"] = phase["run_id"].astype(int)
    phase["signal_group"] = phase["signal_group"].astype(int)
    phase = phase[phase["run_id"].isin([int(r) for r in RUN_IDS])].copy()
    if "controller" in phase.columns and phase["controller"].notna().any():
        phase = phase[(phase["controller"].isna()) | (phase["controller"].astype("Int64") == int(PLOT_CONTROLLER))].copy()
    return phase.sort_values(["run_id", "signal_group", "simsec"]).reset_index(drop=True)



# =============================================================================
# Phase ribbon and plotting
# =============================================================================

def phase_intervals_for_run(phase: Optional[pd.DataFrame], run_id: int) -> pd.DataFrame:
    if phase is None or phase.empty:
        return pd.DataFrame(columns=["start", "end", "state"])

    p = phase[(phase["run_id"] == int(run_id)) & (phase["signal_group"] == int(PLOT_SIGNAL_GROUP))].copy()
    if p.empty:
        return pd.DataFrame(columns=["start", "end", "state"])

    p = p.sort_values("simsec").drop_duplicates(subset=["simsec"], keep="last").reset_index(drop=True)
    p["start"] = p["simsec"].astype(float)
    p["end"] = p["start"].shift(-1)

    if len(p) > 1:
        fallback_step = float(np.nanmedian(np.diff(p["start"].to_numpy(dtype=float))))
        if not np.isfinite(fallback_step) or fallback_step <= 0:
            fallback_step = 1.0
    else:
        fallback_step = 1.0
    p["end"] = p["end"].fillna(p["start"] + fallback_step)
    p = p[p["end"] > p["start"]].copy()
    return p[["start", "end", "state"]].reset_index(drop=True)


def add_phase_ribbon(ax, intervals: pd.DataFrame, t0: float, t1: float) -> None:
    if intervals.empty:
        return
    color_map = {"green": "#7fc97f", "amber": "#fdc086", "yellow": "#fdc086", "red": "#f0027f"}
    y0, y1 = -0.12, -0.045
    trans = ax.get_xaxis_transform()
    view = intervals[(intervals["end"] >= t0) & (intervals["start"] <= t1)].copy()
    for _, row in view.iterrows():
        start = max(float(row["start"]), t0)
        end = min(float(row["end"]), t1)
        state = str(row["state"]).lower()
        color = color_map.get(state, "0.8")
        ax.axvspan(start, end, ymin=y0, ymax=y1, color=color, alpha=0.8, transform=trans, clip_on=False)


def green_start_times(intervals: pd.DataFrame, t_min: float, t_max: float) -> np.ndarray:
    if intervals.empty:
        return np.array([], dtype=float)
    states = intervals["state"].astype(str).str.lower().to_numpy()
    starts = intervals["start"].to_numpy(dtype=float)
    out = []
    prev = None
    for i, st in enumerate(states):
        if st == "green" and prev != "green":
            out.append(float(starts[i]))
        prev = st
    arr = np.asarray(out, dtype=float)
    return np.sort(arr[(arr >= t_min - 1e-9) & (arr <= t_max + 1e-9)])


def get_cycle_windows(qgrid: pd.DataFrame, intervals: pd.DataFrame) -> list[tuple[float, float]]:
    t_min = float(qgrid["time_sec"].min())
    t_max = float(qgrid["time_sec"].max())
    if not intervals.empty:
        starts = green_start_times(intervals, t_min, t_max)
        if len(starts) >= 2:
            return [(float(starts[i]), float(starts[i + 1])) for i in range(min(len(starts) - 1, MAX_CYCLES_TO_PLOT))]

    # Fallback fixed windows only if phase file is missing or unusable.
    windows = []
    start = t_min
    while start < t_max and len(windows) < MAX_CYCLES_TO_PLOT:
        end = min(start + FALLBACK_CYCLE_LENGTH_SEC, t_max)
        windows.append((start, end))
        start = end
    return windows


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("Revised GT queue-join, queue-length, and cumulative-event extraction")
    print("=" * 90)
    print(f"Project root     : {PROJECT_ROOT}")
    print(f"Input trajectory : {INPUT_TRAJ_CSV}")
    print("Phase candidates :")
    for _p in PHASE_FILE_CANDIDATES:
        print(f"  - {_p}")
    print(f"Output directory : {OUT_DIR}")
    print(f"Compute RUN_IDS  : {RUN_IDS}")
    print(f"Detection corridor: {CORRIDOR_MIN_FT:.0f} to {CORRIDOR_MAX_FT:.0f} ft relative to stopbar")
    print("=" * 90)

    phase = load_phase_data() if MAKE_PLOTS else None

    all_events = []
    all_qgrids = []
    all_cum = []
    all_spacing = []
    summary_rows = []

    for run_id in RUN_IDS:
        run_id = int(run_id)

        print(f"\n[Run {run_id:03d}] loading trajectory rows...")
        run_df = load_filtered_trajectory_for_run(run_id)

        if run_df.empty:
            print(f"[WARN] Run {run_id:03d}: no rows found; skipping.")
            continue

        print(f"\n[Run {run_id:03d}] vehicles={run_df['veh_uid'].nunique():,}, rows={len(run_df):,}")

        events, _debug_store = identify_events_for_run(run_df, run_id)
        events_path = OUT_DIR / f"gt_queue_join_events_run{run_id:03d}.csv"
        events.to_csv(events_path, index=False)
        print(f"[Saved] {events_path}")

        intervals = phase_intervals_for_run(phase, run_id)

        spacing = compute_empirical_spacing_values(events, run_id)
        spacing_path = OUT_DIR / f"gt_empirical_spacing_run{run_id:03d}.csv"
        pd.DataFrame([spacing]).to_csv(spacing_path, index=False)
        all_spacing.append(spacing)
        print(f"[Saved] {spacing_path}")
        print(pd.Series(spacing).to_string())

        qgrid = build_gt_queue_length_timegrid(events, run_df, run_id, intervals, spacing)
        qgrid_path = OUT_DIR / f"gt_queue_length_timegrid_run{run_id:03d}.csv"
        qgrid.to_csv(qgrid_path, index=False)
        print(f"[Saved] {qgrid_path}")

        cum = build_gt_cumulative_events(events, run_id)
        cum_path = OUT_DIR / f"gt_cumulative_events_run{run_id:03d}.csv"
        cum.to_csv(cum_path, index=False)
        print(f"[Saved] {cum_path}")

        summary = summarize_events(events, run_id)
        summary_rows.append(summary)
        print(pd.Series(summary).to_string())

        all_events.append(events)
        all_qgrids.append(qgrid)
        all_cum.append(cum)


    if all_events:
        all_events_df = pd.concat(all_events, ignore_index=True)
        all_events_path = OUT_DIR / "gt_queue_join_events_allruns.csv"
        all_events_df.to_csv(all_events_path, index=False)
        print(f"\n[Saved] {all_events_path}")

    if all_qgrids:
        all_qgrids_df = pd.concat(all_qgrids, ignore_index=True)
        all_qgrids_path = OUT_DIR / "gt_queue_length_timegrid_allruns.csv"
        all_qgrids_df.to_csv(all_qgrids_path, index=False)
        print(f"[Saved] {all_qgrids_path}")

    if all_cum:
        all_cum_df = pd.concat(all_cum, ignore_index=True)
        all_cum_path = OUT_DIR / "gt_cumulative_events_allruns.csv"
        all_cum_df.to_csv(all_cum_path, index=False)
        print(f"[Saved] {all_cum_path}")


    if all_spacing:
        spacing_df = pd.DataFrame(all_spacing)
        spacing_all_path = OUT_DIR / "gt_empirical_spacing_by_run.csv"
        spacing_df.to_csv(spacing_all_path, index=False)
        print(f"[Saved] {spacing_all_path}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = OUT_DIR / "gt_queue_join_summary_by_run.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[Saved] {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
