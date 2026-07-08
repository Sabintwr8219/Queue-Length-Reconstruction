"""
Generate decision-oriented cycle-level labels for queue reconstruction.

Purpose
-------
This script creates GT operational-decision labels from the simulation time-grid:

1) residual_queue_ft
   Queue length at the end of green.

2) cycle_failure
   1 if residual_queue_ft >= FAILURE_THRESHOLD_FT, else 0.

It also creates simple traffic-condition groups for later robustness analysis:
    low / moderate / congested

Input
-----
output/intermediate_csv/cv_features/timegrid_features_allruns_allrates.csv

Output
------
output/intermediate_csv/decision_labels/
    decision_labels_cycle_level.csv
    decision_label_summary_by_split.csv
    decision_label_summary_by_run.csv
    decision_label_summary_by_traffic_condition.csv
    decision_label_phase_state_counts.csv
    decision_label_skipped_cycles.csv
"""

from __future__ import annotations

from config import (
    PROJECT_ROOT,
    TRAIN_RUN_IDS,
    VALIDATION_RUN_IDS,
    TEST_RUN_IDS,
)

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

FEATURE_FILE = (
    PROJECT_ROOT
    / "output"
    / "intermediate_csv"
    / "cv_features"
    / "timegrid_features_allruns_allrates.csv"
)

OUT_DIR = PROJECT_ROOT / "output" / "intermediate_csv" / "decision_labels"

# Main decision threshold.
# Starting value: approximately one queued vehicle.
FAILURE_THRESHOLD_FT = 25.0

# We currently define residual queue at the end of green, not yellow.
# If later we decide to use "end of effective green", change this to True.
USE_EFFECTIVE_GREEN_END = False

# Simple fixed traffic-condition grouping.
# These are used only for stratified analysis, not model training.
MODERATE_PEAK_QUEUE_FT = 75.0
CONGESTED_PEAK_QUEUE_FT = 150.0

# Very short red-to-red intervals are ignored as invalid/incomplete cycles.
MIN_CYCLE_DURATION_SEC = 10.0


# =============================================================================
# Utility functions
# =============================================================================

def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def assign_ml_split(run_id: int) -> str:
    if int(run_id) in TRAIN_RUN_IDS:
        return "train"
    if int(run_id) in VALIDATION_RUN_IDS:
        return "validation"
    if int(run_id) in TEST_RUN_IDS:
        return "test"
    return "other"


def classify_phase_state(value) -> str:
    """
    Convert phase_state strings into broad classes.

    Expected common values are red/green/yellow, but this is intentionally
    tolerant so the script does not fail if labels contain small variations.
    """
    p = str(value).strip().lower()

    if p in {"", "nan", "none", "null"}:
        return "unknown"

    if p in {"red", "r"} or p.startswith("red"):
        return "red"

    if p in {"yellow", "y", "amber"} or p.startswith("yellow") or "amber" in p:
        return "yellow"

    if p in {"green", "g"} or p.startswith("green") or "green" in p:
        return "green"

    return "other"


def assign_traffic_condition(
    residual_queue_ft: float,
    gt_peak_queue_ft: float,
    cycle_failure: int,
) -> str:
    """
    Simple cycle-level traffic-condition group.

    Congested means either a residual queue exists after green or the cycle peak
    queue is large. This gives us a useful congested-cycle subset for later
    decision-oriented robustness checks.
    """
    if not np.isfinite(gt_peak_queue_ft):
        return "unknown"

    if int(cycle_failure) == 1 or gt_peak_queue_ft >= CONGESTED_PEAK_QUEUE_FT:
        return "congested"

    if gt_peak_queue_ft >= MODERATE_PEAK_QUEUE_FT:
        return "moderate"

    return "low"


# =============================================================================
# Loading
# =============================================================================

def load_gt_timegrid() -> pd.DataFrame:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find feature file:\n{FEATURE_FILE}\n"
            "Run build_cv_features.py first."
        )

    df = pd.read_csv(FEATURE_FILE)

    required = [
        "run_id",
        "cv_rate_pct",
        "time_sec",
        "phase_state",
        "q_gt_ft",
    ]
    require_columns(df, required, "feature table")

    for col in ["run_id", "cv_rate_pct", "time_sec", "q_gt_ft"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["run_id", "cv_rate_pct", "time_sec", "q_gt_ft"]).copy()
    df["run_id"] = df["run_id"].astype(int)
    df["cv_rate_pct"] = df["cv_rate_pct"].astype(int)

    model_runs = sorted(set(TRAIN_RUN_IDS + VALIDATION_RUN_IDS + TEST_RUN_IDS))
    df = df[df["run_id"].isin(model_runs)].copy()

    # The feature table has one copy per CV rate. GT labels should be one row
    # per run/time, so deduplicate by run/time and keep the first CV-rate row.
    df = (
        df.sort_values(["run_id", "time_sec", "cv_rate_pct"])
        .drop_duplicates(subset=["run_id", "time_sec"], keep="first")
        .sort_values(["run_id", "time_sec"])
        .reset_index(drop=True)
    )

    df["phase_state"] = df["phase_state"].astype(str).str.strip().str.lower()
    df["phase_class"] = df["phase_state"].apply(classify_phase_state)
    df["ml_split"] = df["run_id"].apply(assign_ml_split)

    return df


# =============================================================================
# Cycle inference and label generation
# =============================================================================

def infer_red_to_red_cycles(run_df: pd.DataFrame) -> list[tuple[int, int, int]]:
    """
    Infer red-to-red cycles using phase_state.

    Returns:
        list of tuples:
            (cycle_number, start_pos, next_red_start_pos)

    The cycle contains rows start_pos : next_red_start_pos.
    The next red-start row itself is treated as the boundary of the next cycle.
    """
    work = run_df.sort_values("time_sec").reset_index(drop=True)
    is_red = work["phase_class"].eq("red").to_numpy()

    red_start_positions = []
    prev_red = False

    for i, current_red in enumerate(is_red):
        if bool(current_red) and not bool(prev_red):
            red_start_positions.append(i)
        prev_red = bool(current_red)

    cycles = []
    for k in range(len(red_start_positions) - 1):
        start_pos = int(red_start_positions[k])
        next_red_pos = int(red_start_positions[k + 1])

        if next_red_pos <= start_pos:
            continue

        cycles.append((k + 1, start_pos, next_red_pos))

    return cycles


def choose_green_end_row(cycle_df: pd.DataFrame) -> tuple[pd.Series | None, str]:
    """
    Select the row used for residual queue.

    Default:
        last explicit green row.

    Fallback:
        last non-red row if no explicit green is available.
    """
    if USE_EFFECTIVE_GREEN_END:
        mask = cycle_df["phase_class"].isin(["green", "yellow"])
        source = "green_or_yellow_effective_green"
    else:
        mask = cycle_df["phase_class"].eq("green")
        source = "explicit_green"

    candidate = cycle_df[mask].copy()

    if candidate.empty:
        fallback_mask = ~cycle_df["phase_class"].eq("red")
        candidate = cycle_df[fallback_mask].copy()
        source = "nonred_fallback"

    if candidate.empty:
        return None, "missing_green_or_nonred"

    candidate = candidate.sort_values("time_sec")
    return candidate.iloc[-1], source


def generate_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    skipped = []

    for run_id, run_df in df.groupby("run_id", sort=True):
        run_df = run_df.sort_values("time_sec").reset_index(drop=True)
        cycles = infer_red_to_red_cycles(run_df)

        if not cycles:
            skipped.append(
                {
                    "run_id": int(run_id),
                    "cycle_number": np.nan,
                    "reason": "no_red_to_red_cycles_found",
                }
            )
            continue

        for cycle_number, start_pos, next_red_pos in cycles:
            cycle_df = run_df.iloc[start_pos:next_red_pos].copy()

            if cycle_df.empty:
                skipped.append(
                    {
                        "run_id": int(run_id),
                        "cycle_number": int(cycle_number),
                        "reason": "empty_cycle",
                    }
                )
                continue

            cycle_start_time = float(run_df.loc[start_pos, "time_sec"])
            cycle_end_time = float(run_df.loc[next_red_pos, "time_sec"])
            cycle_duration = cycle_end_time - cycle_start_time

            if not np.isfinite(cycle_duration) or cycle_duration < MIN_CYCLE_DURATION_SEC:
                skipped.append(
                    {
                        "run_id": int(run_id),
                        "cycle_number": int(cycle_number),
                        "reason": "short_or_invalid_cycle_duration",
                        "cycle_duration_sec": cycle_duration,
                    }
                )
                continue

            green_end_row, green_end_source = choose_green_end_row(cycle_df)

            if green_end_row is None:
                skipped.append(
                    {
                        "run_id": int(run_id),
                        "cycle_number": int(cycle_number),
                        "reason": green_end_source,
                        "cycle_start_time_sec": cycle_start_time,
                        "cycle_end_time_sec": cycle_end_time,
                    }
                )
                continue

            q_series = pd.to_numeric(cycle_df["q_gt_ft"], errors="coerce")
            valid_q = q_series[np.isfinite(q_series)]

            if valid_q.empty:
                skipped.append(
                    {
                        "run_id": int(run_id),
                        "cycle_number": int(cycle_number),
                        "reason": "no_valid_q_gt",
                    }
                )
                continue

            peak_idx = q_series.idxmax()
            gt_peak_queue_ft = float(q_series.loc[peak_idx])
            gt_peak_time_sec = float(cycle_df.loc[peak_idx, "time_sec"])

            residual_queue_ft = float(green_end_row["q_gt_ft"])
            cycle_failure = int(residual_queue_ft >= FAILURE_THRESHOLD_FT)

            traffic_condition = assign_traffic_condition(
                residual_queue_ft=residual_queue_ft,
                gt_peak_queue_ft=gt_peak_queue_ft,
                cycle_failure=cycle_failure,
            )

            green_rows = cycle_df[cycle_df["phase_class"].eq("green")]
            if green_rows.empty:
                green_start_time = np.nan
            else:
                green_start_time = float(green_rows["time_sec"].min())

            cycle_uid = f"run{int(run_id):03d}_cycle{int(cycle_number):04d}"

            rows.append(
                {
                    "cycle_uid": cycle_uid,
                    "run_id": int(run_id),
                    "ml_split": assign_ml_split(int(run_id)),
                    "cycle_number": int(cycle_number),

                    "cycle_start_time_sec": cycle_start_time,
                    "cycle_end_time_sec": cycle_end_time,
                    "cycle_duration_sec": float(cycle_duration),

                    "green_start_time_sec": green_start_time,
                    "green_end_time_sec": float(green_end_row["time_sec"]),
                    "green_end_source": green_end_source,
                    "green_end_phase_state": str(green_end_row["phase_state"]),
                    "green_end_to_cycle_end_sec": float(
                        cycle_end_time - float(green_end_row["time_sec"])
                    ),

                    "residual_queue_ft": residual_queue_ft,
                    "failure_threshold_ft": float(FAILURE_THRESHOLD_FT),
                    "cycle_failure": cycle_failure,

                    "residual_queue_veh_25ft": residual_queue_ft / 25.0,

                    "gt_peak_queue_ft": gt_peak_queue_ft,
                    "gt_peak_time_sec": gt_peak_time_sec,
                    "gt_mean_queue_ft": float(valid_q.mean()),
                    "gt_median_queue_ft": float(valid_q.median()),
                    "gt_queue_at_cycle_start_ft": float(cycle_df.iloc[0]["q_gt_ft"]),
                    "gt_queue_at_last_sample_before_next_red_ft": float(
                        cycle_df.iloc[-1]["q_gt_ft"]
                    ),

                    "traffic_condition": traffic_condition,

                    "n_timegrid_samples": int(len(cycle_df)),
                    "n_red_samples": int(cycle_df["phase_class"].eq("red").sum()),
                    "n_green_samples": int(cycle_df["phase_class"].eq("green").sum()),
                    "n_yellow_samples": int(cycle_df["phase_class"].eq("yellow").sum()),
                    "n_other_phase_samples": int(cycle_df["phase_class"].eq("other").sum()),
                }
            )

    labels = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)

    return labels, skipped_df


# =============================================================================
# Summaries
# =============================================================================

def summarize_labels(labels: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()

    out = (
        labels.groupby(group_cols, as_index=False)
        .agg(
            n_cycles=("cycle_uid", "count"),
            mean_residual_queue_ft=("residual_queue_ft", "mean"),
            median_residual_queue_ft=("residual_queue_ft", "median"),
            rmse_like_residual_queue_ft=("residual_queue_ft", lambda x: float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))),
            max_residual_queue_ft=("residual_queue_ft", "max"),
            cycle_failure_rate=("cycle_failure", "mean"),
            mean_gt_peak_queue_ft=("gt_peak_queue_ft", "mean"),
            median_gt_peak_queue_ft=("gt_peak_queue_ft", "median"),
            max_gt_peak_queue_ft=("gt_peak_queue_ft", "max"),
            mean_cycle_duration_sec=("cycle_duration_sec", "mean"),
        )
        .reset_index(drop=True)
    )

    out["cycle_failure_rate_pct"] = 100.0 * out["cycle_failure_rate"]
    out = out.drop(columns=["cycle_failure_rate"])

    return out


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("Generating decision-oriented labels")
    print("=" * 96)
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Feature file       : {FEATURE_FILE}")
    print(f"Output dir         : {OUT_DIR}")
    print(f"Failure threshold  : {FAILURE_THRESHOLD_FT:.1f} ft")
    print(f"Effective green?   : {USE_EFFECTIVE_GREEN_END}")
    print("=" * 96)

    df = load_gt_timegrid()

    print(f"Loaded unique GT time-grid rows: {len(df):,}")
    print("Runs:")
    print(
        df.groupby(["ml_split", "run_id"], sort=True)
        .size()
        .reset_index(name="timegrid_rows")
        .to_string(index=False)
    )

    phase_counts = (
        df.groupby(["phase_state", "phase_class"], as_index=False)
        .size()
        .rename(columns={"size": "n_rows"})
        .sort_values(["phase_class", "phase_state"])
    )
    phase_counts.to_csv(OUT_DIR / "decision_label_phase_state_counts.csv", index=False)

    print("\nPhase-state counts:")
    print(phase_counts.to_string(index=False))

    labels, skipped = generate_labels(df)

    if labels.empty:
        raise RuntimeError(
            "No decision labels were generated. Check phase_state values and cycle inference."
        )

    labels = labels.sort_values(["run_id", "cycle_number"]).reset_index(drop=True)

    labels_path = OUT_DIR / "decision_labels_cycle_level.csv"
    labels.to_csv(labels_path, index=False)

    skipped_path = OUT_DIR / "decision_label_skipped_cycles.csv"
    skipped.to_csv(skipped_path, index=False)

    summary_by_split = summarize_labels(labels, ["ml_split"])
    summary_by_run = summarize_labels(labels, ["ml_split", "run_id"])
    summary_by_condition = summarize_labels(labels, ["ml_split", "traffic_condition"])

    summary_by_split.to_csv(
        OUT_DIR / "decision_label_summary_by_split.csv",
        index=False,
    )
    summary_by_run.to_csv(
        OUT_DIR / "decision_label_summary_by_run.csv",
        index=False,
    )
    summary_by_condition.to_csv(
        OUT_DIR / "decision_label_summary_by_traffic_condition.csv",
        index=False,
    )

    print("\nSaved:")
    print(f"  {labels_path}")
    print(f"  {OUT_DIR / 'decision_label_summary_by_split.csv'}")
    print(f"  {OUT_DIR / 'decision_label_summary_by_run.csv'}")
    print(f"  {OUT_DIR / 'decision_label_summary_by_traffic_condition.csv'}")
    print(f"  {skipped_path}")

    print("\nDecision-label summary by split:")
    print(summary_by_split.round(3).to_string(index=False))

    print("\nDecision-label summary by traffic condition:")
    print(summary_by_condition.round(3).to_string(index=False))

    print("\nSkipped cycles:")
    if skipped.empty:
        print("  None")
    else:
        print(skipped.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()