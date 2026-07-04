"""
Generate cumulative-count theory files.

This script:
1. Extracts upstream-arrival and stopbar-departure event times.
2. Builds A(t), D(t), V(t), and baseline B(t) curves.
3. Saves a vehicle-level baseline B join file.
4. Saves a plot-ready cumulative-curve file for plot.py.

Outputs:
    data/processed_data/ad_times_all_runs.csv
    data/processed_data/vb_curves_all_runs.csv
    data/processed_data/baseline_b_join_all_runs.csv
    data/processed_data/cumulative_curves_plot_ready_allruns.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    RUN_IDS,
    TRAJ_NB_FILTERED_CSV,
    AD_TIMES_CSV,
    VB_CURVES_CSV,
    BASELINE_B_JOIN_CSV,
    CUMULATIVE_CURVES_PLOT_READY_CSV,
    Y_UPSTREAM_DETECTOR_M,
    Y_STOPBAR_M,
    DETECTOR_SPACING_FT,
    VF_FPS,
    VQ_FPS,
    VF_MPH,
    VQ_MPH,
    T_FF_SEC,
    BETA,
    ensure_project_directories,
)


# ============================================================
# User-adjustable local settings
# ============================================================

EPS_SEC = 0.1


# ============================================================
# Helpers
# ============================================================

def require_columns(df: pd.DataFrame, required_columns: set[str], label: str) -> None:
    """Raise an error if required columns are missing."""
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)}")


def first_cross_time(vehicle_df: pd.DataFrame, detector_position_m: float) -> float:
    """
    Return the first time a vehicle crosses a detector line.

    Crossing is defined as the first observation where:
        s_rel_stop_m >= detector_position_m

    Returns NaN if the detector is never crossed.
    """
    g = vehicle_df.sort_values("Total_Sim_Time_Sec")

    t = g["Total_Sim_Time_Sec"].to_numpy(dtype=float)
    y = g["s_rel_stop_m"].to_numpy(dtype=float)

    crossing_idx = np.where(y >= detector_position_m)[0]

    if crossing_idx.size == 0:
        return np.nan

    return float(t[crossing_idx[0]])


def build_curve_rows(
    run_id: int,
    curve_type: str,
    event_times: np.ndarray,
) -> pd.DataFrame:
    """
    Build plot-ready cumulative curve rows.

    Parameters
    ----------
    run_id:
        Simulation run ID.
    curve_type:
        Curve label such as A, D, V, or B.
    event_times:
        Event times associated with cumulative count order.

    Returns
    -------
    pd.DataFrame
        Columns: run_id, curve_type, N, time_sec
    """
    times = np.asarray(event_times, dtype=float)
    times = times[np.isfinite(times)]
    times.sort()

    if times.size == 0:
        return pd.DataFrame(columns=["run_id", "curve_type", "N", "time_sec"])

    return pd.DataFrame(
        {
            "run_id": int(run_id),
            "curve_type": curve_type,
            "N": np.arange(1, len(times) + 1, dtype=int),
            "time_sec": times,
        }
    )


# ============================================================
# A/D event-time generation
# ============================================================

def build_ad_times() -> pd.DataFrame:
    """
    Build upstream-arrival and stopbar-departure event times using chunked
    processing so the full trajectory file is never copied into memory.

    Returns
    -------
    pd.DataFrame
        Vehicle-level A/D event-time table.
    """
    print("=" * 80)
    print("Building A/D event times from filtered trajectories")
    print("=" * 80)

    if not TRAJ_NB_FILTERED_CSV.exists():
        raise FileNotFoundError(f"Missing filtered trajectory file: {TRAJ_NB_FILTERED_CSV}")

    header = pd.read_csv(TRAJ_NB_FILTERED_CSV, nrows=0)
    header_cols = set(header.columns)

    required = {
        "run_id",
        "veh_uid",
        "Total_Sim_Time_Sec",
        "s_rel_stop_m",
    }
    missing = required - header_cols
    if missing:
        raise ValueError(f"traj_nb_filtered.csv missing required columns: {sorted(missing)}")

    has_vehid = "vehID" in header_cols

    usecols = [
        "run_id",
        "veh_uid",
        "Total_Sim_Time_Sec",
        "s_rel_stop_m",
    ]
    if has_vehid:
        usecols.append("vehID")

    # Store one compact record per vehicle only.
    records: dict[tuple[int, str], dict] = {}

    chunksize = 1_000_000
    total_rows = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            TRAJ_NB_FILTERED_CSV,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
        ),
        start=1,
    ):
        total_rows += len(chunk)

        chunk["run_id"] = pd.to_numeric(chunk["run_id"], errors="coerce")
        chunk["veh_uid"] = chunk["veh_uid"].astype(str).str.strip()
        chunk["Total_Sim_Time_Sec"] = pd.to_numeric(chunk["Total_Sim_Time_Sec"], errors="coerce")
        chunk["s_rel_stop_m"] = pd.to_numeric(chunk["s_rel_stop_m"], errors="coerce")

        if has_vehid:
            chunk["vehID"] = pd.to_numeric(chunk["vehID"], errors="coerce")

        chunk = chunk.dropna(
            subset=[
                "run_id",
                "veh_uid",
                "Total_Sim_Time_Sec",
                "s_rel_stop_m",
            ]
        )

        if chunk.empty:
            continue

        chunk["run_id"] = chunk["run_id"].astype(int)
        chunk = chunk[chunk["run_id"].isin(RUN_IDS)]

        if chunk.empty:
            continue

        # Register vehicles seen in this chunk.
        veh_rows = chunk[["run_id", "veh_uid"]].drop_duplicates()
        for _, r in veh_rows.iterrows():
            key = (int(r["run_id"]), str(r["veh_uid"]))
            if key not in records:
                rec = {
                    "run_id": key[0],
                    "veh_uid": key[1],
                    "t_arr_updet_sec": np.nan,
                    "t_dep_stopbar_sec": np.nan,
                }
                if has_vehid:
                    rec["vehID"] = np.nan
                records[key] = rec

        # Store vehID if available.
        if has_vehid:
            veh_id_rows = (
                chunk.dropna(subset=["vehID"])
                .sort_values("Total_Sim_Time_Sec")
                .drop_duplicates(["run_id", "veh_uid"], keep="first")
            )
            for _, r in veh_id_rows.iterrows():
                key = (int(r["run_id"]), str(r["veh_uid"]))
                if key in records and pd.isna(records[key].get("vehID", np.nan)):
                    records[key]["vehID"] = int(r["vehID"])

        # First upstream detector crossing.
        arr = chunk[chunk["s_rel_stop_m"] >= float(Y_UPSTREAM_DETECTOR_M)]
        if not arr.empty:
            arr_min = (
                arr.groupby(["run_id", "veh_uid"], as_index=False)["Total_Sim_Time_Sec"]
                .min()
                .rename(columns={"Total_Sim_Time_Sec": "t_arr_updet_sec"})
            )
            for _, r in arr_min.iterrows():
                key = (int(r["run_id"]), str(r["veh_uid"]))
                val = float(r["t_arr_updet_sec"])
                old = records[key]["t_arr_updet_sec"]
                if pd.isna(old) or val < old:
                    records[key]["t_arr_updet_sec"] = val

        # First stopbar crossing.
        dep = chunk[chunk["s_rel_stop_m"] >= float(Y_STOPBAR_M)]
        if not dep.empty:
            dep_min = (
                dep.groupby(["run_id", "veh_uid"], as_index=False)["Total_Sim_Time_Sec"]
                .min()
                .rename(columns={"Total_Sim_Time_Sec": "t_dep_stopbar_sec"})
            )
            for _, r in dep_min.iterrows():
                key = (int(r["run_id"]), str(r["veh_uid"]))
                val = float(r["t_dep_stopbar_sec"])
                old = records[key]["t_dep_stopbar_sec"]
                if pd.isna(old) or val < old:
                    records[key]["t_dep_stopbar_sec"] = val

        print(
            f"[Chunk {chunk_idx:03d}] rows read: {total_rows:,}, "
            f"vehicles tracked: {len(records):,}"
        )

    ad = pd.DataFrame(records.values())

    if ad.empty:
        raise ValueError("No A/D event-time rows were created.")

    output_cols = ["run_id"]

    if has_vehid:
        output_cols.append("vehID")

    output_cols += [
        "veh_uid",
        "t_arr_updet_sec",
        "t_dep_stopbar_sec",
    ]

    ad = ad[output_cols].copy()
    ad = ad.sort_values(["run_id", "veh_uid"]).reset_index(drop=True)

    AD_TIMES_CSV.parent.mkdir(parents=True, exist_ok=True)
    ad.to_csv(AD_TIMES_CSV, index=False)

    print(f"\nSaved A/D event times to: {AD_TIMES_CSV}")
    print(f"Rows: {len(ad):,}")

    return ad


# ============================================================
# V/B curve generation
# ============================================================

def build_vb_curves(ad: pd.DataFrame) -> pd.DataFrame:
    """
    Build V(t) and baseline B(t) curves for all runs.

    The baseline B curve is computed using:
        V(t) = A(t + T_ff)
        beta = 1 - vq / vf
        B = D - (D - V) / beta

    Returns
    -------
    pd.DataFrame
        V/B curve table for all runs.
    """
    print("=" * 80)
    print("Building V and baseline B curves")
    print("=" * 80)

    if not np.isfinite(T_FF_SEC) or T_FF_SEC <= 0:
        raise ValueError("Invalid T_FF_SEC. Check VF_FPS and detector spacing.")

    if not np.isfinite(BETA) or BETA <= 0:
        raise ValueError("Invalid BETA. Check VF_FPS and VQ_FPS.")

    print(f"Detector spacing : {DETECTOR_SPACING_FT:.1f} ft")
    print(f"Free-flow speed  : {VF_FPS:.3f} ft/s ({VF_MPH:.3f} mph)")
    print(f"Creep speed      : {VQ_FPS:.3f} ft/s ({VQ_MPH:.3f} mph)")
    print(f"T_ff             : {T_FF_SEC:.3f} s")
    print(f"Beta             : {BETA:.6f}")

    required = {
        "run_id",
        "veh_uid",
        "t_arr_updet_sec",
        "t_dep_stopbar_sec",
    }
    require_columns(ad, required, "ad_times_all_runs.csv")

    ad["run_id"] = pd.to_numeric(ad["run_id"], errors="coerce")
    ad["t_arr_updet_sec"] = pd.to_numeric(ad["t_arr_updet_sec"], errors="coerce")
    ad["t_dep_stopbar_sec"] = pd.to_numeric(ad["t_dep_stopbar_sec"], errors="coerce")

    ad = ad.dropna(subset=["run_id"]).copy()
    ad["run_id"] = ad["run_id"].astype(int)

    vb_parts = []

    for run_id in RUN_IDS:
        run_ad = ad[ad["run_id"] == int(run_id)].copy()

        if run_ad.empty:
            print(f"[Run {run_id:03d}] No A/D rows found. Skipping.")
            continue

        t_arrival = np.sort(run_ad["t_arr_updet_sec"].dropna().to_numpy(dtype=float))
        t_departure = np.sort(run_ad["t_dep_stopbar_sec"].dropna().to_numpy(dtype=float))

        t_virtual = t_arrival + T_FF_SEC

        matched_count = min(len(t_virtual), len(t_departure))

        if matched_count == 0:
            print(f"[Run {run_id:03d}] No valid matched V/D events. Skipping.")
            continue

        t_v = t_virtual[:matched_count]
        t_d = t_departure[:matched_count]

        delay = t_d - t_v
        delay = np.maximum(delay, 0.0)

        t_b = t_d - (delay / BETA)

        # Enforce monotonicity in cumulative order.
        t_b = np.maximum.accumulate(t_b)

        run_vb = pd.DataFrame(
            {
                "run_id": int(run_id),
                "N": np.arange(1, matched_count + 1, dtype=int),
                "tV_sec": t_v,
                "tD_sec": t_d,
                "w_sec": delay,
                "beta": float(BETA),
                "vf_fps": float(VF_FPS),
                "vq_fps": float(VQ_FPS),
                "T_ff_sec": float(T_FF_SEC),
                "tB_sec": t_b,
            }
        )

        vb_parts.append(run_vb)
        print(f"[Run {run_id:03d}] V/B rows: {len(run_vb):,}")

    if not vb_parts:
        raise ValueError("No V/B curve rows were created.")

    vb = pd.concat(vb_parts, ignore_index=True)

    VB_CURVES_CSV.parent.mkdir(parents=True, exist_ok=True)
    vb.to_csv(VB_CURVES_CSV, index=False)

    print(f"\nSaved V/B curves to: {VB_CURVES_CSV}")
    print(f"Rows: {len(vb):,}")

    return vb


# ============================================================
# Baseline B vehicle-level join file
# ============================================================

def build_baseline_b_join_file(ad: pd.DataFrame, vb: pd.DataFrame) -> pd.DataFrame:
    """
    Build a vehicle-level baseline B join file.

    Vehicles are ordered by stopbar departure time. Baseline B event time is
    attached by cumulative order N. If the baseline B event occurs before
    departure, the vehicle is considered baseline-queued; otherwise the
    departure time is used as its operational event time.
    """
    print("=" * 80)
    print("Building baseline B vehicle-level join file")
    print("=" * 80)

    baseline_parts = []

    has_vehid = "vehID" in ad.columns

    for run_id in RUN_IDS:
        run_ad = ad[ad["run_id"] == int(run_id)].copy()
        run_vb = vb[vb["run_id"] == int(run_id)].copy()

        if run_ad.empty or run_vb.empty:
            print(f"[Run {run_id:03d}] Missing AD or VB data. Skipping baseline join file.")
            continue

        run_ad["t_dep_stopbar_sec"] = pd.to_numeric(
            run_ad["t_dep_stopbar_sec"],
            errors="coerce",
        )

        leave = run_ad.dropna(subset=["veh_uid", "t_dep_stopbar_sec"]).copy()
        leave["veh_uid"] = leave["veh_uid"].astype(str).str.strip()
        leave = leave.sort_values("t_dep_stopbar_sec").reset_index(drop=True)
        leave["N"] = np.arange(1, len(leave) + 1, dtype=int)

        run_vb = run_vb[["N", "tB_sec"]].copy()
        run_vb["N"] = pd.to_numeric(run_vb["N"], errors="coerce").astype(int)
        run_vb["tB_sec"] = pd.to_numeric(run_vb["tB_sec"], errors="coerce")

        leave = leave.merge(run_vb, on="N", how="left")

        leave["baseline_joined_queue"] = (
            leave["tB_sec"].notna()
            & (leave["tB_sec"] < leave["t_dep_stopbar_sec"] - EPS_SEC)
        ).astype(int)

        leave["t_join_base_sec"] = leave["t_dep_stopbar_sec"]
        leave.loc[
            leave["baseline_joined_queue"] == 1,
            "t_join_base_sec",
        ] = leave.loc[
            leave["baseline_joined_queue"] == 1,
            "tB_sec",
        ]

        leave["run_id"] = int(run_id)

        cols = [
            "run_id",
            "veh_uid",
            "N",
            "t_dep_stopbar_sec",
            "tB_sec",
            "baseline_joined_queue",
            "t_join_base_sec",
        ]

        if has_vehid:
            cols.insert(1, "vehID")

        baseline_parts.append(leave[cols].copy())
        print(f"[Run {run_id:03d}] Baseline join rows: {len(leave):,}")

    if not baseline_parts:
        raise ValueError("No baseline B join rows were created.")

    baseline_join = pd.concat(baseline_parts, ignore_index=True)

    BASELINE_B_JOIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    baseline_join.to_csv(BASELINE_B_JOIN_CSV, index=False)

    print(f"\nSaved baseline B join file to: {BASELINE_B_JOIN_CSV}")
    print(f"Rows: {len(baseline_join):,}")

    return baseline_join


# ============================================================
# Plot-ready cumulative curves
# ============================================================

def build_plot_ready_cumulative_curves(ad: pd.DataFrame, vb: pd.DataFrame) -> pd.DataFrame:
    """
    Save one long-format plot-ready file for A, D, V, and B curves.

    Output columns:
        run_id
        curve_type
        N
        time_sec
    """
    print("=" * 80)
    print("Building plot-ready A/D/V/B cumulative curve file")
    print("=" * 80)

    plot_parts = []

    for run_id in RUN_IDS:
        run_ad = ad[ad["run_id"] == int(run_id)].copy()
        run_vb = vb[vb["run_id"] == int(run_id)].copy()

        if run_ad.empty:
            continue

        plot_parts.append(
            build_curve_rows(
                run_id=run_id,
                curve_type="A",
                event_times=run_ad["t_arr_updet_sec"].to_numpy(dtype=float),
            )
        )

        plot_parts.append(
            build_curve_rows(
                run_id=run_id,
                curve_type="D",
                event_times=run_ad["t_dep_stopbar_sec"].to_numpy(dtype=float),
            )
        )

        if not run_vb.empty:
            plot_parts.append(
                build_curve_rows(
                    run_id=run_id,
                    curve_type="V",
                    event_times=run_vb["tV_sec"].to_numpy(dtype=float),
                )
            )

            plot_parts.append(
                build_curve_rows(
                    run_id=run_id,
                    curve_type="B",
                    event_times=run_vb["tB_sec"].to_numpy(dtype=float),
                )
            )

    if not plot_parts:
        raise ValueError("No plot-ready cumulative curve rows were created.")

    plot_ready = pd.concat(plot_parts, ignore_index=True)

    plot_ready = plot_ready.dropna(subset=["time_sec"]).copy()
    plot_ready["run_id"] = plot_ready["run_id"].astype(int)
    plot_ready["N"] = plot_ready["N"].astype(int)

    CUMULATIVE_CURVES_PLOT_READY_CSV.parent.mkdir(parents=True, exist_ok=True)
    plot_ready.to_csv(CUMULATIVE_CURVES_PLOT_READY_CSV, index=False)

    print(f"Saved plot-ready cumulative curves to: {CUMULATIVE_CURVES_PLOT_READY_CSV}")
    print(f"Rows: {len(plot_ready):,}")

    return plot_ready


# ============================================================
# Main pipeline
# ============================================================

def run_cumulative_count_pipeline() -> None:
    """
    Run the cumulative-count theory stage.
    """
    ensure_project_directories()

    ad = build_ad_times()
    vb = build_vb_curves(ad)
    build_baseline_b_join_file(ad, vb)
    build_plot_ready_cumulative_curves(ad, vb)

    print("\nCumulative-count theory processing complete.")


def main() -> None:
    """Script entry point."""
    run_cumulative_count_pipeline()


if __name__ == "__main__":
    main()