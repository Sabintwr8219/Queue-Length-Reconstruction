"""
Preprocess raw VISSIM trajectory and signal files.

This script performs two main tasks:
1. Parse raw VISSIM FZP and LSA files into master CSV files.
2. Filter northbound through trajectories and compute corridor-based stationing.

Outputs:
    data/processed_data/master_trajectory.csv
    data/processed_data/master_phase_time.csv
    data/processed_data/traj_nb_filtered.csv
    output/results/veh_summary.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from config import (
    VISSIM_RESULTS_DIR,
    VISSIM_BASE_NAME,
    RUN_IDS,
    CHUNKSIZE_FZP,
    MASTER_TRAJECTORY_CSV,
    MASTER_PHASE_CSV,
    VEHICLE_FILTER_SUMMARY_CSV,
    TRAJ_NB_FILTERED_CSV,
    C1_LAT,
    C1_LON,
    C2_LAT,
    C2_LON,
    CRS_WGS84,
    CRS_PROJECTED,
    M_TO_FT,
    UPSTREAM_DETECTOR_OFFSET_M,
    STOPBAR_OFFSET_M,
    MAX_PERP_DIST_M,
    BUFFER_BEFORE_M,
    BUFFER_AFTER_M,
    DIR_LOOKBACK_SEC,
    STOPBAR_STORE_BAND_M,
    ensure_project_directories,
)


# ============================================================
# Column settings
# ============================================================

RUN_COL = "run_id"
VEH_COL = "vehID"
UID_COL = "veh_uid"
TIME_COL = "Total_Sim_Time_Sec"
LAT_COL = "Lat"
LON_COL = "Lon"

FZP_REQUIRED_COLS = [
    "SIMSEC",
    "NO",
    "SPEED",
    "COORDFRONTX",
    "COORDFRONTY",
    "LONGWGS84",
    "LATWGS84",
]

LSA_COLNAMES = [
    "simsec",
    "simtime",
    "controller",
    "signal_group",
    "state",
    "duration",
    "controller_type",
    "program",
]


# ============================================================
# Raw VISSIM parsing helpers
# ============================================================

def find_fzp_vehicle_header(fzp_path: Path) -> tuple[int, list[str]]:
    """
    Locate the $VEHICLE header line in a VISSIM FZP file.

    Parameters
    ----------
    fzp_path:
        Path to the raw FZP file.

    Returns
    -------
    tuple
        Header line index and parsed column names.
    """
    with fzp_path.open("r", errors="ignore") as file:
        for line_idx, line in enumerate(file):
            if line.startswith("$VEHICLE:"):
                raw_cols = line.replace("$VEHICLE:", "").strip().split(";")
                raw_cols = [col.strip() for col in raw_cols if col.strip()]
                return line_idx, raw_cols

    raise ValueError(f"Could not find '$VEHICLE:' header in {fzp_path}")


def parse_fzp_file(
    fzp_path: Path,
    run_id: int,
    output_csv: Path,
    first_write: bool,
    chunksize: int,
) -> bool:
    """
    Parse one FZP file and append cleaned trajectory rows to the master trajectory CSV.

    Returns
    -------
    bool
        Updated first_write flag.
    """
    print(f"[FZP] Parsing {fzp_path.name} for run {run_id:03d}")

    header_idx, raw_cols = find_fzp_vehicle_header(fzp_path)

    missing = [col for col in FZP_REQUIRED_COLS if col not in raw_cols]
    if missing:
        raise ValueError(
            f"{fzp_path} is missing required FZP columns: {missing}\n"
            f"Available columns: {raw_cols}"
        )

    reader = pd.read_csv(
        fzp_path,
        sep=";",
        skiprows=header_idx + 1,
        names=raw_cols,
        header=None,
        engine="python",
        chunksize=chunksize,
        usecols=FZP_REQUIRED_COLS,
        on_bad_lines="skip",
    )

    rows_written = 0

    for chunk_idx, chunk in enumerate(reader, start=1):
        for col in FZP_REQUIRED_COLS:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunk = chunk.dropna(subset=["SIMSEC", "NO"])
        if chunk.empty:
            continue

        vehicle_id = chunk["NO"].astype(int)
        speed_mph = chunk["SPEED"].astype(float)
        speed_fps = speed_mph * 1.4666666667

        veh_uid = pd.Series(run_id, index=chunk.index).astype(str) + "_" + vehicle_id.astype(str)

        out = pd.DataFrame(
            {
                "run_id": int(run_id),
                "veh_uid": veh_uid,
                "Total_Sim_Time_Sec": chunk["SIMSEC"].astype(float),
                "vehID": vehicle_id,
                "X": chunk["COORDFRONTX"].astype(float),
                "Y": chunk["COORDFRONTY"].astype(float),
                "Speed_mph": speed_mph,
                "Speed_fps": speed_fps,
                "Lat": chunk["LATWGS84"].astype(float),
                "Lon": chunk["LONGWGS84"].astype(float),
            }
        )

        out.to_csv(
            output_csv,
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )

        first_write = False
        rows_written += len(out)

        if chunk_idx % 20 == 0:
            print(f"    chunk {chunk_idx}: rows written so far = {rows_written:,}")

    print(f"[FZP] Done run {run_id:03d}. Rows written = {rows_written:,}")
    return first_write


def parse_lsa_file(lsa_path: Path, run_id: int, output_csv: Path, first_write: bool) -> bool:
    """
    Parse one LSA file and append cleaned signal-state rows to the master phase CSV.

    Returns
    -------
    bool
        Updated first_write flag.
    """
    print(f"[LSA] Parsing {lsa_path.name} for run {run_id:03d}")

    start_idx = None

    with lsa_path.open("r", errors="ignore") as file:
        for line_idx, line in enumerate(file):
            if re.match(r"^\s*\d+(\.\d+)?\s*;\s*\d", line):
                start_idx = line_idx
                break

    if start_idx is None:
        raise ValueError(f"Could not find first numeric data row in {lsa_path}")

    df = pd.read_csv(
        lsa_path,
        sep=";",
        skiprows=start_idx,
        header=None,
        engine="python",
        on_bad_lines="skip",
    )

    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 6:
        raise ValueError(f"{lsa_path} has too few columns: {df.shape[1]}")

    if df.shape[1] > len(LSA_COLNAMES):
        df = df.iloc[:, : len(LSA_COLNAMES)]

    df.columns = LSA_COLNAMES[: df.shape[1]]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    for col in ["simsec", "simtime", "controller", "signal_group", "duration", "program"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "state" in df.columns:
        state_map = {
            "r": "red",
            "red": "red",
            "0": "red",
            "g": "green",
            "green": "green",
            "1": "green",
            "y": "amber",
            "yellow": "amber",
            "amber": "amber",
        }
        state_raw = df["state"].astype(str).str.strip().str.lower()
        df["state"] = state_raw.map(lambda value: state_map.get(value, value))

    required_subset = [col for col in ["simsec", "controller", "signal_group", "state"] if col in df.columns]
    df = df.dropna(subset=required_subset)

    if df.empty:
        print(f"[LSA] No valid rows found for run {run_id:03d}")
        return first_write

    df.insert(0, "run_id", int(run_id))

    df.to_csv(
        output_csv,
        index=False,
        mode="w" if first_write else "a",
        header=first_write,
    )

    print(f"[LSA] Done run {run_id:03d}. Rows written = {len(df):,}")
    return False


def build_master_vissim_files() -> None:
    """
    Build master trajectory and master phase CSV files from all configured VISSIM runs.
    """
    MASTER_TRAJECTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    MASTER_PHASE_CSV.parent.mkdir(parents=True, exist_ok=True)

    first_write_traj = True
    first_write_phase = True

    print("=" * 80)
    print("Building master VISSIM files")
    print("=" * 80)
    print(f"Raw VISSIM directory : {VISSIM_RESULTS_DIR}")
    print(f"Run IDs              : {RUN_IDS}")
    print(f"Trajectory output    : {MASTER_TRAJECTORY_CSV}")
    print(f"Phase output         : {MASTER_PHASE_CSV}")
    print("=" * 80)

    for run_id in RUN_IDS:
        fzp_file = VISSIM_RESULTS_DIR / f"{VISSIM_BASE_NAME}_{run_id:03d}.fzp"
        lsa_file = VISSIM_RESULTS_DIR / f"{VISSIM_BASE_NAME}_{run_id:03d}.lsa"

        if not fzp_file.exists():
            print(f"[WARN] Missing FZP for run {run_id:03d}: {fzp_file}")
            continue

        if not lsa_file.exists():
            print(f"[WARN] Missing LSA for run {run_id:03d}: {lsa_file}")
            continue

        print(f"\n--- Processing run {run_id:03d} ---")

        first_write_traj = parse_fzp_file(
            fzp_path=fzp_file,
            run_id=run_id,
            output_csv=MASTER_TRAJECTORY_CSV,
            first_write=first_write_traj,
            chunksize=CHUNKSIZE_FZP,
        )

        first_write_phase = parse_lsa_file(
            lsa_path=lsa_file,
            run_id=run_id,
            output_csv=MASTER_PHASE_CSV,
            first_write=first_write_phase,
        )

    print("\nMaster VISSIM parsing complete.")


# ============================================================
# Corridor projection and filtering helpers
# ============================================================

def compute_corridor_geometry() -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], float, float]:
    """
    Compute corridor geometry using the two intersection reference points.

    Returns
    -------
    tuple
        (C1_XY, C2_XY, unit_axis, corridor_length_m, stopbar_station_m)
    """
    transformer = Transformer.from_crs(CRS_WGS84, CRS_PROJECTED, always_xy=True)

    c1_x, c1_y = transformer.transform(C1_LON, C1_LAT)
    c2_x, c2_y = transformer.transform(C2_LON, C2_LAT)

    dx = c2_x - c1_x
    dy = c2_y - c1_y
    corridor_length_m = float(np.hypot(dx, dy))

    if corridor_length_m <= 1e-6:
        raise ValueError("C1 and C2 are too close or identical after projection.")

    ux = dx / corridor_length_m
    uy = dy / corridor_length_m

    stopbar_station_m = corridor_length_m - STOPBAR_OFFSET_M

    return (c1_x, c1_y), (c2_x, c2_y), (ux, uy), corridor_length_m, stopbar_station_m


def latlon_to_projected(lon_array: np.ndarray, lat_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert lon/lat arrays to projected coordinates.
    """
    transformer = Transformer.from_crs(CRS_WGS84, CRS_PROJECTED, always_xy=True)
    x, y = transformer.transform(lon_array, lat_array)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def project_to_corridor(
    x: np.ndarray,
    y: np.ndarray,
    c1_x: float,
    c1_y: float,
    ux: float,
    uy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project coordinates onto the corridor axis.

    Returns
    -------
    tuple
        Along-corridor station in meters and perpendicular distance in meters.
    """
    dx = x - c1_x
    dy = y - c1_y

    s_m = dx * ux + dy * uy
    d_perp_m = dx * (-uy) + dy * ux

    return s_m, d_perp_m


def initialize_vehicle_record(vehicle_stats: dict, stopbar_points: dict, key: tuple[int, int]) -> None:
    """
    Initialize per-vehicle summary containers.
    """
    vehicle_stats[key] = {
        "min_s": np.inf,
        "max_s": -np.inf,
        "max_abs_perp": 0.0,
        "t_min": np.inf,
        "t_max": -np.inf,
        "s_first": np.nan,
        "s_last": np.nan,
    }

    stopbar_points[key] = []


def update_vehicle_stats(
    vehicle_stats: dict,
    key: tuple[int, int],
    t_array: np.ndarray,
    s_array: np.ndarray,
    perp_array: np.ndarray,
) -> None:
    """
    Update per-vehicle trajectory statistics.
    """
    st = vehicle_stats[key]

    st["min_s"] = min(st["min_s"], float(np.nanmin(s_array)))
    st["max_s"] = max(st["max_s"], float(np.nanmax(s_array)))
    st["max_abs_perp"] = max(st["max_abs_perp"], float(np.nanmax(np.abs(perp_array))))

    idx_first = int(np.nanargmin(t_array))
    idx_last = int(np.nanargmax(t_array))

    t_first = float(t_array[idx_first])
    t_last = float(t_array[idx_last])

    if t_first < st["t_min"]:
        st["t_min"] = t_first
        st["s_first"] = float(s_array[idx_first])

    if t_last > st["t_max"]:
        st["t_max"] = t_last
        st["s_last"] = float(s_array[idx_last])


def strict_nb_through_filter(vehicle_record: dict, corridor_length_m: float) -> tuple[bool, float]:
    """
    Apply the strict northbound through-vehicle filter.
    """
    if np.isnan(vehicle_record["s_first"]) or np.isnan(vehicle_record["s_last"]):
        delta_s_net = np.nan
    else:
        delta_s_net = vehicle_record["s_last"] - vehicle_record["s_first"]

    keep = (
        vehicle_record["min_s"] <= (0.0 - BUFFER_BEFORE_M)
        and vehicle_record["max_s"] >= (corridor_length_m + BUFFER_AFTER_M)
        and vehicle_record["max_abs_perp"] <= MAX_PERP_DIST_M
        and delta_s_net > 0
    )

    return bool(keep), float(delta_s_net)


def robust_stopbar_crossing(t: np.ndarray, s: np.ndarray, r: np.ndarray) -> dict:
    """
    Detect stopbar crossing using stopbar-relative position.

    Parameters
    ----------
    t:
        Time values.
    s:
        Along-corridor station values.
    r:
        Stopbar-relative station values.

    Returns
    -------
    dict
        Crossing information.
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    r = np.asarray(r, dtype=float)

    if r.size < 2:
        return {"crossed": False}

    a = r[:-1]
    b = r[1:]

    candidate = (a <= 0) & (b >= 0) & ~((a == 0) & (b == 0))

    if np.any(candidate):
        i = int(np.argmax(candidate))
        idx = i + 1
        return {
            "crossed": True,
            "idx": idx,
            "t_cross": float(t[idx]),
            "s_cross": float(s[idx]),
            "r_cross": float(r[idx]),
        }

    if np.nanmin(r) <= 0 and np.nanmax(r) >= 0:
        idxs = np.where(r >= 0)[0]
        if idxs.size:
            idx = int(idxs[0])
            return {
                "crossed": True,
                "idx": idx,
                "t_cross": float(t[idx]),
                "s_cross": float(s[idx]),
                "r_cross": float(r[idx]),
            }

    return {"crossed": False}


def delta_s_around_crossing(t: np.ndarray, s: np.ndarray, idx: int) -> float:
    """
    Estimate movement direction around the stopbar crossing.
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)

    crossing_time = float(t[idx])

    back = np.where(t == (crossing_time - DIR_LOOKBACK_SEC))[0]
    if back.size:
        return float(s[idx] - s[back[-1]])

    forward = np.where(t == (crossing_time + DIR_LOOKBACK_SEC))[0]
    if forward.size:
        return float(s[forward[0]] - s[idx])

    return np.nan


def collect_vehicle_stats_and_stopbar_points(
    c1_x: float,
    c1_y: float,
    ux: float,
    uy: float,
    stopbar_station_m: float,
) -> tuple[dict, dict]:
    """
    First pass through master trajectory file.

    Collects per-vehicle movement statistics and stores only stopbar-band points
    for the robust stopbar-crossing recovery step.
    """
    vehicle_stats: dict[tuple[int, int], dict] = {}
    stopbar_points: dict[tuple[int, int], list] = {}

    usecols = [RUN_COL, VEH_COL, UID_COL, TIME_COL, LAT_COL, LON_COL]

    for chunk in pd.read_csv(MASTER_TRAJECTORY_CSV, chunksize=CHUNKSIZE_FZP, usecols=lambda c: c in usecols):
        chunk[TIME_COL] = pd.to_numeric(chunk[TIME_COL], errors="coerce")
        chunk[RUN_COL] = pd.to_numeric(chunk[RUN_COL], errors="coerce")
        chunk[VEH_COL] = pd.to_numeric(chunk[VEH_COL], errors="coerce")
        chunk[LAT_COL] = pd.to_numeric(chunk[LAT_COL], errors="coerce")
        chunk[LON_COL] = pd.to_numeric(chunk[LON_COL], errors="coerce")

        chunk = chunk.dropna(subset=[RUN_COL, VEH_COL, TIME_COL, LAT_COL, LON_COL]).copy()

        if chunk.empty:
            continue

        chunk[RUN_COL] = chunk[RUN_COL].astype(int)
        chunk[VEH_COL] = chunk[VEH_COL].astype(int)

        lon = chunk[LON_COL].to_numpy(dtype=float)
        lat = chunk[LAT_COL].to_numpy(dtype=float)

        x, y = latlon_to_projected(lon, lat)
        s_m, d_perp_m = project_to_corridor(x, y, c1_x, c1_y, ux, uy)
        s_rel_stop_m = s_m - stopbar_station_m

        temp = pd.DataFrame(
            {
                RUN_COL: chunk[RUN_COL].to_numpy(dtype=int),
                VEH_COL: chunk[VEH_COL].to_numpy(dtype=int),
                "time_sec": chunk[TIME_COL].to_numpy(dtype=float),
                "s_m": s_m,
                "d_perp_m": d_perp_m,
                "s_rel_stop_m": s_rel_stop_m,
            }
        )

        for (run_id, veh_id), group in temp.groupby([RUN_COL, VEH_COL], sort=False):
            key = (int(run_id), int(veh_id))

            if key not in vehicle_stats:
                initialize_vehicle_record(vehicle_stats, stopbar_points, key)

            t_group = group["time_sec"].to_numpy(dtype=float)
            s_group = group["s_m"].to_numpy(dtype=float)
            p_group = group["d_perp_m"].to_numpy(dtype=float)
            r_group = group["s_rel_stop_m"].to_numpy(dtype=float)

            update_vehicle_stats(vehicle_stats, key, t_group, s_group, p_group)

            band_mask = np.abs(r_group) <= STOPBAR_STORE_BAND_M
            if np.any(band_mask):
                rows = np.column_stack(
                    [
                        t_group[band_mask],
                        s_group[band_mask],
                        r_group[band_mask],
                        p_group[band_mask],
                    ]
                )
                stopbar_points[key].extend(rows.tolist())

    return vehicle_stats, stopbar_points


def recover_midlink_stopbar_crossers(
    failed_keys: set[tuple[int, int]],
    stopbar_points: dict,
) -> tuple[set[tuple[int, int]], dict]:
    """
    Recover valid vehicles that fail the strict full-corridor filter but
    clearly cross the downstream stopbar in the northbound direction.
    """
    recovered_keys: set[tuple[int, int]] = set()
    crossing_info: dict[tuple[int, int], dict] = {}

    for key in failed_keys:
        points = stopbar_points.get(key, [])

        if not points:
            continue

        arr = np.asarray(points, dtype=float)

        t = arr[:, 0]
        s = arr[:, 1]
        r = arr[:, 2]
        p = arr[:, 3]

        order = np.argsort(t)
        t, s, r, p = t[order], s[order], r[order], p[order]

        if float(np.nanmax(np.abs(p))) > MAX_PERP_DIST_M:
            continue

        crossing = robust_stopbar_crossing(t, s, r)
        if not crossing.get("crossed", False):
            continue

        idx = int(crossing["idx"])
        ds_around_cross = delta_s_around_crossing(t, s, idx)

        if np.isnan(ds_around_cross):
            continue

        if ds_around_cross > 0:
            recovered_keys.add(key)
            crossing_info[key] = {
                "t_cross": crossing["t_cross"],
                "delta_s_around_cross": ds_around_cross,
            }

    return recovered_keys, crossing_info


def write_filtered_trajectories(
    keep_keys: set[tuple[int, int]],
    c1_x: float,
    c1_y: float,
    ux: float,
    uy: float,
    stopbar_station_m: float,
) -> None:
    """
    Second pass through master trajectory file.

    Writes filtered northbound trajectories with corridor geometry columns.
    """
    TRAJ_NB_FILTERED_CSV.parent.mkdir(parents=True, exist_ok=True)

    first_write = True
    keep_set = set(keep_keys)

    for chunk in pd.read_csv(MASTER_TRAJECTORY_CSV, chunksize=CHUNKSIZE_FZP):
        chunk[TIME_COL] = pd.to_numeric(chunk[TIME_COL], errors="coerce")
        chunk[RUN_COL] = pd.to_numeric(chunk[RUN_COL], errors="coerce")
        chunk[VEH_COL] = pd.to_numeric(chunk[VEH_COL], errors="coerce")
        chunk[LAT_COL] = pd.to_numeric(chunk[LAT_COL], errors="coerce")
        chunk[LON_COL] = pd.to_numeric(chunk[LON_COL], errors="coerce")

        chunk = chunk.dropna(subset=[RUN_COL, VEH_COL, TIME_COL, LAT_COL, LON_COL]).copy()

        if chunk.empty:
            continue

        chunk[RUN_COL] = chunk[RUN_COL].astype(int)
        chunk[VEH_COL] = chunk[VEH_COL].astype(int)

        keep_mask = [
            (int(run_id), int(veh_id)) in keep_set
            for run_id, veh_id in zip(chunk[RUN_COL], chunk[VEH_COL])
        ]

        out = chunk.loc[keep_mask].copy()

        if out.empty:
            continue

        lon = out[LON_COL].to_numpy(dtype=float)
        lat = out[LAT_COL].to_numpy(dtype=float)

        x, y = latlon_to_projected(lon, lat)
        s_m, d_perp_m = project_to_corridor(x, y, c1_x, c1_y, ux, uy)
        s_rel_stop_m = s_m - stopbar_station_m

        out["s_m"] = s_m
        out["d_perp_m"] = d_perp_m
        out["s_rel_stop_m"] = s_rel_stop_m

        out["s_ft"] = out["s_m"] * M_TO_FT
        out["d_perp_ft"] = out["d_perp_m"] * M_TO_FT
        out["s_rel_stop_ft"] = out["s_rel_stop_m"] * M_TO_FT

        if UID_COL not in out.columns:
            out[UID_COL] = out[RUN_COL].astype(str) + "_" + out[VEH_COL].astype(str)

        out.to_csv(
            TRAJ_NB_FILTERED_CSV,
            mode="w" if first_write else "a",
            index=False,
            header=first_write,
        )

        first_write = False


def filter_northbound_trajectories() -> None:
    """
    Filter northbound through trajectories and save the plot-ready trajectory file.
    """
    print("=" * 80)
    print("Filtering northbound through trajectories")
    print("=" * 80)

    (
        (c1_x, c1_y),
        (_c2_x, _c2_y),
        (ux, uy),
        corridor_length_m,
        stopbar_station_m,
    ) = compute_corridor_geometry()

    vehicle_stats, stopbar_points = collect_vehicle_stats_and_stopbar_points(
        c1_x=c1_x,
        c1_y=c1_y,
        ux=ux,
        uy=uy,
        stopbar_station_m=stopbar_station_m,
    )

    strict_keep_keys: set[tuple[int, int]] = set()
    strict_fail_keys: set[tuple[int, int]] = set()
    summary_rows = []

    for (run_id, veh_id), stats in vehicle_stats.items():
        keep, delta_s_net = strict_nb_through_filter(stats, corridor_length_m)

        if keep:
            strict_keep_keys.add((run_id, veh_id))
        else:
            strict_fail_keys.add((run_id, veh_id))

        summary_rows.append(
            {
                "run_id": int(run_id),
                "vehID": int(veh_id),
                "min_s_m": stats["min_s"],
                "max_s_m": stats["max_s"],
                "max_abs_perp_m": stats["max_abs_perp"],
                "delta_s_net_m": delta_s_net,
                "strict_keep": int(keep),
            }
        )

    recovered_keys, crossing_info = recover_midlink_stopbar_crossers(
        failed_keys=strict_fail_keys,
        stopbar_points=stopbar_points,
    )

    final_keep_keys = strict_keep_keys.union(recovered_keys)

    summary = pd.DataFrame(summary_rows)

    summary["recovered_by_stopbar_crossing"] = summary.apply(
        lambda row: (int(row["run_id"]), int(row["vehID"])) in recovered_keys,
        axis=1,
    ).astype(int)

    summary["final_keep"] = summary.apply(
        lambda row: (int(row["run_id"]), int(row["vehID"])) in final_keep_keys,
        axis=1,
    ).astype(int)

    summary["t_cross_sec"] = np.nan
    summary["delta_s_around_cross_m"] = np.nan

    for (run_id, veh_id), info in crossing_info.items():
        mask = (summary["run_id"] == run_id) & (summary["vehID"] == veh_id)
        summary.loc[mask, "t_cross_sec"] = info["t_cross"]
        summary.loc[mask, "delta_s_around_cross_m"] = info["delta_s_around_cross"]

    VEHICLE_FILTER_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(VEHICLE_FILTER_SUMMARY_CSV, index=False)

    write_filtered_trajectories(
        keep_keys=final_keep_keys,
        c1_x=c1_x,
        c1_y=c1_y,
        ux=ux,
        uy=uy,
        stopbar_station_m=stopbar_station_m,
    )

    print("\nFiltering complete.")
    print(f"Total unique vehicles     : {len(vehicle_stats):,}")
    print(f"Kept by strict filter     : {len(strict_keep_keys):,}")
    print(f"Recovered at stopbar      : {len(recovered_keys):,}")
    print(f"Final kept vehicles       : {len(final_keep_keys):,}")
    print(f"Vehicle summary saved to  : {VEHICLE_FILTER_SUMMARY_CSV}")
    print(f"Filtered trajectories     : {TRAJ_NB_FILTERED_CSV}")


# ============================================================
# Main entry point
# ============================================================

def main() -> None:
    """
    Run full preprocessing:
    1. Parse raw VISSIM files.
    2. Filter northbound trajectories.
    """
    ensure_project_directories()
    build_master_vissim_files()
    filter_northbound_trajectories()


if __name__ == "__main__":
    main()