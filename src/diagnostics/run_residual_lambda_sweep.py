"""
Residual sequence lambda sweep diagnostic.

Run several lambda settings for train_residual_sequence_diagnostics.py and save
separate metrics/plots for each setting.
"""

from __future__ import annotations

import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


THIS_FILE = Path(__file__).resolve()
DIAGNOSTICS_DIR = THIS_FILE.parent
SRC_DIR = DIAGNOSTICS_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

for p in [str(SRC_DIR), str(DIAGNOSTICS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

diag = importlib.import_module("train_residual_sequence_diagnostics")


SWEEP_ROOT = PROJECT_ROOT / "output" / "intermediate_csv" / "residual_sequence_lambda_sweep"

# GRU only for first sweep because GRU already performed better than LSTM.
# Change this to True later if you want to sweep LSTM too.
TRAIN_GRU_FOR_SWEEP = True
TRAIN_LSTM_FOR_SWEEP = False
TRAIN_XGBOOST_FOR_SWEEP = False

# This avoids writing many per-run/rate CSVs for every sweep case.
SAVE_PER_RUN_RATE_FILES = False


@dataclass(frozen=True)
class SweepSetting:
    name: str
    zero_queue: float
    residual_dq: float
    residual_d2q: float
    notes: str


SWEEP_SETTINGS = [
    SweepSetting("s00_current", 0.20, 0.08, 0.22, "Current reference setting."),
    SweepSetting("s01_mild", 0.30, 0.10, 0.30, "Mildly stronger chatter control."),
    SweepSetting("s02_balanced", 0.40, 0.12, 0.35, "Balanced first tuning setting."),
    SweepSetting("s03_stronger_zero", 0.55, 0.12, 0.35, "More false-queue suppression."),
    SweepSetting("s04_stronger_smooth", 0.40, 0.16, 0.50, "More residual smoothness."),
    SweepSetting("s05_aggressive", 0.65, 0.18, 0.60, "Aggressive test for over-smoothing."),
]


def apply_setting(setting: SweepSetting) -> Path:
    case_dir = SWEEP_ROOT / setting.name

    diag.OUT_DIR = case_dir
    diag.MODEL_DIR = case_dir / "trained_models"
    diag.FIG_DIR = case_dir / "figures"

    diag.TRAIN_XGBOOST = TRAIN_XGBOOST_FOR_SWEEP
    diag.TRAIN_GRU = TRAIN_GRU_FOR_SWEEP
    diag.TRAIN_LSTM = TRAIN_LSTM_FOR_SWEEP
    diag.SAVE_PER_RUN_RATE_FILES = SAVE_PER_RUN_RATE_FILES

    diag.LAMBDA_ZERO_QUEUE_MATCH = float(setting.zero_queue)
    diag.LAMBDA_RESIDUAL_DQ = float(setting.residual_dq)
    diag.LAMBDA_RESIDUAL_D2Q = float(setting.residual_d2q)

    return case_dir


def read_metrics(case_dir: Path, setting: SweepSetting, elapsed_sec: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_path = case_dir / "residual_sequence_metrics_by_model_run.csv"
    rate_path = case_dir / "residual_sequence_metrics_by_model_run_rate.csv"

    common = {
        "sweep_name": setting.name,
        "lambda_zero_queue_match": setting.zero_queue,
        "lambda_residual_dq": setting.residual_dq,
        "lambda_residual_d2q": setting.residual_d2q,
        "sweep_notes": setting.notes,
        "elapsed_sec": round(float(elapsed_sec), 2),
    }

    run_df = pd.DataFrame()
    rate_df = pd.DataFrame()

    if run_path.exists():
        run_df = pd.read_csv(run_path)
        for k, v in common.items():
            run_df[k] = v

    if rate_path.exists():
        rate_df = pd.read_csv(rate_path)
        for k, v in common.items():
            rate_df[k] = v

    return run_df, rate_df


def build_ranking(all_run_metrics: pd.DataFrame) -> pd.DataFrame:
    if all_run_metrics.empty:
        return pd.DataFrame()

    metric_cols = [
        c for c in ["mae_ft", "rmse_ft", "max_abs_error_ft", "area_abs_error_ft_s"]
        if c in all_run_metrics.columns
    ]

    group_cols = [
        "sweep_name",
        "family",
        "model",
        "lambda_zero_queue_match",
        "lambda_residual_dq",
        "lambda_residual_d2q",
        "sweep_notes",
    ]

    ranking = (
        all_run_metrics.groupby(group_cols, as_index=False)[metric_cols]
        .mean()
        .sort_values(["rmse_ft", "mae_ft"])
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank_by_rmse", range(1, len(ranking) + 1))
    return ranking


def main() -> None:
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("Residual sequence lambda sweep")
    print("=" * 90)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Sweep root   : {SWEEP_ROOT}")
    print(f"GRU enabled  : {TRAIN_GRU_FOR_SWEEP}")
    print(f"LSTM enabled : {TRAIN_LSTM_FOR_SWEEP}")
    print(f"Cases        : {len(SWEEP_SETTINGS)}")
    print("=" * 90)

    run_parts = []
    rate_parts = []

    for i, setting in enumerate(SWEEP_SETTINGS, start=1):
        print("\n" + "-" * 90)
        print(f"[{i}/{len(SWEEP_SETTINGS)}] {setting.name}")
        print(
            f"zero={setting.zero_queue}, "
            f"residual_dq={setting.residual_dq}, "
            f"residual_d2q={setting.residual_d2q}"
        )
        print(setting.notes)

        case_dir = apply_setting(setting)
        t0 = time.perf_counter()

        diag.main()

        elapsed = time.perf_counter() - t0
        run_df, rate_df = read_metrics(case_dir, setting, elapsed)

        if not run_df.empty:
            run_parts.append(run_df)
        if not rate_df.empty:
            rate_parts.append(rate_df)

        print(f"[Completed] {setting.name} in {elapsed / 60.0:.2f} minutes")
        print(f"[Folder] {case_dir}")

    all_run = pd.concat(run_parts, ignore_index=True) if run_parts else pd.DataFrame()
    all_rate = pd.concat(rate_parts, ignore_index=True) if rate_parts else pd.DataFrame()
    ranking = build_ranking(all_run)

    if not all_run.empty:
        out = SWEEP_ROOT / "lambda_sweep_metrics_by_model_run.csv"
        all_run.to_csv(out, index=False)
        print(f"\n[Saved] {out}")

    if not all_rate.empty:
        out = SWEEP_ROOT / "lambda_sweep_metrics_by_model_run_rate.csv"
        all_rate.to_csv(out, index=False)
        print(f"[Saved] {out}")

    if not ranking.empty:
        out = SWEEP_ROOT / "lambda_sweep_ranking_by_model.csv"
        ranking.to_csv(out, index=False)
        print(f"[Saved] {out}")

        show_cols = [
            "rank_by_rmse",
            "sweep_name",
            "family",
            "model",
            "mae_ft",
            "rmse_ft",
            "max_abs_error_ft",
            "lambda_zero_queue_match",
            "lambda_residual_dq",
            "lambda_residual_d2q",
        ]
        show_cols = [c for c in show_cols if c in ranking.columns]

        print("\nTop sweep settings:")
        print(ranking[show_cols].head(12).round(3).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()