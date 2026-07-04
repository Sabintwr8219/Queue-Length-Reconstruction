"""
Generate publication/output plots and tables for the Queue Length Reconstruction project.

Place this file at:
    src/generate_publication_outputs.py

Purpose
-------
This is a PLOTTING/TABLE script only.

It reads saved intermediate/evaluation outputs and creates selected figures/tables.
It does not rerun preprocessing, model training, evaluation, transformation, or
anchor adjustment.

Main plot groups
----------------
1. Trajectory time-space plot
2. Cumulative-count theory plot
3. Cumulative-count-space comparison plot
4. Full queue-length-space comparison plot
5. Cycle-wise queue-length-space comparison plot
6. Cycle peak queue plot and table
7. Method-family metric comparison plot and table
8. ML model metric comparison plot and table
9. Interpolation metric comparison plot and table
10. Event timing-shift plot and table

Important design
----------------
Use the MAKE_* switches in each plot section to create one figure/table or many
outputs in a single run.

Queue-length method-family plots read:
    output/intermediate_csv/method_family_queue_length_evaluation

Trajectory/theory/cumulative plots read the existing core pipeline outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import math
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =============================================================================
# 0. COMMON CONFIGURATION
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

INTERMEDIATE_DIR = PROJECT_ROOT / "output" / "intermediate_csv"

PREPROCESS_DIR = INTERMEDIATE_DIR / "preprocessing"
GT_DIR = INTERMEDIATE_DIR / "gt"
BASELINE_DIR = INTERMEDIATE_DIR / "baseline"
CV_FEATURE_DIR = INTERMEDIATE_DIR / "cv_features"
CUM_TRANSFORMED_DIR = INTERMEDIATE_DIR / "cumulative_transformed"
EVENT_TIMING_DIR = INTERMEDIATE_DIR / "evaluation_cumulative_event_timing"
METHOD_EVAL_DIR = INTERMEDIATE_DIR / "method_family_queue_length_evaluation"

FIGURES_DIR = PROJECT_ROOT / "output" / "final_plots" / "publication_outputs"
TABLES_DIR = PROJECT_ROOT / "output" / "tables" / "publication_outputs"

RUN_ID = 12
CV_RATE_PCT = 10
SPLIT_TO_PLOT = "test"

COLOR_MODE = "color"
# Options:
#   "color"
#   "black_white"

SHOW_GRID = True
SHOW_LEGEND = True
SHOW_FIGURES = False
FIGURE_DPI = 300

DEFAULT_FIGSIZE = (14, 6)
WIDE_FIGSIZE = (16, 5.8)
CYCLE_FIGSIZE = (16, 10)
METRIC_FIGSIZE = (10.8, 6)
TABLE_FONT_SIZE = 8.5
TABLE_HEADER_FONT_SIZE = 9.0

# Common time window applies only to selected explanation/cumulative plots.
# It does NOT apply to full queue-length comparison, because the full queue plot
# is intentionally full-run. Use cycle-wise plot for zoomed cycle views.
COMMON_TIME_WINDOW_SEC = None
# Example:
# COMMON_TIME_WINDOW_SEC = (1000, 1600) or None for full run

COMMON_Y_WINDOW = None
# Example:
# COMMON_Y_WINDOW = (0, 900)


# =============================================================================
# 1. TRAJECTORY TIME-SPACE PLOT CONFIG
# =============================================================================

MAKE_TRAJECTORY_PLOT = False

TRAJ_SAMPLE_MODE = "every_n_within_cycle"
# Options:
#   "all"
#   "every_n_vehicle"
#   "every_n_within_cycle"

TRAJ_EVERY_N = 5
TRAJ_MAX_VEHICLES_IF_ALL = 300
TRAJ_SHOW_STOPBAR_LINE = True
TRAJ_SHOW_UPSTREAM_DETECTOR_LINE = True
TRAJ_UPSTREAM_DETECTOR_FT = -1800.0
TRAJ_LINE_ALPHA = 0.45
TRAJ_LINEWIDTH = 0.65
TRAJ_USE_NEUTRAL_TRAJECTORY_COLOR = True

# =============================================================================
# 2. CUMULATIVE-COUNT THEORY PLOT CONFIG
# =============================================================================

MAKE_CUMULATIVE_THEORY_PLOT = False

CUM_THEORY_SELECTED_CURVES = ["A", "D", "V", "B"]
# Any subset allowed:
#   ["A"]
#   ["A", "D"]
#   ["A", "D", "V", "B"]


# =============================================================================
# 3. CUMULATIVE-COUNT-SPACE COMPARISON CONFIG
# =============================================================================

MAKE_CUMULATIVE_SPACE_PLOT = False

CUM_SPACE_SELECTED_CURVES = [
    "Ground Truth",
    "Cumulative Count Theory + GRU + CV",
]
# Available labels if corresponding columns exist:
#   "GT"
#   "Baseline"
#   "Cumulative Count Theory + GRU + CV"

CUM_SPACE_SHOW_CV_EVENT_POINTS = True


# =============================================================================
# 4. FULL QUEUE-LENGTH COMPARISON CONFIG
# =============================================================================

MAKE_QUEUE_FULL_PLOT = False

QUEUE_SELECTED_CURVES = [
    "GT",
    "Physics + ML + CV",
]
# Can select by family:
#   "GT"
#   "Physics baseline"
#   "CV-only"
#   "Physics + CV"
#   "ML-only"
#   "ML + CV"
#   "Physics + ML"
#   "Physics + ML + CV"
#
# Or by curve_id from method_family_curve_catalog.csv:
#   "cv_only_linear"
#   "ml_only_gru"
#   "physics_ml_cv_gru"
#   etc.

QUEUE_USE_BEST_VALIDATION_MODEL = True
QUEUE_ONE_CURVE_PER_FAMILY = True
QUEUE_SHOW_CV_ANCHORS = True
QUEUE_SHOW_PHASE_RIBBON = True


# =============================================================================
# 5. CYCLE-WISE QUEUE-LENGTH COMPARISON CONFIG
# =============================================================================

MAKE_QUEUE_CYCLEWISE_PLOT = False

CYCLE_SELECTION_MODE = "first_n"
# Options:
#   "first_n"
#   "specific"

MAX_CYCLES_TO_PLOT = 12
SELECTED_CYCLE_IDS = [1, 2, 3]

CYCLE_NCOLS = 3
SHARE_Y_CYCLE_PLOTS = False


# =============================================================================
# 6. CYCLE PEAK QUEUE PLOT/TABLE CONFIG
# =============================================================================

MAKE_CYCLE_PEAK_PLOT = False
MAKE_CYCLE_PEAK_TABLE = False

CYCLE_PEAK_VALUE_MODE = "peak"
# Options:
#   "peak"  -> plots/table of peak queue length
#   "error" -> plots/table of peak queue error

CYCLE_PEAK_SELECTED_CURVES = [
    "GT",
    "Physics baseline",
    "Physics + ML + CV",
]
# Internal names are used here.
# Display labels will be converted to:
#   GT                 -> Ground Truth
#   Physics baseline   -> Cumulative Count Theory
#   Physics + ML + CV  -> Cumulative Count Theory + GRU + CV

# =============================================================================
# 7. METHOD-FAMILY METRIC COMPARISON PLOT/TABLE CONFIG
# =============================================================================


MAKE_METHOD_FAMILY_METRIC_PLOT = True
MAKE_METHOD_FAMILY_METRIC_TABLE = True

METHOD_METRICS_TO_PLOT = [
    "rmse_ft",
    # "rmse_ft",
    "mae_ft",
    # "abc_ft_s",
]

METHOD_TABLE_RATE_MODE = "all_rates"
# Options:
#   "selected_rate"
#   "all_rates"

METHOD_TABLE_METRICS = [
    "mae_ft",
    "rmse_ft",
    "abc_ft_s",
    "mean_cycle_peak_error_ft",
]

METHOD_FAMILY_SELECTED_CURVES = [
    "Physics baseline",
    "ML-only",
    "ML + CV",
    "Physics + ML",
    "Physics + ML + CV",
]
# Optional, for supplemental/full comparison:
# METHOD_FAMILY_SELECTED_CURVES = [
#     "Physics baseline",
#     "CV-only",
#     "Physics + CV",
#     "ML-only",
#     "ML + CV",
#     "Physics + ML",
#     "Physics + ML + CV",
# ]

METHOD_TABLE_STYLE = "final_model_by_rate"
# Options:
#   "selected_methods"      -> regular method comparison table
#   "final_model_by_rate"   -> final model performance vs CV penetration rate

FINAL_MODEL_TABLE_FAMILY = "Physics + ML + CV"
FINAL_MODEL_TABLE_MODEL = "GRU"

# =============================================================================
# 8. ML MODEL METRIC COMPARISON PLOT/TABLE CONFIG
# =============================================================================
# Purpose:
#   Compare XGBoost, GRU, and LSTM within one selected model family.
#
# Typical use:
#   - Use this for model selection.
#   - It answers: which ML architecture performs best?
#
# Available MODEL_FAMILY_TO_PLOT values:
#   "ML-only"
#   "ML + CV"
#   "Physics + ML"
#   "Physics + ML + CV"
#
# Available metrics:
#   "mae_ft"
#   "rmse_ft"
#   "abc_ft_s"
#   "mean_cycle_peak_error_ft"
#   "median_cycle_peak_error_ft"
#   "rmse_cycle_peak_error_ft"
#
# Table:
#   The table is fixed to CV_RATE_PCT for compact comparison.

MAKE_ML_MODEL_METRIC_PLOT = True
MAKE_ML_MODEL_METRIC_TABLE = True

ML_MODEL_FAMILY_TO_PLOT = "Physics + ML + CV"

ML_MODEL_METRICS_TO_PLOT = [
    "rmse_ft",
    "mae_ft",
    "abc_ft_s",
    "mean_cycle_peak_error_ft",
]


# =============================================================================
# 9. INTERPOLATION METRIC COMPARISON PLOT/TABLE CONFIG
# =============================================================================
# Purpose:
#   Optional diagnostic comparison for interpolation choices.
#
# Typical use:
#   - Disabled in the current default workflow.
#   - X-axis = CV penetration rate.
#   - Y-axis = selected metric.
#   - One line per interpolation method.
#
# Best families to use here:
#   "CV-only"
#   "Physics + CV"
#
# Available interpolation methods:
#   "linear"
#
# Available metrics:
#   "mae_ft"
#   "rmse_ft"
#   "abc_ft_s"
#   "mean_cycle_peak_error_ft"
#   "median_cycle_peak_error_ft"
#   "rmse_cycle_peak_error_ft"
#
# Table:
#   The table is fixed to CV_RATE_PCT for compact comparison.

MAKE_INTERPOLATION_METRIC_PLOT = False
MAKE_INTERPOLATION_METRIC_TABLE = False

INTERPOLATION_FAMILY_TO_PLOT = "CV-only"
# Options:
#   "CV-only"
#   "Physics + CV"

INTERPOLATION_METHODS_TO_PLOT = [
    "linear",
]

INTERPOLATION_METRICS_TO_PLOT = [
    "rmse_ft",
    # "mae_ft",
    # "abc_ft_s",
    # "mean_cycle_peak_error_ft",
]


# =============================================================================
# 10. EVENT TIMING-SHIFT PLOT/TABLE CONFIG
# =============================================================================

MAKE_EVENT_SHIFT_PLOT = True
MAKE_EVENT_SHIFT_TABLE = True

EVENT_SHIFT_SELECTED_CURVES = [
    "Baseline B",
    "CCT + GRU + CV",
]
# Uses curve_label from event_timing_errors_all.csv:
#   "Baseline B"
#   "CCT + GRU + CV"

EVENT_SHIFT_VALUE_MODE = "error"
# Options:
#   "error" -> timing_error_sec
#   "abs_error" -> abs_timing_error_sec

EVENT_SHIFT_MAX_EVENTS_TO_PLOT = 3600

# =============================================================================
# 11. FINAL SELECTED MODEL IMPROVEMENT PLOT/TABLE CONFIG
# =============================================================================
# Purpose:
#   Show performance improvement of the final selected model over the
#   Cumulative Count Theory baseline across CV penetration rates.
#
# Improvement basis:
#   improvement (%) =
#       100 * (baseline_error - proposed_error) / baseline_error
#
# Baseline:
#   Physics baseline internally, displayed as Cumulative Count Theory
#
# Proposed:
#   Physics + ML + CV / GRU internally, displayed as
#   Cumulative Count Theory + GRU + CV

MAKE_FINAL_IMPROVEMENT_PLOT = True
MAKE_FINAL_IMPROVEMENT_TABLE = True

IMPROVEMENT_BASELINE_FAMILY = "Physics baseline"
IMPROVEMENT_PROPOSED_FAMILY = "Physics + ML + CV"
IMPROVEMENT_PROPOSED_MODEL = "GRU"

IMPROVEMENT_METRICS_TO_PLOT = [
    "rmse_ft",
]

IMPROVEMENT_TABLE_METRICS = [
    "mae_ft",
    "rmse_ft",
    "abc_ft_s",
    "mean_cycle_peak_error_ft",
]


# =============================================================================
# FILE PATHS
# =============================================================================

TRAJECTORY_FILE = PREPROCESS_DIR / "traj_nb_filtered.csv"

METHOD_PREDICTIONS_FILE = METHOD_EVAL_DIR / "method_family_predictions_allruns_allrates.csv"
METHOD_CATALOG_FILE = METHOD_EVAL_DIR / "method_family_curve_catalog.csv"
METHOD_BEST_FILE = METHOD_EVAL_DIR / "method_family_best_models_validation.csv"
METHOD_METRICS_FILE = METHOD_EVAL_DIR / "method_family_metrics_summary_by_split_rate_curve.csv"
METHOD_CYCLE_PEAKS_FILE = METHOD_EVAL_DIR / "method_family_plot_ready_cycle_peaks.csv"

EVENT_TIMING_ERRORS_FILE = EVENT_TIMING_DIR / "event_timing_errors_all.csv"
EVENT_TIMING_SUMMARY_FILE = EVENT_TIMING_DIR / "cumulative_event_timing_metrics_summary_by_rate_model.csv"

TIME_COL = "time_sec"
GT_Q_COL = "q_gt_ft"
PHASE_COL = "phase_state"


# =============================================================================
# STYLE
# =============================================================================

PHASE_COLORS = {
    "green": "#b7e4c7",
    "amber": "#ffd166",
    "yellow": "#ffd166",
    "red": "#ffadad",
    "unknown": "#d9d9d9",
    "nan": "#d9d9d9",
}

COLOR_BY_FAMILY = {
    "GT": "black",
    "Physics baseline": "#6f6f6f",
    "CV-only": "#1f77b4",
    "Physics + CV": "#8c564b",
    "ML-only": "#ff7f0e",
    "ML + CV": "#2ca02c",
    "Physics + ML": "#d62728",
    "Physics + ML + CV": "#9467bd",
}

BW_BY_FAMILY = {
    "GT": "black",
    "Physics baseline": "0.35",
    "CV-only": "0.05",
    "Physics + CV": "0.25",
    "ML-only": "0.50",
    "ML + CV": "0.15",
    "Physics + ML": "0.65",
    "Physics + ML + CV": "0.00",
}

LINESTYLE_BY_FAMILY = {
    "GT": "-",
    "Physics baseline": "--",
    "CV-only": "-.",
    "Physics + CV": ":",
    "ML-only": "-",
    "ML + CV": "-",
    "Physics + ML": "-",
    "Physics + ML + CV": "-",
}

MARKER_BY_FAMILY = {
    "GT": "o",
    "Physics baseline": "s",
    "CV-only": "^",
    "Physics + CV": "D",
    "ML-only": "v",
    "ML + CV": "P",
    "Physics + ML": "X",
    "Physics + ML + CV": "o",
}

FALLBACK_COLORS = [
    "black",
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:cyan",
]


# =============================================================================
# BASIC HELPERS
# =============================================================================

def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, label: str = "file") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}:\n{path}")
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def find_col(df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find {label}. Tried: {list(candidates)}")


def safe_filename(text: str) -> str:
    out = str(text).strip().lower()
    for ch in [" ", "+", "/", "\\", "(", ")", ",", "%", ":", ";", "|", ".", "·"]:
        out = out.replace(ch, "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")

def infer_model_name_from_row(row) -> str:
    """Infer model name from a catalog/metric row."""
    model = str(row.get("model_name", "")).strip()

    if model and model.lower() not in {"nan", "none", ""}:
        if model.lower() in {"xgb", "xgboost"}:
            return "XGBoost"
        if model.lower() in {"gru", "lstm"}:
            return model.upper()
        return model

    search_text = (
        str(row.get("curve_id", ""))
        + " "
        + str(row.get("curve_label", ""))
        + " "
        + str(row.get("prediction_col", ""))
    ).lower()

    if "xgb" in search_text or "xgboost" in search_text:
        return "XGBoost"
    if "lstm" in search_text:
        return "LSTM"
    if "gru" in search_text:
        return "GRU"

    return "GRU"


def display_family_label(method_family: str, model_name: str = "GRU", short: bool = False) -> str:
    """Convert internal method-family names to publication display labels."""
    family = str(method_family).strip()
    model = str(model_name).strip() or "GRU"

    if short:
        mapping = {
            "GT": "GT",
            "Physics baseline": "CCT",
            "CV-only": "CV",
            "Physics + CV": "CCT + CV",
            "ML-only": model,
            "ML + CV": f"{model} + CV",
            "Physics + ML": f"CCT + {model}",
            "Physics + ML + CV": f"CCT + {model} + CV",
        }
    else:
        mapping = {
            "GT": "Ground Truth",
            "Physics baseline": "Cumulative Count Theory",
            "CV-only": "CV",
            "Physics + CV": "Cumulative Count Theory + CV",
            "ML-only": model,
            "ML + CV": f"{model} + CV",
            "Physics + ML": f"Cumulative Count Theory + {model}",
            "Physics + ML + CV": f"Cumulative Count Theory + {model} + CV",
        }

    return mapping.get(family, family)


def display_curve_label_from_row(row, short: bool = False) -> str:
    """Final display label from a catalog or metrics row."""
    family = str(row.get("method_family", "")).strip()
    model = infer_model_name_from_row(row)

    if family:
        return display_family_label(family, model, short=short)

    label = str(row.get("curve_label", "")).strip()
    return display_curve_label_text(label, short=short)


def display_curve_label_text(label: str, short: bool = False) -> str:
    """Final display label for manually named curves."""
    text = str(label).strip()

    if short:
        mapping = {
            "GT": "GT",
            "Ground Truth": "GT",
            "Baseline": "CCT",
            "Baseline B": "CCT",
            "Physics baseline": "CCT",
            "Physics + ML + CV": "CCT + GRU + CV",
            "CCT + GRU + CV": "CCT + GRU + CV",
            "Cumulative Count Theory": "CCT",
            "Cumulative Count Theory + GRU + CV": "CCT + GRU + CV",
        }
    else:
        mapping = {
            "GT": "Ground Truth",
            "Ground Truth": "Ground Truth",
            "Baseline": "Cumulative Count Theory",
            "Baseline B": "Cumulative Count Theory",
            "Physics baseline": "Cumulative Count Theory",
            "Physics + ML + CV": "Cumulative Count Theory + GRU + CV",
            "CCT + GRU + CV": "Cumulative Count Theory + GRU + CV",
            "Cumulative Count Theory": "Cumulative Count Theory",
            "Cumulative Count Theory + GRU + CV": "Cumulative Count Theory + GRU + CV",
        }

    return mapping.get(text, text)

def apply_time_window(df: pd.DataFrame, time_col: str = "time_sec", window=None) -> pd.DataFrame:
    if window is None:
        return df.copy()
    t0, t1 = window
    return df[(df[time_col] >= float(t0)) & (df[time_col] <= float(t1))].copy()


def apply_y_window(ax, y_window=None) -> None:
    if y_window is not None:
        ax.set_ylim(float(y_window[0]), float(y_window[1]))


def style_axis(ax) -> None:
    ax.grid(SHOW_GRID, alpha=0.25)
    ax.tick_params(axis="both", labelsize=10)


def metric_axis_label(metric: str) -> str:
    labels = {
        "mae_ft": "MAE (ft)",
        "rmse_ft": "RMSE (ft)",
        "abc_ft_s": "ABC (ft·s)",
        "mean_cycle_peak_error_ft": "Mean cycle peak queue error (ft)",
        "median_cycle_peak_error_ft": "Median cycle peak queue error (ft)",
        "rmse_cycle_peak_error_ft": "RMSE cycle peak queue error (ft)",
    }
    return labels.get(metric, metric)


def metric_short_label(metric: str) -> str:
    labels = {
        "mae_ft": "MAE (ft)",
        "rmse_ft": "RMSE (ft)",
        "abc_ft_s": "ABC (ft·s)",
        "mean_cycle_peak_error_ft": "Cycle peak error (ft)",
        "median_cycle_peak_error_ft": "Median peak error (ft)",
        "rmse_cycle_peak_error_ft": "RMSE peak error (ft)",
    }
    return labels.get(metric, metric)


def format_metric_value(metric: str, value) -> str:
    try:
        x = float(value)
    except Exception:
        return ""
    if not np.isfinite(x):
        return ""
    if metric in {"abc_ft_s", "abc_count_sec"}:
        return f"{x:,.0f}"
    return f"{x:.2f}"


def save_figure(fig, filename: str) -> None:
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"[Saved figure] {out_path}")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def get_family_style(method_family: str, curve_label: str = "", idx: int = 0) -> dict:
    family = str(method_family)
    label = str(curve_label)

    if COLOR_MODE == "black_white":
        color = BW_BY_FAMILY.get(family, f"0.{min(8, 2 + idx)}")
    else:
        color = COLOR_BY_FAMILY.get(family, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])

    style = {
        "color": color,
        "linestyle": LINESTYLE_BY_FAMILY.get(family, "-"),
        "linewidth": 2.2,
        "marker": MARKER_BY_FAMILY.get(family, "o"),
    }

    if family == "GT" or label == "GT":
        style.update({"color": "black", "linewidth": 2.6, "linestyle": "-", "marker": "o"})

    if family == "Physics baseline":
        style.update({"linewidth": 2.2, "linestyle": "--"})

    if family == "CV-only":
        style.update({"linewidth": 2.2, "linestyle": "-."})

    return style


# =============================================================================
# METHOD-FAMILY LOADERS AND SELECTION
# =============================================================================

def load_method_predictions() -> pd.DataFrame:
    df = read_csv(METHOD_PREDICTIONS_FILE, "method-family predictions")
    require_columns(df, ["run_id", "cv_rate_pct", "time_sec", "q_gt_ft"], "method-family predictions")
    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce").astype("Int64")
    df["cv_rate_pct"] = pd.to_numeric(df["cv_rate_pct"], errors="coerce").astype("Int64")
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    df = df.dropna(subset=["run_id", "cv_rate_pct", "time_sec"]).copy()
    df["run_id"] = df["run_id"].astype(int)
    df["cv_rate_pct"] = df["cv_rate_pct"].astype(int)
    if "phase_state" in df.columns:
        df["phase_state"] = df["phase_state"].astype(str).str.lower().str.strip()
    else:
        df["phase_state"] = "unknown"
    return df.sort_values(["run_id", "cv_rate_pct", "time_sec"]).reset_index(drop=True)


def load_method_catalog() -> pd.DataFrame:
    catalog = read_csv(METHOD_CATALOG_FILE, "curve catalog")
    require_columns(
        catalog,
        ["curve_id", "method_family", "curve_label", "prediction_col", "is_reference"],
        "curve catalog",
    )
    return catalog


def load_best_validation() -> pd.DataFrame:
    if not METHOD_BEST_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(METHOD_BEST_FILE)


def load_method_metrics() -> pd.DataFrame:
    df = read_csv(METHOD_METRICS_FILE, "method-family metric summary")
    require_columns(df, ["ml_split", "cv_rate_pct", "curve_id", "method_family"], "method metrics")
    return df


def load_cycle_peaks() -> pd.DataFrame:
    df = read_csv(METHOD_CYCLE_PEAKS_FILE, "cycle peak table")
    require_columns(df, ["run_id", "cv_rate_pct", "cycle_id", "curve_id", "curve_label", "q_peak_ft"], "cycle peaks")
    return df


def resolve_selected_curves(
    catalog: pd.DataFrame,
    best: pd.DataFrame,
    selected: list[str],
    include_gt: bool = True,
) -> pd.DataFrame:
    rows = []

    c = catalog.copy()
    c["curve_id_l"] = c["curve_id"].astype(str).str.lower()
    c["curve_label_l"] = c["curve_label"].astype(str).str.lower()
    c["method_family_l"] = c["method_family"].astype(str).str.lower()

    best_ids_by_family = {}
    if not best.empty and {"method_family", "curve_id"}.issubset(best.columns):
        for _, r in best.iterrows():
            best_ids_by_family[str(r["method_family"]).lower()] = str(r["curve_id"])

    for item in selected:
        item_raw = str(item).strip()
        item_l = item_raw.lower()

        if item_l == "gt":
            if include_gt:
                gt = c[c["curve_id_l"] == "gt"].copy()
                if not gt.empty:
                    rows.append(gt)
            continue

        exact_id = c[c["curve_id_l"] == item_l].copy()
        if not exact_id.empty:
            rows.append(exact_id)
            continue

        exact_label = c[c["curve_label_l"] == item_l].copy()
        if not exact_label.empty:
            rows.append(exact_label)
            continue

        family_rows = c[c["method_family_l"] == item_l].copy()
        if family_rows.empty:
            print(f"[WARN] Could not resolve selected curve/family: {item_raw}")
            continue

        if QUEUE_USE_BEST_VALIDATION_MODEL and item_l in best_ids_by_family:
            best_id = best_ids_by_family[item_l]
            fam_best = family_rows[family_rows["curve_id"].astype(str) == best_id].copy()
            if not fam_best.empty:
                rows.append(fam_best)
                continue

        if QUEUE_ONE_CURVE_PER_FAMILY:
            nonref = family_rows[family_rows["is_reference"] == 0].copy()
            rows.append(nonref.head(1) if not nonref.empty else family_rows.head(1))
        else:
            rows.append(family_rows)

    if not rows:
        raise ValueError("No selected curves resolved. Check QUEUE_SELECTED_CURVES.")

    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates("curve_id").reset_index(drop=True)
    return out


def selected_prediction_cols(catalog_sel: pd.DataFrame) -> list[str]:
    cols = []
    for c in catalog_sel["prediction_col"].astype(str):
        if c not in cols:
            cols.append(c)
    return cols


def load_cv_anchors(run_id: int, rate: int) -> pd.DataFrame:
    path = CV_FEATURE_DIR / f"cv_anchors_run{int(run_id):03d}_rate{int(rate):03d}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["cv_anchor_time_sec", "cv_anchor_q_ft"])
    df = pd.read_csv(path)
    if "cv_anchor_time_sec" not in df.columns or "cv_anchor_q_ft" not in df.columns:
        return pd.DataFrame(columns=["cv_anchor_time_sec", "cv_anchor_q_ft"])
    df = df[["cv_anchor_time_sec", "cv_anchor_q_ft"]].copy()
    df["cv_anchor_time_sec"] = pd.to_numeric(df["cv_anchor_time_sec"], errors="coerce")
    df["cv_anchor_q_ft"] = pd.to_numeric(df["cv_anchor_q_ft"], errors="coerce")
    return df.dropna(subset=["cv_anchor_time_sec", "cv_anchor_q_ft"]).sort_values("cv_anchor_time_sec")


# =============================================================================
# PHASE AND CYCLE HELPERS
# =============================================================================

def phase_intervals_from_timegrid(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "phase_state" not in df.columns:
        return pd.DataFrame(columns=["start", "end", "state"])

    d = df[["time_sec", "phase_state"]].copy()
    d["time_sec"] = pd.to_numeric(d["time_sec"], errors="coerce")
    d["phase_state"] = d["phase_state"].astype(str).str.lower().str.strip()
    d = d.dropna(subset=["time_sec"]).sort_values("time_sec").drop_duplicates("time_sec", keep="last")

    if len(d) < 2:
        return pd.DataFrame(columns=["start", "end", "state"])

    d["changed"] = d["phase_state"].ne(d["phase_state"].shift()).astype(int)
    d["segment_id"] = d["changed"].cumsum()

    intervals = (
        d.groupby("segment_id", as_index=False)
        .agg(start=("time_sec", "first"), state=("phase_state", "first"))
        .sort_values("start")
        .reset_index(drop=True)
    )
    intervals["end"] = intervals["start"].shift(-1)

    dt = np.nanmedian(np.diff(d["time_sec"].to_numpy(dtype=float))) if len(d) > 2 else 0.1
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.1

    intervals["end"] = intervals["end"].fillna(float(d["time_sec"].max()) + dt)
    intervals = intervals[intervals["end"] > intervals["start"]].copy()
    return intervals[["start", "end", "state"]].reset_index(drop=True)


def red_to_red_cycle_windows(df: pd.DataFrame) -> pd.DataFrame:
    intervals = phase_intervals_from_timegrid(df)
    if intervals.empty:
        return pd.DataFrame(columns=["cycle_id", "cycle_start_sec", "cycle_end_sec"])

    intervals["prev_state"] = intervals["state"].shift()
    red_starts = intervals[
        (intervals["state"] == "red")
        & (intervals["prev_state"] != "red")
    ]["start"].to_numpy(dtype=float)

    red_starts = np.sort(red_starts[np.isfinite(red_starts)])
    if len(red_starts) < 2:
        return pd.DataFrame(columns=["cycle_id", "cycle_start_sec", "cycle_end_sec"])

    raw = [(float(red_starts[i]), float(red_starts[i + 1])) for i in range(len(red_starts) - 1)]

    # Drop first and last boundary/partial cycles.
    if len(raw) <= 2:
        return pd.DataFrame(columns=["cycle_id", "cycle_start_sec", "cycle_end_sec"])

    kept = raw[1:-1]
    return pd.DataFrame(
        [
            {"cycle_id": i, "cycle_start_sec": s, "cycle_end_sec": e}
            for i, (s, e) in enumerate(kept, start=1)
        ]
    )


def select_cycle_windows(cycles: pd.DataFrame) -> pd.DataFrame:
    if cycles.empty:
        return cycles
    mode = str(CYCLE_SELECTION_MODE).lower().strip()
    if mode == "specific":
        return cycles[cycles["cycle_id"].isin(SELECTED_CYCLE_IDS)].copy()
    if mode == "first_n":
        return cycles.head(int(MAX_CYCLES_TO_PLOT)).copy()
    raise ValueError("CYCLE_SELECTION_MODE must be 'first_n' or 'specific'.")


def add_phase_ribbon(ax, intervals: pd.DataFrame, t0: float, t1: float) -> list[Patch]:
    if not QUEUE_SHOW_PHASE_RIBBON or intervals.empty:
        return []

    y0 = -0.12
    y1 = -0.045
    trans = ax.get_xaxis_transform()
    handles = []
    seen = set()

    view = intervals[(intervals["end"] >= t0) & (intervals["start"] <= t1)].copy()

    for _, row in view.iterrows():
        start = max(float(row["start"]), float(t0))
        end = min(float(row["end"]), float(t1))
        state = str(row["state"]).lower().strip()
        color = PHASE_COLORS.get(state, PHASE_COLORS["unknown"])

        ax.axvspan(
            start,
            end,
            ymin=y0,
            ymax=y1,
            facecolor=color,
            edgecolor="none",
            alpha=0.75,
            transform=trans,
            clip_on=False,
            zorder=0,
        )

        norm = "amber" if state == "yellow" else state
        if norm in {"green", "amber", "red"} and norm not in seen:
            seen.add(norm)
            handles.append(Patch(facecolor=color, edgecolor="none", alpha=0.75, label=f"phase: {norm}"))

    return handles


# =============================================================================
# TABLE HELPERS
# =============================================================================


def make_method_metric_table_data(
    metric_rows: pd.DataFrame,
    metrics: list[str],
    include_rate_in_method: bool = False,
    short_labels: bool = True,
) -> pd.DataFrame:
    out = pd.DataFrame()

    labels = metric_rows.apply(
        lambda r: display_curve_label_from_row(r, short=short_labels),
        axis=1,
    )

    if include_rate_in_method:
        out["CV penetration rate (%)"] = pd.to_numeric(
            metric_rows["cv_rate_pct"],
            errors="coerce",
        ).astype("Int64")
        out["Method"] = labels.astype(str)
    else:
        out["Method"] = labels.astype(str)

    for metric in metrics:
        out[metric_short_label(metric)] = metric_rows[metric].apply(
            lambda x: format_metric_value(metric, x)
        )

    return out

def save_csv_table(df: pd.DataFrame, filename: str) -> None:
    path = TABLES_DIR / filename
    df.to_csv(path, index=False)
    print(f"[Saved table CSV] {path}")




def save_table_png(table_df: pd.DataFrame, filename: str, title: str) -> None:
    """Save readable PNG table with automatic sizing and wrapped text."""
    if table_df.empty:
        print(f"[WARN] Empty table skipped: {filename}")
        return

    display = table_df.copy()

    for col in display.columns:
        display[col] = display[col].apply(lambda x: "" if pd.isna(x) else str(x))

    # Wrap long text columns so they do not overlap or smear in PNG output.
    for col in display.columns:
        col_l = str(col).lower()
        if col_l in {"method", "curve", "model"}:
            display[col] = display[col].apply(
                lambda x: "\n".join(textwrap.wrap(str(x), width=24))
            )
        elif len(str(col)) > 16:
            display[col] = display[col].apply(
                lambda x: "\n".join(textwrap.wrap(str(x), width=18))
            )

    n_rows = len(display)
    n_cols = len(display.columns)
    fig_width = max(9.5, 1.75 * n_cols)
    fig_height = max(3.2, 0.62 * n_rows + 1.9)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=18)

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.88],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_FONT_SIZE)
    table.scale(1.0, 1.55)

    try:
        table.auto_set_column_width(col=list(range(n_cols)))
    except Exception:
        pass

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_text_props(weight="bold", fontsize=TABLE_HEADER_FONT_SIZE)
            cell.set_facecolor("#f0f0f0")
            cell.set_height(cell.get_height() * 1.25)
        if col == 0 and row > 0:
            cell.set_text_props(ha="left")
        if row > 0:
            cell.set_height(cell.get_height() * 1.25)

    fig.tight_layout()
    png_path = FIGURES_DIR / filename
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved table PNG] {png_path}")
# =============================================================================
# PLOT 1: TRAJECTORY
# =============================================================================

def load_trajectory_run(run_id: int) -> pd.DataFrame:
    if not TRAJECTORY_FILE.exists():
        raise FileNotFoundError(f"Missing trajectory file:\n{TRAJECTORY_FILE}")

    header = pd.read_csv(TRAJECTORY_FILE, nrows=0)
    cols = set(header.columns)

    required_base = {"run_id", "veh_uid", "Total_Sim_Time_Sec"}
    missing = required_base - cols
    if missing:
        raise ValueError(f"Trajectory file missing columns: {missing}")

    usecols = ["run_id", "veh_uid", "Total_Sim_Time_Sec"]
    if "vehID" in cols:
        usecols.append("vehID")
    if "s_rel_stop_ft" in cols:
        usecols.append("s_rel_stop_ft")
    elif "s_rel_stop_m" in cols:
        usecols.append("s_rel_stop_m")
    else:
        raise ValueError("Trajectory file needs s_rel_stop_ft or s_rel_stop_m.")

    chunks = []
    for chunk in pd.read_csv(TRAJECTORY_FILE, usecols=usecols, chunksize=1_000_000, low_memory=False):
        chunk["run_id"] = pd.to_numeric(chunk["run_id"], errors="coerce")
        chunk = chunk[chunk["run_id"] == int(run_id)].copy()
        if chunk.empty:
            continue
        chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No trajectory rows found for run {run_id:03d}")

    df = pd.concat(chunks, ignore_index=True)
    df["time_sec"] = pd.to_numeric(df["Total_Sim_Time_Sec"], errors="coerce")
    df["veh_uid"] = df["veh_uid"].astype(str).str.strip()

    if "s_rel_stop_ft" not in df.columns:
        df["s_rel_stop_ft"] = pd.to_numeric(df["s_rel_stop_m"], errors="coerce") * 3.28084
    else:
        df["s_rel_stop_ft"] = pd.to_numeric(df["s_rel_stop_ft"], errors="coerce")

    df = df.dropna(subset=["time_sec", "s_rel_stop_ft", "veh_uid"]).copy()
    return df.sort_values(["veh_uid", "time_sec"]).reset_index(drop=True)


def sample_trajectory_vehicles(df: pd.DataFrame, cycle_df: pd.DataFrame | None = None) -> list[str]:
    veh_first = (
        df.groupby("veh_uid", as_index=False)["time_sec"]
        .min()
        .sort_values("time_sec")
        .reset_index(drop=True)
    )

    mode = str(TRAJ_SAMPLE_MODE).lower().strip()

    if mode == "all":
        return veh_first["veh_uid"].head(TRAJ_MAX_VEHICLES_IF_ALL).astype(str).tolist()

    if mode == "every_n_vehicle":
        return veh_first.iloc[:: max(1, int(TRAJ_EVERY_N))]["veh_uid"].astype(str).tolist()

    if mode == "every_n_within_cycle" and cycle_df is not None and not cycle_df.empty:
        selected = []
        for _, cyc in cycle_df.iterrows():
            s = float(cyc["cycle_start_sec"])
            e = float(cyc["cycle_end_sec"])
            g = veh_first[(veh_first["time_sec"] >= s) & (veh_first["time_sec"] < e)].copy()
            selected.extend(g.iloc[:: max(1, int(TRAJ_EVERY_N))]["veh_uid"].astype(str).tolist())
        return sorted(set(selected), key=lambda x: veh_first.set_index("veh_uid").loc[x, "time_sec"])

    return veh_first.iloc[:: max(1, int(TRAJ_EVERY_N))]["veh_uid"].astype(str).tolist()



def make_trajectory_plot() -> None:
    df = load_trajectory_run(RUN_ID)
    df = apply_time_window(df, "time_sec", COMMON_TIME_WINDOW_SEC)

    cycles = pd.DataFrame()
    if METHOD_PREDICTIONS_FILE.exists():
        try:
            pred = load_method_predictions()
            one = pred[(pred["run_id"] == RUN_ID) & (pred["cv_rate_pct"] == CV_RATE_PCT)].copy()
            cycles = red_to_red_cycle_windows(one)
        except Exception as e:
            print(f"[WARN] Could not build cycles for trajectory sampling: {e}")

    selected_veh = sample_trajectory_vehicles(df, cycles)
    d = df[df["veh_uid"].isin(selected_veh)].copy()

    fig, ax = plt.subplots(figsize=WIDE_FIGSIZE)
    use_neutral = bool(globals().get("TRAJ_USE_NEUTRAL_TRAJECTORY_COLOR", True))

    if use_neutral:
        for _, g in d.groupby("veh_uid", sort=False):
            ax.plot(
                g["time_sec"],
                g["s_rel_stop_ft"],
                color="0.35",
                linewidth=TRAJ_LINEWIDTH,
                alpha=TRAJ_LINE_ALPHA,
            )
    elif COLOR_MODE == "color":
        unique = d["veh_uid"].drop_duplicates().tolist()
        color_map = {veh: FALLBACK_COLORS[i % len(FALLBACK_COLORS)] for i, veh in enumerate(unique)}
        for veh, g in d.groupby("veh_uid", sort=False):
            ax.plot(
                g["time_sec"],
                g["s_rel_stop_ft"],
                color=color_map[veh],
                linewidth=TRAJ_LINEWIDTH,
                alpha=TRAJ_LINE_ALPHA,
            )
    else:
        for _, g in d.groupby("veh_uid", sort=False):
            ax.plot(
                g["time_sec"],
                g["s_rel_stop_ft"],
                color="0.25",
                linewidth=TRAJ_LINEWIDTH,
                alpha=TRAJ_LINE_ALPHA,
            )

    if TRAJ_SHOW_STOPBAR_LINE:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.5, label="Stopbar")

    if TRAJ_SHOW_UPSTREAM_DETECTOR_LINE:
        ax.axhline(
            TRAJ_UPSTREAM_DETECTOR_FT,
            color="0.45",
            linestyle=":",
            linewidth=1.5,
            label="Upstream detector",
        )

    ax.set_title(f"Trajectory time-space diagram | Run {RUN_ID:03d}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Stopbar-relative distance (ft)")
    style_axis(ax)

    if SHOW_LEGEND:
        ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    save_figure(fig, f"plot1_trajectory_run{RUN_ID:03d}_{safe_filename(TRAJ_SAMPLE_MODE)}_{COLOR_MODE}.png")

# =============================================================================
# PLOT 2: CUMULATIVE THEORY
# =============================================================================

def load_cumulative_theory_run(run_id: int) -> pd.DataFrame:
    candidates = [
        PREPROCESS_DIR / "cumulative_curves_plot_ready_allruns.csv",
        INTERMEDIATE_DIR / "cumulative_count_theory" / "cumulative_curves_plot_ready_allruns.csv",
        BASELINE_DIR / f"baseline_queue_count_timegrid_run{int(run_id):03d}.csv",
        PREPROCESS_DIR / "vb_curves_all_runs.csv",
    ]

    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("Could not find cumulative-count theory input file.")

    df = read_csv(path, "cumulative-count theory input")
    require_columns(df, ["run_id"], "cumulative-count theory input")

    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce")
    df = df[df["run_id"] == int(run_id)].copy()
    if df.empty:
        raise ValueError(f"No cumulative-count theory rows found for run {run_id:03d}")

    # Long plot-ready format: run_id, curve_type, N, time_sec
    if {"curve_type", "N", "time_sec"}.issubset(df.columns):
        df["curve_type"] = df["curve_type"].astype(str).str.strip().str.upper()
        df["N"] = pd.to_numeric(df["N"], errors="coerce")
        df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
        df = df.dropna(subset=["curve_type", "N", "time_sec"]).copy()

        wide_parts = []
        for curve in ["A", "D", "V", "B"]:
            g = df[df["curve_type"] == curve][["time_sec", "N"]].copy()
            if g.empty:
                continue
            g = g.rename(columns={"N": curve})
            wide_parts.append(g)

        if not wide_parts:
            raise ValueError(f"No A/D/V/B curves found in long-format file: {path}")

        out = wide_parts[0]
        for part in wide_parts[1:]:
            out = out.merge(part, on="time_sec", how="outer")

        out["run_id"] = int(run_id)
        return out.sort_values("time_sec").reset_index(drop=True)

    # Wide time-grid format.
    time_col = find_col(df, ["time_sec", "sim_time_sec"], "cumulative theory time column")
    df = df.rename(columns={time_col: "time_sec"})
    rename_map = {}
    for old, new in [
        ("A_count", "A"),
        ("D_count", "D"),
        ("V_count", "V"),
        ("B_count", "B"),
        ("A", "A"),
        ("D", "D"),
        ("V", "V"),
        ("B", "B"),
    ]:
        if old in df.columns:
            rename_map[old] = new
    df = df.rename(columns=rename_map)
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    return df.dropna(subset=["time_sec"]).sort_values("time_sec").reset_index(drop=True)


def make_cumulative_theory_plot() -> None:
    """
    Plot cumulative-count theory curves A(t), D(t), V(t), and B(t).

    Final publication style:
    - continuous solid lines
    - different colors only
    - no markers
    - no dashed/dotted patterns
    """
    df = load_cumulative_theory_run(RUN_ID)
    df = apply_time_window(df, "time_sec", COMMON_TIME_WINDOW_SEC)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    label_map = {
        "A": "Arrival A(t)",
        "D": "Departure D(t)",
        "V": "Virtual arrival V(t)",
        "B": "Back-of-queue B(t)",
    }

    if COLOR_MODE == "black_white":
        colors = {
            "A": "black",
            "D": "0.35",
            "V": "0.55",
            "B": "0.15",
        }
    else:
        colors = {
            "A": "black",
            "D": "tab:blue",
            "V": "tab:orange",
            "B": "tab:green",
        }

    linewidths = {
        "A": 2.6,
        "D": 2.3,
        "V": 2.3,
        "B": 2.6,
    }

    selected = [str(c).upper() for c in CUM_THEORY_SELECTED_CURVES]
    plot_order = [c for c in ["A", "D", "V", "B"] if c in selected]

    for c in plot_order:
        if c not in df.columns:
            raise ValueError(f"Cumulative theory requested curve '{c}' but column is missing.")

        ax.plot(
            df["time_sec"],
            pd.to_numeric(df[c], errors="coerce"),
            label=label_map.get(c, c),
            color=colors[c],
            linewidth=linewidths[c],
            linestyle="-",
            alpha=0.95,
        )

    ax.set_title(f"Cumulative-count theory curves | Run {RUN_ID:03d}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative vehicle count")
    style_axis(ax)
    apply_y_window(ax, COMMON_Y_WINDOW)

    if SHOW_LEGEND:
        ax.legend(
            loc="upper left",
            frameon=True,
            fontsize=10,
            ncol=1,
            handlelength=2.6,
            borderpad=0.6,
            labelspacing=0.4,
        )

    fig.tight_layout()
    tag = "".join(plot_order)
    save_figure(fig, f"plot2_cumulative_theory_run{RUN_ID:03d}_{tag}_{COLOR_MODE}.png")

# =============================================================================
# PLOT 3: CUMULATIVE SPACE
# =============================================================================

def load_transformed_run_rate(run_id: int, rate: int) -> pd.DataFrame:
    path = CUM_TRANSFORMED_DIR / f"cumulative_B_transformed_run{int(run_id):03d}_rate{int(rate):03d}.csv"
    df = read_csv(path, "transformed cumulative B curve")
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    return df.dropna(subset=["time_sec"]).sort_values("time_sec").reset_index(drop=True)


def load_gt_cumulative_events(run_id: int) -> pd.DataFrame:
    path = GT_DIR / f"gt_cumulative_events_run{int(run_id):03d}.csv"
    df = read_csv(path, "GT cumulative events")
    t_col = find_col(df, ["t_event_sec"], "GT cumulative event time column")
    n_col = find_col(df, ["N_gt"], "GT cumulative event count column")
    df = df.rename(columns={t_col: "t_event_sec", n_col: "N_gt"})
    df["t_event_sec"] = pd.to_numeric(df["t_event_sec"], errors="coerce")
    df["N_gt"] = pd.to_numeric(df["N_gt"], errors="coerce")
    return df.dropna(subset=["t_event_sec", "N_gt"]).sort_values(["N_gt", "t_event_sec"]).reset_index(drop=True)


def gt_step_on_timegrid(time_sec: np.ndarray, gt_events: pd.DataFrame) -> np.ndarray:
    t = np.asarray(time_sec, dtype=float)
    event_times = np.sort(gt_events["t_event_sec"].to_numpy(dtype=float))
    return np.searchsorted(event_times, t, side="right").astype(float)


def make_cumulative_space_plot() -> None:
    df = load_transformed_run_rate(RUN_ID, CV_RATE_PCT)
    df = apply_time_window(df, "time_sec", COMMON_TIME_WINDOW_SEC)

    gt_events = load_gt_cumulative_events(RUN_ID)
    gt_events_win = apply_time_window(
        gt_events.rename(columns={"t_event_sec": "time_sec"}),
        "time_sec",
        COMMON_TIME_WINDOW_SEC,
    ).rename(columns={"time_sec": "t_event_sec"})

    curve_map = {
        "Cumulative Count Theory": "B_baseline_count",
        "Cumulative Count Theory + GRU + CV": "B_physics_ml_cv_gru_count",
    }

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    if any(display_curve_label_text(c) == "Ground Truth" for c in CUM_SPACE_SELECTED_CURVES):
        gt_y = gt_step_on_timegrid(df["time_sec"].to_numpy(dtype=float), gt_events)
        ax.step(
            df["time_sec"],
            gt_y,
            where="post",
            color="black",
            linewidth=2.4,
            label="Ground Truth",
        )

    for i, label in enumerate(CUM_SPACE_SELECTED_CURVES):
        if display_curve_label_text(label) == "Ground Truth":
            continue
        col = curve_map.get(label)
        if col is None:
            print(f"[WARN] Unknown cumulative-space curve label: {label}")
            continue
        if col not in df.columns:
            print(f"[WARN] Missing cumulative-space column: {col}")
            continue

        color = FALLBACK_COLORS[(i + 1) % len(FALLBACK_COLORS)] if COLOR_MODE == "color" else f"0.{2 + i}"
        ax.plot(df["time_sec"], pd.to_numeric(df[col], errors="coerce"), linewidth=2.2, label=label, color=color)

    if CUM_SPACE_SHOW_CV_EVENT_POINTS and not gt_events_win.empty:
        ax.scatter(gt_events_win["t_event_sec"], gt_events_win["N_gt"], s=14, c="black", alpha=0.35, label="GT events")

    ax.set_title(f"Cumulative-count space comparison | Run {RUN_ID:03d} | CV {CV_RATE_PCT}%")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative vehicle count")
    style_axis(ax)
    apply_y_window(ax, COMMON_Y_WINDOW)

    if SHOW_LEGEND:
        ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    save_figure(fig, f"plot3_cumulative_space_run{RUN_ID:03d}_rate{CV_RATE_PCT:03d}_{COLOR_MODE}.png")


# =============================================================================
# PLOTS 4-6: QUEUE LENGTH, CYCLEWISE, CYCLE PEAK
# =============================================================================

def plot_curve_set(ax, df: pd.DataFrame, catalog_sel: pd.DataFrame) -> None:
    t = df["time_sec"].to_numpy(dtype=float)

    for i, (_, row) in enumerate(catalog_sel.iterrows()):
        col = str(row["prediction_col"])
        if col not in df.columns:
            print(f"[WARN] Missing prediction column for plot: {col}")
            continue

        family = str(row["method_family"])
        label = display_curve_label_from_row(row)
        style = get_family_style(family, label, i)

        # Full time-series plots usually look cleaner without markers.
        style_no_marker = style.copy()
        style_no_marker.pop("marker", None)

        alpha = 0.82
        if family == "GT":
            alpha = 0.92
        elif family == "Physics baseline":
            alpha = 0.75

        ax.plot(
            t,
            pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float),
            label=label,
            alpha=alpha,
            zorder=3 + i,
            **style_no_marker,
        )


def set_queue_ylim(ax, df: pd.DataFrame, cols: list[str]) -> None:
    ymax = 0.0
    for c in cols:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            if np.any(np.isfinite(vals)):
                ymax = max(ymax, float(np.nanmax(vals)))
    ax.set_ylim(bottom=0, top=max(10.0, ymax * 1.08))


def add_cv_anchor_dots(ax, anchors: pd.DataFrame) -> None:
    if not QUEUE_SHOW_CV_ANCHORS or anchors.empty:
        return
    ax.scatter(
        anchors["cv_anchor_time_sec"],
        anchors["cv_anchor_q_ft"],
        s=42,
        c="green" if COLOR_MODE == "color" else "white",
        edgecolors="black",
        linewidths=0.5,
        label="CV anchors",
        zorder=20,
    )



def make_queue_full_plot() -> None:
    pred = load_method_predictions()
    catalog = load_method_catalog()
    best = load_best_validation()
    catalog_sel = resolve_selected_curves(catalog, best, QUEUE_SELECTED_CURVES, include_gt=True)

    df = pred[(pred["run_id"] == RUN_ID) & (pred["cv_rate_pct"] == CV_RATE_PCT)].copy()
    if df.empty:
        raise ValueError(f"No queue prediction rows for run {RUN_ID:03d}, CV {CV_RATE_PCT}%.")

    cols = selected_prediction_cols(catalog_sel)
    intervals = phase_intervals_from_timegrid(df)
    anchors = load_cv_anchors(RUN_ID, CV_RATE_PCT)

    fig, ax = plt.subplots(figsize=WIDE_FIGSIZE)

    plot_curve_set(ax, df, catalog_sel)
    add_cv_anchor_dots(ax, anchors)

    phase_handles = add_phase_ribbon(ax, intervals, float(df["time_sec"].min()), float(df["time_sec"].max()))

    ax.set_title(f"Queue length comparison | Run {RUN_ID:03d} | CV {CV_RATE_PCT}%")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Queue length (ft)")
    style_axis(ax)
    set_queue_ylim(ax, df, cols)

    if SHOW_LEGEND:
        handles, labels = ax.get_legend_handles_labels()
        handles += phase_handles
        labels += [h.get_label() for h in phase_handles]
        seen = set()
        final_h, final_l = [], []
        for h, lab in zip(handles, labels):
            if lab not in seen:
                final_h.append(h)
                final_l.append(lab)
                seen.add(lab)
        ax.legend(final_h, final_l, loc="best", fontsize=9, ncol=2, frameon=True)

    fig.tight_layout()
    save_figure(fig, f"plot4_queue_full_run{RUN_ID:03d}_rate{CV_RATE_PCT:03d}_{COLOR_MODE}.png")


def make_queue_cyclewise_plot() -> None:
    pred = load_method_predictions()
    catalog = load_method_catalog()
    best = load_best_validation()
    catalog_sel = resolve_selected_curves(catalog, best, QUEUE_SELECTED_CURVES, include_gt=True)

    df = pred[(pred["run_id"] == RUN_ID) & (pred["cv_rate_pct"] == CV_RATE_PCT)].copy()
    if df.empty:
        raise ValueError(f"No queue prediction rows for run {RUN_ID:03d}, CV {CV_RATE_PCT}%.")

    cycles = select_cycle_windows(red_to_red_cycle_windows(df))
    if cycles.empty:
        raise ValueError("No red-to-red cycles found for queue cycle-wise plot.")

    intervals = phase_intervals_from_timegrid(df)
    anchors = load_cv_anchors(RUN_ID, CV_RATE_PCT)
    cols = selected_prediction_cols(catalog_sel)

    n = len(cycles)
    ncols = int(CYCLE_NCOLS)
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=CYCLE_FIGSIZE,
        sharey=SHARE_Y_CYCLE_PLOTS,
        constrained_layout=True,
    )

    axes_flat = np.atleast_1d(axes).ravel()
    for ax in axes_flat:
        ax.axis("off")

    for i, (_, cyc) in enumerate(cycles.iterrows()):
        ax = axes_flat[i]
        ax.axis("on")

        t0 = float(cyc["cycle_start_sec"])
        t1 = float(cyc["cycle_end_sec"])
        cycle_id = int(cyc["cycle_id"])

        g = df[(df["time_sec"] >= t0) & (df["time_sec"] < t1)].copy()
        a = anchors[(anchors["cv_anchor_time_sec"] >= t0) & (anchors["cv_anchor_time_sec"] < t1)].copy()

        if g.empty:
            continue

        plot_curve_set(ax, g, catalog_sel)
        add_cv_anchor_dots(ax, a)
        add_phase_ribbon(ax, intervals, t0, t1)

        ax.set_title(f"Cycle {cycle_id}: {t0:.1f}-{t1:.1f}s", fontsize=9)
        style_axis(ax)
        ax.tick_params(axis="both", labelsize=8)

        if not SHARE_Y_CYCLE_PLOTS:
            set_queue_ylim(ax, g, cols)
        else:
            ax.set_ylim(bottom=0)

    legend_handles = []
    for i, (_, row) in enumerate(catalog_sel.iterrows()):
        family = str(row["method_family"])
        label = display_curve_label_from_row(row)
        legend_handles.append(Line2D([0], [0], label=label, **get_family_style(family, label, i)))

    if QUEUE_SHOW_CV_ANCHORS:
        legend_handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor="green", markeredgecolor="black", label="CV anchors")
        )

    fig.suptitle(f"Cycle-wise queue length | Run {RUN_ID:03d} | CV {CV_RATE_PCT}%", fontsize=14)
    fig.supxlabel("Time (s)")
    fig.supylabel("Queue length (ft)")

    if SHOW_LEGEND:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(4, len(legend_handles)),
            fontsize=9,
            frameon=True,
        )

    save_figure(fig, f"plot5_queue_cyclewise_run{RUN_ID:03d}_rate{CV_RATE_PCT:03d}_{CYCLE_SELECTION_MODE}_{COLOR_MODE}.png")


def make_cycle_peak_plot() -> None:
    peaks = load_cycle_peaks()
    catalog = load_method_catalog()
    best = load_best_validation()
    catalog_sel = resolve_selected_curves(catalog, best, CYCLE_PEAK_SELECTED_CURVES, include_gt=True)
    curve_ids = set(catalog_sel["curve_id"].astype(str))

    df = peaks[
        (pd.to_numeric(peaks["run_id"], errors="coerce") == RUN_ID)
        & (pd.to_numeric(peaks["cv_rate_pct"], errors="coerce") == CV_RATE_PCT)
        & peaks["curve_id"].astype(str).isin(curve_ids)
    ].copy()

    if CYCLE_SELECTION_MODE == "specific":
        df = df[df["cycle_id"].isin(SELECTED_CYCLE_IDS)].copy()
    else:
        keep = sorted(df["cycle_id"].dropna().unique())[: int(MAX_CYCLES_TO_PLOT)]
        df = df[df["cycle_id"].isin(keep)].copy()

    if df.empty:
        raise ValueError("No cycle peak rows after filtering.")

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    mode = str(CYCLE_PEAK_VALUE_MODE).lower().strip()
    if mode == "peak":
        y_col = "q_peak_ft"
        y_label = "Peak queue length (ft)"
        title = f"Cycle peak queue comparison | Run {RUN_ID:03d} | CV {CV_RATE_PCT}%"
    elif mode == "error":
        y_col = "peak_queue_error_ft"
        y_label = "Peak queue error (ft)"
        title = f"Cycle peak queue error | Run {RUN_ID:03d} | CV {CV_RATE_PCT}%"
    else:
        raise ValueError("CYCLE_PEAK_VALUE_MODE must be 'peak' or 'error'.")

    for i, (_, row) in enumerate(catalog_sel.iterrows()):
        cid = str(row["curve_id"])
        g = df[df["curve_id"].astype(str) == cid].copy()
        if g.empty:
            continue
        g = g.sort_values("cycle_id")
        family = str(row["method_family"])
        label = display_curve_label_from_row(row)
        style = get_family_style(family, label, i)
        ax.plot(
            g["cycle_id"].to_numpy(dtype=int),
            pd.to_numeric(g[y_col], errors="coerce").to_numpy(dtype=float),
            label=label,
            **style,
        )

    ax.set_title(title)
    ax.set_xlabel("Cycle number")
    ax.set_ylabel(y_label)
    style_axis(ax)

    if SHOW_LEGEND:
        ax.legend(loc="best", fontsize=9, ncol=2, frameon=True)

    fig.tight_layout()
    save_figure(fig, f"plot6_cycle_peak_{mode}_run{RUN_ID:03d}_rate{CV_RATE_PCT:03d}_{COLOR_MODE}.png")


def make_cycle_peak_table() -> None:
    peaks = load_cycle_peaks()
    catalog = load_method_catalog()
    best = load_best_validation()
    catalog_sel = resolve_selected_curves(catalog, best, CYCLE_PEAK_SELECTED_CURVES, include_gt=True)
    curve_ids = set(catalog_sel["curve_id"].astype(str))

    df = peaks[
        (pd.to_numeric(peaks["run_id"], errors="coerce") == RUN_ID)
        & (pd.to_numeric(peaks["cv_rate_pct"], errors="coerce") == CV_RATE_PCT)
        & peaks["curve_id"].astype(str).isin(curve_ids)
    ].copy()

    if CYCLE_SELECTION_MODE == "specific":
        df = df[df["cycle_id"].isin(SELECTED_CYCLE_IDS)].copy()
    else:
        keep = sorted(df["cycle_id"].dropna().unique())[: int(MAX_CYCLES_TO_PLOT)]
        df = df[df["cycle_id"].isin(keep)].copy()

    if df.empty:
        raise ValueError("No cycle peak rows for table after filtering.")

    mode = str(CYCLE_PEAK_VALUE_MODE).lower().strip()
    value_col = "q_peak_ft" if mode == "peak" else "peak_queue_error_ft"

    order = {
        str(r["curve_id"]): display_curve_label_from_row(r, short=True)
        for _, r in catalog_sel.iterrows()
    }

    pivot = (
        df.assign(curve_label_ordered=df["curve_id"].astype(str).map(order))
        .pivot_table(
            index="cycle_id",
            columns="curve_label_ordered",
            values=value_col,
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"cycle_id": "Cycle"})
    )

    ordered_cols = ["Cycle"] + [lab for lab in order.values() if lab in pivot.columns]
    pivot = pivot[ordered_cols].copy()

    for c in pivot.columns:
        if c != "Cycle":
            pivot[c] = pivot[c].apply(lambda x: "" if pd.isna(x) else f"{float(x):.1f}")

    csv_name = f"cycle_peak_table_{mode}_run{RUN_ID:03d}_rate{CV_RATE_PCT:03d}.csv"
    png_name = f"cycle_peak_table_{mode}_run{RUN_ID:03d}_rate{CV_RATE_PCT:03d}.png"
    save_csv_table(pivot, csv_name)
    title = f"Cycle peak queue {'comparison' if mode == 'peak' else 'error'} | Run {RUN_ID:03d} | CV {CV_RATE_PCT}%"
    save_table_png(pivot, png_name, title)

# =============================================================================
# METRIC PLOTS/TABLES
# =============================================================================

def filter_selected_metric_rows(metrics: pd.DataFrame, catalog_sel: pd.DataFrame, rate_mode: str) -> pd.DataFrame:
    curve_ids = set(catalog_sel["curve_id"].astype(str))
    df = metrics[
        metrics["ml_split"].astype(str).str.contains(str(SPLIT_TO_PLOT), na=False)
        & metrics["curve_id"].astype(str).isin(curve_ids)
    ].copy()

    if rate_mode == "selected_rate":
        df = df[pd.to_numeric(df["cv_rate_pct"], errors="coerce") == int(CV_RATE_PCT)].copy()

    order_map = {cid: i for i, cid in enumerate(catalog_sel["curve_id"].astype(str))}
    df["_curve_order"] = df["curve_id"].astype(str).map(order_map).fillna(999).astype(int)
    df = df.sort_values(["cv_rate_pct", "_curve_order"]).reset_index(drop=True)
    return df



def make_method_family_metric_plot(metric: str) -> None:
    metrics = load_method_metrics()
    catalog = load_method_catalog()
    best = load_best_validation()
    catalog_sel = resolve_selected_curves(
        catalog,
        best,
        METHOD_FAMILY_SELECTED_CURVES,
        include_gt=False,
    )

    if metric not in metrics.columns:
        raise ValueError(f"Metric not found: {metric}")

    curve_ids = set(catalog_sel["curve_id"].astype(str))
    df = metrics[
        metrics["ml_split"].astype(str).str.contains(SPLIT_TO_PLOT, na=False)
        & metrics["curve_id"].astype(str).isin(curve_ids)
    ].copy()

    if df.empty:
        raise ValueError("No method-family metric rows after filtering.")

    fig, ax = plt.subplots(figsize=METRIC_FIGSIZE)

    for i, (_, row) in enumerate(catalog_sel.iterrows()):
        cid = str(row["curve_id"])
        g = df[df["curve_id"].astype(str) == cid].copy()
        if g.empty:
            continue

        g = g.sort_values("cv_rate_pct")
        family = str(row["method_family"])
        label = display_curve_label_from_row(row)
        style = get_family_style(family, label, i)

        ax.plot(
            pd.to_numeric(g["cv_rate_pct"], errors="coerce"),
            pd.to_numeric(g[metric], errors="coerce"),
            label=label,
            **style,
        )

    ax.set_title(f"Queue reconstruction performance | {SPLIT_TO_PLOT} | {metric_short_label(metric)}")
    ax.set_xlabel("CV penetration rate (%)")
    ax.set_ylabel(metric_axis_label(metric))
    style_axis(ax)

    if SHOW_LEGEND:
        ax.legend(loc="best", fontsize=9, ncol=2, frameon=True)

    fig.tight_layout()
    save_figure(fig, f"plot7_method_family_{safe_filename(metric)}_{SPLIT_TO_PLOT}_{COLOR_MODE}.png")


def make_method_family_metric_table() -> None:
    metrics = load_method_metrics()
    catalog = load_method_catalog()
    best = load_best_validation()

    for metric in METHOD_TABLE_METRICS:
        if metric not in metrics.columns:
            raise ValueError(f"Metric not found for table: {metric}")

    table_style = str(globals().get("METHOD_TABLE_STYLE", "selected_methods")).lower().strip()
    rate_mode = str(METHOD_TABLE_RATE_MODE).lower().strip()

    if rate_mode not in {"selected_rate", "all_rates"}:
        raise ValueError("METHOD_TABLE_RATE_MODE must be 'selected_rate' or 'all_rates'.")

    if table_style == "final_model_by_rate":
        df = metrics[
            metrics["ml_split"].astype(str).str.contains(SPLIT_TO_PLOT, na=False)
            & (metrics["method_family"].astype(str) == FINAL_MODEL_TABLE_FAMILY)
        ].copy()

        if "model_name" in df.columns:
            df = df[
                df["model_name"].astype(str).str.lower()
                == str(FINAL_MODEL_TABLE_MODEL).lower()
            ].copy()
        else:
            search_text = (
                df["curve_id"].astype(str)
                + " "
                + df["curve_label"].astype(str)
            ).str.lower()
            df = df[search_text.str.contains(str(FINAL_MODEL_TABLE_MODEL).lower(), na=False)].copy()

        if df.empty:
            raise ValueError("No rows found for final model by CV penetration rate table.")

        df["cv_rate_pct"] = pd.to_numeric(df["cv_rate_pct"], errors="coerce")
        df = df.dropna(subset=["cv_rate_pct"]).copy()
        df = df.sort_values("cv_rate_pct").drop_duplicates("cv_rate_pct", keep="first")

        table_df = pd.DataFrame()
        table_df["CV penetration rate (%)"] = df["cv_rate_pct"].astype(int)

        for metric in METHOD_TABLE_METRICS:
            table_df[metric_short_label(metric)] = df[metric].apply(
                lambda x: format_metric_value(metric, x)
            )

        csv_name = f"final_model_performance_by_rate_{SPLIT_TO_PLOT}.csv"
        png_name = f"final_model_performance_by_rate_{SPLIT_TO_PLOT}.png"

        save_csv_table(table_df, csv_name)
        save_table_png(
            table_df,
            png_name,
            "Final model (CCT + GRU + CV) performance vs CV penetration rate",
        )
        return

    catalog_sel = resolve_selected_curves(
        catalog,
        best,
        METHOD_FAMILY_SELECTED_CURVES,
        include_gt=False,
    )

    df = filter_selected_metric_rows(metrics, catalog_sel, rate_mode)
    if df.empty:
        raise ValueError("No rows for method-family metric table.")

    include_rate = rate_mode == "all_rates"
    table_df = make_method_metric_table_data(
        df,
        METHOD_TABLE_METRICS,
        include_rate_in_method=include_rate,
        short_labels=True,
    )

    rate_tag = f"rate{CV_RATE_PCT:03d}" if rate_mode == "selected_rate" else "allrates"
    csv_name = f"method_family_table_{SPLIT_TO_PLOT}_{rate_tag}.csv"
    png_name = f"method_family_table_{SPLIT_TO_PLOT}_{rate_tag}.png"

    save_csv_table(table_df, csv_name)

    title_parts = ["Queue-length performance comparison", SPLIT_TO_PLOT]
    if rate_mode == "selected_rate":
        title_parts.append(f"CV {CV_RATE_PCT}%")
    else:
        title_parts.append("all CV rates")

    save_table_png(table_df, png_name, " | ".join(title_parts))

# =============================================================================
# FINAL SELECTED MODEL IMPROVEMENT PLOT/TABLE
# =============================================================================

def select_improvement_metric_rows(
    metrics: pd.DataFrame,
    method_family: str,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Select one metric row per CV rate for a method family/model."""
    df = metrics[
        metrics["ml_split"].astype(str).str.contains(SPLIT_TO_PLOT, na=False)
        & (metrics["method_family"].astype(str) == str(method_family))
    ].copy()

    if model_name is not None:
        model_l = str(model_name).lower()

        if "model_name" in df.columns:
            mask_model = df["model_name"].astype(str).str.lower().str.contains(model_l, na=False)
        else:
            search_text = (
                df.get("curve_id", "").astype(str)
                + " "
                + df.get("curve_label", "").astype(str)
            ).str.lower()
            mask_model = search_text.str.contains(model_l, na=False)

        df = df[mask_model].copy()

    if df.empty:
        raise ValueError(
            f"No metric rows found for family='{method_family}', model='{model_name}'."
        )

    df["cv_rate_pct"] = pd.to_numeric(df["cv_rate_pct"], errors="coerce")
    df = df.dropna(subset=["cv_rate_pct"]).copy()
    df["cv_rate_pct"] = df["cv_rate_pct"].astype(int)

    # Keep one row per CV rate. If duplicates exist, keep the first after sorting.
    sort_cols = ["cv_rate_pct"]
    if "curve_id" in df.columns:
        sort_cols.append("curve_id")

    df = df.sort_values(sort_cols).drop_duplicates("cv_rate_pct", keep="first")
    return df.reset_index(drop=True)


def compute_final_improvement_table(metrics_to_include: list[str]) -> pd.DataFrame:
    """Compute improvement of final selected model over Cumulative Count Theory."""
    metrics = load_method_metrics()

    baseline = select_improvement_metric_rows(
        metrics=metrics,
        method_family=IMPROVEMENT_BASELINE_FAMILY,
        model_name=None,
    )

    proposed = select_improvement_metric_rows(
        metrics=metrics,
        method_family=IMPROVEMENT_PROPOSED_FAMILY,
        model_name=IMPROVEMENT_PROPOSED_MODEL,
    )

    base_cols = ["cv_rate_pct"] + metrics_to_include
    prop_cols = ["cv_rate_pct"] + metrics_to_include

    base = baseline[base_cols].copy()
    prop = proposed[prop_cols].copy()

    merged = base.merge(
        prop,
        on="cv_rate_pct",
        how="inner",
        suffixes=("_baseline", "_proposed"),
    )

    if merged.empty:
        raise ValueError("No overlapping CV rates between baseline and proposed metric rows.")

    for metric in metrics_to_include:
        b_col = f"{metric}_baseline"
        p_col = f"{metric}_proposed"
        i_col = f"{metric}_improvement_pct"

        b = pd.to_numeric(merged[b_col], errors="coerce")
        p = pd.to_numeric(merged[p_col], errors="coerce")

        merged[i_col] = np.where(
            np.isfinite(b) & (b != 0),
            100.0 * (b - p) / b,
            np.nan,
        )

    merged = merged.sort_values("cv_rate_pct").reset_index(drop=True)
    return merged


def make_final_improvement_plot(metric: str) -> None:
    df = compute_final_improvement_table([metric])

    i_col = f"{metric}_improvement_pct"
    if i_col not in df.columns:
        raise ValueError(f"Improvement column not found: {i_col}")

    fig, ax = plt.subplots(figsize=METRIC_FIGSIZE)

    ax.plot(
        pd.to_numeric(df["cv_rate_pct"], errors="coerce"),
        pd.to_numeric(df[i_col], errors="coerce"),
        marker="o",
        linewidth=2.4,
        label="Cumulative Count Theory + GRU + CV",
    )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)

    ax.set_title(f"{metric_short_label(metric)} improvement over Cumulative Count Theory")
    ax.set_xlabel("CV penetration rate (%)")
    ax.set_ylabel(f"{metric_short_label(metric)} improvement (%)")
    style_axis(ax)

    if SHOW_LEGEND:
        ax.legend(loc="best", fontsize=9, frameon=True)

    fig.tight_layout()
    save_figure(
        fig,
        f"plot11_final_improvement_{safe_filename(metric)}_{SPLIT_TO_PLOT}_{COLOR_MODE}.png",
    )


def make_final_improvement_table() -> None:
    df = compute_final_improvement_table(IMPROVEMENT_TABLE_METRICS)

    out = pd.DataFrame()
    out["CV penetration (%)"] = df["cv_rate_pct"].astype(int)

    for metric in IMPROVEMENT_TABLE_METRICS:
        b_col = f"{metric}_baseline"
        p_col = f"{metric}_proposed"
        i_col = f"{metric}_improvement_pct"

        short = metric_short_label(metric)

        out[f"CCT {short}"] = df[b_col].apply(lambda x: format_metric_value(metric, x))
        out[f"CCT + GRU + CV {short}"] = df[p_col].apply(lambda x: format_metric_value(metric, x))
        out[f"{short} improvement (%)"] = df[i_col].apply(
            lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
        )

    csv_name = f"final_model_improvement_table_{SPLIT_TO_PLOT}.csv"
    png_name = f"final_model_improvement_table_{SPLIT_TO_PLOT}.png"

    save_csv_table(out, csv_name)
    save_table_png(
        out,
        png_name,
        f"Final model improvement over Cumulative Count Theory | {SPLIT_TO_PLOT}",
    )    
    



def make_ml_model_metric_plot(metric: str) -> None:
    metrics = load_method_metrics()
    if metric not in metrics.columns:
        raise ValueError(f"Metric not found: {metric}")

    df = metrics[
        metrics["ml_split"].astype(str).str.contains(SPLIT_TO_PLOT, na=False)
        & (metrics["method_family"].astype(str) == ML_MODEL_FAMILY_TO_PLOT)
    ].copy()

    if df.empty:
        raise ValueError(f"No rows for ML model family: {ML_MODEL_FAMILY_TO_PLOT}")

    fig, ax = plt.subplots(figsize=METRIC_FIGSIZE)

    for i, (model_name, g) in enumerate(df.groupby("model_name", sort=True)):
        g = g.sort_values("cv_rate_pct")
        style = {
            "color": FALLBACK_COLORS[i % len(FALLBACK_COLORS)] if COLOR_MODE == "color" else f"0.{2 + i}",
            "linestyle": "-",
            "linewidth": 2.2,
            "marker": "o",
        }
        ax.plot(g["cv_rate_pct"], pd.to_numeric(g[metric], errors="coerce"), label=str(model_name), **style)

    ax.set_title(
        f"ML model comparison | {display_family_label(ML_MODEL_FAMILY_TO_PLOT, 'GRU')} | {SPLIT_TO_PLOT}"
    )
    ax.set_xlabel("CV penetration rate (%)")
    ax.set_ylabel(metric_axis_label(metric))
    style_axis(ax)

    if SHOW_LEGEND:
        ax.legend(loc="best", fontsize=9, frameon=True)

    fig.tight_layout()
    save_figure(fig, f"plot8_ml_model_{safe_filename(ML_MODEL_FAMILY_TO_PLOT)}_{safe_filename(metric)}_{SPLIT_TO_PLOT}.png")


def make_ml_model_metric_table() -> None:
    metrics = load_method_metrics()
    df = metrics[
        metrics["ml_split"].astype(str).str.contains(SPLIT_TO_PLOT, na=False)
        & (metrics["method_family"].astype(str) == ML_MODEL_FAMILY_TO_PLOT)
        & (pd.to_numeric(metrics["cv_rate_pct"], errors="coerce") == int(CV_RATE_PCT))
    ].copy()

    if df.empty:
        raise ValueError("No rows for ML model metric table.")

    out = pd.DataFrame()
    out["Method"] = df.apply(lambda r: display_curve_label_from_row(r, short=True), axis=1).astype(str)
    for metric in ML_MODEL_METRICS_TO_PLOT:
        out[metric_short_label(metric)] = df[metric].apply(lambda x: format_metric_value(metric, x))

    csv_name = f"ml_model_table_{safe_filename(ML_MODEL_FAMILY_TO_PLOT)}_{SPLIT_TO_PLOT}_rate{CV_RATE_PCT:03d}.csv"
    png_name = f"ml_model_table_{safe_filename(ML_MODEL_FAMILY_TO_PLOT)}_{SPLIT_TO_PLOT}_rate{CV_RATE_PCT:03d}.png"
    save_csv_table(out, csv_name)
    save_table_png(out, png_name, f"ML model comparison | {display_family_label(ML_MODEL_FAMILY_TO_PLOT, 'GRU')} | CV {CV_RATE_PCT}%")

def make_interpolation_metric_plot(metric: str) -> None:
    metrics = load_method_metrics()
    if metric not in metrics.columns:
        raise ValueError(f"Metric not found: {metric}")

    df = metrics[
        metrics["ml_split"].astype(str).str.contains(SPLIT_TO_PLOT, na=False)
        & (metrics["method_family"].astype(str) == INTERPOLATION_FAMILY_TO_PLOT)
        & metrics["interpolation_method"].astype(str).isin(INTERPOLATION_METHODS_TO_PLOT)
    ].copy()

    if df.empty:
        raise ValueError("No rows for interpolation metric plot.")

    fig, ax = plt.subplots(figsize=METRIC_FIGSIZE)

    for i, (method, g) in enumerate(df.groupby("interpolation_method", sort=True)):
        g = g.sort_values("cv_rate_pct")
        style = {
            "color": FALLBACK_COLORS[i % len(FALLBACK_COLORS)] if COLOR_MODE == "color" else f"0.{2 + i}",
            "linestyle": "-" if str(method).lower() == "linear" else "--",
            "linewidth": 2.2,
            "marker": "o",
        }
        ax.plot(g["cv_rate_pct"], pd.to_numeric(g[metric], errors="coerce"), label=str(method), **style)

    ax.set_title(f"Interpolation comparison | {INTERPOLATION_FAMILY_TO_PLOT} | {SPLIT_TO_PLOT}")
    ax.set_xlabel("CV penetration rate (%)")
    ax.set_ylabel(metric_axis_label(metric))
    style_axis(ax)

    if SHOW_LEGEND:
        ax.legend(loc="best", fontsize=9, frameon=True)

    fig.tight_layout()
    save_figure(fig, f"plot9_interpolation_{safe_filename(INTERPOLATION_FAMILY_TO_PLOT)}_{safe_filename(metric)}_{SPLIT_TO_PLOT}.png")


def make_interpolation_metric_table() -> None:
    metrics = load_method_metrics()
    df = metrics[
        metrics["ml_split"].astype(str).str.contains(SPLIT_TO_PLOT, na=False)
        & (metrics["method_family"].astype(str) == INTERPOLATION_FAMILY_TO_PLOT)
        & metrics["interpolation_method"].astype(str).isin(INTERPOLATION_METHODS_TO_PLOT)
        & (pd.to_numeric(metrics["cv_rate_pct"], errors="coerce") == int(CV_RATE_PCT))
    ].copy()

    if df.empty:
        raise ValueError("No rows for interpolation metric table.")

    out = pd.DataFrame()
    out["Method"] = df.apply(display_curve_label_from_row, axis=1).astype(str)
    for metric in INTERPOLATION_METRICS_TO_PLOT:
        out[metric_short_label(metric)] = df[metric].apply(lambda x: format_metric_value(metric, x))

    csv_name = f"interpolation_table_{safe_filename(INTERPOLATION_FAMILY_TO_PLOT)}_{SPLIT_TO_PLOT}_rate{CV_RATE_PCT:03d}.csv"
    png_name = f"interpolation_table_{safe_filename(INTERPOLATION_FAMILY_TO_PLOT)}_{SPLIT_TO_PLOT}_rate{CV_RATE_PCT:03d}.png"
    save_csv_table(out, csv_name)
    save_table_png(out, png_name, f"Interpolation comparison | {INTERPOLATION_FAMILY_TO_PLOT} | CV {CV_RATE_PCT}%")


# =============================================================================
# EVENT SHIFT
# =============================================================================

def load_event_timing_errors() -> pd.DataFrame:
    df = read_csv(EVENT_TIMING_ERRORS_FILE, "event timing errors")
    require_columns(
        df,
        ["run_id", "cv_rate_pct", "curve_label", "N_gt", "timing_error_sec", "abs_timing_error_sec"],
        "event timing errors",
    )
    return df



def make_event_shift_plot() -> None:
    df = load_event_timing_errors()
    df = df[
        (pd.to_numeric(df["run_id"], errors="coerce") == int(RUN_ID))
        & (pd.to_numeric(df["cv_rate_pct"], errors="coerce") == int(CV_RATE_PCT))
        & df["curve_label"].astype(str).isin(EVENT_SHIFT_SELECTED_CURVES)
    ].copy()

    if df.empty:
        raise ValueError("No event timing rows after filtering.")

    value_col = "timing_error_sec" if EVENT_SHIFT_VALUE_MODE == "error" else "abs_timing_error_sec"

    max_events = int(EVENT_SHIFT_MAX_EVENTS_TO_PLOT)
    if max_events > 0:
        keep_n = sorted(df["N_gt"].dropna().unique())[:max_events]
        df = df[df["N_gt"].isin(keep_n)].copy()

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    for i, (label, g) in enumerate(df.groupby("curve_label", sort=False)):
        g = g.sort_values("N_gt")
        color = FALLBACK_COLORS[i % len(FALLBACK_COLORS)] if COLOR_MODE == "color" else f"0.{2 + i}"
        ax.plot(
            pd.to_numeric(g["N_gt"], errors="coerce"),
            pd.to_numeric(g[value_col], errors="coerce"),
            label=display_curve_label_text(str(label)),
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3.5,
        )

    ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title(f"Event timing shift | Run {RUN_ID:03d} | CV {CV_RATE_PCT}%")
    ax.set_xlabel("GT event number")
    ax.set_ylabel("Timing shift (s)" if EVENT_SHIFT_VALUE_MODE == "error" else "Absolute timing shift (s)")
    style_axis(ax)

    if SHOW_LEGEND:
        ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    save_figure(fig, f"plot10_event_shift_{EVENT_SHIFT_VALUE_MODE}_run{RUN_ID:03d}_rate{CV_RATE_PCT:03d}.png")


def make_event_shift_table() -> None:
    summary = read_csv(EVENT_TIMING_SUMMARY_FILE, "event timing summary")
    required = ["cv_rate_pct", "curve_label", "mae_sec_avg", "rmse_sec_avg", "bias_sec_avg"]
    require_columns(summary, required, "event timing summary")

    df = summary[
        (pd.to_numeric(summary["cv_rate_pct"], errors="coerce") == int(CV_RATE_PCT))
        & summary["curve_label"].astype(str).isin(EVENT_SHIFT_SELECTED_CURVES)
    ].copy()

    if df.empty:
        raise ValueError("No rows for event timing summary table.")

    out = pd.DataFrame()
    out["Method"] = df["curve_label"].astype(str).apply(lambda x: display_curve_label_text(x, short=True))
    out["MAE shift (s)"] = df["mae_sec_avg"].apply(lambda x: format_metric_value("mae", x))
    out["RMSE shift (s)"] = df["rmse_sec_avg"].apply(lambda x: format_metric_value("rmse", x))
    out["Bias shift (s)"] = df["bias_sec_avg"].apply(lambda x: format_metric_value("bias", x))

    csv_name = f"event_shift_table_rate{CV_RATE_PCT:03d}.csv"
    png_name = f"event_shift_table_rate{CV_RATE_PCT:03d}.png"
    save_csv_table(out, csv_name)
    save_table_png(out, png_name, f"Event timing shift summary | CV {CV_RATE_PCT}%")

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_dirs()

    print("=" * 96)
    print("Generate publication/output figures and tables")
    print("=" * 96)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Figures dir  : {FIGURES_DIR}")
    print(f"Tables dir   : {TABLES_DIR}")
    print(f"Run / CV     : {RUN_ID:03d} / {CV_RATE_PCT}%")
    print(f"Split        : {SPLIT_TO_PLOT}")
    print(f"Color mode   : {COLOR_MODE}")
    print("=" * 96)

    if MAKE_TRAJECTORY_PLOT:
        print("\n[1] Trajectory time-space plot")
        make_trajectory_plot()

    if MAKE_CUMULATIVE_THEORY_PLOT:
        print("\n[2] Cumulative-count theory plot")
        make_cumulative_theory_plot()

    if MAKE_CUMULATIVE_SPACE_PLOT:
        print("\n[3] Cumulative-count-space comparison plot")
        make_cumulative_space_plot()

    if MAKE_QUEUE_FULL_PLOT:
        print("\n[4] Full queue-length comparison plot")
        make_queue_full_plot()

    if MAKE_QUEUE_CYCLEWISE_PLOT:
        print("\n[5] Cycle-wise queue-length comparison plot")
        make_queue_cyclewise_plot()

    if MAKE_CYCLE_PEAK_PLOT:
        print("\n[6] Cycle peak queue plot")
        make_cycle_peak_plot()

    if MAKE_CYCLE_PEAK_TABLE:
        print("\n[6T] Cycle peak queue table")
        make_cycle_peak_table()

    if MAKE_METHOD_FAMILY_METRIC_PLOT:
        for metric in METHOD_METRICS_TO_PLOT:
            print(f"\n[7] Method-family metric plot: {metric}")
            make_method_family_metric_plot(metric)

    if MAKE_METHOD_FAMILY_METRIC_TABLE:
        print("\n[7T] Method-family metric table")
        make_method_family_metric_table()

    if MAKE_ML_MODEL_METRIC_PLOT:
        for metric in ML_MODEL_METRICS_TO_PLOT:
            print(f"\n[8] ML model metric plot: {metric}")
            make_ml_model_metric_plot(metric)

    if MAKE_ML_MODEL_METRIC_TABLE:
        print("\n[8T] ML model metric table")
        make_ml_model_metric_table()

    if MAKE_INTERPOLATION_METRIC_PLOT:
        for metric in INTERPOLATION_METRICS_TO_PLOT:
            print(f"\n[9] Interpolation metric plot: {metric}")
            make_interpolation_metric_plot(metric)

    if MAKE_INTERPOLATION_METRIC_TABLE:
        print("\n[9T] Interpolation metric table")
        make_interpolation_metric_table()

    if MAKE_EVENT_SHIFT_PLOT:
        print("\n[10] Event timing-shift plot")
        make_event_shift_plot()

    if MAKE_EVENT_SHIFT_TABLE:
        print("\n[10T] Event timing-shift table")
        make_event_shift_table()
        
    if MAKE_FINAL_IMPROVEMENT_PLOT:
        for metric in IMPROVEMENT_METRICS_TO_PLOT:
            print(f"\n[11] Final selected model improvement plot: {metric}")
            make_final_improvement_plot(metric)

    if MAKE_FINAL_IMPROVEMENT_TABLE:
        print("\n[11T] Final selected model improvement table")
        make_final_improvement_table()

    print("\nDone.")


if __name__ == "__main__":
    main()
