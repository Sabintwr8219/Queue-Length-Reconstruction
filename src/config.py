"""Central configuration for the Queue Length Reconstruction project.

This file is the shared source of project paths, run splits, CV rates,
traffic-flow constants, queue-detection thresholds, and model settings. Keep
stage-specific switches inside the stage script, but keep shared assumptions here.
"""

from __future__ import annotations

from pathlib import Path


# =============================================================================
# Repository paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
SAMPLE_DATA_DIR = DATA_DIR / "sample_data"
VISSIM_RESULTS_DIR = RAW_DATA_DIR / "vissim_results"

OUTPUT_DIR = PROJECT_ROOT / "output"
INTERMEDIATE_CSV_DIR = OUTPUT_DIR / "intermediate_csv"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
FINAL_PLOTS_DIR = OUTPUT_DIR / "final_plots"
MODELS_DIR = OUTPUT_DIR / "models"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
DOCS_DIR = PROJECT_ROOT / "docs"

# Canonical intermediate output folders.
PROCESSED_DATA_DIR = INTERMEDIATE_CSV_DIR / "preprocessing"
CUMULATIVE_COUNT_THEORY_DIR = INTERMEDIATE_CSV_DIR / "cumulative_count_theory"
GT_DIR = INTERMEDIATE_CSV_DIR / "gt"
BASELINE_DIR = INTERMEDIATE_CSV_DIR / "baseline"
CV_FEATURES_DIR = INTERMEDIATE_CSV_DIR / "cv_features"
ML_RAW_PREDICTIONS_DIR = INTERMEDIATE_CSV_DIR / "ml_raw_predictions"
ML_DIRECT_PREDICTIONS_DIR = INTERMEDIATE_CSV_DIR / "ml_direct_predictions"
ML_RESIDUAL_CV_PREDICTIONS_DIR = INTERMEDIATE_CSV_DIR / "ml_residual_cv_predictions"
METHOD_FAMILY_EVALUATION_DIR = INTERMEDIATE_CSV_DIR / "method_family_queue_length_evaluation"
CUMULATIVE_TRANSFORMED_DIR = INTERMEDIATE_CSV_DIR / "cumulative_transformed"
EVALUATION_EVENT_TIMING_DIR = INTERMEDIATE_CSV_DIR / "evaluation_cumulative_event_timing"
SUPPLEMENTAL_ROBUSTNESS_ABLATION_DIR = INTERMEDIATE_CSV_DIR / "supplemental_robustness_ablation"

# Diagnostic-only folders.
ML_ONLY_SEQUENCE_DIAGNOSTIC_DIR = INTERMEDIATE_CSV_DIR / "ml_only_sequence_diagnostics"
RESIDUAL_SEQUENCE_DIAGNOSTIC_DIR = INTERMEDIATE_CSV_DIR / "residual_sequence_diagnostics"

# Backward-compatible aliases retained for older scripts that are still being
# cleaned. Remove only after all scripts import the canonical names above.
RESULTS_DIR = TABLES_DIR
DOC_DIR = DOCS_DIR
CV_ANCHOR_ADJUSTED_DIR = INTERMEDIATE_CSV_DIR / "cv_anchor_adjusted"
EVALUATION_QUEUE_LENGTH_DIR = INTERMEDIATE_CSV_DIR / "evaluation_queue_length"
EVALUATION_CUMULATIVE_COUNT_DIR = INTERMEDIATE_CSV_DIR / "evaluation_cumulative_count"
ML_ONLY_PREDICTIONS_DIR = INTERMEDIATE_CSV_DIR / "ml_only_predictions"
MODEL_SELECTION_DIR = INTERMEDIATE_CSV_DIR / "model_selection"
METHOD_FAMILY_DIR = METHOD_FAMILY_EVALUATION_DIR
CV_ANCHOR_DIAGNOSTICS_DIR = INTERMEDIATE_CSV_DIR / "cv_anchor_diagnostics"
ABLATION_DIR = INTERMEDIATE_CSV_DIR / "ablation"
ROBUSTNESS_DIR = INTERMEDIATE_CSV_DIR / "robustness"
CORRECTED_CURVES_DIR = INTERMEDIATE_CSV_DIR / "corrected_curves"


# =============================================================================
# VISSIM input settings
# =============================================================================

VISSIM_BASE_NAME = "Project"
CHUNKSIZE_FZP = 200_000
CHUNKSIZE_TRAJ = 1_000_000


# =============================================================================
# Simulation runs and CV rates
# =============================================================================

RUN_IDS = list(range(5, 15))
TRAIN_RUN_IDS = list(range(5, 11))
VALIDATION_RUN_IDS = [11]
TEST_RUN_IDS = list(range(12, 15))

CV_RATES_PCT = [1, 2, 5, 10, 20, 50, 100]
RANDOM_SEED = 42


# =============================================================================
# Common file paths
# =============================================================================

MASTER_TRAJECTORY_CSV = PROCESSED_DATA_DIR / "master_trajectory.csv"
MASTER_PHASE_CSV = PROCESSED_DATA_DIR / "master_phase_time.csv"
VEHICLE_FILTER_SUMMARY_CSV = TABLES_DIR / "veh_summary.csv"
TRAJ_NB_FILTERED_CSV = PROCESSED_DATA_DIR / "traj_nb_filtered.csv"

AD_TIMES_CSV = PROCESSED_DATA_DIR / "ad_times_all_runs.csv"
VB_CURVES_CSV = PROCESSED_DATA_DIR / "vb_curves_all_runs.csv"
BASELINE_B_JOIN_CSV = PROCESSED_DATA_DIR / "baseline_b_join_all_runs.csv"
CUMULATIVE_CURVES_PLOT_READY_CSV = PROCESSED_DATA_DIR / "cumulative_curves_plot_ready_allruns.csv"
GT_CURVE_PLOT_READY_CSV = PROCESSED_DATA_DIR / "gt_curve_plot_ready_allruns.csv"

# Legacy/event-model paths retained until older analysis scripts are fully moved
# to legacy.
GT_JOIN_PATTERN = PROCESSED_DATA_DIR / "gt_queue_join_run{run_id:03d}.csv"
GT_QUEUE_LENGTH_PATTERN = PROCESSED_DATA_DIR / "gt_queue_length_run{run_id:03d}_0p1s.csv"
CV_ALLOC_RUN_PATTERN = PROCESSED_DATA_DIR / "cv_allocation_run{run_id:03d}_rate{rate:03d}.csv"
CV_ALLOC_ALLRUNS_PATTERN = PROCESSED_DATA_DIR / "cv_allocation_allruns_rate{rate:03d}.csv"
TIMEGRID_FEATURES_PATTERN = PROCESSED_DATA_DIR / "timegrid_vehicle_features_allruns_rate{rate:03d}.csv"
EVENT_FEATURES_ALLRATES_CSV = PROCESSED_DATA_DIR / "event_features_allrates.csv"
EVENT_PREDICTIONS_RAW_ALLRATES_CSV = PROCESSED_DATA_DIR / "event_predictions_raw_allrates.csv"
SEGMENTED_PRED_PATTERN = CORRECTED_CURVES_DIR / "event_predictions_segmented_allruns_rate{rate:03d}.csv"
RAW_MODEL_METRICS_CSV = TABLES_DIR / "raw_model_metrics.csv"
EVALUATION_BY_RUN_RATE_CSV = TABLES_DIR / "evaluation_by_run_and_rate.csv"
EVALUATION_SUMMARY_BY_RATE_CSV = TABLES_DIR / "evaluation_summary_by_rate.csv"
EVALUATION_SUMMARY_BY_SPLIT_CSV = TABLES_DIR / "evaluation_summary_by_split.csv"
XGB_MODEL_FILE = MODELS_DIR / "xgb_event_residual_model.joblib"


# =============================================================================
# Corridor geometry and detector settings
# =============================================================================

C1_LAT = 32.749832768069375
C1_LON = -97.09732266524595
C2_LAT = 32.75554890701484
C2_LON = -97.09727179559502

CRS_WGS84 = "EPSG:4326"
CRS_PROJECTED = "EPSG:32614"

FT_TO_M = 0.3048
M_TO_FT = 3.280839895
MPH_PER_FPS = 1.0 / 1.4666666667

UPSTREAM_DETECTOR_OFFSET_FT = 20.0
STOPBAR_OFFSET_FT = 20.0
UPSTREAM_DETECTOR_OFFSET_M = UPSTREAM_DETECTOR_OFFSET_FT * FT_TO_M
STOPBAR_OFFSET_M = STOPBAR_OFFSET_FT * FT_TO_M

DETECTOR_SPACING_FT = 1800.0
Y_STOPBAR_FT = 0.0
Y_UPSTREAM_DETECTOR_FT = -1800.0
Y_STOPBAR_M = Y_STOPBAR_FT * FT_TO_M
Y_UPSTREAM_DETECTOR_M = Y_UPSTREAM_DETECTOR_FT * FT_TO_M

MAX_PERP_DIST_M = 25.0
BUFFER_BEFORE_M = 0.0
BUFFER_AFTER_M = 0.0
DIR_LOOKBACK_SEC = 1.0
STOPBAR_STORE_BAND_M = 120.0


# =============================================================================
# Ground-truth queue detection settings
# =============================================================================

TIMEGRID_DT_SEC = 0.10
ASOF_TOL_SEC = 0.11

CORRIDOR_MIN_FT = -1800.0
CORRIDOR_MAX_FT = 0.0
STOPBAR_FT = 0.0

V_STOP_FPS = 5.0
STOP_PERSIST_SEC = 3.0

CREEP_DROP_FPS = 10.0
CREEP_LOOKBACK_SEC = 1.0
CREEP_PERSIST_SEC = 2.0

USE_NEIGHBOR_SUPPORT_FOR_CREEP = True
NEIGHBOR_TIME_TOL_SEC = 1.0
NEIGHBOR_MAX_GAP_FT = 80.0
NEIGHBOR_LOW_SPEED_FPS = 12.0
NEIGHBOR_DROP_FPS = 8.0

JOIN_PERSIST_SEC = 1.0
BACKTRACK_MAX_SEC = 4.0
ONSET_DROP_EPS_FPS = 2.0
GROW_TOL_FT = 1.0
DEFAULT_L_EFF_FT_PER_VEH = 25.0

SPACING_MIN_FT = 5.0
SPACING_MAX_FT = 120.0
SPACING_FALLBACK_FT = 25.0
DISCHARGE_ONLY_DURING_GREEN = True


# =============================================================================
# Cumulative-count theory settings
# =============================================================================

VF_FPS = 40.584
VQ_FPS = 11.947
VF_MPH = VF_FPS * MPH_PER_FPS
VQ_MPH = VQ_FPS * MPH_PER_FPS
T_FF_SEC = DETECTOR_SPACING_FT / VF_FPS
BETA = 1.0 - (VQ_FPS / VF_FPS)


# =============================================================================
# Model settings
# =============================================================================

XGB_RANDOM_STATE = RANDOM_SEED
XGB_RANDOM_SEED = RANDOM_SEED
XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.04
XGB_SUBSAMPLE = 0.85
XGB_COLSAMPLE_BYTREE = 0.85
XGB_REG_LAMBDA = 2.0
XGB_OBJECTIVE = "reg:squarederror"

NN_RANDOM_SEED = RANDOM_SEED
NN_SEQUENCE_LEN = 256
NN_SEQUENCE_STRIDE = 128
NN_HIDDEN_SIZE = 64
NN_NUM_LAYERS = 2
NN_DROPOUT = 0.10
NN_BATCH_SIZE = 64
NN_EPOCHS = 35
NN_LEARNING_RATE = 1e-3
NN_WEIGHT_DECAY = 1e-5
NN_GRAD_CLIP_NORM = 1.0


# =============================================================================
# Cumulative transformation and event-timing evaluation
# =============================================================================

L_EFF_FALLBACK_FT = 25.0
CLIP_Q_NONNEGATIVE = True
BOUND_B_BETWEEN_D_AND_A = True
FORCE_B_MONOTONE = True
ROUND_TIME_DECIMALS = 3

USE_LINEAR_CROSSING_INTERPOLATION = True
DROP_UNREACHED_EVENTS = True


# =============================================================================
# Utility
# =============================================================================

def ensure_project_directories() -> None:
    """Create the standard directory tree used by the active workflow."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        SAMPLE_DATA_DIR,
        VISSIM_RESULTS_DIR,
        OUTPUT_DIR,
        INTERMEDIATE_CSV_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        FINAL_PLOTS_DIR,
        MODELS_DIR,
        DIAGNOSTICS_DIR,
        DOCS_DIR,
        PROCESSED_DATA_DIR,
        CUMULATIVE_COUNT_THEORY_DIR,
        GT_DIR,
        BASELINE_DIR,
        CV_FEATURES_DIR,
        ML_RAW_PREDICTIONS_DIR,
        ML_DIRECT_PREDICTIONS_DIR,
        ML_RESIDUAL_CV_PREDICTIONS_DIR,
        METHOD_FAMILY_EVALUATION_DIR,
        CUMULATIVE_TRANSFORMED_DIR,
        EVALUATION_EVENT_TIMING_DIR,
        SUPPLEMENTAL_ROBUSTNESS_ABLATION_DIR,
        ML_ONLY_SEQUENCE_DIAGNOSTIC_DIR,
        RESIDUAL_SEQUENCE_DIAGNOSTIC_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
