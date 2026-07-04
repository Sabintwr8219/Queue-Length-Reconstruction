"""Run the active Queue Length Reconstruction workflow.

The core pipeline is intentionally narrow: it creates the current method-family
outputs from raw/processed inputs. Publication plots, supplemental robustness
tests, and standalone diagnostics are callable, but they are not part of the
default core run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from config import PROJECT_ROOT, ensure_project_directories


SRC_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PipelineStep:
    key: str
    script: str
    group: str
    description: str


CORE_STEPS: list[PipelineStep] = [
    PipelineStep("preprocess", "preprocess_vissim.py", "data", "Parse and filter VISSIM trajectories/signals."),
    PipelineStep("gt", "extract_ground_truth.py", "data", "Extract GT queue events and queue-length profiles."),
    PipelineStep("cct", "build_cumulative_count_theory.py", "physics", "Build cumulative-count theory curves."),
    PipelineStep("baseline", "build_cumulative_baseline.py", "physics", "Build CCT queue-length baseline/time-grid features."),
    PipelineStep("cv", "build_cv_features.py", "features", "Build CV anchor/context features for all CV rates."),
    PipelineStep("ml_direct", "train_ml_direct_models.py", "models", "Train ML-only and ML + CV direct models."),
    PipelineStep("cct_ml", "train_residual_models.py", "models", "Train CCT + ML residual models."),
    PipelineStep("cct_ml_cv", "train_residual_cv_models.py", "models", "Train CCT + ML + CV residual models."),
    PipelineStep("family_eval", "evaluate_method_family_queue_length.py", "evaluation", "Evaluate method families and select validation-best models."),
    PipelineStep("cumulative_transform", "transform_to_cumulative.py", "evaluation", "Transform selected queue curves back to cumulative-count space."),
    PipelineStep("event_timing", "evaluate_event_timing.py", "evaluation", "Evaluate cumulative-count event timing shifts."),
]

OPTIONAL_STEPS: list[PipelineStep] = [
    PipelineStep("plots", "plot_method_family_queue_length.py", "outputs", "Generate selected publication figures/tables."),
    PipelineStep("supplemental", "run_supplemental_robustness_ablation.py", "supplemental", "Run robustness and ablation outputs."),
]


def all_steps(include_plots: bool = False, include_supplemental: bool = False) -> list[PipelineStep]:
    steps = list(CORE_STEPS)
    if include_plots:
        steps.append(OPTIONAL_STEPS[0])
    if include_supplemental:
        steps.append(OPTIONAL_STEPS[1])
    return steps


def print_steps(steps: list[PipelineStep]) -> None:
    width = max(len(step.key) for step in steps)
    for idx, step in enumerate(steps, start=1):
        print(f"{idx:02d}. {step.key:<{width}}  [{step.group}]  {step.script}")
        print(f"    {step.description}")


def select_steps(
    steps: list[PipelineStep],
    only: list[str] | None = None,
    start_at: str | None = None,
    stop_after: str | None = None,
) -> list[PipelineStep]:
    by_key = {step.key: step for step in steps}

    if only:
        unknown = [key for key in only if key not in by_key]
        if unknown:
            raise ValueError(f"Unknown --only step(s): {unknown}")
        return [by_key[key] for key in only]

    start_idx = 0
    stop_idx = len(steps) - 1

    if start_at:
        keys = [step.key for step in steps]
        if start_at not in keys:
            raise ValueError(f"Unknown --start-at step: {start_at}")
        start_idx = keys.index(start_at)

    if stop_after:
        keys = [step.key for step in steps]
        if stop_after not in keys:
            raise ValueError(f"Unknown --stop-after step: {stop_after}")
        stop_idx = keys.index(stop_after)

    if stop_idx < start_idx:
        raise ValueError("--stop-after occurs before --start-at")

    return steps[start_idx : stop_idx + 1]


def validate_step_files(steps: list[PipelineStep]) -> None:
    missing = [step.script for step in steps if not (SRC_DIR / step.script).exists()]
    if missing:
        formatted = "\n".join(f"  - {name}" for name in missing)
        raise FileNotFoundError(
            "Pipeline scripts are missing from src/:\n"
            f"{formatted}\n\n"
            "Add or move these scripts into the active src/ folder before running this pipeline."
        )


def run_step(step: PipelineStep) -> None:
    """Run one pipeline script with the active Python interpreter."""
    script_path = SRC_DIR / step.script
    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline step not found: {script_path}")

    print("=" * 88)
    print(f"Running [{step.group}]: {step.key} -> {step.script}")
    print("=" * 88)
    subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List pipeline steps and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected steps without running them.")
    parser.add_argument("--include-plots", action="store_true", help="Run publication plot/table generation after core evaluation.")
    parser.add_argument("--include-supplemental", action="store_true", help="Run supplemental robustness/ablation after core evaluation.")
    parser.add_argument("--only", nargs="+", help="Run only the listed step keys, in the order provided.")
    parser.add_argument("--start-at", help="Start at this step key.")
    parser.add_argument("--stop-after", help="Stop after this step key.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Create folders and execute all current core stages in order."""
    args = build_parser().parse_args(argv)
    steps = all_steps(
        include_plots=args.include_plots,
        include_supplemental=args.include_supplemental,
    )

    if args.list:
        print_steps(steps)
        return

    selected_steps = select_steps(
        steps,
        only=args.only,
        start_at=args.start_at,
        stop_after=args.stop_after,
    )

    ensure_project_directories()
    validate_step_files(selected_steps)

    print("=" * 88)
    print("Queue Length Reconstruction pipeline")
    print("=" * 88)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Source dir  : {SRC_DIR}")
    print("Selected steps:")
    print_steps(selected_steps)
    print("=" * 88)

    if args.dry_run:
        print("Dry run only. No scripts executed.")
        return

    for step in selected_steps:
        run_step(step)

    print("=" * 88)
    print("Selected pipeline stages completed.")
    print("=" * 88)


if __name__ == "__main__":
    main()
