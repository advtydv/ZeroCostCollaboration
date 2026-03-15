#!/usr/bin/env python3
"""
Run the paper experiment matrix across two simulation codebases.

Codebase routing:
- Regular experiments: information_asymmetry_simulation
- Automated modes only: information_asymmetry_simulation_automated
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parent
REGULAR_SIM_ROOT = "information_asymmetry_simulation"
AUTOMATED_SIM_ROOT = "information_asymmetry_simulation_automated"
REGULAR_CONFIG = "information_asymmetry_simulation/config.yaml"
AUTO_REQUEST_CONFIG = "information_asymmetry_simulation_automated/config_automated_request.yaml"
AUTO_FULFILL_CONFIG = "information_asymmetry_simulation_automated/config_automated_fulfill.yaml"

MODEL_SHORTCUTS = {
    "gpt54": "gpt-5.4-2026-03-05",
    "gpt5_4": "gpt-5.4-2026-03-05",
    "gpt-5.4": "gpt-5.4-2026-03-05",
    "claudeopus46": "claude-opus-4-6",
    "opus46": "claude-opus-4-6",
    "claude-opus-4.6": "claude-opus-4-6",
}

HETERO_MODEL_REQUIREMENTS = ["o3", "claude-sonnet-4-20250514"]


@dataclass
class ExperimentSpec:
    name: str
    command: List[str]


def run_validation(models: List[str]) -> bool:
    cmd = [
        sys.executable,
        "validate_setup.py",
        "--skip-test",
        "--require-models",
        *models,
    ]
    result = subprocess.run(cmd, capture_output=False, cwd=str(REPO_ROOT))
    return result.returncode == 0


def build_specs(models: List[str], runs: int) -> List[ExperimentSpec]:
    specs: List[ExperimentSpec] = []

    for model in models:
        # Regular baseline
        specs.append(
            ExperimentSpec(
                name=f"Regular Baseline ({model})",
                command=[
                    "run_agent_types_experiments.py",
                    "--neutral", "10",
                    "--model", model,
                    "--runs", str(runs),
                    "--config", REGULAR_CONFIG,
                    "--simulation-root", REGULAR_SIM_ROOT,
                    "--output-dir", "paper_regular_baseline",
                ],
            )
        )

        # Automated request
        specs.append(
            ExperimentSpec(
                name=f"Automated Request ({model})",
                command=[
                    "run_agent_types_experiments.py",
                    "--neutral", "10",
                    "--model", model,
                    "--runs", str(runs),
                    "--config", AUTO_REQUEST_CONFIG,
                    "--simulation-root", AUTOMATED_SIM_ROOT,
                    "--output-dir", "paper_auto_request",
                ],
            )
        )

        # Automated fulfill (auto baseline)
        specs.append(
            ExperimentSpec(
                name=f"Automated Fulfill ({model})",
                command=[
                    "run_agent_types_experiments.py",
                    "--neutral", "10",
                    "--model", model,
                    "--runs", str(runs),
                    "--config", AUTO_FULFILL_CONFIG,
                    "--simulation-root", AUTOMATED_SIM_ROOT,
                    "--output-dir", "paper_auto_fulfill",
                ],
            )
        )

        # Policy intervention
        specs.append(
            ExperimentSpec(
                name=f"Policy Intervention ({model})",
                command=[
                    "run_agent_types_experiments.py",
                    "--policy", "10",
                    "--model", model,
                    "--runs", str(runs),
                    "--config", REGULAR_CONFIG,
                    "--simulation-root", REGULAR_SIM_ROOT,
                    "--output-dir", "paper_intervention_policy",
                ],
            )
        )

        # Incentive intervention
        specs.append(
            ExperimentSpec(
                name=f"Incentive Intervention ({model})",
                command=[
                    "run_agent_types_experiments.py",
                    "--neutral", "10",
                    "--model", model,
                    "--runs", str(runs),
                    "--config", REGULAR_CONFIG,
                    "--simulation-root", REGULAR_SIM_ROOT,
                    "--sharing-incentive", "1000",
                    "--output-dir", "paper_intervention_incentive",
                ],
            )
        )

        # Visibility intervention
        specs.append(
            ExperimentSpec(
                name=f"Visibility Intervention ({model})",
                command=[
                    "run_agent_types_experiments.py",
                    "--neutral", "10",
                    "--model", model,
                    "--runs", str(runs),
                    "--config", REGULAR_CONFIG,
                    "--simulation-root", REGULAR_SIM_ROOT,
                    "--revenue", "limited",
                    "--output-dir", "paper_intervention_visibility",
                ],
            )
        )

    # Heterogeneous runs (regular codebase)
    specs.extend(
        [
            ExperimentSpec(
                name="Heterogeneous (5 O3, 5 Claude Sonnet)",
                command=[
                    "run_heterogeneous_experiments.py",
                    "-o3", "5",
                    "-claude", "5",
                    "--runs", str(runs),
                    "--config", REGULAR_CONFIG,
                    "--simulation-root", REGULAR_SIM_ROOT,
                    "--output-dir", "paper_heterogeneous",
                ],
            ),
            ExperimentSpec(
                name="Heterogeneous (1 O3, 9 Claude Sonnet)",
                command=[
                    "run_heterogeneous_experiments.py",
                    "-claude", "9",
                    "-o3", "1",
                    "--runs", str(runs),
                    "--config", REGULAR_CONFIG,
                    "--simulation-root", REGULAR_SIM_ROOT,
                    "--output-dir", "paper_heterogeneous",
                ],
            ),
            ExperimentSpec(
                name="Heterogeneous (9 O3, 1 Claude Sonnet)",
                command=[
                    "run_heterogeneous_experiments.py",
                    "-claude", "1",
                    "-o3", "9",
                    "--runs", str(runs),
                    "--config", REGULAR_CONFIG,
                    "--simulation-root", REGULAR_SIM_ROOT,
                    "--output-dir", "paper_heterogeneous",
                ],
            ),
        ]
    )

    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full paper experiment matrix")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["claudeopus46", "gpt54"],
        help="Model shortcuts or full model IDs for baseline/automation/intervention runs",
    )
    parser.add_argument("--runs", type=int, default=5, help="Runs per experiment (default: 5)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validate_setup preflight")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue if a command fails")
    args = parser.parse_args()

    models = [MODEL_SHORTCUTS.get(m, m) for m in args.models]
    specs = build_specs(models, args.runs)

    print("\n" + "=" * 70)
    print("PAPER EXPERIMENT MATRIX")
    print("=" * 70)
    print(f"Models (baseline/automation/interventions): {', '.join(models)}")
    print(f"Runs per experiment: {args.runs}")
    print(f"Total experiment commands: {len(specs)}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN]")
        for idx, spec in enumerate(specs, start=1):
            print(f"\n{idx}. {spec.name}")
            print(f"   {sys.executable} {' '.join(spec.command)}")
        return 0

    if not args.skip_validation:
        required_models = sorted(set(models + HETERO_MODEL_REQUIREMENTS))
        if not run_validation(required_models):
            print("\nValidation failed. Fix issues or re-run with --skip-validation.")
            return 1

    start = time.time()
    failures = 0

    for idx, spec in enumerate(specs, start=1):
        print("\n" + "-" * 70)
        print(f"[{idx}/{len(specs)}] {spec.name}")
        print("-" * 70)

        cmd = [sys.executable, *spec.command]
        result = subprocess.run(cmd, capture_output=False, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            failures += 1
            print(f"FAILED ({result.returncode}): {spec.name}")
            if not args.continue_on_error:
                break

    duration = time.time() - start
    print("\n" + "=" * 70)
    print("PAPER EXPERIMENT MATRIX SUMMARY")
    print("=" * 70)
    print(f"Duration: {duration:.1f}s")
    print(f"Failed commands: {failures}")
    print("Results root: experiments/")
    print("=" * 70)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
