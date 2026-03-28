#!/usr/bin/env python3
"""
Run the current zero-cost environment experiment matrix.

This runner is designed for collaborator handoff:
- validates required API-key setup first (including AWS Secrets Manager refs)
- runs 3 seeds per model in parallel by default
- stores outputs under experiments/<experiment_name>/<model>/run_XXX
- writes per-run launcher logs and top-level summary artifacts
- supports resume/skip of already-completed runs
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "zero_cost_transfer_simulation" / "config.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments"

MODEL_SHORTCUTS = {
    "o3mini": "o3-mini-2025-01-31",
    "o3-mini": "o3-mini-2025-01-31",
    "claude": "claude-sonnet-4-20250514",
    "claudesonnet": "claude-sonnet-4-20250514",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "o3": "o3",
}

SUPPORTED_MODELS = {
    "o3-mini-2025-01-31",
    "claude-sonnet-4-20250514",
    "o3",
}


@dataclass(frozen=True)
class RunSpec:
    model: str
    model_short: str
    seed: int
    run_id: str
    model_dir: Path
    run_dir: Path
    launcher_log: Path
    cmd: List[str]
    env: Dict[str, str]


def resolve_models(model_tokens: Iterable[str]) -> List[str]:
    resolved: List[str] = []
    for token in model_tokens:
        model = MODEL_SHORTCUTS.get(token.lower(), token)
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {token}")
        resolved.append(model)
    return resolved


def short_model_name(model: str) -> str:
    if model == "o3-mini-2025-01-31":
        return "o3mini"
    if model == "claude-sonnet-4-20250514":
        return "claude_sonnet4"
    if model == "o3":
        return "o3"
    return model.replace("/", "_").replace("-", "_")


def run_validation(models: List[str]) -> bool:
    cmd = [
        sys.executable,
        "validate_setup.py",
        "--skip-test",
        "--require-models",
        *models,
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=False)
    return result.returncode == 0


def build_specs(
    models: List[str],
    runs: int,
    config_path: Path,
    experiment_dir: Path,
    log_level: str,
) -> List[RunSpec]:
    specs: List[RunSpec] = []
    for model in models:
        model_short = short_model_name(model)
        model_dir = experiment_dir / model_short
        for seed in range(1, runs + 1):
            run_id = f"run_{seed:03d}"
            run_dir = model_dir / run_id
            launcher_log = run_dir / "launcher.log"
            cmd = [
                sys.executable,
                "zero_cost_transfer_simulation/main.py",
                "--config",
                str(config_path),
                "--model",
                model,
                "--seed",
                str(seed),
                "--output-dir",
                str(model_dir),
                "--sim-id",
                run_id,
                "--log-level",
                log_level,
            ]
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = str(seed)
            env.setdefault("PYTHONUNBUFFERED", "1")
            specs.append(
                RunSpec(
                    model=model,
                    model_short=model_short,
                    seed=seed,
                    run_id=run_id,
                    model_dir=model_dir,
                    run_dir=run_dir,
                    launcher_log=launcher_log,
                    cmd=cmd,
                    env=env,
                )
            )
    return specs


def is_completed_run(run_dir: Path) -> bool:
    return (run_dir / "results.yaml").exists() and (run_dir / "analysis_results.json").exists()


def execute_run(spec: RunSpec, resume: bool) -> Dict[str, object]:
    spec.model_dir.mkdir(parents=True, exist_ok=True)
    spec.run_dir.mkdir(parents=True, exist_ok=True)

    if resume and is_completed_run(spec.run_dir):
        return {
            "model": spec.model,
            "model_short": spec.model_short,
            "seed": spec.seed,
            "run_id": spec.run_id,
            "status": "skipped",
            "returncode": 0,
            "duration_sec": 0.0,
            "run_dir": str(spec.run_dir),
            "launcher_log": str(spec.launcher_log),
        }

    start = time.time()
    with open(spec.launcher_log, "a", encoding="utf-8") as handle:
        handle.write("\n" + "=" * 80 + "\n")
        handle.write(f"Started: {datetime.now().isoformat()}\n")
        handle.write(f"Model:   {spec.model}\n")
        handle.write(f"Seed:    {spec.seed}\n")
        handle.write(f"Command: {shlex.join(spec.cmd)}\n")
        handle.write("=" * 80 + "\n\n")
        handle.flush()

        result = subprocess.run(
            spec.cmd,
            cwd=str(REPO_ROOT),
            env=spec.env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

        duration = time.time() - start
        handle.write("\n" + "-" * 80 + "\n")
        handle.write(f"Finished: {datetime.now().isoformat()}\n")
        handle.write(f"Return code: {result.returncode}\n")
        handle.write(f"Duration (s): {duration:.2f}\n")
        handle.write("-" * 80 + "\n")

    status = "success" if result.returncode == 0 and is_completed_run(spec.run_dir) else "failed"
    return {
        "model": spec.model,
        "model_short": spec.model_short,
        "seed": spec.seed,
        "run_id": spec.run_id,
        "status": status,
        "returncode": result.returncode,
        "duration_sec": round(duration, 2),
        "run_dir": str(spec.run_dir),
        "launcher_log": str(spec.launcher_log),
    }


def load_run_metrics(run_dir: Path) -> Dict[str, Optional[float]]:
    results_path = run_dir / "results.yaml"
    analysis_path = run_dir / "analysis_results.json"
    metrics: Dict[str, Optional[float]] = {
        "total_tasks_completed": None,
        "messages_per_completed_task": None,
        "total_pieces_transferred": None,
        "request_success_rate": None,
        "gini_coefficient": None,
    }

    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as handle:
            results = yaml.safe_load(handle)
        metrics["total_tasks_completed"] = results.get("total_tasks_completed")

    if analysis_path.exists():
        with open(analysis_path, "r", encoding="utf-8") as handle:
            analysis = json.load(handle)
        analysis_metrics = analysis.get("metrics", {})
        metrics["messages_per_completed_task"] = (
            analysis_metrics.get("communication_efficiency", {}).get("messages_per_completed_task")
        )
        metrics["total_pieces_transferred"] = (
            analysis_metrics.get("information_transfer_rate", {}).get("total_pieces_transferred")
        )
        metrics["request_success_rate"] = (
            analysis_metrics.get("negotiation_cycle_time", {}).get("success_rate")
        )
        metrics["gini_coefficient"] = (
            analysis_metrics.get("revenue_distribution", {}).get("gini_coefficient")
        )

    return metrics


def write_summary(experiment_dir: Path, results: List[Dict[str, object]]) -> None:
    summary_json_path = experiment_dir / "runner_summary.json"
    summary_md_path = experiment_dir / "runner_summary.md"

    by_model: Dict[str, List[Dict[str, object]]] = {}
    for result in results:
        by_model.setdefault(str(result["model_short"]), []).append(result)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "experiment_dir": str(experiment_dir),
        "runs": results,
    }
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    lines = [
        "# Zero-Cost Environment Run Summary",
        "",
        f"Experiment directory: `{experiment_dir}`",
        "",
        "| Model | Seed | Status | Tasks | Msgs/Task | Pieces Transferred | Request Success | Gini |",
        "|------|------|--------|-------|-----------|--------------------|-----------------|------|",
    ]

    def fmt(value: Optional[float]) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    for model_short in sorted(by_model.keys()):
        for result in sorted(by_model[model_short], key=lambda item: int(item["seed"])):
            metrics = load_run_metrics(Path(str(result["run_dir"]))) if result["status"] != "failed" else {}
            lines.append(
                f"| {model_short} | {result['seed']} | {result['status']} | "
                f"{fmt(metrics.get('total_tasks_completed'))} | "
                f"{fmt(metrics.get('messages_per_completed_task'))} | "
                f"{fmt(metrics.get('total_pieces_transferred'))} | "
                f"{fmt(metrics.get('request_success_rate'))} | "
                f"{fmt(metrics.get('gini_coefficient'))} |"
            )

    with open(summary_md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run zero-cost environment experiments")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["o3mini", "claude", "o3"],
        help="Model shortcuts or full model IDs",
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per model")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Base config for all runs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root experiments directory",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Folder name under experiments/ for this batch",
    )
    parser.add_argument(
        "--max-parallel-runs",
        type=int,
        default=3,
        help="How many seeds to run in parallel for each model",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level passed through to the simulation",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validate_setup preflight")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip already-completed runs",
    )
    args = parser.parse_args()

    models = resolve_models(args.models)
    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"zero_cost_access_authorization_{timestamp}"
    experiment_dir = args.output_root.resolve() / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs(
        models=models,
        runs=args.runs,
        config_path=config_path,
        experiment_dir=experiment_dir,
        log_level=args.log_level,
    )

    print("\n" + "=" * 76)
    print("ZERO-COST ENVIRONMENT EXPERIMENT MATRIX")
    print("=" * 76)
    print(f"Models: {', '.join(models)}")
    print(f"Runs per model: {args.runs}")
    print(f"Parallel runs per model: {min(args.max_parallel_runs, args.runs)}")
    print(f"Config: {config_path}")
    print(f"Output: {experiment_dir}")
    print("=" * 76)

    if args.dry_run:
        for spec in specs:
            print(f"\n[{spec.model_short} | seed {spec.seed}]")
            print(shlex.join(spec.cmd))
        return 0

    if not args.skip_validation:
        if not run_validation(models):
            print("\nValidation failed. Fix setup issues or rerun with --skip-validation.")
            return 1

    all_results: List[Dict[str, object]] = []
    resume = not args.no_resume
    failures = 0

    for model in models:
        model_short = short_model_name(model)
        model_specs = [spec for spec in specs if spec.model == model]

        print("\n" + "-" * 76)
        print(f"MODEL GROUP: {model} ({model_short})")
        print("-" * 76)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.max_parallel_runs, len(model_specs))
        ) as executor:
            future_map = {
                executor.submit(execute_run, spec, resume): spec
                for spec in model_specs
            }
            for future in concurrent.futures.as_completed(future_map):
                spec = future_map[future]
                result = future.result()
                all_results.append(result)
                status = str(result["status"]).upper()
                print(
                    f"[{status}] {spec.model_short}/{spec.run_id} "
                    f"(seed={spec.seed}, code={result['returncode']}, "
                    f"log={result['launcher_log']})"
                )
                if result["status"] == "failed":
                    failures += 1

    write_summary(experiment_dir, all_results)

    print("\n" + "=" * 76)
    print("ZERO-COST ENVIRONMENT RUN SUMMARY")
    print("=" * 76)
    print(f"Experiment directory: {experiment_dir}")
    print(f"Failed runs: {failures}")
    print(f"Summary JSON: {experiment_dir / 'runner_summary.json'}")
    print(f"Summary MD:   {experiment_dir / 'runner_summary.md'}")
    print("=" * 76)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
