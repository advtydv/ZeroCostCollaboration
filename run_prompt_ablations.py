#!/usr/bin/env python3
"""
Run the prompt ablation rebuttal matrix for the prompting_ablation package.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parent
PROMPTING_ROOT = REPO_ROOT / "prompting_ablation"
PROMPTING_MAIN = PROMPTING_ROOT / "main.py"
DEFAULT_CONFIG = PROMPTING_ROOT / "config.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments" / "prompt_ablations"
DEFAULT_ABLATIONS = ("A", "B", "C")
DEFAULT_SEEDS = (1, 2, 3)

MODEL_SHORTCUTS = {
    "claude": "claude-sonnet-4-20250514",
    "claudesonnet": "claude-sonnet-4-20250514",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "o3mini": "o3-mini-2025-01-31",
    "o3-mini": "o3-mini-2025-01-31",
    "o3": "o3",
}

@dataclass
class RunSpec:
    model: str
    model_short: str
    ablation: str
    seed: int
    config_path: Path
    ablation_dir: Path
    run_dir: Path
    log_path: Path
    cmd: List[str]

    @property
    def label(self) -> str:
        return f"{self.model_short} | {self.ablation} | seed {self.seed}"


def resolve_models(model_tokens: Iterable[str]) -> List[str]:
    resolved: List[str] = []
    for token in model_tokens:
        model = MODEL_SHORTCUTS.get(token.lower(), token)
        resolved.append(model)
    return resolved


def short_model_name(model: str) -> str:
    if model == "claude-sonnet-4-20250514":
        return "claude"
    if model == "o3-mini-2025-01-31":
        return "o3mini"
    if model == "o3":
        return "o3"
    return model.replace("/", "_").replace("-", "_")


def normalize_ablations(values: Sequence[str]) -> List[str]:
    ablations: List[str] = []
    for value in values:
        normalized = value.strip().upper()
        if normalized not in DEFAULT_ABLATIONS:
            raise ValueError(f"Unsupported ablation '{value}'. Expected A, B, or C.")
        ablations.append(normalized)
    return ablations


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.dump(payload, handle, default_flow_style=False, sort_keys=False)


def prepare_ablation_config(base_config: Path, model: str, ablation: str, resolved_config_path: Path, rounds: int | None) -> None:
    config = load_yaml(base_config)
    config.setdefault("agents", {})
    config["agents"]["model"] = model
    config.setdefault("prompting_ablation", {})
    config["prompting_ablation"]["mode"] = ablation
    config.setdefault("simulation", {})
    config["simulation"]["fail_fast_on_agent_error"] = True
    if rounds is not None:
        config["simulation"]["rounds"] = rounds
    save_yaml(resolved_config_path, config)


def build_specs(
    models: Sequence[str],
    ablations: Sequence[str],
    seeds: Sequence[int],
    base_config: Path,
    output_root: Path,
    rounds: int | None,
) -> Dict[str, List[RunSpec]]:
    grouped: Dict[str, List[RunSpec]] = {}

    for model in models:
        model_short = short_model_name(model)
        model_dir = output_root / model_short
        specs: List[RunSpec] = []

        for ablation in ablations:
            ablation_dir = model_dir / ablation.lower()
            resolved_config_path = ablation_dir / "resolved_config.yaml"
            prepare_ablation_config(base_config, model, ablation, resolved_config_path, rounds)

            for seed in seeds:
                run_id = f"run_{seed:03d}"
                run_dir = ablation_dir / run_id
                log_path = run_dir / "runner.log"
                cmd = [
                    sys.executable,
                    str(PROMPTING_MAIN),
                    "--config",
                    str(resolved_config_path),
                    "--ablation",
                    ablation,
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(ablation_dir),
                    "--sim-id",
                    run_id,
                ]
                specs.append(
                    RunSpec(
                        model=model,
                        model_short=model_short,
                        ablation=ablation,
                        seed=seed,
                        config_path=resolved_config_path,
                        ablation_dir=ablation_dir,
                        run_dir=run_dir,
                        log_path=log_path,
                        cmd=cmd,
                    )
                )

        grouped[model] = specs

    return grouped


def run_validation(models: Sequence[str]) -> bool:
    cmd = [
        sys.executable,
        "validate_setup.py",
        "--skip-test",
        "--require-models",
        *models,
    ]
    result = subprocess.run(cmd, capture_output=False, cwd=str(REPO_ROOT))
    return result.returncode == 0


def write_run_spec(spec: RunSpec) -> None:
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": spec.model,
        "model_short": spec.model_short,
        "ablation": spec.ablation,
        "seed": spec.seed,
        "config_path": str(spec.config_path),
        "command": spec.cmd,
        "generated_at": datetime.now().isoformat(),
    }
    save_yaml(spec.run_dir / "run_spec.yaml", payload)


def tail_log(log_path: Path, max_lines: int = 20) -> str:
    if not log_path.exists():
        return ""
    with open(log_path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    return "".join(lines[-max_lines:]).strip()


def run_single_spec(spec: RunSpec) -> Dict[str, object]:
    results_path = spec.run_dir / "results.yaml"
    analysis_path = spec.run_dir / "analysis_results.json"
    write_run_spec(spec)

    if results_path.exists():
        return {
            "label": spec.label,
            "status": "skipped",
            "model": spec.model,
            "ablation": spec.ablation,
            "seed": spec.seed,
            "run_dir": str(spec.run_dir),
            "log_path": str(spec.log_path),
        }

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(spec.seed)

    start = time.time()
    with open(spec.log_path, "w", encoding="utf-8") as log_handle:
        log_handle.write(f"# {spec.label}\n")
        log_handle.write(f"# started_at: {datetime.now().isoformat()}\n")
        log_handle.write(f"# command: {' '.join(spec.cmd)}\n\n")
        log_handle.flush()
        process = subprocess.run(
            spec.cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
        )

    duration = time.time() - start
    status = "completed" if process.returncode == 0 and results_path.exists() else "failed"
    payload: Dict[str, object] = {
        "label": spec.label,
        "status": status,
        "model": spec.model,
        "ablation": spec.ablation,
        "seed": spec.seed,
        "duration_seconds": round(duration, 2),
        "returncode": process.returncode,
        "run_dir": str(spec.run_dir),
        "log_path": str(spec.log_path),
        "results_path": str(results_path) if results_path.exists() else None,
        "analysis_path": str(analysis_path) if analysis_path.exists() else None,
    }
    if status != "completed":
        payload["log_tail"] = tail_log(spec.log_path)
    return payload


def run_specs_for_model(model: str, specs: Sequence[RunSpec], max_workers: int) -> List[Dict[str, object]]:
    print("\n" + "=" * 72)
    print(f"MODEL: {model}")
    print(f"Launching {len(specs)} runs with up to {max_workers} workers")
    print("=" * 72)

    results: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_spec, spec): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            result = future.result()
            results.append(result)
            if result["status"] in {"completed", "skipped"}:
                print(f"[ok] {spec.label} -> {spec.run_dir}")
            else:
                print(f"[failed] {spec.label} -> {spec.run_dir}")

    return results


def write_summary(output_root: Path, summary: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_root / f"summary_{timestamp}.json"
    output_root.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    latest_path = output_root / "latest_summary.json"
    with open(latest_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary_path


def print_plan(
    models: Sequence[str],
    ablations: Sequence[str],
    seeds: Sequence[int],
    output_root: Path,
    max_workers_per_model: int,
    parallel_models: int,
) -> None:
    total_runs = len(models) * len(ablations) * len(seeds)
    print("\n" + "=" * 72)
    print("PROMPT ABLATION MATRIX")
    print("=" * 72)
    print(f"Models: {', '.join(models)}")
    print(f"Ablations: {', '.join(ablations)}")
    print(f"Seeds: {', '.join(str(seed) for seed in seeds)}")
    print(f"Total runs: {total_runs}")
    print(f"Output root: {output_root}")
    print(f"Workers per model: {max_workers_per_model}")
    print(f"Parallel models: {parallel_models}")
    print("Layout: experiments/prompt_ablations/<model>/<ablation>/run_###")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run prompt ablation rebuttal experiments")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["claude", "o3mini", "o3"],
        help="Model shortcuts or full model IDs",
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=list(DEFAULT_ABLATIONS),
        help="Subset of ablations to run (default: A B C)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seed values to run (default: 1 2 3)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Base prompting_ablation config file",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for experiment outputs",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Optional override for simulation rounds",
    )
    parser.add_argument(
        "--max-workers-per-model",
        type=int,
        default=9,
        help="Concurrent runs per model (default: 9)",
    )
    parser.add_argument(
        "--parallel-models",
        type=int,
        default=1,
        help="How many models to process concurrently (default: 1)",
    )
    parser.add_argument("--skip-validation", action="store_true", help="Skip validate_setup preflight")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue to next model after failures")
    parser.add_argument("--list-models", action="store_true", help="List model shortcuts and exit")
    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable model shortcuts:")
        print("=" * 40)
        for shortcut, full_name in sorted(MODEL_SHORTCUTS.items()):
            print(f"  {shortcut:14} -> {full_name}")
        return 0

    models = resolve_models(args.models)
    ablations = normalize_ablations(args.ablations)
    seeds = list(args.seeds)

    if not seeds:
        raise ValueError("At least one seed is required.")
    if any(seed <= 0 for seed in seeds):
        raise ValueError("Seeds must be positive integers.")

    base_config = Path(args.config)
    if not base_config.is_absolute():
        base_config = (REPO_ROOT / base_config).resolve()
    if not base_config.exists():
        raise FileNotFoundError(f"Config file not found: {base_config}")

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()

    max_workers_per_model = max(1, min(args.max_workers_per_model, len(ablations) * len(seeds)))
    parallel_models = max(1, min(args.parallel_models, len(models)))

    grouped_specs = build_specs(models, ablations, seeds, base_config, output_root, args.rounds)
    print_plan(models, ablations, seeds, output_root, max_workers_per_model, parallel_models)

    if args.dry_run:
        for model in models:
            print(f"\n[{model}]")
            for spec in grouped_specs[model]:
                print(" ".join(spec.cmd))
        return 0

    if not args.skip_validation:
        if not run_validation(models):
            print("\nValidation failed. Fix issues or rerun with --skip-validation.")
            return 1

    start = time.time()
    all_results: List[Dict[str, object]] = []
    failed_models: List[str] = []

    if parallel_models == 1:
        for model in models:
            model_results = run_specs_for_model(model, grouped_specs[model], max_workers_per_model)
            all_results.extend(model_results)
            if any(result["status"] == "failed" for result in model_results):
                failed_models.append(model)
                if not args.continue_on_error:
                    break
    else:
        with ThreadPoolExecutor(max_workers=parallel_models) as executor:
            futures = {
                executor.submit(run_specs_for_model, model, grouped_specs[model], max_workers_per_model): model
                for model in models
            }
            for future in as_completed(futures):
                model = futures[future]
                model_results = future.result()
                all_results.extend(model_results)
                if any(result["status"] == "failed" for result in model_results):
                    failed_models.append(model)

    duration = round(time.time() - start, 2)
    summary = {
        "generated_at": datetime.now().isoformat(),
        "models": models,
        "ablations": ablations,
        "seeds": seeds,
        "duration_seconds": duration,
        "max_workers_per_model": max_workers_per_model,
        "parallel_models": parallel_models,
        "results": all_results,
    }
    summary_path = write_summary(output_root, summary)

    completed = sum(1 for result in all_results if result["status"] == "completed")
    skipped = sum(1 for result in all_results if result["status"] == "skipped")
    failed = sum(1 for result in all_results if result["status"] == "failed")

    print("\n" + "=" * 72)
    print("PROMPT ABLATION SUMMARY")
    print("=" * 72)
    print(f"Completed: {completed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Duration:  {duration}s")
    print(f"Summary:   {summary_path}")
    if failed_models:
        print(f"Failed models: {', '.join(sorted(set(failed_models)))}")
    print("=" * 72)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
