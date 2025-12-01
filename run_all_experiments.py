#!/usr/bin/env python3
"""
Run All Experiments Script

This script runs all experiments defined in README.md in sequence.
It provides a convenient way to execute the complete experiment suite.

Usage:
    python run_all_experiments.py              # Run all experiments
    python run_all_experiments.py --dry-run    # Show what would be run
    python run_all_experiments.py --help       # Show all options
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


# Default models (excluding gemini which requires OpenRouter)
DEFAULT_MODELS = ['o3', 'claude', 'gpt41mini', 'gpt5mini', 'deepseek', 'o3mini']

# All models including gemini
ALL_MODELS = DEFAULT_MODELS + ['gemini', 'geminiflash']


def get_agent_types_experiments(models: List[str], include_gemini: bool = False) -> List[Dict[str, Any]]:
    """Get the list of agent types experiments to run"""
    model_str = ' '.join(models)

    experiments = [
        {
            'name': 'Agent Types: Mostly Cooperative (9 neutral, 1 uncooperative)',
            'script': 'run_agent_types_experiments.py',
            'args': ['--neutral', '9', '--uncooperative', '1', '--models'] + models,
        },
        {
            'name': 'Agent Types: Equal Mix (5 neutral, 5 uncooperative)',
            'script': 'run_agent_types_experiments.py',
            'args': ['--neutral', '5', '--uncooperative', '5', '--models'] + (['all'] if include_gemini else models),
        },
        {
            'name': 'Agent Types: Fully Uncooperative Baseline (10 uncooperative)',
            'script': 'run_agent_types_experiments.py',
            'args': ['--uncooperative', '10', '--models'] + models,
        },
    ]

    return experiments


def get_heterogeneous_experiments() -> List[Dict[str, Any]]:
    """Get the list of heterogeneous model experiments to run"""
    experiments = [
        {
            'name': 'Heterogeneous: Half O3, Half Claude (5-5 split)',
            'script': 'run_heterogeneous_experiments.py',
            'args': ['-o3', '5', '-claude', '5'],
        },
        {
            'name': 'Heterogeneous: One Weak Agent (1 O3 + 9 Claude)',
            'script': 'run_heterogeneous_experiments.py',
            'args': ['-claude', '9', '-o3', '1'],
        },
        {
            'name': 'Heterogeneous: One Strong Agent (9 O3 + 1 Claude)',
            'script': 'run_heterogeneous_experiments.py',
            'args': ['-claude', '1', '-o3', '9'],
        },
        {
            'name': 'Heterogeneous: Three-way Mix (4 O3 + 3 Claude + 3 DeepSeek)',
            'script': 'run_heterogeneous_experiments.py',
            'args': ['-o3', '4', '-claude', '3', '-deepseek', '3'],
        },
        {
            'name': 'Heterogeneous: Complex Config (2 O3 + 3 GPT5mini + 3 O3mini + 2 Claude)',
            'script': 'run_heterogeneous_experiments.py',
            'args': ['-o3', '2', '-gpt5mini', '3', '-o3mini', '3', '-claude', '2'],
        },
    ]

    return experiments


def run_validation() -> bool:
    """Run the validation script"""
    print("\n" + "=" * 60)
    print("  PREFLIGHT VALIDATION")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, 'validate_setup.py', '--skip-test'],
        capture_output=False
    )

    return result.returncode == 0


def print_experiment_list(experiments: List[Dict[str, Any]], title: str):
    """Print a formatted list of experiments"""
    print(f"\n{title}")
    print("-" * 50)
    for i, exp in enumerate(experiments, 1):
        cmd = f"python {exp['script']} {' '.join(exp['args'])}"
        print(f"  {i}. {exp['name']}")
        print(f"     Command: {cmd}")


def run_experiment(experiment: Dict[str, Any], index: int, total: int) -> bool:
    """Run a single experiment"""
    print(f"\n{'=' * 60}")
    print(f"  EXPERIMENT {index}/{total}: {experiment['name']}")
    print(f"{'=' * 60}")

    cmd = [sys.executable, experiment['script']] + experiment['args']
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=False)
        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"\n  ✓ Completed successfully in {duration:.1f}s")
            return True
        else:
            print(f"\n  ✗ Failed with exit code {result.returncode} after {duration:.1f}s")
            return False

    except KeyboardInterrupt:
        print(f"\n  ⚠ Interrupted by user")
        raise
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run all experiments from README.md',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_experiments.py                    # Run all experiments
  python run_all_experiments.py --dry-run          # Show what would be run
  python run_all_experiments.py --agent-types-only # Only agent types experiments
  python run_all_experiments.py --heterogeneous-only # Only heterogeneous experiments
  python run_all_experiments.py --include-gemini   # Include gemini models (needs OpenRouter)
"""
    )

    parser.add_argument('--dry-run', action='store_true',
                       help='Show what experiments would be run without executing them')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip the preflight validation check')
    parser.add_argument('--agent-types-only', action='store_true',
                       help='Only run agent types experiments')
    parser.add_argument('--heterogeneous-only', action='store_true',
                       help='Only run heterogeneous model experiments')
    parser.add_argument('--include-gemini', action='store_true',
                       help='Include gemini models (requires OPENROUTER_API_KEY)')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Override default models for agent types experiments')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue running experiments even if one fails')

    args = parser.parse_args()

    # Determine which models to use
    if args.models:
        models = args.models
    elif args.include_gemini:
        models = ALL_MODELS
    else:
        models = DEFAULT_MODELS

    # Build experiment list
    experiments = []

    if not args.heterogeneous_only:
        experiments.extend(get_agent_types_experiments(models, args.include_gemini))

    if not args.agent_types_only:
        experiments.extend(get_heterogeneous_experiments())

    if not experiments:
        print("Error: No experiments selected!")
        return 1

    # Print header
    print("\n" + "=" * 60)
    print("  INFORMATION ASYMMETRY SIMULATION - RUN ALL EXPERIMENTS")
    print("=" * 60)
    print(f"  Total experiments to run: {len(experiments)}")
    print(f"  Models: {', '.join(models)}")
    print(f"  Include Gemini: {args.include_gemini}")

    # Dry run mode
    if args.dry_run:
        print("\n  [DRY RUN MODE - No experiments will be executed]")

        if not args.heterogeneous_only:
            print_experiment_list(
                get_agent_types_experiments(models, args.include_gemini),
                "Agent Types Experiments:"
            )

        if not args.agent_types_only:
            print_experiment_list(
                get_heterogeneous_experiments(),
                "Heterogeneous Model Experiments:"
            )

        print("\n" + "=" * 60)
        print("  To run these experiments, remove the --dry-run flag")
        print("=" * 60)
        return 0

    # Run validation unless skipped
    if not args.skip_validation:
        if not run_validation():
            print("\n  ⚠ Validation found issues. Fix them or use --skip-validation to proceed anyway.")
            response = input("\n  Continue anyway? [y/N]: ").strip().lower()
            if response != 'y':
                return 1

    # Run experiments
    print("\n" + "=" * 60)
    print("  STARTING EXPERIMENT SUITE")
    print("=" * 60)

    results = []
    start_time = time.time()

    try:
        for i, experiment in enumerate(experiments, 1):
            success = run_experiment(experiment, i, len(experiments))
            results.append({
                'name': experiment['name'],
                'success': success
            })

            if not success and not args.continue_on_error:
                print("\n  Stopping due to experiment failure.")
                print("  Use --continue-on-error to keep running after failures.")
                break

    except KeyboardInterrupt:
        print("\n\n  Experiment suite interrupted by user.")

    # Print summary
    total_duration = time.time() - start_time
    successful = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])

    print("\n" + "=" * 60)
    print("  EXPERIMENT SUITE SUMMARY")
    print("=" * 60)
    print(f"  Total time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print(f"  Experiments run: {len(results)}/{len(experiments)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

    if results:
        print("\n  Results:")
        for r in results:
            status = "✓" if r['success'] else "✗"
            print(f"    {status} {r['name']}")

    if failed > 0:
        print("\n  ⚠ Some experiments failed. Check the output above for details.")
        return 1
    elif len(results) < len(experiments):
        print("\n  ⚠ Not all experiments were run.")
        return 1
    else:
        print("\n  ✓ All experiments completed successfully!")
        print(f"\n  Results are stored in: experiments/")
        return 0


if __name__ == "__main__":
    sys.exit(main())
