#!/usr/bin/env python3
"""
Run All Experiments - Central Script

This script runs all experiments in sequence. Edit the EXPERIMENTS list below
to customize which experiments to run.

Usage:
    python run_all_experiments.py              # Run all experiments
    python run_all_experiments.py --dry-run    # Show what would be run
    python run_all_experiments.py --help       # Show all options
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Dict, Any

# =============================================================================
#                        EXPERIMENT CONFIGURATION
# =============================================================================
# Edit this section to add, remove, or modify experiments.
# Each experiment is a dictionary with:
#   - 'name': Human-readable description (shown in output)
#   - 'command': The full command to run (as a list of strings)
#   - 'enabled': Set to False to skip this experiment
#
# Available model shortcuts: o3, o3mini, gpt41mini, gpt5mini, deepseek, claude, 
#                            gemini, geminiflash, perfect
# =============================================================================

EXPERIMENTS = [
    # -------------------------------------------------------------------------
    # AGENT TYPES EXPERIMENTS
    # These test different mixes of cooperative vs uncooperative agent behavior
    # -------------------------------------------------------------------------
    {
        'name': 'Agent Types: Mostly Cooperative (9 neutral, 1 uncooperative)',
        'command': [
            'python', 'run_agent_types_experiments.py',
            '--neutral', '9', '--uncooperative', '1',
            '--models', 'o3', 'claude', 'gpt41mini', 'gpt5mini', 'deepseek', 'o3mini'
        ],
        'enabled': True,
    },
    {
        'name': 'Agent Types: Equal Mix (5 neutral, 5 uncooperative)',
        'command': [
            'python', 'run_agent_types_experiments.py',
            '--neutral', '5', '--uncooperative', '5',
            '--models', 'o3', 'claude', 'gpt41mini', 'gpt5mini', 'deepseek', 'o3mini'
            # Change to: '--models', 'all'  # If you have Gemini access
        ],
        'enabled': True,
    },
    {
        'name': 'Agent Types: Fully Uncooperative Baseline',
        'command': [
            'python', 'run_agent_types_experiments.py',
            '--uncooperative', '10',
            '--models', 'o3', 'claude', 'gpt41mini', 'gpt5mini', 'deepseek', 'o3mini'
        ],
        'enabled': True,
    },

    # -------------------------------------------------------------------------
    # HETEROGENEOUS MODEL EXPERIMENTS  
    # These test mixing different AI models in the same simulation
    # -------------------------------------------------------------------------
    {
        'name': 'Heterogeneous: Half O3, Half Claude (5-5 split)',
        'command': [
            'python', 'run_heterogeneous_experiments.py',
            '-o3', '5', '-claude', '5'
        ],
        'enabled': True,
    },
    {
        'name': 'Heterogeneous: One Weak Agent (1 O3 + 9 Claude)',
        'command': [
            'python', 'run_heterogeneous_experiments.py',
            '-claude', '9', '-o3', '1'
        ],
        'enabled': True,
    },
    {
        'name': 'Heterogeneous: One Strong Agent (9 O3 + 1 Claude)',
        'command': [
            'python', 'run_heterogeneous_experiments.py',
            '-claude', '1', '-o3', '9'
        ],
        'enabled': True,
    },
    {
        'name': 'Heterogeneous: Three-way Mix (4 O3 + 3 Claude + 3 DeepSeek)',
        'command': [
            'python', 'run_heterogeneous_experiments.py',
            '-o3', '4', '-claude', '3', '-deepseek', '3'
        ],
        'enabled': True,
    },
    {
        'name': 'Heterogeneous: Complex Config (2 O3 + 3 GPT5mini + 3 O3mini + 2 Claude)',
        'command': [
            'python', 'run_heterogeneous_experiments.py',
            '-o3', '2', '-gpt5mini', '3', '-o3mini', '3', '-claude', '2'
        ],
        'enabled': True,
    },
]

# =============================================================================
#                        END OF CONFIGURATION
# =============================================================================


def run_validation() -> bool:
    """Run the validation script to check environment setup."""
    print("\n" + "=" * 60)
    print("  PREFLIGHT VALIDATION")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, 'validate_setup.py', '--skip-test'],
        capture_output=False
    )
    
    return result.returncode == 0


def run_experiment(experiment: Dict[str, Any], index: int, total: int) -> bool:
    """Run a single experiment and return success status."""
    print(f"\n{'=' * 60}")
    print(f"  EXPERIMENT {index}/{total}: {experiment['name']}")
    print(f"{'=' * 60}")
    
    # Replace 'python' with the current Python executable
    cmd = experiment['command'].copy()
    if cmd[0] == 'python':
        cmd[0] = sys.executable
    
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
        description='Run all experiments from the configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_experiments.py                    # Run all enabled experiments
  python run_all_experiments.py --dry-run          # Show what would be run
  python run_all_experiments.py --continue-on-error # Keep going after failures

To customize experiments:
  Edit the EXPERIMENTS list at the top of this file.
  Set 'enabled': False to skip specific experiments.
"""
    )
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what experiments would be run without executing them')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip the preflight validation check')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue running experiments even if one fails')
    parser.add_argument('--list', action='store_true',
                       help='List all experiments with their enabled status')
    
    args = parser.parse_args()
    
    # Filter to enabled experiments only
    enabled_experiments = [e for e in EXPERIMENTS if e.get('enabled', True)]
    disabled_count = len(EXPERIMENTS) - len(enabled_experiments)
    
    # List mode
    if args.list:
        print("\n" + "=" * 60)
        print("  ALL EXPERIMENTS")
        print("=" * 60)
        for i, exp in enumerate(EXPERIMENTS, 1):
            status = "✓ ENABLED" if exp.get('enabled', True) else "✗ DISABLED"
            print(f"\n  {i}. [{status}] {exp['name']}")
            print(f"     {' '.join(exp['command'])}")
        print("\n" + "=" * 60)
        print(f"  Total: {len(EXPERIMENTS)} | Enabled: {len(enabled_experiments)} | Disabled: {disabled_count}")
        print("=" * 60)
        return 0
    
    # Print header
    print("\n" + "=" * 60)
    print("  INFORMATION ASYMMETRY SIMULATION - RUN ALL EXPERIMENTS")
    print("=" * 60)
    print(f"  Experiments to run: {len(enabled_experiments)}")
    if disabled_count > 0:
        print(f"  Experiments disabled: {disabled_count}")
    
    # Dry run mode
    if args.dry_run:
        print("\n  [DRY RUN MODE - No experiments will be executed]")
        print("\n  Experiments that would run:")
        print("-" * 50)
        for i, exp in enumerate(enabled_experiments, 1):
            print(f"\n  {i}. {exp['name']}")
            print(f"     {' '.join(exp['command'])}")
        
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
        for i, experiment in enumerate(enabled_experiments, 1):
            success = run_experiment(experiment, i, len(enabled_experiments))
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
    print(f"  Experiments run: {len(results)}/{len(enabled_experiments)}")
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
    elif len(results) < len(enabled_experiments):
        print("\n  ⚠ Not all experiments were run.")
        return 1
    else:
        print("\n  ✓ All experiments completed successfully!")
        print(f"\n  Results are stored in: experiments/")
        return 0


if __name__ == "__main__":
    sys.exit(main())
