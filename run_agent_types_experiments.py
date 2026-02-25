#!/usr/bin/env python3
"""
Run experiments with different distributions of agent behavioral types.

Allows configuring the mix of neutral, uncooperative, and competitive agents
to study how different behavioral compositions affect system outcomes.
Can test multiple models systematically.
"""

import argparse
import yaml
import sys
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import time

# Define the models to test
MODELS_TO_TEST = [
    "o3-mini-2025-01-31",
    "o3",
    "gpt-4.1-mini",
    "gpt-5-mini",
    "deepseek-ai/DeepSeek-R1-0528-Turbo",
    "claude-sonnet-4-20250514",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash"
]

# Model shortcuts for convenience
MODEL_SHORTCUTS = {
    'o3mini': 'o3-mini-2025-01-31',
    'o3': 'o3',
    'gpt41mini': 'gpt-4.1-mini',
    'gpt5mini': 'gpt-5-mini',
    'gpt52': 'gpt-5.2-2025-12-11',
    'gpt5_2': 'gpt-5.2-2025-12-11',
    'gpt-5.2': 'gpt-5.2-2025-12-11',
    'deepseek': 'deepseek-ai/DeepSeek-R1-0528-Turbo',
    'claude': 'claude-sonnet-4-20250514',
    'claudesonnet': 'claude-sonnet-4-20250514',
    'claudeopus46': 'claude-opus-4-6',
    'opus46': 'claude-opus-4-6',
    'claude-opus-4.6': 'claude-opus-4-6',
    'gemini': 'google/gemini-2.5-pro',
    'gemini25': 'google/gemini-2.5-pro',
    'gemini25pro': 'google/gemini-2.5-pro',
    'geminiflash': 'google/gemini-2.5-flash',
}

def parse_agent_distribution(
    neutral: int,
    uncooperative: int,
    competitive: int,
    policy: int,
    total: Optional[int] = None,
) -> Tuple[int, int, int, int, int]:
    """
    Parse and validate agent distribution.
    Returns (neutral, uncooperative, competitive, policy, total)
    """
    # Calculate actual total
    specified_total = neutral + uncooperative + competitive + policy

    if total is not None:
        if specified_total != total:
            print(f"⚠️  WARNING: Specified counts ({specified_total}) don't match --total ({total})")
            print(
                f"   Using specified counts: {neutral} neutral + {uncooperative} uncooperative + "
                f"{competitive} competitive + {policy} policy = {specified_total} total"
            )

    return neutral, uncooperative, competitive, policy, specified_total

def get_model_short_name(model: str) -> str:
    """Get short name for model to use in experiment naming."""
    if model == "o3-mini-2025-01-31":
        return "o3mini"
    elif model == "o3":
        return "o3"
    elif model == "gpt-4.1-mini":
        return "gpt41mini"
    elif model == "gpt-5-mini":
        return "gpt5mini"
    elif model in {"gpt-5.2", "gpt-5.2-2025-12-11"}:
        return "gpt52"
    elif model == "deepseek-ai/DeepSeek-R1-0528-Turbo":
        return "deepseek"
    elif model == "claude-sonnet-4-20250514":
        return "claudesonnet"
    elif model in {"claude-opus-4.6", "claude-opus-4-6"}:
        return "claudeopus46"
    elif model == "google/gemini-2.5-pro":
        return "gemini25pro"
    elif model == "google/gemini-2.5-flash":
        return "gemini25flash"
    else:
        # Fallback: replace special chars
        return model.replace('-', '').replace('.', '').replace('/', '')[:10]

def create_experiment_name(
    neutral: int,
    uncooperative: int,
    competitive: int,
    policy: int,
    total: int,
    model: str,
) -> str:
    """Create a descriptive experiment name."""
    model_short = get_model_short_name(model)

    # Build descriptive parts
    parts = []
    if uncooperative > 0:
        parts.append(f"{uncooperative}u")  # u for uncooperative
    if competitive > 0:
        parts.append(f"{competitive}c")  # c for competitive
    if policy > 0:
        parts.append(f"{policy}p")  # p for policy
    if neutral > 0 and (uncooperative > 0 or competitive > 0 or policy > 0):
        parts.append(f"{neutral}n")  # n for neutral (only if there are other types)

    type_spec = "_".join(parts) if parts else "all_neutral"
    return f"agent_types_{model_short}_{total}agents_{type_spec}"

def get_latest_experiment_path(output_dir: str) -> Optional[Path]:
    """Get the latest experiment folder path under experiments/output_dir."""
    experiments_root = Path("experiments") / output_dir
    if not experiments_root.exists():
        return None
    candidates = [p for p in experiments_root.iterdir() if p.is_dir() and p.name.startswith("exp_")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def create_agent_types_config(base_config_path: Path,
                             neutral: int,
                             uncooperative: int,
                             competitive: int,
                             policy: int,
                             total: int,
                             model: str,
                             sharing_incentive: Optional[int] = None) -> dict:
    """Create configuration with specified agent type distribution."""
    # Load base config
    with open(base_config_path, 'r') as f:
        sim_config = yaml.safe_load(f)

    # Update agent counts
    sim_config['simulation']['agents'] = total
    sim_config['simulation']['fail_fast_on_agent_error'] = True
    sim_config['agents']['uncooperative_count'] = uncooperative
    sim_config['agents']['competitive_count'] = competitive
    sim_config['agents']['policy_count'] = policy
    sim_config['agents']['model'] = model

    if sharing_incentive is not None:
        sim_config.setdefault('revenue', {})
        sim_config['revenue']['information_sharing'] = sharing_incentive

    # Note: neutral agents are implicit in total, while special types are explicit counts.

    return sim_config

def run_agent_types_experiment(neutral: int,
                              uncooperative: int,
                              competitive: int,
                              policy: int,
                              model: str,
                              base_config_path: Path,
                              output_dir: str,
                              runs: int,
                              rounds: int,
                              revenue_visibility: str,
                              simulation_root: str,
                              sharing_incentive: Optional[int] = None) -> bool:
    """Run experiment with specified agent type distribution."""

    # Parse distribution
    neutral, uncooperative, competitive, policy, total = parse_agent_distribution(
        neutral, uncooperative, competitive, policy
    )

    # Create experiment name
    experiment_name = create_experiment_name(
        neutral, uncooperative, competitive, policy, total, model
    )

    # Create simulation config
    sim_config = create_agent_types_config(
        base_config_path, neutral, uncooperative, competitive, policy, total, model,
        sharing_incentive=sharing_incentive
    )

    # Update rounds if specified
    if rounds:
        sim_config['simulation']['rounds'] = rounds

    # Update revenue visibility
    if revenue_visibility == 'full':
        sim_config['simulation']['show_full_revenue'] = True
    elif revenue_visibility == 'limited':
        sim_config['simulation']['show_full_revenue'] = False

    # Create experiment description
    description_parts = []
    if neutral > 0:
        description_parts.append(f"{neutral} neutral (standard)")
    if uncooperative > 0:
        description_parts.append(f"{uncooperative} uncooperative")
    if competitive > 0:
        description_parts.append(f"{competitive} competitive")
    if policy > 0:
        description_parts.append(f"{policy} policy")

    model_short = get_model_short_name(model)

    # Create experiment config
    exp_config = {
        'experiment': {
            'name': experiment_name,
            'description': f"Agent type distribution experiment with {total} agents.\n" +
                          f"Distribution: {', '.join(description_parts)}\n" +
                          f"Model: {model}\n" +
                          f"Revenue visibility: {revenue_visibility}\n" +
                          f"Testing how different behavioral types affect cooperation and performance.",
            'num_runs': runs,
            'parallel': True,
            'max_workers': min(runs, 5),
            'tags': ['agent-types', model_short, f'{uncooperative}u{competitive}c{policy}p', revenue_visibility]
        },
        'runner': {
            'simulation_root': simulation_root
        },
        'simulation_config': sim_config,
        'analysis': {
            'key_metrics': [
                'total_tasks_completed',
                'revenue_distribution.gini_coefficient',
                'communication_efficiency.messages_per_completed_task',
                'agents_with_zero_revenue',
                'network_hub_analysis.hub_concentration',
                'agent_type_performance'  # Analysis by agent type
            ],
            'generate_plots': True,
            'generate_report': True
        }
    }

    # Save temp config
    temp_config_dir = Path("experiment_framework/configs/temp")
    temp_config_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_config_path = temp_config_dir / f"agent_types_{timestamp}.yaml"

    with open(temp_config_path, 'w') as f:
        yaml.dump(exp_config, f, default_flow_style=False, sort_keys=False)

    # Print configuration summary
    print(f"\n{'='*60}")
    print(f"Running Agent Types Experiment")
    print(f"{'='*60}")
    print(f"Total agents: {total}")
    print(f"\nAgent distribution:")
    print(f"  • {neutral:2} Neutral agents (cooperative)")
    print(f"  • {uncooperative:2} Uncooperative agents (won't share)")
    print(f"  • {competitive:2} Competitive agents (strategic)")
    print(f"  • {policy:2} Policy agents (policy + neutral objective)")
    print(f"\nModel: {model}")
    print(f"Rounds: {sim_config['simulation']['rounds']}")
    print(f"Revenue visibility: {revenue_visibility}")
    print(f"Fail-fast agent errors: {sim_config['simulation'].get('fail_fast_on_agent_error', False)}")
    print(f"Simulation root: {simulation_root}")
    if sharing_incentive is not None:
        print(f"Information sharing incentive: ${sharing_incentive:,} per piece")
    print(f"Number of runs: {runs}")
    print(f"\nExperiment name: {experiment_name}")
    print(f"Output directory: experiments/{output_dir}/")
    print(f"{'='*60}")

    # Run experiment
    cmd = [
        sys.executable,
        "experiment_framework/run_experiment.py",
        "--config", str(temp_config_path),
        "--experiments-dir", f"experiments/{output_dir}"
    ]

    try:
        print(f"\nStarting experiment...")
        start_time = time.time()

        process = subprocess.run(cmd, capture_output=False, text=True)

        duration = time.time() - start_time

        if process.returncode == 0:
            latest_path = get_latest_experiment_path(output_dir)
            print(f"\n✓ Successfully completed agent types experiment")
            print(f"  Duration: {duration:.1f} seconds")
            if latest_path:
                print(f"  Results: {latest_path}")
            else:
                print(f"  Results: experiments/{output_dir}/")
            return True
        else:
            print(f"\n✗ Failed to run agent types experiment")
            print(f"  Exit code: {process.returncode}")
            return False

    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error running agent types experiment: {str(e)}")
        return False
    finally:
        # Clean up temp config
        if temp_config_path.exists():
            temp_config_path.unlink()

def main():
    parser = argparse.ArgumentParser(
        description='Run experiments with different agent behavioral type distributions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
AGENT TYPES:
  • Neutral: Standard cooperative agents (default behavior)
  • Uncooperative: Refuse to share information with others
  • Competitive: Strategic behavior, may mislead others
  • Policy: Follow explicit sharing policy while keeping neutral objective

EXAMPLES:
  # 10 neutral agents (baseline)
  python run_agent_types_experiments.py --neutral 10

  # Mix of types (5 neutral, 3 uncooperative, 2 competitive, 0 policy)
  python run_agent_types_experiments.py --neutral 5 --uncooperative 3 --competitive 2 --policy 0

  # Mostly uncooperative environment
  python run_agent_types_experiments.py --neutral 2 --uncooperative 8

  # Policy intervention only
  python run_agent_types_experiments.py --policy 10

  # Specify model and other parameters
  python run_agent_types_experiments.py --neutral 5 --uncooperative 5 --model gemini --runs 10

  # Test with limited revenue visibility
  python run_agent_types_experiments.py --neutral 10 --revenue limited

QUICK PRESETS:
  # Cooperative baseline (all neutral)
  python run_agent_types_experiments.py --neutral 10

  # Hostile environment (mostly uncooperative)
  python run_agent_types_experiments.py --neutral 2 --uncooperative 8

  # Competitive market (mostly competitive)
  python run_agent_types_experiments.py --neutral 2 --competitive 8

  # Chaos mode (equal mix)
  python run_agent_types_experiments.py --neutral 4 --uncooperative 3 --competitive 3

MODEL SHORTCUTS:
  o3mini      → o3-mini-2025-01-31
  o3          → o3
  gpt41mini   → gpt-4.1-mini
  gpt5mini    → gpt-5-mini
  gpt52       → gpt-5.2
  deepseek    → deepseek-ai/DeepSeek-R1-0528-Turbo
  claude      → claude-sonnet-4-20250514
  claudeopus46 → claude-opus-4.6
  gemini      → google/gemini-2.5-pro
  geminiflash → google/gemini-2.5-flash

NOTES:
  • If no counts are specified, defaults to 10 neutral agents
  • Total agent count is sum of all specified types
  • Agents are randomly assigned their types at initialization
""")

    # Agent type arguments
    parser.add_argument('--neutral', type=int, default=None,
                       help='Number of neutral (cooperative) agents')
    parser.add_argument('--uncooperative', type=int, default=None,
                       help='Number of uncooperative agents')
    parser.add_argument('--competitive', type=int, default=None,
                       help='Number of competitive agents')
    parser.add_argument('--policy', type=int, default=None,
                       help='Number of policy agents')

    # Optional total for validation
    parser.add_argument('--total', type=int, default=None,
                       help='Expected total agents (for validation)')

    # Model selection - can specify one or more models
    parser.add_argument('--model', type=str, default=None,
                       help='Single model to use (shortcut or full name)')
    parser.add_argument('--models', nargs='+', default=None,
                       help='List of models to test (shortcuts or full names)')

    # Experiment parameters
    parser.add_argument('--config', type=str,
                       default='information_asymmetry_simulation/config.yaml',
                       help='Base configuration file (default: config.yaml)')
    parser.add_argument('--simulation-root', type=str,
                       default='information_asymmetry_simulation',
                       help='Simulation code directory to execute (default: information_asymmetry_simulation)')
    parser.add_argument('--output-dir', type=str,
                       default='agent_types',
                       help='Output directory under experiments/ (default: agent_types)')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of simulation runs (default: 5)')
    parser.add_argument('--rounds', type=int, default=None,
                       help='Number of rounds per simulation (default: from config)')
    parser.add_argument('--revenue', type=str, choices=['full', 'limited'],
                       default='full',
                       help='Revenue visibility mode (default: full)')
    parser.add_argument('--sharing-incentive', type=int, default=None,
                       help='Override revenue.information_sharing (e.g., 1000)')

    # Utility arguments
    parser.add_argument('--list-models', action='store_true',
                       help='List available model shortcuts and exit')
    parser.add_argument('--sequential', action='store_true',
                       help='Add delay between models when testing multiple')

    # Preset configurations
    parser.add_argument('--preset', type=str,
                       choices=['baseline', 'hostile', 'competitive', 'chaos', 'mixed'],
                       help='Use a preset configuration')

    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        print("\nAvailable model shortcuts:")
        print("="*40)
        for shortcut, full_name in sorted(MODEL_SHORTCUTS.items()):
            print(f"  {shortcut:12} → {full_name}")
        print("\nPredefined model list (for testing all):")
        for model in MODELS_TO_TEST:
            print(f"  • {model}")
        print("\nUsage:")
        print("  --model gemini              # Test single model")
        print("  --models o3 gpt5mini claude # Test specific models")
        print("  --models all                # Test all predefined models")
        sys.exit(0)

    # Handle presets
    if args.preset:
        if args.preset == 'baseline':
            neutral, uncooperative, competitive, policy = 10, 0, 0, 0
            print("Using baseline preset: 10 neutral agents")
        elif args.preset == 'hostile':
            neutral, uncooperative, competitive, policy = 2, 8, 0, 0
            print("Using hostile preset: 2 neutral, 8 uncooperative")
        elif args.preset == 'competitive':
            neutral, uncooperative, competitive, policy = 2, 0, 8, 0
            print("Using competitive preset: 2 neutral, 8 competitive")
        elif args.preset == 'chaos':
            neutral, uncooperative, competitive, policy = 4, 3, 3, 0
            print("Using chaos preset: 4 neutral, 3 uncooperative, 3 competitive")
        elif args.preset == 'mixed':
            neutral, uncooperative, competitive, policy = 6, 2, 2, 0
            print("Using mixed preset: 6 neutral, 2 uncooperative, 2 competitive")
    else:
        type_args_provided = any(
            x is not None for x in [args.neutral, args.uncooperative, args.competitive, args.policy]
        )
        if type_args_provided:
            # Only explicitly provided counts are used; unspecified types default to 0
            neutral = args.neutral if args.neutral is not None else 0
            uncooperative = args.uncooperative if args.uncooperative is not None else 0
            competitive = args.competitive if args.competitive is not None else 0
            policy = args.policy if args.policy is not None else 0
        else:
            # Default baseline
            neutral, uncooperative, competitive, policy = 10, 0, 0, 0
            print("No agent counts specified, using default: 10 neutral agents")

    # Validate at least one agent
    total = neutral + uncooperative + competitive + policy
    if total == 0:
        print("Error: Must specify at least one agent!")
        print("\nExample: python run_agent_types_experiments.py --neutral 10")
        sys.exit(1)

    # Determine which models to test
    if args.models:
        # Multiple models specified
        if len(args.models) == 1 and args.models[0].lower() == 'all':
            models = MODELS_TO_TEST
            print("Testing all predefined models")
        else:
            # Expand shortcuts for each model
            models = []
            for m in args.models:
                models.append(MODEL_SHORTCUTS.get(m, m))
    elif args.model:
        # Single model specified
        models = [MODEL_SHORTCUTS.get(args.model, args.model)]
    else:
        # Default to o3mini
        models = ['o3-mini-2025-01-31']
        print("No model specified, using default: o3-mini")

    # Check base config exists
    base_config_path = Path(args.config)
    if not base_config_path.exists():
        print(f"Error: Config file not found: {base_config_path}")
        print("\nAvailable configs:")
        print("  • information_asymmetry_simulation/config.yaml (standard)")
        print("  • information_asymmetry_simulation/config_mixed.yaml (mixed mode)")
        print("  • information_asymmetry_simulation/config_perfect.yaml (perfect agents)")
        sys.exit(1)

    repo_root = Path(__file__).resolve().parent
    simulation_root_path = Path(args.simulation_root)
    if not simulation_root_path.is_absolute():
        simulation_root_path = (repo_root / simulation_root_path).resolve()
    if not simulation_root_path.exists() or not (simulation_root_path / "main.py").exists():
        print(f"Error: Simulation root is invalid: {args.simulation_root}")
        print("Expected a directory containing main.py")
        sys.exit(1)
    simulation_root = str(simulation_root_path)

    # Display configuration summary
    print("\n" + "="*60)
    print("AGENT TYPES EXPERIMENT CONFIGURATION")
    print("="*60)
    print(
        f"Agent distribution: {neutral} neutral, {uncooperative} uncooperative, "
        f"{competitive} competitive, {policy} policy"
    )
    print(f"Total agents: {total}")
    print(f"Models to test: {len(models)}")
    for model in models:
        print(f"  • {model}")
    print(f"Runs per model: {args.runs}")
    print(f"Simulation root: {simulation_root}")
    if args.sharing_incentive is not None:
        print(f"Information sharing incentive override: ${args.sharing_incentive:,}")
    print(f"Output directory: experiments/{args.output_dir}/")
    print("="*60)

    # Run experiments for each model
    all_success = True
    for i, model in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] Testing model: {model}")

        success = run_agent_types_experiment(
            neutral=neutral,
            uncooperative=uncooperative,
            competitive=competitive,
            policy=policy,
            model=model,
            base_config_path=base_config_path,
            output_dir=args.output_dir,
            runs=args.runs,
            rounds=args.rounds,
            revenue_visibility=args.revenue,
            simulation_root=simulation_root,
            sharing_incentive=args.sharing_incentive
        )
        all_success = all_success and success

        # Add delay between models if requested
        if args.sequential and i < len(models) - 1:
            print(f"\nWaiting 5 seconds before next model...")
            time.sleep(5)

    print("\n" + "="*60)
    if all_success:
        print("ALL AGENT TYPES EXPERIMENTS COMPLETED")
    else:
        print("AGENT TYPES EXPERIMENTS FINISHED WITH FAILURES")
    print(f"Results are stored in: experiments/{args.output_dir}/")
    print("="*60)
    if not all_success:
        sys.exit(1)

if __name__ == "__main__":
    main()
