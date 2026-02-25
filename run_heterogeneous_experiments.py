#!/usr/bin/env python3
"""
Run experiments with heterogeneous agent groups where different agents can use different models.

This script configures mixed mode (LLM vs perfect) and writes per-agent model overrides
via agents.agent_models so LLM agents can use different models in the same run.
"""

import argparse
import yaml
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time

# Model name mappings for convenience
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
    'perfect': 'perfect',  # Special case for perfect agents
}

def parse_agent_configs(args: List[str]) -> List[Tuple[str, int]]:
    """
    Parse command line arguments for agent configurations.
    Format: -model count -model count ...
    """
    configs = []
    i = 0
    while i < len(args):
        if args[i].startswith('-'):
            model_key = args[i][1:]  # Remove the dash
            if i + 1 < len(args):
                try:
                    count = int(args[i + 1])
                    configs.append((model_key, count))
                    i += 2
                except ValueError:
                    print(f"Error: Model '{args[i]}' must be followed by a number")
                    sys.exit(1)
            else:
                print(f"Error: Model '{args[i]}' must be followed by a count")
                sys.exit(1)
        else:
            print(f"Error: Unexpected argument '{args[i]}'. Use format: -model count")
            sys.exit(1)
    return configs

def expand_agent_configs(configs: List[Tuple[str, int]]) -> List[Dict[str, str]]:
    """
    Expand agent configurations into a list of agent specifications.
    Returns list of dicts with 'model' and 'mode' keys.
    """
    agents = []
    agent_num = 1

    for model_key, count in configs:
        # Get full model name
        if model_key in MODEL_SHORTCUTS:
            model = MODEL_SHORTCUTS[model_key]
        else:
            # Try to use as-is if not in shortcuts
            model = model_key

        # Determine mode (perfect or llm)
        mode = 'perfect' if model == 'perfect' else 'llm'

        for _ in range(count):
            agents.append({
                'id': f'agent_{agent_num}',
                'model': model,
                'mode': mode,
                'model_key': model_key  # Keep the shortcut for naming
            })
            agent_num += 1

    return agents

def create_experiment_name(configs: List[Tuple[str, int]], total_agents: int) -> str:
    """Create a descriptive experiment name from agent configurations."""
    # Create a short descriptive name
    parts = []
    for model_key, count in configs:
        # Use abbreviated format for compactness
        parts.append(f"{model_key}{count}")

    # Join with underscore and add total agent count
    name_suffix = "_".join(parts)
    return f"heterogeneous_{total_agents}agents_{name_suffix}"

def get_latest_experiment_path(output_dir: str) -> Optional[Path]:
    """Get the latest experiment folder path under experiments/output_dir."""
    experiments_root = Path("experiments") / output_dir
    if not experiments_root.exists():
        return None
    candidates = [p for p in experiments_root.iterdir() if p.is_dir() and p.name.startswith("exp_")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def create_mixed_mode_config(base_config_path: Path, agent_specs: List[Dict[str, str]]) -> dict:
    """
    Create a configuration for mixed-mode simulation.
    Supports per-agent mode assignment and per-agent model assignment.
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        sim_config = yaml.safe_load(f)

    # Update agent count
    sim_config['simulation']['agents'] = len(agent_specs)
    sim_config['simulation']['fail_fast_on_agent_error'] = True

    # Set to mixed mode
    sim_config['agents']['mode'] = 'mixed'

    # Create agent_modes dictionary
    agent_modes = {}
    agent_models = {}

    for agent_spec in agent_specs:
        agent_modes[agent_spec['id']] = agent_spec['mode']
        if agent_spec['mode'] == 'llm':
            agent_models[agent_spec['id']] = agent_spec['model']

    sim_config['agents']['agent_modes'] = agent_modes
    sim_config['agents']['agent_models'] = agent_models

    # Keep a default model for backward compatibility in case a run has no per-agent map.
    if agent_models:
        first_llm_model = next(iter(agent_models.values()))
        sim_config['agents']['model'] = first_llm_model

    return sim_config

def create_heterogeneous_config(base_config_path: Path, agent_specs: List[Dict[str, str]]) -> dict:
    """
    Create a configuration for true heterogeneous agents.
    This format could work if the simulation is extended to support per-agent models.
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        sim_config = yaml.safe_load(f)

    # Update agent count
    sim_config['simulation']['agents'] = len(agent_specs)
    sim_config['simulation']['fail_fast_on_agent_error'] = True

    # Create agent-specific configurations (for future compatibility)
    sim_config['agents']['heterogeneous'] = True
    sim_config['agents']['agent_configs'] = []

    for agent_spec in agent_specs:
        agent_config = {
            'id': agent_spec['id'],
            'model': agent_spec['model'],
            'mode': agent_spec['mode'],
            'type': 'neutral'  # Default to neutral
        }
        sim_config['agents']['agent_configs'].append(agent_config)

    # Also set mixed mode for backward compatibility
    sim_config['agents']['mode'] = 'mixed'
    agent_modes = {spec['id']: spec['mode'] for spec in agent_specs}
    sim_config['agents']['agent_modes'] = agent_modes

    # Set a default model for compatibility
    llm_models = [spec['model'] for spec in agent_specs if spec['mode'] == 'llm']
    if llm_models:
        sim_config['agents']['model'] = llm_models[0]

    return sim_config

def run_heterogeneous_experiment(configs: List[Tuple[str, int]],
                                base_config_path: Path,
                                output_dir: str,
                                runs: int,
                                simulation_root: str,
                                use_mixed_mode: bool = True) -> bool:
    """Run experiment with heterogeneous agent configuration."""

    # Expand agent configurations
    agent_specs = expand_agent_configs(configs)
    total_agents = len(agent_specs)

    if total_agents == 0:
        print("Error: No agents specified!")
        sys.exit(1)

    # Create experiment name
    experiment_name = create_experiment_name(configs, total_agents)

    # Create simulation config
    if use_mixed_mode:
        # Use the current mixed-mode approach (works with existing simulation)
        sim_config = create_mixed_mode_config(base_config_path, agent_specs)
        config_desc = "Mixed-mode configuration (LLM vs perfect agents)"
    else:
        # Use heterogeneous config format (for potential future support)
        sim_config = create_heterogeneous_config(base_config_path, agent_specs)
        config_desc = "Heterogeneous configuration (requires simulation extension)"

    # Count agent types for description
    llm_count = sum(1 for spec in agent_specs if spec['mode'] == 'llm')
    perfect_count = sum(1 for spec in agent_specs if spec['mode'] == 'perfect')

    # Create detailed description
    model_distribution = {}
    for spec in agent_specs:
        model_key = spec['model_key']
        model_distribution[model_key] = model_distribution.get(model_key, 0) + 1

    description_parts = [f"{count} {model}" for model, count in model_distribution.items()]

    # Create experiment config
    exp_config = {
        'experiment': {
            'name': experiment_name,
            'description': f"Heterogeneous agent experiment with {total_agents} agents.\n" +
                          f"Configuration: {', '.join(description_parts)}\n" +
                          f"Breakdown: {llm_count} LLM agents, {perfect_count} perfect agents\n" +
                          f"Mode: {config_desc}",
            'num_runs': runs,
            'parallel': True,
            'max_workers': min(runs, 5),
            'tags': ['heterogeneous', 'mixed-agents'] + list(set([c[0] for c in configs]))
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
                'per_agent_metrics'  # Include detailed per-agent analysis
            ],
            'generate_plots': True,
            'generate_report': True
        }
    }

    # Save temp config
    temp_config_dir = Path("experiment_framework/configs/temp")
    temp_config_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_config_path = temp_config_dir / f"heterogeneous_{timestamp}.yaml"

    with open(temp_config_path, 'w') as f:
        yaml.dump(exp_config, f, default_flow_style=False, sort_keys=False)

    # Print configuration summary
    print(f"\n{'='*60}")
    print(f"Running Heterogeneous Agent Experiment")
    print(f"{'='*60}")
    print(f"Total agents: {total_agents}")
    print(f"\nAgent configuration:")
    for model_key, count in configs:
        model_name = MODEL_SHORTCUTS.get(model_key, model_key)
        print(f"  • {count:2} agents: {model_name}")
    print(f"\nExperiment name: {experiment_name}")
    print(f"Output directory: experiments/{output_dir}/")
    print(f"Number of runs: {runs}")
    print(f"Simulation root: {simulation_root}")
    print(f"Fail-fast agent errors: {sim_config['simulation'].get('fail_fast_on_agent_error', False)}")
    print(f"Config mode: {config_desc}")
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
            print(f"\n✓ Successfully completed heterogeneous experiment")
            print(f"  Duration: {duration:.1f} seconds")
            if latest_path:
                print(f"  Results: {latest_path}")
            else:
                print(f"  Results: experiments/{output_dir}/")
            return True
        else:
            print(f"\n✗ Failed to run heterogeneous experiment")
            print(f"  Exit code: {process.returncode}")
            return False

    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error running heterogeneous experiment: {str(e)}")
        return False
    finally:
        # Clean up temp config
        if temp_config_path.exists():
            temp_config_path.unlink()

def main():
    parser = argparse.ArgumentParser(
        description='Run experiments with heterogeneous agent groups',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Run with 5 O3 agents and 5 Gemini agents
  python run_heterogeneous_experiments.py -o3 5 -gemini 5

  # Mix of LLM and perfect agents
  python run_heterogeneous_experiments.py -o3mini 3 -perfect 7

  # Complex configuration with multiple models
  python run_heterogeneous_experiments.py -o3 2 -gpt5mini 3 -deepseek 2 -claude 1 -perfect 2

  # Specify custom output directory
  python run_heterogeneous_experiments.py -o3 5 -gemini 5 --output-dir my_experiment

  # Use different number of runs
  python run_heterogeneous_experiments.py -o3mini 10 --runs 10

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
  perfect     → Perfect agents (non-LLM, optimal behavior)

NOTES:
  • Mixed mode supports per-agent model assignment for LLM agents.
  • The --heterogeneous-mode config format is still experimental.
  • Agent IDs are assigned sequentially: agent_1, agent_2, etc.
""")

    # Known arguments
    parser.add_argument('--config', type=str,
                       default='information_asymmetry_simulation/config.yaml',
                       help='Base configuration file (default: config.yaml)')
    parser.add_argument('--simulation-root', type=str,
                       default='information_asymmetry_simulation',
                       help='Simulation code directory to execute (default: information_asymmetry_simulation)')
    parser.add_argument('--output-dir', type=str,
                       default='heterogeneous',
                       help='Output directory name under experiments/ (default: heterogeneous)')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of simulation runs (default: 5)')
    parser.add_argument('--list-models', action='store_true',
                       help='List available model shortcuts and exit')
    parser.add_argument('--heterogeneous-mode', action='store_true',
                       help='Use heterogeneous config format (experimental)')

    # Parse known args and collect the rest for model specifications
    args, remaining = parser.parse_known_args()

    # Handle --list-models
    if args.list_models:
        print("\nAvailable model shortcuts:")
        print("="*40)
        for shortcut, full_name in sorted(MODEL_SHORTCUTS.items()):
            if shortcut != 'perfect':
                print(f"  {shortcut:12} → {full_name}")
        print(f"  {'perfect':12} → Perfect agents (optimal behavior)")
        print("\nUse these shortcuts with -shortcut count")
        print("Example: python run_heterogeneous_experiments.py -o3 5 -gemini 5")
        sys.exit(0)

    # Parse model configurations from remaining arguments
    if not remaining:
        print("Error: No agent configurations specified!")
        print("\nUsage: python run_heterogeneous_experiments.py -model count [-model count ...]")
        print("\nExample: python run_heterogeneous_experiments.py -o3 5 -gemini 5")
        print("\nUse --help for more information or --list-models to see available models")
        sys.exit(1)

    try:
        configs = parse_agent_configs(remaining)
    except ValueError as e:
        print(f"Error parsing agent configurations: {e}")
        parser.print_help()
        sys.exit(1)

    if not configs:
        print("Error: No valid agent configurations found!")
        sys.exit(1)

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

    # Run the experiment
    success = run_heterogeneous_experiment(
        configs,
        base_config_path,
        args.output_dir,
        args.runs,
        simulation_root,
        use_mixed_mode=not args.heterogeneous_mode
    )
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
