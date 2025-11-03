#!/usr/bin/env python3
"""
Run perfect mode experiments
Follows the same pattern as run_mixed_mode_experiments.py but for perfect agents only
No LLM models needed as all agents use hard-coded optimal behavior
"""

import yaml
import subprocess
import sys
from pathlib import Path
import argparse
import time

def load_config(config_path: Path) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: dict, config_path: Path):
    """Save YAML configuration file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def create_experiment_config() -> dict:
    """Create experiment framework config for perfect mode"""
    
    # Load the base perfect config from information_asymmetry_simulation
    sim_config = load_config(Path("information_asymmetry_simulation/config_perfect.yaml"))
    
    # Create experiment config (similar to experiment_template.yaml structure)
    config = {
        'experiment': {
            'name': 'perfect_mode_10agents_20rounds',
            'description': "Perfect mode experiment with all 10 agents using theoretically optimal behavior.\n" + 
                          "Establishes theoretical maximum performance baseline for the simulation environment.",
            'num_runs': 5,
            'parallel': True,  # Enable parallel execution of the 5 runs
            'max_workers': 5,  # Run all 5 simulations in parallel
            'tags': ['perfect-mode', 'baseline', 'optimal', '10-agents']
        },
        'simulation_config': sim_config,
        'analysis': {
            'key_metrics': [
                'total_tasks_completed',
                'revenue_distribution.gini_coefficient',
                'communication_efficiency.messages_per_completed_task',
                'agents_with_zero_revenue',
                'network_hub_analysis.hub_concentration'
            ],
            'generate_plots': True,
            'generate_report': True
        }
    }
    
    return config

def run_perfect_experiment(experiments_subdir: str = "perfect_mode_20agents"):
    """Run the perfect mode experiment"""
    
    # Create the experiment config
    config = create_experiment_config()
    
    # Create a temporary config file for this run
    temp_config_dir = Path("experiment_framework/configs/temp")
    temp_config_dir.mkdir(exist_ok=True)
    
    temp_config_path = temp_config_dir / "perfect_mode.yaml"
    save_config(config, temp_config_path)
    
    print(f"\n{'='*60}")
    print(f"Running perfect mode experiment")
    print(f"Config: {temp_config_path}")
    print(f"{'='*60}")
    
    # Run the experiment using experiment_framework with custom output directory
    cmd = [
        sys.executable,
        "experiment_framework/run_experiment.py",
        "--config", str(temp_config_path),
        "--experiments-dir", f"experiments/{experiments_subdir}"
    ]
    
    try:
        process = subprocess.run(cmd, capture_output=False, text=True)
        
        if process.returncode == 0:
            print(f"✓ Successfully completed perfect mode experiment")
        else:
            print(f"✗ Failed to run perfect mode experiment (exit code: {process.returncode})")
            
    except Exception as e:
        print(f"✗ Error running perfect mode experiment: {str(e)}")
    
    # Clean up temp config
    temp_config_path.unlink()

def main():
    parser = argparse.ArgumentParser(description='Run perfect mode experiments')
    parser.add_argument('--output-dir', type=str, default='perfect_mode_30',
                       help='Subdirectory in experiments/ for output (default: perfect_mode)')
    
    args = parser.parse_args()
    
    # Path to config_perfect.yaml (SOURCE CONFIG)
    perfect_config_path = Path("information_asymmetry_simulation/config_perfect.yaml")
    
    # Check if config exists
    if not perfect_config_path.exists():
        print(f"Error: Perfect config not found at {perfect_config_path}")
        sys.exit(1)
    
    print("="*60)
    print("PERFECT MODE EXPERIMENT CONFIGURATION")
    print("="*60)
    print(f"Source config: {perfect_config_path}")
    print(f"Agent behavior: All agents use theoretically optimal strategy")
    print(f"Number of agents: 10")
    print(f"Runs: 5 (in parallel)")
    print(f"Output directory: experiments/{args.output_dir}/")
    print("="*60)
    
    # Run the experiment
    run_perfect_experiment(args.output_dir)
    
    print("\n" + "="*60)
    print("PERFECT MODE EXPERIMENT COMPLETED")
    print(f"Results are stored in: experiments/{args.output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()