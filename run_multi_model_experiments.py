#!/usr/bin/env python3
"""
Script to run multi-model experiments with show_full_revenue variations
Saves results to experiments/multi_model directory following mixed_mode pattern
"""

import yaml
import subprocess
import sys
from pathlib import Path
import argparse
from typing import List
import time

# Define the models you want to test
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

def load_config(config_path: Path) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: dict, config_path: Path):
    """Save YAML configuration file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def create_experiment_config_for_model(base_config: Path, model: str, show_full_revenue: bool) -> dict:
    """Create experiment framework config for multi-model experiment with specific settings"""
    
    # Load the base config from information_asymmetry_simulation
    sim_config = load_config(base_config)
    
    # Update the model for the agents
    sim_config['agents']['model'] = model
    
    # Update show_full_revenue setting
    sim_config['simulation']['show_full_revenue'] = show_full_revenue
    
    # Create a short version of model name for the experiment name
    if model == "o3-mini-2025-01-31":
        model_short = "o3mini"
    elif model == "o3":
        model_short = "o3"
    elif model == "gpt-4.1-mini":
        model_short = "gpt41mini"
    elif model == "gpt-5-mini":
        model_short = "gpt5mini"
    elif model == "deepseek-ai/DeepSeek-R1-0528-Turbo":
        model_short = "deepseek"
    elif model == "claude-sonnet-4-20250514":
        model_short = "claudesonnet"
    elif model == "google/gemini-2.5-pro":
        model_short = "gemini25pro"
    elif model == "google/gemini-2.5-flash":
        model_short = "gemini25flash"
    elif model == "x-ai/grok-4":
        model_short = "grok4"
    else:
        # Fallback: replace special chars
        model_short = model.replace('-', '').replace('.', '').replace('/', '')[:10]
    
    # Create revenue visibility label
    revenue_visibility = "full" if show_full_revenue else "limited"
    
    config = {
        'experiment': {
            'name': f'multi_model_{model_short}_{revenue_visibility}_revenue',
            'description': f"Multi-model experiment with {model}.\n" + 
                          f"Revenue visibility: {'Full (all agents see all revenue)' if show_full_revenue else 'Limited (agents only see their own revenue position)'}.\n" +
                          "Testing model performance with standard information asymmetry setup.",
            'num_runs': 5,
            'parallel': True,  # Enable parallel execution of the 5 runs
            'max_workers': 5,  # Run all 5 simulations in parallel
            'tags': ['multi-model', model_short, f'{revenue_visibility}-revenue']
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

def run_experiment_with_model(base_config_path: Path, model: str, show_full_revenue: bool, experiments_subdir: str = "multi_model_20_agents"):
    """Run a multi-model experiment with specific settings"""
    
    # Create the experiment config for this model and revenue setting
    config = create_experiment_config_for_model(base_config_path, model, show_full_revenue)
    
    # Create a temporary config file for this run
    temp_config_dir = Path("experiment_framework/configs/temp")
    temp_config_dir.mkdir(exist_ok=True)
    
    revenue_label = "full" if show_full_revenue else "limited"
    temp_config_path = temp_config_dir / f"multi_{model.replace('/', '_').replace('.', '_')}_{revenue_label}.yaml"
    save_config(config, temp_config_path)
    
    print(f"\n{'='*60}")
    print(f"Running multi-model experiment")
    print(f"Model: {model}")
    print(f"Revenue visibility: {revenue_label}")
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
            print(f"✓ Successfully completed experiment with {model} ({revenue_label} revenue)")
        else:
            print(f"✗ Failed to run experiment with {model} ({revenue_label} revenue) - exit code: {process.returncode}")
            
    except Exception as e:
        print(f"✗ Error running experiment with {model} ({revenue_label} revenue): {str(e)}")
    
    # Clean up temp config
    temp_config_path.unlink()

def main():
    parser = argparse.ArgumentParser(description='Run multi-model experiments with revenue visibility variations')
    parser.add_argument('--revenue', choices=['full', 'limited', 'both'], 
                       default='both',
                       help='Revenue visibility mode: full (agents see all), limited (agents see only their position), or both (default: both)')
    parser.add_argument('--models', nargs='+', 
                       default=None,
                       help='List of models to test (default: uses predefined list)')
    parser.add_argument('--sequential', action='store_true',
                       help='Add delay between experiments (default: no delay)')
    parser.add_argument('--config', type=str,
                       default='information_asymmetry_simulation/config.yaml',
                       help='Base config file to use (default: information_asymmetry_simulation/config.yaml)')
    
    args = parser.parse_args()
    
    # Use provided models or default list
    models = args.models if args.models else MODELS_TO_TEST
    
    # Path to base config (SOURCE CONFIG)
    base_config_path = Path(args.config)
    
    # Check if config exists
    if not base_config_path.exists():
        print(f"Error: Base config not found at {base_config_path}")
        print("Available configs in information_asymmetry_simulation:")
        print("  - config.yaml (standard LLM config)")
        print("  - config_mixed.yaml (mixed mode with 1 LLM + 9 perfect agents)")
        print("  - config_perfect.yaml (all perfect agents)")
        sys.exit(1)
    
    print("="*60)
    print("MULTI-MODEL EXPERIMENT CONFIGURATION")
    print("="*60)
    print(f"Source config: {base_config_path}")
    print(f"Models to test: {models}")
    print(f"Revenue visibility modes: {args.revenue}")
    print(f"Runs per configuration: 5 (in parallel)")
    print(f"Output directory: experiments/multi_model_20_agents/")
    print("="*60)
    
    # Run experiments based on revenue visibility mode
    if args.revenue == 'full':
        for i, model in enumerate(models):
            run_experiment_with_model(base_config_path, model, show_full_revenue=True)
            if args.sequential and i < len(models) - 1:
                print(f"\nWaiting 5 seconds before next model...")
                time.sleep(5)
                
    elif args.revenue == 'limited':
        for i, model in enumerate(models):
            run_experiment_with_model(base_config_path, model, show_full_revenue=False)
            if args.sequential and i < len(models) - 1:
                print(f"\nWaiting 5 seconds before next model...")
                time.sleep(5)
                
    elif args.revenue == 'both':
        # Run all full revenue experiments first, then all limited
        print("\n" + "="*60)
        print("RUNNING EXPERIMENTS WITH FULL REVENUE VISIBILITY")
        print("(All agents can see everyone's revenue)")
        print("="*60)
        for i, model in enumerate(models):
            run_experiment_with_model(base_config_path, model, show_full_revenue=True)
            if args.sequential and i < len(models) - 1:
                time.sleep(5)
        
        print("\n" + "="*60)
        print("RUNNING EXPERIMENTS WITH LIMITED REVENUE VISIBILITY")
        print("(Agents only see their own revenue position)")
        print("="*60)
        for i, model in enumerate(models):
            run_experiment_with_model(base_config_path, model, show_full_revenue=False)
            if args.sequential and i < len(models) - 1:
                time.sleep(5)
    
    print("\n" + "="*60)
    print("ALL MULTI-MODEL EXPERIMENTS COMPLETED")
    print("Results are stored in: experiments/multi_model_20_agents/")
    print("="*60)

if __name__ == "__main__":
    main()