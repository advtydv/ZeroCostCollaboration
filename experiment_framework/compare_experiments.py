#!/usr/bin/env python3
"""
Compare multiple experiments side by side
"""

import argparse
import json
import yaml
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from experiment_framework.utils.registry import ExperimentRegistry


def load_experiment_metrics(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """Load aggregate metrics for an experiment"""
    metrics_file = exp_dir / "aggregate_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None


def extract_metric_value(metrics: Dict[str, Any], metric_path: str) -> Optional[float]:
    """Extract a metric value from nested dictionary using dot notation"""
    if not metrics or 'statistical_summary' not in metrics:
        return None
    
    parts = metric_path.split('.')
    current = metrics['statistical_summary']
    
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    
    # If we got a dict with 'mean', return that
    if isinstance(current, dict) and 'mean' in current:
        return current['mean']
    
    # If it's a number, return it
    if isinstance(current, (int, float)):
        return current
    
    return None


def format_value(value: Optional[float], precision: int = 3) -> str:
    """Format a value for display"""
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def format_value_with_std(metrics: Dict[str, Any], metric_path: str, precision: int = 3) -> str:
    """Format a value with standard deviation"""
    if not metrics or 'statistical_summary' not in metrics:
        return "N/A"
    
    parts = metric_path.split('.')
    current = metrics['statistical_summary']
    
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return "N/A"
    
    if isinstance(current, dict) and 'mean' in current:
        mean = current['mean']
        std = current.get('std', 0)
        return f"{mean:.{precision}f} ± {std:.{precision}f}"
    
    return "N/A"


def create_comparison_table(experiments: List[str], metrics_to_compare: List[str], 
                           experiments_dir: Path, include_std: bool = False) -> List[List[str]]:
    """Create a comparison table of experiments"""
    table = []
    
    # Header row
    header = ["Metric"] + experiments
    table.append(header)
    
    # Load all experiment metrics
    exp_metrics = {}
    for exp_id in experiments:
        exp_dir = experiments_dir / exp_id
        metrics = load_experiment_metrics(exp_dir)
        exp_metrics[exp_id] = metrics
    
    # Add rows for each metric
    for metric_path in metrics_to_compare:
        row = [metric_path]
        
        for exp_id in experiments:
            if include_std:
                value_str = format_value_with_std(exp_metrics[exp_id], metric_path)
            else:
                value = extract_metric_value(exp_metrics[exp_id], metric_path)
                value_str = format_value(value)
            row.append(value_str)
        
        table.append(row)
    
    return table


def print_comparison_table(table: List[List[str]]):
    """Print a formatted comparison table"""
    # Calculate column widths
    col_widths = []
    for col_idx in range(len(table[0])):
        max_width = max(len(str(row[col_idx])) for row in table)
        col_widths.append(max_width)
    
    # Print header
    header = table[0]
    header_str = " | ".join(str(header[i]).ljust(col_widths[i]) for i in range(len(header)))
    print(header_str)
    print("-" * len(header_str))
    
    # Print data rows
    for row in table[1:]:
        row_str = " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row)))
        print(row_str)


def save_comparison_csv(table: List[List[str]], output_path: Path):
    """Save comparison table as CSV"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(table)


def main():
    parser = argparse.ArgumentParser(description='Compare multiple experiments')
    parser.add_argument('experiments', nargs='+', help='Experiment IDs to compare')
    parser.add_argument('--metrics', type=str, 
                       default='total_tasks_completed,revenue_distribution.gini_coefficient,communication_efficiency.messages_per_completed_task',
                       help='Comma-separated list of metrics to compare')
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                       help='Directory containing experiments')
    parser.add_argument('--output', type=str, help='Output CSV file')
    parser.add_argument('--include-std', action='store_true', 
                       help='Include standard deviation in values')
    parser.add_argument('--show-config', action='store_true',
                       help='Also show configuration differences')
    
    args = parser.parse_args()
    
    experiments_dir = Path(args.experiments_dir)
    
    # Verify all experiments exist
    for exp_id in args.experiments:
        exp_dir = experiments_dir / exp_id
        if not exp_dir.exists():
            print(f"Error: Experiment {exp_id} not found")
            sys.exit(1)
    
    # Parse metrics
    metrics_to_compare = [m.strip() for m in args.metrics.split(',')]
    
    # Load registry for additional information
    registry_path = experiments_dir / "registry.yaml"
    registry = ExperimentRegistry(registry_path) if registry_path.exists() else None
    
    # Print experiment information
    print("\n" + "="*60)
    print("EXPERIMENT COMPARISON")
    print("="*60)
    
    # Show basic info for each experiment
    for exp_id in args.experiments:
        if registry:
            exp_info = registry.get_experiment_info(exp_id)
            if exp_info:
                print(f"\n{exp_id}:")
                print(f"  Status: {exp_info['status']}")
                print(f"  Model: {exp_info['config_summary']['model']}")
                print(f"  Agents: {exp_info['config_summary']['num_agents']}")
                print(f"  Description: {exp_info['description'][:80]}...")
    
    # Show configuration differences if requested
    if args.show_config:
        print("\n" + "="*60)
        print("CONFIGURATION COMPARISON")
        print("="*60)
        
        configs = {}
        for exp_id in args.experiments:
            config_file = experiments_dir / exp_id / "experiment_config.yaml"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    configs[exp_id] = yaml.safe_load(f)
        
        # Compare key configuration parameters
        config_params = [
            ('Model', ['simulation_config', 'agents', 'model']),
            ('Agents', ['simulation_config', 'simulation', 'agents']),
            ('Rounds', ['simulation_config', 'simulation', 'rounds']),
            ('Uncooperative', ['simulation_config', 'agents', 'uncooperative_count']),
            ('Show Full Revenue', ['simulation_config', 'simulation', 'show_full_revenue']),
            ('Task Revenue', ['simulation_config', 'revenue', 'task_completion']),
            ('First Bonus', ['simulation_config', 'revenue', 'bonus_for_first'])
        ]
        
        config_table = [["Parameter"] + args.experiments]
        
        for param_name, path in config_params:
            row = [param_name]
            for exp_id in args.experiments:
                if exp_id in configs:
                    value = configs[exp_id]
                    for key in path:
                        if isinstance(value, dict) and key in value:
                            value = value[key]
                        else:
                            value = "N/A"
                            break
                    row.append(str(value))
                else:
                    row.append("N/A")
            config_table.append(row)
        
        print()
        print_comparison_table(config_table)
    
    # Create metrics comparison table
    print("\n" + "="*60)
    print("METRICS COMPARISON")
    print("="*60)
    print()
    
    table = create_comparison_table(args.experiments, metrics_to_compare, 
                                   experiments_dir, args.include_std)
    
    print_comparison_table(table)
    
    # Save to CSV if requested
    if args.output:
        output_path = Path(args.output)
        save_comparison_csv(table, output_path)
        print(f"\nComparison saved to {output_path}")
    
    # Statistical comparison (if more than one experiment)
    if len(args.experiments) > 1:
        print("\n" + "="*60)
        print("STATISTICAL COMPARISON")
        print("="*60)
        
        # Load raw values for statistical tests
        raw_values = {}
        for exp_id in args.experiments:
            metrics_file = experiments_dir / exp_id / "aggregate_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    if 'raw_values' in metrics:
                        raw_values[exp_id] = metrics['raw_values']
        
        # Compare first metric as example
        if metrics_to_compare and raw_values:
            metric = metrics_to_compare[0]
            print(f"\nMetric: {metric}")
            
            # Find best performing experiment
            mean_values = {}
            for exp_id in args.experiments:
                if exp_id in raw_values and metric in raw_values[exp_id]:
                    mean_values[exp_id] = np.mean(raw_values[exp_id][metric])
            
            if mean_values:
                best_exp = max(mean_values, key=mean_values.get)
                worst_exp = min(mean_values, key=mean_values.get)
                
                if best_exp != worst_exp:
                    improvement = (mean_values[best_exp] - mean_values[worst_exp]) / abs(mean_values[worst_exp]) * 100
                    print(f"Best: {best_exp} ({mean_values[best_exp]:.3f})")
                    print(f"Worst: {worst_exp} ({mean_values[worst_exp]:.3f})")
                    print(f"Improvement: {improvement:.1f}%")


if __name__ == "__main__":
    main()