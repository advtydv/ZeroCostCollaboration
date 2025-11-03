#!/usr/bin/env python3
"""
Find experiments matching specific criteria
"""

import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import sys

sys.path.append(str(Path(__file__).parent.parent))
from experiment_framework.utils.registry import ExperimentRegistry


def format_experiment_info(exp_info: Dict[str, Any], exp_id: str, verbose: bool = False) -> str:
    """Format experiment information for display"""
    lines = []
    
    # Basic info
    lines.append(f"\n{exp_id}")
    lines.append("-" * len(exp_id))
    lines.append(f"Status: {exp_info['status']}")
    lines.append(f"Created: {exp_info['created'][:19]}")  # Trim microseconds
    lines.append(f"Description: {exp_info['description'][:100]}...")
    
    if verbose:
        # Configuration details
        config = exp_info['config_summary']
        lines.append(f"Model: {config['model']}")
        lines.append(f"Agents: {config['num_agents']}")
        lines.append(f"Rounds: {config['rounds']}")
        lines.append(f"Uncooperative: {config['uncooperative_count']}")
        
        # Git info
        lines.append(f"Git: {exp_info['git_branch']} @ {exp_info['git_commit'][:8]}")
        if exp_info.get('has_uncommitted'):
            lines.append("  ⚠️  Has uncommitted changes")
        
        # Results if completed
        if exp_info['status'] == 'completed' and 'key_metrics' in exp_info:
            metrics = exp_info['key_metrics']
            if metrics:
                lines.append("Results:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        lines.append(f"  {key}: {value:.3f}")
                    else:
                        lines.append(f"  {key}: {value}")
    
    # Tags
    if exp_info.get('tags'):
        lines.append(f"Tags: {', '.join(exp_info['tags'])}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Find experiments matching criteria')
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                       help='Directory containing experiments')
    parser.add_argument('--model', type=str, help='Filter by model name')
    parser.add_argument('--status', type=str, choices=['initialized', 'running', 'completed', 'failed'],
                       help='Filter by status')
    parser.add_argument('--tag', type=str, help='Filter by tag')
    parser.add_argument('--after', type=str, help='Filter by date (YYYY-MM-DD)')
    parser.add_argument('--before', type=str, help='Filter by date (YYYY-MM-DD)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed information')
    parser.add_argument('--list-all', action='store_true', help='List all experiments')
    
    args = parser.parse_args()
    
    # Load registry
    registry_path = Path(args.experiments_dir) / "registry.yaml"
    if not registry_path.exists():
        print(f"No experiments found in {args.experiments_dir}")
        return
    
    registry = ExperimentRegistry(registry_path)
    
    # Build filters
    filters = {}
    if args.model:
        filters['model'] = args.model
    if args.status:
        filters['status'] = args.status
    if args.tag:
        filters['tag'] = args.tag
    if args.after:
        filters['after'] = args.after + "T00:00:00"
    if args.before:
        filters['before'] = args.before + "T23:59:59"
    
    # Find experiments
    if args.list_all:
        experiments = registry.list_all_experiments()
        matching_ids = [exp['id'] for exp in experiments]
    else:
        matching_ids = registry.find_experiments(**filters)
    
    if not matching_ids:
        print("No experiments found matching the criteria")
        return
    
    # Display results
    print(f"\nFound {len(matching_ids)} experiment(s):")
    
    for exp_id in matching_ids:
        exp_info = registry.get_experiment_info(exp_id)
        if exp_info:
            print(format_experiment_info(exp_info, exp_id, args.verbose))
    
    # Summary statistics
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    # Count by status
    status_counts = {}
    for exp_id in matching_ids:
        exp_info = registry.get_experiment_info(exp_id)
        status = exp_info['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("By Status:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    
    # Count by model
    model_counts = {}
    for exp_id in matching_ids:
        exp_info = registry.get_experiment_info(exp_id)
        model = exp_info['config_summary']['model']
        model_counts[model] = model_counts.get(model, 0) + 1
    
    if len(model_counts) > 1:
        print("\nBy Model:")
        for model, count in sorted(model_counts.items()):
            print(f"  {model}: {count}")


if __name__ == "__main__":
    main()