#!/usr/bin/env python3
"""
Recover and complete analysis for an experiment that had errors
"""

import argparse
import json
import yaml
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from experiment_framework.utils.registry import ExperimentRegistry
from experiment_framework.utils.statistics import StatisticalAggregator
from experiment_framework.utils.analysis_fixed import ExperimentAnalyzer


def recover_experiment(experiment_id: str, experiments_dir: str = "experiments"):
    """Recover and complete analysis for an experiment"""
    
    experiments_base = Path(experiments_dir)
    experiment_dir = experiments_base / experiment_id
    
    if not experiment_dir.exists():
        print(f"Error: Experiment {experiment_id} not found in {experiments_base}")
        return False
    
    print(f"Recovering experiment: {experiment_id}")
    
    # Load experiment config
    config_file = experiment_dir / "experiment_config.yaml"
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        return False
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Find completed runs
    runs_dir = experiment_dir / "runs"
    if not runs_dir.exists():
        print(f"Error: No runs directory found")
        return False
    
    # Collect results from all completed runs
    results = []
    completed_runs = 0
    failed_runs = 0
    
    for run_dir in sorted(runs_dir.iterdir()):
        if run_dir.is_dir():
            run_id = run_dir.name
            sim_results_file = run_dir / "simulation" / "results.yaml"
            
            if sim_results_file.exists():
                print(f"  Found completed run: {run_id}")
                
                # Load simulation results
                with open(sim_results_file, 'r') as f:
                    sim_results = yaml.safe_load(f)
                
                # Check for analysis results
                analysis_file = run_dir / "simulation" / "analysis_results.json"
                if analysis_file.exists():
                    with open(analysis_file, 'r') as f:
                        analysis_results = json.load(f)
                else:
                    print(f"    Warning: No analysis results found for {run_id}")
                    analysis_results = None
                
                results.append({
                    'run_id': run_id,
                    'status': 'completed',
                    'results': sim_results,
                    'analysis': analysis_results
                })
                completed_runs += 1
            else:
                print(f"  Run {run_id} appears incomplete")
                failed_runs += 1
    
    if not results:
        print("Error: No completed runs found")
        return False
    
    print(f"\nFound {completed_runs} completed runs and {failed_runs} incomplete runs")
    
    # Run statistical aggregation
    print("\nAggregating statistics...")
    try:
        aggregator = StatisticalAggregator(experiment_dir, results)
        aggregate_metrics = aggregator.aggregate()
        
        # Save aggregate metrics
        metrics_file = experiment_dir / "aggregate_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(aggregate_metrics, f, indent=2)
        print(f"Saved aggregate metrics to {metrics_file}")
        
    except Exception as e:
        print(f"Error during aggregation: {e}")
        aggregate_metrics = {
            'error': str(e),
            'total_runs': len(results),
            'successful_runs': completed_runs
        }
    
    # Generate analysis report
    if config.get('analysis', {}).get('generate_report', True):
        print("\nGenerating analysis report...")
        try:
            analyzer = ExperimentAnalyzer(experiment_dir, config, aggregate_metrics)
            report_path = analyzer.generate_report()
            print(f"Report saved to {report_path}")
        except Exception as e:
            print(f"Error generating report: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("RECOVERY SUMMARY")
    print("="*60)
    print(f"Experiment: {experiment_id}")
    print(f"Completed runs: {completed_runs}")
    print(f"Failed runs: {failed_runs}")
    
    if 'statistical_summary' in aggregate_metrics:
        print("\nKEY METRICS:")
        stats = aggregate_metrics['statistical_summary']
        
        # Show a few key metrics
        metrics_to_show = [
            'total_tasks_completed',
            'revenue_distribution.gini_coefficient',
            'communication_efficiency.messages_per_completed_task'
        ]
        
        for metric in metrics_to_show:
            if metric in stats:
                metric_data = stats[metric]
                if isinstance(metric_data, dict) and 'mean' in metric_data:
                    print(f"  {metric}: {metric_data['mean']:.3f} ± {metric_data.get('std', 0):.3f}")
    
    # Update registry if it exists
    registry_path = experiments_base / "registry.yaml"
    if registry_path.exists():
        print("\nUpdating registry...")
        registry = ExperimentRegistry(registry_path)
        
        key_metrics = {}
        if 'statistical_summary' in aggregate_metrics:
            stats = aggregate_metrics['statistical_summary']
            if 'revenue_distribution.gini_coefficient' in stats:
                gini = stats['revenue_distribution.gini_coefficient']
                if isinstance(gini, dict):
                    key_metrics['avg_gini'] = gini.get('mean')
            if 'total_tasks_completed' in stats:
                tasks = stats['total_tasks_completed']
                if isinstance(tasks, dict):
                    key_metrics['avg_tasks_completed'] = tasks.get('mean')
        
        registry.update_experiment_status(experiment_id, 'completed', key_metrics)
        print("Registry updated")
    
    print("\nRecovery complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Recover and complete analysis for an experiment')
    parser.add_argument('experiment_id', help='Experiment ID to recover (e.g., exp_006_baseline_o3mini)')
    parser.add_argument('--experiments-dir', default='experiments', 
                       help='Directory containing experiments')
    
    args = parser.parse_args()
    
    success = recover_experiment(args.experiment_id, args.experiments_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()