"""
Analysis and reporting tools for experiments - Fixed version
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


class ExperimentAnalyzer:
    """Generates analysis reports and visualizations for experiments"""
    
    def __init__(self, experiment_dir: Path, config: Dict[str, Any], 
                 aggregate_metrics: Dict[str, Any]):
        self.experiment_dir = Path(experiment_dir)
        self.config = config
        self.aggregate_metrics = aggregate_metrics
        self.experiment_id = experiment_dir.name
    
    def generate_report(self) -> Path:
        """Generate comprehensive markdown report"""
        report_path = self.experiment_dir / "analysis_report.md"
        
        # Build complete report content in memory first
        content = []
        
        # Header
        content.append(f"# Experiment Analysis Report: {self.experiment_id}\n")
        content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Experiment Overview
        content.append("\n## Experiment Overview\n")
        content.append(f"**Description:** {self.config['experiment']['description']}\n")
        content.append(f"**Tags:** {', '.join(self.config['experiment'].get('tags', []))}\n")
        
        # Configuration Summary
        content.append("\n## Configuration Summary\n")
        sim_config = self.config['simulation_config']
        content.append(f"- **Model:** {sim_config['agents']['model']}")
        content.append(f"- **Number of Agents:** {sim_config['simulation']['agents']}")
        content.append(f"- **Rounds:** {sim_config['simulation']['rounds']}")
        content.append(f"- **Uncooperative Agents:** {sim_config['agents'].get('uncooperative_count', 0)}")
        content.append(f"- **Show Full Revenue:** {sim_config['simulation'].get('show_full_revenue', False)}")
        content.append(f"- **Task Completion Revenue:** ${sim_config['revenue']['task_completion']:,}")
        content.append(f"- **First Completion Bonus:** ${sim_config['revenue']['bonus_for_first']:,}")
        
        # Run Summary
        content.append("\n## Run Summary\n")
        content.append(f"- **Total Runs:** {self.aggregate_metrics.get('total_runs', 0)}")
        content.append(f"- **Successful Runs:** {self.aggregate_metrics.get('successful_runs', 0)}")
        content.append(f"- **Failed Runs:** {self.aggregate_metrics.get('failed_runs', 0)}")
        
        # Statistical Summary
        if 'statistical_summary' in self.aggregate_metrics:
            content.append("\n## Statistical Summary\n")
            content.append("### Key Metrics (Mean ± Std [95% CI])\n")
            
            stats = self.aggregate_metrics['statistical_summary']
            
            # Create table
            content.append("| Metric | Mean | Std Dev | 95% CI | Min | Max |")
            content.append("|--------|------|---------|--------|-----|-----|")
            
            for metric_name in self.config['analysis'].get('key_metrics', []):
                if metric_name in stats:
                    metric = stats[metric_name]
                    if isinstance(metric, dict) and 'mean' in metric:
                        ci_low, ci_high = metric.get('ci_95', [metric['mean'], metric['mean']])
                        content.append(f"| {self._format_metric_name(metric_name)} | "
                                     f"{metric['mean']:.3f} | "
                                     f"{metric.get('std', 0):.3f} | "
                                     f"[{ci_low:.3f}, {ci_high:.3f}] | "
                                     f"{metric.get('min', 0):.3f} | "
                                     f"{metric.get('max', 0):.3f} |")
            
            # Additional metrics
            content.append("\n### Additional Metrics\n")
            for metric_name, metric in stats.items():
                if metric_name not in self.config['analysis'].get('key_metrics', []):
                    if isinstance(metric, dict) and 'mean' in metric:
                        content.append(f"- **{self._format_metric_name(metric_name)}:** "
                                     f"{metric['mean']:.3f} ± {metric.get('std', 0):.3f}")
        
        # Convergence Analysis
        if 'convergence_analysis' in self.aggregate_metrics:
            conv = self.aggregate_metrics['convergence_analysis']
            if conv:
                content.append("\n## Convergence Analysis\n")
                if 'steady_state_round' in conv:
                    content.append(f"- **Steady State Round:** {conv['steady_state_round']}")
                if 'variance_reduction' in conv:
                    content.append(f"- **Variance Reduction:** {conv['variance_reduction']:.2%}")
        
        # Per-Round Evolution
        if 'per_round_evolution' in self.aggregate_metrics:
            evolution = self.aggregate_metrics['per_round_evolution']
            if evolution:
                content.append("\n## Per-Round Evolution\n")
                
                # Extract round numbers and sort
                rounds = sorted([int(k.split('_')[1]) for k in evolution.keys() 
                               if k.startswith('round_')])
                
                if rounds:
                    content.append("### Tasks Completed Per Round\n")
                    content.append("| Round | Mean | Std Dev |")
                    content.append("|-------|------|---------|")
                    
                    for round_num in rounds[:5]:  # Show first 5 rounds
                        round_key = f'round_{round_num}'
                        if round_key in evolution and 'tasks_completed' in evolution[round_key]:
                            tc = evolution[round_key]['tasks_completed']
                            content.append(f"| {round_num} | {tc['mean']:.2f} | {tc.get('std', 0):.2f} |")
                    
                    if len(rounds) > 5:
                        content.append(f"| ... | ... | ... |")
                        # Show last round
                        round_key = f'round_{rounds[-1]}'
                        if round_key in evolution and 'tasks_completed' in evolution[round_key]:
                            tc = evolution[round_key]['tasks_completed']
                            content.append(f"| {rounds[-1]} | {tc['mean']:.2f} | {tc.get('std', 0):.2f} |")
        
        # Raw Values Distribution
        if 'raw_values' in self.aggregate_metrics:
            content.append("\n## Raw Value Distributions\n")
            content.append("Individual run values for key metrics:\n")
            
            for metric_name in self.config['analysis'].get('key_metrics', []):
                if metric_name in self.aggregate_metrics['raw_values']:
                    values = self.aggregate_metrics['raw_values'][metric_name]
                    content.append(f"**{self._format_metric_name(metric_name)}:** "
                                 f"{', '.join([f'{v:.3f}' for v in values])}\n")
        
        # Generate plots if requested
        if self.config['analysis'].get('generate_plots', True):
            content.append("\n## Visualizations\n")
            try:
                self._generate_plots_safe()
                if (self.experiment_dir / "plots" / "distributions.png").exists():
                    content.append("![Distribution Plot](plots/distributions.png)\n")
                if (self.experiment_dir / "plots" / "evolution.png").exists():
                    content.append("![Evolution Plot](plots/evolution.png)\n")
            except Exception as e:
                content.append(f"_Error generating plots: {e}_\n")
        
        # Notes and Observations
        content.append("\n## Notes and Observations\n")
        content.append("_Add your observations and interpretations here_\n")
        
        # Metadata
        content.append("\n## Experiment Metadata\n")
        metadata_file = self.experiment_dir / "metadata.yaml"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = yaml.safe_load(f)
            
            content.append(f"- **Created:** {metadata.get('created_at', 'Unknown')}")
            content.append(f"- **Git Commit:** {metadata.get('git', {}).get('commit', 'Unknown')[:8]}")
            content.append(f"- **Git Branch:** {metadata.get('git', {}).get('branch', 'Unknown')}")
            content.append(f"- **Uncommitted Changes:** {metadata.get('git', {}).get('has_uncommitted_changes', False)}")
            content.append(f"- **Python Version:** {metadata.get('environment', {}).get('python_version', 'Unknown')}")
        
        # Write complete content to file at once
        with open(report_path, 'w') as f:
            f.write('\n'.join(content))
        
        return report_path
    
    def _format_metric_name(self, metric_name: str) -> str:
        """Format metric name for display"""
        formatted = metric_name.replace('.', ' - ').replace('_', ' ')
        return formatted.title()
    
    def _generate_plots_safe(self):
        """Generate visualization plots with error handling"""
        plots_dir = self.experiment_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self._plot_distributions_safe(plots_dir)
        except Exception as e:
            print(f"Warning: Could not generate distribution plot: {e}")
        
        try:
            self._plot_evolution_safe(plots_dir)
        except Exception as e:
            print(f"Warning: Could not generate evolution plot: {e}")
    
    def _plot_distributions_safe(self, plots_dir: Path):
        """Plot distributions of key metrics with proper error handling"""
        if 'raw_values' not in self.aggregate_metrics:
            return
        
        key_metrics = self.config['analysis'].get('key_metrics', [])
        raw_values = self.aggregate_metrics['raw_values']
        
        # Filter to only plottable metrics
        metrics_to_plot = []
        for m in key_metrics:
            if m in raw_values:
                vals = raw_values[m]
                if vals and all(isinstance(v, (int, float)) for v in vals):
                    metrics_to_plot.append(m)
        
        if not metrics_to_plot:
            return
        
        n_metrics = len(metrics_to_plot)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        # Create figure
        fig = plt.figure(figsize=(5*n_cols, 4*n_rows))
        
        for idx, metric_name in enumerate(metrics_to_plot):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            
            values = raw_values[metric_name]
            
            # Create box plot
            bp = ax.boxplot(values, vert=True)
            
            # Add individual points
            x = np.ones(len(values))
            ax.scatter(x, values, alpha=0.5, s=50, zorder=3)
            
            # Add mean line
            mean_val = np.mean(values)
            ax.axhline(y=mean_val, color='r', linestyle='--', alpha=0.5)
            
            # Labels
            ax.set_title(self._format_metric_name(metric_name), fontsize=10)
            ax.set_ylabel('Value', fontsize=9)
            ax.set_xticklabels([''])
            ax.text(0.02, 0.98, f'Mean: {mean_val:.3f}', 
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Metric Distributions - {self.experiment_id}', fontsize=12)
        plt.tight_layout()
        plt.savefig(plots_dir / 'distributions.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    def _plot_evolution_safe(self, plots_dir: Path):
        """Plot evolution of metrics over rounds with proper error handling"""
        if 'per_round_evolution' not in self.aggregate_metrics:
            return
        
        evolution = self.aggregate_metrics['per_round_evolution']
        if not evolution:
            return
        
        # Extract round numbers
        rounds = sorted([int(k.split('_')[1]) for k in evolution.keys() 
                       if k.startswith('round_')])
        
        if not rounds:
            return
        
        # Prepare data
        tasks_means = []
        tasks_stds = []
        messages_means = []
        messages_stds = []
        
        for round_num in rounds:
            round_key = f'round_{round_num}'
            if round_key in evolution:
                if 'tasks_completed' in evolution[round_key]:
                    tasks_means.append(evolution[round_key]['tasks_completed'].get('mean', 0))
                    tasks_stds.append(evolution[round_key]['tasks_completed'].get('std', 0))
                else:
                    tasks_means.append(0)
                    tasks_stds.append(0)
                
                if 'messages_sent' in evolution[round_key]:
                    messages_means.append(evolution[round_key]['messages_sent'].get('mean', 0))
                    messages_stds.append(evolution[round_key]['messages_sent'].get('std', 0))
                else:
                    messages_means.append(0)
                    messages_stds.append(0)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Tasks completed
        if tasks_means:
            ax1.errorbar(rounds, tasks_means, yerr=tasks_stds, 
                        marker='o', capsize=5, capthick=1, linewidth=1.5)
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Tasks Completed')
            ax1.set_title('Tasks Completed Per Round')
            ax1.grid(True, alpha=0.3)
        
        # Messages sent
        if messages_means:
            ax2.errorbar(rounds, messages_means, yerr=messages_stds, 
                        marker='s', capsize=5, capthick=1, linewidth=1.5, color='orange')
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Messages Sent')
            ax2.set_title('Messages Sent Per Round')
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Metric Evolution - {self.experiment_id}', fontsize=12)
        plt.tight_layout()
        plt.savefig(plots_dir / 'evolution.png', dpi=100, bbox_inches='tight')
        plt.close()
