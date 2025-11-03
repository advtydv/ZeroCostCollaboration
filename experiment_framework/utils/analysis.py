"""
Analysis and reporting tools for experiments
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
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
        
        # Build report content first to avoid file I/O issues
        report_content = []
        
        # Header
        report_content.append(f"# Experiment Analysis Report: {self.experiment_id}\n\n")
        report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Experiment Overview
            f.write("## Experiment Overview\n\n")
            f.write(f"**Description:** {self.config['experiment']['description']}\n\n")
            f.write(f"**Tags:** {', '.join(self.config['experiment'].get('tags', []))}\n\n")
            
            # Configuration Summary
            f.write("## Configuration Summary\n\n")
            sim_config = self.config['simulation_config']
            f.write(f"- **Model:** {sim_config['agents']['model']}\n")
            f.write(f"- **Number of Agents:** {sim_config['simulation']['agents']}\n")
            f.write(f"- **Rounds:** {sim_config['simulation']['rounds']}\n")
            f.write(f"- **Uncooperative Agents:** {sim_config['agents'].get('uncooperative_count', 0)}\n")
            f.write(f"- **Show Full Revenue:** {sim_config['simulation'].get('show_full_revenue', False)}\n")
            f.write(f"- **Task Completion Revenue:** ${sim_config['revenue']['task_completion']:,}\n")
            f.write(f"- **First Completion Bonus:** ${sim_config['revenue']['bonus_for_first']:,}\n\n")
            
            # Run Summary
            f.write("## Run Summary\n\n")
            f.write(f"- **Total Runs:** {self.aggregate_metrics.get('total_runs', 0)}\n")
            f.write(f"- **Successful Runs:** {self.aggregate_metrics.get('successful_runs', 0)}\n")
            f.write(f"- **Failed Runs:** {self.aggregate_metrics.get('failed_runs', 0)}\n\n")
            
            # Statistical Summary
            if 'statistical_summary' in self.aggregate_metrics:
                f.write("## Statistical Summary\n\n")
                f.write("### Key Metrics (Mean ± Std [95% CI])\n\n")
                
                stats = self.aggregate_metrics['statistical_summary']
                
                # Create table
                f.write("| Metric | Mean | Std Dev | 95% CI | Min | Max |\n")
                f.write("|--------|------|---------|--------|-----|-----|\n")
                
                for metric_name in self.config['analysis'].get('key_metrics', []):
                    if metric_name in stats:
                        metric = stats[metric_name]
                        if isinstance(metric, dict) and 'mean' in metric:
                            ci_low, ci_high = metric.get('ci_95', [metric['mean'], metric['mean']])
                            f.write(f"| {self._format_metric_name(metric_name)} | "
                                   f"{metric['mean']:.3f} | "
                                   f"{metric.get('std', 0):.3f} | "
                                   f"[{ci_low:.3f}, {ci_high:.3f}] | "
                                   f"{metric.get('min', 0):.3f} | "
                                   f"{metric.get('max', 0):.3f} |\n")
                
                f.write("\n")
                
                # Additional metrics
                f.write("### Additional Metrics\n\n")
                for metric_name, metric in stats.items():
                    if metric_name not in self.config['analysis'].get('key_metrics', []):
                        if isinstance(metric, dict) and 'mean' in metric:
                            f.write(f"- **{self._format_metric_name(metric_name)}:** "
                                   f"{metric['mean']:.3f} ± {metric.get('std', 0):.3f}\n")
                f.write("\n")
            
            # Convergence Analysis
            if 'convergence_analysis' in self.aggregate_metrics:
                conv = self.aggregate_metrics['convergence_analysis']
                if conv:
                    f.write("## Convergence Analysis\n\n")
                    if 'steady_state_round' in conv:
                        f.write(f"- **Steady State Round:** {conv['steady_state_round']}\n")
                    if 'variance_reduction' in conv:
                        f.write(f"- **Variance Reduction:** {conv['variance_reduction']:.2%}\n")
                    f.write("\n")
            
            # Per-Round Evolution
            if 'per_round_evolution' in self.aggregate_metrics:
                evolution = self.aggregate_metrics['per_round_evolution']
                if evolution:
                    f.write("## Per-Round Evolution\n\n")
                    
                    # Extract round numbers and sort
                    rounds = sorted([int(k.split('_')[1]) for k in evolution.keys() 
                                   if k.startswith('round_')])
                    
                    if rounds:
                        f.write("### Tasks Completed Per Round\n\n")
                        f.write("| Round | Mean | Std Dev |\n")
                        f.write("|-------|------|---------||\n")
                        
                        for round_num in rounds[:5]:  # Show first 5 rounds
                            round_key = f'round_{round_num}'
                            if round_key in evolution and 'tasks_completed' in evolution[round_key]:
                                tc = evolution[round_key]['tasks_completed']
                                f.write(f"| {round_num} | {tc['mean']:.2f} | {tc.get('std', 0):.2f} |\n")
                        
                        if len(rounds) > 5:
                            f.write(f"| ... | ... | ... |\n")
                            # Show last round
                            round_key = f'round_{rounds[-1]}'
                            if round_key in evolution and 'tasks_completed' in evolution[round_key]:
                                tc = evolution[round_key]['tasks_completed']
                                f.write(f"| {rounds[-1]} | {tc['mean']:.2f} | {tc.get('std', 0):.2f} |\n")
                        
                        f.write("\n")
            
            # Raw Values Distribution
            if 'raw_values' in self.aggregate_metrics:
                f.write("## Raw Value Distributions\n\n")
                f.write("Individual run values for key metrics:\n\n")
                
                for metric_name in self.config['analysis'].get('key_metrics', []):
                    if metric_name in self.aggregate_metrics['raw_values']:
                        values = self.aggregate_metrics['raw_values'][metric_name]
                        f.write(f"**{self._format_metric_name(metric_name)}:** "
                               f"{', '.join([f'{v:.3f}' for v in values])}\n\n")
            
            # Notes and Observations
            f.write("## Notes and Observations\n\n")
            f.write("_Add your observations and interpretations here_\n\n")
            
            # Metadata
            f.write("## Experiment Metadata\n\n")
            metadata_file = self.experiment_dir / "metadata.yaml"
            if metadata_file.exists():
                with open(metadata_file, 'r') as mf:
                    metadata = yaml.safe_load(mf)
                
                f.write(f"- **Created:** {metadata.get('created_at', 'Unknown')}\n")
                f.write(f"- **Git Commit:** {metadata.get('git', {}).get('commit', 'Unknown')[:8]}\n")
                f.write(f"- **Git Branch:** {metadata.get('git', {}).get('branch', 'Unknown')}\n")
                f.write(f"- **Uncommitted Changes:** {metadata.get('git', {}).get('has_uncommitted_changes', False)}\n")
                f.write(f"- **Python Version:** {metadata.get('environment', {}).get('python_version', 'Unknown')}\n")
        
        # Generate plots if requested
        if self.config['analysis'].get('generate_plots', True):
            try:
                self._generate_plots()
                f.write("\n## Visualizations\n\n")
                f.write("![Distribution Plot](plots/distributions.png)\n\n")
                f.write("![Evolution Plot](plots/evolution.png)\n\n")
            except Exception as e:
                f.write(f"\n## Visualizations\n\n")
                f.write(f"_Error generating plots: {e}_\n\n")
        
        return report_path
    
    def _format_metric_name(self, metric_name: str) -> str:
        """Format metric name for display"""
        # Replace dots and underscores with spaces, capitalize
        formatted = metric_name.replace('.', ' - ').replace('_', ' ')
        return formatted.title()
    
    def _generate_plots(self):
        """Generate visualization plots"""
        plots_dir = self.experiment_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Generate distribution plots
        self._plot_distributions(plots_dir)
        
        # Generate evolution plots
        self._plot_evolution(plots_dir)
    
    def _plot_distributions(self, plots_dir: Path):
        """Plot distributions of key metrics"""
        if 'raw_values' not in self.aggregate_metrics:
            return
        
        key_metrics = self.config['analysis'].get('key_metrics', [])
        raw_values = self.aggregate_metrics['raw_values']
        
        # Determine number of subplots needed
        metrics_to_plot = [m for m in key_metrics if m in raw_values]
        if not metrics_to_plot:
            return
        
        n_metrics = len(metrics_to_plot)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        
        for idx, metric_name in enumerate(metrics_to_plot):
            row = idx // n_cols
            col = idx % n_cols
            
            # Get the correct axis
            if n_rows == 1 and n_cols == 1:
                ax = axes[0][0]
            elif n_rows == 1:
                ax = axes[0][col] if isinstance(axes[0], list) else axes[col]
            elif n_cols == 1:
                ax = axes[row][0]
            else:
                ax = axes[row][col]
            
            values = raw_values[metric_name]
            
            # Create box plot with individual points
            ax.boxplot(values, vert=True)
            ax.scatter([1]*len(values), values, alpha=0.5, s=50)
            
            # Add mean line
            mean_val = np.mean(values)
            ax.axhline(y=mean_val, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.3f}')
            
            ax.set_title(self._format_metric_name(metric_name))
            ax.set_ylabel('Value')
            ax.set_xticklabels([''])
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_metrics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            
            # Get the correct axis to hide
            if n_rows == 1 and n_cols == 1:
                continue  # Only one subplot, nothing to hide
            elif n_rows == 1:
                ax = axes[0][col] if isinstance(axes[0], list) else axes[col]
            elif n_cols == 1:
                ax = axes[row][0]
            else:
                ax = axes[row][col]
            ax.set_visible(False)
        
        plt.suptitle(f'Metric Distributions - {self.experiment_id}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(plots_dir / 'distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_evolution(self, plots_dir: Path):
        """Plot evolution of metrics over rounds"""
        if 'per_round_evolution' not in self.aggregate_metrics:
            return
        
        evolution = self.aggregate_metrics['per_round_evolution']
        if not evolution:
            return
        
        # Extract round numbers and sort
        rounds = sorted([int(k.split('_')[1]) for k in evolution.keys() 
                       if k.startswith('round_')])
        
        if not rounds:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot tasks completed
        tasks_means = []
        tasks_stds = []
        for round_num in rounds:
            round_key = f'round_{round_num}'
            if round_key in evolution and 'tasks_completed' in evolution[round_key]:
                tasks_means.append(evolution[round_key]['tasks_completed']['mean'])
                tasks_stds.append(evolution[round_key]['tasks_completed'].get('std', 0))
            else:
                tasks_means.append(0)
                tasks_stds.append(0)
        
        ax1.errorbar(rounds, tasks_means, yerr=tasks_stds, marker='o', capsize=5)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Tasks Completed')
        ax1.set_title('Tasks Completed Per Round')
        ax1.grid(True, alpha=0.3)
        
        # Plot messages sent
        messages_means = []
        messages_stds = []
        for round_num in rounds:
            round_key = f'round_{round_num}'
            if round_key in evolution and 'messages_sent' in evolution[round_key]:
                messages_means.append(evolution[round_key]['messages_sent']['mean'])
                messages_stds.append(evolution[round_key]['messages_sent'].get('std', 0))
            else:
                messages_means.append(0)
                messages_stds.append(0)
        
        ax2.errorbar(rounds, messages_means, yerr=messages_stds, marker='s', capsize=5, color='orange')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Messages Sent')
        ax2.set_title('Messages Sent Per Round')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Metric Evolution - {self.experiment_id}', fontsize=14)
        plt.tight_layout()
        plt.savefig(plots_dir / 'evolution.png', dpi=150, bbox_inches='tight')
        plt.close()