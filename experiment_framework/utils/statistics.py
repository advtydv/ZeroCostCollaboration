"""
Statistical aggregation for experiment results
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import scipy.stats as stats


class StatisticalAggregator:
    """Aggregates statistics across multiple simulation runs"""
    
    def __init__(self, experiment_dir: Path, run_results: List[Dict[str, Any]]):
        self.experiment_dir = Path(experiment_dir)
        self.run_results = run_results
        self.successful_runs = [r for r in run_results if r['status'] == 'completed']
        self.logger = logging.getLogger(__name__)
    
    def _log_skipped_metric(self, metric_name: str, values: List[Any]):
        """Log when a metric is skipped due to non-numeric values"""
        self.logger.debug(f"Skipping non-numeric metric '{metric_name}': {type(values[0]) if values else 'empty'}")
    
    def aggregate(self) -> Dict[str, Any]:
        """Aggregate all statistics"""
        if not self.successful_runs:
            return {
                'error': 'No successful runs to aggregate',
                'total_runs': len(self.run_results),
                'successful_runs': 0
            }
        
        aggregate = {
            'total_runs': len(self.run_results),
            'successful_runs': len(self.successful_runs),
            'failed_runs': len(self.run_results) - len(self.successful_runs),
            'statistical_summary': {},
            'per_round_evolution': {},
            'convergence_analysis': {},
            'raw_values': {}
        }
        
        # Extract metrics from all runs
        metrics_by_run = self._extract_metrics_from_runs()
        
        # Calculate statistical summaries for each metric
        for metric_name, values in metrics_by_run.items():
            if values and all(v is not None for v in values):
                # Only calculate statistics for numeric values
                if all(isinstance(v, (int, float)) for v in values):
                    aggregate['statistical_summary'][metric_name] = self._calculate_statistics(values)
                    aggregate['raw_values'][metric_name] = values
                else:
                    # Skip non-numeric metrics (like dicts)
                    self._log_skipped_metric(metric_name, values)
        
        # Analyze per-round evolution
        aggregate['per_round_evolution'] = self._analyze_round_evolution()
        
        # Analyze convergence
        aggregate['convergence_analysis'] = self._analyze_convergence()
        
        return aggregate
    
    def _extract_metrics_from_runs(self) -> Dict[str, List[float]]:
        """Extract metrics from all successful runs"""
        metrics = defaultdict(list)
        
        for run in self.successful_runs:
            # Get analysis results if available
            if run.get('analysis') and 'metrics' in run['analysis']:
                analysis_metrics = run['analysis']['metrics']
                
                # Extract common metrics
                if 'total_tasks_completed' in analysis_metrics:
                    metrics['total_tasks_completed'].append(analysis_metrics['total_tasks_completed'])
                
                if 'revenue_distribution' in analysis_metrics:
                    gini = analysis_metrics['revenue_distribution'].get('gini_coefficient')
                    if gini is not None:
                        metrics['revenue_distribution.gini_coefficient'].append(gini)
                
                if 'communication_efficiency' in analysis_metrics:
                    efficiency = analysis_metrics['communication_efficiency'].get('messages_per_completed_task')
                    if efficiency is not None:
                        metrics['communication_efficiency.messages_per_completed_task'].append(efficiency)
                    
                    # Also extract total_messages from communication efficiency
                    total_msgs = analysis_metrics['communication_efficiency'].get('total_messages')
                    if total_msgs is not None:
                        metrics['communication_efficiency.total_messages'].append(total_msgs)
                
                if 'agents_with_zero_revenue' in analysis_metrics:
                    metrics['agents_with_zero_revenue'].append(analysis_metrics['agents_with_zero_revenue'])
                
                if 'network_hub_analysis' in analysis_metrics:
                    hub_conc = analysis_metrics['network_hub_analysis'].get('hub_concentration')
                    if hub_conc is not None:
                        metrics['network_hub_analysis.hub_concentration'].append(hub_conc)
                    
                    top_hub_msgs = analysis_metrics['network_hub_analysis'].get('top_hub_messages')
                    if top_hub_msgs is not None:
                        metrics['network_hub_analysis.top_hub_messages'].append(top_hub_msgs)
                
                # Extract new metrics
                if 'information_transfer_rate' in analysis_metrics:
                    transfer_rate = analysis_metrics['information_transfer_rate'].get('transfer_rate_per_round')
                    if transfer_rate is not None:
                        metrics['information_transfer_rate.transfer_rate_per_round'].append(transfer_rate)
                    
                    total_transferred = analysis_metrics['information_transfer_rate'].get('total_pieces_transferred')
                    if total_transferred is not None:
                        metrics['information_transfer_rate.total_pieces_transferred'].append(total_transferred)
                
                if 'manipulation_rate' in analysis_metrics:
                    manip_rate = analysis_metrics['manipulation_rate'].get('manipulation_rate')
                    if manip_rate is not None:
                        metrics['manipulation_rate.manipulation_rate'].append(manip_rate)
                    
                    tasks_with_penalties = analysis_metrics['manipulation_rate'].get('tasks_with_penalties')
                    if tasks_with_penalties is not None:
                        metrics['manipulation_rate.tasks_with_penalties'].append(tasks_with_penalties)
                
                if 'revenue_spread' in analysis_metrics:
                    min_max_ratio = analysis_metrics['revenue_spread'].get('min_max_ratio')
                    if min_max_ratio is not None:
                        metrics['revenue_spread.min_max_ratio'].append(min_max_ratio)
                    
                    spread = analysis_metrics['revenue_spread'].get('revenue_spread')
                    if spread is not None:
                        metrics['revenue_spread.spread'].append(spread)
            
            # Also extract from raw results
            if 'results' in run and run['results']:
                results = run['results']
                
                # Total messages
                if 'total_messages' in results:
                    metrics['total_messages'].append(results['total_messages'])
                
                # Winner's revenue
                if 'final_revenue_board' in results:
                    revenues = list(results['final_revenue_board'].values())
                    if revenues:
                        metrics['max_revenue'].append(max(revenues))
                        metrics['min_revenue'].append(min(revenues))
                        metrics['mean_revenue'].append(np.mean(revenues))
        
        return dict(metrics)
    
    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical measures for a set of values"""
        values_array = np.array(values)
        
        stats_dict = {
            'mean': float(np.mean(values_array)),
            'median': float(np.median(values_array)),
            'std': float(np.std(values_array, ddof=1)) if len(values_array) > 1 else 0.0,
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'values': [float(v) for v in values_array]  # Store raw values
        }
        
        # Calculate 95% confidence interval if we have enough samples
        if len(values_array) > 1:
            sem = stats.sem(values_array)
            ci = stats.t.interval(0.95, len(values_array)-1, 
                                 loc=np.mean(values_array), scale=sem)
            stats_dict['ci_95'] = [float(ci[0]), float(ci[1])]
            stats_dict['sem'] = float(sem)  # Standard error of mean
        else:
            stats_dict['ci_95'] = [stats_dict['mean'], stats_dict['mean']]
            stats_dict['sem'] = 0.0
        
        # Calculate coefficient of variation (relative variability)
        if stats_dict['mean'] != 0:
            stats_dict['cv'] = stats_dict['std'] / abs(stats_dict['mean'])
        else:
            stats_dict['cv'] = 0.0
        
        return stats_dict
    
    def _analyze_round_evolution(self) -> Dict[str, Any]:
        """Analyze how metrics evolve across rounds"""
        evolution = {}
        
        # Extract per-round data from all runs
        rounds_data = defaultdict(lambda: defaultdict(list))
        
        for run in self.successful_runs:
            if 'results' in run and 'rounds' in run['results']:
                for round_data in run['results']['rounds']:
                    round_num = round_data['round']
                    
                    # Tasks completed in this round
                    if 'tasks_completed' in round_data:
                        rounds_data[f'round_{round_num}']['tasks_completed'].append(
                            round_data['tasks_completed']
                        )
                    
                    # Messages sent in this round
                    if 'messages_sent' in round_data:
                        rounds_data[f'round_{round_num}']['messages_sent'].append(
                            round_data['messages_sent']
                        )
        
        # Calculate statistics for each round
        for round_key, metrics in rounds_data.items():
            evolution[round_key] = {}
            for metric_name, values in metrics.items():
                if values:
                    evolution[round_key][metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    }
        
        return evolution
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence patterns in the simulations"""
        convergence = {}
        
        # Analyze task completion convergence
        task_completions_by_round = []
        
        for run in self.successful_runs:
            if 'results' in run and 'rounds' in run['results']:
                run_completions = []
                cumulative = 0
                for round_data in run['results']['rounds']:
                    if 'tasks_completed' in round_data:
                        cumulative += round_data['tasks_completed']
                        run_completions.append(cumulative)
                
                if run_completions:
                    task_completions_by_round.append(run_completions)
        
        if task_completions_by_round:
            # Calculate variance reduction over rounds
            max_rounds = max(len(run) for run in task_completions_by_round)
            variance_by_round = []
            
            for round_idx in range(max_rounds):
                round_values = []
                for run in task_completions_by_round:
                    if round_idx < len(run):
                        round_values.append(run[round_idx])
                
                if len(round_values) > 1:
                    variance_by_round.append(np.var(round_values))
            
            if variance_by_round:
                # Find when variance stabilizes (reduces by 50% from peak)
                peak_variance = max(variance_by_round)
                for idx, var in enumerate(variance_by_round):
                    if var < peak_variance * 0.5:
                        convergence['steady_state_round'] = idx + 1
                        break
                
                # Calculate overall variance reduction
                if variance_by_round[0] > 0:
                    convergence['variance_reduction'] = float(
                        1 - variance_by_round[-1] / variance_by_round[0]
                    )
        
        return convergence
    
    def get_metric_for_plotting(self, metric_name: str) -> Optional[Tuple[List[float], Dict]]:
        """Get a specific metric's values and statistics for plotting"""
        metrics = self._extract_metrics_from_runs()
        
        if metric_name in metrics:
            values = metrics[metric_name]
            if values:
                stats = self._calculate_statistics(values)
                return values, stats
        
        return None, None