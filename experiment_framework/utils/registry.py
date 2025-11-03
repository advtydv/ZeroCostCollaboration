"""
Experiment registry management for tracking all experiments
"""

import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading


class ExperimentRegistry:
    """Manages the central registry of all experiments"""
    
    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        self.lock = threading.Lock()
        
        # Load existing registry or create new one
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                self.registry = yaml.safe_load(f) or {'experiments': {}}
        else:
            self.registry = {
                'experiments': {},
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat(),
                    'total_experiments': 0,
                    'next_experiment_number': 1
                }
            }
            self._save()
    
    def _save(self):
        """Save registry to file"""
        self.registry['metadata']['last_updated'] = datetime.now().isoformat()
        with open(self.registry_path, 'w') as f:
            yaml.dump(self.registry, f, default_flow_style=False, sort_keys=False)
    
    def get_next_experiment_number(self) -> int:
        """Get the next available experiment number"""
        with self.lock:
            if 'metadata' not in self.registry:
                self.registry['metadata'] = {'next_experiment_number': 1}
            
            next_num = self.registry['metadata'].get('next_experiment_number', 1)
            self.registry['metadata']['next_experiment_number'] = next_num + 1
            self._save()
            return next_num
    
    def register_experiment(self, experiment_id: str, metadata: Dict[str, Any], 
                           config: Dict[str, Any]):
        """Register a new experiment"""
        with self.lock:
            self.registry['experiments'][experiment_id] = {
                'created': metadata['created_at'],
                'description': metadata['description'],
                'tags': metadata.get('tags', []),
                'config_summary': {
                    'model': config['simulation_config']['agents']['model'],
                    'num_agents': config['simulation_config']['simulation']['agents'],
                    'rounds': config['simulation_config']['simulation']['rounds'],
                    'uncooperative_count': config['simulation_config']['agents'].get('uncooperative_count', 0)
                },
                'git_commit': metadata['git'].get('commit', 'unknown'),
                'git_branch': metadata['git'].get('branch', 'unknown'),
                'has_uncommitted': metadata['git'].get('has_uncommitted_changes', False),
                'runs_requested': metadata['runs_requested'],
                'runs_completed': 0,
                'status': 'initialized'
            }
            
            self.registry['metadata']['total_experiments'] = len(self.registry['experiments'])
            self._save()
    
    def update_experiment_status(self, experiment_id: str, status: str, 
                                key_metrics: Optional[Dict[str, Any]] = None):
        """Update the status of an experiment"""
        with self.lock:
            if experiment_id in self.registry['experiments']:
                self.registry['experiments'][experiment_id]['status'] = status
                self.registry['experiments'][experiment_id]['last_updated'] = datetime.now().isoformat()
                
                if status == 'completed' and key_metrics:
                    self.registry['experiments'][experiment_id]['key_metrics'] = key_metrics
                
                self._save()
    
    def find_experiments(self, **filters) -> List[str]:
        """Find experiments matching filters"""
        matching = []
        
        for exp_id, exp_data in self.registry['experiments'].items():
            match = True
            
            # Check each filter
            for key, value in filters.items():
                if key == 'model':
                    if exp_data['config_summary']['model'] != value:
                        match = False
                elif key == 'status':
                    if exp_data['status'] != value:
                        match = False
                elif key == 'tag':
                    if value not in exp_data.get('tags', []):
                        match = False
                elif key == 'after':
                    if exp_data['created'] < value:
                        match = False
                elif key == 'before':
                    if exp_data['created'] > value:
                        match = False
            
            if match:
                matching.append(exp_id)
        
        return sorted(matching)
    
    def get_experiment_info(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific experiment"""
        return self.registry['experiments'].get(experiment_id)
    
    def list_all_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments with their information"""
        experiments = []
        for exp_id, exp_data in self.registry['experiments'].items():
            exp_info = exp_data.copy()
            exp_info['id'] = exp_id
            experiments.append(exp_info)
        
        return sorted(experiments, key=lambda x: x['created'], reverse=True)