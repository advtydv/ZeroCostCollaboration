#!/usr/bin/env python3
"""
Research-grade experiment runner for Information Asymmetry Simulations
Manages multiple runs with statistical aggregation and reproducibility tracking
"""

import argparse
import json
import yaml
import logging
import subprocess
import sys
import os
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from experiment_framework.utils.registry import ExperimentRegistry
from experiment_framework.utils.statistics import StatisticalAggregator
from experiment_framework.utils.analysis_fixed import ExperimentAnalyzer


# Static function for parallel execution (avoids pickling issues)
def run_single_simulation_static(run_number: int, experiment_dir: Path, experiment_id: str) -> Dict[str, Any]:
    """Static version of run_single_simulation that can be pickled for parallel execution"""
    run_id = f"run_{run_number:03d}"
    run_dir = experiment_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger for this process WITH handlers (critical for subprocess visibility)
    logger = logging.getLogger(f"exp_runner_{run_id}")
    logger.setLevel(logging.INFO)
    # Clear any existing handlers to avoid duplicates
    logger.handlers = []
    # Add console handler so errors are visible to user
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    # Check if this run already completed
    results_file = run_dir / "results.yaml"
    if results_file.exists():
        logger.info(f"Run {run_id} already completed, skipping")
        with open(results_file, 'r') as f:
            results = yaml.safe_load(f)
        return {
            'run_id': run_id,
            'status': 'completed',
            'skipped': True,
            'results': results
        }
    
    logger.info(f"Starting {run_id}")
    start_time = time.time()
    
    # Prepare simulation command
    sim_script = Path(__file__).parent.parent / "information_asymmetry_simulation" / "main.py"
    sim_config = experiment_dir / "simulation_config.yaml"
    
    # Use absolute paths to avoid relative path issues
    cmd = [
        sys.executable,
        str(sim_script.absolute()),
        "--config", str(sim_config.absolute()),
        "--output-dir", str(run_dir.absolute()),
        "--sim-id", "simulation",
        "--log-level", "INFO"
    ]
    
    try:
        # Run simulation
        env = os.environ.copy()
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path(__file__).parent.parent / "information_asymmetry_simulation")
        )

        duration = time.time() - start_time

        # CRITICAL: Always print subprocess output so errors are visible to user
        if process.stdout:
            print(process.stdout)
        if process.stderr:
            print(process.stderr, file=sys.stderr)

        if process.returncode == 0:
            # Load results
            sim_results_file = run_dir / "simulation" / "results.yaml"
            if sim_results_file.exists():
                with open(sim_results_file, 'r') as f:
                    sim_results = yaml.safe_load(f)
                
                # Also check for analysis results
                analysis_file = run_dir / "simulation" / "analysis_results.json"
                if analysis_file.exists():
                    with open(analysis_file, 'r') as f:
                        analysis_results = json.load(f)
                else:
                    analysis_results = None
                
                logger.info(f"Completed {run_id} in {duration:.2f}s")
                
                return {
                    'run_id': run_id,
                    'status': 'completed',
                    'duration': duration,
                    'results': sim_results,
                    'analysis': analysis_results
                }
            else:
                raise FileNotFoundError(f"Results file not found for {run_id}")
        else:
            logger.error(f"Failed {run_id}: Exit code {process.returncode}")
            logger.error(f"Error output: {process.stderr}")
            return {
                'run_id': run_id,
                'status': 'failed',
                'duration': duration,
                'error': process.stderr,
                'exit_code': process.returncode
            }
            
    except Exception as e:
        logger.error(f"Exception in {run_id}: {str(e)}")
        return {
            'run_id': run_id,
            'status': 'error',
            'duration': time.time() - start_time,
            'error': str(e)
        }


class ExperimentRunner:
    """Manages research-grade experiments with multiple simulation runs"""
    
    def __init__(self, config_path: str, experiments_dir: str = "experiments", 
                 resume_id: Optional[str] = None):
        self.config_path = Path(config_path).absolute()
        self.experiments_base = Path(experiments_dir)
        self.experiments_base.mkdir(exist_ok=True)
        
        # Initialize registry
        self.registry = ExperimentRegistry(self.experiments_base / "registry.yaml")
        
        if resume_id:
            # Resume existing experiment
            self.experiment_id = resume_id
            self.experiment_dir = self.experiments_base / resume_id
            if not self.experiment_dir.exists():
                raise ValueError(f"Experiment {resume_id} not found")
            
            # Load existing metadata and config
            with open(self.experiment_dir / "metadata.yaml", 'r') as f:
                self.metadata = yaml.safe_load(f)
            with open(self.experiment_dir / "experiment_config.yaml", 'r') as f:
                self.config = yaml.safe_load(f)
                
            self.logger = self._setup_logging()
            self.logger.info(f"Resuming experiment {resume_id}")
        else:
            # Load configuration for new experiment
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Generate experiment ID
            self.experiment_id = self._generate_experiment_id()
            self.experiment_dir = self.experiments_base / self.experiment_id
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup logging
            self.logger = self._setup_logging()
            
            # Initialize metadata
            self.metadata = self._initialize_metadata()
            
            # Save copies of config and metadata
            self._save_experiment_files()
            
            # Register experiment
            self.registry.register_experiment(self.experiment_id, self.metadata, self.config)
            
            self.logger.info(f"Created new experiment: {self.experiment_id}")
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID with timestamp"""
        # Get next experiment number from registry
        exp_num = self.registry.get_next_experiment_number()
        name = self.config['experiment']['name'].lower().replace(' ', '_')
        # Add timestamp to make each experiment run unique
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"exp_{exp_num:03d}_{name}_{timestamp}"
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the experiment"""
        log_file = self.experiment_dir / "experiment.log"
        
        # Create logger
        logger = logging.getLogger(f"experiment_{self.experiment_id}")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _initialize_metadata(self) -> Dict[str, Any]:
        """Initialize experiment metadata"""
        # Get git information
        git_info = self._get_git_info()
        
        # Get environment information
        env_info = {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': sys.platform,
            'cwd': str(Path.cwd())
        }
        
        # Get package versions
        packages = {}
        try:
            import openai
            packages['openai'] = openai.__version__
        except:
            pass
        
        try:
            import numpy
            packages['numpy'] = numpy.__version__
        except:
            pass
        
        metadata = {
            'experiment_id': self.experiment_id,
            'created_at': datetime.now().isoformat(),
            'description': self.config['experiment']['description'],
            'tags': self.config['experiment'].get('tags', []),
            'config_source': str(self.config_path),
            'git': git_info,
            'environment': env_info,
            'packages': packages,
            'status': 'initialized',
            'runs_requested': self.config['experiment']['num_runs'],
            'runs_completed': 0,
            'runs_failed': 0
        }
        
        return metadata
    
    def _get_git_info(self) -> Dict[str, Any]:
        """Get git repository information"""
        git_info = {}
        
        try:
            # Get current commit hash
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, check=True)
            git_info['commit'] = result.stdout.strip()
            
            # Get current branch
            result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                  capture_output=True, text=True, check=True)
            git_info['branch'] = result.stdout.strip()
            
            # Check for uncommitted changes
            result = subprocess.run(['git', 'status', '--porcelain'],
                                  capture_output=True, text=True, check=True)
            git_info['has_uncommitted_changes'] = bool(result.stdout.strip())
            
            if git_info['has_uncommitted_changes']:
                self.logger.warning("WARNING: Uncommitted changes detected! Results may not be reproducible.")
                # Save diff for reference
                result = subprocess.run(['git', 'diff'], capture_output=True, text=True)
                diff_file = self.experiment_dir / "uncommitted_changes.diff"
                with open(diff_file, 'w') as f:
                    f.write(result.stdout)
                git_info['diff_file'] = str(diff_file)
                
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Git information unavailable: {e}")
            git_info['error'] = str(e)
        
        return git_info
    
    def _save_experiment_files(self):
        """Save experiment configuration and metadata"""
        # Save frozen copy of experiment config
        config_copy = self.experiment_dir / "experiment_config.yaml"
        with open(config_copy, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Save metadata
        metadata_file = self.experiment_dir / "metadata.yaml"
        with open(metadata_file, 'w') as f:
            yaml.dump(self.metadata, f, default_flow_style=False)
        
        # Create simulation config for each run
        sim_config = self.config['simulation_config']
        sim_config_file = self.experiment_dir / "simulation_config.yaml"
        with open(sim_config_file, 'w') as f:
            yaml.dump(sim_config, f, default_flow_style=False)
    
    def run_single_simulation(self, run_number: int) -> Dict[str, Any]:
        """Run a single simulation"""
        # Call the static method to avoid pickling issues
        return run_single_simulation_static(
            run_number=run_number,
            experiment_dir=self.experiment_dir,
            experiment_id=self.experiment_id
        )
    
    def run_experiment(self):
        """Run the complete experiment"""
        self.logger.info(f"Starting experiment with {self.config['experiment']['num_runs']} runs")
        
        # Update status
        self.metadata['status'] = 'running'
        self.metadata['started_at'] = datetime.now().isoformat()
        self._save_metadata()
        self.registry.update_experiment_status(self.experiment_id, 'running')
        
        # Determine which runs to perform
        existing_runs = []
        runs_dir = self.experiment_dir / "runs"
        if runs_dir.exists():
            for run_dir in runs_dir.iterdir():
                if run_dir.is_dir() and (run_dir / "simulation" / "results.yaml").exists():
                    run_num = int(run_dir.name.split('_')[1])
                    existing_runs.append(run_num)
        
        runs_to_do = [i for i in range(1, self.config['experiment']['num_runs'] + 1) 
                      if i not in existing_runs]
        
        if not runs_to_do:
            self.logger.info("All runs already completed")
        else:
            self.logger.info(f"Running {len(runs_to_do)} simulations (already completed: {len(existing_runs)})")
        
        # Run simulations
        results = []
        use_parallel = self.config['experiment'].get('parallel', False)
        
        if use_parallel and len(runs_to_do) > 1:
            max_workers = self.config['experiment'].get('max_workers', 4)
            self.logger.info(f"Running in parallel with {max_workers} workers")
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Use the static function with experiment_dir and experiment_id
                futures = {executor.submit(run_single_simulation_static, run_num, 
                                         self.experiment_dir, self.experiment_id): run_num 
                          for run_num in runs_to_do}
                
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    
                    # Log the result
                    run_id = result.get('run_id', f'run_{futures[future]:03d}')
                    if result['status'] == 'completed':
                        self.logger.info(f"Completed {run_id} in {result.get('duration', 0):.2f}s")
                        self.metadata['runs_completed'] += 1
                    else:
                        self.logger.error(f"Failed {run_id}: {result.get('error', 'Unknown error')}")
                        self.metadata['runs_failed'] += 1
                    self._save_metadata()
        else:
            for run_num in runs_to_do:
                result = self.run_single_simulation(run_num)
                results.append(result)
                
                # Log the result  
                run_id = result.get('run_id', f'run_{run_num:03d}')
                if result['status'] == 'completed':
                    self.logger.info(f"Completed {run_id} in {result.get('duration', 0):.2f}s")
                    self.metadata['runs_completed'] += 1
                else:
                    self.logger.error(f"Failed {run_id}: {result.get('error', 'Unknown error')}")
                    self.metadata['runs_failed'] += 1
                self._save_metadata()
        
        # Include existing runs in results
        for run_num in existing_runs:
            run_id = f"run_{run_num:03d}"
            run_dir = self.experiment_dir / "runs" / run_id
            
            # Load existing results
            with open(run_dir / "simulation" / "results.yaml", 'r') as f:
                sim_results = yaml.safe_load(f)
            
            analysis_file = run_dir / "simulation" / "analysis_results.json"
            if analysis_file.exists():
                with open(analysis_file, 'r') as f:
                    analysis_results = json.load(f)
            else:
                analysis_results = None
            
            results.append({
                'run_id': run_id,
                'status': 'completed',
                'results': sim_results,
                'analysis': analysis_results,
                'existing': True
            })
        
        # Aggregate statistics
        self.logger.info("Aggregating statistics across runs...")
        aggregator = StatisticalAggregator(self.experiment_dir, results)
        aggregate_metrics = aggregator.aggregate()
        
        # Save aggregate metrics
        metrics_file = self.experiment_dir / "aggregate_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(aggregate_metrics, f, indent=2)
        
        # Generate analysis report
        if self.config['analysis'].get('generate_report', True):
            self.logger.info("Generating analysis report...")
            analyzer = ExperimentAnalyzer(self.experiment_dir, self.config, aggregate_metrics)
            report_path = analyzer.generate_report()
            self.logger.info(f"Report saved to {report_path}")
        
        # Update final metadata
        self.metadata['status'] = 'completed'
        self.metadata['completed_at'] = datetime.now().isoformat()
        # Handle case where no runs succeeded (statistical_summary may not exist)
        if 'statistical_summary' in aggregate_metrics:
            self.metadata['aggregate_metrics_summary'] = {
                'avg_gini': aggregate_metrics['statistical_summary'].get('revenue_distribution.gini_coefficient', {}).get('mean'),
                'avg_tasks_completed': aggregate_metrics['statistical_summary'].get('total_tasks_completed', {}).get('mean')
            }
        else:
            self.metadata['aggregate_metrics_summary'] = {
                'avg_gini': None,
                'avg_tasks_completed': None,
                'error': aggregate_metrics.get('error', 'No successful runs')
            }
        self._save_metadata()
        
        # Update registry
        self.registry.update_experiment_status(
            self.experiment_id, 
            'completed',
            key_metrics=self.metadata['aggregate_metrics_summary']
        )
        
        self.logger.info(f"Experiment {self.experiment_id} completed successfully!")
        self.logger.info(f"Results saved to {self.experiment_dir}")
        
        # Print summary
        self._print_summary(aggregate_metrics)
    
    def _save_metadata(self):
        """Save updated metadata"""
        metadata_file = self.experiment_dir / "metadata.yaml"
        with open(metadata_file, 'w') as f:
            yaml.dump(self.metadata, f, default_flow_style=False)
    
    def _print_summary(self, aggregate_metrics: Dict[str, Any]):
        """Print experiment summary"""
        print("\n" + "="*60)
        print(f"EXPERIMENT SUMMARY: {self.experiment_id}")
        print("="*60)
        print(f"Description: {self.config['experiment']['description'][:100]}...")
        print(f"Runs completed: {self.metadata['runs_completed']}/{self.metadata['runs_requested']}")
        
        if 'statistical_summary' in aggregate_metrics:
            print("\nKEY METRICS (mean ± std):")
            stats = aggregate_metrics['statistical_summary']
            
            for metric_name in self.config['analysis'].get('key_metrics', []):
                if metric_name in stats:
                    metric = stats[metric_name]
                    if isinstance(metric, dict) and 'mean' in metric:
                        print(f"  {metric_name}: {metric['mean']:.3f} ± {metric.get('std', 0):.3f}")
        
        print("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run research-grade experiments')
    parser.add_argument('--config', type=str, help='Path to experiment configuration file')
    parser.add_argument('--resume', type=str, help='Resume an existing experiment by ID')
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                       help='Directory to store experiments (default: experiments)')
    
    args = parser.parse_args()
    
    if not args.config and not args.resume:
        parser.error("Either --config or --resume must be provided")
    
    if args.resume:
        runner = ExperimentRunner(
            config_path=None,
            experiments_dir=args.experiments_dir,
            resume_id=args.resume
        )
    else:
        runner = ExperimentRunner(
            config_path=args.config,
            experiments_dir=args.experiments_dir
        )
    
    runner.run_experiment()


if __name__ == "__main__":
    main()