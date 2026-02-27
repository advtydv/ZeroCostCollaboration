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


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Static function for parallel execution (avoids pickling issues)
def run_single_simulation_static(
    run_number: int,
    experiment_dir: Path,
    experiment_id: str,
    simulation_root: str = "information_asymmetry_simulation",
) -> Dict[str, Any]:
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
    results_file = run_dir / "simulation" / "results.yaml"
    legacy_results_file = run_dir / "results.yaml"
    if results_file.exists() or legacy_results_file.exists():
        logger.info(f"Run {run_id} already completed, skipping")
        source_file = results_file if results_file.exists() else legacy_results_file
        with open(source_file, 'r') as f:
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
    simulation_root_path = Path(simulation_root)
    if not simulation_root_path.is_absolute():
        simulation_root_path = (PROJECT_ROOT / simulation_root_path).resolve()
    sim_script = simulation_root_path / "main.py"
    sim_config = experiment_dir / "simulation_config.yaml"

    if not sim_script.exists():
        error_msg = f"Simulation entrypoint not found: {sim_script}"
        logger.error(error_msg)
        return {
            "run_id": run_id,
            "status": "error",
            "duration": time.time() - start_time,
            "error": error_msg,
        }
    
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
        # Run simulation with real-time output streaming
        env = os.environ.copy()
        
        # Use Popen for real-time output instead of buffered subprocess.run
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout for unified streaming
            text=True,
            env=env,
            cwd=str(simulation_root_path),
            bufsize=1  # Line buffered for real-time output
        )
        
        # Stream output in real-time
        output_lines = []
        try:
            for line in process.stdout:
                print(line, end='', flush=True)  # Real-time display
                output_lines.append(line)
        except Exception as read_error:
            logger.warning(f"Error reading subprocess output: {read_error}")
        
        # Wait for process to complete (no timeout - experiments can be long-running)
        process.wait()

        duration = time.time() - start_time




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
            error_output = "".join(output_lines[-50:]).strip()
            if not error_output:
                error_output = "No subprocess output captured."
            logger.error(f"Error output (last {min(len(output_lines), 50)} lines):\n{error_output}")
            return {
                'run_id': run_id,
                'status': 'failed',
                'duration': duration,
                'error': error_output,
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
    
    def __init__(
        self,
        config_path: Optional[str],
        experiments_dir: str = "experiments",
        resume_id: Optional[str] = None,
        simulation_root: Optional[str] = None,
    ):
        self.config_path = Path(config_path).absolute() if config_path else None
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
            self.simulation_root = self._resolve_simulation_root(simulation_root)
            self.logger.info(f"Resuming experiment {resume_id}")
            self.logger.info(f"Using simulation root: {self.simulation_root}")
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
            self.simulation_root = self._resolve_simulation_root(simulation_root)
            self.logger.info(f"Using simulation root: {self.simulation_root}")
            
            # Initialize metadata
            self.metadata = self._initialize_metadata()
            
            # Save copies of config and metadata
            self._save_experiment_files()
            
            # Register experiment
            self.registry.register_experiment(self.experiment_id, self.metadata, self.config)
            
            self.logger.info(f"Created new experiment: {self.experiment_id}")

    def _resolve_simulation_root(self, cli_override: Optional[str]) -> str:
        """Resolve simulation root with CLI override > config > default precedence."""
        if cli_override:
            return cli_override

        runner_cfg = self.config.get("runner", {}) if isinstance(self.config, dict) else {}
        if isinstance(runner_cfg, dict):
            root = runner_cfg.get("simulation_root")
            if root:
                return str(root)

        root = self.config.get("simulation_root")
        if root:
            return str(root)

        return "information_asymmetry_simulation"

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
            'config_source': str(self.config_path) if self.config_path else None,
            'git': git_info,
            'environment': env_info,
            'packages': packages,
            'simulation_root': self.simulation_root,
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
                
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
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
            experiment_id=self.experiment_id,
            simulation_root=self.simulation_root,
        )
    
    def run_experiment(self) -> bool:
        """Run the complete experiment and return True only when all runs succeed."""
        self.logger.info(f"Starting experiment with {self.config['experiment']['num_runs']} runs")
        
        # Update status
        self.metadata['status'] = 'running'
        self.metadata['started_at'] = datetime.now().isoformat()
        self._save_metadata()
        self.registry.update_experiment_status(self.experiment_id, 'running')
        try:
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
                                               self.experiment_dir, self.experiment_id,
                                               self.simulation_root): run_num
                               for run_num in runs_to_do}

                    for future in as_completed(futures):
                        run_num = futures[future]
                        run_id = f"run_{run_num:03d}"
                        try:
                            result = future.result()
                        except Exception as worker_error:
                            error_msg = f"Worker process crashed for {run_id}: {worker_error}"
                            self.logger.error(error_msg)
                            result = {
                                'run_id': run_id,
                                'status': 'error',
                                'duration': 0.0,
                                'error': error_msg,
                            }
                        results.append(result)

                        # Log the result
                        result_run_id = result.get('run_id', run_id)
                        if result['status'] == 'completed':
                            self.logger.info(f"Completed {result_run_id} in {result.get('duration', 0):.2f}s")
                            self.metadata['runs_completed'] += 1
                        else:
                            self.logger.error(f"Failed {result_run_id}: {result.get('error', 'Unknown error')}")
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
                try:
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
                except Exception as load_error:
                    error_msg = f"Failed to load existing run {run_id}: {load_error}"
                    self.logger.error(error_msg)
                    results.append({
                        'run_id': run_id,
                        'status': 'error',
                        'error': error_msg,
                        'existing': True
                    })

            # Recompute final run counts from collected results so metadata matches reality
            completed_runs = sum(1 for r in results if r.get('status') == 'completed')
            failed_runs = sum(1 for r in results if r.get('status') in {'failed', 'error'})
            self.metadata['runs_completed'] = completed_runs
            self.metadata['runs_failed'] = failed_runs

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
            overall_success = (
                failed_runs == 0 and
                completed_runs == self.config['experiment']['num_runs']
            )
            final_status = 'completed' if overall_success else 'failed'
            self.metadata['status'] = final_status
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
                final_status,
                key_metrics=self.metadata['aggregate_metrics_summary']
            )

            if overall_success:
                self.logger.info(f"Experiment {self.experiment_id} completed successfully!")
            else:
                self.logger.error(
                    f"Experiment {self.experiment_id} completed with failures "
                    f"({failed_runs} failed, {completed_runs} succeeded)."
                )
            self.logger.info(f"Results saved to {self.experiment_dir}")

            # Print summary
            self._print_summary(aggregate_metrics)
            return overall_success
        except Exception as e:
            self.logger.exception(f"Unhandled exception while running experiment {self.experiment_id}: {e}")
            self.metadata['status'] = 'failed'
            self.metadata['completed_at'] = datetime.now().isoformat()
            self.metadata['aggregate_metrics_summary'] = {
                'avg_gini': None,
                'avg_tasks_completed': None,
                'error': str(e)
            }
            self._save_metadata()
            try:
                self.registry.update_experiment_status(
                    self.experiment_id,
                    'failed',
                    key_metrics=self.metadata['aggregate_metrics_summary']
                )
            except Exception as registry_error:
                self.logger.error(f"Failed to update registry status after exception: {registry_error}")
            return False
    
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


def main() -> int:
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run research-grade experiments')
    parser.add_argument('--config', type=str, help='Path to experiment configuration file')
    parser.add_argument('--resume', type=str, help='Resume an existing experiment by ID')
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                       help='Directory to store experiments (default: experiments)')
    parser.add_argument('--simulation-root', type=str, default=None,
                       help='Simulation code directory to execute (default: from config or information_asymmetry_simulation)')
    
    args = parser.parse_args()
    
    if not args.config and not args.resume:
        parser.error("Either --config or --resume must be provided")
    
    if args.resume:
        runner = ExperimentRunner(
            config_path=None,
            experiments_dir=args.experiments_dir,
            resume_id=args.resume,
            simulation_root=args.simulation_root,
        )
    else:
        runner = ExperimentRunner(
            config_path=args.config,
            experiments_dir=args.experiments_dir,
            simulation_root=args.simulation_root,
        )
    
    success = runner.run_experiment()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
