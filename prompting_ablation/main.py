#!/usr/bin/env python3
"""
Prompting Ablation Simulation
Main entry point for prompt-ablation runs.
"""

import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

import yaml

from simulation.analysis import run_analysis
from simulation.logger import SimulationLogger, setup_logging
from simulation.prompt_ablation import (
    IMPLEMENTED_PROMPT_ABLATIONS,
    normalize_prompt_ablation_mode,
)
from simulation.simulation import SimulationManager


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def apply_ablation_override(config: dict, ablation_mode: str = None) -> str:
    """Apply CLI ablation overrides onto the loaded config."""
    configured_mode = config.get('prompting_ablation', {}).get('mode')
    resolved_mode = normalize_prompt_ablation_mode(ablation_mode or configured_mode)
    config.setdefault('prompting_ablation', {})
    config['prompting_ablation']['mode'] = resolved_mode
    return resolved_mode


def apply_seed_override(config: dict, seed: int = None) -> None:
    """Apply deterministic seeding for simulation randomness."""
    if seed is None:
        return

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    config.setdefault('simulation', {})
    config['simulation']['seed'] = seed


def run_simulation(
    config_path: str,
    log_level: str = 'INFO',
    output_dir: str = 'logs',
    sim_id: str = None,
    variant: str = None,
    custom_prompt: str = None,
    ablation_mode: str = None,
    seed: int = None,
):
    """Run one prompt-ablation simulation."""
    if sim_id:
        log_dir = Path(output_dir) / sim_id
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(output_dir) / f"simulation_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_dir, log_level)
    logger = logging.getLogger(__name__)

    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    resolved_mode = apply_ablation_override(config, ablation_mode)
    apply_seed_override(config, seed)
    logger.info(f"Using prompt ablation mode: {resolved_mode}")
    if seed is not None:
        logger.info(f"Using random seed: {seed}")

    variant_overlay = {}
    if variant:
        from variants.variant_loader import VariantLoader
        variant_overlay = VariantLoader.load_variant(variant)
        if variant_overlay:
            logger.info(f"Loaded variant: {variant}")

    sim_logger = SimulationLogger(log_dir)

    try:
        logger.info("Initializing simulation...")
        simulation = SimulationManager(
            config,
            sim_logger,
            variant_overlay=variant_overlay,
            custom_prompt=custom_prompt,
        )

        logger.info("Starting simulation...")
        results = simulation.run()
    except Exception:
        sim_logger.close()
        raise

    results_path = log_dir / "results.yaml"
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)

    logger.info(f"Simulation completed. Results saved to {results_path}")
    logger.info(f"Logs available in {log_dir}")

    logger.info("Running post-simulation analysis...")
    try:
        analysis_results = run_analysis(log_dir)
        logger.info("Analysis completed successfully")

        metrics = analysis_results.get('metrics', {})
        logger.info(f"Total tasks completed: {metrics.get('total_tasks_completed', 0)}")

        gini = metrics.get('revenue_distribution', {}).get('gini_coefficient')
        if gini is not None:
            logger.info(f"Revenue distribution Gini coefficient: {gini}")

        comm_efficiency = metrics.get('communication_efficiency', {}).get('messages_per_completed_task')
        if comm_efficiency:
            logger.info(f"Communication efficiency: {comm_efficiency} messages per task")

    except Exception as e:
        logger.error(f"Failed to run analysis: {e}")
        import traceback
        logger.debug(f"Analysis error traceback: {traceback.format_exc()}")

    summary_title = simulation.prompt_ablation.display_summary_title()
    total_label = simulation.prompt_ablation.display_total_label()

    print("\n=== Simulation Summary ===")
    print(f"Total rounds: {results['total_rounds']}")
    print(f"Total tasks completed: {results['total_tasks_completed']}")
    print(f"Total messages sent: {results['total_messages']}")
    print(f"\n{summary_title}:")
    for i, (agent_id, revenue) in enumerate(results['final_revenue_board'].items(), 1):
        formatted_amount = simulation.prompt_ablation.format_summary_amount(revenue)
        print(f"{i}. Agent {agent_id}: {formatted_amount}")

    total_revenue = sum(results['final_revenue_board'].values())
    print(f"\n{total_label}: {simulation.prompt_ablation.format_summary_amount(total_revenue)}")

    return results, log_dir, resolved_mode


def main():
    parser = argparse.ArgumentParser(description='Run prompting ablation simulation')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--output-dir', type=str, default='logs',
                        help='Directory for simulation outputs')
    parser.add_argument('--sim-id', type=str, default=None,
                        help='Unique simulation ID (used by batch runner)')
    parser.add_argument('--variant', type=str, default=None,
                        help='Environment variant (marketplace, space_station)')
    parser.add_argument('--custom-prompt', type=str, default=None,
                        help='Path to custom prompt file')
    parser.add_argument('--ablation', type=str, default=None,
                        help='Prompt ablation mode: none, A, B, or C')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional random seed for deterministic runs')

    args = parser.parse_args()

    resolved_mode = normalize_prompt_ablation_mode(args.ablation)
    if resolved_mode not in IMPLEMENTED_PROMPT_ABLATIONS:
        raise NotImplementedError(
            f"Ablation {resolved_mode} is not implemented yet in prompting_ablation."
        )

    run_simulation(
        config_path=args.config,
        log_level=args.log_level,
        output_dir=args.output_dir,
        sim_id=args.sim_id,
        variant=args.variant,
        custom_prompt=args.custom_prompt,
        ablation_mode=resolved_mode,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
