#!/usr/bin/env python3
"""
Convenience runner for prompting ablation conditions.
"""

import argparse
from datetime import datetime
from pathlib import Path

from main import run_simulation
from simulation.prompt_ablation import normalize_prompt_ablation_mode


IMPLEMENTED_BATCH_ABLATIONS = ("A", "B", "C")


def main():
    parser = argparse.ArgumentParser(description="Run one or more prompt ablation conditions")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--ablation", type=str, default="all",
                        help="A, B, C, none, or all")
    parser.add_argument("--output-dir", type=str, default="logs/prompt_ablation",
                        help="Base directory for outputs")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--variant", type=str, default=None,
                        help="Environment variant (marketplace, space_station)")
    parser.add_argument("--custom-prompt", type=str, default=None,
                        help="Path to custom prompt file")

    args = parser.parse_args()

    requested = args.ablation.strip().lower()
    if requested == "all":
        ablations = list(IMPLEMENTED_BATCH_ABLATIONS)
    else:
        normalized = normalize_prompt_ablation_mode(args.ablation)
        if normalized == "none":
            ablations = ["none"]
        elif normalized == "C":
            raise NotImplementedError("Ablation C is not implemented yet in prompting_ablation.")
        else:
            ablations = [normalized]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = Path(args.output_dir)

    for ablation in ablations:
        run_name = ablation.lower()
        run_output_dir = base_output / run_name
        sim_id = f"{run_name}_{timestamp}"
        run_simulation(
            config_path=args.config,
            log_level=args.log_level,
            output_dir=str(run_output_dir),
            sim_id=sim_id,
            variant=args.variant,
            custom_prompt=args.custom_prompt,
            ablation_mode=ablation,
        )


if __name__ == "__main__":
    main()
