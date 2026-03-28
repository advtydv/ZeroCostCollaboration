#!/usr/bin/env python3
"""
Small wrapper around the local simulation analysis module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simulation.analysis import run_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a zero_cost_transfer_simulation run")
    parser.add_argument("log_dir", type=str, help="Path to a simulation log directory")
    args = parser.parse_args()

    results = run_analysis(Path(args.log_dir))
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
