#!/usr/bin/env python3
"""
Validate Setup Script for Information Asymmetry Simulation

This script performs preflight checks to ensure your environment is properly
configured before running experiments. It checks:
1. Python version
2. Required dependencies
3. API keys (warns if missing)
4. Config files
5. Runs a quick test with perfect mode (no API keys needed)

Run this before sending experiments to collaborators or running real experiments.
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict
import argparse


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_status(check: str, passed: bool, message: str = ""):
    """Print a status line with checkmark or X"""
    status = "✓" if passed else "✗"
    color_start = "\033[92m" if passed else "\033[91m"
    color_end = "\033[0m"

    if message:
        print(f"  {color_start}{status}{color_end} {check}: {message}")
    else:
        print(f"  {color_start}{status}{color_end} {check}")


def check_python_version() -> bool:
    """Check Python version is 3.8+"""
    version = sys.version_info
    passed = version.major >= 3 and version.minor >= 8
    print_status(
        "Python version",
        passed,
        f"{version.major}.{version.minor}.{version.micro}" + ("" if passed else " (need 3.8+)")
    )
    return passed


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check all required dependencies are installed"""
    required_packages = {
        'openai': 'openai',
        'anthropic': 'anthropic',
        'yaml': 'pyyaml',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
    }

    missing = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print_status(f"Package '{package_name}'", True, "installed")
        except ImportError:
            print_status(f"Package '{package_name}'", False, "NOT INSTALLED")
            missing.append(package_name)

    return len(missing) == 0, missing


def check_api_keys() -> Dict[str, bool]:
    """Check which API keys are set"""
    api_keys = {
        'OPENAI_API_KEY': 'OpenAI (GPT models)',
        'ANTHROPIC_API_KEY': 'Anthropic (Claude models)',
        'DEEPINFRA_TOKEN': 'DeepInfra (DeepSeek models)',
        'OPENROUTER_API_KEY': 'OpenRouter (Gemini models)',
    }

    results = {}
    for key, description in api_keys.items():
        is_set = bool(os.environ.get(key))
        results[key] = is_set
        if is_set:
            # Show first/last few chars for verification
            value = os.environ.get(key, "")
            masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
            print_status(f"{key}", True, f"set ({masked})")
        else:
            print_status(f"{key}", False, f"NOT SET - {description} won't work")

    return results


def check_config_files() -> bool:
    """Check required config files exist"""
    config_files = [
        'information_asymmetry_simulation/config.yaml',
        'experiment_framework/run_experiment.py',
    ]

    all_exist = True
    for config_file in config_files:
        exists = Path(config_file).exists()
        print_status(f"Config '{config_file}'", exists, "found" if exists else "MISSING")
        if not exists:
            all_exist = False

    return all_exist


def run_perfect_mode_test(quick: bool = True) -> bool:
    """Run a quick test using perfect mode (no API keys needed)"""
    print("\n  Running quick pipeline test with perfect mode...")
    print("  (This validates the entire pipeline without needing API keys)")

    # Use a minimal configuration for quick testing
    cmd = [
        sys.executable,
        "run_heterogeneous_experiments.py",
        "-perfect", "5",
        "--runs", "1",
        "--output-dir", "validation_test"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )

        if result.returncode == 0:
            print_status("Perfect mode pipeline test", True, "completed successfully")

            # Clean up test output
            import shutil
            test_dir = Path("experiments/validation_test")
            if test_dir.exists():
                shutil.rmtree(test_dir)
                print("  (Cleaned up test output)")

            return True
        else:
            print_status("Perfect mode pipeline test", False, "FAILED")
            if result.stderr:
                print(f"\n  Error output:\n{result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print_status("Perfect mode pipeline test", False, "TIMEOUT (>120s)")
        return False
    except Exception as e:
        print_status("Perfect mode pipeline test", False, f"ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Validate setup for Information Asymmetry Simulation experiments'
    )
    parser.add_argument('--skip-test', action='store_true',
                       help='Skip the perfect mode pipeline test')
    parser.add_argument('--quick', action='store_true',
                       help='Run minimal checks only')
    args = parser.parse_args()

    print_header("SETUP VALIDATION")
    print("  Checking your environment for running experiments...")

    all_passed = True
    warnings = []

    # Check 1: Python version
    print_header("1. Python Version")
    if not check_python_version():
        all_passed = False

    # Check 2: Dependencies
    print_header("2. Required Dependencies")
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        all_passed = False
        print(f"\n  To install missing packages:")
        print(f"  pip install {' '.join(missing)}")

    # Check 3: API Keys
    print_header("3. API Keys")
    api_keys = check_api_keys()
    if not any(api_keys.values()):
        warnings.append("No API keys are set - you can only run 'perfect' mode experiments")
    elif not all(api_keys.values()):
        missing_keys = [k for k, v in api_keys.items() if not v]
        warnings.append(f"Some API keys missing: {', '.join(missing_keys)}")

    # Check 4: Config files
    print_header("4. Configuration Files")
    if not check_config_files():
        all_passed = False

    # Check 5: Pipeline test (optional)
    if not args.skip_test and deps_ok:
        print_header("5. Pipeline Test (Perfect Mode)")
        if not run_perfect_mode_test():
            all_passed = False
    elif args.skip_test:
        print_header("5. Pipeline Test (Perfect Mode)")
        print("  Skipped (--skip-test flag)")
    elif not deps_ok:
        print_header("5. Pipeline Test (Perfect Mode)")
        print("  Skipped (missing dependencies)")

    # Summary
    print_header("SUMMARY")

    if warnings:
        print("\n  ⚠️  Warnings:")
        for warning in warnings:
            print(f"    - {warning}")

    if all_passed:
        print("\n  ✓ All checks passed!")
        print("\n  You're ready to run experiments. Try:")
        print("    python run_all_experiments.py --dry-run")
        print("\n  Or run individual experiments from README.md")
        return 0
    else:
        print("\n  ✗ Some checks failed!")
        print("\n  Please fix the issues above before running experiments.")
        print("  If you only have some API keys, you can still run experiments")
        print("  with the models you have access to.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
