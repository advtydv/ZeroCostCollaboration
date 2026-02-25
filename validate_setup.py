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


MODEL_SHORTCUTS = {
    'o3mini': 'o3-mini-2025-01-31',
    'o3': 'o3',
    'gpt41mini': 'gpt-4.1-mini',
    'gpt5mini': 'gpt-5-mini',
    'gpt52': 'gpt-5.2-2025-12-11',
    'gpt5_2': 'gpt-5.2-2025-12-11',
    'gpt-5.2': 'gpt-5.2-2025-12-11',
    'deepseek': 'deepseek-ai/DeepSeek-R1-0528-Turbo',
    'claude': 'claude-sonnet-4-20250514',
    'claudesonnet': 'claude-sonnet-4-20250514',
    'claudeopus46': 'claude-opus-4-6',
    'opus46': 'claude-opus-4-6',
    'claude-opus-4.6': 'claude-opus-4-6',
    'gemini': 'google/gemini-2.5-pro',
    'gemini25': 'google/gemini-2.5-pro',
    'gemini25pro': 'google/gemini-2.5-pro',
    'geminiflash': 'google/gemini-2.5-flash',
    'perfect': 'perfect',
}

OPENROUTER_PROVIDERS = {'google', 'meta', 'mistralai', 'cohere', 'databricks', 'amazon', 'x-ai'}


def resolve_model_name(model_token: str) -> str:
    """Resolve model shortcut to full model name."""
    return MODEL_SHORTCUTS.get(model_token.lower(), model_token)


def required_api_key_for_model(model_name: str) -> str:
    """Map a model name to the required API-key environment variable."""
    model_lower = model_name.lower()
    if model_lower == 'perfect':
        return ""
    if model_lower.startswith('claude') or model_lower.startswith('anthropic/claude'):
        return 'ANTHROPIC_API_KEY'
    if '/' in model_name:
        provider = model_name.split('/')[0].lower()
        if provider in OPENROUTER_PROVIDERS:
            return 'OPENROUTER_API_KEY'
        return 'DEEPINFRA_TOKEN'
    return 'OPENAI_API_KEY'


def get_required_api_keys(models: List[str]) -> Dict[str, List[str]]:
    """Build mapping: required_env_var -> list of models that need it."""
    required: Dict[str, List[str]] = {}
    for token in models:
        resolved = resolve_model_name(token)
        env_key = required_api_key_for_model(resolved)
        if not env_key:
            continue
        required.setdefault(env_key, []).append(resolved)
    return required


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
    parser.add_argument('--require-models', nargs='+', default=None,
                       help='Require API keys needed by this list of models')
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

    # Optional strict API-key check based on models that must run
    if args.require_models:
        required_keys = get_required_api_keys(args.require_models)
        missing_required = [key for key in required_keys.keys() if not api_keys.get(key, False)]
        if missing_required:
            all_passed = False
            print_status(
                "Required model API keys",
                False,
                f"missing: {', '.join(sorted(missing_required))}"
            )
            for missing_key in sorted(missing_required):
                models_for_key = ", ".join(sorted(set(required_keys[missing_key])))
                print(f"    {missing_key} is required for: {models_for_key}")
        else:
            print_status("Required model API keys", True, "all required keys are set")

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
        print("    python run_paper_experiments.py --dry-run")
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
