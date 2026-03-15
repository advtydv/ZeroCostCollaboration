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
import json
import base64
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
from functools import lru_cache

REPO_ROOT = Path(__file__).resolve().parent


MODEL_SHORTCUTS = {
    'o3mini': 'o3-mini-2025-01-31',
    'o3': 'o3',
    'gpt41mini': 'gpt-4.1-mini',
    'gpt5mini': 'gpt-5-mini',
    'gpt54': 'gpt-5.4-2026-03-05',
    'gpt5_4': 'gpt-5.4-2026-03-05',
    'gpt-5.4': 'gpt-5.4-2026-03-05',
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


def is_secret_reference(value: str) -> bool:
    """Return True when the env value points at AWS Secrets Manager."""
    normalized = value.strip().lower()
    return normalized.startswith("aws-secretsmanager://") or normalized.startswith("arn:aws:secretsmanager:")


def normalize_secret_id(reference: str) -> str:
    """Convert aws-secretsmanager:// refs into SecretId values accepted by AWS."""
    stripped = reference.strip()
    if stripped.lower().startswith("aws-secretsmanager://"):
        return stripped[len("aws-secretsmanager://"):]
    return stripped


def infer_secret_region(secret_id: str) -> Optional[str]:
    """Extract region from a Secrets Manager ARN when available."""
    if secret_id.startswith("arn:aws:secretsmanager:"):
        parts = secret_id.split(":", 5)
        if len(parts) >= 4 and parts[3]:
            return parts[3]
    return None


def extract_secret_value(secret_payload: str, env_name: str, provider_label: str) -> str:
    """Support plain-string secrets and common JSON secret shapes."""
    text = secret_payload.strip()
    if not text:
        raise ValueError(f"Resolved AWS secret for {env_name} was empty")

    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text

        if isinstance(payload, dict):
            candidate_keys = [
                env_name,
                env_name.lower(),
                provider_label,
                provider_label.lower(),
                f"{provider_label.lower()}_api_key",
                f"{provider_label.lower()}_token",
                "api_key",
                "apikey",
                "token",
                "key",
                "secret",
                "value",
            ]
            for key in candidate_keys:
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

            string_values = [value.strip() for value in payload.values() if isinstance(value, str) and value.strip()]
            if len(string_values) == 1:
                return string_values[0]

            raise ValueError(
                f"Resolved AWS secret for {env_name} is JSON but does not contain a recognizable key"
            )

    return text


def fetch_secret_via_botocore(secret_id: str, region_name: Optional[str]) -> str:
    """Fetch a secret using botocore without requiring boto3."""
    from botocore.config import Config
    from botocore.session import Session

    session = Session()
    client = session.create_client(
        "secretsmanager",
        region_name=region_name,
        config=Config(
            connect_timeout=5,
            read_timeout=5,
            retries={"max_attempts": 1, "mode": "standard"},
        ),
    )
    response = client.get_secret_value(SecretId=secret_id)
    if "SecretString" in response and response["SecretString"] is not None:
        return response["SecretString"]
    if "SecretBinary" in response and response["SecretBinary"] is not None:
        return base64.b64decode(response["SecretBinary"]).decode("utf-8")
    raise ValueError(f"AWS secret {secret_id} did not contain SecretString or SecretBinary")


def fetch_secret_via_aws_cli(secret_id: str, region_name: Optional[str]) -> str:
    """Fetch a secret using the AWS CLI as a fallback."""
    if not shutil.which("aws"):
        raise FileNotFoundError("aws CLI not found")

    cmd = ["aws", "secretsmanager", "get-secret-value", "--secret-id", secret_id, "--output", "json"]
    if region_name:
        cmd.extend(["--region", region_name])

    result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
    response = json.loads(result.stdout)
    if "SecretString" in response and response["SecretString"] is not None:
        return response["SecretString"]
    if "SecretBinary" in response and response["SecretBinary"] is not None:
        return base64.b64decode(response["SecretBinary"]).decode("utf-8")
    raise ValueError(f"AWS CLI returned no secret payload for {secret_id}")


@lru_cache(maxsize=32)
def resolve_aws_secret_reference(reference: str, env_name: str, provider_label: str) -> str:
    """Resolve an AWS Secrets Manager reference into the actual provider key."""
    secret_id = normalize_secret_id(reference)
    region_name = infer_secret_region(secret_id)
    errors = []

    try:
        secret_payload = fetch_secret_via_botocore(secret_id, region_name)
        return extract_secret_value(secret_payload, env_name, provider_label)
    except Exception as exc:
        errors.append(f"botocore: {exc}")

    try:
        secret_payload = fetch_secret_via_aws_cli(secret_id, region_name)
        return extract_secret_value(secret_payload, env_name, provider_label)
    except Exception as exc:
        errors.append(f"awscli: {exc}")

    raise ValueError(
        f"Could not resolve {env_name} via AWS Secrets Manager reference. "
        f"Tried botocore/awscli. Details: {' | '.join(errors)}"
    )


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


def check_api_keys() -> Tuple[Dict[str, bool], Dict[str, str]]:
    """Check which API keys are set"""
    api_keys = {
        'OPENAI_API_KEY': 'OpenAI (GPT models)',
        'ANTHROPIC_API_KEY': 'Anthropic (Claude models)',
        'DEEPINFRA_TOKEN': 'DeepInfra (DeepSeek models)',
        'OPENROUTER_API_KEY': 'OpenRouter (Gemini models)',
    }
    provider_labels = {
        'OPENAI_API_KEY': 'OpenAI',
        'ANTHROPIC_API_KEY': 'Claude',
        'DEEPINFRA_TOKEN': 'DeepInfra',
        'OPENROUTER_API_KEY': 'OpenRouter',
    }

    results = {}
    invalid_values = {}
    for key, description in api_keys.items():
        value = os.environ.get(key)
        is_set = bool(value)
        if is_set and value and is_secret_reference(value):
            try:
                resolve_aws_secret_reference(value, key, provider_labels[key])
                results[key] = True
                print_status(f"{key}", True, "resolved via AWS Secrets Manager")
            except Exception as exc:
                results[key] = False
                invalid_values[key] = f"set to AWS secret reference but resolution failed: {exc}"
                print_status(f"{key}", False, invalid_values[key])
        elif is_set and value:
            results[key] = True
            # Show first/last few chars for verification
            masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
            print_status(f"{key}", True, f"set ({masked})")
        else:
            results[key] = False
            print_status(f"{key}", False, f"NOT SET - {description} won't work")

    return results, invalid_values


def check_config_files() -> bool:
    """Check required config files exist"""
    config_files = [
        'information_asymmetry_simulation/config.yaml',
        'experiment_framework/run_experiment.py',
    ]

    all_exist = True
    for config_file in config_files:
        exists = (REPO_ROOT / config_file).exists()
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
        str(REPO_ROOT / "run_heterogeneous_experiments.py"),
        "-perfect", "5",
        "--runs", "1",
        "--output-dir", "validation_test"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=str(REPO_ROOT)
        )

        if result.returncode == 0:
            print_status("Perfect mode pipeline test", True, "completed successfully")

            # Clean up test output
            import shutil
            test_dir = REPO_ROOT / "experiments" / "validation_test"
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
    api_keys, invalid_api_values = check_api_keys()
    if not any(api_keys.values()):
        if invalid_api_values:
            warnings.append("No usable API keys are set - one or more values are unresolved secret references")
        else:
            warnings.append("No API keys are set - you can only run 'perfect' mode experiments")
    elif not all(api_keys.values()):
        missing_keys = [k for k, v in api_keys.items() if not v]
        warnings.append(f"Some API keys missing: {', '.join(missing_keys)}")
    if invalid_api_values:
        all_passed = False
        for key, reason in invalid_api_values.items():
            print(f"    {key}: {reason}")
        print("    Fix: provide AWS credentials that can read the referenced secret, or export the raw provider key string.")

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
