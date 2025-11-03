#!/usr/bin/env python3
"""
Synchronization tool to keep experiment configs in sync with main simulation config
"""

import yaml
import argparse
from pathlib import Path
from datetime import datetime
import difflib
import sys


def load_yaml(path: Path):
    """Load a YAML file"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data):
    """Save data to YAML file"""
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def extract_simulation_config(main_config_path: Path):
    """Extract the simulation configuration from main config"""
    config = load_yaml(main_config_path)
    return config


def update_experiment_template(template_path: Path, main_config: dict):
    """Update experiment template with latest main config"""
    # Load existing template
    template = load_yaml(template_path)
    
    # Update simulation_config section
    template['simulation_config'] = main_config
    
    # Add sync timestamp comment at the top
    timestamp = datetime.now().strftime('%Y-%m-%d')
    
    # Save with comment
    with open(template_path, 'r') as f:
        lines = f.readlines()
    
    # Update the sync date in the header comment
    for i, line in enumerate(lines):
        if 'Last synced with' in line:
            lines[i] = f"# Last synced with information_asymmetry_simulation/config.yaml on {timestamp}\n"
            break
    else:
        # If no sync comment found, add it after the first line
        lines.insert(2, f"# Last synced with information_asymmetry_simulation/config.yaml on {timestamp}\n")
    
    # Write back
    with open(template_path, 'w') as f:
        f.writelines(lines[:4])  # Write header with sync comment
        f.write('\n')
    
    # Append the updated config
    with open(template_path, 'a') as f:
        # Write experiment section
        f.write("experiment:\n")
        exp_section = template.get('experiment', {})
        for key, value in exp_section.items():
            if isinstance(value, str) and '\n' in value:
                # Multi-line string
                f.write(f"  {key}: |\n")
                for line in value.split('\n'):
                    f.write(f"    {line}\n")
            elif isinstance(value, list):
                f.write(f"  {key}:\n")
                for item in value:
                    f.write(f"    - {item}\n")
            else:
                f.write(f"  {key}: {value}\n")
        
        f.write("\n# Simulation configuration that will be used for all runs\n")
        f.write("# This directly maps to the information_asymmetry_simulation config.yaml format\n")
        f.write("simulation_config:\n")
        
        # Write simulation config with proper indentation
        def write_dict(d, indent=2):
            for key, value in d.items():
                if isinstance(value, dict):
                    f.write(" " * indent + f"{key}:\n")
                    write_dict(value, indent + 2)
                elif isinstance(value, list):
                    f.write(" " * indent + f"{key}:\n")
                    for item in value:
                        f.write(" " * (indent + 2) + f"- {yaml.dump(item, default_flow_style=True).strip()}\n")
                elif isinstance(value, str) and '#' in value:
                    # Preserve comments in strings
                    f.write(" " * indent + f"{key}: {value}\n")
                else:
                    # Handle comments for specific keys
                    if key == 'show_full_revenue':
                        f.write(" " * indent + f"{key}: {value}  # If true, agents see all revenue. If false, only their own revenue position.\n")
                    elif key == 'report_frequency':
                        f.write(" " * indent + f"{key}: {value}  # Request strategic reports every N rounds (0 to disable)\n")
                    elif key == 'model':
                        f.write(" " * indent + f'{key}: "{value}"  # Can be changed to any OpenAI-compatible model\n')
                    elif key == 'temperature':
                        f.write(" " * indent + f"# {key}: 0.7  # Note: o3-mini doesn't support temperature parameter\n")
                    elif key == 'uncooperative_count':
                        f.write(" " * indent + f"{key}: {value}  # Number of uncooperative agents (0 = all neutral/standard agents)\n")
                    elif key == 'competitive_count':
                        f.write(" " * indent + f"{key}: {value}  # Number of competitive agents (0 = no competitive agents)\n")
                    elif key == 'incorrect_value_penalty':
                        f.write(" " * indent + f"{key}: {value}  # Revenue reduction for submitting tasks with incorrect information values\n")
                    elif key == 'max_actions_per_turn':
                        f.write(" " * indent + f"{key}: {value}\n")
                    elif key == 'unique_distribution':
                        f.write(" " * indent + f"{key}: {value}  # If true, each piece exists exactly once (no duplicates)\n")
                        f.write(" " * (indent + len(str(value)) + len(key) + 2) + "# If false, pieces can be held by multiple agents (1-3 copies)\n")
                    else:
                        f.write(" " * indent + f"{key}: {value}\n")
        
        write_dict(main_config)
        
        # Write analysis section
        f.write("\n# Analysis settings\n")
        f.write("analysis:\n")
        analysis_section = template.get('analysis', {})
        for key, value in analysis_section.items():
            if isinstance(value, list):
                f.write(f"  {key}:\n")
                for item in value:
                    f.write(f"    - {item}\n")
            else:
                f.write(f"  {key}: {value}\n")
            if key == 'key_metrics':
                f.write("  \n")
            elif key == 'generate_plots':
                f.write("  \n")


def check_differences(template_path: Path, main_config_path: Path):
    """Check for differences between template and main config"""
    template = load_yaml(template_path)
    main_config = load_yaml(main_config_path)
    
    template_sim = template.get('simulation_config', {})
    
    # Convert to string for comparison
    template_str = yaml.dump(template_sim, default_flow_style=False)
    main_str = yaml.dump(main_config, default_flow_style=False)
    
    if template_str != main_str:
        print("Differences found between template and main config:")
        print("-" * 60)
        
        diff = difflib.unified_diff(
            template_str.splitlines(keepends=True),
            main_str.splitlines(keepends=True),
            fromfile='experiment_template.yaml (simulation_config)',
            tofile='main config.yaml',
            n=3
        )
        
        for line in diff:
            if line.startswith('+'):
                print(f"\033[92m{line}\033[0m", end='')  # Green for additions
            elif line.startswith('-'):
                print(f"\033[91m{line}\033[0m", end='')  # Red for deletions
            else:
                print(line, end='')
        
        return True
    else:
        print("✓ Template is up-to-date with main config")
        return False


def update_experiment_configs(experiments_dir: Path, main_config: dict, dry_run: bool = False):
    """Update all experiment configs with new fields from main config"""
    updated = []
    
    for config_file in experiments_dir.glob("*.yaml"):
        if config_file.name == "experiment_template.yaml":
            continue
            
        try:
            config = load_yaml(config_file)
            
            # Check if this is an experiment config
            if 'experiment' not in config or 'simulation_config' not in config:
                continue
            
            # Deep merge: add new fields without overwriting existing values
            def merge_configs(old, new):
                """Recursively merge new fields into old config"""
                for key, value in new.items():
                    if key not in old:
                        print(f"  Adding new field: {key}")
                        old[key] = value
                    elif isinstance(value, dict) and isinstance(old[key], dict):
                        merge_configs(old[key], value)
            
            original = yaml.dump(config['simulation_config'])
            merge_configs(config['simulation_config'], main_config)
            modified = yaml.dump(config['simulation_config'])
            
            if original != modified:
                updated.append(config_file.name)
                if not dry_run:
                    save_yaml(config_file, config)
                    print(f"✓ Updated {config_file.name}")
                else:
                    print(f"Would update {config_file.name}")
                    
        except Exception as e:
            print(f"Error processing {config_file.name}: {e}")
    
    return updated


def main():
    parser = argparse.ArgumentParser(description='Sync experiment configs with main simulation config')
    parser.add_argument('--main-config', type=str, 
                       default='information_asymmetry_simulation/config.yaml',
                       help='Path to main simulation config')
    parser.add_argument('--template', type=str,
                       default='experiment_framework/configs/experiment_template.yaml',
                       help='Path to experiment template')
    parser.add_argument('--check', action='store_true',
                       help='Only check for differences without updating')
    parser.add_argument('--update-all', action='store_true',
                       help='Update all experiment configs with new fields')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be updated without making changes')
    
    args = parser.parse_args()
    
    main_config_path = Path(args.main_config)
    template_path = Path(args.template)
    
    if not main_config_path.exists():
        print(f"Error: Main config not found: {main_config_path}")
        sys.exit(1)
    
    if not template_path.exists():
        print(f"Error: Template not found: {template_path}")
        sys.exit(1)
    
    # Load main config
    main_config = extract_simulation_config(main_config_path)
    
    if args.check:
        # Just check for differences
        has_diff = check_differences(template_path, main_config_path)
        sys.exit(1 if has_diff else 0)
    
    # Update template
    print(f"Updating template from {main_config_path}")
    update_experiment_template(template_path, main_config)
    print(f"✓ Template updated: {template_path}")
    
    # Update all experiment configs if requested
    if args.update_all:
        experiments_dir = template_path.parent
        print(f"\nUpdating experiment configs in {experiments_dir}")
        updated = update_experiment_configs(experiments_dir, main_config, args.dry_run)
        
        if updated:
            print(f"\n{'Would update' if args.dry_run else 'Updated'} {len(updated)} config(s)")
        else:
            print("\nNo configs needed updating")


if __name__ == "__main__":
    main()