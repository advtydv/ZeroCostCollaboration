# Configuration Synchronization Guide

## The Problem
As your simulation evolves, the main `information_asymmetry_simulation/config.yaml` may change (new fields, new agent types, etc.), but experiment configurations could fall behind.

## The Solution: `sync_config.py`

A synchronization tool that keeps experiment configs in sync with the main simulation configuration.

## Usage

### 1. Check for Differences
See if your template is out of sync:
```bash
python experiment_framework/sync_config.py --check
```

### 2. Update Template
Sync the template with latest main config:
```bash
python experiment_framework/sync_config.py
```

### 3. Update All Experiments (Safely)
Add new fields to existing experiment configs without overwriting your custom values:
```bash
# Preview what would change
python experiment_framework/sync_config.py --update-all --dry-run

# Actually update
python experiment_framework/sync_config.py --update-all
```

## How It Works

1. **Template Sync**: Updates `experiment_template.yaml` with the exact structure from main config
2. **Smart Merging**: When updating existing experiments, it only adds NEW fields, never overwrites your custom values
3. **Preserves Comments**: Keeps helpful comments in the config files
4. **Tracks Sync Date**: Adds a timestamp showing when last synced

## Example Scenario

Let's say the main simulation adds a new field `competitive_count`:

**Before sync:**
```yaml
agents:
  model: "gpt-4"
  uncooperative_count: 2
  # competitive_count is missing!
```

**After sync:**
```yaml
agents:
  model: "gpt-4"  # Your custom value preserved
  uncooperative_count: 2  # Your custom value preserved
  competitive_count: 0  # New field added with default
```

## Best Practices

1. **Before Major Experiments**: Always run sync to ensure you have latest fields
   ```bash
   python experiment_framework/sync_config.py --check
   ```

2. **After Simulation Updates**: When you modify the main simulation, sync all experiments
   ```bash
   python experiment_framework/sync_config.py --update-all
   ```

3. **Version Control**: Commit after syncing so you can track changes
   ```bash
   git add experiment_framework/configs/
   git commit -m "Sync experiment configs with latest simulation"
   ```

## Manual Sync (If Needed)

If automatic sync misses something, you can manually ensure your experiment config has all fields from `information_asymmetry_simulation/config.yaml`:

1. Copy the entire config section from main config
2. Paste into `simulation_config:` section of your experiment
3. Modify only the values you want to change

## Troubleshooting

**"Differences found" but they look the same?**
- Check for trailing spaces, different quote styles, or ordering differences
- The tool compares the actual YAML structure, not just text

**New field not appearing?**
- Make sure it's in the main `config.yaml` first
- Run sync without `--dry-run` flag

**Lost custom values?**
- The tool should never overwrite existing values
- Check git history if needed: `git diff experiment_framework/configs/`