## Prerequisites

1. **Python 3.8+** installed
2. **API Keys** for the models you want to test:
   ```bash
   export OPENAI_API_KEY="your-key-here"        # For GPT models
   export ANTHROPIC_API_KEY="your-key-here"     # For Claude models
   export DEEPINFRA_TOKEN="your-key-here"       # For DeepSeek models
   export OPENROUTER_API_KEY="your-key-here"    # For Gemini models via OpenRouter
   ```

3. **Install dependencies**:
   ```bash
   pip install openai anthropic pyyaml numpy matplotlib
   # Or use the requirements file:
   pip install -r information_asymmetry_simulation/requirements.txt
   ```

## Available Models

Use these shortcuts for convenience:

| Shortcut | Full Model Name |
|----------|----------------|
| `o3mini` | o3-mini-2025-01-31 |
| `o3` | o3 |
| `gpt41mini` | gpt-4.1-mini |
| `gpt5mini` | gpt-5-mini |
| `deepseek` | deepseek-ai/DeepSeek-R1-0528-Turbo |
| `claude` | claude-sonnet-4-20250514 |
| `gemini` | google/gemini-2.5-pro |
| `geminiflash` | google/gemini-2.5-flash |
| `perfect` | Optimal behavior (non-LLM) |


## Two Ways to Run Experiments

### 1️⃣ Agent Behavior Types (`run_agent_types_experiments.py`)

Configure the mix of cooperative, uncooperative, and competitive agents to study behavioral dynamics. Focusing primarily on mix of uncooperative and cooperative right now.

Can choose which models to run with. If gemini model access is available, then the 'all' tag can be used as shown in the example below.

#### Experiments to run

```bash
# Mostly cooperative agents
python run_agent_types_experiments.py --neutral 9 --uncooperative 1 --models o3 claude gpt41mini gpt5mini deepseek o3mini

# Equal Mix (Replace all tag with models if gemini models not available)
python run_agent_types_experiments.py --neutral 5 --uncooperative 5 --models all

# Fully uncooperative baseline
python run_agent_types_experiments.py --uncooperative 10 --models o3 claude gpt41mini gpt5mini deepseek o3mini
```

### 2️⃣ Heterogeneous Models (`run_heterogeneous_experiments.py`)

Mix different AI models in the same experiment to study cross-model interactions.

#### Experiments to run

```bash
# Half O3, half Claude: one is a low performer, another is a high performer (can replace claude with gemini if access is available)
python run_heterogeneous_experiments.py -o3 5 -claude 5

# One weak agent in system with stronger agents
python run_heterogeneous_experiments.py -claude 9 -o3 1

# One strong agent in system with weaker agents
python run_heterogeneous_experiments.py -claude 1 -o3 9

# Three-way model mix
python run_heterogeneous_experiments.py -o3 4 -claude 3 -deepseek 3

# Complex configuration
python run_heterogeneous_experiments.py -o3 2 -gpt5mini 3 -o3mini 3 -claude 2
```

## Results

Each experiment creates organized outputs in `experiments/[category]/[experiment_name]/`:

```
experiment_folder/
├── aggregate_metrics.json
├── analysis_report.md
├── experiment_config.yaml
├── plots/
└── runs/
    └── run_001/
        └── simulation/
            ├── results.yaml    # Raw results
            ├── simulation.log  # Detailed logs
            └── analysis_results.json
```

## Configuration Options

Both scripts support these parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--runs` | Number of simulations | 5 |
| `--output-dir` | Results directory name | Script-specific |
| `--config` | Base configuration file | config.yaml |

Agent types script additional options:
- `--rounds`: Simulation rounds (default: 20)
- `--revenue`: Visibility mode (`full` or `limited`)
- `--preset`: Quick configurations

---