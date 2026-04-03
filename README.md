# Zero cost Paper Experiments

## Setup

1. Create a clean virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install Python deps:
```bash
pip install -r requirements.txt
```

3. Export API credentials.
For the standard paper experiments in this repo, only `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are required. `OPENROUTER_API_KEY` and `DEEPINFRA_TOKEN` are only needed if you want to run Gemini/OpenRouter or DeepSeek/DeepInfra models manually.
Raw provider keys work, and AWS Secrets Manager references also work if the runtime has AWS credentials that can read those secrets:
```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export OPENROUTER_API_KEY="..."
export DEEPINFRA_TOKEN="..."
```

4. Validate:
```bash
python validate_setup.py --skip-test --require-models gpt54 claudeopus46 o3 claude
```

If you are using AWS secret references such as `aws-secretsmanager://arn:aws:secretsmanager:...`,
`validate_setup.py` now verifies that they can actually be resolved before you launch experiments.

## Run full paper matrix
This runs:
- baseline (regular) for GPT-5.4 (`gpt-5.4-2026-03-05`) and Claude Opus 4.6 (`claude-opus-4-6`)
- automated request + automated fulfill (auto baseline) for both models
- interventions (policy, incentive=1000, visibility=limited) for both models
- heterogeneous runs:
  - `-o3 5 -claude 5`
  - `-claude 9 -o3 1`
  - `-claude 1 -o3 9`

```bash
python run_paper_experiments.py
```

## Reward sweep runs

To run the current baseline-environment reward sweep, use:

```bash
python3 run_reward_sweep.py --experiment-name reward_sweep
```

This launches the main environment with no prompt ablation and varies only the
task-completion reward:
- reward levels: `$100`, `$1,000`, `$10,000`, `$100,000`
- models: Claude Sonnet 4, O3-mini, and O3
- one run per reward/model condition at the shared base settings

Execution order:
- models are processed sequentially
- within each model, the 4 reward conditions run in parallel

Outputs are written under:

```text
experiments/reward_sweep/<model>/reward_<amount>/run_001
```

The runner performs a preflight validation first, uses the shared config at
`prompting_ablation/config.yaml`, forces `ablation=none`, and overrides only:
- `agents.model`
- `revenue.task_completion`
- `simulation.rounds` (default `20`, unless changed via `--rounds`)

The configured task-completion reward is also shown explicitly in the agent
prompt, so each condition is prompt-consistent as well as config-consistent.
