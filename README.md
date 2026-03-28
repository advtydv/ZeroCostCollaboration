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

## Prompting ablation rebuttal runs

To run the rebuttal prompt-ablation matrix, use:

```bash
python3 run_prompt_ablations.py
```

This launches all 27 prompt-ablation runs:
- 3 models: Claude Sonnet 4, O3-mini, and O3
- 3 prompt ablations: A, B, and C
- 3 seeds per model/ablation

Outputs are written under:

```text
experiments/prompt_ablations/<model>/<a|b|c>/run_###
```

Before launching, make sure the environment is set up and that `OPENAI_API_KEY`
and `ANTHROPIC_API_KEY` are available, either directly or via supported AWS
Secrets Manager references.

## Zero-cost environment batch runs

To run the current zero-cost environment matrix, use:

```bash
python3 run_zero_cost_transfer_experiments.py --experiment-name zero_cost_current_env_main
```

This launches:
- 3 models: O3-mini, Claude Sonnet 4, and O3
- 3 seeds per model
- the 3 seeds for each model in parallel

Outputs are written under:

```text
experiments/zero_cost_current_env_main/<model>/run_###
```

The runner performs a preflight validation first, uses the shared config at
`zero_cost_transfer_simulation/config.yaml`, and only changes model and seed
between runs. Re-running the same command resumes automatically by skipping runs
that already completed successfully.
