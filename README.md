# Zero cost Paper Experiments

## Setup
Use one interpreter consistently for install + run commands (for example, if deps are installed in `python`, run scripts with `python`, not a different `python3`).

1. Install Python deps:
```bash
pip install -r information_asymmetry_simulation/requirements.txt
```

2. Export API keys:
```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

3. Validate:
```bash
python validate_setup.py --require-models gpt52 claudeopus46 o3 claude
```

## Run full paper matrix
This runs:
- baseline (regular) for GPT-5.2 (`gpt-5.2-2025-12-11`) and Claude Opus 4.6 (`claude-opus-4-6`)
- automated request + automated fulfill (auto baseline) for both models
- interventions (policy, incentive=1000, visibility=limited) for both models
- heterogeneous runs:
  - `-o3 5 -claude 5`
  - `-claude 9 -o3 1`
  - `-claude 1 -o3 9`

```bash
python run_paper_experiments.py
```