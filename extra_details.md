Useful options:
```bash
python run_paper_experiments.py --dry-run
python run_paper_experiments.py --runs 3
python run_paper_experiments.py --continue-on-error
python run_paper_experiments.py --models claudeopus46 gpt54
```

## Exact config changes per experiment
Baseline reference:
- Regular experiments start from `information_asymmetry_simulation/config.yaml`.
- Automated experiments start from:
  - `information_asymmetry_simulation_automated/config_automated_request.yaml`
  - `information_asymmetry_simulation_automated/config_automated_fulfill.yaml`

Per experiment, these are the applied overrides:

1. Regular baseline (`paper_regular_baseline`)
- `simulation_root`: `information_asymmetry_simulation`
- `agents.model`: selected model
- `simulation.agents=10`
- `agents.uncooperative_count=0`
- `agents.competitive_count=0`
- `agents.policy_count=0`

2. Automated request (`paper_auto_request`)
- `simulation_root`: `information_asymmetry_simulation_automated`
- Base config already sets `simulation.automation_mode=automated_request`
- `agents.model`: selected model
- `simulation.agents=10`
- `agents.uncooperative_count=0`
- `agents.competitive_count=0`
- `agents.policy_count=0`

3. Automated fulfill / auto baseline (`paper_auto_fulfill`)
- `simulation_root`: `information_asymmetry_simulation_automated`
- Base config already sets `simulation.automation_mode=automated_fulfill`
- `agents.model`: selected model
- `simulation.agents=10`
- `agents.uncooperative_count=0`
- `agents.competitive_count=0`
- `agents.policy_count=0`

4. Policy intervention (`paper_intervention_policy`)
- `simulation_root`: `information_asymmetry_simulation`
- `agents.model`: selected model
- `simulation.agents=10`
- `agents.policy_count=10`
- `agents.uncooperative_count=0`
- `agents.competitive_count=0`
- Policy text comes from `agent_type=="policy"` in `information_asymmetry_simulation/simulation/agent.py`

5. Incentive intervention (`paper_intervention_incentive`)
- `simulation_root`: `information_asymmetry_simulation`
- `agents.model`: selected model
- `simulation.agents=10`
- `agents.uncooperative_count=0`
- `agents.competitive_count=0`
- `agents.policy_count=0`
- `revenue.information_sharing=1000`

6. Visibility intervention (`paper_intervention_visibility`)
- `simulation_root`: `information_asymmetry_simulation`
- `agents.model`: selected model
- `simulation.agents=10`
- `agents.uncooperative_count=0`
- `agents.competitive_count=0`
- `agents.policy_count=0`
- `simulation.show_full_revenue=false` (via `--revenue limited`)

7. Heterogeneous runs (`paper_heterogeneous`)
- `simulation_root`: `information_asymmetry_simulation`
- Base config: `information_asymmetry_simulation/config.yaml`
- Dynamic overrides from `run_heterogeneous_experiments.py`:
  - `simulation.agents=<sum of provided counts>`
  - `agents.mode=mixed`
  - `agents.agent_modes` set per agent (`llm` vs `perfect`)
  - `agents.agent_models` set per LLM agent (exact requested model per agent)
  - `agents.model=<first requested LLM model>` kept as a backward-compatible fallback

## Run manually (if needed)

Regular baseline (example model):
```bash
python run_agent_types_experiments.py \
  --neutral 10 \
  --model gpt54 \
  --config information_asymmetry_simulation/config.yaml \
  --simulation-root information_asymmetry_simulation \
  --output-dir paper_regular_baseline
```

Automated request:
```bash
python run_agent_types_experiments.py \
  --neutral 10 \
  --model gpt54 \
  --config information_asymmetry_simulation_automated/config_automated_request.yaml \
  --simulation-root information_asymmetry_simulation_automated \
  --output-dir paper_auto_request
```

Automated fulfill:
```bash
python run_agent_types_experiments.py \
  --neutral 10 \
  --model gpt54 \
  --config information_asymmetry_simulation_automated/config_automated_fulfill.yaml \
  --simulation-root information_asymmetry_simulation_automated \
  --output-dir paper_auto_fulfill
```

Policy intervention:
```bash
python run_agent_types_experiments.py \
  --policy 10 \
  --model gpt54 \
  --config information_asymmetry_simulation/config.yaml \
  --simulation-root information_asymmetry_simulation \
  --output-dir paper_intervention_policy
```

Incentive intervention:
```bash
python run_agent_types_experiments.py \
  --neutral 10 \
  --model gpt54 \
  --config information_asymmetry_simulation/config.yaml \
  --simulation-root information_asymmetry_simulation \
  --sharing-incentive 1000 \
  --output-dir paper_intervention_incentive
```

Visibility intervention:
```bash
python run_agent_types_experiments.py \
  --neutral 10 \
  --model gpt54 \
  --config information_asymmetry_simulation/config.yaml \
  --simulation-root information_asymmetry_simulation \
  --revenue limited \
  --output-dir paper_intervention_visibility
```

Heterogeneous runs:
```bash
python run_heterogeneous_experiments.py -o3 5 -claude 5 --simulation-root information_asymmetry_simulation --output-dir paper_heterogeneous
python run_heterogeneous_experiments.py -claude 9 -o3 1 --simulation-root information_asymmetry_simulation --output-dir paper_heterogeneous
python run_heterogeneous_experiments.py -claude 1 -o3 9 --simulation-root information_asymmetry_simulation --output-dir paper_heterogeneous
```
## Codebase split
- Use `information_asymmetry_simulation` for:
  - regular baseline
  - policy / incentive / visibility interventions
  - heterogeneous mixes
- Use `information_asymmetry_simulation_automated` only for:
  - automated request
  - automated fulfill (auto baseline)

The scripts in this repo now route these explicitly.
