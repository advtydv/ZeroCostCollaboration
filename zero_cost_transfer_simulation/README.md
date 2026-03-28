# Zero-Cost Transfer Simulation

This folder now runs a fork of the original information asymmetry environment inside the `zero_cost_transfer_simulation/` workspace.

The reason for this reset is empirical: the original request/response environment reliably reproduced the `o3-mini > o3` ordering, while the newer project-contribution variants did not. The active code path here therefore mirrors the original environment's mechanics much more closely, but with several deliberate differences from the original simulator:

- there is no public broadcast channel; coordination is direct-message-only
- each task is owner-anchored by default, meaning one required piece comes from the task owner's original portfolio
- the remaining required pieces are drawn from distinct outside owners when possible
- each deliverable is a structured workflow packet, typically requiring one piece from each information family rather than an arbitrary set of pieces
- the information pool and starting agent portfolios are family-balanced, so each agent begins with a structured mix of manifests, compliance memos, calibration records, inventory snapshots, and handoff codes
- cooperation is access-based rather than transfer-based: agents request and grant reusable access to components instead of copying them into each other's inventories
- the surface domain is now operations / dispatch / handoff packets rather than the original quarterly-revenue vocabulary

- 10 agents
- round count is controlled by `config.yaml`
- 2 active personal tasks per agent
- 4 required information pieces per task
- reusable, non-rivalrous access grants
- direct messaging, revenue board, system notifications
- explicit `grant_access` and `submit_task` actions

The cooperative act remains zero-cost in the narrow sense used in the paper: granting access does not remove the information from the sender.

## Manual run

```bash
python3 zero_cost_transfer_simulation/main.py --config zero_cost_transfer_simulation/config.yaml --sim-id manual_test
```

Outputs are written under `zero_cost_transfer_simulation/logs/` by default unless `--output-dir` is overridden.

Key output files per run:

- `results.yaml`
- `analysis_results.json`
- `simulation.log`
- `simulation_log.jsonl`

## Notes

- The existing `logs/` directory was preserved intentionally.
- `config_perfect.yaml`, `config_sharing_incentive.yaml`, and `config_mixed.yaml` are copied from the original simulator for convenience.
- `config_auto_request.yaml` and `config_auto_fulfill.yaml` are now simple copies of the base config so they do not point at the obsolete contribution-only environment.
