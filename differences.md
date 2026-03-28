# Original Environment vs. Current Zero-Cost Environment

## Purpose of this note

This document summarizes:

1. What the original `information_asymmetry_simulation` environment is.
2. What the current `zero_cost_transfer_simulation` environment is.
3. What has been preserved across the two.
4. Where they now diverge in a meaningful way.
5. Whether the current environment can reasonably be described as a new environment with a different cooperation mechanism.
6. What we should expect to happen to the `o3-mini > o3` gap as the number of rounds increases.


## 1. Original Environment

The original environment is a multi-agent information-sharing task environment with the following core properties:

- There are 10 agents and 20 rounds.
- Each agent owns personal tasks.
- Each task requires 4 information pieces.
- Agents begin with inventories of reusable information pieces.
- Information is discoverable through a public directory that shows who holds what.
- Agents can:
  - send direct messages,
  - broadcast publicly,
  - send information to other agents,
  - submit completed tasks.
- Sending information is zero-cost in the narrow mechanical sense:
  - the sender keeps the information,
  - the receiver gets a copy.
- Only the task owner receives the completion reward.
- Information can be shared truthfully or with manipulated values, and manipulated values trigger submission penalties.
- The environment has strong corporate/economic framing:
  - revenue board,
  - budget / sales / market data / performance metrics,
  - report / forecast / validation tasks.

Conceptually, the original environment tests whether agents will share useful information with one another, even when doing so directly benefits another agent's private task outcome.

The real-world setting it most closely mimics is a corporate information-sharing workflow:

- teams or departments hold pieces of reusable internal knowledge,
- individuals own revenue-relevant tasks,
- they must identify who has what,
- request it,
- and decide whether to share or withhold information that benefits someone else's deliverable.


## 2. Current Zero-Cost Environment

The current environment in `zero_cost_transfer_simulation` preserves the same broad multi-agent scaffold, but changes the task ecology and the cooperative act itself.

Current key properties:

- Still 10 agents and 20 rounds by default.
- Agents still own personal deliverables.
- Each deliverable still requires 4 pieces.
- Agents still start with reusable resource inventories.
- A directory still reveals ownership information.
- Agents now coordinate only through direct messages.
- There is no public broadcast channel.
- Tasks are not generic bundles anymore.
  - They are structured workflow packets.
  - Information belongs to explicit operational families:
    - manifest,
    - compliance,
    - calibration,
    - inventory,
    - handoff.
- Task generation is owner-anchored and multi-party:
  - one required piece is typically anchored in the owner's initial portfolio,
  - the remaining pieces are usually drawn from distinct outside holders.
- Information portfolios are family-balanced, so the environment is more structured than the original.
- The cooperative act is no longer transfer/copy.
  - Helpers use `grant_access`.
  - The recipient does not receive a new inventory item.
  - Instead, the recipient receives a temporary authorization to use that piece for submission.
- Access grants expire.
- Deliverables also expire after a short service window and are replaced.
- The environment remains zero-cost:
  - granting access does not consume the sender's resource,
  - the sender retains full use of the same resource afterward.

Conceptually, the current environment is a time-bounded workflow-authorization environment rather than a plain information-copy environment.

The real-world setting it most closely mimics is a regulated or operational approval workflow:

- individuals own operational deliverables,
- required components come from different functional domains,
- other agents do not hand over permanent ownership of the underlying resource,
- instead they authorize its use for a limited time,
- and missing a coordination window can invalidate the opportunity and force the workflow to restart.


## 3. What Is Still the Same

The two environments are intentionally comparable.

The following design features are shared:

- Same broad agent/task/reward scaffold.
- Same personal-task ownership structure.
- Same need to obtain help from other agents to complete private work.
- Same zero-cost cooperation principle:
  - helping someone does not remove the helper's resource.
- Same explicit submission step.
- Same revenue board logic.
- Same value-manipulation channel and manipulation penalty.
- Same direct-message request/response social structure.
- Same general prompt layout:
  - revenue status,
  - owned tasks,
  - owned resources,
  - ownership directory,
  - message history,
  - notifications,
  - past actions,
  - private thoughts history.

This means results remain meaningfully comparable to the original paper setup.


## 4. What Is Meaningfully Different

### 4.1 Domain and task ecology

The original environment is framed as business analytics and corporate reporting.

The current environment is framed as operational workflow execution:

- handoff packets,
- delivery authorizations,
- execution bundles,
- field operations briefs,
- dispatch packages.

This is not just a rename. The required information is now structured into operational families, and tasks are packet-like rather than generic combinations.


### 4.1a Real-world analogue

This difference matters because the environments mimic different organizational problems.

The original environment resembles:

- knowledge sharing inside a firm,
- analytics/reporting collaboration,
- internal data exchange for individually owned business tasks.

The current environment resembles:

- operational execution,
- packet assembly / clearance / authorization workflows,
- time-sensitive coordination where resources remain owned by one party but can be temporarily authorized for use by another.


### 4.2 Communication surface

The original environment includes public broadcasting.

The current environment does not.

This matters because it removes a global coordination mechanism and forces cooperation into bilateral interactions. That makes the social choice more local and more directly comparable to one-to-one help decisions.


### 4.3 Resource structure

The original environment uses a flatter information pool.

The current environment uses:

- explicit information families,
- balanced category generation,
- balanced initial portfolios,
- owner-anchored deliverables,
- distinct external dependencies when possible.

This creates a more structured dependency graph than the original.


### 4.4 Temporal structure

This is one of the biggest differences.

The original environment does not have short-lived task service windows or expiring permissions.

The current environment does:

- each deliverable has a service window,
- each access grant has a validity window,
- overdue deliverables are replaced,
- expired grants must be re-obtained.

This changes the strategic texture of the environment substantially. Delay now has mechanical consequences.


### 4.5 Cooperation mechanism

This is the clearest mechanism difference.

Original cooperation mechanism:

- request information,
- helper sends information,
- receiver gets a copied item in inventory,
- copied item can then be used normally.

Current cooperation mechanism:

- request access to a resource,
- helper grants access,
- receiver does not gain ownership of a copied inventory item,
- receiver gets a temporary right to use that resource for submission.

This is a real mechanism change, not a cosmetic one.

The original environment is best described as:

- zero-cost information transfer / copying.

The current environment is best described as:

- zero-cost temporary workflow authorization.


## 5. Can we claim this is a new environment?

Yes.

The strongest accurate claim is:

> This is a new zero-cost workflow-authorization environment built on the same multi-agent request/response scaffold as the original information-sharing environment.

That claim is defensible because the current environment differs from the original in:

- domain,
- task structure,
- communication surface,
- temporal dynamics,
- and the cooperative act itself.

It is not merely a prompt ablation or cosmetic reskin.


## 6. Can we claim it uses a different cooperation mechanism?

Yes, but the wording should be precise.

Recommended wording:

> The environment preserves the original bilateral request/response structure, but replaces zero-cost information transfer with zero-cost temporary access granting under expiring service windows.

That is the right level of strength:

- it does not overclaim a totally different interaction paradigm,
- but it does accurately identify a meaningful mechanism change.


## 7. Bottom line

The current environment is best described as:

- a new, structured, time-bounded, zero-cost workflow-authorization environment
- that preserves the original multi-agent request/response scaffold
- while changing the cooperative act from transfer to temporary access granting
- and introducing time pressure that strongly penalizes strategic delay.

That is enough to claim both:

- a new environment,
- and a different cooperation mechanism.
