# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project scope
This repository implements DIA (Discover–Intervene–Adapt) for causal RL: learn causal structure (PCG), execute skill interventions (SIG/options), and adapt behavior in an interleaved loop.

## Setup and installation commands
- Create CPU environment:
  - `make env-cpu`
- Create GPU environment:
  - `make env-gpu`
- Create legacy SB3 environment (Gym 0.21 / SB3 1.8):
  - `make env-sb3-legacy`
- Install package in editable mode (from activated env):
  - `make install`
  - equivalent: `python -m pip install -e .`

## Core run commands
- List supported DIA environments/backends:
  - `python scripts/dia_cli.py list`
- Unified run entrypoint:
  - `python scripts/dia_cli.py run --env <minecraft2d|coinrun|causalworld|cartpole|montezuma> --pcg <notears|variational|simple> --steps 300 --logdir runs/<name>`
- Makefile shortcuts:
  - `make mc2d`
  - `make coinrun`
  - `make causalworld`
  - `make cartpole`
  - `make montezuma`
- Interactive launcher:
  - `make menu`
- Multi-env sweep:
  - `make sweep-long`
- TensorBoard:
  - `make tb`

## Testing commands
- Run all tests:
  - `python -m pytest tests`
- Run a single test file:
  - `python -m pytest tests/test_plan_search.py`
- Run a single test case:
  - `python -m pytest tests/test_plan_search.py::test_plan_from_prereqs`

## Lint/build status in this repo
- There is no dedicated lint target or linter configuration in the repository root (`pyproject.toml` only contains packaging metadata).
- “Build/install” is setuptools editable install via `python -m pip install -e .`.

## High-level architecture (big picture)
### 1) Entrypoints and experiment wiring
- `scripts/dia_cli.py` is the main dispatcher used by Make targets; it maps `--env` to environment-specific `train_*.py` scripts and forwards shared arguments (`--pcg`, `--steps`, fit/buffer settings, goals, PPO flags).
- Environment-focused runners in `scripts/train_*_dia.py` assemble:
  - environment + EVGS adapter,
  - PCG backend,
  - initial SIG,
  - `InterventionSelector`,
  - `DIARunner`.

### 2) Main DIA loop orchestration
- `src/dia/rollout.py` (`DIARunner`) is the center of the runtime loop:
  1. select skill via planner (`InterventionSelector`),
  2. execute option policy,
  3. store macro and per-step transitions in `PCGBuffer`,
  4. periodically fit/update PCG,
  5. optionally auto-expand SIG from updated causal probabilities,
  6. emit metrics via `TBLogger`.
- This file is the primary place to understand interleaving of Discover/Intervene/Adapt behavior.

### 3) Causal graph learning (PCG)
- `src/dia/pcg.py`: `SimplePCG` baseline with Bernoulli edge probabilities and entropy/IG utilities.
- `src/dia/pcg_learner.py`: differentiable NOTEARS-style learner (`DifferentiablePCG`) with acyclicity penalty and interventional masking.
- `src/dia/pcg_variational.py`: variational relaxed-Bernoulli PCG (`VariationalPCG`) with Monte Carlo relaxed sampling and acyclicity on mean probabilities.
- Runner logic relies on a common PCG interface (`probs`, `entropy`, `apply_update`/`fit`).

### 4) Skill graph and planning
- `src/dia/sig.py`: `SIGraph` and `Skill` objects, prerequisites, readiness checks, topological utilities, and skill success/effect statistics.
- `src/dia/plan_search.py`: goal-directed prerequisite planning over SIG.
- `src/dia/planner.py`: phase-based selector (`novel` / `confirm` / `goal`) plus goal-aware plan-following when a task subgoal is provided.
- `src/dia/sig_auto.py`: automatic SIG edge/skill expansion and pruning from PCG posterior thresholds.

### 5) Goal-space abstraction and options
- `src/dia/evgs.py`: EVGS abstraction and subgoal predicate evaluation (`UP`, `DOWN`, `EQUAL`, `REACH`).
- `src/dia/evgs_adapters.py` (+ env-specific EVGS modules): converts env observations/info into interpretable variable vectors.
- `src/dia/options.py`: option execution interface (`OptionPolicy`) with `RandomOption`, `FixedActionOption`, and optional `PPOOption` (SB3-dependent).

### 6) Persistence and logging
- `src/dia/checkpoint.py`: JSON checkpoint save/load for PCG + SIG (+ lightweight option metadata).
- `src/dia/logging_utils.py`: TensorBoard logging wrapper used throughout runners.

## Practical navigation pointers
- Start with `scripts/dia_cli.py` and one env script (for example `scripts/train_minecraft2d_dia.py`) to see end-to-end wiring.
- Then read `src/dia/rollout.py` to understand the control loop and update cadence.
- For algorithmic changes, modify PCG modules and planner/SIG modules before touching env-specific scripts.
