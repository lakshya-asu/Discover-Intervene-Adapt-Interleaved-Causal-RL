# Discover-Intervene-Adapt (DIA)

## Project Goals
- Develop causal reinforcement learning agents capable of interleaved discovery, intervention, and adaptation across complex environments.
- Provide a modular research codebase that accelerates experimentation with novel causal discovery and policy learning algorithms.
- Enable reproducible experimentation workflows, from synthetic benchmarks to real-world-inspired domains.

## Repository Layout
- `src/dia/agents/` — Agent interfaces, planners, and specific algorithm implementations.
- `src/dia/envs/` — Environment wrappers, simulators, and instrumentation utilities for interventions.
- `src/dia/models/` — Structural causal models, representation learners, and dynamics predictors.
- `src/dia/policies/` — Policy parameterizations and policy-gradient friendly abstractions.
- `src/dia/training/` — Training pipelines, loss functions, and optimization utilities.
- `src/dia/evaluation/` — Evaluation suites, metrics, and benchmarking harnesses.
- `src/dia/experiments/` — Experiment configurations, sweeps, and reproducibility artifacts.
- `src/dia/utils/` — Shared helpers (logging, configuration, common math/functions).
- `docs/` *(future)* — Technical notes, design docs, and paper-style writeups.
- `tests/` *(future)* — Unit, integration, and regression tests.

Each package currently exposes a placeholder `__init__.py` so that future modules can be added incrementally without refactoring the import tree.

## Development Guidelines
- Prefer `src/dia` for all importable Python code. Keep notebooks, exploratory scripts, and generated assets outside of the `src/` tree.
- Group algorithms, utilities, and experiment definitions by responsibility. If a module spans multiple concerns, consider refactoring into shared utilities or composing from smaller pieces.
- Maintain clear separation between experiment configuration (`src/dia/experiments`) and reusable logic (`src/dia/*`). Config files should be declarative (YAML/TOML/JSON) whenever possible.
- Document any non-trivial module with concise docstrings and, where appropriate, high-level README snippets to aid collaborators.
- Keep dependencies minimal. Add new packages only when they are broadly useful and document them in the project requirements.
- Write tests alongside new features. Favor deterministic tests; if stochasticity is unavoidable, control randomness with fixed seeds.

## Getting Started
1. Create a virtual environment (e.g., `python -m venv .venv && source .venv/bin/activate`).
2. Install required tooling and dependencies (to be documented as the codebase matures).
3. Use the provided package skeleton as a guide for placing new modules and expanding the DIA research platform.

Contributions should follow the guidelines above to keep the structure consistent while the project evolves.
