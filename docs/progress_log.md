mkdir -p docs

cat > docs/progress_log.md <<'MD'
# Progress Log: Discover–Intervene–Adapt (Interleaved Causal RL)

## High-level plan (from proposal + sprints design)
- **Goal**: Build a framework where an agent simultaneously:
  1. Learns **causal structure** of the environment.
  2. Learns **interventional skills** in a structured, curriculum-like manner.
  3. Interleaves discovery and interventions (not assuming prior intervention sets).
- **Target environments**:
  - CartPole with hidden params (length/mass randomization).
  - Procgen CoinRun (procedural visual RL).
  - CausalWorld Pushing/Stacking (continuous manipulation with causal factors).
- **Sprints**:
  - **Sprint 1**: repo + env wrappers + smoke tests.
  - **Sprint 2**: training loop with SB3 + Hydra, simple PPO baselines on wrappers.
  - Later: causal curiosity baseline, intervention graph, curriculum, etc.

---

## Sprint 1 — repo + env wrappers

### Repository setup
- Repo: [Discover-Intervene-Adapt-Interleaved-Causal-RL](https://github.com/lakshya-asu/Discover-Intervene-Adapt-Interleaved-Causal-RL)
- Structure:
dia/
envs/
init.py
base.py
cartpole_hp.py
procgen_coinrun.py
causalworld_env.py
tests/
test_cartpole_hidden_params.py
test_procgen_coinrun.py
test_causalworld_wrapper.py
test_env_api.py
scripts/
train.py

### Environment wrappers
1. **CartPoleHiddenParamsEnv**
 - Wraps `gym.make("CartPole-v1")`.
 - Randomizes pole length & mass.
 - Provides `.observation_space`, `.action_space`, `.reset()`, `.step()`, `.close()`.

2. **ProcgenCoinRunEnv**
 - Wraps `procgen2.ProcgenGym3Env`.
 - Handles tricky `observe()` API: detects where RGB lives.
 - Returns `(obs, info)` in Gymnasium style.
 - Provides `.NUM_ACTIONS`, `.level_seed`.
 - Shapes standardized to `(64,64,3)` uint8.

3. **CausalWorldPushingEnv**
 - Stubbed reset/step for now (`NotImplementedError` → replaced with stub obs for tests).
 - Returns `obs = {"rgb": np.zeros((84,84,3),dtype=uint8), "proprio": ...}`, `info={"factors":{}}`.

### Tests
- Smoke tests check:
- CartPole randomization.
- Procgen wrapper returns proper RGB array + info dict.
- CausalWorld wrapper returns dict with `"rgb"`, `"factors"`.
- `EnvAPI` base class compliance.

### Issues solved
- Added `pyproject.toml` with editable install.
- Fixed `pytest` import errors via `tests/conftest.py`.
- Fixed NumPy 2.x incompatibility (CausalWorld and procgen2 needed `<2.0.0`):
- Pinned to **NumPy 1.26.4**.
- Installed `procgen2==0.10.7`, `gym3==0.3.3`, `gym==0.26.2`.
- Installed `gymnasium==0.29.1`, `stable-baselines3==2.3.2`, `shimmy==1.3.0` with `--no-deps`.
- Added `.unwrapped` shim to wrappers for SB3 compatibility.
- Added `.NUM_ACTIONS`, `.level_seed` attributes to Procgen wrapper for tests.

**Result**: All smoke tests passed.

---

## Sprint 2 — training loop & PPO baseline

### Training script (`scripts/train.py`)
- Uses Hydra for config overrides.
- Arguments: `env=... total_timesteps=... vec_envs=... log_dir=... seed=...`.
- Builds vectorized env:
- `DummyVecEnv` for parallel rollout.
- `VecTransposeImage` for channel-order.
- `VecMonitor` (instead of `Monitor`) for SB3 logging.
- PPO with `"CnnPolicy"`.
- Saves model at `log_dir/ppo_final.zip`.
- Seeds `numpy`, `random`, `torch`.

### Fixes applied
- Replaced `Monitor` with `VecMonitor` (fixed error: `DummyVecEnv.reset got unexpected keyword argument 'seed'`).
- Added `.unwrapped = self` shim inside wrappers for SB3 uniqueness checks.
- Removed unused imports flagged by `ruff`.
- Added `level_seed` attribute to Procgen wrapper.

### Tests after Sprint 2
- `pytest -q -k procgen_coinrun` passed once `.level_seed` added.
- Training ran successfully:

python -m scripts.train env=procgen_coinrun total_timesteps=3000 vec_envs=1 log_dir=runs/procgen_debug
→ PPO initialized, rollout started, logs written.

---

## Outstanding / Next Steps
- Implement **real CausalWorld reset/step** (currently stubbed).
- Implement **causal curiosity baseline** (structured intervention graph).
- Sprint 3+:
- Logging & TensorBoard integration for interventions.
- Curriculum learning setup.
- Causal graph representation learning modules.
- Benchmarks vs SB3 PPO baseline.

---

## Environment / Dependency notes
- **Working combo**:
- `numpy==1.26.4` (pinned).
- `procgen2==0.10.7`, `gym3==0.3.3`, `gym==0.26.2`.
- `gymnasium==0.29.1`.
- `stable-baselines3[extra]==2.3.2`.
- `shimmy==1.3.0`.
- `causal-world==1.2` (requires `gym==0.17.2`, but stub used).
- Installed extras: `hydra-core`, `tensorboard`, `opencv-python==4.8.1.78`.

---

✅ **Status now**:
- Repo scaffolding done.
- Wrappers implemented + tested.
- PPO baseline training runs on Procgen CoinRun.
- Env compatibility issues (NumPy/gym/procgen) solved with careful pinning.
- CausalWorld currently stubbed.
