# DIA CoinRun Agent — Current Plan & Status

Date: 2026-03-13

## Goal
Produce a video of the DIA agent on ProcGen CoinRun showing:
1. Agent executing skills via interventional causal learning (not random drift)
2. PCG (7×7 interventional edge probability heatmap) updating in real time
3. SIG discovering prerequisite edges only when there is sufficient evidence
4. All 7 EVGS variables updating per frame

---

## Agent Coordination

**Agent2** owns Montezuma's Revenge. Do NOT touch:
- `src/dia/pcg_granger.py` — Agent2's GrangerPCG (3-signal mask-aware)
- `src/dia/evgs_montezuma.py` — Agent2's 8-variable wrapper
- `scripts/train_montezuma_gym.py` — Agent2's entry point
- `monterun1/` output directory

**This agent (Agent1)** owns ProcGen CoinRun AND Minecraft 2D→3D:
- `src/dia/interventional_pcg.py` — interventional PCG for CoinRun
- `src/dia/evgs_procgen.py` — 7-variable pixel detector
- `src/dia/evgs_adapters.py` — make_coinrun_evgs()
- `src/dia/options.py` — FixedActionOption + PPOOption
- `scripts/watch_coinrun_dia_v4.py` — SIP algorithm entry point (CoinRun)
- `scripts/watch_coinrun_dia.py` — v3 (kept for reference)
- `src/dia/evgs_minedojo.py` — NEW: MineDojo inventory → 9 binary vars EVGS
- `src/dia/options_minedojo.py` — NEW: MinedojoPPOOption + ItemRewardWrapper
- `scripts/train_minecraft2d_sip.py` — NEW: Phase 1 (2D SIP → pcg_2d.npy + sig_2d.json)
- `scripts/train_minedojo_skill.py` — NEW: PPO training for individual MineDojo skills
- `scripts/transfer_minecraft3d_dia.py` — NEW: Phase 2 (3D execution from 2D SIG)

---

## What Has Been Done

### v3 (Completed — reference only)
- 7-variable EVGS (coin × 4, platform, saw, creature)
- FixedActionOption primitive skills
- Produced `coinrun_dia_v3_rich.mp4` (score=10, 8 edges discovered)
- **Known problem**: planner looped on dodge/evade; PCG learned correlations not causality

### v4 Architecture — Structured Interventional Probing (SIP) [Current]

#### Root causes fixed from v3

| Bug | v3 behaviour | v4 fix |
|-----|-------------|--------|
| Observational PCG | Mixed all skill transitions; computed P(j↑\|state[i]=1) | InterventionalPCG: each skill updates only its own row |
| Dodge/evade loop | saw/creature always ~1 → always high entropy → always selected | Round-robin discovery: each skill tried equally N times before selection |
| Premature edges | Edge added at threshold=0.62 with only 15 buffer samples | Conservative: add_threshold=0.78 AND exec_count >= min_obs=5 |
| Goal phase lockout | Once approach→collect added, collect unreachable without approach | InterventionalPCG won't add spurious edge (pressing RIGHT rarely collects coin) |

#### New files

**`src/dia/interventional_pcg.py`**
- `InterventionalPCG(num_vars, alpha, min_obs)` class
- `update(cause_var, x_before, x_after)` — updates row `cause_var` only
- `exec_count[k]` — how many times skill targeting var k has executed
- `probs[k, j]` = P(j↑ | skill_k executes), stays at 0.5 until exec_count >= min_obs
- `row_entropy(k)` — uncertainty about effects of skill k
- Compatible `.probs` attribute for `expand_sig_from_pcg`

**`scripts/watch_coinrun_dia_v4.py`**
- `InformationDirectedSelector` — 3-phase selection:
  - `discover`: round-robin (least-executed skill first)
  - `confirm`: skill with highest row entropy (still uncertain effects)
  - `goal`: follow causal plan from SIG prerequisites
- Direct per-step PCG update (no buffer batching)
- SIG expansion gated on `exec_count[cause_var] >= min_obs`
- PCG heatmap labelled with exec counts per row
- Output: `coinrun_dia_v4_sip.mp4`

### EVGS (7 Variables) — Complete (from v3)
1. `coin_visible`    – yellow coin present in frame
2. `coin_close`      – coin centroid in left 80%
3. `coin_collected`  – level completed (`prev_level_complete`)
4. `coin_elevated`   – coin centroid in upper 45%
5. `platform_above`  – brown tiles in upper 2/3 of screen
6. `saw_visible`     – gray circular saw hazard in mid-frame
7. `creature_visible`– colorful enemy in right half of mid-frame

### Skills (7) — Complete (from v3)
- `see_coin↑`   → FixedAction(RIGHT, max 100 steps)
- `approach↑`   → FixedAction(RIGHT, max 150 steps)
- `collect↑`    → PhaseAwareOption(PPO stochastic/deterministic)
- `climb↑`      → FixedAction([JUMP_RIGHT×2, RIGHT, UP], max 60 steps)
- `level_coin↓` → FixedAction([JUMP_RIGHT×2, RIGHT], max 80 steps)
- `dodge↓`      → FixedAction([JUMP_RIGHT×2, RIGHT×2], max 60 steps)
- `evade↓`      → FixedAction([JUMP_RIGHT×2, RIGHT×2], max 60 steps)

### Models Available
- `models/coinrun_cnn_ppo_v2.zip` — 3M-step PPO model (best)

---

## Current Status

### v4 — READY TO RUN

```bash
conda run -n dia --cwd /home/flux/DIA/Discover-Intervene-Adapt-Interleaved-Causal-RL \
  python scripts/watch_coinrun_dia_v4.py \
    --model models/coinrun_cnn_ppo_v2.zip \
    --macro_steps 120 \
    --option_steps 500 \
    --num_levels 200 \
    --fps 30 \
    --discover_per_skill 8 \
    --add_threshold 0.78 \
    --out coinrun_dia_v4_sip.mp4
```

Expected behaviour:
- Steps 1–56: round-robin discovery (each of 7 skills tried ~8 times)
- Steps 57+: confirm phase — skills with uncertain effects probed more
- Steps ~80+: goal phase — causal plan drives see_coin→approach→collect
- Score should be higher than v3 (no dodge/evade lockout)

---

## Minecraft 2D → 3D PCG Transfer [NEW]

### Goal
Learn the Minecraft crafting causal graph cheaply in a symbolic 2D env, then
transfer it as a fixed prior to guide a 3D MineDojo agent with PPO skills.

### Causal chain (shared between 2D and 3D)
```
wood, stone → stone_pickaxe → iron_ore
coal + iron_ore + furnace → iron
iron + wood → iron_pickaxe → diamond
```

### Reference: ADAM (OpenCausaLab)
ADAM (arxiv: 2410.22194) uses LLM + intervention-based CD directly in 3D Minecraft
to build a "technology tree" (causal graph). Key similarity: their intervention-based
CD (remove item c, retry action — edge confirmed if effect fails without c) is
analogous to DIA's InterventionalPCG signal. ADAM does NOT do 2D→3D transfer.
DIA's advantage: the 2D→3D transfer makes Phase 2 skip discovery entirely.

### Architecture

```
Phase 1: 2D Symbolic Minecraft (fast, cheap)
  MinecraftChainEnv → SimplePCG (observational, trajectory buffer)
  MinecraftSIPSelector: discover (round-robin) → confirm → goal
  Saves: pcg_2d.npy + sig_2d.json

Phase 2: 3D MineDojo (expensive, real Minecraft)
  Load sig_2d.json → build SIG → topological order
  For each skill in topo order:
    Load models/minedojo/skill_{var}.zip (pre-trained PPO)
    Run MinedojoPPOOption until item acquired or timeout
  Fixed prior: PCG is NOT re-learned in 3D — 2D SIG is the plan
```

### PCG Transfer Key

Phase 1's `SimplePCG` learns:
- `probs[stone, stonepickaxe]` ≈ high (stone present → stonepickaxe crafted)
- `probs[stonepickaxe, ironore]` ≈ high (pickaxe present → ironore mined)
- etc. (correct observational prerequisites from trajectory data)

`sig_2d.json` encodes the discovered edges. Phase 2 reads them directly
and executes skills in the correct causal order without any 3D discovery.

### Environment: MineDojo
- Obs: `obs['inventory']['name'/'quantity']` → 9 binary DIA vars (via evgs_minedojo.py)
- Task: `open-ended` (free play with reward shaping per skill)
- PPO obs: `{"rgb": (64,64,3), "inventory": (9,)}` (MultiInputPolicy)
- Java 11 available on this machine ✓; `pip install minedojo` needed

### Phase 1: Running

```bash
conda run -n dia --cwd /home/flux/DIA/Discover-Intervene-Adapt-Interleaved-Causal-RL \
  python scripts/train_minecraft2d_sip.py \
    --macro_steps 200 \
    --discover_per_skill 12 \
    --add_threshold 0.72 \
    --out minecraft2d_sip.mp4
```

Output: `pcg_2d.npy`, `sig_2d.json`, `minecraft2d_sip.mp4`

### Phase 2a: Train PPO Skills (must be done once)

```bash
# MineDojo install (one-time):
pip install minedojo

# Train each skill (9 total, simplest first):
for var in wood stone coal furnace stonepickaxe ironore iron ironpickaxe diamond; do
  python scripts/train_minedojo_skill.py --var $var --steps 1_000_000 &
done
```

Models saved to: `models/minedojo/skill_{var}.zip`

### Phase 2b: Transfer Execution

```bash
conda run -n dia --cwd /home/flux/DIA/Discover-Intervene-Adapt-Interleaved-Causal-RL \
  python scripts/transfer_minecraft3d_dia.py \
    --sig sig_2d.json \
    --model_dir models/minedojo \
    --max_steps_per_skill 2000 \
    --out transfer_minecraft3d.mp4
```

### Status

| Item | Status |
|------|--------|
| `evgs_minedojo.py` (9 binary vars from MineDojo inventory) | DONE |
| `options_minedojo.py` (MinedojoPPOOption + ItemRewardWrapper) | DONE |
| `train_minecraft2d_sip.py` (Phase 1 SIP script) | DONE |
| `train_minedojo_skill.py` (PPO training script) | DONE |
| `transfer_minecraft3d_dia.py` (Phase 2 transfer script) | DONE |
| `pip install minedojo` | PENDING |
| Phase 1 run (produce pcg_2d.npy + sig_2d.json) | PENDING |
| Phase 2a: Train 9 PPO skills | PENDING |
| Phase 2b: Transfer execution video | PENDING |

---

## Future: GrangerPCG Integration

Agent2 built `GrangerPCG` with 3-signal mask-aware learning:
- Signal 1 (Prerequisite): P(j success↑ | X_t[i]=1, targeting j) — detects "i required for j"
- Signal 2 (Co-change): P(j changes | targeting i, vs baseline) — detects downstream effects
- Signal 3 (Granger): observational P(X_t[i]=1 → j changes) — correlational support

This is strictly more powerful than `InterventionalPCG` (Signal 2 only).
**Once Agent2's run validates, adopt `GrangerPCG` for CoinRun** by swapping
`InterventionalPCG` in `watch_coinrun_dia_v4.py`.

---

## Known Issues / Observations
- `saw_visible` and `creature_visible` have high false-positive rates (background noise).
  Their DOWN-predicate skills (dodge↓, evade↓) will appear to "succeed" due to frame-to-frame
  variation, but the InterventionalPCG won't add spurious edges since their success is noisy.
- Score depends heavily on PPO model quality for `collect↑`. The v2 model works well.
- H stays high if all 7 skills' rows have high entropy — expected during discovery phase.

---

## Files Summary

### CoinRun
| File | Role | State |
|------|------|-------|
| `src/dia/interventional_pcg.py` | InterventionalPCG (CoinRun) | Done |
| `scripts/watch_coinrun_dia_v4.py` | SIP algorithm (CoinRun) | Done |
| `src/dia/evgs_procgen.py` | 7-var pixel detector | Done |
| `src/dia/evgs_adapters.py` | make_coinrun_evgs() | Done |
| `src/dia/options.py` | FixedActionOption | Done |
| `scripts/watch_coinrun_dia.py` | v3 reference | Done |

### Minecraft 2D→3D Transfer
| File | Role | State |
|------|------|-------|
| `src/dia/evgs_minedojo.py` | MineDojo inventory → 9 binary vars | NEW |
| `src/dia/options_minedojo.py` | MinedojoPPOOption + ItemRewardWrapper | NEW |
| `scripts/train_minecraft2d_sip.py` | Phase 1: 2D SIP → pcg/sig files | NEW |
| `scripts/train_minedojo_skill.py` | PPO training per MineDojo skill | NEW |
| `scripts/transfer_minecraft3d_dia.py` | Phase 2: 3D execution from 2D SIG | NEW |
| `AGENT_PLAN.md` | This file | Current |
