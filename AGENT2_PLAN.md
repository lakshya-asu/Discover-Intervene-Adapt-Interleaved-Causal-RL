# AGENT2_PLAN — DIA Causal RL Improvement for Montezuma's Revenge

## Problem Statement
The original DIA system defaults to a degenerate "always dodge" policy on
Montezuma's Revenge because three cascading bugs prevent any real causal
learning from occurring.  The agent "wins" by luck/repetition rather than
by following a causally-grounded plan.

---

## Root Causes Diagnosed

### Bug 1 — Entropy starts below entropy_low → immediate goal phase
With 4 variables and `init_edge_prob=0.05`:
  entropy = 12 × H(0.05) ≈ 3.5 < entropy_low=5.0
The planner starts in "goal" phase before any learning.  It tracks a fixed
skill (whichever has the best random success rate) and never explores.

Fix: Use 8 semantically richer variables AND `init_edge_prob=0.5` (maximum
uncertainty).  Starting entropy ≈ 38.8 >> entropy_high=25 → correct "novel"
phase at launch.

### Bug 2 — Intervention mask ignored during causal learning
`DIARunner._maybe_fit_pcg` calls `_transition_fit_probs(X_t, X_tp1)` which
does NOT receive the mask.  Interventional and observational transitions are
pooled together, erasing the clean causal signal interventions provide.

Fix: `GrangerDIARunner` subclass overrides `_maybe_fit_pcg` to call
`GrangerPCG.fit_from_transitions(X_t, X_tp1, mask)`.

### Bug 3 — Wrong causal signal for prerequisite-type relationships
The original estimator asks: "does X_t[i]>0.5 predict j increasing?"
This does NOT detect prerequisite relationships like "you must have a key
before you can open a door".

Fix: Three-signal GrangerPCG:
  Signal 1 (Prerequisite): Among steps targeting skill-j, does X_t[i]=1
    increase the success rate?  → Detects "i required for j".
  Signal 2 (Co-change): When targeting i, does j also change more than
    baseline?  → Detects direct downstream effects.
  Signal 3 (Granger): In clean observational steps, does X_t[i]=1 predict
    j changing?  → Supplementary correlational support.

### Bug 4 — RAM addresses unpopulated / wrong defaults
`MontezumaDetectorsConfig` had all None defaults.  Skull X was mis-set to
byte 66 (static).  Score detection used BCD parsing on wrong bytes.

Fix: Validated addresses from live ALE probe:
  player_x = RAM[42]  (range 0-152, start=77)
  player_y = RAM[43]  (range ~148-235, higher = lower on screen)
  room     = RAM[3]   (value 1 = first room)
  skull_x  = RAM[47]  (oscillates 28-58, confirmed moving)
  Skull Y  = fixed constant (145) for room-1 platform height
  has_key / door_open = detected from per-step env reward (≥100 / ≥300)

### Bug 5 — RandomOption generates zero-signal data
Pure random actions in Montezuma almost never achieve predicate changes
(getting the key, reaching a zone).  The buffer fills with noise transitions
and the PCG gets nothing to learn from.

Fix: `DirectedOption` — 75% of actions are drawn from a small preferred
set appropriate for each target variable (e.g., UP/UPRIGHT for on_upper_level,
RIGHT for at_key_zone); 25% are fully random for exploration.

### Bug 6 — _get_ram() fails for gymnasium env stack
gymnasium wraps the raw AtariEnv in two layers: `OrderEnforcing` and
`PassiveEnvChecker`, neither of which has an `ale` attribute.  The original
`.env`-chain walk reaches AtariEnv at depth 2 *in theory*, but the actual
gymnasium stack uses property accessors that can short-circuit early.

Fix: Added `.unwrapped` as a direct fallback in `_get_ram()`:
```python
# After .env chain walk fails:
unwrapped = self.env.unwrapped
ale = getattr(unwrapped, "ale", None)
if ale is not None: return ale.getRAM()
```
Confirmed by live test: RAM is now read correctly, player x/y updated properly.

### Bug 7 — Zone boundaries too tight (player fell outside all zones)
After 40 LEFT actions, probe showed player at X=63, Y=192.
The original `near_rope` zone X=(65, 95) missed X=63 by 2 pixels.
`on_upper_level` threshold Y<175 was too strict for the mid-platform Y=192.

Fix:
  `_ROOM1_ROPE_X = (55, 102)` – covers both LEFT (X=63) and RIGHT (X=89) rope approach
  `_ROOM1_ROPE_Y = (160, 220)` – slightly widened
  `_ROOM1_UPPER_Y_MAX = 185`  – now includes Y=180-184 (staircase top)
Confirmed: after 40 LEFT → near_rope=1.0 (was 0.0); after 40 RIGHT → near_rope=1.0.

### Bug 8 — DIARunner / RunnerConfig import missing in training script
`GrangerDIARunner(DIARunner)` and `RunnerConfig` were used without import.

Fix: Added `from dia.rollout import DIARunner, RunnerConfig` to train script.

### Bug 9 — Game startup animation prevents movement
For ~30 frames after env.reset() the player cannot take actions (game animation).
Options executing immediately after reset waste steps on frozen state.

Fix: Added 60-frame NOOP warmup in training script after env creation:
```python
env.reset()
for _ in range(60): env.step(0)
```

---

## New Architecture

### New Files
```
src/dia/pcg_granger.py          — GrangerPCG (3-signal mask-aware learner)
scripts/train_montezuma_gym.py  — New training entry-point
```

### Modified Files
```
src/dia/evgs_montezuma.py       — MontezumaRichWrapper + make_montezuma_evgs_rich()
                                  (8 variables, correct RAM addresses, reward latches)
```

### Key Classes

**GrangerPCG** (`pcg_granger.py`)
- `fit_from_transitions(X_t, X_tp1, mask)` — main 3-signal fitting method
- `apply_update(new_probs)` — momentum-blended update, returns KL (IG)
- `entropy()` — Bernoulli entropy over all off-diagonal edges
- `top_edges(k, var_names)` — diagnostic: highest-probability edges

**MontezumaRichWrapper** (`evgs_montezuma.py`)
- Wraps raw gymnasium Atari env
- Injects 8 semantic variables into info dict via RAM reads + reward tracking
- Supports frame recording (`start_recording()`, `get_frames()`)
- Variables: has_key, door_open, at_key_zone, at_door_zone,
             on_upper_level, near_rope, skull_near, score_gained

**GrangerDIARunner** (`train_montezuma_gym.py`)
- Subclass of DIARunner
- Overrides `_maybe_fit_pcg` to pass mask to `fit_from_transitions`

**DirectedOption** (`train_montezuma_gym.py`)
- Biased random option per variable's target zone
- Preferred actions calibrated to Atari action set

---

## Variable Semantic Design (8 vars)

| # | Name           | Detection              | Expected Causal Role              |
|---|----------------|------------------------|-----------------------------------|
| 0 | has_key        | env reward ≥ 100       | terminal goal prerequisite        |
| 1 | door_open      | env reward ≥ 300       | main task goal                    |
| 2 | at_key_zone    | RAM[42/43] vs zone box | spatial prereq for has_key        |
| 3 | at_door_zone   | RAM[42/43] vs zone box | spatial prereq for door_open      |
| 4 | on_upper_level | RAM[43] < 180          | intermediate navigation state     |
| 5 | near_rope      | RAM[42/43] vs rope box | intermediate navigation state     |
| 6 | skull_near     | dist(player, skull)    | danger signal (anti-target)       |
| 7 | score_gained   | env reward > 0         | proxy for any forward progress    |

Expected discoverable causal structure for Room 1:
  near_rope → on_upper_level
  on_upper_level → at_key_zone
  at_key_zone → has_key
  has_key + at_door_zone → door_open
  has_key → score_gained
  door_open → score_gained

---

## Output Structure (monterun1/)

```
monterun1/
├── videos/
│   ├── episode_step0200.mp4    (recorded at each PCG print checkpoint)
│   └── ...
├── pcg/
│   ├── pcg_step0200.png        (labelled heatmap)
│   └── pcg_final.png
├── sig/
│   ├── sig_step0200.png        (labelled graph)
│   └── sig_final.png
├── tensorboard/                (TBLogger scalar logs)
└── summary.txt                 (end-of-run stats)
```

---

## Entropy Calibration (8 vars, 56 off-diagonal edges)

| State                         | Approx entropy | Phase        |
|-------------------------------|----------------|--------------|
| All edges at 0.5 (start)      | 56 × ln2 ≈ 38.8 | novel (> 25) |
| Most → 0.1, few → 0.8         | ~18–22         | confirm      |
| Most → 0.02, true edges → 0.9 | ~5–8           | goal (< 5)   |

Planner thresholds: entropy_high=25.0, entropy_low=5.0 (unchanged defaults).

---

## Status

| Item                                      | Status      |
|-------------------------------------------|-------------|
| GrangerPCG (3-signal)                     | DONE        |
| MontezumaRichWrapper (validated RAM)      | DONE        |
| make_montezuma_evgs_rich (8 vars)         | DONE        |
| GrangerDIARunner                          | DONE        |
| DirectedOption                            | DONE        |
| train_montezuma_gym.py (base)             | DONE        |
| Fix env to gymnasium                      | DONE        |
| Fix reward-based latch detection          | DONE        |
| PCG heatmap visualisation                 | DONE        |
| SIG graph visualisation                   | DONE        |
| Video recording                           | DONE        |
| monterun1/ output structure               | DONE        |
| Fix _get_ram() gymnasium fallback (Bug 6) | DONE        |
| Recalibrate zone boundaries (Bug 7)       | DONE        |
| Fix missing DIARunner import (Bug 8)      | DONE        |
| Startup NOOP warmup (Bug 9)              | DONE        |
| VLM scene analyzer (vlm_scene.py)         | DONE        |
| VLM-RAM fusion EVGS (evgs_vlm.py)        | DONE        |
| --use_vlm flag + VLM-annotated PCG/SIG    | DONE        |
| End-to-end run + validation               | DONE        |

---

## VLM Integration (Claude claude-opus-4-6)

### New Files
- `src/dia/vlm_scene.py` — `VLMSceneAnalyzer`, `VLMSceneConfig`, `VLMSceneResult`
- `src/dia/evgs_vlm.py` — `VLMEnrichedEVGS` drop-in EVGS wrapper

### Architecture
```
  game frame (RGB 210x160) from obs["obs"]
        │
        ▼  (every call_every=200 primitive steps; cached between)
  VLMSceneAnalyzer
    ├─ Upscale 2× → 420×320 PNG
    ├─ base64-encode → Anthropic Messages API
    ├─ tool_use: report_scene_state  (guaranteed structured output)
    └─ VLMSceneResult:
         has_key, door_open, at_key_zone, at_door_zone,
         on_upper_level, near_rope, skull_near, score_gained  ∈ [0,1]
         scene_description (str)  +  player_action (enum)
        │
        ▼  (every primitive step, zero extra latency after first call)
  VLMEnrichedEVGS.extract(obs)
    ├─ RAM hard binary    (from MontezumaRichWrapper info dict)
    ├─ VLM soft probs     (from VLMSceneResult cache)
    └─ fused = α·RAM + (1-α)·VLM     default α=0.6
        │
        ▼
  PCGBuffer              (soft transitions → richer gradient signal)
  GrangerPCG.fit()       (benefits from continuous [0,1] values)
  PCG/SIG plots          (subtitle annotated with latest VLM description)
```

### Usage
```bash
export CLAUDE_API_KEY=sk-ant-...    # your personal Anthropic API key
conda activate dia
python scripts/train_montezuma_gym.py \
    --steps 400 --outdir monterun1 \
    --use_vlm --vlm_call_every 200 --vlm_alpha 0.6 --vlm_verbose
```

### Graceful fallback
If `CLAUDE_API_KEY` is unset or any API error occurs, the VLM silently
falls back to pure RAM values.  Error message printed once only.  Training
continues at full speed with no API dependency.

### Note on proxy keys
`ANTHROPIC_API_KEY` inside Claude Code is an internal proxy key that returns
raw SSE text instead of parsed Message objects.  Always set `CLAUDE_API_KEY`
to your own personal key for direct VLM API access.

---

## Running

```bash
conda activate dia
cd Discover-Intervene-Adapt-Interleaved-Causal-RL
pip install -e .
python scripts/train_montezuma_gym.py \
    --steps 400 \
    --outdir monterun1 \
    --goal has_key \
    --checkpoint_every 50
```
