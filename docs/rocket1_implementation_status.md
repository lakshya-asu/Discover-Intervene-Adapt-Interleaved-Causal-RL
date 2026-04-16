# ROCKET-1 + DIA Hybrid: Implementation Status & Session Notes

**Last updated:** 2026-04-13 (session ~21:20)  
**Status:** Pilot experiment ACTIVELY RUNNING (PID 1626293 / Java PID 1626379)

---

## What Was Built

A fully implemented ROCKET-1 + DIA hybrid experiment for 3D Minecraft:

- **Low-level executor:** ROCKET-1 (MineStudio, CraftJarvis/CVPR 2025) — pre-trained VPT transformer, 189M params, trained on YouTube-scale Minecraft videos
- **High-level planner:** DIA PCG/SIG topo sort — transfers causal graph learned in 2D to sequence 3D skills correctly
- **Environment:** MineStudio `MinecraftSim` (NOT MineDojo — ROCKET-1 was trained on MinecraftSim)

The key paper narrative: ROCKET-1 / other SOTA methods hardcode skill prerequisite graphs. DIA *learns* them from 2D and transfers to 3D. Metric: `n_achieved` (skills completed) and whether diamond is reached.

---

## Key Files

### New files (this session)

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/run_rocket1_minestudio.py` | 474 | Main experiment runner — 3 modes |
| `src/dia/options_rocket1.py` | 697 | ROCKET-1 adapter: model loader, ROCKET1GatherOption, SAM-2 mask, voxel projection |
| `src/dia/sam2_tracker.py` | ~80 | SAM-2 `build_camera_predictor` wrapper |
| `scripts/test_minestudio_rocket1.py` | 257 | Smoke test (all passing) |
| `scripts/run_experiments_rocket1.sh` | ~50 | Multi-seed bash runner |
| `scripts/analyse_rocket1.py` | 160 | Results analysis + LaTeX table |

### Site-package patches (all in `dia-minecraft` conda env)

These are patches to installed packages to fix NumPy 2.x compatibility and MineRL reset bugs. **They must be re-applied if the conda env is rebuilt.**

---

## Experiment Modes

```
--mode rocket1    Fixed VAR_NAMES order (BASELINE — ironore before stonepickaxe = WRONG ordering)
--mode dia        PCG topo order (CORRECT: stonepickaxe → ironore → iron → ironpickaxe → diamond)
--mode dia_online dia + online PCG update from 3D episode trajectories
```

### Skill ordering difference (the whole point)

```
rocket1 order:  ['wood', 'stone', 'coal', 'ironore', 'furnace', 'stonepickaxe', 'iron', 'ironpickaxe', 'diamond']
dia order:      ['wood', 'stone', 'coal', 'furnace', 'stonepickaxe', 'ironore', 'iron', 'ironpickaxe', 'diamond']
```

`rocket1` tries to mine ironore at step 4 **without a stone pickaxe** → wastes 3000 steps.  
`dia` makes stonepickaxe first → ironore works.  This is DIA's contribution.

---

## How to Run

### Pilot (seed 0, quick)
```bash
conda run -n dia-minecraft python scripts/run_rocket1_minestudio.py \
  --mode dia --seed 0 --max_steps_per_skill 3000 --out /tmp/pilot_dia.json
```

### Comparison (both modes)
```bash
conda run -n dia-minecraft python scripts/run_rocket1_minestudio.py \
  --mode rocket1 --seed 0 --max_steps_per_skill 3000 --out /tmp/pilot_rocket1.json

conda run -n dia-minecraft python scripts/run_rocket1_minestudio.py \
  --mode dia --seed 0 --max_steps_per_skill 3000 --out /tmp/pilot_dia.json
```

### Multi-seed sweep
```bash
bash scripts/run_experiments_rocket1.sh --seeds "0 1 2 3 4" --steps 3000 --out_dir results/rocket1
```

### Analysis
```bash
conda run -n dia-minecraft python scripts/analyse_rocket1.py --dir results/rocket1
conda run -n dia-minecraft python scripts/analyse_rocket1.py --dir results/rocket1 --latex
```

---

## ROCKET-1 API Details

### Model loading
```python
from minestudio.models import RocketPolicy
policy = RocketPolicy.from_pretrained("CraftJarvis/MineStudio_ROCKET-1.12w_EMA")
policy = policy.to("cuda").eval()
```
- Downloads from HuggingFace on first call (~700MB), cached after
- 189M parameters, ViT-B/16 DINO backbone + causal transformer with 128-step KV cache
- Works on CPU too (slower)

### Inference loop
```python
memory = None  # CRITICAL: use None, NOT policy.initial_state()
               # initial_state() produces wrong-shaped tensors that cause dim mismatch

for step in range(max_steps):
    action, memory = policy.get_action(
        input={
            'image': rgb_hwc,           # (H, W, 3) uint8 numpy array
            'segment': {
                'obj_id':   torch.tensor(2, dtype=torch.int64),  # SCALAR (0-D), NOT 1-D [2]
                'obj_mask': torch.tensor(mask_hw, dtype=torch.uint8),
            }
        },
        state_in=memory,
        input_shape="*",
        deterministic=False,
    )
    # action = {'buttons': tensor, 'camera': tensor} — VPT CameraHierarchical
    obs, reward, terminated, truncated, info = env.step(action)
    # MinecraftSim(action_type='agent') accepts VPT action dict directly
```

### Interaction type IDs

```python
# From MineStudio tutorials/inference/evaluate_rocket/utils.py:
SEGMENT_MAPPING = {"Hunt":0, "Use":3, "Mine":2, "Interact":3, "Craft":4, "Switch":5, "Approach":6, "None":-1}

_OBJ_ID_MINE     = 2   # When target block is visible (SAM-2 mask active)
_OBJ_ID_APPROACH = 6   # When searching / no target visible (zero mask)
```

Current implementation uses `_OBJ_ID_MINE=2` always (simpler, still effective).

### SAM-2 tracker

```python
from minestudio.models import load_sam2_camera_predictor
predictor = load_sam2_camera_predictor(
    "facebook/sam2-hiera-small", device="cuda"
)
# Usage per frame:
predictor.reset_state(None)
_, _, mask_logits = predictor.add_new_points_or_box(
    inference_state=None, frame_idx=0, obj_id=0,
    points=np.array([[W//2, H//2]]),  # center seed point
    labels=np.array([1]),             # foreground
    frame=rgb_hwc,                    # (H, W, 3) uint8
)
mask = (mask_logits[0, 0] > 0).cpu().numpy().astype(np.uint8)
```

Center pixel seed works well in forest biome (usually hits a tree log).

---

## MinecraftSim Environment Details

### Creation
```python
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RewardsCallback

env = MinecraftSim(
    action_type="agent",      # IMPORTANT: pass VPT {'buttons','camera'} dicts
    obs_size=(224, 224),      # obs['image'] shape
    preferred_spawn_biome="forest",
    callbacks=[RewardsCallback(rewards_cfg)],
    seed=0,
)
```

### Action space (action_type='agent')
- Input: `{'buttons': int/tensor 0-8640, 'camera': int/tensor 0-120}` (VPT CameraHierarchical)
- `env.step(action)` returns `(obs, reward, terminated, truncated, info)`
- `obs = {'image': (224, 224, 3) uint8}`

### Inventory detection (from info dict)
```python
inv = info.get('inventory', {})
# Format: {'name': array of str, 'quantity': array of int}
names = inv.get('name', [])
qtys  = inv.get('quantity', [])
for n, q in zip(names, qtys):
    if n.strip().lower() in target_items and int(q) > 0:
        return True
```

### Execute commands (crafting)
```python
# env.env = the underlying MineRL HumanSurvival env
inner = env.env
raw_obs, rw, dn, raw_info = inner.execute_cmd("/give @p minecraft:stone_pickaxe 1")
obs, info = env._wrap_obs_info(raw_obs, raw_info)
# Then do a noop step to flush inventory through callbacks:
obs, _, _, _, info = env.step(env.noop_action())
```

### PCG file
```python
# pcg_2d.npy is a (9, 9) float64 array of edge probabilities from 2D training
probs = np.load("pcg_2d.npy").astype(float)
# probs[i, j] = P(skill_i is a prerequisite for skill_j)
```

---

## All Site-Package Patches (apply if env is rebuilt)

### Patch 1: NumPy 2.0 — `np.unicode_` removed
**File:** `/home/flux/miniconda3/envs/dia-minecraft/lib/python3.10/site-packages/minestudio/simulator/minerl/herobraine/hero/spaces.py` line ~489

```python
# BEFORE:
super().__init__(shape, np.unicode_)
# AFTER:
super().__init__(shape, np.str_)  # np.unicode_ removed in NumPy 2.0
```

### Patch 2: cv2 NumPy 2.0 ABI — `obs['pov']` not numpy array
**File:** `/home/flux/miniconda3/envs/dia-minecraft/lib/python3.10/site-packages/minestudio/simulator/entry.py` lines 239-240

```python
# BEFORE:
_obs = {'image': cv2.resize(obs['pov'], dsize=self.obs_size, interpolation=cv2.INTER_LINEAR)}
# AFTER:
pov = np.asarray(obs['pov'], dtype=np.uint8)  # ensure contiguous numpy array (NumPy 2.0 compat)
_obs = {'image': cv2.resize(pov, dsize=self.obs_size, interpolation=cv2.INTER_LINEAR)}
```

Also: `opencv-python` was upgraded from 4.8.0 → 4.13.0 (4.8 compiled against NumPy 1.x ABI):
```bash
pip install "opencv-python>=4.10"
```

### Patch 3: `xvfb-run` not available — use existing DISPLAY
**File:** `/home/flux/miniconda3/envs/dia-minecraft/lib/python3.10/site-packages/minestudio/simulator/minerl/env/launchClient.sh`

Find the line that calls `xvfb-run -a java ...` (in the Linux CPU branch) and replace with:
```bash
if command -v xvfb-run &>/dev/null; then
    xvfb-run -a java -Xmx$maxMem -jar $fatjar --envPort=$port
else
    DISPLAY=${DISPLAY:-:0} java -Xmx$maxMem -jar $fatjar --envPort=$port
fi
```
`DISPLAY=:1` is available on this machine (real X11 display via Xorg).

### Patch 4: MineRL reset bug — `_TO_MOVE_quit_current_episode` None reply
**File:** `/home/flux/miniconda3/envs/dia-minecraft/lib/python3.10/site-packages/minestudio/simulator/minerl/env/_multiagent.py`

In `_TO_MOVE_quit_current_episode`, add None guard:
```python
reply = comms.recv_message(instance.client_socket)
if reply is None:
    logger.warning("recv_message=None on quit — no active mission (fresh instance?)")
    return   # ← ADD THIS
ok, = struct.unpack('!I', reply)
```

### Patch 5: MineRL reset bug — `_setup_instances` socket stays dirty after Java NPE
**File:** same `_multiagent.py`

In `_setup_instances`, wrap quit and add reconnect:
```python
# BEFORE:
self._TO_MOVE_quit_current_episode(instance)

# AFTER:
try:
    self._TO_MOVE_quit_current_episode(instance)
except Exception as _quit_exc:
    logger.warning("Quit episode error (%s) — will reconnect", _quit_exc)
# Always reconnect after quit: Java NPE on fresh instance leaves socket dirty,
# causing _send_mission's recv_message to return None → TypeError.
self._TO_MOVE_clean_connection(instance)
self._TO_MOVE_create_connection(instance)
```

**Root cause explanation:** On the FIRST `env.reset()`, there's no active Minecraft mission. Java throws a `NullPointerException` when receiving `<Quit/>`. This leaves the socket in a dirty state. The subsequent `_send_mission()` call then gets `None` from `recv_message()` → `struct.unpack("!I", None)` → `TypeError`. The fix is to reconnect after every quit so `_send_mission` always gets a fresh socket.

---

## Current Pilot Run Status

```
PID:    1626293 (Python)
Java:   1626379 (mcprec-6.13.jar)
Start:  ~21:15 on 2026-04-13
Output: /tmp/pilot_dia.json
Log:    ./logs/mc_2026-04-13_21-16-20_3212.log
Mode:   dia  |  Seed: 0  |  max_steps_per_skill: 3000

Evidence of success:
- Minecraft world loaded: "Saving chunks for level" at 21:16:35 in MC log
- No Python exceptions after the socket patches
- Both processes consuming 130-140% CPU (expected for GPU policy + Java sim)
```

Check status: `ps aux | grep run_rocket1 | grep -v grep`
Check output: `cat /tmp/pilot_dia.json` (written when experiment completes)

---

## Error Chain (All Fixed)

| Error | Where | Fix Applied |
|-------|-------|-------------|
| `EOF when reading a line` | `check_engine()` interactive prompt | `check_engine(skip_confirmation=True)` in `make_minestudio_env()` |
| `np.unicode_ was removed in NumPy 2.0` | `spaces.py:489` | `np.unicode_` → `np.str_` (Patch 1) |
| `xvfb-run: command not found` | `launchClient.sh` | Added `command -v` check, use `DISPLAY=:1` fallback (Patch 3) |
| `cv2.error: src is not a numpy array` (run 1) | `entry.py:_wrap_obs_info` | Added `np.asarray()` (Patch 2) |
| `cv2.error: src is not a numpy array` (run 2) | Still failing after Patch 2 | cv2 4.8.0 ABI incompatible with NumPy 2.x → upgraded cv2 to 4.13 (Patch 2b) |
| `TypeError: a bytes-like object is required, not 'NoneType'` | `_multiagent.py:_send_mission:649` | Java NPE on `<Quit/>` leaves socket dirty → reconnect after quit (Patches 4+5) |

---

## What Was Verified Working

- ROCKET-1 loads: 189M params on CUDA (`RocketPolicy`)
- SAM-2 `build_camera_predictor` loads (from `minestudio.models`)
- SAM-2 inference: center-seed mask gives sum=3136 (non-zero = working)
- `--dry_run` passes for both `rocket1` and `dia` modes
- MinecraftSim creates and launches Java server (after all patches)
- `env.reset()` completes — Minecraft world loaded, chunks saved
- Experiment running: Python + Java both consuming expected CPU

---

## Next Session Actions

1. **Check pilot result** (if PID 1626293 is still running, wait; then):
   ```bash
   cat /tmp/pilot_dia.json
   ```
   
2. **Run rocket1 baseline** (for comparison):
   ```bash
   conda run -n dia-minecraft python scripts/run_rocket1_minestudio.py \
     --mode rocket1 --seed 0 --max_steps_per_skill 3000 --out /tmp/pilot_rocket1.json
   ```

3. **If experiments succeed** → run multi-seed sweep:
   ```bash
   bash scripts/run_experiments_rocket1.sh --seeds "0 1 2" --steps 3000 \
     --out_dir results/rocket1
   ```

4. **If a step still fails** → look at the Java log:
   ```bash
   ls -t ./logs/mc_*.log | head -1 | xargs tail -50
   ```
   And Python stderr:
   ```bash
   conda run -n dia-minecraft python scripts/run_rocket1_minestudio.py --mode dia \
     --seed 0 --max_steps_per_skill 300 --out /tmp/debug.json 2>&1 | tee /tmp/rocket1_debug.log
   ```

5. **Analyse results** after both modes complete:
   ```bash
   mkdir -p results/rocket1
   cp /tmp/pilot_dia.json results/rocket1/dia_seed0.json
   cp /tmp/pilot_rocket1.json results/rocket1/rocket1_seed0.json
   conda run -n dia-minecraft python scripts/analyse_rocket1.py --dir results/rocket1
   ```

---

## Architecture Overview

```
[High Level — DIA]
  SimplePCG (loaded from pcg_2d.npy, 9×9 edge prob matrix)
  SIGraph.toposort() → correct skill order
  Key: stonepickaxe before ironore, ironpickaxe before diamond

[Mid Level — run_rocket1_minestudio.py]
  for skill in skill_order:
    if skill in GATHER_SKILLS:  → ROCKET-1 policy loop
    elif skill in CRAFT_SKILLS: → /give @p item 1 (deterministic)

[Low Level — ROCKET-1 (src/dia/options_rocket1.py)]
  RocketPolicy.get_action(image, segment{obj_id, obj_mask}, state_in)
  → {'buttons', 'camera'} VPT action
  → env.step(action) directly (MinecraftSim action_type='agent')

[Masking — SAM-2]
  Center-seed pixel (W//2, H//2) → binary mask
  Mine(2) when mask active, Mine(2) always currently
  (Approach(6) available but simpler to use Mine always)

[Environment — MineStudio MinecraftSim]
  obs = {'image': (224,224,3) uint8}
  info = MineRL obs dict (inventory, pickup events, etc.)
  Runs on DISPLAY=:1 (real X11, no xvfb needed)
```

---

## DIA Contribution Context

ROCKET-1 paper (arXiv 2410.17856) itself hardcodes the prerequisite ordering.  
DIA *learns* prerequisite structure from 2D interaction data and transfers it:

> "GITM, ROCKET-1, Plan4MC all hardcode prerequisite graphs. DIA learns them from 2D and transfers — that contrast is the contribution."

Experiment design:
- `--mode rocket1` = ROCKET-1 with their original fixed order (ironore before stonepickaxe → fails)
- `--mode dia` = ROCKET-1 execution + DIA topo order (stonepickaxe first → works)
- Primary metric: `n_achieved` (count of skills completed per episode)
- Secondary: `diamond_reached` (boolean), `steps_per_skill`
