# DIA+GROOT Experiment Changelog

Tracks code changes and results for each run iteration.
All runs: `mode=dia`, `seed=0` (and `seed=1` where noted), MineStudio env, GrootPolicy pretrained.

Skill order: `wood → woodpickaxe → stone → coal → furnace → stonepickaxe → ironore → iron → ironpickaxe → diamond`
Craft skills (woodpickaxe, furnace, stonepickaxe, iron, ironpickaxe) use GROOT first, `/give` fallback if GROOT fails.

---

## it2 — Baseline with PCG ordering

**Results (seed 0):**
| Skill | Success | Steps |
|-------|---------|-------|
| wood | ✅ | 754 |
| stone | ✅ | 999 |
| coal | ❌ | 2500 |
| ironore | ❌ | 2500 |
| diamond | ❌ | 2500 |
| n_achieved | **7/10** | total=31,763 |

**Notes:** `max_steps_per_skill=2500`. Underground skills (coal, ironore) had no navigation primer — GROOT ran blind from surface spawn with no descent phase. Consistently hit budget.

---

## it3 — Increased step budget

**Changes from it2:**
- `max_steps_per_skill`: 2500 → 3000

**Results (seed 0):**
| Skill | Success | Steps |
|-------|---------|-------|
| wood | ✅ | 149 |
| stone | ✅ | 344 |
| coal | ❌ | 3000 |
| ironore | ❌ | 3000 |
| diamond | ❌ | 3000 |
| n_achieved | **7/10** | total=36,503 |

**Notes:** Larger budget confirmed coal/ironore were not being found — not a budget issue, a navigation issue. Underground skills still had no descent primer.

---

## it4 — Underground staircase primer introduced ⭐ first ironore

**Changes from it3:**
- Added `_UNDERGROUND_SWEEP`: 200-step primer for coal/ironore/diamond skills
  - 3 steps pitch camera down to 45°, GROOT controls buttons
  - 150 steps forced attack+forward (diagonal staircase dig)
  - ~47 steps camera sweep (wall scan) — GROOT controls attack during scan
- Added `_SURFACE_SWEEP`: 38-step horizontal yaw sweep for wood/stone
- `_run_primer()` added to `GrootExecutor`
- `min_qty` parameter added to `run_skill()` for quantity-based success gating
- Online PCG update via `_update_pcg_from_inventory()` (inventory snapshot before/after each skill)
- `_apply_experiment_setup()`: hunger disabled (saturation), time locked to day
- `_show_minecraft_window()` with persistent keeper thread (30s re-raise)

**Results:**
| Seed | n_achieved | wood | coal | ironore | notes |
|------|-----------|------|------|---------|-------|
| 0 | 7 | 166 steps | ❌ 3000 | ❌ 3000 | stone as byproduct |
| 1 | **8** | 436 steps | ❌ 3000 | ✅ **0** (byproduct!) | ironore found during staircase descent |
| 2 | 7 | — | ❌ | ❌ | |
| 3 | 7 | — | ❌ | ❌ | |

**Key finding:** Seed 1 terrain has an iron vein that intersects the staircase path. Ironore was picked up as a byproduct at step 0 (already in inventory when ironore skill was scheduled). The 150-step forced staircase is reaching ore-bearing layers.

---

## it5 — Resource quantity gating (regression on wood)

**Changes from it4:**
- `_GATHER_MIN_QTY["wood"]`: 1 → **5** (agent farms until it holds 5 logs)
- `_GATHER_MIN_QTY["stone"]`: 1 → **11** (furnace=8 + stonepickaxe=3)
- `_GATHER_MIN_QTY["ironore"]`: 1 → **3** (iron_pickaxe needs 3 ingots)

**Results:**
| Seed | n_achieved | wood | coal | ironore | notes |
|------|-----------|------|------|---------|-------|
| 0 | 7 | ❌ **2883 steps** | ❌ 3000 | ❌ 3000 | Wood min_qty=5 caused severe over-farming |
| 1 | 7 | 0 (spawn) | ❌ 3000 | ❌ 3000 | Ironore regression vs it4 s1 |

**Regression:** `min_qty=5` for wood was too aggressive. Seed 0 spent 2883/3000 steps farming wood alone. The scan phase during underground skills still had GROOT controlling attack, meaning GROOT was not reliably breaking ore blocks during the wall sweep.

**User note:** Agent behavior quality was good (effective tree-chopping visible in game), but resource farming was over-budget.

---

## it6 — Wood over-farming fix + forced-attack scan phase

**Changes from it5:**
- `_GATHER_MIN_QTY["wood"]`: 5 → **3** (3 logs → 12 planks, sufficient for all pickaxes)
- `_UNDERGROUND_SWEEP` scan phase: added `force_attack=1` during ALL wall-sweep steps
  - Previously: 8+8+4+22 = 42 scan steps had GROOT controlling attack (GROOT was not attacking ore)
  - Now: all 42 scan steps force attack=1, agent mines whatever the camera points at
- Restored 150 **uninterrupted** staircase dig steps (a prior experimental variant had added collection pauses, which reduced descent depth)

**Results:**
| Seed | n_achieved | wood | coal | ironore | notes |
|------|-----------|------|------|---------|-------|
| 0 | 7 | 0 (spawn) | ❌ 3000 | ❌ 3000 | stone took 2800 steps |
| 1 | **8** | 0 (spawn) | ✅ **0** (spawn) | ❌ 3000 | Both seeds spawned with items from environment |

**Note:** Both seeds had a "cavern/biome spawn" — wood, stone, coal were already in inventory from the environment. This confounds the wood fix (can't measure) and inflated seed 1's score. Ironore remains unsolved.

---

## it7 — Survival effects, torches, video recording (no logic changes)

**Changes from it6:**
- `_apply_experiment_setup()`: added 3 new commands:
  - `/effect give @p minecraft:water_breathing 1000000 255 true` — prevents drowning in underground streams
  - `/effect give @p minecraft:fire_resistance 1000000 255 true` — prevents burning in lava pockets
  - `/give @p minecraft:torch 64` — agent carries torches (for illumination in deep caves)
- Video recording: `--video_dir` CLI flag + `_VideoRecordingEnv` wrapper + `_save_video()` (cv2 MP4 output)
- **No changes to primer logic** — `_UNDERGROUND_SWEEP` and `_run_primer()` identical to it6

**Attempted but reverted:**
- Jump suppression in underground primer (`is_underground → jump=0`) — caused lake-dive in tested run
- `force_fwd=-1` for pitch-down steps — reverted with jump suppression

**Results:**
| Seed | n_achieved | wood | stone | coal | ironore | notes |
|------|-----------|------|-------|------|---------|-------|
| 0 | 6 | ✅ 0 (spawn) | ❌ 3000 | ❌ 3000 | ❌ 3000 | Stone primer failed; terrain variance |
| 1 | 4 | ❌ 3000 | ❌ 3000 | ❌ 3000 | ❌ 3000 | Wood GROOT failed; ironpickaxe /give broken |

**Bug found — `_VideoRecordingEnv` wrapper broke `/give` fallback:**
`_give_item()` uses `getattr(env, "env", None)` to reach the inner `execute_cmd` env.
When `env` is wrapped as `_VideoRecordingEnv`, `env.env` routes through `__getattr__`
and resolves to `None` (MineStudio env has no `env.env` chain accessible this way).
Result: all craft `/give` fallbacks silently returned `False` on this run.
Fix: unwrap one level with `getattr(env, "_env", env)` before looking up `.env`.

---

## it8 — Revert to it5 logic + hotbar equip fix + playthrough demo recorder

**Changes from it7:**
- Reverted to **it5 scan-phase logic**: `_UNDERGROUND_SWEEP` scan phase uses `force_attack=0` (GROOT controls attack), restoring visual goal-following behaviour that was lost in it6
- `_GATHER_MIN_QTY["wood"]`: reverted to 5 (it5 value); it6/it7 used 3 and correlated with worse behaviour
- `_GATHER_MIN_QTY["stone"]`: 11 (unchanged)
- Added `_equip_pickaxe()`: uses `/item replace entity @p hotbar.0 <pickaxe> 1` to put the correct pickaxe in hotbar slot 0 before each underground gather skill
  - stone → wooden_pickaxe, coal/ironore → stone_pickaxe, diamond → iron_pickaxe
  - Fixes bare-hands ironore mining (zero drops) observed in it7: agent had stone_pickaxe in inventory but not equipped
- Added `scripts/record_playthrough_demos.py`: full tech-tree sequential demo recorder
  - Redesigned as diamond-gated free-play (one world per seed, records continuously until diamond in inventory)
  - F5 to annotate segments, F12 to quit, diamond auto-advances to next seed
  - Saves MP4 + BC `.npz` + annotation JSON per seed
- Retained from it7: water_breathing, fire_resistance, torch give, `/give` wrapper fix

**Root cause identified (coal never found):**
User observed during it7 run: agent walked past multiple coal blocks in a cavern without attacking them.
GROOT's visual goal-following triggers attack when the reference clip matches — but clips show mid-mining state
(already adjacent to ore face). Agent hasn't learned to approach and initiate attack from a distance.
Primary fix path: collect full-playthrough demos via `record_playthrough_demos.py` where the human
demonstrates approach + initiation for each skill.

**Results:**
| Seed | n_achieved | notes |
|------|-----------|-------|
| TBD | TBD | — |

---

## recorder — PlayCallback integration + Malmo timing fixes

**Changes to `scripts/record_playthrough_demos.py`:**

### PlayCallback integration (exclusive mouse capture)
- Replaced pynput global listener with MineStudio's `PlayCallback` for all input capture
  - PlayCallback creates a dedicated pyglet window with exclusive mouse lock
  - 'C' key captures/releases mouse; Ctrl+C or F12 quits
  - No more desktop bleed-through: mouse/keyboard only affect the game window
- `_RecordingPlayCallback` wraps PlayCallback as a **standalone GUI controller** (NOT a MineStudio callback)
  - Key insight: registering as a callback interfered with `env.reset()` internals
  - Instead: `get_action()` dispatches pyglet events + returns human action directly; `update_display()` pushes frames
  - F5/F12 detected via `_capture_all_keys()` inside `get_action()` (called once per game tick)
- Patched `minestudio/simulator/utils/gui.py` line 340: `layout.update(x, y)` → `layout.x = ...; layout.y = ...`
  - pyglet 2.x removed `TextLayout.update()` — replaced with direct attribute assignment

### Malmo connection timing fix
- `env.reset()` retry delay: **5s → 25s**
  - Root cause: after a Malmo restart, the JVM reaches "process ready" after ~15s but the Malmo socket needs ~10s more before accepting connections. 5s retries were consistently firing into that 10s window, each triggering another Minecraft restart. 25s clears the full window.
- `play_cb.show_resetting()` shown before each `env.reset()` attempt (user feedback)
- `play_cb.update_display(info)` called after reset + setup commands to show the initial frame

### Architecture: `action_type="env"` (unchanged from it8)
- Produces individual-key action dict, compatible with PlayCallback's `_get_human_action()` output

---

## Known Issues / Open Problems

| Problem | Status | Evidence |
|---------|--------|----------|
| Coal never found by primer | ❌ Unsolved | 0/8 attempts across it2–it7 |
| Ironore only found as terrain byproduct | ⚠️ Terrain-dependent | it4 s1, it6 s1 (spawn) |
| Underground primer walks agent into water | ⚠️ Terrain-dependent | GROOT outputs jump+forward during pitch-down on some spawns |
| Wood min_qty=3 not yet measured on normal spawn | ⏳ Pending | it6 seeds had spawn wood; it7 s1 wood failed entirely |
| _VideoRecordingEnv breaks /give fallback | ✅ Fixed (it7.1) | `_give_item` now unwraps `_env` before seeking `execute_cmd` |
| Iron ore mined with bare hands (zero drops) | ✅ Fixed (it8) | `_equip_pickaxe()` puts correct pickaxe in hotbar.0 before each underground gather |
| Diamond never found | ❌ Unsolved | Requires ironore → iron → ironpickaxe first |
| recorder: env.reset() loops on ConnectionRefusedError | ✅ Fixed (recorder) | Retry delay 5s→25s; PlayCallback decoupled from MinecraftSim callback chain |
| recorder: pynput bleeds input to desktop | ✅ Fixed (recorder) | Replaced with PlayCallback pyglet window (exclusive mouse) |
| gui.py TextLayout.update() crash (pyglet 2.x) | ✅ Fixed (site-pkg patch) | `layout.update(x,y)` → `layout.x=; layout.y=` |

---

## Performance Summary

| Iteration | Best n_achieved | Coal found? | Ironore found? | Key change |
|-----------|----------------|-------------|----------------|-----------|
| it2 | 7 | ❌ | ❌ | Baseline |
| it3 | 7 | ❌ | ❌ | +step budget |
| it4 | **8** (s1) | ❌ | ✅ byproduct | +staircase primer |
| it5 | 7 | ❌ | ❌ | min_qty=5 (wood over-farms but agent behaviour quality high) |
| it6 | **8** (s1) | ✅ spawn | ❌ | min_qty=3, force-attack scan — agent visibly dumber |
| it7 | 6/4 | ❌ | ❌ | +video/torches/effects; /give bug from wrapper |
| ft† | **8** (s0) | ❌ | ✅ spawn | Fine-tuned GROOT ckpt |
| **it8** | TBD | TBD | TBD | **Reverted to it5 logic** + it7 additions |

†`ft`: separate run with fine-tuned GROOT checkpoint, seed 0.

**Best legitimate performance:** it4 seed 1 (ironore found by staircase, not spawn luck).  
**Best agent behaviour quality:** it4/it5 — GROOT's visual goal-following during scan sweep produced purposeful ore-seeking. Force-attack override in it6 removed this and made the agent dumber.  
**Active codebase (it8):** it5 logic (wood min_qty=5, GROOT controls scan attack) + torches + water/fire effects + video recording + /give wrapper fix + hotbar pickaxe equip before underground gather.
