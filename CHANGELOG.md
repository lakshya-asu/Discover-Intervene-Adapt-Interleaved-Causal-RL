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

**Results (seed 0, partial — run aborted):**
- Agent spawned above ground near a lake, jumped in during GROOT-controlled pitch-down steps, then dug down through water. Run killed before completion. Water-breathing prevented death but pathing was wrong.

---

## Known Issues / Open Problems

| Problem | Status | Evidence |
|---------|--------|----------|
| Coal never found by primer | ❌ Unsolved | 0/8 attempts across it2–it7 |
| Ironore only found as terrain byproduct | ⚠️ Terrain-dependent | it4 s1, it6 s1 (spawn) |
| Underground primer walks agent into water | ⚠️ New (it7) | GROOT outputs jump+forward during pitch-down |
| Wood min_qty=3 not yet measured on normal spawn | ⏳ Pending | it6 seeds had spawn wood |
| Diamond never found | ❌ Unsolved | Requires ironore → iron → ironpickaxe first |

---

## Performance Summary

| Iteration | Best n_achieved | Coal found? | Ironore found? | Key change |
|-----------|----------------|-------------|----------------|-----------|
| it2 | 7 | ❌ | ❌ | Baseline |
| it3 | 7 | ❌ | ❌ | +step budget |
| it4 | **8** (s1) | ❌ | ✅ byproduct | +staircase primer |
| it5 | 7 | ❌ | ❌ | min_qty regression |
| it6 | **8** (s1) | ✅ spawn | ❌ | +force-attack scan |
| ft† | **8** (s0) | ❌ | ✅ spawn | Fine-tuned GROOT ckpt |

†`ft`: separate run with fine-tuned GROOT checkpoint, seed 0.

**Best legitimate performance:** it4 seed 1 (ironore found by staircase, not spawn luck).  
**Closest to SOTA path:** Getting ironore reliably requires either consistent staircase geometry hitting ore veins, or active cave detection.
