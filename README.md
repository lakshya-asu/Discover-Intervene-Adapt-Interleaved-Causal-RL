# Discover, Intervene, Adapt (DIA)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey?logo=linux)](https://ubuntu.com)
[![Status](https://img.shields.io/badge/Status-Active%20Research-blue)]()

**Interpretable and adaptive causal RL via interleaved discovery and structured interventions.**

DIA agents learn a probabilistic causal graph (PCG) from skill interventions, organise skills into a prerequisite-linked Skill-Intervention Graph (SIG), and plan goal-directed sequences using causal structure rather than reward correlation. The key result: structure learned in a fast 2D symbolic environment transfers directly to guide skill ordering in 3D Minecraft, without any additional 3D training.

The current active system pairs **DIA's causal planner** with **GROOT** (a goal-conditioned imitation learning policy) as the low-level executor — achieving up to 8/10 tech-tree skills in a single episode without any task-specific reward shaping.

---

## Architecture: DIA + GROOT Hybrid

```
┌─────────────────────────────────────────────┐
│              DIA Causal Planner              │
│                                             │
│  PCG  →  SIG topo-sort  →  skill_order      │
│   ↑                              │          │
│   └── online update (inventory) ─┘          │
└──────────────────┬──────────────────────────┘
                   │ next skill + reference clip
┌──────────────────▼──────────────────────────┐
│            GrootExecutor                    │
│                                             │
│  Navigation primer  →  GROOT goal-following │
│  (scripted camera+force)   (video-conditioned│
│                             IL policy)      │
└──────────────────┬──────────────────────────┘
                   │ actions
          MineStudio / Minecraft
```

**DIA layer** — `src/dia/pcg.py`, `src/dia/sig.py`
- Maintains a probabilistic causal graph (PCG) over 9 Minecraft skill variables
- Builds a Skill-Intervention Graph (SIG) encoding the tech-tree prerequisite structure
- Topological sort gives an adaptive skill execution order
- Online PCG update after every skill attempt (inventory-state causal learning)

**GROOT layer** — `src/dia/groot_executor.py`, `CraftJarvis/MineStudio_GROOT.18w_EMA`
- Goal-conditioned imitation learning policy (237M params)
- Conditioned on a reference video clip per skill (no per-task reward needed)
- Navigation primer: scripted camera overrides + forced dig to reach underground ore depth
- `/give` fallback for craft skills (woodpickaxe, furnace, etc.) if GROOT fails to craft

---

## Performance (it6, seed 1 — best result)

| Skill | Achieved | Method |
|-------|----------|--------|
| wood | ✅ | GROOT (surface sweep) |
| woodpickaxe | ✅ | GROOT craft / /give fallback |
| stone | ✅ | Byproduct of underground primer |
| coal | ✅ | Found in spawn environment |
| furnace | ✅ | GROOT craft / /give fallback |
| stonepickaxe | ✅ | GROOT craft / /give fallback |
| ironore | ❌ | Primer reaches depth, seam not found |
| iron | ✅ | /give fallback |
| ironpickaxe | ✅ | GROOT craft / /give fallback |
| diamond | ❌ | Requires ironpickaxe to mine |
| **Total** | **8 / 10** | |

See [CHANGELOG.md](CHANGELOG.md) for full per-iteration results across it2–it7.

---

## Setup

**Requirements:** Linux, Miniconda, Java 11, CUDA GPU (for GROOT inference).

```bash
# 2D environments (CPU only)
conda env create -f environment.yml
conda activate dia
pip install -e .

# 3D MineStudio experiments (GPU required)
conda env create -f environment-gpu.yml
conda activate dia-minecraft
pip install minestudio
pip install -e .
```

MineDojo requires a one-line NumPy 2.0 compatibility patch in `minedojo/sim/spaces.py`
(replace `np.unicode_` with `np.str_`). See `scripts/setup_vpt_data.sh`.

---

## Quick Start

### 2D Causal Prior Training (CPU, ~5 min)

```bash
python scripts/train_minecraft2d_dia.py \
    --pcg simple --steps 5000 --logdir runs/minecraft2d
```

Saves `pcg_2d.npy` and `sig_2d.json` — the causal prior used for 3D transfer.

### DIA + GROOT Hybrid (MineStudio, GPU)

```bash
conda run -n dia-minecraft python scripts/run_groot_dia_minestudio.py \
    --mode dia \
    --seed 0 \
    --max_steps_per_skill 3000 \
    --max_total_steps 45000 \
    --out results/dia_groot_s0.json \
    --video_dir results/videos/
```

**Modes:**
- `groot` — fixed VAR_NAMES skill order, pure GROOT execution (baseline)
- `dia` — DIA PCG/SIG topological order + GROOT execution
- `dia_online` — `dia` + online PCG update from trajectories

**Key flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | 0 | Env + PCG seed |
| `--max_steps_per_skill` | 3000 | Budget per skill attempt |
| `--max_total_steps` | 45000 | Episode step cap |
| `--primer_steps` | 80 | Surface sweep length (wood/stone) |
| `--video_dir` | None | Save per-run MP4 (disabled if unset) |
| `--no_ft_ckpt` | False | Use pretrained GROOT (skip fine-tuned ckpt) |

### Multi-seed Parallel Run

```bash
for SEED in 0 1 2 3; do
  DISPLAY=:1 conda run -n dia-minecraft python scripts/run_groot_dia_minestudio.py \
    --mode dia --seed $SEED \
    --out /tmp/dia_groot_s${SEED}.json \
    --video_dir /tmp/dia_videos/ &
done
wait
```

### Crafting Demonstration Recorder

Record human crafting demonstrations for GROOT BC fine-tuning:

```bash
conda run -n dia-minecraft python scripts/record_crafting_demos.py \
    --skill woodpickaxe \
    --seed 0 \
    --out_dir data/crafting_demos
# Controls: F10=start, F11=stop/save, F12=quit
```

Saves reference MP4 clips and compressed BC arrays (obs + action dicts).

---

## Repository Layout

```
src/dia/
    evgs*.py              # EVGS adapters: symbolic → binary variables
    pcg*.py               # PCG backends: Simple, Differentiable, Variational, Granger
    sig.py                # SIG + topo-sort planner
    groot_executor.py     # GrootExecutor: primer + GROOT skill runner
    options*.py           # Option policies (MineDojo, MineRL, ROCKET-1)
    rollout.py            # DIARunner: interleaved training loop
    planner.py            # Three-phase intervention selector

scripts/
    run_groot_dia_minestudio.py    # ★ Main DIA+GROOT experiment (active)
    record_crafting_demos.py       # Human demo recorder for crafting BC
    run_rocket1_minestudio.py      # ROCKET-1 backbone variant
    train_minecraft2d_dia.py       # 2D causal prior training
    pretrain_bc_minerl.py          # BC policy pre-training from MineRL demos
    run_transfer_minedojo.py       # MineDojo transfer experiment (legacy)
    finetune_groot.py              # GROOT fine-tuning on collected BC data

skill_clips/
    manifest.json                  # Skill → reference clip paths
    wood/, stone/, coal/, ...      # Per-skill .mp4 reference clips

data/
    minerl_bc/                     # BC demo arrays (obs.npy + actions.npy)
    groot_ft/                      # Fine-tuned GROOT checkpoint (if available)

CHANGELOG.md                       # Per-iteration results and code changes
```

---

## How It Works

### EVGS (Environment Variable Goal Space)
Maps raw observations to a fixed set of interpretable binary variables
(`has_wood`, `has_stone`, `has_coal`, ...). Options target predicates over these variables.

### PCG (Probabilistic Causal Graph)
Maintains a posterior over directed edges between skill variables.
Each skill execution is a causal intervention; the PCG updates from the observed
inventory-state delta. Edge entropy drives an information-gain exploration bonus.
Online update rule: if items {i} are in inventory before a successful skill j,
edge i→j probability increases; if absent before a failure, it decreases slightly.

### SIG (Skill-Intervention Graph)
Encodes skill prerequisites as directed edges. Its topological ordering gives the
planner a causal execution sequence. When transferred from 2D to 3D, this ordering
ensures e.g. `stonepickaxe` is attempted before `ironore`.

Hard-coded tech-tree edges (from domain knowledge):
```
wood → stone, coal          (via wooden pickaxe — tier 1 unlock)
stone + wood → ironore      (via stone pickaxe — tier 2 unlock)
iron + wood → diamond       (via iron pickaxe — tier 3 unlock)
```

### Underground Navigation Primer
Since GROOT's BC demonstrations for ore-gathering skills start in dark caves,
the policy produces near-random actions when the agent is at the bright surface.
The primer bridges this gap:

1. **Pitch-down** (3 steps): camera tilts to 45°, GROOT controls buttons
2. **Forced staircase** (150 steps): camera held at 45°, `attack=1, forward=1` forced — digs a diagonal shaft reaching coal/ironore depth
3. **Wall sweep** (42 steps): camera sweeps left/right, `attack=1` forced — mines exposed ore seams

---

## Environments Supported

| Environment | EVGS adapter | Notes |
|-------------|-------------|-------|
| 2D Minecraft chain (symbolic) | `evgs_minecraft.py` | Fast CPU training |
| MineDojo (3D, open-ended) | `evgs_minedojo.py` | Legacy experiments |
| MineRL (ObtainDiamond) | `evgs_minerl.py` | BC data source |
| MineStudio (3D, GROOT/ROCKET-1) | `groot_executor.py` | **Active** |
| CoinRun (ProcGen) | `evgs_procgen.py` | |
| CausalWorld | `evgs_causalworld.py` | |
| Montezuma's Revenge | `evgs_montezuma.py` | |
| Crafter | `evgs_crafter.py` | |

---

## Citation

```bibtex
@phdthesis{Jain2025DIA,
  title   = {Discover, Intervene, Adapt: Interpretable and Adaptive Causal RL
             via Interleaved Discovery and Structured Interventions},
  author  = {Jain, Lakshya Tushar},
  school  = {Arizona State University},
  year    = {2025},
  type    = {MS Dissertation},
  note    = {ProQuest Dissertations \& Theses, No.\ 32283045}
}
```

---

## License

MIT. See [LICENSE](LICENSE).
