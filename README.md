# Discover, Intervene, Adapt (DIA)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey?logo=linux)](https://ubuntu.com)
[![Status](https://img.shields.io/badge/Status-Research%20Code-orange)]()

**Interpretable and adaptive causal RL via interleaved discovery and structured interventions.**

DIA agents learn a probabilistic causal graph (PCG) from skill interventions, organise skills into a prerequisite-linked Skill-Intervention Graph (SIG), and plan goal-directed sequences using causal structure rather than reward correlation. The key result: structure learned in a fast 2D symbolic environment transfers directly to guide skill ordering in 3D Minecraft, without any additional 3D training.

---

## Setup

**Requirements:** Linux, Miniconda, Java 11 (for Minecraft experiments).

```bash
# 2D environments (CPU only)
conda env create -f environment.yml
conda activate dia
pip install -e .

# 3D MineDojo experiments (GPU)
conda env create -f environment-gpu.yml
conda activate dia-minecraft
pip install minedojo
pip install -e .
```

MineDojo requires patching one NumPy 2.0 compatibility issue in `minedojo/sim/spaces.py` (replace `np.unicode_` with `np.str_`). A script for this is in `scripts/setup_vpt_data.sh`.

---

## Quick Start

**Train the 2D causal prior (5 min on CPU):**

```bash
python scripts/train_minecraft2d_dia.py \
    --pcg simple --steps 5000 --logdir runs/minecraft2d
```

This saves `pcg_2d.npy` and `sig_2d.json`, the causal prior used for 3D transfer.

**Run the 3D MineDojo transfer experiment:**

```bash
# Transfer condition: SIG ordering from 2D prior
conda run -n dia-minecraft python scripts/run_transfer_minedojo.py \
    --mode transfer --seed 0 \
    --pcg_path pcg_2d.npy \
    --bc_dir data/minerl_policies \
    --out results/transfer_seed0.json

# Baseline: no ordering prior
conda run -n dia-minecraft python scripts/run_transfer_minedojo.py \
    --mode baseline --seed 0 \
    --out results/baseline_seed0.json
```

**Pre-train BC gathering policies from MineRL demos:**

```bash
python scripts/pretrain_bc_minerl.py \
    --skill wood --data_dir data/minerl_bc \
    --out data/minerl_policies/wood.pt
```

Pre-trained checkpoints for wood, stone, coal, ironore, and diamond are already in `data/minerl_policies/`.

---

## Repository Layout

```
src/dia/
    evgs*.py          # environment-variable goal space adapters (per env)
    pcg*.py           # PCG backends: Simple, Differentiable, Variational, Granger
    sig.py            # Skill-Intervention Graph and topo-sort planner
    options*.py       # skill option policies per environment
    rollout.py        # DIARunner: the interleaved training loop
    planner.py        # three-phase intervention selector (novelty / confirm / goal)

scripts/
    train_minecraft2d_dia.py      # 2D training
    pretrain_bc_minerl.py         # BC policy training from MineRL demos
    run_transfer_minedojo.py      # main 3D transfer experiment
    run_transfer_pool_minedojo.py # parallel multi-seed runner
    train_coinrun_dia.py          # CoinRun experiments
    train_causalworld_dia.py      # CausalWorld experiments
```

---

## How It Works

**EVGS** maps raw observations to a fixed set of interpretable binary variables (`has_wood`, `has_stone`, etc.). Options target predicates over these variables.

**PCG** maintains a posterior over directed edges between variables. Each skill execution is an intervention; the PCG updates from the observed variable deltas. Edge entropy drives an information-gain exploration bonus.

**SIG** encodes skill prerequisites as directed edges. Its topological ordering gives the planner a causal execution sequence. When transferred from 2D to 3D, this ordering ensures skills like `stonepickaxe` are attempted before `ironore` (which requires a stone pickaxe to succeed).

---

## Supported Environments

| Environment | EVGS adapter |
|-------------|-------------|
| 2D Minecraft chain (symbolic) | `evgs_minecraft.py` |
| MineDojo (3D, open-ended) | `evgs_minedojo.py` |
| MineRL (ObtainDiamond) | `evgs_minerl.py` |
| CoinRun (ProcGen) | `evgs_procgen.py` |
| CausalWorld | `evgs_causalworld.py` |
| Montezuma's Revenge | `evgs_montezuma.py` |
| Crafter | `evgs_crafter.py` |

---

## Citation

```bibtex
@phdthesis{Jain2025DIA,
  title   = {Discover, Intervene, Adapt: Interpretable and Adaptive Causal RL
             via Interleaved Discovery and Structured Interventions},
  author  = {Jain, Lakshya Tushar},
  school  = {Arizona State University},
  year    = {2025},
  type    = {Doctoral Dissertation},
  note    = {ProQuest Dissertations \& Theses, No.\ 32283045}
}
```

---

## License

MIT. See [LICENSE](LICENSE).
