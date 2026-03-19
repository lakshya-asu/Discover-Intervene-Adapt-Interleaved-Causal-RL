# DIA-Enhanced: Design Specification
**Date:** 2026-03-19
**Author:** Lakshya Tushar Jain, Arizona State University
**Status:** Approved for implementation

---

## 1. Overview

This spec defines DIA-Enhanced — an upgrade to the Discover-Intervene-Adapt framework that replaces the hand-crafted variable readout `g(·)` and flat variational PCG parameters with principled learned modules. The SIG is upgraded with hyperbolic geometry. A human demonstration warm-starting pipeline is introduced for both Overcooked-AI and Minecraft 3D.

The existing encoder `f_ψ` is unchanged. The probe replaces only `g(·)`. RSSM integration is out of scope for this spec (covered separately in `DIA_WM_architecture_spec.md`).

**Three core problems addressed:**
1. Hand-engineered `X_t` extraction does not scale across environments
2. Flat variational PCG parameters cannot share structure across variables
3. Flat SIG adjacency does not capture hierarchical depth

---

## 2. Architecture

### 2.1 Intervention-Contrastive Probe (replaces `g(·)`)

**Module:** `probes/contrastive_probe.py`

MLP with M independent output heads:
```
z_t → Linear(512) → ReLU → Linear(256) → ReLU → [head_1, ..., head_M]
```
Each head produces one environment variable estimate `X_{i,t} ∈ R`.

**Contrastive training signal:**
When the PCG tags a transition `(z_t, a_t, z_{t+1})` as an intervention on variable `X_i` (via option k targeting X_i):
- Positive: probe output for `X_i` should change
- Negatives: probe outputs for all `X_j, j ≠ i` should stay stable

Loss:
```
L_contrastive = -log(sim(X_i,t+1, X_i,t + δ_i)) + log(Σ_{j≠i} exp(sim(X_j,t+1, X_j,t)))
```
where `sim` is cosine similarity and `δ_i` is the expected change direction from the option's subgoal.

**Edge cases:**
- High PCG uncertainty (early training): soften targets by weighting with edge probability
- Multiple variables changing simultaneously: weight by PCG confidence that option k targets X_i
- No active option (random exploration): collect as observational data, skip contrastive loss
- Overcooked partner acting simultaneously: down-weight by `1 - P(partner_caused_X_i | partner_action)`, estimated from BC partner model

**Gradient isolation:** Probe trains on `z_t.detach()`. No backprop into encoder.

**References:** Lachapelle et al. 2022, Zimmermann et al. 2021, CausalVAE (Yang et al. 2021)

---

### 2.2 TC/Partial-Correlation Regularizer (probe stabilizer)

**Module:** `probes/tc_regularizer.py`

Total Correlation penalty on probe outputs, estimated via minibatch:
```
TC ≈ (1/N) Σ_n [log q(x_n) - Σ_i log q(x_{i,n})]
```

**Partial Correlation variant (preferred):** Once PCG has edges with probability > τ_group, group causally connected variables. Penalize TC only *between* groups, not within:
```
PC = KL(q(X_1,...,X_M) || Π_{g=1}^G q(X_g))
```

**Annealing schedule:**
- Epochs 0–T_warm (first 20% of training): λ_TC = 0
- Epochs T_warm–T_anneal: linear ramp 0 → λ_TC_max
- After T_anneal: hold at λ_TC_max

**References:** β-TCVAE (Chen et al. 2018), PDisVAE (Li et al. 2025)

---

### 2.3 GNN-Based PCG Belief Module

**Module:** `beliefs/gnn_pcg.py`

Replaces flat variational parameters φ with a 2–3 layer Graph Attention Network (GAT).

**Node features (per variable X_i):** mean, variance, change rate under each option
**Edge features:** current edge probability `q_φ(A_{ij})`, intervention co-occurrence counts

**Update rule per batch B from D_int:**
1. Encode each `(X_t, a_t, X_{t+1})` into per-variable change vectors
2. GNN forward pass over current graph structure
3. Output updated edge logits → sigmoid → new `q_φ(A) ∈ [0,1]^{M×M}`
4. Apply NOTEARS acyclicity penalty: `Ω_acyclic = tr(e^{A⊙A}) - M`
5. Apply sparsity penalty: `||E_{q_φ}[A]||_1`

**PCG objective (unchanged structurally):**
```
L(φ,θ) = E_{A~q_φ}[Σ_B log p_θ(X_{t+1}|X_t,a_t,A)] - λ_DAG·Ω_acyclic - λ_sp·||E[A]||_1
```
φ now refers to GNN weights, not raw logits.

**References:** DAG-GNN (Yu et al. 2019), AVICI (Lorch et al. 2022)

---

### 2.4 Hyperbolic SIG Embedding

**Module:** `beliefs/hyperbolic_sig.py`

Each option k gets an embedding `e_k ∈ B^n` (Poincaré ball, n=32).

**Distance:**
```
d_P(e_k, e_k') = arcosh(1 + 2·||e_k - e_k'||² / ((1-||e_k||²)(1-||e_k'||²)))
```

**Training:**
- Prerequisite edges from PCG: minimize `d_P(e_k, e_k')` while keeping prerequisite `e_k` closer to origin
- Effect similarity: options with similar Δ_k signatures attract
- Optimizer: Riemannian SGD (geoopt library)

**Planning:** Backward chaining becomes geodesic path-finding from current state to goal, through prerequisite nodes ordered by distance from origin (most fundamental = closest to center).

**References:** Nickel & Kiela 2017, Chami et al. 2019

---

## 3. Demo Warm-Starting Pipeline

### 3.1 RSSM Post-Hoc Processing (all environments)

**Module:** `warmstart/rssm_demo_processor.py`

Given demonstration trajectories `{τ_demo}`:
1. Run pre-trained RSSM to extract `z_t` per timestep
2. Apply probe to get `X_t` estimates
3. Extract transition tuples `(X_t, a_t, X_{t+1})`
4. Detect pseudo-interventions: segments where one factor changes significantly (> δ_i) while others are stable
5. Warm-start PCG on tagged transitions → output `A_init`
6. Extract subgoal ordering from successful task completions → warm-start SIG edge weights
7. Output: `A_init`, SIG edge weights, pre-trained probe weights

### 3.2 Overcooked Human Demo Integration

**Module:** `warmstart/overcooked_demo_loader.py`

**Data source:** Carroll et al. 2019 human-human gameplay data (available in `overcooked_ai/data/`)

**Pipeline:**
1. Load human-human trajectories; separate per-agent action sequences
2. Compute `X_t` from Overcooked state (pot contents, held items, positions, deliveries)
3. Tag single-agent-act transitions (other agent idle) as clean pseudo-interventions
4. Tag joint-act transitions as "joint interventions" with reduced contrastive weight
5. Run PCG warm-start on tagged transitions → `A_init`
6. Extract subgoal ordering from successful deliveries → SIG edge weights
7. Train BC partner model from human data (`warmstart/bc_partner_trainer.py`)

**BC partner model serves two purposes:**
- Co-player policy during DIA training
- Attribution baseline for disentangling self vs. partner causal effects

### 3.3 Minecraft 3D Human Demo Collection

**Module:** `warmstart/minecraft_recorder.py`

**Goal:** Collect human gameplay trajectories in Minecraft 3D (MineRL) across world seeds different from the training trials.

**Recording pipeline:**
1. Launch MineRL environment with a recorder wrapper that captures keyboard/mouse input
2. Human plays; recorder stores `(obs_t, action_t, timestamp_t)` per step as compressed numpy arrays
3. Log the world seed alongside each trajectory (demo seeds ≠ trial seeds by design)
4. Save to `data/demos/minecraft3d/` as trajectory files

**Post-hoc processing:** Same as Task 3.1 — raw `(obs, action)` only at collection time; RSSM/probe applied offline.

**Design rationale:** Keeping the recorder minimal (no online processing) allows the RSSM and probe to evolve independently of when demos were recorded. Re-processing is always possible.

---

## 4. Training Loop Integration

### 4.1 Modifications to Algorithm 1

```
MODIFICATIONS TO ALGORITHM 1:
─────────────────────────────
Line "Initialize q_φ(A)":
  → Load A_init from demo warm-start
  → Initialize GNN-PCG weights from demo pre-training

New: After UpdatePCG:
  → Update contrastive probe on intervention-tagged transitions
  → If epoch > T_warm: compute TC/PC, add to probe loss
  → Recompute PCG causal groups for partial TC

Line "RefreshSIG":
  → Update Poincaré embeddings via RSGD
  → PCG prerequisite edges inform hyperbolic distances
  → Effect similarity from Δ_k signatures adds attraction

New: Logging
  → DCI / MIG disentanglement score
  → PCG entropy H[q_φ(A)]
  → TC and PC values over training
  → Hyperbolic SIG embedding norms (hierarchy depth proxy)
  → Contrastive loss convergence
```

### 4.2 Overcooked-Specific Modifications

```
OVERCOOKED ADDITIONS:
─────────────────────
Partner:
  → Load BC partner from warmstart pipeline
  → Optionally re-train BC on mixed (human + DIA agent) data every 50–200 episodes
  → Tag each transition with which agent caused state change

Credit assignment at delivery:
  → Trace PCG backward: who loaded pot? who plated? who delivered?
  → Use attribution for option-level reward shaping

Multi-agent contrastive weighting:
  → Partner idle during DIA option execution: full contrastive weight
  → Partner also acted: weight = 1 - P(partner_caused_X_i | partner_action)
    estimated from BC model
```

### 4.3 Loss Summary

```
L_probe = L_contrastive + λ_TC(t)·L_TC + λ_recon·L_recon
L_PCG   = -E_{A~q_φ}[Σ_B log p_θ(X_{t+1}|X_t,a_t,A)] + λ_DAG·Ω_acyclic + λ_sp·||E[A]||_1
r_t     = r_ext + λ_sub·r_sub + β·IG_t   (unchanged)
```

**Gradient isolation (strict):**
- Probe trains on `z_t.detach()`
- PCG trains on `X_t.detach()` from probe
- Encoder `f_ψ` receives no gradient from probe or PCG

---

## 5. Experimental Domains

### Domain 1: 2D-Minecraft — Trade Token (Primary Ablation)
- 10+ variable causal chain (Wood → Plank → Stick → … → TradeToken), ground truth known
- M ≈ 15, binary/count variables
- Run full ablation table here
- Baselines: DIA baseline, CDHRL

### Domain 2: NetHack (via NLE)
- Long-horizon, deeply hierarchical, no ground truth graph
- Tests whether PCG discovers meaningful structure without supervision
- Variables: player_level, hp, item_identified, door_unlocked, corridor_explored, staircase_reached, …
- Baselines: DIA baseline, RIDE, NovelD

### Domain 3: ProcGen CoinRun (Generalization)
- Procedural train/test split; test zero-shot transfer of causal structure
- M = 6, key metric: train/test performance gap
- Baselines: DIA baseline, PPO, IBAC

### Domain 4: CausalWorld (Robotic Manipulation)
- Continuous physical variables, purpose-built for causal transfer
- M ≈ 8–12, exposes all causal variables for intervention
- Baselines: DIA baseline, SAC, CausalWorld built-ins

### Domain 5: Overcooked-AI (Multi-Agent + Demos)
- Partner as latent confounder on shared state; real human demo data available
- Layouts: Cramped Room (dev), Asymmetric Advantages, Counter Circuit, Coordination Ring
- M ≈ 10–14, mixed continuous/categorical
- Baselines: PPO self-play, PPO+BC, CausalPlan, IReCa

---

## 6. Evaluation

### 6.1 Metrics

**Representation quality:**
- DCI Disentanglement score (Eastwood & Williams 2018)
- Mutual Information Gap / MIG (Chen et al. 2018)
- Interventional robustness: X_i prediction accuracy under targeted vs. non-targeted interventions

**Causal discovery:**
- PCG entropy half-life `t_{1/2}(H[q_φ])`
- Structural Hamming Distance (SHD) to ground truth (2D-Minecraft, CausalWorld)
- Expected Calibration Error (ECE) on edge posteriors
- Intervention precision

**RL performance:**
- Normalized return (mean ± std, 10 seeds)
- Frames to 75% of final return (sample efficiency)

**Enhancement-specific:**
- Contrastive loss convergence curve
- TC/PC values over training
- Poincaré distance vs. true prerequisite depth correlation
- Demo warm-start lift: PCG convergence speed with vs. without demos

**Overcooked-specific:**
- Soups delivered per episode
- Coordination efficiency (useful actions / total actions)
- Causal attribution accuracy (does PCG correctly identify which agent caused state changes?)
- Zero-shot coordination score (ZSC-Eval, Wang et al. NeurIPS 2024)

### 6.2 Ablation Table (run on 2D-Minecraft Trade Token)

| Configuration | Contrastive Probe | GNN-PCG | Hyperbolic SIG | TC Reg | Demo Warmstart |
|---|---|---|---|---|---|
| DIA (baseline) | ✗ | ✗ | ✗ | ✗ | ✗ |
| + Contrastive only | ✓ | ✗ | ✗ | ✗ | ✗ |
| + GNN-PCG | ✓ | ✓ | ✗ | ✗ | ✗ |
| + Hyperbolic SIG | ✓ | ✓ | ✓ | ✗ | ✗ |
| + TC regularizer | ✓ | ✓ | ✓ | ✓ | ✗ |
| Full system | ✓ | ✓ | ✓ | ✓ | ✓ |
| TC only (no contrastive) | ✗ | ✗ | ✗ | ✓ | ✗ |

### 6.3 Per-Domain Evaluation Focus

| Domain | Primary Question | Key Comparison |
|---|---|---|
| 2D-Minecraft | Does the full pipeline work? | Enhanced DIA vs. DIA baseline vs. CDHRL |
| NetHack | Does PCG discover structure without ground truth? | Enhanced DIA vs. DIA baseline vs. RIDE |
| CoinRun | Does causal structure improve generalization? | Train/test gap: enhanced vs. baseline |
| CausalWorld | Does it work with continuous physical variables? | Enhanced DIA vs. DIA baseline vs. SAC |
| Overcooked | Does it handle multi-agent + demos + confounders? | Enhanced DIA vs. PPO-BC vs. CausalPlan vs. IReCa |

---

## 7. Implementation Order

1. **Contrastive probe** (`probes/contrastive_probe.py`) — test on 2D-Minecraft Cramped variant
2. **GNN-PCG** (`beliefs/gnn_pcg.py`) — test on 2D-Minecraft Trade Token (known ground truth)
3. **TC regularizer** (`probes/tc_regularizer.py`) — compare DCI/MIG with/without on 2D-Minecraft
4. **Demo warm-start** (Phases 2.1–2.3) — test PCG convergence speed lift on 2D-Minecraft and CoinRun
5. **Overcooked integration** — Domain 5 setup, Task 2.2, multi-agent contrastive signal (Task 3.2)
6. **Hyperbolic SIG** (`beliefs/hyperbolic_sig.py`) — self-contained, add last
7. **Full integration and evaluation** across all five domains

---

## 8. File Structure

```
dia_enhanced/
├── probes/
│   ├── contrastive_probe.py
│   └── tc_regularizer.py
├── beliefs/
│   ├── gnn_pcg.py
│   └── hyperbolic_sig.py
├── warmstart/
│   ├── rssm_demo_processor.py
│   ├── minecraft_recorder.py
│   ├── overcooked_demo_loader.py
│   ├── bc_partner_trainer.py
│   └── warmstart_pipeline.py
├── envs/
│   ├── overcooked_wrapper.py
│   ├── overcooked_variables.py
│   └── env_registry.py
├── training/
│   ├── dia_enhanced_loop.py
│   ├── overcooked_training.py
│   └── losses.py
├── eval/
│   ├── disentanglement_metrics.py
│   ├── causal_metrics.py
│   ├── overcooked_metrics.py
│   └── ablation_runner.py
└── configs/
    ├── minecraft_trade.yaml
    ├── nethack.yaml
    ├── coinrun.yaml
    ├── causalworld.yaml
    └── overcooked.yaml
```

---

## 9. Key Hyperparameters

| Parameter | Suggested Range | Controls |
|---|---|---|
| λ_TC_max | 0.05–0.5 | TC regularizer strength at full anneal |
| T_warm | 15–25% of training | When TC begins |
| τ_group | 0.5–0.8 | PCG edge threshold for partial TC grouping |
| GNN layers | 2–3 | GNN-PCG depth |
| GNN hidden dim | 64–128 | GNN-PCG width |
| Poincaré ball dim | 16–64 | SIG embedding dimension |
| Contrastive temperature | 0.05–0.2 | Contrastive loss sharpness |
| Demo warmstart epochs | 50–200 | Pre-training duration on demos |
| Partner attribution discount | 0.3–0.7 | Down-weight when partner also acted (Overcooked) |
| BC partner update frequency | every 50–200 episodes | BC re-training cadence (Overcooked) |

---

## 10. Constraints

- Framework: PyTorch
- New dependencies: `geoopt` (Riemannian optimization), `torch_geometric` (GNN), `overcooked-ai` (HumanCompatibleAI)
- Probe replaces only `g(·)`, not `f_ψ`. Backward compatible when all enhancements disabled.
- For Overcooked: DIA controls one agent only. Partner is BC model during training, held-out policy at eval.
- Implement DIA baseline on Overcooked before layering enhancements.
- RSSM integration is out of scope (see `DIA_WM_architecture_spec.md`).

---

## 11. References

- Jain, L.T. (2025). Discover, Intervene, Adapt. MS Thesis, ASU.
- Peng et al. (2022). CDHRL. NeurIPS 2022.
- Lachapelle et al. (2022). Linear Causal Disentanglement via Interventions. ICML 2022.
- Zimmermann et al. (2021). Contrastive Learning Inverts the Data Generating Process. ICML 2021.
- Yang et al. (2021). CausalVAE. CVPR 2021.
- Yu et al. (2019). DAG-GNN. ICML 2019.
- Lorch et al. (2022). AVICI. NeurIPS 2022.
- Chen et al. (2018). β-TCVAE. NeurIPS 2018.
- Nickel & Kiela (2017). Poincaré Embeddings. NeurIPS 2017.
- Carroll et al. (2019). Overcooked. NeurIPS 2019.
- CausalPlan (2025). arXiv:2508.13721.
- IReCa (2024). arXiv:2408.07877.
- Wang et al. (2024). ZSC-Eval. NeurIPS 2024.
