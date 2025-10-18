# Discover–Intervene–Adapt (DIA): Interpretable & Adaptive Causal RL

> A research codebase implementing the DIA paradigm: **Discover** causal structure, **Intervene** with learned skills, and **Adapt** policies—interleaved in one loop—to achieve **generalizable** and **explainable** behavior.

---

## Why DIA?

Modern RL often overfits to correlations and misses the **mechanisms** that generate outcomes. DIA integrates **causal discovery** with **hierarchical control**, so agents *learn why things work*, not just *what works*.

- **Interleaved learning**: structure learning and policy optimization proceed in tandem, each informing the other.
- **Probabilistic causal beliefs**: a **Probabilistic Causal Graph (PCG)** maintains a posterior over environment-variable dependencies.
- **Interventional skills**: a **Skill–Intervention Graph (SIG)** organizes options (skills) as interventions over those variables.
- **Rational exploration**: an **information-gain** bonus prioritizes novel, informative experiments during training.

---

## DIA at a glance

### System overview

```mermaid
flowchart LR
    subgraph Perception
      O[Observations] --> E[Encoder f_ψ]
      E --> X[Environment Variables X]
    end

    subgraph Causal Side
      X --> PCG[Probabilistic Causal Graph q_φ(A)]
      PCG --> IG[Information Gain I G_t]
      IG -->|intrinsic bonus| HL[High-Level Planner/Selector]
    end

    subgraph Control Side
      X --> HL
      HL -->|choose subgoal g=(X_i, F)| SIG[Skill–Intervention Graph]
      SIG --> OPT[Option Policy π_k]
      OPT --> ACT[Primitive Actions]
      ACT --> O
    end

    classDef dim fill:#f7f7f7,stroke:#bbb,color:#333;
    class Perception,Causal Side,Control Side dim;
