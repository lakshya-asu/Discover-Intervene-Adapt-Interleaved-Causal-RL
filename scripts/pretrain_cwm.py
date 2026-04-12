#!/usr/bin/env python3
"""
scripts/pretrain_cwm.py  —  Offline pre-training of the Causal World Model (CWM).

Loads the (x_t, skill_idx, x_tp1) buffer saved by train_minecraft2d_sip.py
and trains a CausalWorldModel on it. The trained model is saved as cwm_2d.pt
for use in transfer_minecraft3d_dia.py (Phase 2 online adaptation).

Why offline pre-training?
--------------------------
The 2D symbolic Minecraft env produces thousands of clean causal transitions
in seconds (no Java server, no rendering). Pre-training the CWM here means
that when Phase 2 starts, the model already knows the causal structure:
  - "if stone=1 and skill=stonepickaxe, then stonepickaxe↑ is likely"
  - "if ironore=1, furnace=1, coal=1 and skill=iron, then iron↑ is likely"

During Phase 2 (3D MineRL), the model is fine-tuned online to adapt to
differences between the symbolic 2D env and the real 3D environment.

Usage
-----
  # Step 1: Run Phase 1 to generate the CWM buffer
  conda run -n dia python scripts/train_minecraft2d_sip.py \\
      --macro_steps 200 --cwm_buf_out cwm_buffer_2d

  # Step 2: Pre-train the CWM
  conda run -n dia python scripts/pretrain_cwm.py \\
      --buf cwm_buffer_2d.npz --pcg pcg_2d.npy --out cwm_2d.pt

  # Step 3: Use in Phase 2 (transfer_minecraft3d_dia.py)
  conda run -n dia-minecraft python scripts/transfer_minecraft3d_dia.py \\
      --sig sig_2d.json --cwm cwm_2d.pt --out transfer_minecraft3d.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from dia.cwm import CausalWorldModel, CWMConfig, CWMTrainer
from dia.evgs_minerl import VAR_NAMES

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


def main():
    ap = argparse.ArgumentParser(
        description="Offline pre-training of CWM from 2D Minecraft transitions.")
    ap.add_argument("--buf",    type=str, default="cwm_buffer_2d.npz",
                    help="Path to .npz buffer from train_minecraft2d_sip.py")
    ap.add_argument("--pcg",    type=str, default="pcg_2d.npy",
                    help="Path to pcg_2d.npy (PCG edge probs from Phase 1)")
    ap.add_argument("--out",    type=str, default="cwm_2d.pt",
                    help="Output path for trained CWM weights")
    ap.add_argument("--epochs", type=int, default=0,
                    help="Training epochs (0 = use CWMConfig default of 50)")
    ap.add_argument("--lr",     type=float, default=0.0,
                    help="Learning rate (0 = use CWMConfig default of 1e-3)")
    ap.add_argument("--batch",  type=int, default=0,
                    help="Batch size (0 = use CWMConfig default of 64)")
    ap.add_argument("--device", type=str, default="auto",
                    help="'cpu', 'cuda', or 'auto'")
    ap.add_argument("--edge_hidden",   type=int, default=16,
                    help="Hidden units in per-edge MLP")
    ap.add_argument("--skill_emb_dim", type=int, default=16,
                    help="Skill embedding dimension")
    args = ap.parse_args()

    if not TORCH_OK:
        print("ERROR: PyTorch is required. Install torch.")
        return

    # ── Load buffer ───────────────────────────────────────────────────────────
    buf_path = args.buf
    if not os.path.exists(buf_path):
        # Try without extension
        if os.path.exists(buf_path + ".npz"):
            buf_path = buf_path + ".npz"
        else:
            print(f"ERROR: buffer not found: {buf_path}")
            print("  Run train_minecraft2d_sip.py first to generate the buffer.")
            return

    data = np.load(buf_path)
    x_t_all     = data["x_t"].astype(np.float32)    # (N, M)
    skill_all   = data["skill_idx"].astype(np.int64) # (N,)
    x_tp1_all   = data["x_tp1"].astype(np.float32)  # (N, M)
    N, M = x_t_all.shape

    print(f"Loaded CWM buffer: {buf_path}")
    print(f"  Transitions: {N}")
    print(f"  Variables:   {M}  ({VAR_NAMES})")
    print(f"  Skills seen: {np.unique(skill_all).tolist()}")

    # ── Load PCG probs ────────────────────────────────────────────────────────
    pcg_probs = None
    if os.path.exists(args.pcg):
        pcg_probs = np.load(args.pcg).astype(np.float32)
        np.fill_diagonal(pcg_probs, 0.0)
        print(f"Loaded PCG probs: {args.pcg}  shape={pcg_probs.shape}")
        # Print top edges
        print("  Top PCG edges (p >= 0.5):")
        for i in range(M):
            for j in range(M):
                if i != j and pcg_probs[i, j] >= 0.5:
                    print(f"    {VAR_NAMES[i]} → {VAR_NAMES[j]}: {pcg_probs[i,j]:.3f}")
    else:
        print(f"  PCG file not found ({args.pcg}); using uniform edge probs (0.5)")

    # ── Build CWMConfig ───────────────────────────────────────────────────────
    cfg = CWMConfig(
        num_vars     = M,
        num_skills   = M,    # one skill per DIA variable
        edge_hidden  = args.edge_hidden,
        skill_emb_dim= args.skill_emb_dim,
        offline_lr   = args.lr     if args.lr    > 0 else 1e-3,
        offline_epochs = args.epochs if args.epochs > 0 else 50,
        offline_batch  = args.batch  if args.batch  > 0 else 64,
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # ── Build trainer ─────────────────────────────────────────────────────────
    trainer = CWMTrainer(cfg, pcg=None, device=device)

    if pcg_probs is not None:
        pcg_tensor = torch.tensor(pcg_probs, dtype=torch.float32, device=trainer.device)
    else:
        uniform = np.full((M, M), 0.5, dtype=np.float32)
        np.fill_diagonal(uniform, 0.0)
        pcg_tensor = torch.tensor(uniform, dtype=torch.float32, device=trainer.device)

    # Store pcg tensor directly for offline training
    trainer._offline_pcg = pcg_tensor

    # ── Build transitions list ────────────────────────────────────────────────
    transitions = [
        (x_t_all[i], int(skill_all[i]), x_tp1_all[i])
        for i in range(N)
    ]

    print(f"\nStarting offline training:")
    print(f"  Transitions: {N}")
    print(f"  Epochs:      {cfg.offline_epochs}")
    print(f"  Batch size:  {cfg.offline_batch}")
    print(f"  LR:          {cfg.offline_lr}")

    # ── Train ─────────────────────────────────────────────────────────────────
    # Override fit_offline to use our pre-loaded pcg_tensor
    losses = _fit_offline_with_pcg(trainer, transitions, pcg_tensor, cfg)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print(f"\nEvaluation on training data (final accuracy per variable):")
    trainer.model.eval()
    with torch.no_grad():
        xs  = torch.tensor(x_t_all,   device=trainer.device)
        ks  = torch.tensor(skill_all, device=trainer.device)
        xps = torch.tensor(x_tp1_all, device=trainer.device)

        proba = trainer.model.predict_proba(xs, ks, pcg_tensor)  # [N, M]
        preds = (proba >= 0.5).float()
        acc_per_var = (preds == xps).float().mean(dim=0).cpu().numpy()

    for i, (name, acc) in enumerate(zip(VAR_NAMES, acc_per_var)):
        bar = "#" * int(acc * 20)
        print(f"  {name:<14}: {acc:.3f}  [{bar:<20}]")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    trainer.save(args.out)
    print(f"\nSaved CWM: {args.out}")
    print(f"  Final loss: {losses[-1]:.4f}" if losses else "  (no training done)")
    print(f"\nNext step:")
    print(f"  conda run -n dia-minecraft python scripts/transfer_minecraft3d_dia.py \\")
    print(f"      --sig sig_2d.json --cwm {args.out} --out transfer_minecraft3d.mp4")


def _fit_offline_with_pcg(trainer, transitions, pcg_tensor, cfg):
    """
    Like CWMTrainer.fit_offline but uses a pre-built PCG tensor.
    Returns list of per-epoch losses.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np

    if not transitions:
        return []

    losses = []
    model  = trainer.model
    opt    = trainer._offline_opt

    model.train()
    for epoch in range(cfg.offline_epochs):
        epoch_losses = []
        n = len(transitions)
        idx = np.random.permutation(n)
        for start in range(0, n, cfg.offline_batch):
            batch_idx = idx[start: start + cfg.offline_batch]
            xs  = torch.tensor(
                np.stack([transitions[i][0] for i in batch_idx]),
                dtype=torch.float32, device=trainer.device)
            ks  = torch.tensor(
                [transitions[i][1] for i in batch_idx],
                dtype=torch.int64,  device=trainer.device)
            xps = torch.tensor(
                np.stack([transitions[i][2] for i in batch_idx]),
                dtype=torch.float32, device=trainer.device)

            opt.zero_grad()
            logits = model(xs, ks, pcg_tensor)
            loss   = F.binary_cross_entropy_with_logits(logits, xps)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  [CWM offline] epoch {epoch+1}/{cfg.offline_epochs}  loss={mean_loss:.4f}")

    model.eval()
    return losses


if __name__ == "__main__":
    main()
