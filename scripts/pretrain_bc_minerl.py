#!/usr/bin/env python3
"""
scripts/pretrain_bc_minerl.py  —  BC warm-start for 3D MineRL skill options.

Loads per-skill BC data extracted by filter_minerl_demos.py and trains a small
CNN policy via cross-entropy behavioural cloning.  Saved .pt files are then
consumed by run_transfer_3d.py to warm-start the 3D skills before PPO fine-tune.

Usage
-----
  # Full run (all 9 skills, 50 epochs):
  conda run -n dia-minecraft python scripts/pretrain_bc_minerl.py \\
      --bc_dir data/minerl_bc \\
      --out_dir data/minerl_policies \\
      --epochs 50

  # Smoke test with tiny dataset (5 epochs):
  conda run -n dia-minecraft python scripts/pretrain_bc_minerl.py \\
      --bc_dir /tmp/minerl_bc_test \\
      --out_dir /tmp/minerl_policies_test \\
      --epochs 5

Data format (input)
-------------------
  <bc_dir>/<skill>/obs.npy     — uint8 (N, H, W, 3)  HWC layout
  <bc_dir>/<skill>/actions.npy — int32 (N,)           indices into 25-action set

Policy format (output)
----------------------
  <out_dir>/<skill>.pt  — torch checkpoint with keys:
      'state_dict'  : BCPolicy state_dict
      'n_actions'   : int (25)
      'skill'       : str
      'train_acc'   : float (final training accuracy)
      'epochs'      : int
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Repo path setup ─────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ── Skill names (same order as EVGS 9-variable vector) ──────────────────────
SKILL_NAMES: List[str] = [
    "wood", "stone", "coal", "ironore", "furnace",
    "stonepickaxe", "iron", "ironpickaxe", "diamond",
]

N_ACTIONS: int = 25   # mirrors options_minerl.py _ACTION_DEFS length
OBS_H: int = 64
OBS_W: int = 64


# ---------------------------------------------------------------------------
# BCPolicy — small CNN policy for one skill
# ---------------------------------------------------------------------------

def _build_bc_policy(n_actions: int = N_ACTIONS):
    """Import torch and return a BCPolicy instance.

    Deferred import so the module can be loaded (for --dry_run) even if torch
    is not installed.
    """
    import torch
    import torch.nn as nn

    class BCPolicy(nn.Module):
        """
        Three-conv + two-fc policy.

        Input:  (B, 3, 64, 64)  float32  [0, 1]
        Output: (B, n_actions)  logits
        """

        def __init__(self, n_actions: int = N_ACTIONS) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),  # → (32, 15, 15)
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → (64, 6, 6)
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → (64, 4, 4)
                nn.ReLU(),
            )
            # After three convs: 64 channels × 4 × 4 = 1024
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions),
            )
            self.n_actions = n_actions

        def forward(self, x):
            return self.fc(self.conv(x))

        def predict(self, obs_hwc: np.ndarray) -> int:
            """Single-step inference.  obs_hwc: uint8 (H, W, 3)."""
            import torch
            self.eval()
            with torch.no_grad():
                x = (
                    torch.from_numpy(obs_hwc.astype(np.float32) / 255.0)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )
                logits = self(x)
                return int(logits.argmax(dim=-1).item())

    return BCPolicy(n_actions=n_actions)


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def load_bc_data(bc_dir: str, skill: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load obs.npy + actions.npy for one skill.  Returns None if files missing."""
    obs_path = os.path.join(bc_dir, skill, "obs.npy")
    act_path = os.path.join(bc_dir, skill, "actions.npy")
    if not os.path.exists(obs_path) or not os.path.exists(act_path):
        return None
    obs = np.load(obs_path)    # (N, H, W, 3) uint8
    acts = np.load(act_path)   # (N,) int32
    assert obs.ndim == 4 and obs.shape[1:3] == (OBS_H, OBS_W), (
        f"Expected obs shape (N, {OBS_H}, {OBS_W}, 3), got {obs.shape}"
    )
    assert len(obs) == len(acts), (
        f"obs/actions length mismatch: {len(obs)} vs {len(acts)}"
    )
    return obs, acts.astype(np.int64)


def make_dataloader(obs: np.ndarray, acts: np.ndarray, batch_size: int = 64):
    """Yield (obs_batch, act_batch) mini-batches indefinitely (one pass = one epoch)."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    # Convert HWC uint8 → CHW float32 [0, 1]
    obs_chw = torch.from_numpy(obs.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    act_t   = torch.from_numpy(acts)
    dataset = TensorDataset(obs_chw, act_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_skill(
    skill: str,
    obs: np.ndarray,
    acts: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device_str: str,
    verbose: bool,
) -> Tuple[object, float]:
    """Train BCPolicy for one skill.  Returns (policy, final_train_acc)."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device(device_str)
    policy = _build_bc_policy(n_actions=N_ACTIONS).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loader = make_dataloader(obs, acts, batch_size=batch_size)

    final_acc = 0.0
    for epoch in range(1, epochs + 1):
        policy.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            optimizer.zero_grad()
            logits = policy(obs_batch)
            loss = criterion(logits, act_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(act_batch)
            preds = logits.argmax(dim=-1)
            correct += (preds == act_batch).sum().item()
            total += len(act_batch)

        final_acc = correct / max(1, total)
        if verbose or epoch == epochs or epoch % max(1, epochs // 5) == 0:
            avg_loss = total_loss / max(1, total)
            print(
                f"  [{skill}] epoch {epoch:3d}/{epochs}  "
                f"loss={avg_loss:.4f}  acc={final_acc:.3f}"
            )

    return policy, final_acc


# ---------------------------------------------------------------------------
# Save / Load helpers (used by run_transfer_3d.py)
# ---------------------------------------------------------------------------

def save_bc_policy(policy, out_path: str, skill: str, train_acc: float, epochs: int) -> None:
    """Save BCPolicy weights + metadata to a .pt file."""
    import torch
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "n_actions":  N_ACTIONS,
            "skill":      skill,
            "train_acc":  float(train_acc),
            "epochs":     int(epochs),
        },
        out_path,
    )


def load_bc_policy(pt_path: str):
    """Load a BCPolicy from disk.  Returns (BCPolicy, metadata_dict)."""
    import torch
    ckpt = torch.load(pt_path, map_location="cpu")
    policy = _build_bc_policy(n_actions=ckpt.get("n_actions", N_ACTIONS))
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy, ckpt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Behavioural-cloning warm-start for DIA 3D MineRL skills.\n"
            "Trains a small CNN (BCPolicy) per skill from obs.npy + actions.npy.\n"
            "Outputs: <out_dir>/<skill>.pt  (torch checkpoint)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--bc_dir", type=str, default="data/minerl_bc",
        help="Directory containing per-skill BC data (default: data/minerl_bc)",
    )
    ap.add_argument(
        "--out_dir", type=str, default="data/minerl_policies",
        help="Output directory for .pt policy files (default: data/minerl_policies)",
    )
    ap.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs per skill (default: 50)",
    )
    ap.add_argument(
        "--batch_size", type=int, default=64,
        help="Mini-batch size (default: 64)",
    )
    ap.add_argument(
        "--lr", type=float, default=3e-4,
        help="Adam learning rate (default: 3e-4)",
    )
    ap.add_argument(
        "--device", type=str, default="auto",
        help="Compute device: 'cpu', 'cuda', 'auto' (default: auto)",
    )
    ap.add_argument(
        "--skills", type=str, nargs="*", default=None,
        help="Subset of skills to train (default: all 9)",
    )
    ap.add_argument(
        "--verbose", action="store_true",
        help="Print loss/acc every epoch (default: every 20%% of epochs)",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Load data and print shapes; skip training and saving",
    )
    args = ap.parse_args()

    # ── Device selection ─────────────────────────────────────────────────────
    if args.device == "auto":
        try:
            import torch
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device_str = "cpu"
    else:
        device_str = args.device

    skills_to_train = args.skills if args.skills else SKILL_NAMES

    print("BC Pretraining — DIA MineRL skill policies")
    print(f"  bc_dir   : {args.bc_dir}")
    print(f"  out_dir  : {args.out_dir}")
    print(f"  epochs   : {args.epochs}")
    print(f"  device   : {device_str}")
    print(f"  skills   : {skills_to_train}")
    if args.dry_run:
        print("  [DRY RUN — no training or saving]")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    results: Dict[str, dict] = {}

    for skill in skills_to_train:
        data = load_bc_data(args.bc_dir, skill)
        if data is None:
            print(f"  [{skill}] SKIP — obs.npy / actions.npy not found in {args.bc_dir}/{skill}/")
            continue

        obs, acts = data
        n = len(obs)
        print(f"  [{skill}] loaded {n} transitions  obs={obs.shape}  actions={acts.shape}")

        if args.dry_run:
            results[skill] = {"n": n, "skipped": True}
            continue

        policy, train_acc = train_skill(
            skill=skill,
            obs=obs,
            acts=acts,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device_str=device_str,
            verbose=args.verbose,
        )

        out_path = os.path.join(args.out_dir, f"{skill}.pt")
        save_bc_policy(policy, out_path, skill=skill, train_acc=train_acc, epochs=args.epochs)
        print(f"  [{skill}] saved → {out_path}  (train_acc={train_acc:.3f})")
        results[skill] = {"n": n, "train_acc": train_acc, "path": out_path}

    print()
    print("Summary:")
    for skill in skills_to_train:
        r = results.get(skill)
        if r is None:
            print(f"  {skill:<16}  missing data")
        elif r.get("skipped"):
            print(f"  {skill:<16}  dry-run (n={r['n']})")
        else:
            print(
                f"  {skill:<16}  n={r['n']:6d}  acc={r['train_acc']:.3f}  → {r['path']}"
            )

    if not args.dry_run:
        trained = [s for s in skills_to_train if s in results and not results[s].get("skipped")]
        print(f"\nDone. Trained {len(trained)}/{len(skills_to_train)} skills → {args.out_dir}/")


if __name__ == "__main__":
    main()
