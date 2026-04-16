"""Fine-tune GrootPolicy on DIA skill-specific BC demonstrations.

Fine-tunes the decoder and pi_head (and value_head) on 9 per-skill BC
datasets collected from MineRL environments.  Backbone, video_encoder,
image_encoder, updim, and fuser are kept frozen.

Usage:
    python scripts/finetune_groot.py [--epochs 3] [--seq_len 16] \
        [--batch_size 2] [--lr 5e-5]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("finetune_groot")

# ---------------------------------------------------------------------------
# Project root (scripts/ lives one level below repo root)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all RNG sources for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Seed set to %d (torch/numpy/random/cudnn deterministic).", seed)


# ---------------------------------------------------------------------------
# Progress tracking (written after each skill so remote watchers can poll)
# ---------------------------------------------------------------------------

PROGRESS_FILE = REPO_ROOT / "data" / "groot_ft" / "progress.json"


def _load_progress() -> Dict[str, Any]:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    return {"skills": {}, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def _save_progress(progress: Dict[str, Any]) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    progress["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


# ---------------------------------------------------------------------------
# DIA action → VPT action mapping
# ---------------------------------------------------------------------------

# (fore_back, left_right, sprint_sneak, use, attack, jump, pitch_deg, yaw_deg)
DIA_TO_VPT: Dict[int, Tuple] = {
    0:  ('none', 'none', 'none', 'none', 'none', 'none',  0.0,  0.0),
    1:  ('forward', 'none', 'none', 'none', 'none', 'none',  0.0,  0.0),
    2:  ('back', 'none', 'none', 'none', 'none', 'none',  0.0,  0.0),
    3:  ('none', 'left', 'none', 'none', 'none', 'none',  0.0,  0.0),
    4:  ('none', 'right', 'none', 'none', 'none', 'none',  0.0,  0.0),
    5:  ('none', 'none', 'none', 'none', 'none', 'jump',  0.0,  0.0),
    6:  ('none', 'none', 'none', 'none', 'attack', 'none',  0.0,  0.0),
    7:  ('none', 'none', 'none', 'use', 'none', 'none',  0.0,  0.0),
    8:  ('forward', 'none', 'none', 'none', 'attack', 'none',  0.0,  0.0),
    9:  ('forward', 'none', 'none', 'none', 'none', 'jump',  0.0,  0.0),
    10: ('forward', 'none', 'sprint', 'none', 'none', 'none',  0.0,  0.0),
    11: ('none', 'none', 'none', 'none', 'none', 'none', -15.0,  0.0),
    12: ('none', 'none', 'none', 'none', 'none', 'none',  15.0,  0.0),
    13: ('none', 'none', 'none', 'none', 'none', 'none',  0.0, -15.0),
    14: ('none', 'none', 'none', 'none', 'none', 'none',  0.0,  15.0),
    15: ('none', 'none', 'sprint', 'none', 'attack', 'none',  0.0,  0.0),
    16: ('none', 'none', 'none', 'none', 'none', 'none',  -5.0,  0.0),
    17: ('none', 'none', 'none', 'none', 'none', 'none',   5.0,  0.0),
    18: ('none', 'none', 'none', 'none', 'none', 'none',  0.0,  -5.0),
    19: ('none', 'none', 'none', 'none', 'none', 'none',  0.0,   5.0),
    20: ('none', 'none', 'none', 'none', 'none', 'none',  0.0,  0.0),
    21: ('none', 'none', 'none', 'none', 'none', 'none',  0.0,  0.0),
    22: ('forward', 'none', 'none', 'use', 'none', 'none',  0.0,  0.0),
    23: ('none', 'none', 'none', 'none', 'attack', 'jump',  0.0,  0.0),
    24: ('forward', 'none', 'none', 'none', 'attack', 'jump',  0.0,  0.0),
}

SKILLS: List[str] = [
    "wood", "stone", "coal", "ironore",
    "furnace", "stonepickaxe", "iron", "ironpickaxe", "diamond",
]


def build_dia_to_vpt_map() -> Tuple[Dict[int, int], Dict[int, int]]:
    """Build DIA action index → (buttons_idx, camera_idx) mappings."""
    from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping

    mapper = CameraHierarchicalMapping(n_camera_bins=11)

    def cam_bin(deg: float) -> int:
        return int(np.clip(round((deg / 15.0) * 5) + 5, 0, 10))

    dia_to_buttons: Dict[int, int] = {}
    dia_to_camera: Dict[int, int] = {}

    for dia_idx, (fb, lr, ss, use, atk, jmp, pitch, yaw) in DIA_TO_VPT.items():
        cam_move = 'camera' if (pitch != 0.0 or yaw != 0.0) else 'none'
        # hotbar always 'none'; drop always 'none'
        key = ('none', fb, lr, ss, use, 'none', atk, jmp, cam_move)
        btn_idx = mapper.BUTTONS_COMBINATION_TO_IDX.get(key, 0)
        pitch_bin = cam_bin(pitch)
        yaw_bin = cam_bin(yaw)
        cam_idx = pitch_bin * 11 + yaw_bin
        dia_to_buttons[dia_idx] = btn_idx
        dia_to_camera[dia_idx] = cam_idx

    logger.info("DIA→VPT button map sample: %s",
                {k: v for k, v in list(dia_to_buttons.items())[:6]})
    return dia_to_buttons, dia_to_camera


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GrootBCDataset(Dataset):
    """Sliding-window BC dataset for a single skill."""

    def __init__(
        self,
        skill: str,
        dia_to_buttons: Dict[int, int],
        dia_to_camera: Dict[int, int],
        seq_len: int = 16,
    ) -> None:
        data_dir = REPO_ROOT / "data" / "minerl_bc" / skill
        obs_path = data_dir / "obs.npy"
        act_path = data_dir / "actions.npy"

        if not obs_path.exists():
            raise FileNotFoundError(f"BC obs not found: {obs_path}")
        if not act_path.exists():
            raise FileNotFoundError(f"BC actions not found: {act_path}")

        self.obs = np.load(str(obs_path), mmap_mode='r')   # (N, 64, 64, 3)
        self.act = np.load(str(act_path), mmap_mode='r')   # (N,)
        self.seq_len = seq_len
        self.dia_to_buttons = dia_to_buttons
        self.dia_to_camera = dia_to_camera

        stride = seq_len // 2
        n = len(self.obs)
        self.windows = [
            (i, i + seq_len)
            for i in range(0, n - seq_len, stride)
        ]
        logger.info(
            "  skill=%-15s  frames=%d  windows=%d  seq_len=%d",
            skill, n, len(self.windows), seq_len,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        s, e = self.windows[i]
        frames = self.obs[s:e].copy()   # (T, 64, 64, 3) uint8

        # Upsample 64→224 using PIL (avoids cv2 / numpy 2.x issues)
        frames_224 = np.stack([
            np.array(
                PIL.Image.fromarray(f).resize((224, 224), PIL.Image.BILINEAR)
            )
            for f in frames
        ])  # (T, 224, 224, 3)

        dia_actions = self.act[s:e]  # (T,)
        btn_labels = np.array(
            [self.dia_to_buttons.get(int(a), 0) for a in dia_actions],
            dtype=np.int64,
        )
        cam_labels = np.array(
            [self.dia_to_camera.get(int(a), 60) for a in dia_actions],
            dtype=np.int64,
        )

        return (
            torch.from_numpy(frames_224),          # (T, 224, 224, 3) uint8
            torch.from_numpy(btn_labels),           # (T,)  int64
            torch.from_numpy(cam_labels),           # (T,)  int64
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def freeze_module(mod: torch.nn.Module) -> None:
    for p in mod.parameters():
        p.requires_grad = False


def unfreeze_module(mod: torch.nn.Module) -> None:
    for p in mod.parameters():
        p.requires_grad = True


def train_skill(
    policy: torch.nn.Module,
    skill: str,
    dia_to_buttons: Dict[int, int],
    dia_to_camera: Dict[int, int],
    epochs: int,
    seq_len: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int = 42,
    progress: Dict[str, Any] | None = None,
) -> None:
    """Fine-tune policy on one skill's BC data."""
    logger.info("=" * 60)
    logger.info("Training skill: %s  (epochs=%d)", skill, epochs)
    set_seed(seed)          # deterministic per-skill reset

    dataset = GrootBCDataset(skill, dia_to_buttons, dia_to_camera, seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scaler = torch.cuda.amp.GradScaler()

    total_steps = 0
    first_loss = mid_loss = final_loss = float('nan')
    mid_step = (len(loader) * epochs) // 2

    policy.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_steps = 0

        for step, (frames, btn_labels, cam_labels) in enumerate(loader, 1):
            # frames: (B, T, 224, 224, 3)  uint8
            # labels: (B, T)
            frames = frames.to(device)           # keep uint8, GROOT casts internally
            btn_labels = btn_labels.to(device)   # (B, T)
            cam_labels = cam_labels.to(device)   # (B, T)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                latents, _ = policy({'image': frames}, memory=None)
                # buttons: (B, T, 1, 8641) — squeeze the extra dim
                btn_logits = latents['pi_logits']['buttons'].squeeze(2)  # (B, T, 8641)
                cam_logits = latents['pi_logits']['camera'].squeeze(2)   # (B, T, 121)

                btn_loss = F.cross_entropy(
                    btn_logits.reshape(-1, 8641),
                    btn_labels.reshape(-1),
                )
                cam_loss = F.cross_entropy(
                    cam_logits.reshape(-1, 121),
                    cam_labels.reshape(-1),
                )
                loss = btn_loss + 0.5 * cam_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1
            total_steps += 1

            if total_steps == 1:
                first_loss = loss_val

            if total_steps == mid_step:
                mid_loss = loss_val

            if step % 50 == 0 or step == len(loader):
                logger.info(
                    "  [%s] epoch %d/%d  step %d/%d  "
                    "loss=%.4f  btn=%.4f  cam=%.4f",
                    skill, epoch, epochs, step, len(loader),
                    loss_val, btn_loss.item(), cam_loss.item(),
                )

        logger.info(
            "  [%s] epoch %d done  avg_loss=%.4f",
            skill, epoch, epoch_loss / max(epoch_steps, 1),
        )

    final_loss = loss_val  # noqa: F821  (set on last iteration)

    logger.info(
        "[%s] loss trajectory: first=%.4f  mid=%.4f  final=%.4f",
        skill, first_loss, mid_loss, final_loss,
    )

    # --- deterministic validation hook: one forward pass, no grad ----------
    policy.eval()
    val_losses: List[float] = []
    with torch.no_grad():
        for val_step, (vframes, vbtn, vcam) in enumerate(loader):
            if val_step >= 20:
                break
            vframes = vframes.to(device)
            vbtn = vbtn.to(device)
            vcam = vcam.to(device)
            with torch.amp.autocast("cuda"):
                latents, _ = policy({"image": vframes}, memory=None)
                vbtn_logits = latents["pi_logits"]["buttons"].squeeze(2)
                vcam_logits = latents["pi_logits"]["camera"].squeeze(2)
                v_loss = (
                    F.cross_entropy(vbtn_logits.reshape(-1, 8641), vbtn.reshape(-1))
                    + 0.5 * F.cross_entropy(vcam_logits.reshape(-1, 121), vcam.reshape(-1))
                )
            val_losses.append(v_loss.item())
    val_loss_mean = float(np.mean(val_losses)) if val_losses else float("nan")
    logger.info("[%s] val_loss (20-batch hook) = %.4f", skill, val_loss_mean)
    policy.train()

    # --- write per-skill progress record ------------------------------------
    if progress is not None:
        progress["skills"][skill] = {
            "status":     "done",
            "first_loss": round(first_loss, 4),
            "mid_loss":   round(mid_loss, 4),
            "final_loss": round(final_loss, 4),
            "val_loss":   round(val_loss_mean, 4),
            "total_steps": total_steps,
        }
        _save_progress(progress)
        logger.info("[%s] progress written to %s", skill, PROGRESS_FILE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune GrootPolicy on DIA BC data")
    p.add_argument("--epochs",     type=int,   default=3,    help="Epochs per skill")
    p.add_argument("--seq_len",    type=int,   default=16,   help="Sequence length")
    p.add_argument("--batch_size", type=int,   default=2,    help="Batch size")
    p.add_argument("--lr",         type=float, default=5e-5, help="Learning rate")
    p.add_argument("--seed",       type=int,   default=42,   help="Global RNG seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    logger.info(
        "Config: epochs=%d  seq_len=%d  batch_size=%d  lr=%g  seed=%d",
        args.epochs, args.seq_len, args.batch_size, args.lr, args.seed,
    )

    # Global seed before anything
    set_seed(args.seed)

    # Load / create progress tracker
    progress = _load_progress()
    progress["config"] = {
        "epochs": args.epochs, "seq_len": args.seq_len,
        "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
    }
    _save_progress(progress)

    # Build action map
    logger.info("Building DIA→VPT action map …")
    dia_to_buttons, dia_to_camera = build_dia_to_vpt_map()

    # Load GROOT policy
    logger.info("Loading GrootPolicy from HF cache …")
    from minestudio.models.groot_one.body import load_groot_policy
    policy = load_groot_policy()
    policy = policy.to(device)

    # Freeze everything first
    for p in policy.parameters():
        p.requires_grad = False

    # Unfreeze trainable modules
    for mod_name in ("decoder", "pi_head", "value_head"):
        mod = getattr(policy, mod_name, None)
        if mod is not None:
            unfreeze_module(mod)
            logger.info("Unfrozen: %s", mod_name)
        else:
            logger.warning("Module not found on policy: %s", mod_name)

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    logger.info("Trainable params: %d / %d (%.1f%%)", trainable, total,
                100.0 * trainable / total)

    # Train each skill sequentially; skip already-done skills (resume support)
    for skill in SKILLS:
        if progress["skills"].get(skill, {}).get("status") == "done":
            logger.info("Skipping %s — already in progress file.", skill)
            continue
        try:
            train_skill(
                policy=policy,
                skill=skill,
                dia_to_buttons=dia_to_buttons,
                dia_to_camera=dia_to_camera,
                epochs=args.epochs,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                seed=args.seed,
                progress=progress,
            )
        except FileNotFoundError as exc:
            logger.warning("Skipping skill %s: %s", skill, exc)
            progress["skills"][skill] = {"status": "skipped", "reason": str(exc)}
            _save_progress(progress)
        except torch.cuda.OutOfMemoryError:
            logger.error(
                "CUDA OOM on skill=%s  batch_size=%d — try --batch_size 1",
                skill, args.batch_size,
            )
            torch.cuda.empty_cache()
            progress["skills"][skill] = {"status": "oom"}
            _save_progress(progress)

    # Save combined checkpoint
    out_dir = REPO_ROOT / "data" / "groot_ft"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "groot_ft.pt"

    logger.info("Saving fine-tuned checkpoint to %s …", ckpt_path)
    torch.save(policy.state_dict(), str(ckpt_path))

    ckpt_size_mb = ckpt_path.stat().st_size / (1024 ** 2)
    progress["checkpoint"] = {"path": str(ckpt_path), "size_mb": round(ckpt_size_mb, 1)}
    progress["status"] = "complete"
    _save_progress(progress)

    logger.info("Checkpoint saved: %.1f MB", ckpt_size_mb)
    logger.info("Fine-tuning complete.")


if __name__ == "__main__":
    main()
