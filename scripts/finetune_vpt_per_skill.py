#!/usr/bin/env python3
"""Per-skill behavioural-cloning fine-tune of the OpenAI VPT foundation model.

Fine-tunes MinecraftAgentPolicy on contractor demo segments where a target
skill variable (inventory count) increases, using the VPT repo's own
action-conversion utilities.

Usage:
    # Dry-run: count available frames for the 'wood' skill
    python scripts/finetune_vpt_per_skill.py --skill wood --dry_run

    # Full fine-tune (GPU recommended)
    python scripts/finetune_vpt_per_skill.py --skill wood --steps 1000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# VPT repo on sys.path
# ---------------------------------------------------------------------------
_VPT_ROOT = Path("/home/flux/DIA/baselines/vpt")
sys.path.insert(0, str(_VPT_ROOT))

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skill definitions (mirrors filter_minerl_demos.py)
# ---------------------------------------------------------------------------
SKILLS: List[str] = [
    "wood", "stone", "coal", "ironore", "furnace",
    "stonepickaxe", "iron", "ironpickaxe", "diamond",
]
SKILL_VARS: Dict[str, List[str]] = {
    "wood":         ["oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log",
                     "dark_oak_log", "log", "log2", "oak_wood", "birch_wood",
                     "spruce_wood", "jungle_wood", "acacia_wood", "dark_oak_wood"],
    "stone":        ["cobblestone", "cobbled_deepslate", "stone"],
    "coal":         ["coal"],       "ironore":      ["iron_ore"],
    "furnace":      ["furnace"],    "stonepickaxe": ["stone_pickaxe"],
    "iron":         ["iron_ingot"], "ironpickaxe":  ["iron_pickaxe"],
    "diamond":      ["diamond"],
}

# VPT agent resolution (must match AGENT_RESOLUTION in agent.py)
_AGENT_RES = (128, 128)

# ---------------------------------------------------------------------------
# Contractor JSONL → MineRL env-format action
# ---------------------------------------------------------------------------
_KEY_MAP: Dict[str, str] = dict(zip(
    ["key.keyboard.escape", "key.keyboard.s", "key.keyboard.q", "key.keyboard.w",
     "key.keyboard.1", "key.keyboard.2", "key.keyboard.3", "key.keyboard.4",
     "key.keyboard.5", "key.keyboard.6", "key.keyboard.7", "key.keyboard.8",
     "key.keyboard.9", "key.keyboard.e", "key.keyboard.space", "key.keyboard.a",
     "key.keyboard.d", "key.keyboard.left.shift", "key.keyboard.left.control",
     "key.keyboard.f"],
    ["ESC", "back", "drop", "forward",
     "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4",
     "hotbar.5", "hotbar.6", "hotbar.7", "hotbar.8",
     "hotbar.9", "inventory", "jump", "left",
     "right", "sneak", "sprint", "swapHands"],
))
_CAM_SCALER = 360.0 / 2400.0
_NOOP = {k: 0 for k in
         ["ESC", "back", "drop", "forward", "hotbar.1", "hotbar.2", "hotbar.3",
          "hotbar.4", "hotbar.5", "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9",
          "inventory", "jump", "left", "right", "sneak", "sprint", "swapHands",
          "attack", "use", "pickItem"]}


def _json_to_env_action(step: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Convert one contractor JSONL step to a MineRL env-format action dict."""
    act = dict(_NOOP)
    act["camera"] = np.array([0.0, 0.0])
    null = True
    for key in step.get("keyboard", {}).get("keys", []):
        if key in _KEY_MAP:
            act[_KEY_MAP[key]] = 1
            null = False
    mouse = step.get("mouse", {})
    dx, dy = float(mouse.get("dx", 0.0)), float(mouse.get("dy", 0.0))
    act["camera"][:] = [dy * _CAM_SCALER, dx * _CAM_SCALER]
    if dx or dy:
        null = False
    for btn, name in [(0, "attack"), (1, "use"), (2, "pickItem")]:
        if btn in mouse.get("buttons", []):
            act[name] = 1
            null = False
    return act, null


def _inventory_count(step: Dict[str, Any], item_keys: List[str]) -> int:
    inv = step.get("inventory", [])
    if isinstance(inv, list):
        return sum(int(e.get("quantity", 0))
                   for e in inv if e.get("type", "") in item_keys)
    return sum(int(inv.get(k, 0)) for k in item_keys)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _load_frames(mp4_path: str) -> Optional[np.ndarray]:
    """Load MP4 as uint8 (N,H,W,3) at VPT resolution; None if unreadable."""
    p = Path(mp4_path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            _AGENT_RES, interpolation=cv2.INTER_LINEAR,
        ))
    cap.release()
    return np.stack(frames).astype(np.uint8) if frames else None


def iter_skill_frames(
    data_dir: str, skill: str, window: int = 100,
) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
    """Yield (frame uint8 (H,W,3), env_action_dict) for skill-window frames."""
    item_keys = SKILL_VARS[skill]
    pairs: List[Tuple[str, str]] = []
    for jf in sorted(Path(data_dir).rglob("*.jsonl")):
        if not jf.name.endswith(".mp4.jsonl"):
            pairs.append((str(jf), str(jf.with_suffix(".mp4"))))

    for jsonl_path, mp4_path in pairs:
        if not (Path(mp4_path).exists() and Path(mp4_path).stat().st_size > 0):
            logger.warning("Skipping episode (mp4 missing/0-bytes): %s", mp4_path)
            continue
        frames = _load_frames(mp4_path)
        if frames is None:
            logger.warning("Could not decode mp4: %s", mp4_path)
            continue
        steps = _load_jsonl(jsonl_path)
        n = min(len(frames), len(steps))
        frames, steps = frames[:n], steps[:n]
        env_actions = [_json_to_env_action(s) for s in steps]

        prev = _inventory_count(steps[0], item_keys)
        for t in range(1, n):
            curr = _inventory_count(steps[t], item_keys)
            if curr > prev:
                for idx in range(max(0, t - window), t):
                    act, is_null = env_actions[idx]
                    if not is_null:
                        yield frames[idx], act
            prev = curr


# ---------------------------------------------------------------------------
# VPT model loading (no MineRL env required)
# ---------------------------------------------------------------------------
def load_vpt_agent(model_path: str, weights_path: str, device: str):
    """Return a namespace with .policy, .action_mapper, .action_transformer, .device."""
    import torch as th
    from lib.policy import MinecraftAgentPolicy
    from lib.action_mapping import CameraHierarchicalMapping
    from lib.actions import ActionTransformer
    from lib.torch_util import set_default_torch_device
    from gym3.types import DictType

    AT_KWARGS = dict(camera_binsize=2, camera_maxval=10,
                     camera_mu=10, camera_quantization_scheme="mu_law")
    try:
        params = pickle.load(open(model_path, "rb"))
        pol_kw = params["model"]["args"]["net"]["args"]
        pi_kw = params["model"]["args"]["pi_head_opts"]
        pi_kw["temperature"] = float(pi_kw["temperature"])
    except Exception as exc:
        logger.warning("Could not read .model file (%s); using built-in defaults.", exc)
        pol_kw = dict(
            attention_heads=16, attention_mask_style="clipped_causal",
            attention_memory_size=256, diff_mlp_embedding=False, hidsize=2048,
            img_shape=[128, 128, 3], impala_chans=[16, 32, 32],
            impala_kwargs={"post_pool_groups": 1}, impala_width=8,
            init_norm_kwargs={"batch_norm": False, "group_norm_groups": 1},
            n_recurrence_layers=4, only_img_input=True, pointwise_ratio=4,
            pointwise_use_activation=False, recurrence_is_residual=True,
            recurrence_type="transformer", timesteps=128,
            use_pointwise_layer=True, use_pre_lstm_ln=False,
        )
        pi_kw = dict(temperature=2.0)

    dev = th.device(device if th.cuda.is_available() else "cpu")
    set_default_torch_device(dev)
    mapper = CameraHierarchicalMapping(n_camera_bins=11)
    policy = MinecraftAgentPolicy(
        action_space=DictType(**mapper.get_action_space_update()),
        policy_kwargs=pol_kw, pi_head_kwargs=pi_kw,
    ).to(dev)
    policy.load_state_dict(th.load(weights_path, map_location=dev), strict=False)

    class _A:
        pass
    a = _A()
    a.policy = policy
    a.action_mapper = mapper
    a.action_transformer = ActionTransformer(**AT_KWARGS)
    a.device = dev
    return a


def _to_agent_action(agent: Any, env_action: Dict[str, Any]) -> Optional[Dict]:
    import torch as th
    m = agent.action_transformer.env2policy(env_action)
    if (np.all(m["buttons"] == 0) and
            np.all(m["camera"] == agent.action_transformer.camera_zero_bin)):
        return None
    if m["camera"].ndim == 1:
        m = {k: v[None] for k, v in m.items()}
    a = agent.action_mapper.from_factored(m)
    return {k: th.from_numpy(v).to(agent.device) for k, v in a.items()}


# ---------------------------------------------------------------------------
# Fine-tune loop
# ---------------------------------------------------------------------------
def finetune(args: argparse.Namespace) -> None:
    import torch as th
    from lib.tree_util import tree_map

    device = "cuda" if th.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    logger.info("Loading VPT weights from %s …", args.weights)
    agent = load_vpt_agent(args.model, args.weights, device)
    policy = agent.policy

    # Freeze trunk; unfreeze pi_head + lastlayer + final_ln + last 2 xf blocks
    for p in policy.parameters():
        p.requires_grad_(False)
    for mod in [policy.pi_head, policy.net.lastlayer, policy.net.final_ln]:
        for p in mod.parameters():
            p.requires_grad_(True)
    if hasattr(policy.net, "recurrent_layer") and policy.net.recurrent_layer is not None:
        for block in list(policy.net.recurrent_layer.children())[-2:]:
            for p in block.parameters():
                p.requires_grad_(True)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    logger.info("Trainable params: %d", sum(p.numel() for p in trainable))

    optimizer = th.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    dummy_first = th.from_numpy(np.array((False,))).to(agent.device)
    hidden = policy.initial_state(1)
    data_iter: Iterator = (
        item for _ in iter(int, 1)
        for item in iter_skill_frames(args.data_dir, args.skill, window=100)
    )

    policy.train()
    loss_acc = 0.0
    for step in range(1, args.steps + 1):
        frames, actions = [], []
        while len(frames) < args.batch_size:
            try:
                frame, env_act = next(data_iter)
            except StopIteration:
                break
            ag_act = _to_agent_action(agent, env_act)
            if ag_act is not None:
                frames.append(frame)
                actions.append(ag_act)

        if not frames:
            logger.warning("No frames for skill '%s'. Stopping.", args.skill)
            return

        batch_loss = 0.0
        for frame, ag_act in zip(frames, actions):
            img = th.from_numpy(frame[None]).to(agent.device)
            pd, _v, new_hidden = policy.get_output_for_observation(
                {"img": img}, hidden, dummy_first)
            log_prob = policy.get_logprob_of_action(pd, ag_act)
            hidden = tree_map(lambda x: x.detach(), new_hidden)
            loss = -log_prob / args.batch_size
            batch_loss += loss.item()
            loss.backward()

        th.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        loss_acc += batch_loss

        if step % 100 == 0:
            logger.info("Step %d/%d  avg_loss=%.4f  lr=%.2e",
                        step, args.steps, loss_acc / 100,
                        scheduler.get_last_lr()[0])
            loss_acc = 0.0

    out_dir = Path(args.out_dir) / args.skill
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "finetuned.weights"
    th.save(policy.state_dict(), str(out_path))
    logger.info("Saved fine-tuned weights → %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Per-skill BC fine-tuning of the VPT foundation model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--skill", required=True, choices=SKILLS)
    p.add_argument("--model",
                   default="/home/flux/DIA/baselines/vpt/models/foundation-model-3x.model")
    p.add_argument("--weights",
                   default="/home/flux/DIA/baselines/vpt/models/foundation-model-3x.weights")
    p.add_argument("--data_dir",
                   default="/home/flux/DIA/baselines/vpt/data/contractor")
    p.add_argument("--out_dir",
                   default="/home/flux/DIA/baselines/vpt/models/finetuned/")
    p.add_argument("--steps",      type=int,   default=1000)
    p.add_argument("--lr",         type=float, default=1e-5)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--dry_run",    action="store_true",
                   help="Scan data, report frame count, exit without training.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    try:
        from lib.policy import MinecraftAgentPolicy  # noqa: F401
    except ImportError as exc:
        print(f"[ERROR] VPT repo not importable: {exc}")
        print("VPT repo not found. Run: bash scripts/setup_vpt_data.sh")
        sys.exit(1)

    logger.info("Skill: %s  |  data_dir: %s", args.skill, args.data_dir)

    if args.dry_run:
        logger.info("DRY RUN — scanning for frames (skill=%s) …", args.skill)
        item_keys = SKILL_VARS[args.skill]
        n_eps = n_wins = n_frames = 0
        for jf in sorted(Path(args.data_dir).rglob("*.jsonl")):
            if jf.name.endswith(".mp4.jsonl"):
                continue
            mp4 = jf.with_suffix(".mp4")
            if not (mp4.exists() and mp4.stat().st_size > 0):
                logger.warning("  SKIP (mp4 missing/0-bytes): %s", mp4)
                continue
            steps = _load_jsonl(str(jf))
            n_eps += 1
            prev = _inventory_count(steps[0], item_keys) if steps else 0
            for t in range(1, len(steps)):
                curr = _inventory_count(steps[t], item_keys)
                if curr > prev:
                    n_frames += min(t, 100)
                    n_wins += 1
                prev = curr
        print(f"\n[dry_run] skill={args.skill}")
        print(f"  episodes scanned : {n_eps}")
        print(f"  skill windows    : {n_wins}")
        print(f"  total frames     : {n_frames}")
        if n_frames == 0:
            print(f"  WARNING: No frames found for skill '{args.skill}'. "
                  "Check data_dir or re-download mp4 files.")
        return

    if not Path(args.weights).exists():
        logger.error("Weights file not found: %s", args.weights)
        sys.exit(1)
    if not Path(args.model).exists():
        logger.error("Model file not found: %s", args.model)
        sys.exit(1)

    finetune(args)


if __name__ == "__main__":
    main()
