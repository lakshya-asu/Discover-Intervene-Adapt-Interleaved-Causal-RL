#!/usr/bin/env python3
"""
Smoke test for ROCKET-1 (MineStudio) inference.

Checks:
  1. minestudio is importable
  2. ROCKET-1 model loads from HuggingFace
  3. Dummy inference runs without error
  4. Action output is a dict with MineDojo-compatible keys

Does NOT require a running Minecraft server.

Run:
    conda run -n dia-minecraft python scripts/test_rocket1_inference.py
"""
from __future__ import annotations

import sys
import os
import numpy as np

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = PASS if ok else FAIL
    line = f"  [{mark}] {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return ok


def main() -> int:
    print("=" * 60)
    print("ROCKET-1 / MineStudio smoke test")
    print("=" * 60)

    results: list[bool] = []

    # ------------------------------------------------------------------
    # 1. minestudio importable
    # ------------------------------------------------------------------
    print("\n[1] minestudio package")
    try:
        import minestudio
        ver = getattr(minestudio, "__version__", "unknown version")
        results.append(check("import minestudio", True, ver))
    except ImportError as e:
        results.append(check("import minestudio", False, str(e)))
        print("     Fix: conda run -n dia-minecraft pip install minestudio")
        # Can't continue without the package
        _print_summary(results)
        return 1

    # ------------------------------------------------------------------
    # 2. ROCKET-1 adapter import
    # ------------------------------------------------------------------
    print("\n[2] DIA options_rocket1 adapter")
    try:
        from dia.options_rocket1 import _load_rocket1, ROCKET1GatherOption, is_rocket1_available
        results.append(check("import options_rocket1", True))
    except ImportError as e:
        results.append(check("import options_rocket1", False, str(e)))
        _print_summary(results)
        return 1

    # ------------------------------------------------------------------
    # 3. Model load
    # ------------------------------------------------------------------
    print("\n[3] ROCKET-1 model load (CraftJarvis/ROCKET-1)")
    policy = _load_rocket1()
    ok = policy is not None
    results.append(check(
        "ROCKET1Policy.from_pretrained()",
        ok,
        type(policy).__name__ if ok else "returned None — check warning log above",
    ))
    if not ok:
        _print_summary(results)
        return 1

    # ------------------------------------------------------------------
    # 4. Dummy inference — (1, 3, 224, 224) float32
    # ------------------------------------------------------------------
    print("\n[4] Dummy inference  image=(224,224,3) uint8 + full-frame mask")
    import torch
    dummy_rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    dummy_mask = np.ones((224, 224), dtype=np.uint8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    memory = policy.initial_state()
    rocket_input = {
        "image": dummy_rgb,
        "segment": {
            "obj_id":   torch.tensor([2], dtype=torch.int64).to(device),
            "obj_mask": torch.tensor(dummy_mask, dtype=torch.uint8).to(device),
        },
    }
    try:
        action, memory = policy.get_action(
            input=rocket_input, state_in=memory, input_shape="*", deterministic=False
        )
        results.append(check("get_action() returns without error", True))
    except Exception as e:
        import traceback
        traceback.print_exc()
        results.append(check("get_action() returns without error", False, str(e)))
        _print_summary(results)
        return 1

    # ------------------------------------------------------------------
    # 5. Action format
    # ------------------------------------------------------------------
    print("\n[5] Action output format")
    is_dict = isinstance(action, dict)
    results.append(check("action is dict", is_dict, str(type(action))))
    if is_dict:
        print(f"       keys: {sorted(action.keys())}")
        # Expect MineDojo action keys
        expected_keys = {"forward", "back", "left", "right", "jump", "attack",
                         "use", "camera", "sprint"}
        overlap = expected_keys & set(action.keys())
        results.append(check(
            "action has MineDojo keys",
            len(overlap) >= 3,
            f"{len(overlap)}/{len(expected_keys)} expected keys found: {sorted(overlap)}",
        ))
    else:
        print(f"       action value: {action!r}")
        results.append(False)

    # ------------------------------------------------------------------
    # 6. Recurrent state preserved
    # ------------------------------------------------------------------
    print("\n[6] Recurrent hidden state")
    results.append(check(
        "hidden state returned",
        hidden is not None,
        f"type={type(hidden).__name__}",
    ))

    _print_summary(results)
    return 0 if all(results) else 1


def _print_summary(results: list[bool]) -> None:
    n_pass = sum(results)
    n_total = len(results)
    print("\n" + "=" * 60)
    if n_pass == n_total:
        print(f"\033[92mOVERALL: PASS ({n_pass}/{n_total})\033[0m")
        print("ROCKET-1 is ready. Run the transfer experiment with:")
        print("  conda run -n dia-minecraft python scripts/run_transfer_minedojo.py \\")
        print("    --mode transfer --backbone rocket1 --seed 0 --dry_run --out /tmp/dry.json")
    else:
        print(f"\033[91mOVERALL: FAIL ({n_pass}/{n_total} passed)\033[0m")
        print("Fix the failures above before running the transfer experiment.")
        print("Fallback: use --backbone bc with pre-trained BC policies.")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
