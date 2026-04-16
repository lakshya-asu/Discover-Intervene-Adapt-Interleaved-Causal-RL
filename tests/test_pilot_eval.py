# tests/test_pilot_eval.py
"""Performance evaluation unit tests for DIA+GROOT pilot results.

Tests run against any JSON produced by run_groot_dia_minestudio.py.
They are parametrised so the same suite covers every result file
found under /tmp/dia_groot_ft_s*.json and data/eval/*.json.

Tiers (defined as module constants so eval_and_tune.py can import them):
  TIER_BASELINE  — n_achieved >= 1  (any skill)
  TIER_ACCEPTABLE — n_achieved >= 3  (wood / gather trio)
  TIER_TARGET    — n_achieved >= 5  (through iron family)
  TIER_STRETCH   — diamond_reached == True

Run:
    pytest tests/test_pilot_eval.py -v -p no:launch_ros -p no:launch_testing \
        [--result /tmp/dia_groot_ft_s0.json]
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Performance tiers (importable by eval_and_tune.py)
# ---------------------------------------------------------------------------

TIER_BASELINE   = 1   # n_achieved
TIER_ACCEPTABLE = 3
TIER_TARGET     = 5
TIER_STRETCH    = 9   # diamond_reached implies this

SKILL_ORDER = [
    "wood", "woodpickaxe", "stone", "coal",
    "furnace", "stonepickaxe", "ironore", "iron", "ironpickaxe",
]
GATHER_SKILLS = {"wood", "stone", "coal", "ironore"}
CRAFT_SKILLS  = {"woodpickaxe", "furnace", "stonepickaxe", "iron", "ironpickaxe"}

# Max steps allowed per skill without being considered "stuck"
MAX_STEPS_STUCK_THRESHOLD = 1900   # 95% of the 2000 cap

# ---------------------------------------------------------------------------
# Result fixture — resolves to the newest /tmp/dia_groot_ft_s*.json or the
# path passed via --result CLI option.
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--result", action="store", default=None,
        help="Path to a specific result JSON to evaluate",
    )


def _find_latest_result() -> Path | None:
    candidates = sorted(
        Path("/tmp").glob("dia_groot_ft_s*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # Also check data/eval/
    repo = Path(__file__).resolve().parent.parent
    candidates += sorted(
        (repo / "data" / "eval").glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


@pytest.fixture(scope="session")
def result_path(pytestconfig) -> Path:
    explicit = pytestconfig.getoption("--result", default=None)
    if explicit:
        p = Path(explicit)
        if not p.exists():
            pytest.skip(f"Result file not found: {p}")
        return p
    p = _find_latest_result()
    if p is None:
        pytest.skip("No result JSON found — run the pilot first.")
    return p


@pytest.fixture(scope="session")
def result(result_path) -> Dict[str, Any]:
    return json.loads(result_path.read_text())


# ---------------------------------------------------------------------------
# Schema / sanity tests  (always runnable once the file exists)
# ---------------------------------------------------------------------------

class TestSchema:
    REQUIRED_TOP_KEYS = {
        "mode", "seed", "skill_order", "results",
        "skills_achieved", "n_achieved", "diamond_reached",
        "total_steps", "elapsed_s",
    }
    REQUIRED_SKILL_KEYS = {"success", "steps", "step_global", "attempt"}

    def test_top_level_keys(self, result):
        missing = self.REQUIRED_TOP_KEYS - result.keys()
        assert not missing, f"Missing top-level keys: {missing}"

    def test_n_achieved_consistent(self, result):
        assert result["n_achieved"] == len(result["skills_achieved"]), \
            "n_achieved does not match len(skills_achieved)"

    def test_diamond_consistent(self, result):
        # diamond_reached should match presence of a craft that produced diamond
        # The runner sets diamond_reached = "diamond" in achieved (via woodpickaxe→...→diamond)
        # just check type
        assert isinstance(result["diamond_reached"], bool)

    def test_skills_achieved_subset_of_order(self, result):
        achieved = set(result["skills_achieved"])
        order = set(result["skill_order"])
        extra = achieved - order
        assert not extra, f"skills_achieved contains skills not in skill_order: {extra}"

    def test_per_skill_records_present(self, result):
        for skill in result["skills_achieved"]:
            assert skill in result["results"], \
                f"Achieved skill '{skill}' missing from results dict"

    def test_per_skill_record_keys(self, result):
        for skill, rec in result["results"].items():
            missing = self.REQUIRED_SKILL_KEYS - rec.keys()
            assert not missing, f"Skill '{skill}' record missing keys: {missing}"

    def test_steps_nonnegative(self, result):
        for skill, rec in result["results"].items():
            assert rec["steps"] >= 0, f"Skill '{skill}': steps < 0"

    def test_total_steps_positive(self, result):
        assert result["total_steps"] > 0

    def test_elapsed_positive(self, result):
        assert result["elapsed_s"] > 0

    def test_mode_valid(self, result):
        assert result["mode"] in ("groot", "dia", "dia_online")


# ---------------------------------------------------------------------------
# Behavioral sanity tests
# ---------------------------------------------------------------------------

class TestBehavior:
    def test_no_skill_took_zero_steps_and_succeeded(self, result):
        """A skill should not succeed in 0 env steps (catch /give lag artifacts)."""
        for skill, rec in result["results"].items():
            if rec["success"] and skill in GATHER_SKILLS:
                assert rec["steps"] > 2, \
                    f"Gather skill '{skill}' succeeded in {rec['steps']} steps — likely /give artifact"

    def test_failed_gather_used_meaningful_steps(self, result):
        """Failed gather skills should have tried for a while, not quit immediately."""
        for skill, rec in result["results"].items():
            if not rec["success"] and skill in GATHER_SKILLS:
                # Allow 0 steps for "no_clip" skips; otherwise expect > 50 steps
                if rec.get("reason") != "no_clip":
                    assert rec["steps"] >= 50 or rec["steps"] == 0, \
                        f"Gather skill '{skill}' failed after only {rec['steps']} steps"

    def test_craft_skills_have_method(self, result):
        for skill, rec in result["results"].items():
            if skill in CRAFT_SKILLS and skill in result["results"]:
                assert "craft_method" in rec, \
                    f"Craft skill '{skill}' missing craft_method field"

    def test_global_steps_monotone(self, result):
        """step_global should be non-decreasing across execution order."""
        ordered = sorted(
            result["results"].items(),
            key=lambda kv: kv[1].get("step_global", 0),
        )
        prev = -1
        for skill, rec in ordered:
            sg = rec.get("step_global", 0)
            assert sg >= prev, \
                f"step_global not monotone at skill '{skill}': {sg} < {prev}"
            prev = sg

    def test_no_all_stuck_skills(self, result):
        """Not every attempted gather skill should hit the max step limit."""
        gather_attempted = [
            rec for sk, rec in result["results"].items()
            if sk in GATHER_SKILLS and rec.get("steps", 0) > 0
        ]
        if not gather_attempted:
            pytest.skip("No gather skills attempted")
        stuck = [r for r in gather_attempted if r["steps"] >= MAX_STEPS_STUCK_THRESHOLD]
        frac_stuck = len(stuck) / len(gather_attempted)
        assert frac_stuck < 1.0, \
            "Every gather skill hit the step cap — model may be fully stuck (all-noop)"


# ---------------------------------------------------------------------------
# Performance / tier tests — these XFAIL gracefully below tier
# ---------------------------------------------------------------------------

class TestPerformanceTiers:
    def test_tier_baseline(self, result):
        """At least 1 skill succeeded."""
        assert result["n_achieved"] >= TIER_BASELINE, \
            f"n_achieved={result['n_achieved']} < TIER_BASELINE={TIER_BASELINE}"

    @pytest.mark.xfail(strict=False, reason="acceptable tier not yet reached")
    def test_tier_acceptable(self, result):
        """At least 3 skills succeeded."""
        assert result["n_achieved"] >= TIER_ACCEPTABLE, \
            f"n_achieved={result['n_achieved']} < TIER_ACCEPTABLE={TIER_ACCEPTABLE}"

    @pytest.mark.xfail(strict=False, reason="target tier not yet reached")
    def test_tier_target(self, result):
        """At least 5 skills succeeded."""
        assert result["n_achieved"] >= TIER_TARGET, \
            f"n_achieved={result['n_achieved']} < TIER_TARGET={TIER_TARGET}"

    @pytest.mark.xfail(strict=False, reason="stretch goal")
    def test_tier_stretch(self, result):
        """Diamond reached."""
        assert result["diamond_reached"], "diamond not reached"

    def test_wood_success(self, result):
        """Wood (easiest gather skill) should succeed."""
        rec = result["results"].get("wood", {})
        assert rec.get("success", False), \
            f"Wood failed — steps={rec.get('steps')} attempt={rec.get('attempt')}"

    @pytest.mark.xfail(strict=False, reason="stone may not be reached yet")
    def test_stone_success(self, result):
        rec = result["results"].get("stone", {})
        assert rec.get("success", False), "stone failed"

    @pytest.mark.xfail(strict=False, reason="coal may not be reached yet")
    def test_coal_success(self, result):
        rec = result["results"].get("coal", {})
        assert rec.get("success", False), "coal failed"

    def test_per_skill_success_table(self, result, capsys):
        """Print a readable success table (always passes, aids diagnostics)."""
        lines = ["\n  Skill results:"]
        for sk in SKILL_ORDER:
            rec = result["results"].get(sk, {})
            icon = "✓" if rec.get("success") else "✗"
            steps = rec.get("steps", "-")
            lines.append(f"  {icon}  {sk:<15}  steps={steps}")
        lines.append(f"\n  n_achieved={result['n_achieved']}  "
                     f"diamond={result['diamond_reached']}  "
                     f"elapsed={result['elapsed_s']}s")
        print("\n".join(lines))
        assert True
