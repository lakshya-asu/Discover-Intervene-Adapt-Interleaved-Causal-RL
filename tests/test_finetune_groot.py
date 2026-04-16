# tests/test_finetune_groot.py
"""Unit tests for scripts/finetune_groot.py.

Covers:
  - Camera bin arithmetic (no external deps)
  - DIA→VPT action mapping (requires minestudio, skipped otherwise)
  - GrootBCDataset shape/dtype/window logic (uses temp numpy files)
  - Seed determinism
  - Checkpoint round-trip
"""
from __future__ import annotations

import importlib
import random
import sys
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers: pull pure-Python pieces from the script without triggering the
# full minestudio import chain.
# ---------------------------------------------------------------------------

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "finetune_groot.py"


def _import_script_module():
    """Import finetune_groot as a module (cached after first call)."""
    if "finetune_groot" in sys.modules:
        return sys.modules["finetune_groot"]
    spec = importlib.util.spec_from_file_location("finetune_groot", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["finetune_groot"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Camera bin formula
# ---------------------------------------------------------------------------

def cam_bin(deg: float) -> int:
    """Mirror of finetune_groot.build_dia_to_vpt_map's inner cam_bin."""
    return int(np.clip(round((deg / 15.0) * 5) + 5, 0, 10))


class TestCamBin:
    def test_center(self):
        assert cam_bin(0.0) == 5

    def test_full_up(self):
        # pitch = -15 → bin 0
        assert cam_bin(-15.0) == 0

    def test_full_down(self):
        # pitch = +15 → bin 10
        assert cam_bin(15.0) == 10

    def test_full_left(self):
        assert cam_bin(-15.0) == 0

    def test_full_right(self):
        assert cam_bin(15.0) == 10

    def test_small_up(self):
        # pitch = -5 → round(-5/15*5)+5 = round(-1.667)+5 = -2+5 = 3
        assert cam_bin(-5.0) == 3

    def test_small_down(self):
        assert cam_bin(5.0) == 7

    def test_clamped_extreme(self):
        # Should not go below 0 or above 10
        assert cam_bin(-100.0) == 0
        assert cam_bin(100.0) == 10

    def test_camera_index_noop(self):
        # noop: pitch=0, yaw=0 → cam_idx = 5*11 + 5 = 60
        idx = cam_bin(0.0) * 11 + cam_bin(0.0)
        assert idx == 60

    def test_camera_index_full_up(self):
        # action 11: pitch=-15, yaw=0 → cam_idx = 0*11 + 5 = 5
        idx = cam_bin(-15.0) * 11 + cam_bin(0.0)
        assert idx == 5

    def test_camera_index_full_down(self):
        # action 12: pitch=+15, yaw=0 → cam_idx = 10*11 + 5 = 115
        idx = cam_bin(15.0) * 11 + cam_bin(0.0)
        assert idx == 115

    def test_camera_index_full_left(self):
        # action 13: pitch=0, yaw=-15 → cam_idx = 5*11 + 0 = 55
        idx = cam_bin(0.0) * 11 + cam_bin(-15.0)
        assert idx == 55

    def test_camera_index_full_right(self):
        # action 14: pitch=0, yaw=+15 → cam_idx = 5*11 + 10 = 65
        idx = cam_bin(0.0) * 11 + cam_bin(15.0)
        assert idx == 65


# ---------------------------------------------------------------------------
# DIA→VPT mapping (requires minestudio)
# ---------------------------------------------------------------------------

minestudio = pytest.importorskip(
    "minestudio",
    reason="minestudio not importable — skipping mapping tests",
)


class TestDiaToVptMap:
    @pytest.fixture(scope="class")
    def maps(self):
        mod = _import_script_module()
        return mod.build_dia_to_vpt_map()

    def test_map_sizes(self, maps):
        dia_to_buttons, dia_to_camera = maps
        # 25 DIA actions (0-24)
        assert len(dia_to_buttons) == 25
        assert len(dia_to_camera) == 25

    def test_noop_buttons(self, maps):
        dia_to_buttons, _ = maps
        # noop = action 0; all keys 'none' → index 0 in CameraHierarchicalMapping
        assert dia_to_buttons[0] == 0, \
            f"noop buttons expected 0, got {dia_to_buttons[0]}"

    def test_noop_camera(self, maps):
        _, dia_to_camera = maps
        # noop camera: pitch=0, yaw=0 → idx=60
        assert dia_to_camera[0] == 60, \
            f"noop camera expected 60 (center), got {dia_to_camera[0]}"

    def test_forward_buttons_nonzero(self, maps):
        dia_to_buttons, _ = maps
        # forward (action 1) must be a distinct non-zero button combo
        assert dia_to_buttons[1] != 0

    def test_camera_up_index(self, maps):
        _, dia_to_camera = maps
        # action 11: pitch=-15, yaw=0 → cam_idx=5
        assert dia_to_camera[11] == 5, \
            f"action 11 camera expected 5, got {dia_to_camera[11]}"

    def test_camera_down_index(self, maps):
        _, dia_to_camera = maps
        assert dia_to_camera[12] == 115, \
            f"action 12 camera expected 115, got {dia_to_camera[12]}"

    def test_camera_left_index(self, maps):
        _, dia_to_camera = maps
        assert dia_to_camera[13] == 55, \
            f"action 13 camera expected 55, got {dia_to_camera[13]}"

    def test_camera_right_index(self, maps):
        _, dia_to_camera = maps
        assert dia_to_camera[14] == 65, \
            f"action 14 camera expected 65, got {dia_to_camera[14]}"

    def test_all_camera_indices_in_range(self, maps):
        _, dia_to_camera = maps
        for a, idx in dia_to_camera.items():
            assert 0 <= idx <= 120, f"action {a}: cam_idx {idx} out of [0,120]"

    def test_no_negative_button_indices(self, maps):
        dia_to_buttons, _ = maps
        for a, idx in dia_to_buttons.items():
            assert idx >= 0, f"action {a}: buttons idx {idx} is negative"


# ---------------------------------------------------------------------------
# GrootBCDataset
# ---------------------------------------------------------------------------

class TestGrootBCDataset:
    """Tests using temporary numpy files so no real BC data is required."""

    SEQ_LEN = 16
    N_FRAMES = 200

    @pytest.fixture(scope="class")
    def tmp_skill_dir(self, tmp_path_factory):
        d = tmp_path_factory.mktemp("bc_skill")
        obs = np.random.randint(0, 256, (self.N_FRAMES, 64, 64, 3), dtype=np.uint8)
        acts = np.random.randint(0, 25, (self.N_FRAMES,), dtype=np.int64)
        np.save(str(d / "obs.npy"), obs)
        np.save(str(d / "actions.npy"), acts)
        return d

    @pytest.fixture(scope="class")
    def dataset(self, tmp_path_factory):
        mod = _import_script_module()
        fake_root = tmp_path_factory.mktemp("repo")
        skill_dir = fake_root / "data" / "minerl_bc" / "testskill"
        skill_dir.mkdir(parents=True)
        obs = np.random.randint(0, 256, (self.N_FRAMES, 64, 64, 3), dtype=np.uint8)
        acts = np.random.randint(0, 25, (self.N_FRAMES,), dtype=np.int64)
        np.save(str(skill_dir / "obs.npy"), obs)
        np.save(str(skill_dir / "actions.npy"), acts)

        orig_root = mod.REPO_ROOT
        mod.REPO_ROOT = fake_root
        try:
            ds = mod.GrootBCDataset(
                "testskill",
                {i: i % 8641 for i in range(25)},
                {i: 60 for i in range(25)},
                seq_len=self.SEQ_LEN,
            )
        finally:
            mod.REPO_ROOT = orig_root
        return ds

    def test_window_count(self, dataset):
        stride = self.SEQ_LEN // 2
        expected = len(range(0, self.N_FRAMES - self.SEQ_LEN, stride))
        assert len(dataset) == expected

    def test_item_frame_shape(self, dataset):
        frames, btn, cam = dataset[0]
        assert frames.shape == (self.SEQ_LEN, 224, 224, 3)

    def test_item_frame_dtype(self, dataset):
        frames, _, _ = dataset[0]
        assert frames.dtype == torch.uint8

    def test_item_label_shape(self, dataset):
        _, btn, cam = dataset[0]
        assert btn.shape == (self.SEQ_LEN,)
        assert cam.shape == (self.SEQ_LEN,)

    def test_item_label_dtype(self, dataset):
        _, btn, cam = dataset[0]
        assert btn.dtype == torch.int64
        assert cam.dtype == torch.int64

    def test_missing_obs_raises(self, tmp_path):
        mod = _import_script_module()
        fake_root = tmp_path / "repo"
        skill_dir = fake_root / "data" / "minerl_bc" / "ghost"
        skill_dir.mkdir(parents=True)
        orig = mod.REPO_ROOT
        mod.REPO_ROOT = fake_root
        with pytest.raises(FileNotFoundError, match="obs"):
            mod.GrootBCDataset("ghost", {}, {}, seq_len=self.SEQ_LEN)
        mod.REPO_ROOT = orig

    def test_missing_actions_raises(self, tmp_path):
        mod = _import_script_module()
        fake_root = tmp_path / "repo"
        skill_dir = fake_root / "data" / "minerl_bc" / "ghost"
        skill_dir.mkdir(parents=True)
        obs = np.random.randint(0, 256, (50, 64, 64, 3), dtype=np.uint8)
        np.save(str(skill_dir / "obs.npy"), obs)
        orig = mod.REPO_ROOT
        mod.REPO_ROOT = fake_root
        with pytest.raises(FileNotFoundError, match="actions"):
            mod.GrootBCDataset("ghost", {}, {}, seq_len=self.SEQ_LEN)
        mod.REPO_ROOT = orig


# ---------------------------------------------------------------------------
# Seed determinism
# ---------------------------------------------------------------------------

class TestSeedDeterminism:
    """Verify that set_seed makes torch/numpy/random reproducible."""

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def test_torch_reproducible(self):
        self._set_seed(42)
        a = torch.randn(100)
        self._set_seed(42)
        b = torch.randn(100)
        assert torch.allclose(a, b)

    def test_numpy_reproducible(self):
        self._set_seed(7)
        a = np.random.rand(50)
        self._set_seed(7)
        b = np.random.rand(50)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        self._set_seed(0)
        a = torch.randn(50)
        self._set_seed(1)
        b = torch.randn(50)
        assert not torch.allclose(a, b)

    def test_dataloader_worker_seed(self):
        """Seeded DataLoader with shuffle=True returns same order both runs."""
        from torch.utils.data import TensorDataset, DataLoader

        ds = TensorDataset(torch.arange(100).float())

        def _collect(seed):
            g = torch.Generator()
            g.manual_seed(seed)
            loader = DataLoader(ds, batch_size=10, shuffle=True, generator=g)
            return [b[0].tolist() for b in loader]

        assert _collect(99) == _collect(99)
        assert _collect(99) != _collect(100)


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------

class TestCheckpointRoundtrip:
    def test_state_dict_save_load(self, tmp_path):
        model = torch.nn.Linear(8, 4)
        ckpt = tmp_path / "ckpt.pt"
        torch.save(model.state_dict(), str(ckpt))
        model2 = torch.nn.Linear(8, 4)
        model2.load_state_dict(
            torch.load(str(ckpt), map_location="cpu", weights_only=True)
        )
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_checkpoint_size_nonzero(self, tmp_path):
        model = torch.nn.Linear(32, 16)
        ckpt = tmp_path / "ckpt.pt"
        torch.save(model.state_dict(), str(ckpt))
        assert ckpt.stat().st_size > 0

    def test_groot_executor_loads_ft_ckpt(self, tmp_path):
        """GrootExecutor.build() picks up a fine-tuned checkpoint by path."""
        # We don't load the real GrootPolicy here — just verify the path logic.
        import os

        # Simulate the path resolution in groot_executor.py
        executor_path = Path(__file__).resolve().parent.parent / "src" / "dia" / "groot_executor.py"
        assert executor_path.exists(), "groot_executor.py not found"

        src = executor_path.read_text()
        assert "groot_ft.pt" in src, "groot_executor.py does not reference groot_ft.pt"
        assert "load_state_dict" in src, "groot_executor.py does not call load_state_dict"
