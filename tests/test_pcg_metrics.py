import sys, numpy as np
sys.path.insert(0, 'src')
import pytest
from dia.eval.pcg_metrics import shd, ece, pcg_summary

# 3-variable example: 0->1, 1->2
PROBS_PERFECT = np.array([
    [0.0, 0.9, 0.1],
    [0.1, 0.0, 0.9],
    [0.1, 0.1, 0.0],
], dtype=float)
TRUE_EDGES = [(0, 1), (1, 2)]

def test_shd_perfect():
    assert shd(PROBS_PERFECT, TRUE_EDGES) == 0

def test_shd_one_missing_edge():
    p = PROBS_PERFECT.copy()
    p[1, 2] = 0.1  # drop edge 1->2
    assert shd(p, TRUE_EDGES) == 1

def test_shd_one_extra_edge():
    p = PROBS_PERFECT.copy()
    p[0, 2] = 0.9  # add spurious edge 0->2
    assert shd(p, TRUE_EDGES) == 1

def test_shd_zero_edges():
    p = np.zeros((3, 3))
    assert shd(p, []) == 0

def test_shd_diagonal_ignored():
    p = PROBS_PERFECT.copy()
    p[0, 0] = 0.99  # diagonal should not count
    assert shd(p, TRUE_EDGES) == 0

def test_ece_perfect_calibration():
    # Edge 0->1 exists, predict 1.0; edge 1->2 exists, predict 1.0; all others 0.0
    p = np.zeros((3, 3))
    p[0, 1] = 1.0
    p[1, 2] = 1.0
    assert ece(p, TRUE_EDGES) < 0.01

def test_ece_worst_calibration():
    # Predict 1.0 for non-edges, 0.0 for true edges
    p = np.ones((3, 3)) - np.eye(3)
    p[0, 1] = 0.0
    p[1, 2] = 0.0
    val = ece(p, TRUE_EDGES)
    assert val > 0.5

def test_pcg_summary_keys():
    result = pcg_summary(PROBS_PERFECT, TRUE_EDGES,
                         var_names=['a', 'b', 'c'])
    assert 'shd' in result
    assert 'ece' in result
    assert 'false_positives' in result
    assert 'false_negatives' in result
    assert result['shd'] == 0
