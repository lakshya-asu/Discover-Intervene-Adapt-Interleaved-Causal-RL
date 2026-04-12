"""
Usage:
  python scripts/eval_pcg_shd.py --checkpoint path/to/checkpoint.pt --env 2d_minecraft
  python scripts/eval_pcg_shd.py --checkpoint path/to/checkpoint.pt --env crafter
"""
import argparse, sys, numpy as np
sys.path.insert(0, 'src')
from dia.eval.pcg_metrics import pcg_summary

# Ground-truth edges for each environment (variable indices, see dev-plan.md)
# 2D Minecraft: 9 vars [wood, stone, coal, ironore, furnace, stonepickaxe, iron, ironpickaxe, diamond]
GROUND_TRUTH_2D = [
    (0, 5), (1, 5),           # wood, stone -> stonepickaxe
    (1, 4),                    # stone -> furnace
    (2, 6), (3, 6), (4, 6),   # coal, ironore, furnace -> iron
    (6, 7), (0, 7),            # iron, wood -> ironpickaxe
    (7, 8),                    # ironpickaxe -> diamond
]
VAR_NAMES_2D = ['wood', 'stone', 'coal', 'ironore', 'furnace',
                'stonepickaxe', 'iron', 'ironpickaxe', 'diamond']

def load_probs(checkpoint_path: str) -> np.ndarray:
    """Load PCG probability matrix from a checkpoint file."""
    import torch
    obj = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    # Handle various storage formats
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, 'probs'):
        p = obj.probs
        return p.numpy() if hasattr(p, 'numpy') else np.array(p)
    if isinstance(obj, dict):
        for key in ('probs', 'phi', 'edge_probs', 'pcg_probs'):
            if key in obj:
                p = obj[key]
                return p.numpy() if hasattr(p, 'numpy') else np.array(p)
    raise ValueError(f"Cannot extract probability matrix from checkpoint. Keys: "
                     f"{list(obj.keys()) if isinstance(obj, dict) else type(obj)}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate PCG accuracy vs ground truth')
    parser.add_argument('--checkpoint', required=True, help='Path to PCG checkpoint')
    parser.add_argument('--env', required=True, choices=['2d_minecraft', 'crafter'],
                        help='Environment (determines ground-truth edges)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for binarization')
    args = parser.parse_args()

    probs = load_probs(args.checkpoint)
    print(f"PCG shape: {probs.shape}")

    if args.env == '2d_minecraft':
        true_edges = GROUND_TRUTH_2D
        var_names = VAR_NAMES_2D
    else:  # crafter
        from dia.evgs_crafter import CRAFTER_CAUSAL_EDGES
        true_edges = CRAFTER_CAUSAL_EDGES
        var_names = None

    result = pcg_summary(probs, true_edges, var_names=var_names,
                         threshold=args.threshold)
    print(f"SHD:  {result['shd']}")
    print(f"ECE:  {result['ece']:.4f}")
    if 'false_positives' in result and result['false_positives']:
        print(f"False positive edges: {result['false_positives']}")
    if 'false_negatives' in result and result['false_negatives']:
        print(f"False negative edges (missed): {result['false_negatives']}")

if __name__ == '__main__':
    main()
