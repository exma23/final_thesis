"""
Debug: visualize raw and normalized SPR features as heatmaps.
Usage: python debug.py
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import common
from feat_cpp.fastcpp import FastCpp
from utils import load_data_indices, get_tree_indices, normalize_features
from common import FEAT_NAMES


def main():
    tree_indices = get_tree_indices(common.DATA_PATH)
    tree_start, tree_gt = load_data_indices(common.DATA_PATH, tree_indices)
    cpp = FastCpp()

    # Pick first tree
    idx = tree_indices[0]
    newick = tree_start[idx]
    gt = tree_gt[idx]

    _, actions, feats, rewards = cpp.get_state_action(
        newick, -1, gt, spr_radius=7)

    n_actions = feats.shape[0]
    print(f"Tree {idx}: {n_actions} SPR actions, feats shape = {feats.shape}")

    X_raw = torch.tensor(feats, dtype=torch.float32)
    X_norm = normalize_features(X_raw.clone())

    # ── Sort actions by reward (best on top) ──
    order = np.argsort(rewards)[::-1].copy()
    X_raw_sorted = X_raw[order]
    X_norm_sorted = X_norm[order]
    rewards_sorted = rewards[order]

    # ── Limit to top-100 for readability ──
    # N = min(100, n_actions)
    # X_raw_show = X_raw_sorted[:N].numpy()
    # X_norm_show = X_norm_sorted[:N].numpy()
    # r_show = rewards_sorted[:N]
    X_raw_show = X_raw_sorted.numpy()
    X_norm_show = X_norm_sorted.numpy()
    r_show = rewards_sorted
    N = n_actions

    fig, axes = plt.subplots(1, 2, figsize=(20, 12))

    # Raw features
    im0 = axes[0].imshow(X_raw_show, aspect='auto', cmap='viridis')
    axes[0].set_title(f'Raw features (top-{N} by reward)')
    axes[0].set_xlabel('Feature index')
    axes[0].set_ylabel('Action (sorted by reward)')
    axes[0].set_xticks(range(19))
    axes[0].set_xticklabels(FEAT_NAMES, rotation=90, fontsize=7)
    fig.colorbar(im0, ax=axes[0], shrink=0.6)

    # Normalized features
    im1 = axes[1].imshow(X_norm_show, aspect='auto', cmap='viridis')
    axes[1].set_title(f'Normalized features (top-{N} by reward)')
    axes[1].set_xlabel('Feature index')
    axes[1].set_ylabel('Action (sorted by reward)')
    axes[1].set_xticks(range(19))
    axes[1].set_xticklabels(FEAT_NAMES, rotation=90, fontsize=7)
    fig.colorbar(im1, ax=axes[1], shrink=0.6)

    plt.tight_layout()
    plt.savefig('debug_features_heatmap.png', dpi=150)
    print('Saved -> debug_features_heatmap.png')
    plt.show()

    # ── Per-feature stats ──
    print(f"\n{'Feature':<18s} {'raw min':>10s} {'raw max':>10s} {'raw mean':>10s} {'raw var':>10s}"
          f" | {'norm min':>10s} {'norm max':>10s} {'norm mean':>10s} {'norm var':>10s}")
    print("-" * 110)
    for i, name in enumerate(FEAT_NAMES):
        raw_col = X_raw[:, i].numpy()
        norm_col = X_norm[:, i].numpy()
        print(f"{name:<18s} {raw_col.min():10.4f} {raw_col.max():10.4f} {raw_col.mean():10.4f} {raw_col.var():10.4f}"
              f" | {norm_col.min():10.4f} {norm_col.max():10.4f} {norm_col.mean():10.4f} {norm_col.var():10.4f}")

    # ── Reward stats ──
    print(f"\nReward: min={rewards.min():.4f}  max={rewards.max():.4f}  "
          f"mean={rewards.mean():.4f}  >0: {(rewards > 0).sum()}/{n_actions}")


if __name__ == '__main__':
    main()
