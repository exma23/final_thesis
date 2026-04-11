#!/usr/bin/env python3
"""Greedy inference: load checkpoint, run SPR steps, plot likelihood trajectory."""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import torch

import common
import utils
from agents.agent import Agent
from env.environment import Environment

log = common.logger


def parse_args():
    parser = argparse.ArgumentParser(description="Phylogenetic RL Inference")
    parser.add_argument("--ckp-dir", required=True,
                        help="Checkpoint run directory (e.g. ckps/260411_002111)")
    parser.add_argument("--use-best", action="store_true",
                        help="Use model_best_*.pt instead of model_*.pt")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tree-ids", type=int, nargs="+", default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--out-dir", default="eval_results")
    return parser.parse_args()


def find_checkpoint(ckp_dir, use_best):
    """Find .pt file in ckp_dir and auto-detect loss_type from filename."""
    files = sorted(glob.glob(os.path.join(ckp_dir, "*.pt")))
    if use_best:
        matches = [f for f in files if "model_best_" in os.path.basename(f)]
    else:
        matches = [f for f in files if "model_best_" not in os.path.basename(f)]

    # Fallback: if preferred variant not found, use whatever is available
    if not matches:
        matches = files
    if not matches:
        raise FileNotFoundError(f"No .pt files in {ckp_dir}")

    ckp_path = matches[0]

    # Auto-detect loss_type from filename
    basename = os.path.basename(ckp_path)
    loss_type = "q_learning" if "q_learning" in basename else "reinforce"

    return ckp_path, loss_type


def greedy_rollout(agent, cpp, start_newick, gt_newick, msa_path,
                   n_steps, spr_radius, bl_opt_model):
    """Run n_steps greedy SPR moves, return logL at each step (length n_steps+1)."""
    cur_newick = start_newick

    # Initial logL
    _, cur_logL = utils.evaluate_loglikelihood_raxmlng(
        cur_newick, msa_path, model=bl_opt_model)
    ll_history = [cur_logL]

    # Initial features
    cur_newick, actions, feats, _ = cpp.get_state_action(
        cur_newick, -1, gt_newick, spr_radius=spr_radius)

    for step in range(n_steps):
        X = torch.tensor(feats, dtype=torch.float32, device=agent.device)
        X = utils.normalize_features(X)
        _, action_idx = agent.choose(actions, X)

        # Apply action → new tree
        next_newick, next_actions, next_feats, _ = cpp.get_state_action(
            cur_newick, action_idx, gt_newick, spr_radius=spr_radius)

        # logL of new tree
        _, new_logL = utils.evaluate_loglikelihood_raxmlng(
            next_newick, msa_path, model=bl_opt_model)
        ll_history.append(new_logL)

        cur_newick = next_newick
        actions = next_actions
        feats = next_feats

    return ll_history


def plot_ll_trajectories(all_results, out_dir):
    """One subplot per tree, showing logL over steps."""
    os.makedirs(out_dir, exist_ok=True)
    tree_ids = sorted(all_results.keys())
    n = len(tree_ids)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for i, tid in enumerate(tree_ids):
        ax = axes[i // cols][i % cols]
        ll = all_results[tid]
        ax.plot(range(len(ll)), ll, marker='o', markersize=3, color='tab:red')
        ax.set_title(f"Tree {tid}")
        ax.set_xlabel("Step")
        ax.set_ylabel("LogLikelihood")
        ax.grid(True, alpha=0.3)
        ax.annotate(f"{ll[0]:.1f}", (0, ll[0]), fontsize=7)
        ax.annotate(f"{ll[-1]:.1f}", (len(ll) - 1, ll[-1]), fontsize=7)

    for i in range(n, rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, "ll_trajectories.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Plot saved to {path}")


def main():
    args = parse_args()
    config, raw = utils.load_config(args.config)
    train_cfg = config.train_cfg
    phylo_cfg = config.phylo_cfg
    rl_cfg = config.rl_cfg
    paths = raw["paths"]

    data_dir = paths.get("data_dir", common.DATA_PATH)
    bridge_lib = paths.get("bridge_lib", "feat_cpp/bridge.so")
    n_steps = args.n_steps or rl_cfg.n_steps

    # Find checkpoint, auto-detect loss_type
    ckp_path, loss_type = find_checkpoint(args.ckp_dir, args.use_best)
    log.info(f"Checkpoint: {ckp_path} (loss_type={loss_type})")

    # Tree IDs
    tree_ids = args.tree_ids or utils.get_tree_indices(data_dir)
    log.info(f"Trees: {tree_ids}")

    # Build agent with greedy strategy
    agent = Agent(
        network_type=loss_type,
        in_features=train_cfg.in_features,
        out_features=train_cfg.out_features,
        device=train_cfg.device,
        layers=train_cfg.layers,
        strategy_train=common.Strategy.GREEDY.value,
        strategy_infer=common.Strategy.GREEDY.value,
    )
    agent.load_checkpoint(ckp_path)
    agent.to_eval()

    # Environment
    env = Environment(tree_indices=tree_ids, data_dir=data_dir, lib_path=bridge_lib)

    # Greedy rollout per tree
    all_results = {}
    for tid in tree_ids:
        msa_path = os.path.join(data_dir, str(tid), common.POSTFIX_MSA)
        ll_history = greedy_rollout(
            agent, env.cpp,
            start_newick=env.tree_start[tid],
            gt_newick=env.tree_gt[tid],
            msa_path=msa_path,
            n_steps=n_steps,
            spr_radius=phylo_cfg.spr_radius,
            bl_opt_model=phylo_cfg.bl_opt_model,
        )
        delta = ll_history[-1] - ll_history[0]
        log.info(f"  Tree {tid}: logL {ll_history[0]:.2f} -> {ll_history[-1]:.2f} (delta={delta:.2f})")
        all_results[tid] = ll_history

    plot_ll_trajectories(all_results, args.out_dir)


if __name__ == "__main__":
    main()