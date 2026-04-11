import os
import glob
import common
from typing import Dict, Tuple, List
import torch
import subprocess
import tempfile

def load_data(data_dir):
    tree_start, tree_gt = {}, {}
    for folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        start_path = os.path.join(folder_path, common.POSTFIX_START)
        gt_path = os.path.join(folder_path, common.POSTFIX_GT)
        if os.path.exists(start_path) and os.path.exists(gt_path):
            with open(start_path) as f:
                tree_start[folder] = f.read().strip()
            with open(gt_path) as f:
                tree_gt[folder] = f.read().strip()
    return tree_start, tree_gt


def load_data_indices(data_dir: str, indices: List[int]) -> Tuple[Dict[int, str], Dict[int, str]]:
    all_start, all_gt = load_data(data_dir)
    tree_start = {idx: all_start[str(idx)] for idx in indices}
    tree_gt = {idx: all_gt[str(idx)] for idx in indices}
    return tree_start, tree_gt


def get_tree_indices(data_dir):
    indices = []
    for folder in sorted(os.listdir(data_dir)):
        if os.path.isdir(os.path.join(data_dir, folder)):
            try:
                indices.append(int(folder))
            except ValueError:
                pass
    return sorted(indices)

def normalize_features(X: torch.Tensor, n_taxa: int = 30) -> torch.Tensor:
    """Normalize raw 19-dim SPR features in-place and return X."""
    total_bl = X[:, 0:1] + 1e-8
    X[:, [1, 2, 3, 5, 6]] /= total_bl        # BL features / total_bl
    X[:, [7, 8, 9, 10]] /= float(n_taxa)      # leaf counts / n_taxa
    X[:, [11, 12, 13, 14]] /= total_bl        # subtree BLs / total_bl
    X[:, [15, 16, 17, 18]] /= total_bl        # longest BLs / total_bl
    return X

def optimize_bl_raxmlng(newick: str, msa_path: str,
                        model: str = "GTR+G",
                        raxmlng: str = "raxml-ng") -> str:
    """Gọi raxml-ng --evaluate để optimize BL trên fixed topology."""
    with tempfile.TemporaryDirectory() as tmp:
        tree_file = os.path.join(tmp, "tree.nwk")
        prefix = os.path.join(tmp, "eval")
        with open(tree_file, 'w') as f:
            f.write(newick)
        result = subprocess.run(
            [raxmlng, "--evaluate",
             "--msa", msa_path,
             "--tree", tree_file,
             "--model", model,
             "--prefix", prefix,
             "--threads", "1",
             "--force", "perf_threads"],
            capture_output=True, timeout=60)
        out_file = prefix + ".raxml.bestTree"
        if os.path.exists(out_file):
            with open(out_file) as f:
                return f.read().strip()
    return newick

def evaluate_loglikelihood_raxmlng(newick: str, msa_path: str,
                                   model: str = "GTR+G",
                                   raxmlng: str = "raxml-ng") -> Tuple[str, float]:
    """Call raxml-ng --evaluate, return (optimized_newick, logL)."""
    with tempfile.TemporaryDirectory() as tmp:
        tree_file = os.path.join(tmp, "tree.nwk")
        prefix = os.path.join(tmp, "eval")
        with open(tree_file, 'w') as f:
            f.write(newick)
        result = subprocess.run(
            [raxmlng, "--evaluate",
             "--msa", msa_path,
             "--tree", tree_file,
             "--model", model,
             "--prefix", prefix,
             "--threads", "1",
             "--force", "perf_threads"],
            capture_output=True, timeout=120)

        logL = float('-inf')
        stdout = result.stdout.decode('utf-8', errors='replace')
        for line in stdout.split('\n'):
            if 'Final LogLikelihood:' in line:
                logL = float(line.split(':')[-1].strip())
                break

        opt_newick = newick
        out_file = prefix + ".raxml.bestTree"
        if os.path.exists(out_file):
            with open(out_file) as f:
                opt_newick = f.read().strip()

        return opt_newick, logL

def load_config(path: str = "config.yaml"):
    """Load Config from YAML. Returns (Config, raw_dict)."""
    import yaml
    from trainer.config import Config, TrainConfig, RLConfig, PhyloConfig

    with open(path) as f:
        raw = yaml.safe_load(f)

    t = raw["training"]
    device = t.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_cfg = TrainConfig(
        num_epoch=t["num_epoch"],
        learning_rate=t["learning_rate"],
        in_features=t["in_features"],
        out_features=t["out_features"],
        device=device,
        layers=t["layers"],
        weight_decay=t.get("weight_decay", 1e-4),
        num_epoch_episode=t.get("num_epoch_episode", 5),
    )

    r = raw["rl"]
    rl_cfg = RLConfig(
        n_steps=r["n_steps"],
        loss_type=r["loss_type"],
        gamma=r.get("gamma", 0.9),
        pg_episodes_in_memory=r.get("pg_episodes_in_memory", 2),
        pg_epochs=r.get("pg_epochs", 10),
        pg_batch_size=r.get("pg_batch_size", 64),
        buffer_size=r.get("buffer_size", 10000),
        batch_size=r.get("batch_size", 32),
        target_update_freq=r.get("target_update_freq", 100),
        tau=r.get("tau", 0.005),
        epsilon_start=r.get("epsilon_start", 1.0),
        epsilon_end=r.get("epsilon_end", 0.05),
        epsilon_decay_steps=r.get("epsilon_decay_steps", 5000),
        reward_type=r.get("reward_type", "raw"),
        reward_scale=r.get("reward_scale", 1000),
    )

    p = raw["phylo"]
    phylo_cfg = PhyloConfig(
        move_type=p["move_type"],
        obj_func=p["obj_func"],
        spr_radius=p.get("spr_radius", common.DEFAULT_SPR_RADIUS),
        optimize_bl=p.get("optimize_bl", True),
        bl_opt_every=p.get("bl_opt_every", 5),
        bl_opt_model=p.get("bl_opt_model", "GTR+G"),
    )

    return Config(train_cfg=train_cfg, rl_cfg=rl_cfg, phylo_cfg=phylo_cfg), raw