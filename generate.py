#!/usr/bin/env python3
"""Read config.yaml and run feat_cpp/generate with the right args."""
import subprocess
import sys
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)["generate"]

cmd = [
    "feat_cpp/generate",
    "--taxa",    str(cfg["n_taxa"]),
    "--len",     str(cfg["seq_length"]),
    "--num",     str(cfg["n_trees"]),
    "--seed",    str(cfg["seed"]),
    "--outdir",  cfg["out_dir"],
    "--model",   cfg["subst_model"],
    "--iqtree",  cfg["iqtree_bin"],
    "--raxmlng", cfg["raxmlng_bin"],
]

print("Running:", " ".join(cmd))
sys.exit(subprocess.call(cmd))