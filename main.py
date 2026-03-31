import os
import glob
import torch
import common
from common import Strategy
from agents.agent import Agent
from trainer.trainer import Trainer
from trainer.config import Config, TrainConfig


def load_tree_data(data_dir: str):
    """
    Scans data_dir for files matching {id}_starting.tre / {id}_ground_truth.tre.
    Returns two dicts: {tree_id -> starting_newick} and {tree_id -> gt_newick}.
    """
    dict_tree_newick = {}
    dict_tree_gt_newick = {}

    for path in sorted(glob.glob(os.path.join(data_dir, '*_starting.tre'))):
        tree_id = os.path.basename(path).replace('_starting.tre', '')
        gt_path = os.path.join(data_dir, f'{tree_id}_ground_truth.tre')
        if not os.path.exists(gt_path):
            print(f'[warn] no ground truth for tree {tree_id}, skipping')
            continue
        with open(path) as f:
            dict_tree_newick[tree_id] = f.read().strip()
        with open(gt_path) as f:
            dict_tree_gt_newick[tree_id] = f.read().strip()

    return dict_tree_newick, dict_tree_gt_newick


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dict_tree_newick, dict_tree_gt_newick = load_tree_data(common.DATA_PATH)
    print(f'Loaded {len(dict_tree_newick)} trees: {list(dict_tree_newick.keys())}')

    train_cfg = TrainConfig(
        num_epoch=200,
        n_steps=30,
        learning_rate=1e-3,
        in_features=20,
        out_features=1,
        device=device,
        layers=[128, 128, 128, 128],
        weight_decay=1e-4,
    )
    config = Config(train_cfg=train_cfg)

    agent = Agent(strategy=Strategy.GREEDY.value)

    trainer = Trainer(
        config=config,
        dict_tree_newick=dict_tree_newick,
        dict_tree_gt_newick=dict_tree_gt_newick,
        agent=agent,
    )

    trainer.train()