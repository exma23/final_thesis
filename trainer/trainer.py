from typing import Dict
import numpy as np
import torch
from env.buffer import create_batch
from agents.agent import Agent
from networks.ff_net import FFNet
from trainer.config import Config, TrainConfig
import common


class Trainer:
    def __init__(
        self,
        config: Config,
        dict_tree_newick: Dict[str, str],
        dict_tree_gt_newick: Dict[str, str],
        agent: Agent,
    ):
        self.train_cfg = config.train_cfg
        self.dict_tree_newick = dict_tree_newick
        self.dict_tree_gt_newick = dict_tree_gt_newick
        self.list_trees = list(dict_tree_newick.keys())
        self.agent = agent
        self.FFNet = FFNet(
            self.train_cfg.in_features,
            self.train_cfg.out_features,
            self.train_cfg.device,
            self.train_cfg.layers,
        )
        self.optimizer = torch.optim.AdamW(
            self.FFNet.parameters(),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )

    def train(self):
        for epoch in range(self.train_cfg.num_epoch):
            tree = np.random.choice(self.list_trees)

            newick_old = self.dict_tree_newick[tree]
            gt_newick = self.dict_tree_gt_newick[tree]
            action_chosen = common.INITIAL_ACTION

            for step in range(self.train_cfg.n_steps):
                newick_current, actions_current, X_feat_current, y_true_current = create_batch(
                    newick_old, action_chosen, gt_newick
                )

                X_feat_tensor = torch.tensor(X_feat_current, dtype=torch.float32, device=self.train_cfg.device)
                y_true_tensor = torch.tensor(y_true_current, dtype=torch.float32, device=self.train_cfg.device)

                y_pred_current = self.FFNet(X_feat_tensor)
                self.update_grad(X_feat_tensor, y_true_tensor)

                action_chosen = self.agent.choose(actions_current, y_pred_current)
                newick_old = newick_current

    def update_grad(self, X_feat: torch.Tensor, y: torch.Tensor):
        logits = self.FFNet(X_feat)
        probs = torch.softmax(logits, dim=-1)

        loss = -(probs * y)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()