import os
import numpy as np
import torch
import torch.nn.functional as F
from env.environment import Environment
from agents.agent import Agent
from trainer.config import Config
from trainer.replay_buffer import ReplayBuffer
import common
import utils


class Trainer:
    def __init__(self, config: Config, env: Environment, agent: Agent):
        self.rl_cfg = config.rl_cfg
        self.train_cfg = config.train_cfg
        self.phylo_cfg = config.phylo_cfg
        self.env = env
        self.agent = agent

        self.optimizer = torch.optim.AdamW(
            self.agent.parameters(),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.32
        )

        if self.rl_cfg.loss_type == common.NetworkType.QLearning.value:
            self.buffer = ReplayBuffer(
                self.rl_cfg.buffer_size,
                self.rl_cfg.batch_size,
                self.train_cfg.device,
            )

        for idx in self.env.tree_indices:
            self.env.tree_state[idx] = self.env.tree_start[idx]

    def train(self):
        if self.rl_cfg.loss_type == common.NetworkType.REINFORCE.value:
            self._train_reinforce()
        elif self.rl_cfg.loss_type == common.NetworkType.QLearning.value:
            self._train_qlearning()

    # ================================================================ #
    #  REINFORCE                                                        #
    # ================================================================ #
    def _train_reinforce(self):
        for epoch in range(self.train_cfg.num_epoch):
            tree_cur = np.random.choice(self.env.tree_indices)
            self.env.tree_state[tree_cur] = self.env.tree_start[tree_cur]
            transitions = self._rollout(tree_cur)
            rewards = [t['reward'] for t in transitions]
            returns = self._compute_returns(rewards, self.rl_cfg.gamma)

            for _ in range(self.train_cfg.num_epoch_episode):
                self._update_reinforce(transitions, returns)

            self.scheduler.step()  # ← ADD

            if epoch % 50 == 0:
                obj = self.phylo_cfg.obj_func
                print(f"Epoch {epoch}/{self.train_cfg.num_epoch} "
                      f"tree={tree_cur} reward={sum(rewards):.4f} obj={obj}")

    def _rollout(self, tree_idx):
        cur_newick = self.env.tree_state[tree_idx]
        gt_newick = self.env.tree_gt[tree_idx]
        is_ll = self.phylo_cfg.obj_func == common.ObjFunc.LIKELIHOOD.value
        msa_path = os.path.join(self.env.data_dir,
                                f"{tree_idx}{common.POSTFIX_MSA}")
        transitions = []

        # ── Likelihood: get initial logL ──
        if is_ll:
            _, cur_logL = utils.evaluate_loglikelihood_raxmlng(
                cur_newick, msa_path, model=self.phylo_cfg.bl_opt_model)

        # ── Initial features (no action applied) ──
        cur_newick, actions, feats, rf_rewards = \
            self.env.cpp.get_state_action(
                cur_newick, -1, gt_newick,
                spr_radius=self.phylo_cfg.spr_radius)

        for step in range(self.rl_cfg.n_steps):
            X = torch.tensor(feats, dtype=torch.float32,
                             device=self.train_cfg.device)
            X = utils.normalize_features(X)

            _, action_idx = self.agent.choose(actions, X)

            # Apply action → resulting tree + next features
            next_newick, next_actions, next_feats, next_rf = \
                self.env.cpp.get_state_action(
                    cur_newick, action_idx, gt_newick,
                    spr_radius=self.phylo_cfg.spr_radius)

            # ── Compute reward ──
            if is_ll:
                _, new_logL = utils.evaluate_loglikelihood_raxmlng(
                    next_newick, msa_path,
                    model=self.phylo_cfg.bl_opt_model)
                reward = new_logL - cur_logL
                cur_logL = new_logL
            else:
                reward = float(rf_rewards[action_idx])

            transitions.append({
                'features': X,
                'action_idx': action_idx,
                'reward': reward,
            })

            # BL optimization (RF mode only; likelihood already does it)
            if not is_ll and self.phylo_cfg.optimize_bl \
                    and (step + 1) % self.phylo_cfg.bl_opt_every == 0:
                next_newick = utils.optimize_bl_raxmlng(
                    next_newick, msa_path)

            cur_newick = next_newick
            actions = next_actions
            feats = next_feats
            rf_rewards = next_rf

        self.env.tree_state[tree_idx] = cur_newick
        return transitions

    @staticmethod
    def _compute_returns(rewards, gamma):
        T = len(rewards)
        G = [0.0] * T
        G[-1] = rewards[-1]
        for t in range(T - 2, -1, -1):
            G[t] = rewards[t] + gamma * G[t + 1]
        return G

    def _update_reinforce(self, transitions, returns):
        G = torch.tensor(returns, dtype=torch.float32,
                         device=self.train_cfg.device)
        G = (G - G.mean()) / (G.std() + 1e-8)

        loss = torch.tensor(0.0, device=self.train_cfg.device)
        for t, G_t in zip(transitions, G):
            logits = self.agent.forward(t['features']).squeeze(-1)
            log_probs = torch.log_softmax(logits, dim=0)
            loss += -log_probs[t['action_idx']] * G_t

        loss = loss / len(transitions)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
        self.optimizer.step()

    # ================================================================ #
    #  Q-learning  (unchanged)                                          #
    # ================================================================ #
    def _train_qlearning(self):
        total_steps = 0
        for epoch in range(self.train_cfg.num_epoch):
            tree_cur = np.random.choice(self.env.tree_indices)
            cur_newick = self.env.tree_state[tree_cur]
            gt_newick = self.env.tree_gt[tree_cur]
            epoch_reward = 0.0

            cur_newick, cur_actions, cur_feats, cur_rewards = \
                self.env.cpp.get_state_action(cur_newick, -1, gt_newick)

            for step in range(self.rl_cfg.n_steps):
                X = torch.tensor(cur_feats, dtype=torch.float32,
                                 device=self.train_cfg.device)
                action, action_idx = self.agent.choose(cur_actions, X)
                chosen_feat = X[action_idx].detach()
                reward = float(cur_rewards[action_idx])
                epoch_reward += reward

                next_newick, next_actions, next_feats, next_rewards = \
                    self.env.cpp.get_state_action(
                        cur_newick, action_idx, gt_newick)
                next_X = torch.tensor(next_feats, dtype=torch.float32,
                                      device=self.train_cfg.device)
                with torch.no_grad():
                    next_q = self.agent.network_target(next_X).squeeze(-1)
                    best_next_feat = next_X[torch.argmax(next_q)].detach()

                self.buffer.add(chosen_feat, reward, best_next_feat)
                if len(self.buffer) >= self.rl_cfg.batch_size:
                    self._learn_qlearning()

                total_steps += 1
                if total_steps % self.rl_cfg.target_update_freq == 0:
                    self.agent.update_target()

                cur_newick = next_newick
                cur_actions = next_actions
                cur_feats = next_feats
                cur_rewards = next_rewards

            self.env.tree_state[tree_cur] = cur_newick
            if epoch % 50 == 0:
                print(f"Epoch {epoch}/{self.train_cfg.num_epoch} "
                      f"tree={tree_cur} reward={epoch_reward:.4f} "
                      f"eps={self.agent.epsilon:.3f} buffer={len(self.buffer)}")

    def _learn_qlearning(self):
        sa, r, nsa = self.buffer.sample()
        q_pred = self.agent.forward(sa)
        with torch.no_grad():
            q_next = self.agent.network_target(nsa)
        q_target = r + self.rl_cfg.gamma * q_next
        loss = F.mse_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()