import numpy as np
import torch
import torch.nn.functional as F
from env.environment import Environment
from agents.agent import Agent
from trainer.config import Config
from trainer.replay_buffer import ReplayBuffer
import common
import random

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

        # ── Episode memory for REINFORCE ──
        self.episode_memory = []


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

            # ── Store episode ──
            self.episode_memory.append((transitions, returns))
            if len(self.episode_memory) > self.rl_cfg.pg_episodes_in_memory:
                self.episode_memory.pop(0)

            # ── Train multi-epoch on stored episodes ──
            if len(self.episode_memory) >= self.rl_cfg.pg_episodes_in_memory:
                self._update_reinforce_batched()

            # ── Step LR scheduler ──
            self.scheduler.step()

            if epoch % 50 == 0:
                lr_now = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}/{self.train_cfg.num_epoch} "
                      f"tree={tree_cur} reward={sum(rewards):.4f} lr={lr_now:.6f}")

    def _rollout(self, tree_idx):
        cur_newick = self.env.tree_state[tree_idx]
        gt_newick = self.env.tree_gt[tree_idx]
        action_idx = -1
        transitions = []

        for step in range(self.rl_cfg.n_steps):
            new_newick, actions, feats, rewards = \
                self.env.cpp.get_state_action(
                    cur_newick, action_idx, gt_newick,
                    spr_radius=self.phylo_cfg.spr_radius)

            X = torch.tensor(feats, dtype=torch.float32,
                             device=self.train_cfg.device)
            X[:, [1, 2, 3, 5, 6]] /= (X[:, 0:1] + 1e-8)
            X[:, [7, 8, 9, 10]] /= 30.0
            X[:, [11, 12, 13, 14]] /= (X[:, 0:1] + 1e-8)
            X[:, [15, 16, 17, 18]] /= (X[:, 0:1] + 1e-8)

            if step % 10 == 0:
                with torch.no_grad():
                    logits = self.agent.network(X).squeeze(-1)
                    probs = torch.softmax(logits, dim=0)
                top5 = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)[:5]
                print(f"  Top-5 reward actions (step {step}):")
                for rank, i in enumerate(top5):
                    print(f"    #{rank+1} idx={i} r={rewards[i]:.2f} prob={probs[i].item():.4f}")

            _, action_idx = self.agent.choose(actions, X)

            # ── Clip reward ──
            r = max(0.0, float(rewards[action_idx]))

            transitions.append({
                'features': X,
                'action_idx': action_idx,
                'reward': r,
            })
            cur_newick = new_newick

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

    def _update_reinforce_batched(self):
        """Multi-epoch mini-batch training over stored episodes (à la bmepRL)."""
        # 1. Flatten all episodes into (features, action_idx, return_value)
        all_data = []
        for transitions, returns in self.episode_memory:
            G = torch.tensor(returns, dtype=torch.float32,
                             device=self.train_cfg.device)
            G = (G - G.mean()) / (G.std() + 1e-8)
            for t, g_t in zip(transitions, G):
                all_data.append((t['features'], t['action_idx'], g_t))

        # 2. Train multiple epochs with mini-batches
        bs = self.rl_cfg.pg_batch_size
        for _ in range(self.rl_cfg.pg_epochs):
            random.shuffle(all_data)
            for start in range(0, len(all_data), bs):
                batch = all_data[start:start + bs]
                loss = torch.tensor(0.0, device=self.train_cfg.device)

                for features, action_idx, G_t in batch:
                    logits = self.agent.forward(features).squeeze(-1)
                    log_probs = torch.log_softmax(logits, dim=0)
                    loss += -log_probs[action_idx] * G_t

                    probs = torch.softmax(logits, dim=0)
                    entropy = -(probs * log_probs).sum()
                    loss += -0.01 * entropy

                loss = loss / len(batch)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
                self.optimizer.step()

    def _update_reinforce(self, transitions, returns):
        G = torch.tensor(returns, dtype=torch.float32, device=self.train_cfg.device)
        G = (G - G.mean()) / (G.std() + 1e-8)

        loss = torch.tensor(0.0, device=self.train_cfg.device)
        for t, G_t in zip(transitions, G):
            logits = self.agent.forward(t['features']).squeeze(-1)
            log_probs = torch.log_softmax(logits, dim=0)
            loss += -log_probs[t['action_idx']] * G_t
            probs = torch.softmax(logits, dim=0)
            entropy = -(probs * log_probs).sum()
            loss += -0.01 * entropy

        loss = loss / len(transitions)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
        self.optimizer.step()

    # ================================================================ #
    #  Q-learning                                                       #
    # ================================================================ #
    def _train_qlearning(self):
        total_steps = 0

        for epoch in range(self.train_cfg.num_epoch):
            tree_cur = np.random.choice(self.env.tree_indices)
            cur_newick = self.env.tree_state[tree_cur]
            gt_newick = self.env.tree_gt[tree_cur]
            epoch_reward = 0.0

            cur_newick, cur_actions, cur_feats, cur_rewards = \
                self.env.cpp.get_state_action(
                    cur_newick, -1, gt_newick)            # ← CHANGED

            for step in range(self.rl_cfg.n_steps):
                if step % 10 == 0:
                    print(f'at epoch {epoch}, step {step}')
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
