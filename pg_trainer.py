import copy
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import uuid


class PGTrainer:
    def __init__(self, trainer_cfg, agent, env):
        self._trainer_cfg = trainer_cfg
        self._device = self._trainer_cfg['device']
        self._train_cfg = trainer_cfg['train_cfg']

        self._agent = agent
        self._env = env

        self._memory_size = None
        self._memory_fill = 0
        self._optimizer_cfg = trainer_cfg['optimizer_cfg']
        self._optimizer = self._optimizer_cfg['optimizer_cls'](self._agent.parameters(),
                                                               **self._optimizer_cfg['optimizer_args'])

        self._lr_scheduler = None
        if 'lr_scheduler' in self._optimizer_cfg and self._optimizer_cfg['lr_scheduler'] is not None:
            self._lr_scheduler = self._optimizer_cfg['lr_scheduler'](optimizer=self._optimizer, **self._optimizer_cfg['lr_scheduler_args'])


        self._train_freq = self._train_cfg['eps_in_memory']

    def __eval_agent(self, test_data):
        self._agent.to_eval()
        val_logs, val_traj_logs = self._env.validation_fn(
        test_data, self._agent, self._env, self._train_cfg["n_steps"],
        orig_idxs=self._test_orig_idxs
        )
        self._agent.to_train()
        return val_logs, val_traj_logs

    def __log_losses(self, logger, losses):
        if logger is not None:
            logger['train/batch_loss'].extend(losses)
            logger['train/avg_loss'].append(np.mean(losses))

    def __log_grads(self, logger, grad_logs):
        if logger is not None:
            for k, v in grad_logs.items():
                logger[f'train/grad_{k}'].extend(v)

    def __log_training(self, logger, train_logs, train_traj_logs):
        if logger is not None:
            for k, v in train_logs.items():
                logger[f'train/{k}'].append(np.mean(v))
                for v_idx, v_i in enumerate(v):
                    logger[f'train/instance{v_idx}_{k}'].append(v_i)

            for k, v in train_traj_logs.items():
                k_traj = np.array(v).mean(axis=0)
                logger[f'train/{k}'].extend(k_traj.tolist())

    def __log_validation(self, logger, val_logs, val_traj_logs):
        if logger is not None:
            for k, v in val_logs.items():
                logger[f'validation/{k}'].append(np.mean(v))

                for v_idx, v_i in enumerate(v):
                    logger[f'validation/instance{v_idx}_{k}'].append(v_i)

            for k, v in val_traj_logs.items():
                k_traj = np.array(v).mean(axis=0)
                logger[f'validation/{k}'].extend(k_traj.tolist())

    def __init_memory(self, n_sa):

        self._memory_size = self._train_cfg["eps_in_memory"] * self._train_cfg["n_steps"]
        memory_shape = (self._memory_size, self._train_cfg["problem_batch_size"])

        # n_sa = 3024
        self._sa_memory = torch.zeros(*memory_shape, n_sa,
                                      self._env.n_features(), device=self._device)
        self._next_sa_memory = torch.zeros(*memory_shape, n_sa,
                                           self._env.n_features(), device=self._device)
        self._done_memory = torch.zeros(*memory_shape, 1,
                                        device=self._device)
        self._a_memory = torch.zeros(*memory_shape, 1,
                                     device=self._device)
        self._r_memory = torch.zeros(*memory_shape, 1,
                                     device=self._device)
        self._gt_memory = torch.zeros(*memory_shape, 1,
                                      device=self._device)
        self._bspr_memory = torch.zeros(*memory_shape, 1,
                                        device=self._device)

    def __rollout_loop(self, ep_idx, batch_d, batch_orig_idxs, train_logs, train_traj_logs):
        with torch.no_grad():
            batch_data = self._env.make_batch_data(batch_d[0], batch_orig_idxs)  # <-- was data_class()(batch_d[0])

            batch_state = self._env.reset(batch_data)
            train_init_len = batch_state.best_tree.get_length()
            train_logs['train_init_len'].append(torch.mean(train_init_len).item())
            bspr_len, bspr_traj = batch_state.current_tree.BSPR(self._train_cfg["n_steps"], ret_traj=True)
            full_bspr_improvement = train_init_len - bspr_len
            train_logs['train_bspr_len'].append(bspr_len.mean().item())
            train_logs['train_bspr_improvement'].append(full_bspr_improvement.mean().item())
            train_traj_logs['train_bspr_traj'].append(bspr_traj.mean(axis=0).tolist())

            policy_traj = [train_init_len.mean().item()]
            state_actions = self._env.neighbors_features(batch_data, batch_state)

            for step_i in range(self._train_cfg["n_steps"]):
                act_index = self._agent(state_actions)

                batch_state, reward, bspr_state = self._env.step(batch_data, batch_state, act_index,
                                                                 bspr_baseline_steps=self._train_cfg[
                                                                                         "n_steps"] - step_i)
                bspr_len, bspr_improvement, bspr_value = bspr_state
                policy_traj.append(batch_state.current_tree.get_length().mean().item())

                # episode_memory.append((state_actions, act_index, reward))
                self._sa_memory[ep_idx + step_i] = state_actions
                self._a_memory[ep_idx + step_i] = act_index.unsqueeze(-1)
                self._r_memory[ep_idx + step_i] = reward.unsqueeze(-1)
                self._bspr_memory[ep_idx + step_i] = bspr_value.unsqueeze(-1)

                state_actions = self._env.neighbors_features(batch_data, batch_state)
                self._next_sa_memory[ep_idx + step_i] = state_actions

            train_final_len = batch_state.best_tree.get_length()
            train_logs['train_policy_len'].append(train_final_len.mean().item())
            policy_improvement = train_init_len - train_final_len
            train_logs['train_policy_improvement'].append(policy_improvement.mean().item())
            train_logs['train_policy_gap'].append(((full_bspr_improvement - policy_improvement) / full_bspr_improvement).mean().item()*100)
            train_traj_logs['train_policy_traj'].append(policy_traj)

            self._gt_memory[ep_idx + self._train_cfg["n_steps"] - 1] = self._r_memory[ep_idx + self._train_cfg["n_steps"] - 1]
            for i in range(2, self._train_cfg['n_steps'] + 1):
                r = self._r_memory[ep_idx + self._train_cfg["n_steps"] - i]
                self._gt_memory[ep_idx + self._train_cfg["n_steps"] - i] = r + self._train_cfg['gamma'] * self._gt_memory[ep_idx + self._train_cfg["n_steps"] - i + 1]

            train_logs['train_reward'].append(self._gt_memory[ep_idx].mean().item())

            self._agent.step()
            

    def __train_loop(self):
        print('train_loop is called')
        losses = []
        grad_logs = {n: [] for n, _ in self._agent.named_parameters()}
        for train_epoch_i in range(self._train_cfg['train_epochs']):
            ###### TRAIN ON EPISODE
            train_dset = TensorDataset(self._a_memory[:self._memory_fill].flatten(0, 1),
                                       self._sa_memory[:self._memory_fill].flatten(0, 1),
                                       self._next_sa_memory[:self._memory_fill].flatten(0, 1),
                                       self._gt_memory[:self._memory_fill].flatten(0, 1),
                                       self._r_memory[:self._memory_fill].flatten(0, 1),
                                       self._bspr_memory[:self._memory_fill].flatten(0, 1))
            train_dataloader = DataLoader(train_dset, batch_size=self._train_cfg["training_batch_size"],
                                          shuffle=True)
            for batch in train_dataloader:
                batch_acts, batch_sa, batch_next_sa, batch_gt, batch_r, bspr_value = batch

                _, act_ll, entropies, act_probs = self._agent(batch_sa, act_in=batch_acts, ret_ll=True, ret_entropy=True,
                                                        ret_probs=True)

                selected_act_ll = torch.gather(act_ll, dim=-1, index=batch_acts.long())
                #selected_act_probs = torch.gather(act_probs, dim=-1, index=batch_acts.long())

                #batch_s_value = batch_r + bspr_value

                #batch_adv = -(batch_gt - batch_s_value)
                #batch_adv = batch_r
                #bspr_acts = torch.argmin(batch_sa[..., 7].view(len(batch_sa), -1), dim=-1)
                #is_bspr_act = batch_acts == bspr_acts

                #batch_adv = batch_r + bspr_value
                batch_adv = batch_gt

                #batch_adv = batch_r * is_bspr_act
                loss = -(selected_act_ll * batch_adv.detach())
                #entropy_loss = b_e * entropies.mean()

                loss = loss.mean()

                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._agent.parameters(), max_norm=1)
                losses.append(loss.item())

                for n, p in self._agent.named_parameters():
                    if p.grad is not None:
                        grad_logs[n].append(torch.mean(p.grad.cpu()).item())

                self._optimizer.step()

        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

        return losses, grad_logs

    def train(self, train_data, test_data, logger, save_best=False, checkpoint_frequency=float('inf'), save_dir='',
              run_id=None):
        if run_id is None:
            train_id = str(uuid.uuid4())
        else:
            train_id = run_id

        if save_best or checkpoint_frequency < float('inf'):
            self._save_path = os.path.join(save_dir, f'checkpoints_pg_trainer_{train_id}')
            os.makedirs(self._save_path)

        _train_inst = [train_data.random_instance_with_indices(self._train_cfg['instance_size'])
               for _ in range(self._train_cfg['n_train_instances'])]
        train_dset = torch.cat([x[0].unsqueeze(0) for x in _train_inst])
        self._train_orig_idxs = [x[1] for x in _train_inst]

        _test_inst = [test_data.random_instance_with_indices(self._train_cfg['instance_size'])
                    for _ in range(self._train_cfg['n_test_instances'])]
        test_dset = torch.cat([x[0].unsqueeze(0) for x in _test_inst])
        self._test_orig_idxs = [x[1] for x in _test_inst]

        train_loader = DataLoader(TensorDataset(train_dset), shuffle=False,
                                  batch_size=self._train_cfg["problem_batch_size"])

        # print('####> Init Eval')
        # val_logs, val_traj_logs = self.__eval_agent(test_dset)
        # self.__log_validation(logger, val_logs, val_traj_logs)
        # best_perf = np.mean(val_logs['policy_gap'])
        # print('####')
        best_perf = float('inf')

        n2_training_taxa = 2 * self._train_cfg['instance_size']
        n_training_acts = (n2_training_taxa - 6) * (n2_training_taxa - 7)  # + (n2_training_taxa-6)*3
        self.__init_memory(n_training_acts)
        episode_count = 0
        self._train_count = 0
        for e_i in range(self._train_cfg["n_epochs"]):
            #if e_i == 20:
            #    self._train_cfg['n_steps'] = 40
            #    self.__init_memory(n_training_acts)
            #elif e_i == 40:
            #    self._train_cfg['n_steps'] = 60
            #    self.__init_memory(n_training_acts)
            #elif e_i == 60:
            #    self._train_cfg['n_steps'] = 80
            #    self.__init_memory(n_training_acts)
            #elif e_i == 80:
            #    self._train_cfg['n_steps'] = 100
            #    self.__init_memory(n_training_acts)

            train_logs = {'train_init_len': [],
                          'train_policy_len': [],
                          'train_policy_improvement': [],
                          'train_policy_gap': [],
                          'train_bspr_len': [],
                          'train_bspr_improvement': [],
                          'train_reward': []}

            train_traj_logs = {'train_policy_traj': [],
                               'train_bspr_traj': []}

            problem_idx = 0
            batch_size = self._train_cfg["problem_batch_size"]
            for batch_d in train_loader:
                episode_count += 1
                ep_idx = ((e_i + problem_idx) % self._train_cfg["eps_in_memory"]) * self._train_cfg["n_steps"]
                start = problem_idx * batch_size
                batch_orig_idxs = self._train_orig_idxs[start : start + batch_d[0].shape[0]]
                self.__rollout_loop(ep_idx, batch_d, batch_orig_idxs, train_logs, train_traj_logs)
                self._memory_fill = min(self._memory_size,
                                        self._memory_fill + self._train_cfg["n_steps"])
                problem_idx += 1

                if episode_count % self._train_freq == 0:
                    self._train_count += 1
                    losses, grad_logs = self.__train_loop()
                    self.__log_losses(logger, losses)
                    self.__log_grads(logger, grad_logs)

            self.__log_training(logger, train_logs, train_traj_logs)
            print('####> Epoch Eval')
            val_logs, val_traj_logs = self.__eval_agent(test_dset)
            self.__log_validation(logger, val_logs, val_traj_logs)
            epoch_perf = np.mean(val_logs['policy_gap'])
            if epoch_perf < best_perf:
                best_perf = epoch_perf
                if save_best:
                    self._agent.save_checkpoint(os.path.join(self._save_path, f'best_agent_epoch{e_i}gap{best_perf}'))
            print('####')

            if e_i % checkpoint_frequency == 0 and e_i > 0:
                self._agent.save_checkpoint(os.path.join(self._save_path, f'agent_epoch{e_i}_gap{epoch_perf}'))

            if e_i % self._trainer_cfg['resample_freq'] == 0:
                _train_inst = [train_data.random_instance_with_indices(self._train_cfg['instance_size'])
               for _ in range(self._train_cfg['n_train_instances'])]

                train_dset = torch.cat([x[0].unsqueeze(0) for x in _train_inst])
                self._train_orig_idxs = [x[1] for x in _train_inst]
                
                train_loader = DataLoader(TensorDataset(train_dset), shuffle=False,
                                        batch_size=self._train_cfg["problem_batch_size"])

    def save_path(self):
        return self._save_path