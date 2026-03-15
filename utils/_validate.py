import copy
import torch

from environments.env_utils import BMEPData, LLData


def validation(test_data, agent, bmep, n_steps):
    logs = {'init_len': [],
            'policy_len': [],
            'policy_improvement': []}

    traj_logs = {'policy_traj': []}

    for data_i in test_data:
        batch_data = BMEPData(data_i.squeeze(0))
        batch_state = bmep.reset(batch_data)
        init_state = copy.deepcopy(batch_state)
        init_len = batch_state.best_tree.get_length()

        logs['init_len'].append(init_len)

        policy_traj = [init_len]
        for step_i in range(n_steps):
            state_actions = bmep.neighbors_features(batch_data, batch_state)
            act_index = agent(state_actions.unsqueeze(0)).item()
            batch_state, reward = bmep.step(batch_data, batch_state, act_index)
            policy_traj.append(batch_state.current_tree.get_length())
        policy_len = batch_state.best_tree.get_length()
        logs['policy_len'].append(policy_len)
        logs['policy_improvement'].append(init_len - policy_len)
        traj_logs['policy_traj'].append(policy_traj)

    return logs, traj_logs


def batched_validation(test_data, agent, bmep, n_steps, orig_idxs=None):
    logs = {'init_len': [],
            'policy_len': [],
            'policy_improvement': [],
            'policy_gap': []}

    traj_logs = {'policy_traj': []}

    if orig_idxs is not None:
        batch_data = bmep.make_batch_data(test_data, orig_idxs)
    else:
        batch_data = bmep.data_class()(test_data)
    batch_state = bmep.reset(batch_data)
    init_state = copy.deepcopy(batch_state)
    init_len = batch_state.best_tree.get_length()
    init_len_mean = init_len.mean().item()

    logs['init_len'].append(init_len_mean)

    policy_traj = [init_len_mean]
    for step_i in range(n_steps):
        state_actions = bmep.neighbors_features(batch_data, batch_state)
        act_index = agent(state_actions)
        batch_state, reward = bmep.step(batch_data, batch_state, act_index)
        policy_traj.append(batch_state.current_tree.get_length().mean().item())
    policy_len = batch_state.best_tree.get_length()
    policy_len_mean = policy_len.mean().item()
    logs['policy_len'].append(policy_len_mean)
    policy_improv = init_len - policy_len
    logs['policy_improvement'].append(init_len_mean - policy_len_mean)
    traj_logs['policy_traj'].append(policy_traj)

    return logs, traj_logs
