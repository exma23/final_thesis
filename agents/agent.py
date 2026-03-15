from functools import partial

import torch
import torch.nn as nn
from torch.distributions.multinomial import Multinomial

from agents.selection import GreedySelection, RandomSelection, EpsGreedySelection, SampleSelection, \
    BSPRSelection, ReductionSampleSelection

def feats_or_probs(feats, probs, ret_feats):
    return feats if ret_feats else probs


def combine_feats_and_probs(feats, probs, alpha):
    feat_probs = torch.softmax((1 - feats[..., 7]) / 0.000001, dim=-1)
    return ((1 - alpha) * probs) + (alpha * feat_probs)


class SPRPolicy(nn.Module):
    def __init__(self, policy_cfg, net_cfg):
        super().__init__()
        self.policy_cfg = policy_cfg
        self.net_cfg = net_cfg

        self.policy_network = self.net_cfg['network_cls'](**self.net_cfg['network_args'])

        self.train_selection_method = None
        self.train_input_fn = None
        if self.policy_cfg['train_selection_method'] == 'greedy':
            self.train_selection_method = GreedySelection()
            self.train_input_fn = partial(feats_or_probs, ret_feats=False)
        elif self.policy_cfg['train_selection_method'] == 'random':
            self.train_selection_method = RandomSelection()
            self.train_input_fn = partial(feats_or_probs, ret_feats=False)
        elif self.policy_cfg['train_selection_method'] == 'eps-greedy':
            self.train_selection_method = EpsGreedySelection(self.policy_cfg['init_eps'],
                                                             self.policy_cfg['eps_schedule_cls'](
                                                                 **self.policy_cfg['eps_schedule_cfg']
                                                             ))
            self.train_input_fn = partial(feats_or_probs, ret_feats=False)
            
        elif self.policy_cfg['train_selection_method'] == 'sample':
            self.train_selection_method = SampleSelection()
            self.train_input_fn = partial(feats_or_probs, ret_feats=False)
        elif self.policy_cfg['train_selection_method'] == 'reduction-sample':
            self.train_selection_method = ReductionSampleSelection(**self.policy_cfg['selection_args'])
            self.train_input_fn = partial(feats_or_probs, ret_feats=True)
        elif self.policy_cfg['train_selection_method'] == 'combined-sample':
            self.train_selection_method = SampleSelection()
            self.train_input_fn = partial(combine_feats_and_probs, alpha=self.policy_cfg['alpha'])
        else:
            raise NotImplementedError()

        self.test_selection_method = None
        self.test_input_fn = None
        if self.policy_cfg['test_selection_method'] == 'greedy':
            self.test_selection_method = GreedySelection()
            self.test_input_fn = partial(feats_or_probs, ret_feats=False)
        elif self.policy_cfg['test_selection_method'] == 'random':
            self.test_selection_method = RandomSelection()
            self.test_input_fn = partial(feats_or_probs, ret_feats=False)
        elif self.policy_cfg['test_selection_method'] == 'eps-greedy':
            self.test_selection_method = EpsGreedySelection(self.policy_cfg['init_eps'],
                                                            self.policy_cfg['eps_schedule_cls'](
                                                                **self.policy_cfg['eps_schedule_cfg']
                                                            ))
            self.test_input_fn = partial(feats_or_probs, ret_feats=False)
        elif self.policy_cfg['test_selection_method'] == 'sample':
            self.test_selection_method = SampleSelection()
            self.test_input_fn = partial(feats_or_probs, ret_feats=False)
        elif self.policy_cfg['test_selection_method'] == 'reduction-sample':
            self.test_selection_method = ReductionSampleSelection(**self.policy_cfg['selection_args'])
            self.test_input_fn = partial(feats_or_probs, ret_feats=True)
        else:
            raise NotImplementedError()

        self.selection_method = self.train_selection_method
        self.input_fn = self.train_input_fn

        self.device = self.policy_cfg['device']
        self.to(self.device)

    def forward(self, state_actions, act_in=None, ret_ll=False, ret_entropy=False, ret_probs=False):
        logits = self.policy_network(state_actions).squeeze(-1)
        probs = torch.softmax(logits, dim=-1)

        if act_in is None:
            act = self.selection_method(self.input_fn(state_actions, probs))
        else:
            act = act_in

        out = []
        if ret_ll:
            out += [torch.log_softmax(logits, dim=-1)]
        if ret_entropy:
            out += [Multinomial(probs=probs).entropy()]
        if ret_probs:
            out += [probs]

        if len(out) >= 1:
            return act, *out
        else:
            return act

    def to_train(self):
        self.selection_method = self.train_selection_method
        self.input_fn = self.train_input_fn

    def to_eval(self):
        self.selection_method = self.test_selection_method
        self.input_fn = self.test_input_fn

    def step(self):
        self.train_selection_method.step()
        if 'alpha' in self.policy_cfg:
            self.policy_cfg['alpha'] *= self.policy_cfg['alpha_decay']

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.policy_network.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.policy_network.load_state_dict(torch.load(checkpoint_file))


class SPRValue(nn.Module):
    def __init__(self, value_cfg, net_cfg):
        super().__init__()
        self.value_cfg = value_cfg
        self.net_cfg = net_cfg

        self.value_network = self.net_cfg['network_cls'](**self.net_cfg['network_args'])
        if self.net_cfg['positive_init']:
            last_layer = self.value_network.last_layer()
            last_layer.weight.data = torch.abs(last_layer.weight.data)
            last_layer.bias.data = torch.abs(last_layer.bias.data)

        self.train_selection_method = None
        if self.value_cfg['train_selection_method'] == 'greedy':
            self.train_selection_method = GreedySelection()
        elif self.value_cfg['train_selection_method'] == 'random':
            self.train_selection_method = RandomSelection()
        elif self.value_cfg['train_selection_method'] == 'eps-greedy':
            self.train_selection_method = EpsGreedySelection(self.value_cfg['init_eps'],
                                                             self.value_cfg['eps_schedule_cls'](
                                                                 **self.value_cfg['eps_schedule_cfg']
                                                             ))
        elif self.value_cfg['train_selection_method'] == 'sample':
            self.train_selection_method = SampleSelection()
        else:
            raise NotImplementedError()

        self.test_selection_method = None
        if self.value_cfg['test_selection_method'] == 'greedy':
            self.test_selection_method = GreedySelection()
        elif self.value_cfg['test_selection_method'] == 'random':
            self.test_selection_method = RandomSelection()
        elif self.value_cfg['test_selection_method'] == 'eps-greedy':
            self.test_selection_method = EpsGreedySelection(self.value_cfg['init_eps'])
        elif self.value_cfg['test_selection_method'] == 'sample':
            self.test_selection_method = SampleSelection()
        else:
            raise NotImplementedError()

        self.selection_method = self.train_selection_method

        self.device = self.value_cfg['device']
        self.to(self.device)

        self.bspr_selection = BSPRSelection()
        self.bspr_prob = 0.9
        self.bspr_decay = 0.99

    def forward(self, state_actions, act_in=None, ret_values=False):
        values = self.value_network(state_actions).squeeze(-1)

        if act_in is None:
            probs = values / torch.sum(values, dim=-1).unsqueeze(-1)
            act = self.selection_method(probs)
            # bspr_act = self.bspr_selection(state_actions)
            # r = torch.rand(len(probs)) < self.bspr_prob
            # act[r] = bspr_act[r]

        else:
            act = act_in

        if ret_values:
            return act, values
        else:
            return act

    def step(self):
        self.train_selection_method.step()
        self.bspr_prob *= self.bspr_decay

    def to_train(self):
        self.selection_method = self.train_selection_method

    def to_eval(self):
        self.selection_method = self.test_selection_method

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.value_network.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.value_network.load_state_dict(torch.load(checkpoint_file))
