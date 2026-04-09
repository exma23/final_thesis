import torch
import common
from common import Strategy, NetworkType
from networks.ff_net import FFNet
from agents.selection import GreedySelection, SampleSelection, EpsGreedySelection


class Agent:
    def __init__(
        self,
        network_type: str,
        in_features: int,
        out_features: int,
        device: str,
        layers,
        strategy_train: str = Strategy.EPS_GREEDY.value,
        strategy_infer: str = Strategy.GREEDY.value,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 5000,
    ):
        self.device = device
        self.network_type = network_type
        self.network = FFNet(in_features, out_features, device, layers)
        self.network_target = None
        if network_type == NetworkType.QLearning.value:
            self.network_target = FFNet(in_features, out_features, device, layers)
            self.network_target.load_state_dict(self.network.state_dict())
            self.network_target.eval()

        self.train_strategy = make_selection(strategy_train, epsilon_start)
        self.infer_strategy = make_selection(strategy_infer)
        self._is_training = True

        # --- Epsilon schedule (cho EpsGreedy) ---
        self._eps_start = epsilon_start
        self._eps_end = epsilon_end
        self._eps_decay_steps = epsilon_decay_steps
        self._step_count = 0

    def choose(self, actions, features):
        with torch.no_grad():
            scores = self.network(features).squeeze(-1)  # (n_actions,)

        strategy = self.train_strategy if self._is_training else self.infer_strategy

        if isinstance(strategy, EpsGreedySelection):
            strategy._eps = self._current_epsilon()
            if self._is_training:
                self._step_count += 1

        if isinstance(strategy, SampleSelection):
            probs = torch.softmax(scores, dim=-1).unsqueeze(0)
            idx = int(strategy(probs).item())
        else:
            idx = int(strategy(scores.unsqueeze(0)).item())

        row = actions[idx]
        return tuple(int(x) for x in row), idx

    def forward(self, features):
        return self.network(features)

    def parameters(self):
        return self.network.parameters()

    def to_train(self):
        self._is_training = True
        self.network.train()

    def to_eval(self):
        self._is_training = False
        self.network.eval()

    def update_target(self):
        if self.network_target is not None:
            self.network_target.load_state_dict(self.network.state_dict())

    def soft_update_target(self, tau=0.005):
        if self.network_target is not None:
            for tp, lp in zip(self.network_target.parameters(),
                              self.network.parameters()):
                tp.data.copy_(tau * lp.data + (1 - tau) * tp.data)

    def save_checkpoint(self, path):
        state = {'network': self.network.state_dict()}
        if self.network_target is not None:
            state['network_target'] = self.network_target.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(state['network'])
        if self.network_target is not None and 'network_target' in state:
            self.network_target.load_state_dict(state['network_target'])

    def _current_epsilon(self):
        frac = min(self._step_count / max(self._eps_decay_steps, 1), 1.0)
        return self._eps_start + frac * (self._eps_end - self._eps_start)

    @property
    def epsilon(self):
        return self._current_epsilon()


def make_selection(strategy, epsilon=None):
    if strategy == Strategy.GREEDY.value:
        return GreedySelection()
    elif strategy == Strategy.SOFTMAX.value:
        return SampleSelection()
    elif strategy == Strategy.EPS_GREEDY.value:
        return EpsGreedySelection(epsilon or 1.0)
    raise ValueError(f"Unknown strategy: {strategy}")
