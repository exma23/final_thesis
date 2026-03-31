import abc
import torch

class SelectionMethod:
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def step(self, *args, **kwargs):
        pass


class GreedySelection(SelectionMethod):
    def __call__(self, probs):
        return torch.argmax(probs, dim=-1)

    def step(self):
        pass


class RandomSelection(SelectionMethod):
    def __call__(self, probs):
        return torch.randint(low=0, high=len(probs[0]), size=(len(probs), 1))

    def step(self, *args, **kwargs):
        pass


class EpsGreedySelection(SelectionMethod):
    def __init__(self, init_eps, eps_schedule=None):
        self._eps_schedule = eps_schedule
        if self._eps_schedule is None:
            self._eps_schedule = lambda x: x

        self._eps = init_eps

    def __call__(self, probs):
        r = torch.rand(len(probs)) < self._eps
        act_idx = torch.argmax(probs, dim=-1)
        act_idx[r] = torch.randint(low=0, high=len(probs[0]), size=(sum(r),)).to(probs.device)
        return act_idx

    def step(self):
        self._eps = self._eps_schedule(self._eps)


class SampleSelection(SelectionMethod):
    def __call__(self, probs):
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def step(self):
        pass

class ReductionSampleSelection(SelectionMethod):
    def __init__(self, temp):
        self._temp = temp
    def __call__(self, feats):
        probs = torch.softmax((1 - feats[..., 7]) / self._temp, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def step(self):
        pass

class BSPRSelection(SelectionMethod):
    def __call__(self, feats):
        act_idx = torch.argmin(feats[..., 7].view(len(feats), -1), dim=-1)
        return act_idx

    def step(self):
        pass