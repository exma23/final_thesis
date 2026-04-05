import numpy as np
import torch
import common
from common import Strategy


class Agent:
    def __init__(self, strategy: str = Strategy.GREEDY.value):
        self.strategy = strategy

    def choose(self, actions_current: np.ndarray, y_pred_current: torch.Tensor):
        """
        actions_current: np.ndarray of shape [n, 4]  -- [pd_idx, pn_idx, rd_idx, rn_idx]
        y_pred_current:  torch.Tensor of shape [n, 1] -- predicted scores

        Returns: ((pd_idx, pn_idx), (rd_idx, rn_idx))
        """
        scores = y_pred_current.detach().squeeze(-1)  # shape [n]

        if self.strategy == Strategy.GREEDY.value:
            idx = int(torch.argmax(scores).item())
        elif self.strategy == Strategy.SOFTMAX.value:
            probs = torch.softmax(scores, dim=-1)
            idx = int(torch.multinomial(probs, num_samples=1).item())
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        row = actions_current[idx]
        return (int(row[0]), int(row[1]))