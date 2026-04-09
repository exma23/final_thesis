from collections import deque
import random
import torch


class ReplayBuffer:
    def __init__(self, capacity, batch_size, device):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.device = device

    def add(self, chosen_feat, reward, best_next_feat):
        # chosen_feat:    (feat_dim,) tensor — features của action đã chọn
        # reward:         float
        # best_next_feat: (feat_dim,) tensor — features của best action ở next state
        self.memory.append((
            chosen_feat.cpu(),
            reward,
            best_next_feat.cpu(),
        ))

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        sa, r, nsa = zip(*batch)
        sa = torch.stack(sa).to(self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(-1)
        nsa = torch.stack(nsa).to(self.device)
        return sa, r, nsa

    def __len__(self):
        return len(self.memory)