import itertools
import random
import numpy as np
import torch

class BMEPDataSet:
    def __init__(self, d):
        self._d = d
        self._size = self._d.shape[0]

    def n_taxa(self):
        return self._size
    
    def possible_instances(self, dim):
        dim = dim if dim <= self._size else self._size
        return len(list(itertools.combinations(range(self._size), dim)))

    def random_instance(self, dim):
        dim = dim if dim <= self._size else self._size
        idx = random.sample(range(self._size), k=dim)

        return self._d[idx, :][:, idx]

    def all_instances(self, dim):
        dim = dim if dim <= self._size else self._size
        instances = []
        for idx in itertools.combinations(range(self._size), dim):
            instances.append(self._d[idx, :][:, idx].unsqueeze(0))

        return torch.cat(instances, dim=0)

class LLDataSet:
    pass

class RFDataSet:
    pass
