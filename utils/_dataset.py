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

def dataset_partition(bmep_dataset, prop):
    n1 = int(bmep_dataset.n_taxa() * prop)
    n1_samples = random.sample(range(bmep_dataset.n_taxa()), n1)

    dset1, dset2 = split_dataset(bmep_dataset, n1_samples)

    return dset1, dset2, n1_samples

def split_dataset(bmep_dataset, split_set):
    n1_samples = split_set
    n2_samples = [i_i for i_i in range(bmep_dataset.n_taxa()) if i_i not in n1_samples]

    d1 = bmep_dataset._d[n1_samples, :][:, n1_samples]
    d2 = bmep_dataset._d[n2_samples, :][:, n2_samples]

    dset1 = BMEPDataSet(d1)
    dset2 = BMEPDataSet(d2)

    return dset1, dset2