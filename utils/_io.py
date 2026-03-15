import numpy as np
import torch
import os
from utils._dataset import BMEPDataSet, LLDataSet, RFDataSet

def newick_to_edges(newick: str):
    edges = []
    return edges

def read_bmep_dataset(data_dir, dataset_file):
    d = np.loadtxt(os.path.join(data_dir, dataset_file))
    d = torch.from_numpy(d)
    return BMEPDataSet(d)

def read_ll_dataset(data_dir, dataset_file):
    pass

def read_rf_dataset(data_dir, dataset_file):
    pass