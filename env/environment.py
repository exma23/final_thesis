from feat_cpp.fastcpp import FastCpp
from utils import load_data_indices
from typing import List
import numpy as np


class Environment:
    def __init__(self, tree_indices: List[int], data_dir: str = 'data', lib_path: str = 'feat_cpp/bridge.so'):
        self.tree_indices = tree_indices
        self.cpp = FastCpp(lib_path)
        self.tree_start, self.tree_gt = load_data_indices(data_dir, tree_indices)
        self.tree_state = {}
        self.tree_step = {}
        self.tree_cur = np.random.choice(tree_indices)
