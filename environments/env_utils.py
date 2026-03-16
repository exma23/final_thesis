from dataclasses import dataclass
import torch
import subprocess
import tempfile
import os
from concurrent.futures import ProcessPoolExecutor
from environments.single.single_tree import TrainingTree
from environments.single.single_env import BMEPEnvironment
import environments.batch.batch_tree as batch_env
from agents.agent import SPRPolicy
from utils._lib_call import call_raxml

@dataclass
class BMEPState:
    init_tree: TrainingTree
    best_tree: TrainingTree  # BMEPTree
    current_tree: TrainingTree

@dataclass
class LLState:
    init_tree: TrainingTree
    best_tree: TrainingTree  # BMEPTree
    current_tree: TrainingTree
    init_ll: torch.Tensor
    best_ll: torch.Tensor
    current_ll: torch.Tensor
    alignment_file: str
    model: str
    labels: list
    
@dataclass
class RFState:
    pass


@dataclass
class BMEPData:
    dist: torch.Tensor
@dataclass
class LLData:
    alignment_path: str
    model: str
@dataclass
class RFData:
    dist: torch.Tensor
    alignment_path: str
    model: str


def spr_neighbor_features(current_state, normalize=False):
    return current_state.current_tree.get_features(normalize)
def spr_action(current_state, action):
    current_state.current_tree.action(action)
    return current_state.current_tree


def improve_over_best_score(problem_data, current_state, new_state, score_fn, norm_factor=1, norm_scale=1):
    new_state_score = score_fn(problem_data, current_state.best_tree) - score_fn(problem_data, new_state)
    new_state_score = (new_state_score / norm_factor) * norm_scale
    return torch.maximum(new_state_score, torch.zeros_like(new_state_score))
def improve_over_best_ll(problem_data, current_state, new_state, score_fn, norm_factor=1, norm_scale=1):
    new_state_score = score_fn(problem_data, new_state) - current_state.best_ll
    new_state_score = (new_state_score / norm_factor) * norm_scale
    return torch.maximum(new_state_score, torch.zeros_like(new_state_score))
def improve_over_best_rf(problem_data, current_state, new_state, score_fn, norm_factor=1, norm_scale=1):
    pass

def random_tree_init(problem_data):
    return TrainingTree(problem_data.dist.numpy())
def nj_tree_init(problem_data):
    return TrainingTree(problem_data.dist.numpy(), init_method='nj')
def raxml_tree_init(problem_data):
    return TrainingTree(problem_data.dist.numpy(), init_method='raxml')


def bmep_tree_l(problem_data: BMEPData, tree: TrainingTree):
    return tree.get_length()

def ll_tree(problem_data: LLData, tree: batch_env.TrainingTree):
    args = []
    for i, t_i in enumerate(tree._trees):
        labels_i = problem_data.labels[i]
        args.append((t_i.to_newick(labels_i), problem_data.alignment_path, problem_data.model, labels_i))
    with ProcessPoolExecutor(max_workers=64) as pool:
        lls = list(pool.map(call_raxml, args))
    return torch.tensor(lls, dtype=torch.float32)

def rf_tree(tree, true_tree):
    return tree.get_rf(true_tree)
