import random
from typing import List, Optional, Tuple
from utils._io import newick_to_edges
from utils._nj_tree import nj_torch
from utils._compute import feat_normalization_fn, normalized_rf_distance
import torch
import numpy as np

from Solvers.BNNI_BSPR.fast_cpp import FastCpp
fast_obj = FastCpp()

class Tree:
    def __init__(self, device, d: Optional[np.ndarray] = None, newick_str: str = None, obj: str = 'rf'):
        self.device = device
        self.d = d # distance matrix
        self.n_taxa = d.shape[0] if d is not None else None
        self.m = self.n_taxa * 2 - 2 if d is not None else None # Total number of nodes in the tree 
        self.newick_str = newick_str
        
        self.obj_val = None # rf distance or likelihood

        self.subtrees = []
        self.subtrees_length = {}

        self.edges = None
        self.edge_idx = None
        self.edge_length = None

        self.branch_length = {}
        self.max_branch_length = None
        self.features = {}

    def to_newick(self, labels=None):
        adj = {}
        for u, v in self.edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

        def traverse(node, parent):
            children = [c for c in adj.get(node, []) if c != parent]
            label = (labels[node] if labels is not None else str(node)) if node < self.n_taxa else ''
            if not children:
                return label
            return '(' + ','.join(traverse(c, node) for c in children) + ')' + label
        return traverse(self.n_taxa, -1) + ';'
    
    def set_random_tree(self):
        edges = [(0, self.n_taxa), (1, self.n_taxa), (2, self.n_taxa)]

        for i in range(3, self.n_taxa):
            u, v = random.choice(list(edges))
            edges.remove((u, v))
            edges.append((i, self.n_taxa + i - 2))
            edges.append((u, self.n_taxa + i - 2))
            edges.append((v, self.n_taxa + i - 2))

        self.adj = np.zeros((2 * self.n_taxa - 2, 2 * self.n_taxa - 2), dtype=int)
        for idx in edges:
            self.adj[idx[0], idx[1]] = self.adj[idx[1], idx[0]] = 1
        self.edges = edges
        self.subtrees = self.edges + [(e[1], e[0]) for e in self.edges]
        self.T = np.zeros((self.m, self.m), dtype=int)
    
    def set_raxml_tree(self, newick_str: str):
        pass

    def get_children(self, node: int, parent: int) -> List[Tuple[int, int]]:
        return [(node, l[1]) for l in self.subtrees if l[0] == node and l[1] != parent]

    def apply_SPR_move(
            self, 
            pruned: Tuple[int, int], 
            regrafted: Tuple[int, int]
        ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        children = self.get_children(pruned[0], pruned[1])
        self.edges.remove(children[0] if children[0] in self.edges else (children[0][1], children[0][0]))
        self.edges.remove(children[1] if children[1] in self.edges else (children[1][1], children[1][0]))
        new_pruned = (children[0][1], children[1][1]) if children[0][1] < children[1][1] else (
        children[1][1], children[0][1])
        self.edges.append(new_pruned)

        self.edges.remove(regrafted if regrafted in self.edges else (regrafted[1], regrafted[0]))
        child_1 = (pruned[0], regrafted[0]) if pruned[0] < regrafted[0] else (regrafted[0], pruned[0])
        self.edges.append(child_1)
        child_2 = (pruned[0], regrafted[1]) if pruned[0] < regrafted[1] else (regrafted[1], pruned[0])
        self.edges.append(child_2)

        self.subtrees = self.edges + [(e[1], e[0]) for e in self.edges]

        self.edge_idx = dict(zip([tuple(edge) for edge in self.subtrees], range(len(self.subtrees))))
        self.adj = np.zeros((self.m, self.m), dtype=np.int32)
        for edge in self.edges:
            if edge[0] < self.n_taxa:
                edge = edge[1], edge[0]

        self.adj = np.zeros_like(self.adj)
        for edge in self.edges:
            self.adj[edge[0], edge[1]] = self.adj[edge[1], edge[0]] = 1

        return new_pruned, child_1, child_2

    def compute_features_cpp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return fast_obj.get_features_cpp(self.d, self.edges, self.n_taxa, self.m)
    def run_bspr(self, max_steps=10**6):
        #t = training_Tree._tree
        adjs, objs, spr_counts = fast_obj.run_BSPR(self.d.astype(np.double), self.adj.astype(dtype=np.int32),
                                                   self.n_taxa, self.m, max_steps)

        fast_obj.free_result_memory()
        return objs, spr_counts


class TrainingTree:
    def __init__(self, d, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), init_method='random'):
        self.device = device
        self._n_taxa = d.shape[-1]

        self._initialize_tree(d, init_method)
        self._compute_features()

    def _initialize_tree(self, d, init_method):
        self._tree = Tree(d)
        if init_method == 'random':
            self._tree.set_random_tree()
        elif init_method == 'nj':
            self._tree.set_nj_tree()
        elif init_method == 'raxml':
            self._tree.set_raxml_tree()

    def _compute_features(self):
        self._actions, self._features_tensor = self._tree.compute_features_cpp()
        self._tree_len = self._features_tensor[0][0].item()

    def get_features(self, normalize=False):
        return feat_normalization_fn(self._features_tensor, self._n_taxa) if normalize else self._features_tensor
    def get_length(self):
        return self._tree_len
    def get_likelihood(self):
        return None
    def get_rf(self, true_tree: 'TrainingTree'):
        return normalized_rf_distance(self._tree.adj, true_tree._tree.adj, self._tree.n_taxa)
    
    def action(self, action_idx: int):
        pruned, regrafted = self._actions[action_idx][:2], self._actions[action_idx][2:]
        tree = self._tree.copy()
        tree.apply_SPR_move(tuple(pruned.cpu().tolist()), tuple(regrafted.cpu().tolist()))
        self._tree = tree
        self._actions, self._features_tensor = self._tree.compute_features_cpp()
        self._tree_len = self._features_tensor[0][0].item()