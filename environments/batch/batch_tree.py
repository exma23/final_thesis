import copy
import torch

import numpy as np

from Solvers.BNNI_BSPR.fast_cpp import FastCpp
from environments.single.single_tree import Tree
from utils._compute import feat_normalization_fn

fast_obj = FastCpp()

class TrainingTree:
    def __init__(self, d, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), init_method='random'):
        self.device = device
        self._n_taxa = d.shape[-1]
        self._trees = [Tree(d_i) for d_i in d]
        if init_method == 'random':
            for t_i in self._trees:
                t_i.set_random_tree()
        if init_method == 'nj':
            for t_i in self._trees:
                t_i.set_nj_tree()
        self.n_moves, self._moves, self._features_tensor, self._tree_lens = fast_obj.get_features_batch_cpp(self._trees)

    def get_features(self, normalize=False):
        return feat_normalization_fn(self._features_tensor, self._n_taxa) if normalize else self._features_tensor
    def get_length(self):
        return torch.tensor(self._tree_lens)
    def get_likelihood(self):
        pass
    

    def action(self, action_idx):
        actions = torch.gather(self._moves, dim=1, index=action_idx.cpu().view(-1, 1, 1).repeat(1, 1, 4)).squeeze(
            0).squeeze(1)
        pruned, regrafted = actions[:, :2], actions[:, 2:]

        for i_i, (p_i, r_i) in enumerate(zip(pruned, regrafted)):
            self._trees[i_i].apply_SPR_move(tuple(p_i.cpu().tolist()), tuple(r_i.cpu().tolist()))

        self.n_moves, self._moves, self._features_tensor, self._tree_lens = fast_obj.get_features_batch_cpp(self._trees)

    def update_trees(self, mask, new_trees):
        self._trees = [copy.deepcopy(new_i) if m_i else t_i for new_i, t_i, m_i in
                       zip(new_trees._trees, self._trees, mask)]
        self.n_moves, self._moves, self._features_tensor, self._tree_lens = fast_obj.get_features_batch_cpp(self._trees)

    def set_trees(self, new_trees):
        self._trees = new_trees
        self.n_moves, self._moves, self._features_tensor, self._tree_lens = fast_obj.get_features_batch_cpp(self._trees)

    def BSPR(self, max_steps=10 ** 6, ret_steps=False, ret_traj=False):
        bspr_lens, bspr_trajs, bspr_steps = fast_obj.run_BSPR_batch(self._trees, max_steps)
        if max_steps < 10 ** 6:
            full_traj = np.ones((len(self._trees), max_steps + 1)) * bspr_lens.reshape((-1, 1))
            for i_i, (t_i, s_i) in enumerate(zip(bspr_trajs, bspr_steps)):
                full_traj[i_i, :s_i] = t_i

        out = []
        out += [bspr_steps] if ret_steps else []
        out += [full_traj] if ret_traj else []
        if len(out) > 0:
            return bspr_lens, *out

        return bspr_lens