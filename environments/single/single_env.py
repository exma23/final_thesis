import copy
from functools import partial
from typing import Optional, Callable, Union, List, Type, Tuple

import torch
from environments.env_utils import spr_neighbor_features, \
    random_tree_init, nj_tree_init, raxml_tree_init, spr_action, \
    bmep_tree_l, ll_tree, rf_tree, \
    improve_over_best_score, improve_over_best_ll, improve_over_best_rf, \
    BMEPState, BMEPData, LLState, LLData, RFState, RFData
from utils._io import read_bmep_dataset, read_ll_dataset, read_rf_dataset

class BMEPEnvironment:
    def __init__(
        self, 
        move_type: str = 'spr', 
        tree_init_method: str = 'random', 
        obj_type: str = 'bmep-l', 
        reward_type: str = 'improve-best',
        normalize_reward: str = False, 
        rew_norm_scale: int = 1, 
        feat_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        fixed_start: bool = False,
        normalize_feats: bool = True
    ):
        self._move_type = move_type
        if obj_type == 'bmep-l':
            self._data_cls = BMEPData
            self._state_cls = BMEPState
            self._obj_fn = bmep_tree_l

        elif obj_type == 'raxml':
            self._data_cls = LLData
            self._state_cls = LLState
            self._obj_fn = ll_tree
        
        elif obj_type == 'rf':
            self._data_cls = RFData
            self._state_cls = RFState
            self._obj_fn = rf_tree
        
        # Neighbor features
        self._neighbor_features_fn = None
        self._action_fn = None
        if self._move_type == 'spr':
            self._neighbor_features_fn = spr_neighbor_features
            self._n_features = 20
            self._action_fn = spr_action
        else:
            raise NotImplementedError()

        self._feat_transform = lambda x: x
        if feat_transform is not None:
            self._feat_transform = feat_transform

        # Tree initialization method
        self._tree_init_method = tree_init_method
        self._tree_init_fn = None
        if self._tree_init_method == 'random':
            self._tree_init_fn = random_tree_init
        elif self._tree_init_method == 'nj':
            self._tree_init_fn = nj_tree_init
        elif self._tree_init_method == 'raxml':
            self._tree_init_fn = raxml_tree_init
        
        # Reward fn
        self._reward_type = reward_type
        self._reward_fn = None
        if self._reward_type == 'improve-best' and hasattr(self, '_obj_fn'):
            self._reward_fn = partial(improve_over_best_score,
                                      score_fn=self._obj_fn)  # anche qui ci giochiamo altre valutazioni della obj fun

        self._fixed_start = fixed_start
        self._init_tree = None

        self._normalize_reward = normalize_reward
        self._rew_norm_factor = 1
        self._rew_norm_scale = rew_norm_scale

        self._normalize_feats = normalize_feats

    def data_class(self):
        return self._data_cls

    def neighbors_features(self, problem_data: BMEPData, current_state: BMEPState) -> torch.Tensor:
        feats = self._neighbor_features_fn(problem_data, current_state, self._normalize_feats)
        return self._feat_transform(feats)

    def reset(self, problem_data):
        if self._init_tree is None or not self._fixed_start:
            self._init_tree = self._tree_init_fn(problem_data)

        init_tree = copy.deepcopy(self._init_tree)
        return self._state_cls(init_tree=init_tree, current_tree=copy.deepcopy(init_tree), best_tree=copy.deepcopy(init_tree))

    def _update_state(
        self, 
        problem_data, 
        current_state, 
        new_tree
    ):
        new_obj_vals = self._obj_fn(new_tree)
        best_obj_vals = self._obj_fn(current_state.best_tree)

        improved = new_obj_vals < best_obj_vals

        best_tree = copy.deepcopy(new_tree) if improved else current_state.best_tree

        return self._state_cls(init_tree=current_state.init_tree, best_tree=best_tree, current_tree=new_tree)

    def step(self, problem_data, current_state, action, bspr_baseline_steps=0):

        new_tree = self._action_fn(problem_data, current_state, action)
        rew_norm_factor = current_state.init_tree.get_length() if self._normalize_reward else 1

        reward = self._reward_fn(problem_data, current_state, new_tree, norm_factor=rew_norm_factor, norm_scale=self._rew_norm_scale)

        new_state = self._update_state(problem_data, current_state, new_tree)

        if bspr_baseline_steps > 0:
            bspr_len = new_state.current_tree.BSPR(bspr_baseline_steps)
            bspr_improvement = new_state.current_tree.get_length() - bspr_len
            bspr_value = (bspr_improvement / rew_norm_factor) * self._rew_norm_scale
            bspr_value += reward

            return new_state, reward, [bspr_len, bspr_improvement, bspr_value]

        return new_state, reward

    @staticmethod
    def read_dataset(self, dataset_file, data_dir='data/bmep'):
        if self._obj_type == 'bmep-l':
            return read_bmep_dataset(data_dir, dataset_file)
        elif self._obj_type == 'll':
            return read_ll_dataset(data_dir, dataset_file)
        elif self._obj_fn == 'rf':
            return read_rf_dataset(data_dir, dataset_file)

    def n_features(self):
        return self._n_features

