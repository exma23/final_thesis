from functools import partial

from environments.single.single_env import BMEPEnvironment
from environments.env_utils import spr_neighbor_features, spr_action, \
    random_tree_init, nj_tree_init, raxml_tree_init, \
    bmep_tree_l, ll_tree, rf_tree, \
    improve_over_best_score, improve_over_best_ll, improve_over_best_rf, \
    BMEPState, BMEPData, LLState, LLData, RFState, RFData
from utils._validate import batched_validation
import torch

class BatchedBMEPEnvironment(BMEPEnvironment):

    def __init__(
        self, 
        move_type='spr', 
        tree_init_method='random', 
        obj_type='bmep-l', 
        reward_type='improve-best',
        normalize_reward=False, 
        rew_norm_scale=1, 
        feat_transform=None, 
        fixed_start=False,
        normalize_feats=True
    ):
        super(BatchedBMEPEnvironment, self).__init__(move_type, tree_init_method, obj_type, reward_type,
                                                     normalize_reward, rew_norm_scale, feat_transform, fixed_start,
                                                     normalize_feats)
        self._move_type = move_type
        self._obj_type = obj_type
        self._maximize_obj = False
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

        self.validation_fn = batched_validation

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
        if self._reward_type == 'improve-best':
            if obj_type == 'raxml':
                self._reward_fn = partial(improve_over_best_ll, score_fn=self._obj_fn)
            else:
                self._reward_fn = partial(improve_over_best_score, score_fn=self._obj_fn)

        self._fixed_start = fixed_start
        self._init_tree = None

        self._normalize_reward = normalize_reward
        self._rew_norm_factor = 1
        self._rew_norm_scale = rew_norm_scale

        self._normalize_feats = normalize_feats
        
    def make_batch_data(self, dist_tensor, orig_idxs_list):
        if self._obj_type == 'raxml-ll':
            per_instance_labels = [[self._fasta_labels[j] for j in idxs] for idxs in orig_idxs_list]
            return MLData(dist=dist_tensor, alignment_file=self._alignment_file,
                        model=self._model, labels=per_instance_labels)
        from environments.batched_bmep.batched_bmep_utils import BMEPData
        return BMEPData(dist=dist_tensor)

    def _update_state(self, problem_data, current_state, new_tree):
        new_obj_vals = self._obj_fn(problem_data, new_tree)
        if self._maximize_obj:
            # use cached best_ll to avoid re-evaluating best_tree
            best_obj_vals = current_state.best_ll
            improved = new_obj_vals > best_obj_vals
            current_state.best_ll = torch.where(improved, new_obj_vals, best_obj_vals)
        else:
            best_obj_vals = self._obj_fn(problem_data, current_state.best_tree)
            improved = new_obj_vals < best_obj_vals

        current_state.best_tree.update_trees(improved, new_tree)
        current_state.current_tree = new_tree

        return current_state
    
    def reset(self, problem_data):
        import copy
        if self._init_tree is None or not self._fixed_start:
            self._init_tree = self._tree_init_fn(problem_data)

        init_tree = copy.deepcopy(self._init_tree)
        state = self._state_cls(init_tree=init_tree,
                                current_tree=copy.deepcopy(init_tree),
                                best_tree=copy.deepcopy(init_tree))
        if self._maximize_obj:
            state.best_ll = self._obj_fn(problem_data, state.best_tree)
        return state
