import common
from typing import List, Tuple
import numpy as np

class Environment:
    def __init__(
        self,
        move_type: str = common.MoveType.SPR.value,
        obj_type: str = common.ObjFunc.RF.value,
    ):
        self.move_type = move_type
        self.obj_type = obj_type


def create_batch(
    action: List[Tuple, Tuple],
    adj_old: np.ndarray,
):
    '''
        Input:
            - action_chosen: action chosen in the previous stage
            => a nested tuple to indicate action
            => ((pruned_dad_idx, pruned_child_idx), (regraft_dad_idx, regraft_child_idx))

            - adj_old: a np.ndarray matrix to indicate the previous stage of the tree

        Output:
            - adj_current: a np.ndarray to indicate the current stage of the tree

            (after applying action_chosen on adj_old)

            - action_list: the possible actions from the current stage of the tree

            list of tuple
            [   ((pruned_dad_idx, pruned_child_idx), (regraft_dad_idx, regraft_child_idx)),
                .....]

            - X_feat_current: the features of possible actions from the current stage of the tree
            => tensor feature  (X_feat_current shape: batch_size x num_feat)

            - y_true_current: reward(the possible stage) - reward(the current stage)
            => tensor of scalar number (y_true shape: batch_size x 1)

        what this function does:
            1. apply action on the previous stage of the tree -> to get current state => return adj_current
            2. from new state, considering all possible moves to get "new state"
                => return actions_current
            3. also calculating:
                - all possible features: encoding ("current state" + possible moves) into a vector
                - all possible rewards: "new state" - "current state"

                => return X_feat_current, y_true_current
    '''



    return adj_current, actions_current, X_feat_current, y_true_current