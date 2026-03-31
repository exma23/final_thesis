import ctypes
import numpy as np
from typing import Tuple


class FastCpp:
    def __init__(self, lib_path: str = 'feat_cpp/bridge.so'):
        self.lib = ctypes.CDLL(lib_path)

        # void get_state_action_c(
        #     const char* newick_str,
        #     const int*  action,
        #     const char* gt_newick_str,
        #     char*       out_newick,
        #     int         out_newick_cap,
        #     int*        out_actions,
        #     double*     out_feats,
        #     double*     out_rewards,
        #     int*        out_n_actions
        # )
        self.lib.get_state_action_c.argtypes = [
            ctypes.c_char_p,                  # newick_str
            ctypes.POINTER(ctypes.c_int),     # action[4]
            ctypes.c_char_p,                  # gt_newick_str
            ctypes.c_char_p,                  # out_newick
            ctypes.c_int,                     # out_newick_cap
            ctypes.POINTER(ctypes.c_int),     # out_actions
            ctypes.POINTER(ctypes.c_double),  # out_feats
            ctypes.POINTER(ctypes.c_double),  # out_rewards
            ctypes.POINTER(ctypes.c_int),     # out_n_actions
        ]
        self.lib.get_state_action_c.restype = None

    def get_state_action(
        self,
        newick: str,
        action_chosen: Tuple[Tuple[int, int], Tuple[int, int]],
        gt_newick: str,
        max_actions: int = 10000,
    ) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply a chosen SPR action on the current tree, then enumerate all
        possible next moves from the resulting state.

        Parameters
        ----------
        newick        : current tree as a Newick string
        action_chosen : ((prune_dad_idx, prune_child_idx),
                         (regraft_dad_idx, regraft_child_idx))
                        Use ((-1, -1), (-1, -1)) for the initial step.
        gt_newick     : ground-truth tree as a Newick string
        max_actions   : upper bound on number of SPR moves (pre-allocation)

        Returns
        -------
        newick_current : new tree state as a Newick string
        action_list    : (n_actions, 4) int32  — node indices of each move
        X_feat_current : (n_actions, 20) float64 — feature vectors
        y_true_current : (n_actions,)   float64  — RF reward per move
        """
        feat_dim       = 20
        out_newick_cap = 65536

        action_arr = np.array([
            action_chosen[0][0], action_chosen[0][1],
            action_chosen[1][0], action_chosen[1][1],
        ], dtype=np.int32)

        out_newick_buf = ctypes.create_string_buffer(out_newick_cap)
        out_actions    = np.zeros((max_actions, 4),        dtype=np.int32)
        out_feats      = np.zeros((max_actions, feat_dim), dtype=np.float64)
        out_rewards    = np.zeros((max_actions,),          dtype=np.float64)
        out_n_actions  = ctypes.c_int(0)

        self.lib.get_state_action_c(
            newick.encode('utf-8'),
            action_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            gt_newick.encode('utf-8'),
            out_newick_buf,
            ctypes.c_int(out_newick_cap),
            out_actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_feats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_rewards.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(out_n_actions),
        )

        n = out_n_actions.value
        return (
            out_newick_buf.value.decode('utf-8'),
            out_actions[:n],
            out_feats[:n],
            out_rewards[:n],
        )