import ctypes
import numpy as np
from typing import Tuple
import common

class FastCpp:
    def __init__(self, lib_path: str = 'feat_cpp/bridge.so'):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.get_state_action_c.argtypes = [
            ctypes.c_char_p,               # newick_str
            ctypes.POINTER(ctypes.c_int),  # action[3]
            ctypes.c_char_p,               # gt_newick_str
            ctypes.c_char_p,               # out_newick
            ctypes.c_int,                  # out_newick_cap
            ctypes.POINTER(ctypes.c_int),  # out_actions  (flat n*3)
            ctypes.POINTER(ctypes.c_double),  # out_feats (flat n*19)
            ctypes.POINTER(ctypes.c_double),  # out_rewards  (n,)
            ctypes.POINTER(ctypes.c_int),  # out_n_actions
        ]
        self.lib.get_state_action_c.restype = None

    def get_state_action(
        self,
        newick: str,
        action_chosen: Tuple[int, int, int],
        gt_newick: str,
        max_actions: int = 10000,
    ) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        out_newick_cap = 65536
        action_arr  = np.array(action_chosen, dtype=np.int32)
        out_newick  = ctypes.create_string_buffer(out_newick_cap)
        out_actions = np.zeros(max_actions * common.ACT_DIM, dtype=np.int32)
        out_feats   = np.zeros(max_actions * common.FEAT_DIM, dtype=np.float64)
        out_rewards = np.zeros(max_actions, dtype=np.float64)
        out_n       = ctypes.c_int(0)

        self.lib.get_state_action_c(
            newick.encode('utf-8'),
            action_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            gt_newick.encode('utf-8'),
            out_newick,
            ctypes.c_int(out_newick_cap),
            out_actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_feats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_rewards.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(out_n),
        )

        n = out_n.value
        return (
            out_newick.value.decode('utf-8'),
            out_actions[:n * common.ACT_DIM].reshape(n, common.ACT_DIM),
            out_feats[:n * common.FEAT_DIM].reshape(n, common.FEAT_DIM),
            out_rewards[:n],
        )