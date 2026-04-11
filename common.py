from enum import Enum
import os
import sys
import logging
from datetime import datetime, timezone, timedelta


DATA_PATH = 'data'
CHECKPOINT_PATH = 'ckps'

class MoveType(Enum):
    SPR = "spr"
    NNI = "nni"

class ObjFunc(Enum):
    LIKELIHOOD = "likelihood"
    PARSIMONY = "parsimony"
    RF = "robinson_foulds"

class Strategy(Enum):
    GREEDY = "greedy"
    SOFTMAX = "softmax"
    EPS_GREEDY = "eps_greedy"

class NetworkType(Enum):
    REINFORCE = "reinforce"
    QLearning = "q_learning"

ACT_DIM = 3  # [pruned, pruned_back, regraft]
FEAT_DIM = 19
INITIAL_ACTION = -1
DEFAULT_SPR_RADIUS = 10

NAME_TREEFILE = "newick"
POSTFIX_GT = f"gt.{NAME_TREEFILE}"
POSTFIX_START = f"start.{NAME_TREEFILE}"
POSTFIX_MSA = "data.phy"

FEAT_NAMES = [
    "total_bl",          # 0  — tree total BL (normalization anchor)
    "longest_bl",        # 1  — tree longest BL
    "prune_bl",          # 2  — prune edge BL
    "regraft_bl",        # 3  — regraft edge BL
    "topo_dist",         # 4  — topological distance (int)
    "bl_dist",           # 5  — branch-length distance
    "new_bl",            # 6  — new edge BL after move
    "n_leaves_a",        # 7  — leaves in pruned subtree
    "n_leaves_b",        # 8  — leaves in regraft subtree
    "n_leaves_b1",       # 9  — leaves on q-side
    "n_leaves_b2",       # 10 — leaves on q->back-side
    "total_bl_a",        # 11 — pruned subtree total BL
    "total_bl_b",        # 12 — regraft subtree total BL
    "total_bl_b1",       # 13 — q-side subtree BL
    "total_bl_b2",       # 14 — qback-side subtree BL
    "longest_bl_a",      # 15 — pruned subtree longest BL
    "longest_bl_b",      # 16 — regraft subtree longest BL
    "longest_bl_b1",     # 17 — q-side longest BL
    "longest_bl_b2",     # 18 — qback-side longest BL
]

LOG_PATH = 'logs'

logger = logging.getLogger("phylo_rl")
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logger.addHandler(_stdout_handler)
logger.setLevel(logging.INFO)


def setup_log_file(loss_type, obj_func, num_epoch, num_epoch_episode, n_steps):
    os.makedirs(LOG_PATH, exist_ok=True)
    now = datetime.now(timezone(timedelta(hours=7)))
    ts = now.strftime("%y%m%d_%H%M%S")
    fname = f"{ts}_{loss_type}_{obj_func}_ep{num_epoch}_epe{num_epoch_episode}_steps{n_steps}.log"
    fh = logging.FileHandler(os.path.join(LOG_PATH, fname))
    fh.setFormatter(logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

class RewardDefinition(Enum):
    RAW_DELTA = 'raw'
    NORMALIZED = 'normalized'
    RELU = 'relu'

def reward_definition(old_score, current_score, type: str, scale_norm: int):
    if type == RewardDefinition.RAW_DELTA.value:
        return old_score - current_score
    elif type == RewardDefinition.NORMALIZED.value:
        return ((current_score - old_score)/(-old_score))*scale_norm
    elif type == RewardDefinition.RELU.value:
        return max(current_score- old_score, 0)
    else:
        raise ValueError('Not implemented this reward definition')

