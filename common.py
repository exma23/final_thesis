from enum import Enum

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
POSTFIX_GT = f"_gt.{NAME_TREEFILE}"
POSTFIX_START = f"_start.{NAME_TREEFILE}"
POSTFIX_MSA = ".phy"

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