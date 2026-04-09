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
