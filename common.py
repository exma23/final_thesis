from enum import Enum

DATA_PATH = 'data'

class MoveType(Enum):
    SPR = "spr"
    NNI = "nni"

class ObjFunc(Enum):
    LIKELIHOOD = "likelihood"
    PARSIMONY = "parsimony"
    RF = "rf"

class Strategy(Enum):
    GREEDY = "greedy"
    SOFTMAX = "softmax"

INITIAL_ACTION = ((-1, -1), (-1, -1))