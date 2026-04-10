from dataclasses import dataclass
from typing import List
import common


@dataclass
class TrainConfig:
    num_epoch: int
    learning_rate: float
    in_features: int
    out_features: int
    device: str
    layers: List[int]
    weight_decay: float
    num_epoch_episode: int = 5        # ← NEW: train epochs per episode

@dataclass
class InferConfig:
    pass


@dataclass
class RLConfig:
    n_steps: int
    loss_type: str
    gamma: float = 0.9
    # REINFORCE specific
    pg_episodes_in_memory: int = 2    # ← NEW
    pg_epochs: int = 10               # ← NEW
    pg_batch_size: int = 64           # ← NEW
    # Q-learning specific
    buffer_size: int = 10000
    batch_size: int = 32
    target_update_freq: int = 100
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000


@dataclass
class PhyloConfig:
    move_type: str
    obj_func: str
    spr_radius: int = common.DEFAULT_SPR_RADIUS
    optimize_bl: bool = True          # ← THÊM
    bl_opt_every: int = 5              # ← THÊM: optimize mỗi N bước
    bl_opt_model: str = "GTR+G"       # ← NEW: model for raxml-ng

class Config:
    def __init__(self, train_cfg: TrainConfig, rl_cfg: RLConfig, phylo_cfg: PhyloConfig):
        self.train_cfg = train_cfg
        self.rl_cfg = rl_cfg
        self.phylo_cfg = phylo_cfg
