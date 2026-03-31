import yaml
from typing import List


class TrainConfig:
    def __init__(
        self,
        num_epoch: int,
        n_steps: int,
        learning_rate: float,
        in_features: int,
        out_features: int,
        device: str,
        layers: List[int],
        weight_decay: float,
    ):
        self.num_epoch = num_epoch
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.layers = layers
        self.weight_decay = weight_decay


class Config:
    def __init__(self, train_cfg: TrainConfig):
        self.train_cfg = train_cfg