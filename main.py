import os
import torch
import common
from common import Strategy, NetworkType
from agents.agent import Agent
from env.environment import Environment
from trainer.trainer import Trainer
from trainer.config import Config, TrainConfig, RLConfig, PhyloConfig
from utils import get_tree_indices


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tree_indices = get_tree_indices(common.DATA_PATH)
    print(f'Found {len(tree_indices)} trees: {tree_indices}')

    # --- Loss type ---
    loss_type = NetworkType.REINFORCE.value  # hoặc NetworkType.QLearning.value

    # --- Config ---
    train_cfg = TrainConfig(
        num_epoch=500,
        learning_rate=1e-3,
        in_features=common.FEAT_DIM,
        out_features=1,
        device=device,
        layers=[128, 128, 128, 128],
        weight_decay=1e-4,
    )

    rl_cfg = RLConfig(
        n_steps=30,
        loss_type=loss_type,
    )

    phylo_cfg = PhyloConfig(
        move_type=common.MoveType.SPR.value,
        obj_func=common.ObjFunc.RF.value,
        spr_radius=7,
    )

    config = Config(train_cfg=train_cfg, rl_cfg=rl_cfg, phylo_cfg=phylo_cfg)

    # --- Environment ---
    env = Environment(tree_indices=tree_indices, data_dir=common.DATA_PATH)

    # --- Agent ---
    strategy_train = Strategy.SOFTMAX.value if loss_type == NetworkType.REINFORCE.value \
        else Strategy.EPS_GREEDY.value

    agent = Agent(
        network_type=loss_type,
        in_features=train_cfg.in_features,
        out_features=train_cfg.out_features,
        device=device,
        layers=train_cfg.layers,
        strategy_train=strategy_train,
        strategy_infer=Strategy.GREEDY.value,
    )

    # --- Train ---
    trainer = Trainer(config=config, env=env, agent=agent)
    trainer.train()

    # --- Save ---
    os.makedirs(common.CHECKPOINT_PATH, exist_ok=True)
    save_path = os.path.join(common.CHECKPOINT_PATH, 'model_final.pt')
    agent.save_checkpoint(save_path)
    print(f'Model saved to {save_path}')