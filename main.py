import os
import common
from agents.agent import Agent
from env.environment import Environment
from trainer.trainer import Trainer
from utils import get_tree_indices, load_config


if __name__ == '__main__':
    config, raw = load_config("config.yaml")
    train_cfg = config.train_cfg
    rl_cfg = config.rl_cfg
    paths = raw["paths"]
    agent_cfg = raw["agent"]

    tree_indices = get_tree_indices(paths["data_dir"])
    print(f'Found {len(tree_indices)} trees: {tree_indices}')

    env = Environment(
        tree_indices=tree_indices,
        data_dir=paths["data_dir"],
        lib_path=paths.get("bridge_lib", "feat_cpp/bridge.so"),
    )

    agent = Agent(
        network_type=rl_cfg.loss_type,
        in_features=train_cfg.in_features,
        out_features=train_cfg.out_features,
        device=train_cfg.device,
        layers=train_cfg.layers,
        strategy_train=agent_cfg["strategy_train"],
        strategy_infer=agent_cfg["strategy_infer"],
        epsilon_start=rl_cfg.epsilon_start,
        epsilon_end=rl_cfg.epsilon_end,
        epsilon_decay_steps=rl_cfg.epsilon_decay_steps,
    )

    trainer = Trainer(config=config, env=env, agent=agent)
    trainer.train()

    ckp_dir = paths.get("checkpoint_dir", common.CHECKPOINT_PATH)
    os.makedirs(ckp_dir, exist_ok=True)
    save_path = os.path.join(ckp_dir, 'model_final.pt')
    agent.save_checkpoint(save_path)
    print(f'Model saved to {save_path}')