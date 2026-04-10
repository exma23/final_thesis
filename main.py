import os
import argparse
import common
from agents.agent import Agent
from env.environment import Environment
from trainer.trainer import Trainer
from utils import get_tree_indices, load_config

log = common.logger


def parse_args():
    parser = argparse.ArgumentParser(description="Phylogenetic RL Training")

    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )

    # ── Training ──
    parser.add_argument("--num-epoch", dest="num_epoch", type=int)
    parser.add_argument("--lr", dest="learning_rate", type=float)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float)
    parser.add_argument("--num-epoch-episode", dest="num_epoch_episode", type=int)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--layers", type=int, nargs="+",
                        help="Hidden layer sizes, e.g. --layers 128 128 128 128")

    # ── RL ──
    parser.add_argument("--loss-type", dest="loss_type",
                        choices=["reinforce", "q_learning"])
    parser.add_argument("--n-steps", dest="n_steps", type=int)
    parser.add_argument("--gamma", type=float)

    # ── Agent ──
    parser.add_argument("--strategy-train", dest="strategy_train",
                        choices=["greedy", "softmax", "eps_greedy"])
    parser.add_argument("--strategy-infer", dest="strategy_infer",
                        choices=["greedy", "softmax", "eps_greedy"])

    # ── Phylo ──
    parser.add_argument("--obj-func", dest="obj_func",
                        choices=["robinson_foulds", "likelihood", "parsimony"])
    parser.add_argument("--spr-radius", dest="spr_radius", type=int)
    parser.add_argument("--bl-opt-model", dest="bl_opt_model")
    parser.add_argument("--no-optimize-bl", dest="optimize_bl",
                        action="store_false")

    # ── Paths ──
    parser.add_argument("--data-dir", dest="data_dir")
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir")

    return parser.parse_args()


def run(args):
    # 1. Load defaults from YAML
    config, raw = load_config(args.config)
    train_cfg = config.train_cfg
    rl_cfg = config.rl_cfg
    phylo_cfg = config.phylo_cfg
    paths = raw["paths"]
    agent_cfg = raw["agent"]

    # 2. CLI overrides
    if args.num_epoch is not None:
        train_cfg.num_epoch = args.num_epoch
    if args.learning_rate is not None:
        train_cfg.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        train_cfg.weight_decay = args.weight_decay
    if args.num_epoch_episode is not None:
        train_cfg.num_epoch_episode = args.num_epoch_episode
    if args.device is not None:
        import torch
        train_cfg.device = ("cuda" if torch.cuda.is_available() else "cpu") \
            if args.device == "auto" else args.device
    if args.layers is not None:
        train_cfg.layers = args.layers

    if args.loss_type is not None:
        rl_cfg.loss_type = args.loss_type
    if args.n_steps is not None:
        rl_cfg.n_steps = args.n_steps
    if args.gamma is not None:
        rl_cfg.gamma = args.gamma

    if args.strategy_train is not None:
        agent_cfg["strategy_train"] = args.strategy_train
    if args.strategy_infer is not None:
        agent_cfg["strategy_infer"] = args.strategy_infer

    if args.obj_func is not None:
        phylo_cfg.obj_func = args.obj_func
    if args.spr_radius is not None:
        phylo_cfg.spr_radius = args.spr_radius
    if args.bl_opt_model is not None:
        phylo_cfg.bl_opt_model = args.bl_opt_model
    if args.optimize_bl is False:
        phylo_cfg.optimize_bl = False

    data_dir = args.data_dir or paths.get("data_dir", common.DATA_PATH)
    ckp_dir = args.checkpoint_dir or paths.get("checkpoint_dir", common.CHECKPOINT_PATH)
    bridge_lib = paths.get("bridge_lib", "feat_cpp/bridge.so")

    # 3. Auto strategy: softmax for REINFORCE, eps_greedy for Q-learning
    if args.strategy_train is None and args.loss_type is not None:
        agent_cfg["strategy_train"] = (
            common.Strategy.SOFTMAX.value
            if rl_cfg.loss_type == common.NetworkType.REINFORCE.value
            else common.Strategy.EPS_GREEDY.value
        )

    # 3b. Setup log file
    common.setup_log_file(
        rl_cfg.loss_type, phylo_cfg.obj_func,
        train_cfg.num_epoch, train_cfg.num_epoch_episode, rl_cfg.n_steps)

    # 4. Log config summary
    log.info("=== Config ===")
    log.info(f"  loss_type:    {rl_cfg.loss_type}")
    log.info(f"  obj_func:     {phylo_cfg.obj_func}")
    log.info(f"  num_epoch:    {train_cfg.num_epoch}")
    log.info(f"  n_steps:      {rl_cfg.n_steps}")
    log.info(f"  lr:           {train_cfg.learning_rate}")
    log.info(f"  device:       {train_cfg.device}")
    log.info(f"  strategy:     {agent_cfg['strategy_train']}")
    log.info(f"  data_dir:     {data_dir}")
    log.info("==============")

    # 5. Build components
    tree_indices = get_tree_indices(data_dir)
    log.info(f"Found {len(tree_indices)} trees: {tree_indices}")

    env = Environment(
        tree_indices=tree_indices,
        data_dir=data_dir,
        lib_path=bridge_lib,
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

    # 6. Train
    trainer = Trainer(config=config, env=env, agent=agent, save_dir=ckp_dir)
    trainer.train()

    # 7. Save: model_{loss_type}_{obj_func}.pt
    os.makedirs(ckp_dir, exist_ok=True)
    model_name = (
        f"model_{rl_cfg.loss_type}_{phylo_cfg.obj_func}"
        f"_ep{train_cfg.num_epoch}"
        f"_epe{train_cfg.num_epoch_episode}"
        f"_steps{rl_cfg.n_steps}.pt"
    )
    save_path = os.path.join(ckp_dir, model_name)
    agent.save_checkpoint(save_path)
    log.info(f"Model saved to {save_path}")


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()