import os
import random
import uuid
import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from environments.single.single_env import BMEPEnvironment
from environments.batch.batch_env import BatchedBMEPEnvironment
from environments.bmep.bmep_utils import sa_feat_norm, dataset_partition, zero_bspr_len

from agents.agent import SPRPolicy
from networks.attention_net import AttentionNet
from networks.ff_net import FFNet

from trainers.pg_trainer import PGTrainer
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--config', type=str, default='default_config.yml')

args = parser.parse_args()
cfg = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

log = cfg['log']

if log:
    logger = neptune.init_run(
        project="federicocamerota/phyloRL",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNDM4YTIyMS0zM2M0LTQ5YjItYjBhZi00NDNhZmFjYWZmYTMifQ==",
        mode=cfg['log_mode'],
        flush_period=10
    )
else:
    logger = None


seed = cfg['random_seed'] = args.seed
random.seed(seed)

cfg['run_id'] = str(uuid.uuid4())


## Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg['device'] = device
device = torch.device(device)

env_cfg = cfg['env_cfg']

#### Feature transformation
if env_cfg['feat_transform'] is not None:
    if env_cfg['feat_transform'] == 'sa_normalization':
        env_cfg['feat_transform'] = sa_feat_norm
    if env_cfg['feat_transform'] == 'zero_bspr_len':
        env_cfg['feat_transform'] = zero_bspr_len
    else:
        raise NotImplementedError()

## Environment
if cfg['batched']:
    bmep = BatchedBMEPEnvironment(**env_cfg)
else:
    bmep = BMEPEnvironment(**env_cfg)

# Network configs
network_cfg = cfg['network_cfg']
network_cfg['network_args']['device'] = device

if network_cfg['network_cls'] == 'AttentionNet':
    network_cfg['network_cls'] = AttentionNet
elif network_cfg['network_cls'] == 'FFNet':
    network_cfg['network_cls'] = FFNet
else:
    raise NotImplementedError()

# Training configs
train_cfg = cfg['train_cfg']

## Policy configs
policy_cfg = cfg['policy_cfg']
policy_cfg['device'] = device

## Optimizer configs
optimizer_cfg = cfg['optimizer_cfg']
if optimizer_cfg['optimizer_cls'] == 'Adam':
    optimizer_cfg['optimizer_cls'] = Adam
else:
    raise NotImplementedError()

if optimizer_cfg['lr_scheduler'] is not None:
    if optimizer_cfg['lr_scheduler'] == 'StepLR':
        optimizer_cfg['lr_scheduler'] = StepLR
    else:
        raise NotImplementedError()

## Trainer configs
trainer_cfg = cfg['trainer_cfg']
trainer_cfg['device'] = device
trainer_cfg['train_cfg'] = train_cfg
trainer_cfg['optimizer_cfg'] = optimizer_cfg

## CONFIG LOGGING
def repr_dict(d):
    return {k: repr_dict(v) if isinstance(v, dict) else repr(v) for k, v in d.items()}
## Log configs
if logger is not None:
    logger['cfg'] = repr_dict(cfg)
## TAGS
    tags = cfg['tags']
    logger['sys/tags'].add(tags)

## Data
data = bmep.read_dataset(train_cfg["dataset_filename"])
train_data, test_data, train_idxs = dataset_partition(data, train_cfg['dataset_partition_prop'])


## Agent
agent = SPRPolicy(policy_cfg, network_cfg)
if 'checkpoint' in train_cfg and train_cfg['checkpoint'] is not None:
    agent.load_checkpoint(train_cfg['checkpoint'])
#agent.load_checkpoint('experiments/checkpoints/checkpoints_dqn_trainer_db44ed78-6908-4561-8457-8ecc03406404/agent_epoch980_perf0.28935208905442233')
#agent.load_checkpoint('experiments/checkpoints/checkpoints_dqn_trainer_cbf8056f-78b9-47d3-a3ed-4f9502f5eba3/best_agent_epoch256gap-0.07811148909526727')

## Trainer
trainer = PGTrainer(trainer_cfg, agent, bmep)

## Training
trainer.train(train_data, test_data, logger, checkpoint_frequency=100, save_best=True, save_dir='experiments/checkpoints', run_id=cfg['run_id'])

if cfg['save_split']:
    torch.save(torch.tensor(train_idxs), os.path.join(trainer.save_path(), 'train_data_split.pt'))


