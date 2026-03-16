import os
import random
import uuid
import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from environments.batch.batch_env import BatchedBMEPEnvironment
from utils._compute import sa_feat_norm, zero_bspr_len
from agents.agent import SPRPolicy
from networks.attention_net import AttentionNet
from networks.ff_net import FFNet
import sys
from collections import defaultdict
from utils._dataset import dataset_partition
from pg_trainer import PGTrainer

# random starttree - reinforce - rf distance

if __name__ == "main":
    cfg = defaultdict(list)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg['env_cfg'] = {
      'tree_init_method': 'random',
      'normalize_reward': True,
      'normalize_feats': True,
      'feat_transform': None,
      'fixed_start': False,
      'rew_norm_scale': 10
    }

    cfg['network_cfg'] = {
       'network_cls': 'FFNet',
       'network_args': {
          'in_features': 20,
          'out_features': 1,
          'layers': [128,128,128,128]
       }
    }

    cfg['train_cfg'] = {
        'dataset_folders': [
          ''
        ],
        'n_steps': 30,
        'n_epochs': 200,
        'problem_batch_size': 100,
        'training_batch_size': 512,
        'eps_in_memory': 2, # buffer size
        'train_epochs': 10,
        'gamma': 0.9,
    }

    cfg['policy_cfg'] = {
      'train_selection_method': 'combined-sample',
      'test_selection_method': 'sample',
      'alpha': 1.0,
      'alpha_decay': 0.9,
      'temp': 0.00001
    }

    cfg['optimizer_cfg'] = {
      'optimizer_cls': 'Adam',
      'optimizer_args': {
        'lr': 1e-3,
        'weight_decay': 1e-4
      },
      'lr_scheduler': 'StepLR',
      'lr_scheduler_args': {
        'step_size': 50,
        'gamma': 0.32
      }
    }

    cfg['trainer_cfg'] = {
       'resample_freq': 5,
       'train_freq': 1
    }

    policy_cfg = cfg['policy_cfg']
    network_cfg = cfg['network_cfg']
    train_cfg = cfg['train_cfg']
    optimizer_cfg = cfg['optimizer_cfg']

    trainer_cfg = cfg['trainer_cfg']
    trainer_cfg['device'] = device
    trainer_cfg['train_cfg'] = train_cfg
    trainer_cfg['optimizer_cfg'] = optimizer_cfg

    bmep = BatchedBMEPEnvironment(**cfg['env_cfg'])
    agent = SPRPolicy(policy_cfg, network_cfg)

    datas = []
    for file in trainer_cfg['dataset_folders']:
      datas.append(bmep.read_dataset(file))
    train_data, test_data, train_idxs = dataset_partition(datas, train_cfg['dataset_partition_prop'])

    logger = None
    trainer = PGTrainer(trainer_cfg, agent, bmep)
    trainer.train(train_data, test_data, logger, checkpoint_frequency=100, save_best=True, save_dir='experiments/checkpoints')