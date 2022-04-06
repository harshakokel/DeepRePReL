"""
Run DQN on grid world.
"""

import gym
from torch import nn as nn
import torch
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy, EpsilonGreedyWithDecay
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.dqn.dqn import DQNTrainer
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import officeworld
import argparse
import os
import pickle
import torch

def experiment(variant):
    expl_env = gym.make(variant['env'])
    eval_env = gym.make(variant['env'])
    # data = pickle.load(open(variant['model_file'],"rb"))
    data = torch.load(variant['model_file']) #, map_location='cpu' )
    eval_policy = data['evaluation/policy']
    qf = data['trainer/qf']
    target_qf =data['trainer/target_qf']

    if variant['epsilon_decay']:
        exploration_strategy = EpsilonGreedyWithDecay(
            action_space=expl_env.action_space, num_epochs=variant['algorithm_kwargs']['num_epochs']
        )
    else:
        exploration_strategy = EpsilonGreedy(
            action_space=expl_env.action_space,
        )
    expl_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=exploration_strategy,
        policy=eval_policy,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
        epsilon_decay=variant['epsilon_decay']
    )
    trainer = DQNTrainer(
        qf=qf,
        target_qf=target_qf,
        **variant['trainer_kwargs']
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":




    parser = argparse.ArgumentParser()

    parser.add_argument("--transfer",
                        required=True,
                        help="Path to model for transfer")

    parser.add_argument("--env",
                        default="OfficeWorld-deliver-mail-v0",
                        help="Environment")

    parser.add_argument("--total-epochs",
                        type=int,
                        default=3000,
                        help="Total epochs for training")

    parser.add_argument("--buffer-size",
                        type=int,
                        default=1e6,
                        help="Max buffer size")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.0003,
                        help="Max buffer size")


    parser.add_argument("--max-episode-length",
                        type=int,
                        default=100,
                        help="Max Episode Path Length")


    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
                        help="Batch size")

    parser.add_argument('--decay-epsilon', action='store_true', default=False,
                        help="enable the epsilon decay strategy for exploration")

    args = parser.parse_args()

    # noinspection PyTypeChecker
    variant = dict(
        algorithm="DQN-transfer",
        model_file= args.transfer,
        version=f"DQN-transfer-{args.env}",
        env =args.env,
        epsilon_decay=args.decay_epsilon,
        replay_buffer_size=int(args.buffer_size),
        algorithm_kwargs=dict(
            num_epochs=args.total_epochs,
            num_eval_steps_per_epoch=10*args.max_episode_length,
            num_trains_per_train_loop=10*args.max_episode_length,
            num_expl_steps_per_train_loop=10*args.max_episode_length,
            min_num_steps_before_training=10*args.max_episode_length,
            max_path_length=args.max_episode_length,
            batch_size=args.batch_size,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            learning_rate=args.learning_rate,
        ),
    )
    exp_id = os.getpid()
    setup_logger(variant['version'], variant=variant, snapshot_mode="gap_and_last",snapshot_gap=20,exp_id=exp_id)
    if torch.cuda.is_available():
        ptu.set_gpu_mode('gpu')  # optionally set the GPU (default=False)
    experiment(variant)
