"""
Run DQN on office world.
"""

import gym
from torch import nn as nn

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
from gym.spaces.discrete import Discrete
from gym.spaces import MultiDiscrete
import officeworld
import argparse
import os
import torch


def experiment(variant):
    expl_env = gym.make(variant['env'])
    eval_env = gym.make(variant['env'])
    obs_dim = expl_env.observation_space.shape[0]
    action_dim = eval_env.action_space.n

    qf = Mlp(
        hidden_sizes=variant['net_arch'],
        input_size=obs_dim,
        output_size=action_dim,
    )
    target_qf = Mlp(
        hidden_sizes=variant['net_arch'],
        input_size=obs_dim,
        output_size=action_dim,
    )
    eval_policy = ArgmaxDiscretePolicy(qf)
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

    parser.add_argument("--env",
                        default="OfficeWorld-deliver-mail-v0",
                        help="Environment")

    parser.add_argument("--total-epochs",
                        type=int,
                        default=3000,
                        help="Total epochs for training")

    parser.add_argument('--decay-epsilon', action='store_true', default=False,
                        help="enable the epsilon decay strategy for exploration")

    parser.add_argument("--num-hidden-layers",
                        type=int,
                        default=2,
                        help="Number of hidden layers")

    parser.add_argument("--num-hidden-units",
                        type=int,
                        default=256,
                        help="Number of hidden units")

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
                        default=1000,
                        help="Max Episode Length")

    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
                        help="Batch size")

    args = parser.parse_args()

    # noinspection PyTypeChecker
    variant = dict(
        algorithm="DQN",
        version=f"DQN-{args.env}",
        env=args.env,
        net_arch=[args.num_hidden_units for _ in range(args.num_hidden_layers)],
        replay_buffer_size=int(args.buffer_size),
        epsilon_decay=args.decay_epsilon,
        algorithm_kwargs=dict(
            num_epochs=args.total_epochs,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=args.max_episode_length,
            batch_size=args.batch_size,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            learning_rate=args.learning_rate,
        ),
    )
    exp_id = os.getpid()
    setup_logger(variant['version'],
                 variant=variant,
                 snapshot_mode="gap_and_last",
                 snapshot_gap=20,
                 exp_id=exp_id)
    if torch.cuda.is_available():
        ptu.set_gpu_mode("gpu")  # optionally set the GPU (default=False)
    experiment(variant)
