"""
Run DQN on grid world.
"""

import argparse
import os

import gym
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy, EpsilonGreedyWithDecay
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.dqn.dqn import DQNTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import taxi_domain


def experiment(variant):
    expl_env = gym.make(variant['env'])
    eval_env = gym.make(variant['env'])
    data = torch.load(variant['model_file'])  # , map_location='cpu' )
    eval_policy = data['evaluation/policy']
    qf = data['trainer/qf']
    qf._mask = eval_env.passenger_count
    target_qf = data['trainer/target_qf']
    target_qf._mask = eval_env.passenger_count
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.n

    if variant['epsilon_decay']:
        exploration_strategy = EpsilonGreedyWithDecay(
            action_space=expl_env.action_space, explore_ratio=variant['exploration_ratio'],
            num_epochs=variant['algorithm_kwargs']['num_epochs']
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
                        default="RelationalTaxiWorld-graph-task1-v1",
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
                        default=1000,
                        help="Max Episode Length")

    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
                        help="Batch size")

    parser.add_argument("--exploration-ratio",
                        type=float,
                        default=0.2,
                        help="Exploration Ratio")

    parser.add_argument('--decay-epsilon', action='store_true', default=False,
                        help="Use epsilon decay")

    args = parser.parse_args()

    # noinspection PyTypeChecker
    variant = dict(
        algorithm="GNN-DQN-transfer",
        model_file=args.transfer,
        version=f"GNN-DQN-transfer-{args.env}",
        env=args.env,
        replay_buffer_size=int(args.buffer_size),
        algorithm_kwargs=dict(
            num_epochs=args.total_epochs,
            num_eval_steps_per_epoch=10 * args.max_episode_length,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=10 * args.max_episode_length,
            min_num_steps_before_training=10 * args.max_episode_length,
            max_path_length=args.max_episode_length,
            batch_size=args.batch_size,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            learning_rate=args.learning_rate,
        ),
        epsilon_decay=args.decay_epsilon,
        exploration_ratio=args.exploration_ratio,
    )
    exp_id = os.getpid()
    setup_logger(variant['version'], variant=variant, snapshot_mode="gap_and_last", snapshot_gap=20, exp_id=exp_id)
    if torch.cuda.is_available():
        ptu.set_gpu_mode('gpu')  # optionally set the GPU (default=False)
    experiment(variant)
