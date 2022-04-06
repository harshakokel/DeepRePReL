"""
Run DQN on grid world.
"""

import gym
from torch import nn as nn

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy, EpsilonGreedyWithDecay
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.reprel.reprel_dqn import RePReLDQNTrainer as TRLDQNTrainer
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer, SimpleReplayBufferDiscreteAction
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector.reprel_path_collector import RePReLRollout
from rlkit.samplers.data_collector.reprel_path_collector import RePReLPathCollector as TRLPathCollector
from rlkit.core.reprel_algorithm import RePReLAlgorithm as TRLAlgorithm
import officeworld
import argparse
from examples.office.office_trl_planner import *
import os, torch


def experiment(variant):
    expl_env = gym.make(variant['env'])
    eval_env = gym.make(variant['env'])
    data = torch.load(variant['model_file']) #, map_location='cpu' )
    eval_policy = data['evaluation/policy']
    operator_qfs = data['trainer/operator_qfs']
    operator_target_qfs =data['trainer/operator_target_qfs']
    expl_planner = variant['planner'](expl_env)
    eval_planner = variant['planner'](expl_env)
    operators = expl_planner.get_operators()
    dims = expl_planner.dims
    replay_buffers = {}
    for operator in operators:
        obs_dim, action_dim = dims[operator]
        replay_buffers[operator] = SimpleReplayBufferDiscreteAction(
            max_replay_buffer_size=variant['replay_buffer_size'],
            observation_dim=obs_dim,
            action_dim=action_dim,
            env_info_sizes={})
    if variant['epsilon_decay']:
        exploration_strategy = EpsilonGreedyWithDecay(
            action_space=expl_env.action_space, num_epochs=variant['algorithm_kwargs']['num_epochs']
        )
    else:
        exploration_strategy = EpsilonGreedy(
            action_space=expl_env.action_space,
        )
    eval_path_collector = TRLPathCollector(
        eval_env,
        operator_qfs,
        policy=ArgmaxDiscretePolicy,
        rollout_fn=RePReLRollout,
        planner=eval_planner,
        task_terminal_reward=0
    )
    expl_path_collector = TRLPathCollector(expl_env,
                                           operator_qfs,
                                           strategy=exploration_strategy,
                                           policy=ArgmaxDiscretePolicy,
                                           planner=expl_planner,
                                           rollout_fn=RePReLRollout,
                                           epsilon_decay=variant['epsilon_decay'],
                                           task_terminal_reward=30)
    trainer = TRLDQNTrainer(
        operator_qfs,
        operator_target_qfs,
        **variant['trainer_kwargs']
    )

    algorithm = TRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffers=replay_buffers,
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
        algorithm="TRL-transfer",
        model_file= args.transfer,
        version=f"TRL-transfer-{args.env}",
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
        planner = OfficePlanner
    )
    exp_id = os.getpid()
    setup_logger(variant['version'], variant=variant, snapshot_mode="gap_and_last",snapshot_gap=20,exp_id=exp_id)
    if torch.cuda.is_available():
        ptu.set_gpu_mode('gpu')  # optionally set the GPU (default=False)
    experiment(variant)
