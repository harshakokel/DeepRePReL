"""
This should results in an average return of ~3000 by the end of training.

Usually hits 3000 around epoch 80-100. Within a see, the performance will be
a bit noisy from one epoch to the next (occasionally dips dow to ~2000).

Note that one epoch = 5k steps, so 200 epochs = 1 million steps.
"""

import gym
import os

from rlkit.core import logger, eval_util
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.samplers.data_collector import GoalConditionedPathCollector
from rlkit.torch.her.her import HERTrainer
from mpi4py import MPI
from rlkit.torch.optim.mpi_adam import MpiAdam
from rlkit.launchers.launcher_util import run_experiment
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.torch.relational.modules import *
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
import argparse


def experiment(variant):
    try:
        import fetch_block_construction
    except ImportError as e:
        print(e)

    env = gym.make(variant['env'])
    eval_env = gym.make(variant['env'])
    data = torch.load(variant['model_file'])  #, map_location='cpu')

    policy = data['trainer/policy']
    policy._mask = env.unwrapped.num_blocks
    qf1 = data['trainer/qf1']
    qf1._mask = env.unwrapped.num_blocks
    qf2 = data['trainer/qf2']
    qf2._mask = env.unwrapped.num_blocks
    target_qf1 = data['trainer/target_qf1']
    target_qf1._mask = env.unwrapped.num_blocks
    target_qf2 = data['trainer/target_qf2']
    target_qf2._mask = env.unwrapped.num_blocks

    log_alpha = None
    # log_alpha = data['trainer/log_alpha']
    # q1_optimizer = data['trainer/optimizers/q1_optimizer']
    # q2_optimizer = data['trainer/optimizers/q2_optimizer']
    # policy_optimizer = data['trainer/optimizers/policy_optimizer']
    # alpha_optimizer = data['trainer/optimizers/alpha_optimizer']

    observation_key = 'observation'
    desired_goal_key = 'desired_goal'
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")

    eval_path_collector = GoalConditionedPathCollector(
        eval_env,
        policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        render=True,
        get_action_kwargs=dict(mask=np.ones((1, eval_env.unwrapped.num_blocks)))  # Num_blocks is the MAX num_blocks
    )
    _ = eval_path_collector.collect_new_paths(
        variant['algo_kwargs']['max_path_length'],
        variant['algo_kwargs']['num_eval_steps_per_epoch'],
        discard_incomplete_paths=False,
    )
    logger.record_dict(
        eval_path_collector.get_diagnostics(),
        prefix='eval/'
    )
    paths_all = eval_path_collector.get_epoch_paths()
    logger.record_dict(
        eval_util.get_generic_path_information(paths_all),
        prefix="eval/",
    )
    logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-blocks",
                        type=int,
                        required=True,
                        help="Number of Blocks")

    parser.add_argument("--transfer",
                        required=True,
                        help="Path to model for transfer")

    parser.add_argument("--shape",
                        default="Singletower",
                        help="Case for the environment")

    parser.add_argument("--batch-size",
                        type=int,
                        default=256,
                        help="Batch size")

    parser.add_argument("--num-episodes",
                        type=int,
                        default=100,
                        help="Number of episodes")

    parser.add_argument('--allow-in-air', action='store_true', default=False,
                        help="Allow blocks to be in Air (i.e, stackonly=False)")

    args = parser.parse_args()
    action_dim = 4
    object_dim = 15
    goal_dim = 3

    shared_dim = 10
    filename = args.transfer

    variant = dict(
        algo_kwargs=dict(
            num_epochs=3000 * 10,
            max_path_length=50 * args.num_blocks,
            batch_size=args.batch_size,
            num_trains_per_train_loop=50 * args.num_blocks,
            num_expl_steps_per_train_loop=50 * args.num_blocks,  # Do one episode per block
            num_eval_steps_per_epoch=50 * args.num_blocks * args.num_episodes,  # Do ten episodes per eval
            num_train_loops_per_epoch=10,
        ),
        sac_trainer_kwargs=dict(
            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            discount=0.98,
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1e5),
            fraction_goals_rollout_goals=0.2,  # equal to k = 4 in HER paper
            fraction_goals_env_goals=0.0,
        ),
        render=False,
        env=F"FetchBlockConstruction_{args.num_blocks}Blocks_IncrementalReward_DictstateObs_42Rendersize_{not args.allow_in_air}Stackonly_{args.shape}Case-v1",
        # TODO: make sure FalseStackonly so it goes in the air
        save_video=False,
        save_video_period=50,
        action_dim=action_dim,
        goal_dim=goal_dim,
        her_kwargs=dict(
            exploration_masking=True
        ),
        model_file=args.transfer,
    )
    exp_prefix = F"renn_evaluate_stack{args.num_blocks}_{not args.allow_in_air}stackonly"
    gpu_mode = False
    if torch.cuda.is_available():
        ptu.set_gpu_mode('gpu')  # optionally set the GPU (default=False)
        gpu_mode = 'gpu'
    run_experiment(experiment,
                   exp_prefix=exp_prefix,
                   variant=variant,
                   use_gpu=gpu_mode,
                   snapshot_mode='gap_and_last',
                   snapshot_gap=100,
                   exp_id=os.getpid(),
                   base_log_dir=f"data/renn_evaluate/",
                   prepend_date_to_exp_prefix=True
                   )
