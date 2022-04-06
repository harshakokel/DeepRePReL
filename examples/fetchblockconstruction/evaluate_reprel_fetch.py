"""
This should results in an average return of ~3000 by the end of training.

Usually hits 3000 around epoch 80-100. Within a see, the performance will be
a bit noisy from one epoch to the next (occasionally dips dow to ~2000).

Note that one epoch = 5k steps, so 200 epochs = 1 million steps.
"""
import argparse
import gym
import os

from mpi4py import MPI

from examples.fetchblockconstruction.FetchBlocksPlanner import FetchBlocksPlanner
from rlkit.core.reprel_algorithm import RePReLAlgorithm
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.launchers.launcher_util import run_experiment
from rlkit.samplers.data_collector import RePReLGoalConditionedPathCollector
from rlkit.torch.optim.mpi_adam import MpiAdam
from rlkit.torch.relational.networks import *
from rlkit.torch.reprel.reprel_her import RePReLHERTrainer
from rlkit.torch.reprel.reprel_sac import RePReLSACTrainer
import torch
from rlkit.core import logger, eval_util


def experiment(variant):
    try:
        import fetch_block_construction
    except ImportError as e:
        print(e)

    eval_env = gym.make(variant['env'])
    data = torch.load(variant['model_file']) # , map_location='cpu' )

    observation_key = 'observation'
    desired_goal_key = 'desired_goal'
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")

    eval_planner = variant['planner'](eval_env,
                                      observation_key,
                                      desired_goal_key,
                                      achieved_goal_key,
                                      abstract_obs_type=variant['planner_obs_type'])

    dims = eval_planner.dims
    operators = eval_planner.get_operators()

    operator_qf1s = data['trainer/operator_qf1s']
    operator_qf2s = data['trainer/operator_qf2s']
    operator_target_qf1s = data['trainer/operator_target_qf1s']
    operator_target_qf2s = data['trainer/operator_target_qf2s']
    operator_policies = data['trainer/operator_policies']
    log_alpha = data['trainer/log_alpha']

    operator_eval_policy = operator_policies
    eval_path_collector = RePReLGoalConditionedPathCollector(
        eval_env,
        operator_eval_policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        get_action_kwargs=dict(mask=np.ones((1, 1))),
        agents_passed=True,
        render=True,
        planner=eval_planner,
        task_terminal_reward=1,
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

    parser.add_argument("--num-episodes",
                        type=int,
                        default=100,
                        help="Number of episodes")

    parser.add_argument("--steps-per-block",
                        type=int,
                        default=50,
                        help="Number of steps per block")

    parser.add_argument("--nearest-neighbour", action='store_true', default=False,
                        help="Use nearest neighbour")

    parser.add_argument("--batch-size",
                        type=int,
                        default=256,
                        help="Batch size")

    parser.add_argument("--transfer",
                        required=True,
                        help="Path to model for transfer")

    parser.add_argument('--allow-in-air', action='store_true', default=False,
                        help="Allow blocks to be in Air (i.e, stackonly=False)")

    args = parser.parse_args()

    filename = args.transfer

    variant = dict(
        algo_kwargs=dict(
            num_epochs=3000 * 10,
            max_path_length=args.steps_per_block * args.num_blocks,
            batch_size=args.batch_size,
            num_trains_per_train_loop=args.steps_per_block * args.num_blocks,
            num_expl_steps_per_train_loop=args.steps_per_block * args.num_blocks,  # Do one episode per block
            num_eval_steps_per_epoch=args.steps_per_block * args.num_blocks * args.num_episodes,
            # Do ten episodes per eval
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
        env=F"FetchBlockConstruction_{args.num_blocks}Blocks_IncrementalReward_DictstateObs_42Rendersize_{not args.allow_in_air}Stackonly_SingletowerCase-v1",
        # TODO: make sure FalseStackonly so it goes in the air
        save_video=False,
        save_video_period=args.steps_per_block,
        set_max_episode_steps=args.steps_per_block * args.num_blocks,
        her_kwargs=dict(
            exploration_masking=True
        ),
        model_file=args.transfer,
        planner=FetchBlocksPlanner,
    )
    exp_prefix = F"reprel_evaluate_stack{args.num_blocks}_{not args.allow_in_air}stackonly"
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
                   base_log_dir=f"data/reprel_evaluate/",
                   prepend_date_to_exp_prefix=True)
