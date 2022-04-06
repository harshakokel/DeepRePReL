"""
This should results in an average return of ~3000 by the end of training.

Usually hits 3000 around epoch 80-100. Within a see, the performance will be
a bit noisy from one epoch to the next (occasionally dips dow to ~2000).

Note that one epoch = 5k steps, so 200 epochs = 1 million steps.
"""

import argparse
import gym
import os, time
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.samplers.data_collector import MdpPathCollector, GoalConditionedPathCollector
from rlkit.torch.her.her import HERTrainer

from rlkit.torch.data_management.normalizer import CompositeNormalizer
from rlkit.torch.optim.mpi_adam import MpiAdam
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.relational.networks import *
import torch.nn.functional as F
from rlkit.torch.relational.modules import *
from torch.nn import Parameter
from rlkit.launchers.config import get_infra_settings
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


def experiment(variant):
    try:
        import fetch_block_construction
    except ImportError as e:
        print(e)

    env = gym.make(variant['env'])
    eval_env = gym.make(variant['env'])
    env.unwrapped.render_image_obs = False
    if variant['set_max_episode_steps']:
        env.env._max_episode_steps = variant['set_max_episode_steps']

    action_dim = env.action_space.low.size

    value_graphprop_kwargs = dict(
        graph_module_kwargs=dict(
            # num_heads=num_query_heads,
            # embedding_dim=embedding_dim,
            embedding_dim=variant['embedding_dim'],
            num_heads=1,
        ),
        layer_norm=variant['layer_norm'],
        num_query_heads=variant['num_query_heads'],
        num_relational_blocks=variant['num_relational_blocks'],
        activation_fnx=F.leaky_relu,
        recurrent_graph=variant['recurrent_graph']
    )

    qvalue_graphprop_kwargs = dict(
        graph_module_kwargs=dict(
            num_heads=variant['num_query_heads'],
            embedding_dim=variant['embedding_dim'],
        ),
        layer_norm=variant['layer_norm'],
        num_query_heads=variant['num_query_heads'],
        num_relational_blocks=variant['num_relational_blocks'],
        activation_fnx=F.leaky_relu,
        recurrent_graph=variant['recurrent_graph']
    )

    q1_gp = GraphPropagation(**qvalue_graphprop_kwargs)

    q2_gp = GraphPropagation(**qvalue_graphprop_kwargs)

    policy_gp = GraphPropagation(**value_graphprop_kwargs)

    policy_readout = AttentiveGraphPooling(mlp_kwargs=None)

    qf1_readout = AttentiveGraphPooling(mlp_kwargs=dict(
        hidden_sizes=variant['mlp_hidden_sizes'],
        output_size=1,
        input_size=variant['pooling_heads'] * variant['embedding_dim'],
        layer_norm=variant['layer_norm'],
    ), )
    qf2_readout = AttentiveGraphPooling(mlp_kwargs=dict(
        hidden_sizes=variant['mlp_hidden_sizes'],
        output_size=1,
        input_size=variant['pooling_heads'] * variant['embedding_dim'],
        layer_norm=variant['layer_norm'],
    ), )

    shared_normalizer = CompositeNormalizer(object_dim + shared_dim + goal_dim,
                                            action_dim,
                                            default_clip_range=5,
                                            reshape_blocks=True,
                                            fetch_kwargs=dict(
                                                lop_state_dim=3,
                                                object_dim=object_dim,
                                                goal_dim=goal_dim
                                            ))

    qf1 = QValueReNN(
        graph_propagation=q1_gp,
        readout=qf1_readout,
        mask=env.unwrapped.num_blocks,
        input_module_kwargs=dict(
            normalizer=shared_normalizer,
            object_total_dim=object_dim + shared_dim + goal_dim + action_dim,
            embedding_dim=variant['embedding_dim'],
            layer_norm=variant['layer_norm']
        ),
        composite_normalizer=shared_normalizer,
    )

    qf2 = QValueReNN(
        graph_propagation=q2_gp,
        readout=qf2_readout,
        mask=env.unwrapped.num_blocks,
        input_module_kwargs=dict(
            normalizer=shared_normalizer,
            object_total_dim=object_dim + shared_dim + goal_dim + action_dim,
            embedding_dim=variant['embedding_dim'],
            layer_norm=variant['layer_norm']
        ),
        composite_normalizer=shared_normalizer,
    )
    target_qf1 = QValueReNN(
        graph_propagation=q1_gp,
        readout=qf1_readout,
        mask=env.unwrapped.num_blocks,
        input_module_kwargs=dict(
            normalizer=shared_normalizer,
            object_total_dim=object_dim + shared_dim + goal_dim + action_dim,
            embedding_dim=variant['embedding_dim'],
            layer_norm=variant['layer_norm']
        ),
        composite_normalizer=shared_normalizer,
    )

    target_qf2 = QValueReNN(
        graph_propagation=q2_gp,
        readout=qf2_readout,
        mask=env.unwrapped.num_blocks,
        input_module_kwargs=dict(
            normalizer=shared_normalizer,
            object_total_dim=object_dim + shared_dim + goal_dim + action_dim,
            embedding_dim=variant['embedding_dim'],
            layer_norm=variant['layer_norm']
        ),
        composite_normalizer=shared_normalizer,
    )

    observation_key = 'observation'
    desired_goal_key = 'desired_goal'
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")

    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    policy = PolicyReNN(
        graph_propagation=policy_gp,
        readout=policy_readout,
        out_size=action_dim,
        mask=env.unwrapped.num_blocks,
        input_module_kwargs=dict(
            normalizer=shared_normalizer,
            object_total_dim=object_dim + shared_dim + goal_dim,
            embedding_dim=variant['embedding_dim'],
            layer_norm=variant['layer_norm']
        ),
        num_relational_blocks=variant['num_relational_blocks'],
        num_query_heads=variant['num_query_heads'],
        mlp_class=FlattenTanhGaussianPolicy,  # KEEP IN MIND
        mlp_kwargs=dict(
            hidden_sizes=variant['mlp_hidden_sizes'],
            obs_dim=variant['pooling_heads'] * variant['embedding_dim'],
            action_dim=action_dim,
            output_activation=torch.tanh,
            layer_norm=variant['layer_norm'],
            # init_w=3e-4,
        ),
        composite_normalizer=shared_normalizer
    )
    trainer = SACTrainer(
        env=env,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        policy=policy,
        optimizer_class=MpiAdam,
        **variant['sac_trainer_kwargs']
    )
    eval_policy = policy
    trainer = HERTrainer(trainer)
    eval_path_collector = GoalConditionedPathCollector(
        eval_env,
        eval_policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        get_action_kwargs=dict(mask=np.ones((1, eval_env.unwrapped.num_blocks)))  # Num_blocks is the MAX num_blocks
    )
    expl_path_collector = GoalConditionedPathCollector(
        env,
        policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        get_action_kwargs=dict(mask=np.ones((1, eval_env.unwrapped.num_blocks)))  # Num_blocks is the MAX num_blocks
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--num-hidden-layers",
                        type=int,
                        default=3,
                        help="Number of hidden layers in the last MLP layer")

    parser.add_argument("--num-hidden-units",
                        type=int,
                        default=64,
                        help="Number of hidden units in the last MLP layer")

    parser.add_argument("--num-blocks",
                        type=int,
                        default=1,
                        help="Number of Blocks")

    parser.add_argument("--num-relational-blocks",
                        type=int,
                        default=3,
                        help="Number of relational blocks")

    parser.add_argument("--embedding-dim",
                        type=int,
                        default=64,
                        help="Dimension of the Embedding in the Graph NN")

    parser.add_argument("--num-query-heads",
                        type=int,
                        default=1,
                        help="Number of query heads")


    parser.add_argument('--stack-only', action='store_true', default=False,
                        help="Stack all the blocks (no block in air)")

    parser.add_argument('--recurrent-graph', action='store_true', default=False,
                        help="Use Recurrent Graph")

    parser.add_argument('--no-layer-norm', action='store_false', default=True,
                        help="Do not use layer norm")

    parser.add_argument("--batch-size",
                        type=int,
                        default=256,
                        help="Batch size")

    parser.add_argument("--bounding-box", action='store_true', default=False,
                        help="Use bounding box")

    args = parser.parse_args()

    action_dim = 4
    object_dim = 15
    goal_dim = 3

    shared_dim = 10

    variant = dict(
        algorithm = "RENN",
        algo_kwargs=dict(
            num_epochs=3000 * 10,
            max_path_length=50 * args.num_blocks,
            batch_size=args.batch_size,
            num_trains_per_train_loop=50 * args.num_blocks,
            num_expl_steps_per_train_loop=50 * args.num_blocks,  # Do one episode per block
            num_eval_steps_per_epoch=50 * args.num_blocks * 10,  # Do ten episodes per eval
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
        layer_norm=args.no_layer_norm,
        render=False,
        env=F"FetchBlockConstruction_{args.num_blocks}Blocks_IncrementalReward_DictstateObs_42Rendersize_{args.stack_only}Stackonly_SingletowerCase-v1",
        # TODO: make sure FalseStackonly so it goes in the air
        save_video=False,
        save_video_period=50,
        num_relational_blocks=args.num_relational_blocks,
        set_max_episode_steps=50 * args.num_blocks,
        mlp_hidden_sizes=[args.num_hidden_units for _ in range(args.num_hidden_layers)],
        num_query_heads=args.num_query_heads,
        action_dim=action_dim,
        goal_dim=goal_dim,
        embedding_dim=args.embedding_dim,
        pooling_heads=1,
        her_kwargs=dict(
            exploration_masking=True
        ),
        recurrent_graph=args.recurrent_graph
    )
    exp_prefix = F"renn_task1_stack{args.num_blocks}_numrelblocks{args.num_relational_blocks}_nqh{args.num_query_heads}_{args.stack_only}stackonly_recurrent{args.recurrent_graph}"
    gpu_mode=False
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
                   prepend_date_to_exp_prefix=True
                   )
