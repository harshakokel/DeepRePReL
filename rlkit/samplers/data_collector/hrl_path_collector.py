from collections import deque, OrderedDict
from functools import partial

import numpy as np
import copy
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.data_collector.base import PathCollector
from rlkit.samplers.rollout_functions import rollout
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy, EpsilonGreedyWithDecay
from rlkit.policies.argmax import ArgmaxDiscretePolicy

METACONTROLLER = 'metacontroller'


def HRLRollout(
        env,
        agents,
        intrinsic_critic,
        is_terminal,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        preprocess_obs_for_policy_fn=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
        reset_callback=None,
        metacontroller_agent=None,
        operators_list=None
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x
    raw_obs = []
    raw_next_obs = []
    keys = operators_list+[METACONTROLLER,'all']
    observations = {operator: [] for operator in keys}
    actions = {operator: [] for operator in keys}
    rewards = {operator: [] for operator in keys}
    terminals = {operator: [] for operator in keys}
    agent_infos = {operator: [] for operator in keys}
    env_infos = {operator: [] for operator in keys}
    next_observations = {operator: [] for operator in keys}
    path_length = 0
    # TODO: Implement the roll out
    o = env.reset()
    # if reset_callback:
    #     reset_callback(env, agent, o)
    agent = {}
    s_for_meta = preprocess_obs_for_policy_fn(o)
    current_option_id, metacontroller_agent_info  = metacontroller_agent.get_action(s_for_meta, **get_action_kwargs)
    current_option = operators_list[current_option_id]
    current_agent = agents[current_option]

    if render:
        env.render(**render_kwargs)
    episode_done = False
    F = 0
    while path_length < max_path_length:
        raw_obs.append(o)
        o_for_agent = preprocess_obs_for_policy_fn(o)

        a, agent_info = current_agent.get_action(o_for_agent, **get_action_kwargs)

        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        next_o, f, episode_done, env_info = env.step(copy.deepcopy(a))
        F += f
        r = intrinsic_critic(o, a, next_o, current_option, f)
        task_done = is_terminal(o, a, next_o, current_option)
        if render:
            env.render(**render_kwargs)
        observations[current_option].append(o_for_agent)
        observations['all'].append(o)
        rewards[current_option].append(r)
        rewards['all'].append(f)
        terminals[current_option].append(task_done)
        terminals['all'].append(episode_done)
        actions[current_option].append(a)
        actions['all'].append(a)
        next_observations[current_option].append(next_o)
        next_observations['all'].append(next_o)
        raw_next_obs.append(next_o)
        agent_infos[current_option].append(agent_info)
        agent_infos['all'].append({'task_done': task_done})
        env_infos[current_option].append(env_info)
        env_infos['all'].append(env_info)
        path_length += 1
        if episode_done or task_done:
            next_s_for_meta = preprocess_obs_for_policy_fn(next_o)
            observations[current_option].append(o_for_agent)
            observations[METACONTROLLER].append(s_for_meta)
            rewards[METACONTROLLER].append(F)
            terminals[METACONTROLLER].append(episode_done)
            actions[METACONTROLLER].append(current_option_id)
            next_observations[METACONTROLLER].append(next_s_for_meta)
            agent_infos[METACONTROLLER].append(metacontroller_agent_info)
            env_infos[METACONTROLLER].append(env_info)
            F = 0
        if episode_done:
            break
        if task_done:
            current_option_id, metacontroller_agent_info = metacontroller_agent.get_action(next_s_for_meta, **get_action_kwargs)
            current_option = operators_list[current_option_id]
            current_agent = agents[current_option]

        o = next_o

    # actions = np.array(actions)
    # if len(actions.shape) == 1:
    #     actions = np.expand_dims(actions, 1)
    # observations = np.array(observations)
    # next_observations = np.array(next_observations)
    # if return_dict_obs:
    #     observations = raw_obs
    #     next_observations = raw_next_obs
    # rewards = np.array(rewards)
    # if len(rewards.shape) == 1:
    #     rewards = rewards.reshape(-1, 1)
    return {operator: dict(
        observations=np.array(observations[operator]),
        actions=np.array(actions[operator]).reshape(-1, 1),
        rewards=np.array(rewards[operator]).reshape(-1, 1),
        next_observations=np.array(next_observations[operator]),
        terminals=np.array(terminals[operator]).reshape(-1, 1),
        agent_infos=agent_infos[operator],
        env_infos=env_infos[operator],
    ) for operator in observations.keys()}


class HRLPathCollector(PathCollector):
    def __init__(
            self,
            env,
            operator_qfs,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            rollout_fn=HRLRollout,
            save_env_in_snapshot=True,
            epsilon_decay=False,
            metacontroller_epsilon_decay=False,
            strategy=None,
            metacontroller_strategy=None,
            policy=None,
            intrinsic_critic=None,
            is_terminal=None,
            operators_list=None
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._operator_qfs = operator_qfs
        self._agent_epsilon_decay = epsilon_decay
        self._metacontroller_epsilon_decay = metacontroller_epsilon_decay
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths =  deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._rollout_fn = rollout_fn
        self._policy = policy
        self._strategy = strategy
        self._metacontroller_strategy = metacontroller_strategy
        self._intrinsic_critic = intrinsic_critic
        self._is_terminal = is_terminal
        self._num_steps_total = 0
        self._num_paths_total = 0
        self._save_env_in_snapshot = save_env_in_snapshot
        self._operators_list = operators_list
        self._agents = {}
        for operator in operators_list:
            current_agent = policy(operator_qfs[operator])
            if strategy is not None:
                current_agent = PolicyWrappedWithExplorationStrategy(
                    exploration_strategy=strategy,
                    policy=current_agent,
                )
            self._agents[operator] = current_agent
        metacontroller_agent = policy(operator_qfs[METACONTROLLER])
        if metacontroller_strategy is not None:
            metacontroller_agent = PolicyWrappedWithExplorationStrategy(
                exploration_strategy=metacontroller_strategy,
                policy=metacontroller_agent,
            )
        self._metacontroller_agent = metacontroller_agent
        if self._agent_epsilon_decay:
            self._agent_epsilon =  self._strategy.epsilon
        if self._metacontroller_epsilon_decay:
            self._metacontroller_epsilon = self._metacontroller_strategy.epsilon

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = {operator: [] for operator in self._operator_qfs.keys()}
        paths['all']=[]
        num_steps_collected = 0
        num_paths_total = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            operator_path = self._rollout_fn(
                self._env,
                self._agents,
                self._intrinsic_critic,
                self._is_terminal,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
                operators_list=self._operators_list,
                metacontroller_agent=self._metacontroller_agent
            )
            path_len = len(operator_path['all']['actions'])
            if (discard_incomplete_paths
                and path_len != max_path_length
                and not operator_path['all']['terminals'][-1]
                and not operator_path['all']['agent_infos'][-1]['task_done']
            ):
                break
            num_steps_collected += path_len
            for key, path in operator_path.items(): paths[key].append(path)
            num_paths_total += 1
        self._num_paths_total += num_paths_total
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths['all'])
        if self._agent_epsilon_decay:
            self._agent_epsilon =  self._strategy.epsilon
        if self._metacontroller_epsilon_decay:
            self._metacontroller_epsilon = self._metacontroller_strategy.epsilon
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        if self._agent_epsilon_decay:
            self._strategy.decay()
        if self._metacontroller_epsilon_decay:
            self._metacontroller_strategy.decay()
        self._epoch_paths =  deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        if self._agent_epsilon_decay:
            stats.update(create_stats_ordered_dict(
                "Epsilon value",
                self._agent_epsilon,
                always_show_all_stats=True,
            ))
        if self._metacontroller_epsilon_decay:
            stats.update(create_stats_ordered_dict(
                "MetaController Epsilon value",
                self._metacontroller_epsilon,
                always_show_all_stats=True,
            ))
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        snapshot_dict = dict(
            policy=self._policy,
        )
        if self._save_env_in_snapshot:
            snapshot_dict['env'] = self._env
        return snapshot_dict
