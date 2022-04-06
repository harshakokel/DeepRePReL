from collections import deque, OrderedDict
from functools import partial

import numpy as np
import copy
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.data_collector.base import PathCollector
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy


def RePReLRollout(
        env,
        agents,
        planner,
        max_path_length=np.inf,
        render=False,
        render_kwargs={},
        preprocess_obs_for_policy_fn=lambda x: x,
        get_action_kwargs={},
        return_dict_obs=False,
        task_terminal_reward=100,
        full_o_postprocess_func=None,
        reset_callback=None
):
    raw_obs = []
    raw_next_obs = []
    keys = list(agents.keys()) + ['all']
    observations = {operator: [] for operator in keys}
    actions = {operator: [] for operator in keys}
    rewards = {operator: [] for operator in keys}
    terminals = {operator: [] for operator in keys}
    agent_infos = {operator: [] for operator in keys}
    env_infos = {operator: [] for operator in keys}
    next_observations = {operator: [] for operator in keys}
    path_length = 0
    # TODO: Implement the roll out
    planner.reset()
    o = env.reset()
    # if reset_callback:
    #     reset_callback(env, agent, o)
    agent = {}
    current_operator, current_task = planner.get_next_operator(o)
    while current_operator is None:  # Make sure the goal is not already achieved
        env.reset()
        current_operator, current_task = planner.get_next_operator(o)
    current_agent = agents[current_operator]

    if render:
        env.render(**render_kwargs)
    episode_done = False
    while path_length < max_path_length:
        raw_obs.append(o)
        o_for_agent = preprocess_obs_for_policy_fn(o)

        # Get abstract state from planner
        o_for_agent = planner.get_abstract_state(current_operator, current_task, o_for_agent)
        a, agent_info = current_agent.get_action(o_for_agent, **get_action_kwargs)

        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        next_o, r, episode_done, env_info = env.step(copy.deepcopy(a))
        task_done = planner.is_terminal(current_operator, current_task, next_o)
        next_o_for_agent = planner.get_abstract_state(current_operator, current_task, next_o)
        if render:
            env.render(**render_kwargs)
        observations[current_operator].append(o_for_agent)
        observations['all'].append(o)
        rewards[current_operator].append(task_terminal_reward + r if task_done else r)
        rewards['all'].append(r)
        terminals[current_operator].append(task_done)
        terminals['all'].append(episode_done)
        actions[current_operator].append(a)
        actions['all'].append(a)
        next_observations[current_operator].append(next_o_for_agent)
        next_observations['all'].append(next_o)
        raw_next_obs.append(next_o)
        agent_info.update({'task_done': task_done})
        agent_infos[current_operator].append(agent_info)
        agent_infos['all'].append(agent_info)
        env_infos[current_operator].append(env_info)
        env_infos['all'].append(env_info)
        path_length += 1
        if episode_done:
            break
        if task_done:
            current_operator, current_task = planner.get_next_operator(o)
            if current_operator is None:
                break
            current_agent = agents[current_operator]
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


def RePReLGoalConditionedRollout(
        env,
        agents,
        planner,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        preprocess_obs_for_policy_fn=None,
        get_action_kwargs={},
        return_dict_obs=False,
        task_terminal_reward=100,
        full_o_postprocess_func=None,
        reset_callback=None,
        observation_dict=False
):
    if render_kwargs is None:
        render_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x
    raw_obs = []
    raw_next_obs = []
    keys = list(agents.keys()) + ['all']
    episode_observations = []
    task_paths = {operator: [] for operator in keys}

    episode_actions = []
    episode_rewards = []
    episode_terminals = []
    episode_agent_infos = []
    episode_env_infos = []
    episode_next_observations = []
    task_actions = []
    task_observations = []
    task_rewards = []
    task_terminals = []
    task_agent_infos = []
    task_env_infos = []
    task_next_observations = []
    path_length = 0
    # TODO: Implement the roll out
    planner.reset()
    o = env.reset()
    init_o = o.copy()
    if render:
        env.render(**render_kwargs)
    # if reset_callback:
    #     reset_callback(env, agent, o)
    agent = {}
    current_operator, current_task = planner.get_next_operator(o)
    while current_operator is None:  # Make sure the goal is not already achieved
        env.reset()
        current_operator, current_task = planner.get_next_operator(o)
    current_agent = agents[current_operator]

    episode_done = False
    while path_length < max_path_length:
        raw_obs.append(o)
        o_for_agent = preprocess_obs_for_policy_fn(o)

        # Get abstract state from planner
        o_for_agent = planner.get_abstract_state(current_operator, current_task, o_for_agent)
        a, agent_info = current_agent.get_action(o_for_agent, **get_action_kwargs)

        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        next_o, r, episode_done, env_info = env.step(copy.deepcopy(a))
        next_o_for_agent = preprocess_obs_for_policy_fn(next_o)
        task_done = planner.is_terminal(current_operator, current_task, next_o_for_agent)
        next_o_for_agent = planner.get_abstract_state(current_operator, current_task, next_o_for_agent)
        if render:
            env.render(**render_kwargs)
        # if observation_dict:
        #     o_record = copy.deepcopy(o)
        #     o_record['observation'] = o_for_agent
        #     observations[current_operator].append(o_for_agent)
        # else:
        task_observations.append(planner.get_abstract_dict(current_operator, current_task, o))
        task_rewards.append(task_terminal_reward + r if task_done else r)
        task_next_observations.append(planner.get_abstract_dict(current_operator, current_task, next_o))
        task_actions.append(a)
        task_agent_infos.append(agent_info)
        task_env_infos.append(env_info)

        episode_env_infos.append(env_info)
        episode_actions.append(a)
        episode_agent_infos.append({'task_done': task_done})
        episode_terminals.append(episode_done)
        episode_observations.append(o)
        episode_rewards.append(r)
        episode_next_observations.append(next_o)

        raw_next_obs.append(next_o)
        path_length += 1
        if episode_done:
            task_terminals.append(True)
            task_paths[current_operator].append(dict(
                observations=np.array(task_observations),
                actions=np.array(task_actions),
                rewards=np.array(task_rewards).reshape(-1, 1),
                next_observations=np.array(task_next_observations),
                terminals=np.array(task_terminals).reshape(-1, 1),
                agent_infos=task_agent_infos,
                env_infos=task_env_infos
            ))
            task_actions = []
            task_observations = []
            task_rewards = []
            task_terminals = []
            task_agent_infos = []
            task_env_infos = []
            task_next_observations = []
            break

        if task_done:
            next_operator, next_task = planner.get_next_operator(o)
            if next_operator is not None:
                task_terminals.append(True)
                task_paths[current_operator].append(dict(
                    observations=np.array(task_observations),
                    actions=np.array(task_actions),
                    rewards=np.array(task_rewards).reshape(-1, 1),
                    next_observations=np.array(task_next_observations),
                    terminals=np.array(task_terminals).reshape(-1, 1),
                    agent_infos=task_agent_infos,
                    env_infos=task_env_infos
                ))
                current_operator, current_task = next_operator, next_task
                current_agent = agents[current_operator]
                task_actions = []
                task_observations = []
                task_rewards = []
                task_terminals = []
                task_agent_infos = []
                task_env_infos = []
                task_next_observations = []
            else:
                task_terminals.append(False)

        else:
            task_terminals.append(False)

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

    #
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
    task_paths['all']=dict(
        observations=np.array(episode_observations),
        actions=np.array(episode_actions),
        rewards=np.array(episode_rewards).reshape(-1, 1),
        next_observations=np.array(episode_next_observations),
        terminals=np.array(episode_terminals).reshape(-1, 1),
        agent_infos=episode_agent_infos,
        env_infos=episode_env_infos
    )
    if len(task_actions) != 0:
        task_paths[current_operator].append(dict(
            observations=np.array(task_observations),
            actions=np.array(task_actions),
            rewards=np.array(task_rewards).reshape(-1, 1),
            next_observations=np.array(task_next_observations),
            terminals=np.array(task_terminals).reshape(-1, 1),
            agent_infos=task_agent_infos,
            env_infos=task_env_infos
        ))

    # path = {operator: dict(
    #     observations=np.array(observations[operator]),
    #     actions=np.array(actions[operator]),
    #     rewards=np.array(rewards[operator]).reshape(-1, 1),
    #     next_observations=np.array(next_observations[operator]),
    #     terminals=np.array(terminals[operator]).reshape(-1, 1),
    #     agent_infos=agent_infos[operator],
    #     env_infos=env_infos[operator],
    # ) for operator in observations.keys()}
    # path['all']['full_observations'] = raw_obs
    # path['all']['full_next_observations'] = raw_next_obs,
    return task_paths


# dict(
#     observations=observations,
#     actions=actions,
#     rewards=rewards,
#     next_observations=next_observations,
#     terminals=np.array(terminals).reshape(-1, 1),
#     agent_infos=agent_infos,
#     env_infos=env_infos,
#     full_observations=raw_obs,
#     full_next_observations=raw_obs,
# )


class RePReLPathCollector(PathCollector):
    def __init__(
            self,
            env,
            operator_qfs,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            rollout_fn=RePReLRollout,
            save_env_in_snapshot=True,
            epsilon_decay=False,
            strategy=None,
            policy=None,
            planner=None,
            task_terminal_reward=100,
            get_action_kwargs={},
            agents_passed=False
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._operator_qfs = operator_qfs
        self._epsilon_decay = epsilon_decay
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._rollout_fn = rollout_fn
        self._policy = policy
        self._strategy = strategy
        self._planner = planner
        self._num_steps_total = 0
        self._num_paths_total = 0
        self._save_env_in_snapshot = save_env_in_snapshot
        self.task_terminal_reward = task_terminal_reward
        self._agents = {}
        self._get_action_kwargs = get_action_kwargs
        if not agents_passed:
            for operator in operator_qfs.keys():
                current_agent = policy(operator_qfs[operator])
                if strategy is not None:
                    current_agent = PolicyWrappedWithExplorationStrategy(
                        exploration_strategy=strategy,
                        policy=current_agent,
                    )
                self._agents[operator] = current_agent
        else:
            self._agents = operator_qfs
        if self._epsilon_decay:
            self._epsilon = self._strategy.epsilon

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = {operator: [] for operator in self._operator_qfs.keys()}
        paths['all'] = []
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
                self._planner,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
                task_terminal_reward=self.task_terminal_reward,
                get_action_kwargs=self._get_action_kwargs
            )
            path_len = len(operator_path['all']['actions'])
            if (discard_incomplete_paths
                    and path_len != max_path_length
                    and (not operator_path['all']['terminals'][-1] and
                         not operator_path['all']['agent_infos'][-1]['task_done'])
            ):
                break
            num_steps_collected += path_len
            for key, path in operator_path.items(): paths[key].append(path)
            num_paths_total += 1
        self._num_paths_total += num_paths_total
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths['all'])
        if self._epsilon_decay:
            self._epsilon = self._strategy.epsilon
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        if self._epsilon_decay:
            self._strategy.decay()
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        if self._epsilon_decay:
            stats.update(create_stats_ordered_dict(
                "Epsilon value",
                self._epsilon,
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


class RePReLGoalConditionedPathCollector(RePReLPathCollector):
    def __init__(
            self,
            *args,
            rollout_fn=RePReLGoalConditionedRollout,
            observation_key='observation',
            desired_goal_key='desired_goal',
            goal_sampling_mode=None,
            epsilon_decay=False,
            **kwargs
    ):
        def obs_processor(o):
            return np.hstack((o[observation_key], o[desired_goal_key]))

        rollout_fn = partial(
            rollout_fn,
            preprocess_obs_for_policy_fn=obs_processor,
        )
        super().__init__(*args, rollout_fn=rollout_fn, **kwargs)
        self._observation_key = observation_key
        self.epsilon_decay = epsilon_decay
        self._desired_goal_key = desired_goal_key
        self._goal_sampling_mode = goal_sampling_mode

    # def collect_new_paths(self, *args, **kwargs):
    #     self._env.goal_sampling_mode = self._goal_sampling_mode
    #     return super().collect_new_paths(*args, **kwargs)

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = {operator: [] for operator in self._operator_qfs.keys()}
        paths['all'] = []
        num_steps_collected = 0
        num_paths_total = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
                )
            task_paths = self._rollout_fn(
                self._env,
                self._agents,
                self._planner,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
                task_terminal_reward=self.task_terminal_reward,
                get_action_kwargs=self._get_action_kwargs
            )
            path_len = len(task_paths['all']['actions'])
            if (discard_incomplete_paths
                    and path_len != max_path_length
                    and (not task_paths['all']['terminals'][-1] and
                         not task_paths['all']['agent_infos'][-1]['task_done'])
            ):
                break
            num_steps_collected += path_len
            for key in self._operator_qfs.keys():
                paths[key] =paths[key] + task_paths[key]
            paths['all'].append(task_paths['all'])
            num_paths_total += 1
        self._num_paths_total += num_paths_total
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths['all'])
        if self._epsilon_decay:
            self._epsilon = self._strategy.epsilon
        return paths

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )
        return snapshot
