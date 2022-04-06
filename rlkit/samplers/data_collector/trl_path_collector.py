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


def TRLRollout(
        env,
        agents,
        planner,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        preprocess_obs_for_policy_fn=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        task_terminal_reward=100,
        full_o_postprocess_func=None,
        reset_callback=None,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x
    raw_obs = []
    raw_next_obs = []
    keys = list(agents.keys())+['all']
    observations = []
    actions = []
    rewards = {operator: [] for operator in keys}
    terminals = {operator: [] for operator in keys}
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    # TODO: Implement the roll out
    planner.reset()
    o = env.reset()
    # if reset_callback:
    #     reset_callback(env, agent, o)
    agent = {}
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
        observations.append(o_for_agent)

        for key in keys:
            if key == current_operator:
                rewards[key].append(task_terminal_reward+r if task_done else r)
                terminals[key].append(task_done)
            elif key == 'all':
                rewards[key].append(r)
                terminals[key].append(episode_done)
            else:
                extra_task_terminal = planner.is_terminal(key, current_task, next_o)
                rewards[key].append(task_terminal_reward+r if extra_task_terminal else r)
                terminals[key].append(extra_task_terminal)
        actions.append(a)
        next_observations.append(next_o_for_agent)
        raw_next_obs.append(next_o)
        agent_info.update({'task_done': task_done})
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if episode_done:
            break
        if task_done:
            current_operator, current_task = planner.get_next_operator(o)
            if current_operator is None:
                break
            current_agent = agents[current_operator]
        o = next_o

    return {operator: dict(
        observations=np.array(observations),
        actions=np.array(actions).reshape(-1, 1),
        rewards=np.array(rewards[operator]).reshape(-1, 1),
        next_observations=np.array(next_observations),
        terminals=np.array(terminals[operator]).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    ) for operator in keys}

