import logging
from rlkit.util.pyhop import pyhop as hop
import numpy as np


def achieve_goal(state, goals):
    # Check if goal state reached
    goal_achieved = True
    unachieved_goals = []
    for passenger in goals.at_dest:
        if passenger not in state.dropped:
            goal_achieved = False
            unachieved_goals.append(passenger)
    if goal_achieved:
        return []
    return [('transport', unachieved_goals[0]), ('achieve_goal', goals)]


# If passenger is not in taxi and at location, then pickup
def pickup(state, passenger):
    if passenger not in state.in_taxi:
        location = state.at[passenger]
        state.in_taxi.append(passenger)
        state.taxi_at = location
        state.at.pop(passenger)
        return state
    return False


# If passenger is in taxi, then drop
def drop(state, passenger):
    if passenger in state.in_taxi:
        location = state.dest[passenger]
        state.in_taxi.remove(passenger)
        state.dropped.append(passenger)
        state.taxi_at = location
        return state
    return False


def add_transport_1(p):
    def transport_dynamic_method(state, passenger=p):
        if passenger not in state.dropped or passenger not in state.in_taxi:
            return [('pickup', passenger), ('drop', passenger)]
        return False

    transport_dynamic_method.__name__ = "transport_%s" % (str(p))
    return transport_dynamic_method


def add_transport_2(p):
    def transport_dynamic_method(state, passenger=p):
        if passenger in state.in_taxi:
            return [('drop', passenger)]
        return False

    transport_dynamic_method.__name__ = "transport_%s_intaxi" % (str(p))
    return transport_dynamic_method


def define_dynamic_methods(max_passenger):
    dynamic_methods_1 = []
    dynamic_methods_2 = []
    for passenger in range(1, max_passenger + 1):
        dynamic_methods_1.append(add_transport_1(passenger))
        dynamic_methods_2.append(add_transport_2(passenger))
    dynamic_methods = dynamic_methods_2 + dynamic_methods_1
    hop.declare_methods('transport', *dynamic_methods)


def declare_methods_and_operators(max_passenger):
    hop.declare_methods('achieve_goal', achieve_goal)
    define_dynamic_methods(max_passenger)
    hop.declare_operators(pickup, drop)


def get_environment_state(obs, num_p):
    state = hop.State('state1')
    state.taxi_at = None
    state.in_taxi = []
    obs = obs[-(num_p * 9):]
    # print(obs)
    state.at, state.dest = {}, {}
    for p in range(1, num_p + 1):
        loc = int(np.dot(obs[((p - 1) * 9):(((p - 1) * 9) + 5)], [1, 2, 3, 4, 5]))
        if loc == 0:
            continue
        elif loc == 5:
            state.in_taxi.append(p)
        else:
            state.at[p] = loc
        state.dest[p] = int(np.dot(obs[(((p - 1) * 9) + 5):(((p - 1) * 9) + 9)], [1, 2, 3, 4]))
    state.dropped = []
    # hop.print_state(state)
    goal = hop.State('goal')
    goal.at_dest = list(state.at.keys())
    return state, goal


class TaxiPlanner:

    def __init__(self, env):
        self.max_passenger = env.max_passenger
        declare_methods_and_operators(self.max_passenger)
        self.plan = None
        self.goal = None
        self.operator_list = self.get_operators()
        self.grid_dim = env.grid_dim
        self.dims = {'pickup': (env.grid_dim + 5, env.action_space.n),
                     'drop': (env.grid_dim + 5, env.action_space.n)}

    def set_goal(self, _goal):
        self.goal = _goal

    def get_plan(self, _state):
        state, goal = get_environment_state(_state, self.max_passenger)
        return hop.pyhop(state, [('achieve_goal', goal)])

    def get_next_operator(self, state):
        if self.plan is None:
            sub_tasks = self.get_plan(state)
            sub_tasks.reverse()
            self.plan = sub_tasks
        if not self.plan:
            return None, None
        sub_task = self.plan.pop()
        return sub_task[0], sub_task[1]

    def get_operators(self):
        return list(hop.operators.keys())

    def reset(self):
        self.plan = None
        self.goal = None

    def get_dim(self, operator):
        return self.dims[operator]

    def get_abstract_state(self, operator, p, state):
        obs = state[-(self.max_passenger * 9):]
        if operator == 'pickup':
            return np.concatenate([state[:self.grid_dim], obs[((p - 1) * 9):(((p - 1) * 9) + 5)]])
        elif operator == 'drop':
            return np.concatenate([state[:self.grid_dim], obs[(((p - 1) * 9) + 4):(((p - 1) * 9) + 9)]])

    def is_terminal(self, operator, p, state):
        obs = state[-(self.max_passenger * 9):]
        if operator == 'pickup':
            pick_loc = int(np.dot(obs[((p - 1) * 9):(((p - 1) * 9) + 5)], [1, 2, 3, 4, 5]))
            return pick_loc == 5
        if operator == 'drop':
            dest_loc = int(np.dot(obs[(((p - 1) * 9) + 5):(((p - 1) * 9) + 9)], [1, 2, 3, 4]))
            return dest_loc == 0


class TaxiTRLPlanner(TaxiPlanner):

    def __init__(self, env):
        self.max_passenger =env.max_passenger
        declare_methods_and_operators(self.max_passenger)
        self.plan = None
        self.goal = None
        self.operator_list = self.get_operators()
        self.grid_dim = env.grid_dim
        self.dims = {'pickup': (env.observation_space.low.size,  env.action_space.n),
                     'drop': (env.observation_space.low.size,  env.action_space.n)}

    def get_abstract_state(self, operator, p, state):
        # No abstraction for TRL
        return state




if __name__ == '__main__':
    import taxi_domain
    import gym

    env = gym.make('RelationalTaxiWorld-task2-v1')

    planner = TaxiPlanner(env)
    obs = env.reset()
    print(obs)
    plan = planner.get_plan(obs)
    print(plan)
    #
    p = planner.get_abstract_state('pickup', 1, obs)
    print(p[-5:])
    p = planner.get_abstract_state('drop', 1, obs)
    print(p[-5:])
    p = planner.get_abstract_state('pickup', 2, obs)
    print(p[-5:])
