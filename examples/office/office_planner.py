import logging
from rlkit.util.pyhop import pyhop as hop
from rlkit.util.pyhop.planner import Planner
import numpy as np

visited_a = 0
visited_b = 1
visited_c = 2
visited_d = 3
has_mail = 4
has_coffee = 5
visited_office = 6
delivered_mail = 7
delivered_coffee = 8

OBS_FACTS = [visited_a, visited_b, visited_c, visited_d, has_mail,
             has_coffee, visited_office, delivered_mail, delivered_coffee]


def achieve_goal(state, goals):
    # Check if goal state reached
    goal_achieved = True
    unachieved_goals = []
    for object in goals:
        if object not in state.objects:
            goal_achieved = False
            unachieved_goals.append(object)
    if goal_achieved:
        return []
    return [('solve', unachieved_goals[0]), ('achieve_goal', unachieved_goals)]


# object is collected when location is visited
# loc_{a,b,c,d} -> visited_{a,b,c,d}
# mailroom -> has_mail, breakroom -> has_coffee, office -> visited_office
def pickup(state, object):
    # Special case of office
    if object == visited_office:
        if has_mail in state.objects:
            state.objects.remove(has_mail)
            state.objects.add(delivered_mail)
        if has_coffee in state.objects:
            state.objects.remove(has_coffee)
            state.objects.add(delivered_coffee)
        state.objects.add(object)
        return state
    if object in [visited_a, visited_b, visited_c, visited_d, has_mail, has_coffee]:
        state.objects.add(object)
        return state
    return False


# The object is converted to delivered_object and agent location is updated
# has_coffee @ office -> delivered_coffee
# has_mail @office -> delivered_mail
def deliver(state, object, delivered_object):
    if object in state.objects:
        state.objects.remove(object)
        state.objects.add(delivered_object)
        if delivered_object in [delivered_mail, delivered_coffee]:
            state.objects.add(visited_office)
        return state
    return False


def add_deliver(has_predicate, deliver_predicate):
    def deliver_dynamic_method(state, object):
        if object == deliver_predicate:
            return [('pickup', has_predicate), ('deliver', has_predicate, object)]
        return False

    deliver_dynamic_method.__name__ = "deliver_%s" % (str(has_predicate))
    return deliver_dynamic_method


def add_pickup(visit_predicate):
    def pickup_dynamic_method(state, object):
        if object == visit_predicate:
            return [('pickup', visit_predicate)]
        return False

    pickup_dynamic_method.__name__ = "pickup_%s" % (str(visit_predicate))
    return pickup_dynamic_method


def define_dynamic_methods():
    dynamic_methods = []
    dynamic_methods.append(add_deliver(has_coffee, delivered_coffee))
    dynamic_methods.append(add_deliver(has_mail, delivered_mail))
    dynamic_methods.append(add_pickup(visited_a))
    dynamic_methods.append(add_pickup(visited_b))
    dynamic_methods.append(add_pickup(visited_c))
    dynamic_methods.append(add_pickup(visited_d))
    dynamic_methods.append(add_pickup(has_mail))
    dynamic_methods.append(add_pickup(has_coffee))
    dynamic_methods.append(add_pickup(visited_office))
    hop.declare_methods('solve', *dynamic_methods)


def declare_methods_and_operators():
    hop.declare_methods('achieve_goal', achieve_goal)
    define_dynamic_methods()

    hop.declare_operators(pickup, deliver)


def get_environment_state(obs):
    state = hop.State('state1')
    facts = obs[2:]
    state.objects = set([i for i, x in enumerate(facts) if x])
    state.track = set()
    return state


class OfficePlanner(Planner):

    def __init__(self, env):
        declare_methods_and_operators()
        self.goal = env.target
        self.plan = None
        self.operator_list = self.get_operators()
        self.dims = {'pickup': (env.observation_space.shape[0], env.action_space.n),
                     'deliver': (env.observation_space.shape[0], env.action_space.n)}

    def set_goal(self, goal):
        self.goal = goal

    def get_plan(self, state):
        return hop.pyhop(get_environment_state(state), [('achieve_goal', self.goal)], verbose=0)

    def get_next_operator(self, state):
        if self.plan is None:
            sub_tasks = self.get_plan(state)
            sub_tasks.reverse()
            self.plan = sub_tasks
        if not self.plan:
            return None, None
        sub_task = self.plan.pop()
        return sub_task[0], sub_task[1:]

    def get_operators(self):
        return list(hop.operators.keys())

    def reset(self):
        self.plan = None

    def is_terminal(self, operator, subtask, state):
        facts = state[2:]
        terminal = True
        if operator == 'pickup':
            terminal = terminal and facts[subtask[0]]
        if operator == 'deliver':
            terminal = terminal and facts[subtask[-1]]
        return terminal

    def get_abstract_state(self, operator, subtask, state):
        new_state = state[:2]
        facts = state[2:]
        if operator == 'pickup':
            mask = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ])
            mask[subtask[0]+2] = 1
            new_state = state*mask
        if operator == 'deliver':
            mask = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0 ])
            mask[subtask[0]+2] = 1
            mask[subtask[1]+2] = 1
            new_state = state*mask
        return new_state


if __name__ == '__main__':
    import officeworld
    import gym
    from random import Random

    rng = Random(2019)
    env = gym.make("OfficeWorld-deliver-coffee-v0")
    obs = env.reset()
    # goal = [delivered_mail, delivered_coffee]
    # declare_methods_and_operators()
    # plan = hop.pyhop(get_environment_state(obs), [('achieve_goal', goal)])
    # print(plan)

    planner = OfficePlanner(env)
    print(obs)
    print(planner.get_plan(obs))
    operator, task = planner.get_next_operator(obs)
    print(planner.get_abstract_state(operator, task, obs))
    print(planner.is_terminal(operator, task, obs))
    operator, task = planner.get_next_operator(obs)
    print(planner.get_abstract_state(operator, task, obs))
