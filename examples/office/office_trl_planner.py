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


def get_mail(state, object):
    state.objects.add(object)
    return state


def get_coffee(state, object):
    state.objects.add(object)
    return state


def go_to_office(state, object):
    state.objects.add(visited_office)
    if object == delivered_mail:
        state.objects.remove(has_mail)
        state.objects.add(object)
    elif object  == delivered_coffee:
        state.objects.remove(has_coffee)
        state.objects.add(object)
    state.objects.add(object)
    return state



def deliver_mail(state, object):
    if object == delivered_mail:
        return [('get_mail', has_mail), ('go_to_office', delivered_mail)]
    return False

def deliver_coffee(state, object):
    if object == delivered_coffee:
        return [('get_coffee', has_coffee), ('go_to_office', delivered_coffee)]
    return False




def declare_methods_and_operators():
    hop.declare_methods('achieve_goal', achieve_goal)
    hop.declare_methods('solve', deliver_mail, deliver_coffee)
    hop.declare_operators(get_coffee, get_mail, go_to_office)


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
        self.dims = {'get_coffee': (env.observation_space.shape[0], env.action_space.n),
                     'get_mail': (env.observation_space.shape[0], env.action_space.n),
                     'go_to_office': (env.observation_space.shape[0], env.action_space.n),
                     }

    def set_goal(self, goal):
        self.goal = goal

    def get_plan(self, state):
        return hop.pyhop(get_environment_state(state), [('achieve_goal', self.goal)], verbose=0)

    def get_next_operator(self, state_dict):
        if self.plan is None:
            sub_tasks = self.get_plan(state_dict)
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
        if operator == 'get_coffee':
            terminal = terminal and facts[has_coffee]
        elif operator == 'get_mail':
            terminal = terminal and facts[has_mail]
        elif operator == 'go_to_office':
            terminal = terminal and facts[visited_office]
        return terminal

    def get_abstract_state(self, operator, subtask, state):
        return state


if __name__ == '__main__':
    import officeworld
    import gym
    from random import Random

    rng = Random(2019)
    env = gym.make("OfficeWorld-deliver-coffee-v0")
    obs = env.reset()
    # goal = [delivered_mail, delivered_coffee]
    # declare_methods_and_operators()
    # plan = hop.pyhop(get_environment_state(obs), [('achieve_goal', goal)], verbose=2)
    # print(plan)

    planner = OfficePlanner(env)
    print(obs)
    print(planner.get_plan(obs))
    operator, task = planner.get_next_operator(obs)
    print(planner.get_abstract_state(operator, task, obs))
    print(planner.is_terminal(operator, task, obs))
    operator, task = planner.get_next_operator(obs)
    print(planner.get_abstract_state(operator, task, obs))
