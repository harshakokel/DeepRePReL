import abc
from rlkit.util.pyhop import pyhop as hop


class Planner(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, env):
        pass

    def set_goal(self, _goal):
        self.goal = _goal

    @abc.abstractmethod
    def get_plan(self, _state):
        pass

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

    @abc.abstractmethod
    def get_abstract_state(self, operator, subtask, state):
        pass

    @abc.abstractmethod
    def is_terminal(self, operator, p, state):
        pass
