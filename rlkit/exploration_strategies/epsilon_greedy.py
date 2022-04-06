import random

from rlkit.exploration_strategies.base import RawExplorationStrategy


class EpsilonGreedy(RawExplorationStrategy):
    """
    Take a random discrete action with some probability.
    """
    def __init__(self, action_space, prob_random_action=0.1):
        self.prob_random_action = prob_random_action
        self.action_space = action_space

    def get_action_from_raw_action(self, action, **kwargs):
        if random.random() <= self.prob_random_action:
            return self.action_space.sample()
        return action



class EpsilonGreedyWithDecay(RawExplorationStrategy):
    """
    Take a random discrete action with some probability.
    """
    def __init__(self, action_space, prob_random_action=1, min_epsilon= 0.1, num_epochs=100, explore_ratio=0.1):
        self.epsilon = prob_random_action
        self.action_space = action_space
        self.min_epsilon = min_epsilon
        self.epsilon_decay = (prob_random_action - min_epsilon) / (num_epochs*explore_ratio)

    def decay(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.epsilon_decay
            print(F"epsilon updated to {self.epsilon}")

    def get_action_from_raw_action(self, action, **kwargs):
        if random.random() <= self.epsilon:
            return self.action_space.sample()
        return action