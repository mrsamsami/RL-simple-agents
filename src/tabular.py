import math
import time

import numpy as np
import pandas as pd

from src import utils
from src.agent import Agent


class Tabular(Agent):
    def observation_to_state(self, observation):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(observation[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in
                  range(len(observation))]
        state = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(observation))]
        state = [min(self.buckets[i] - 1, max(0, state[i])) for i in range(len(observation))]
        return tuple(state)

    def before_episode(self):
        raise NotImplementedError()

    def during_episode(self, state, action, reward, next_state, epsilon, alpha):
        raise NotImplementedError()

    def after_episode(self, alpha):
        raise NotImplementedError()

    def get_best_action(self, state):
        return np.argmax(self.Q[state])

    def train(self,
              env,
              n_X=1,
              n_X_dot=1,
              n_theta=6,
              n_theta_dot=12,
              *args, **kwargs):
        self.n_X = n_X
        self.n_X_dot = n_X_dot
        self.n_theta = n_theta
        self.n_theta_dot = n_theta_dot
        self.buckets = (n_X, n_X_dot, n_theta, n_theta_dot)
        self.Q = np.zeros(self.buckets + (env.action_space.n,))
        return super().train(
            env,
            *args, **kwargs
        )
