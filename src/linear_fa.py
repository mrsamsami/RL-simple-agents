import math
import time

import gym
import numpy as np
import pandas as pd

import sklearn
import sklearn.pipeline
import sklearn.preprocessing

from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

from src import utils
from src.agent import Agent


class LinearFA(Agent):
    def observation_to_state(self, observation):
        return self.featurizer.transform(self.scaler.transform([observation]))[0]

    def get_best_action(self, state):
        return np.argmax([self.models[a].predict([state])[0] for a in range(self.env.action_space.n)])

    def get_state_value(self, state):
        return np.max([self.models[a].predict([state])[0] for a in range(self.env.action_space.n)])

    def before_episode(self):
        pass

    def during_episode(self, state, action, reward, next_state, epsilon, alpha):
        td_target = reward + alpha * self.get_state_value(next_state)
        self.models[action].partial_fit([state], [td_target])

    def after_episode(self, alpha):
        pass

    def train(self, env, *args, **kwargs):
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
        observation_examples = np.array([env.observation_space.sample() for _ in range(10000)], dtype='float64')
        self.scaler.fit(observation_examples)
        self.featurizer.fit(self.scaler.transform(observation_examples))
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self.observation_to_state(env.reset())], [0])
            self.models.append(model)
        return super().train(env, *args, **kwargs)
